"""Read-only global Polymarket moneyline screen. No credentials or execution."""
import json
import math
import pathlib
import re
from datetime import datetime, timezone
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'


def stamp(value):
    try:
        d = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return d if d.tzinfo else None
    except ValueError:
        return None


def norm(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def array(value):
    return json.loads(value) if isinstance(value, str) else value


def get(url, **params):
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def discover():
    sports = get(GAMMA + '/sports')
    series = next(s['series'] for s in sports if s['sport'] == 'cfb')
    events = []
    for offset in range(0, 2000, 100):
        page = get(GAMMA + '/events', series_id=series, closed='false',
                   limit=100, offset=offset)
        events.extend(page)
        if len(page) < 100:
            return events
    raise ValueError('Discovery pagination limit reached')


def match(event, market, games, now):
    if (market.get('sportsMarketType') != 'moneyline' or
            market.get('question') != event.get('title') or
            not re.fullmatch(r'cfb-[a-zA-Z0-9-]+', event.get('slug', '')) or
            market.get('active') is not True or market.get('closed') is not False or
            market.get('acceptingOrders') is not True):
        raise ValueError('Not an open full-game moneyline')
    outcomes = array(market.get('outcomes', []))
    tokens = array(market.get('clobTokenIds', []))
    if len(outcomes) != 2 or len(tokens) != 2 or len(set(tokens)) != 2:
        raise ValueError('Ambiguous outcome tokens')
    start = stamp(market.get('gameStartTime'))
    if not start or start <= now:
        raise ValueError('Started or undated game')
    candidates = [g for g in games if not g['played'] and not g['unrated']
                  and {norm(g['home']), norm(g['away'])} == {norm(o) for o in outcomes}
                  and stamp(g.get('kickoff'))
                  and abs((stamp(g['kickoff']) - start).total_seconds()) <= 300]
    if len(candidates) != 1:
        raise ValueError('No exact team and kickoff match in SEC schedule')
    return candidates[0], outcomes, tokens, start


def top(book, token, now):
    if str(book.get('asset_id')) != str(token):
        raise ValueError('Wrong outcome order book')
    age = now.timestamp() - float(book.get('timestamp', 0)) / 1000
    if not -60 <= age <= 900:
        raise ValueError('Stale order book')
    def levels(key):
        return [(float(x['price']), float(x['size'])) for x in book.get(key, [])
                if 0 < float(x['price']) < 1 and math.isfinite(float(x['size']))
                and float(x['size']) > 0]
    bids, asks = levels('bids'), levels('asks')
    if not bids or not asks:
        raise ValueError('Missing bid or ask')
    bid, ask = max(p for p, _ in bids), min(p for p, _ in asks)
    if bid > ask:
        raise ValueError('Crossed book')
    return bid, ask, sum(s for p, s in bids if p == bid), sum(s for p, s in asks if p == ask)


def movement(history, now):
    points = sorted((float(x['t']), float(x['p'])) for x in history
                    if 0 < float(x['p']) < 1 and float(x['t']) <= now.timestamp())
    if not points or now.timestamp() - points[-1][0] > 7200:
        return None
    old = min(points, key=lambda x: abs(x[0] - (now.timestamp() - 86400)))
    if abs(old[0] - (now.timestamp() - 86400)) > 7200:
        return None
    return points[-1][1] - old[1]


def classify(p, other_mid, quote, fee_rate, change):
    bid, ask, bid_size, ask_size = quote
    mid = (bid + ask) / 2
    market_p = mid / (mid + other_mid)
    fee = lambda price: fee_rate * price * (1 - price)
    cost = ask + fee(ask) if fee_rate is not None else None
    edge = p - cost if cost is not None else None
    # Flags describe disagreements; execution considerations are annotations.
    mismatch = p > .5 and market_p < .5
    target = ask + (p - ask) / 2
    gain = target - fee(target) - cost if cost is not None else None
    good_buy = p > ask
    notes = []
    if ask - bid > .04 + 1e-9: notes.append('Wide bid/ask spread')
    if min(bid_size, ask_size) < 100: notes.append('Thin top-of-book liquidity')
    if fee_rate is None: notes.append('Fees unavailable')
    if change is None: notes.append('24-hour movement unavailable')
    elif change <= 0: notes.append('No upward 24-hour momentum')
    if edge is not None and edge <= 0: notes.append('Entry fees erase model edge')
    if gain is not None and gain <= 0: notes.append('Exit scenario does not cover fees')
    return dict(model_probability=p, market_probability=market_p, bid=bid, ask=ask,
                bid_size=bid_size, ask_size=ask_size, entry_cost=cost, net_edge=edge,
                change_24h=change, target=target, scenario_gain=gain,
                scenario_return=gain / cost if cost else None, mismatch=mismatch,
                good_buy=good_buy, model_gap=p-market_p, notes=notes)


def build(model, now=None, fetch=get, events=None):
    now = now or datetime.now(timezone.utc)
    result = dict(checked_at=now.isoformat(), model_at=model['generated_at'],
                  status='ok', venue='Polymarket Global', rows=[], skipped=[],
                  events_scanned=0, matched_games=0)
    checked = stamp(model.get('data_source', {}).get('checked_at'))
    if not checked or not -60 <= (now - checked).total_seconds() <= 8 * 3600:
        raise ValueError('Model results are older than eight hours')
    if not model.get('inseason'):
        raise ValueError('Current in-season model unavailable')
    events = discover() if events is None else events
    result['events_scanned'] = len(events)
    seen = set()
    for event in events:
        for market in event.get('markets', []):
            if market.get('sportsMarketType') != 'moneyline':
                continue
            try:
                game, outcomes, tokens, start = match(event, market, model['games'], now)
                if game['id'] in seen:
                    continue
                p_home = game.get('p_home_current')
                if p_home is None or not 0 < p_home < 1:
                    raise ValueError('Current matchup probability unavailable')
                if market.get('feesEnabled') is False:
                    rate = 0.
                else:
                    schedule = market.get('feeSchedule', {})
                    rate = schedule.get('rate')
                    rate = float(rate) if rate is not None else None
                    if schedule.get('exponent') != 1 or rate is None or not 0 <= rate <= 1:
                        rate = None
                quotes = [top(fetch(CLOB + '/book', token_id=t), t, now) for t in tokens]
                seen.add(game['id'])
                result['matched_games'] += 1
                for i, outcome in enumerate(outcomes):
                    team = game['home'] if norm(outcome) == norm(game['home']) else game['away']
                    p = p_home if team == game['home'] else 1 - p_home
                    try:
                        hist = fetch(CLOB + '/prices-history', market=tokens[i],
                                     startTs=int(now.timestamp()) - 93600,
                                     endTs=int(now.timestamp()), fidelity=60)
                        change = movement(hist.get('history', []), now)
                    except (requests.RequestException, ValueError, KeyError, TypeError):
                        change = None
                    row = classify(p, sum(quotes[1-i][:2]) / 2, quotes[i], rate, change)
                    row.update(team=team, home=game['home'], away=game['away'],
                               kickoff=start.isoformat(), neutral=game['neutral'],
                               url='https://polymarket.com/event/' + event['slug'],
                               fee_rate=rate)
                    result['rows'].append(row)
            except (requests.RequestException, ValueError, KeyError, TypeError) as e:
                result['skipped'].append(dict(event=event.get('title', ''), reason=str(e)))
    result['rows'].sort(key=lambda r: r['model_gap'], reverse=True)
    result['upcoming'] = [dict(home=g['home'], away=g['away'], kickoff=g.get('kickoff'),
                               date=g.get('date'), probability=g.get('p_home_current'),
                               market_found=g['id'] in seen)
                          for g in model['games'] if not g['played']
                          and (stamp(g.get('kickoff')) or now) > now]
    result['upcoming'].sort(key=lambda g: g['kickoff'])
    return result


if __name__ == '__main__':
    model = json.loads((ROOT / 'output/app_data.json').read_text())
    try:
        result = build(model)
    except (requests.RequestException, ValueError, KeyError, StopIteration) as error:
        result = dict(status='unavailable', checked_at=datetime.now(timezone.utc).isoformat(),
                      rows=[], events_scanned=0, matched_games=0,
                      error='Market refresh unavailable; no current candidates published.')
        print('[best bets] refresh unavailable:', type(error).__name__)
    (ROOT / 'output/best_bets.json').write_text(json.dumps(result, allow_nan=False))
    print('[best bets]', result['status'], 'matched', result['matched_games'], 'games')
