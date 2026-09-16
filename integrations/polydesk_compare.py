#!/usr/bin/env python3
"""Read public Polydesk snapshots and score all-FBS full-game moneyline disagreements.
No network, credentials, browser, order placement, or automatic execution handoff.
"""
from __future__ import annotations
import argparse, hashlib, html, json, math, re
from datetime import datetime, timezone
from pathlib import Path


def dt(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed if parsed.tzinfo else None
    except (TypeError, ValueError):
        return None


def normalize(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def list_value(value):
    return json.loads(value) if isinstance(value, str) else value


def estimate(model, home, away, neutral=False):
    ratings = {r['team']: float(r['projected_rating']) for r in model['all_ratings']}
    c = model['calibration']
    if c.get('method') != 'walk-forward preseason predictions':
        raise ValueError('Unrecognized probability calibration')
    if home == away or home not in ratings or away not in ratings:
        raise ValueError('Both distinct teams must have model ratings')
    margin = c['rating_diff_coef'] * (ratings[home] - ratings[away])
    if not neutral: margin += c['home_field_advantage']
    sd = c['neutral_residual_sd'] if neutral else c['residual_sd']
    if not math.isfinite(margin) or not math.isfinite(sd) or sd <= 0: raise ValueError('Invalid calibration spread')
    p = .5 * (1 + math.erf(margin / sd / math.sqrt(2)))
    return {home: p, away: 1-p}


def book_top(book):
    def levels(key):
        out = []
        for row in book.get(key, []):
            try:
                price, size = float(row['price']), float(row['size'])
                if 0 < price < 1 and size > 0 and math.isfinite(size): out.append((price,size))
            except (ValueError, TypeError, KeyError): pass
        return out
    bids, asks = levels('bids'), levels('asks')
    bid = max((p for p,_ in bids), default=None)
    ask = min((p for p,_ in asks), default=None)
    size = sum(s for p,s in asks if p == ask)
    if bid is not None and ask is not None and bid > ask: raise ValueError('Crossed order book')
    return bid, ask, size


def match_game(event, market, scoreboard, ratings):
    outcomes = list_value(market.get('outcomes', []))
    if market.get('sportsMarketType') != 'moneyline' or len(outcomes) != 2:
        raise ValueError('Only two-team full-game moneylines are supported')
    # Period winners can also be tagged moneyline. Require the exact event title.
    if market.get('question') != event.get('title'):
        raise ValueError('Period/prop contract or ambiguous full-game title')
    start = dt(market.get('gameStartTime'))
    if not start: raise ValueError('Market kickoff missing')
    known = {normalize(r['team']):r['team'] for r in ratings}
    candidates=[]
    for game in scoreboard.get('events', []):
        when=dt(game.get('date'))
        if not when or abs((when-start).total_seconds()) > 300: continue
        for comp in game.get('competitions', []):
            teams=comp.get('competitors', [])
            if len(teams)!=2 or not isinstance(comp.get('neutralSite'), bool): continue
            mapped={}
            for t in teams:
                name=known.get(normalize(t.get('team',{}).get('location')))
                if name: mapped[t.get('homeAway')]=name
            if set(mapped)!={'home','away'}: continue
            if {normalize(o) for o in outcomes} != {normalize(n) for n in mapped.values()}: continue
            candidates.append((game,comp,mapped))
    if len(candidates)!=1: raise ValueError('No unique team/date/venue match in ESPN snapshot')
    return candidates[0]


def compare(model, run, now=None, target_venue='polymarket_us'):
    now = now or datetime.now(timezone.utc)
    report=json.loads((run/'report.json').read_text(encoding='utf-8-sig'))
    stamp=dt(report.get('utc'))
    model_stamp=dt(model.get('generated_at'))
    common=[]
    if not stamp or not -60 <= (now-stamp).total_seconds() <= 900: common.append('market_snapshot_stale_or_undated')
    if not model_stamp or not 0 <= (now-model_stamp).total_seconds() <= 7*86400: common.append('model_snapshot_stale_or_undated')
    if model.get('snapshot_type') != 'preseason': common.append('unknown_model_snapshot_type')
    # A preseason projection is a baseline; it does not incorporate new-season results.
    common.append('preseason_only_current_form_unverified')
    source_venue='polymarket_global'
    if target_venue!=source_venue: common.append('research_execution_venue_mismatch')
    common.append('out_of_sample_profitability_unverified')
    scoreboard=json.loads((run/'espn_ncaaf.json').read_text(encoding='utf-8-sig'))
    rows=[]; skips=[]
    for file in sorted(run.glob('evt_cfb-*.json')):
        raw=json.loads(file.read_text(encoding='utf-8-sig'))
        for event in raw if isinstance(raw,list) else [raw]:
            for m in event.get('markets', []):
                if m.get('sportsMarketType') != 'moneyline' or m.get('question') != event.get('title'): continue
                try:
                    if not m.get('active') or m.get('closed') or m.get('acceptingOrders') is False:
                        raise ValueError('Contract not accepting orders')
                    if not re.fullmatch(r'cfb-[A-Za-z0-9_-]+', event.get('slug','')): raise ValueError('Invalid market slug')
                    game, comp, teams=match_game(event,m,scoreboard,model['all_ratings'])
                    if game.get('status',{}).get('type',{}).get('state') != 'pre': raise ValueError('Not a pregame event')
                    if dt(game['date']) <= now: raise ValueError('Kickoff has passed')
                    if dt(game['date']).year != int(model['season']): raise ValueError('Wrong model season')
                    probabilities=estimate(model,teams['home'],teams['away'],comp['neutralSite'])
                    outcomes=list_value(m['outcomes']);tokens=list_value(m.get('clobTokenIds',[]))
                    if len(tokens)!=2 or len(set(tokens))!=2: raise ValueError('Missing or duplicate outcome token mapping')
                    for outcome,token in zip(outcomes,tokens):
                        canonical=next(n for n in teams.values() if normalize(n)==normalize(outcome))
                        book_file=run/('clob_'+event['slug']+'_'+re.sub(r'[^A-Za-z0-9_-]','_',outcome)+'.json')
                        book=json.loads(book_file.read_text(encoding='utf-8-sig'))
                        if str(book.get('token',book.get('asset_id'))) != str(token): raise ValueError('Order book side mismatch')
                        if book.get('snapshot_complete') is not True: raise ValueError('Legacy truncated order book; fresh complete snapshot required')
                        bid,ask,size=book_top(book)
                        reasons=list(common)
                        quote_stamp=dt(book.get('retrieved_at'))
                        if not quote_stamp or not -60 <= (now-quote_stamp).total_seconds() <= 900: reasons.append('book_timestamp_stale_or_missing')
                        if ask is None: reasons.append('no_executable_ask')
                        if m.get('feesEnabled') is not False: reasons.append('fees_not_estimated')
                        if size < 1: reasons.append('insufficient_visible_depth')
                        p=probabilities[canonical]
                        rows.append({'event':event['title'],'game_id':game['id'],'kickoff':game['date'],'team':canonical,
                          'home':teams['home'],'away':teams['away'],'neutral':comp['neutralSite'],
                          'model_probability':round(p,6),'best_bid':bid,'best_ask':ask,'ask_size':size,
                          'gross_difference_pp':round((p-ask)*100,3) if ask is not None else None,
                          'net_edge_pp':None,'source_venue':source_venue,'target_venue':target_venue,
                          'status':'RESEARCH_ONLY','execution_eligible':False,'blockers':reasons,
                          'model_as_of':model.get('generated_at'),'quote_as_of':book.get('retrieved_at')})
                except (ValueError, KeyError, TypeError, StopIteration, OSError) as error:
                    skips.append({'event':event.get('title'),'reason':str(error)})
    return {'generated_at':now.isoformat(),'mode':'research_only','execution_enabled':False,'rows':rows,'skips':skips,
            'coverage':{'rated_teams':len(model['all_ratings']),'matched_outcomes':len(rows),
             'scope':'Full-game college-football moneylines present in the existing Polydesk snapshot; not a market-wide scan'},
            'not_checked':['current injuries and roster changes','US venue contract/orderbook','fees and slippage','profitability','settlement and closing-line performance']}


def write_report(result, out):
    out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=[]
    for r in result['rows']:
        pct=lambda v: '\u2014' if v is None else f'{v*100:.1f}%'
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['event'],r['team'],pct(r['model_probability']),pct(r['best_ask']),r['gross_difference_pp'],r['model_as_of'],r['quote_as_of'],', '.join(r['blockers'])])+'</tr>')
    (out/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hedges & Margins · Research comparison</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#162433}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccd;padding:12px;text-align:left}td:last-child{font-size:12px}table{min-width:700px}.scroll{overflow:auto}</style><h1>College football · model / market comparison</h1><p>Research only. These are disagreements, not verified profitable opportunities. No orders are generated. All figures are snapshots.</p><p>'+html.escape(result['generated_at'])+'</p><div class="scroll"><table><tr><th>Game</th><th>Team</th><th>Model chance</th><th>Market ask</th><th>Gross difference (pp)</th><th>Model snapshot</th><th>Market snapshot</th><th>Unresolved checks</th></tr>'+''.join(rows)+'</table></div><h2>Coverage</h2><p>'+html.escape(result['coverage']['scope'])+'</p><h2>Skipped</h2><pre>'+html.escape(json.dumps(result['skips'],indent=2))+'</pre></html>')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();model=json.loads(args.model.read_text(encoding='utf-8-sig'));result=compare(model,args.run)
    result['model_sha256']=hashlib.sha256(args.model.read_bytes()).hexdigest();write_report(result,args.out)
    print(json.dumps({'rows':len(result['rows']),'skips':len(result['skips']),'execution_enabled':False,'report':str(args.out/'index.html')}))
if __name__=='__main__':main()
