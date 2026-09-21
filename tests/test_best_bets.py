import importlib.util
import pathlib
import sys
import unittest
from datetime import datetime, timezone, timedelta
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
import best_bets as b

NOW = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)


class BestBets(unittest.TestCase):
    def test_winner_mismatch_and_fee_adjusted_cashout(self):
        row = b.classify(.65, .6, (.38, .40, 200, 200), .05, .02)
        self.assertTrue(row['mismatch'])
        self.assertTrue(row['good_buy'])
        self.assertAlmostEqual(row['entry_cost'], .412)
        self.assertAlmostEqual(row['target'], .525)
        self.assertAlmostEqual(row['scenario_gain'], .525-.05*.525*.475-.412)

    def test_good_buy_can_be_a_projected_loser(self):
        row = b.classify(.4, .75, (.23, .25, 200, 200), .05, .02)
        self.assertFalse(row['mismatch'])
        self.assertTrue(row['good_buy'])

    def test_missing_or_negative_momentum_does_not_hide_flag(self):
        for change in [None, -.01, 0]:
            row = b.classify(.7, .6, (.38, .4, 200, 200), .05, change)
            self.assertTrue(row['good_buy'])
            self.assertTrue(row['notes'])

    def test_illiquid_and_wide_books_are_annotated(self):
        for quote in [(.3, .4, 200, 200), (.38, .4, 20, 200)]:
            r = b.classify(.7, .6, quote, .05, .03)
            self.assertTrue(r['mismatch'])
            self.assertTrue(r['good_buy'])
            self.assertTrue(r['notes'])

    def test_tiny_edge_and_unknown_fees_still_flagged(self):
        r = b.classify(.505, .6, (.48, .50, 200, 200), .05, .02)
        self.assertTrue(r['mismatch'])
        self.assertTrue(r['good_buy'])
        self.assertLess(r['net_edge'], 0)
        r = b.classify(.505, .6, (.48, .50, 200, 200), None, None)
        self.assertTrue(r['mismatch'])
        self.assertIsNone(r['entry_cost'])

    def test_book_validation_and_unsorted_levels(self):
        book = dict(asset_id='1', timestamp=NOW.timestamp()*1000,
                    bids=[dict(price='.2', size=100), dict(price='.4', size=120)],
                    asks=[dict(price='.8', size=100), dict(price='.42', size=150)])
        self.assertEqual(b.top(book, '1', NOW), (.4, .42, 120, 150))
        with self.assertRaises(ValueError): b.top(book, '2', NOW)
        with self.assertRaises(ValueError): b.top(book, '1', NOW+timedelta(hours=1))
        book['bids'].append(dict(price='.9', size=100))
        with self.assertRaises(ValueError): b.top(book, '1', NOW)

    def test_exact_match_rejects_props_wrong_dates_and_started_games(self):
        event = dict(title='Georgia vs. Alabama', slug='cfb-uga-bama-2026-09-22')
        m = dict(question=event['title'], sportsMarketType='moneyline', active=True,
                 closed=False, acceptingOrders=True, outcomes=['Georgia','Alabama'],
                 clobTokenIds=['1','2'], gameStartTime='2026-09-22T12:00:00Z')
        g = dict(id='1', home='Alabama', away='Georgia', kickoff=m['gameStartTime'],
                 played=False, unrated=False)
        self.assertEqual(b.match(event, m, [g], NOW)[0], g)
        for patch in [dict(question='First half'), dict(clobTokenIds=['1','1']),
                      dict(gameStartTime='2026-09-21T11:00:00Z'), dict(closed=True),
                      dict(gameStartTime='2026-09-23T12:00:00Z')]:
            with self.assertRaises(ValueError): b.match(event, m | patch, [g], NOW)

    def test_history_needs_a_real_day_and_recent_point(self):
        t = NOW.timestamp()
        self.assertAlmostEqual(b.movement([dict(t=t-86400,p=.4),dict(t=t,p=.43)], NOW), .03)
        self.assertIsNone(b.movement([dict(t=t-100,p=.4),dict(t=t,p=.43)], NOW))
        self.assertIsNone(b.movement([dict(t=t-86400,p=.4),dict(t=t-8000,p=.43)], NOW))

    def test_stale_model_rejected(self):
        with self.assertRaisesRegex(ValueError, 'older'):
            b.build(dict(generated_at=NOW.isoformat(), data_source={}), now=NOW, events=[])

    def test_build_maps_reversed_outcomes_and_deduplicates(self):
        model = dict(generated_at=NOW.isoformat(), inseason={'through_week':3},
                     data_source={'checked_at':NOW.isoformat()}, games=[dict(id='1',
                     home='Alabama', away='Georgia', kickoff='2026-09-22T12:00:00Z',
                     played=False, unrated=False, neutral=False, p_home_current=.35)])
        market = dict(question='Georgia vs. Alabama', sportsMarketType='moneyline',
                      active=True, closed=False, acceptingOrders=True,
                      outcomes='["Georgia", "Alabama"]', clobTokenIds='["1", "2"]',
                      gameStartTime='2026-09-22T12:00:00Z', feesEnabled=True,
                      feeSchedule=dict(rate=.05, exponent=1))
        event = dict(title=market['question'], slug='cfb-uga-bama-2026-09-22', markets=[market])
        def fetch(url, **params):
            if url.endswith('/prices-history'):
                return {'history':[dict(t=NOW.timestamp()-86400,p=.36),dict(t=NOW.timestamp(),p=.39)]}
            token = params['token_id']
            return dict(asset_id=token,timestamp=NOW.timestamp()*1000,
                        bids=[dict(price=.38 if token=='1' else .6,size=200)],
                        asks=[dict(price=.4 if token=='1' else .62,size=200)])
        report = b.build(model, NOW, fetch, [event,event])
        self.assertEqual(report['matched_games'], 1)
        self.assertEqual(len(report['rows']), 2)
        self.assertEqual(report['rows'][0]['team'], 'Georgia')
        self.assertAlmostEqual(report['rows'][0]['model_probability'], .65)
        self.assertTrue(report['rows'][0]['mismatch'])
        market['feeSchedule']['exponent'] = 2
        report = b.build(model, NOW, fetch, [event])
        self.assertEqual(len(report['rows']), 2)
        self.assertIn('Fees unavailable', report['rows'][0]['notes'])

    def test_refresh_cadence_and_eastern_date(self):
        path = pathlib.Path(__file__).resolve().parents[1] / 'scripts/refresh_due.py'
        spec = importlib.util.spec_from_file_location('refresh_due', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = dict(games=[dict(kickoff='2026-09-27T01:00:00Z')], teams=[])
        self.assertTrue(module.due(model, datetime(2026,9,26,18,37,tzinfo=timezone.utc)))
        self.assertFalse(module.due(model, datetime(2026,9,25,18,37,tzinfo=timezone.utc)))
        self.assertTrue(module.due(model, datetime(2026,9,25,18,7,tzinfo=timezone.utc)))
        self.assertTrue(module.due(model, NOW, 'push'))


if __name__ == '__main__': unittest.main()
