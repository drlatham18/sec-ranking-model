import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import availability
import export_data
import matchup
import schedule_data
import weekly
import pandas as pd
from scipy.stats import norm

class LiveForecastTests(unittest.TestCase):
    def test_cli_uses_current_ratings_and_matching_calibration(self):
        data = json.loads((ROOT / 'output/app_data.json').read_text())
        if not data.get('inseason'):
            self.skipTest('No in-season snapshot')
        result = matchup.matchup('Alabama', 'Tennessee', 'Tennessee', data['season'])
        ratings = {r['team']: r['current_rating'] for r in data['all_ratings']}
        c = data['inseason']['calibration']
        margin = c['rating_diff_coef'] * (ratings['Alabama']-ratings['Tennessee']) - c['home_field_advantage']
        self.assertAlmostEqual(result['projected_margin_a'], margin, places=2)
        self.assertAlmostEqual(result['win_prob_a'], norm.cdf(margin/c['residual_sd']), places=4)
        self.assertEqual(result['calibration_method'], 'current results blend')
        self.assertFalse(result['availability_adjusted'])

    def test_weekly_callback_does_not_receive_future_cutoff(self):
        games = [dict(home='A', away='B', week=1, p_home=.5, played=True,
                      home_points=7, away_points=10, conf_game=True),
                 dict(home='A', away='B', week=2, p_home=.5, played=False,
                      home_points=None, away_points=None, conf_game=True)]
        seen = []
        def asof(week):
            seen.append(week)
            return [dict(g, p_home=.5 if week==-1 else .2) for g in games]
        history = weekly.history(games, ['A','B'], 100, asof)
        self.assertEqual(seen, [-1,1])
        self.assertEqual(history[0]['teams'][0]['proj_wins'], 1)
        self.assertEqual(history[1]['teams'][0]['proj_wins'], .2)

    def test_export_historical_ratings_exclude_later_results(self):
        old = json.loads((ROOT/'output/app_data.json').read_text())
        rows=[]
        for g in old['games']:
            rows.append(dict(id=g['id'], week=g['week'], start_date=g['kickoff'],
                home_team=g['home'], away_team=g['away'], home_conference=g['home_conf'],
                away_conference=g['away_conf'], neutral=g['neutral'], conference_game=g['conf_game'],
                completed=g['played'], home_points=g['home_points'], away_points=g['away_points']))
        # Independent synthetic finals, with known opposing evidence in a later week.
        for r in rows:
            r.update(completed=False,home_points=None,away_points=None)
        frame=pd.DataFrame(rows)
        first=frame.iloc[0].copy(); first.update(pd.Series(dict(completed=True,home_points=14,away_points=7)))
        frame.loc[0]=first
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)
            for path in (ROOT/'output').iterdir():
                if path.is_file(): (out/path.name).write_bytes(path.read_bytes())
            with patch.object(export_data,'OUT',out), patch.object(schedule_data,'fetch_consensus',return_value=(frame,{'checked_at':old['generated_at']})), patch.object(availability,'fetch_reports',return_value={'status':'unavailable','reports':[]}), patch.object(export_data,'N_SIMS',50):
                result=export_data.build()
        self.assertIsNotNone(result['inseason'])
        for game in result['games']:
            self.assertEqual(game['p_home'],game['p_home_current'])
            self.assertIn('p_home_preseason',game)
        self.assertEqual(result['neutral_grid'],result['current_grid'])
        self.assertIn('future results',result['history_method'])
        self.assertEqual(result['weekly_history'][0]['label'],'Preseason')
        # The first snapshot must still equal the original preseason probabilities.
        preseason_games=[dict(g,p_home=g['p_home_preseason']) for g in result['games']]
        expected=weekly.simulate(preseason_games,[t['team'] for t in result['teams']],50,through_week=-1)
        self.assertEqual(result['weekly_history'][0]['teams'],expected)

class AvailabilityTests(unittest.TestCase):
    def report(self):
        return {'1':dict(footer={'date':'2026-10-17'}, publishDate='2026-10-15',postedTime='19:10:00',
            ReportType='Update 1',games=[{'teamDisplayName':t,'rows':[{'name':'QB #1 Player','status':'Out'},
            {'name':'Private exemption','status':'Exempt'}]} for t in ['Alabama','Tennessee']])}
    def test_only_public_status_fields_and_timezone_are_kept(self):
        result=availability.parse_reports(self.report())[0]
        self.assertEqual(result['published_at'],'2026-10-16T00:10:00+00:00')
        self.assertEqual(result['teams'][0]['players'],[{'name':'QB #1 Player','status':'Out'}])
    def test_missing_or_changed_feed_is_unknown_not_healthy(self):
        with patch.object(availability.requests,'post',side_effect=availability.requests.ConnectionError):
            result=availability.fetch_reports()
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['reports'],[])
        self.assertFalse(result['adjustment_applied'])
        with self.assertRaises(ValueError): availability.parse_reports([])
