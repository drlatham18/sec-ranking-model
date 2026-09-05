import json
import pathlib
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import pandas as pd
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
import cfbd_client as api
import fit
import build_dataset as dataset

class Refresh(unittest.TestCase):
    def test_dynamic_cache_expiry_and_force(self):
        with tempfile.TemporaryDirectory() as d, patch.object(api,'CACHE',pathlib.Path(d)), patch.object(api,'_key',return_value='fake'), patch.object(api.requests,'get') as request, patch.dict(api.os.environ,{'CFBD_REFRESH':'0'}):
            request.return_value=Mock(status_code=200,json=Mock(return_value=[{'score':1}]))
            year=datetime.now(timezone.utc).year
            self.assertEqual(api.get('games',year=year),[{'score':1}])
            api.get('games',year=year)
            self.assertEqual(request.call_count,1)
            path=next(pathlib.Path(d).glob('*.json'))
            cached=json.loads(path.read_text());cached['fetched_at']-=7*3600
            path.write_text(json.dumps(cached))
            api.get('games',year=year)
            self.assertEqual(request.call_count,2)
            api.get('games',year=year,refresh=True)
            self.assertEqual(request.call_count,3)
            path.write_text('[{"legacy":true}]')
            api.get('games',year=year)
            self.assertEqual(request.call_count,4)
    def test_season_requires_fraction_and_calendar(self):
        with tempfile.TemporaryDirectory() as d, patch.object(fit,'PROC',pathlib.Path(d)):
            root=pathlib.Path(d)
            pd.DataFrame({'season':[2024]*700+[2025]*120+[2026]*700}).to_csv(root/'games.csv',index=False)
            pd.DataFrame([{'season':2024,'scheduled':750,'completed':700},
                          {'season':2025,'scheduled':750,'completed':120},
                          {'season':2026,'scheduled':750,'completed':700}]).to_csv(root/'season_status.csv',index=False)
            self.assertEqual(fit.last_completed_season(datetime(2026,9,5,tzinfo=timezone.utc)),2024)
            self.assertEqual(fit.last_completed_season(datetime(2027,1,20,tzinfo=timezone.utc)),2024)
            self.assertEqual(fit.last_completed_season(datetime(2027,2,1,tzinfo=timezone.utc)),2026)
    def test_missing_provenance_fails(self):
        with tempfile.TemporaryDirectory() as d, patch.object(fit,'PROC',pathlib.Path(d)):
            with self.assertRaisesRegex(RuntimeError,'rebuild'): fit.last_completed_season()
    def test_schedule_counts_unplayed_and_excludes_in_progress(self):
        schedule=[{'home_team':'A','away_team':'B','home_points':10,'away_points':3,'completed':True},
                  {'home_team':'C','away_team':'D','home_points':0,'away_points':0,'completed':False},
                  {'home_team':'E','away_team':'F','home_points':None,'away_points':None}]
        status=[]
        with patch.object(dataset,'safe',return_value=schedule): rows=dataset.games(2026,status)
        self.assertEqual(len(rows),1)
        self.assertEqual((status[0]['completed'],status[0]['scheduled']),(1,3))

if __name__=='__main__': unittest.main()
