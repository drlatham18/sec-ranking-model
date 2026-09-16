import importlib.util, pathlib, unittest, tempfile, json
from datetime import datetime, timezone
P=pathlib.Path(__file__).resolve().parents[1]/'integrations/polydesk_compare.py'
spec=importlib.util.spec_from_file_location('comparison',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ComparisonTests(unittest.TestCase):
 def test_unsorted_orderbook_uses_executable_best_ask(self):
  self.assertEqual(m.book_top({'bids':[{'price':'.1','size':'2'},{'price':'.4','size':'3'}],'asks':[{'price':'.99','size':'1'},{'price':'.42','size':'10'},{'price':'.42','size':'5'}]}),(.4,.42,15))
 def test_crossed_book_rejected(self):
  with self.assertRaises(ValueError):m.book_top({'bids':[{'price':'.8','size':1}],'asks':[{'price':'.4','size':2}]})
 def test_home_effect_and_probability_complements(self):
  model={'all_ratings':[{'team':'A','projected_rating':0},{'team':'B','projected_rating':0}],'calibration':{'method':'walk-forward preseason predictions','rating_diff_coef':1,'home_field_advantage':3,'residual_sd':17,'neutral_residual_sd':16}}
  self.assertAlmostEqual(m.estimate(model,'A','B',True)['A'],.5)
  p=m.estimate(model,'A','B');self.assertGreater(p['A'],.5);self.assertAlmostEqual(sum(p.values()),1)
 def test_unknown_teams_do_not_get_default_ratings(self):
  with self.assertRaises(ValueError):m.estimate({'all_ratings':[],'calibration':{'method':'walk-forward preseason predictions'}},'A','B')
 def test_period_contract_cannot_match_full_game(self):
  with self.assertRaises(ValueError):m.match_game({'title':'A vs B'},{'question':'A vs B: 1H Moneyline','sportsMarketType':'moneyline','outcomes':['A','B']},{'events':[]},[])
 def test_timestamp_requires_timezone(self):
  self.assertIsNone(m.dt('2026-09-15'));self.assertIsNotNone(m.dt('2026-09-15T12:00:00Z'))
class SnapshotTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)
  self.now=datetime(2026,9,15,20,tzinfo=timezone.utc)
  self.model={'season':2026,'snapshot_type':'preseason','generated_at':'2026-08-30T12:00:00Z','all_ratings':[{'team':'A','projected_rating':0},{'team':'B','projected_rating':0}],'calibration':{'method':'walk-forward preseason predictions','rating_diff_coef':1,'home_field_advantage':3,'residual_sd':17,'neutral_residual_sd':16}}
  self.write('report.json',{'utc':'2026-09-15T19:59:00Z'})
  self.write('espn_ncaaf.json',{'events':[{'id':'1','date':'2026-09-19T16:00:00Z','status':{'type':{'state':'pre'}},'competitions':[{'neutralSite':False,'competitors':[{'homeAway':'home','team':{'location':'A'}},{'homeAway':'away','team':{'location':'B'}}]}]}]})
  self.event={'title':'A vs B','slug':'cfb-a-b','markets':[{'sportsMarketType':'moneyline','question':'A vs B','gameStartTime':'2026-09-19T16:00:00Z','outcomes':['A','B'],'clobTokenIds':['1','2'],'active':True,'closed':False,'acceptingOrders':True}]}
  self.write('evt_cfb-a-b.json',[self.event])
  for team,token in [('A','1'),('B','2')]:self.write('clob_cfb-a-b_'+team+'.json',{'token':token,'snapshot_complete':True,'retrieved_at':'2026-09-15T19:59:00Z','bids':[{'price':'.3','size':'10'}],'asks':[{'price':'.99','size':'1'},{'price':'.4','size':'20'}]})
 def tearDown(self): self.tmp.cleanup()
 def write(self,name,x):(self.root/name).write_text(json.dumps(x))
 def test_stale_model_and_venue_and_fees_block_execution(self):
  r=m.compare(self.model,self.root,self.now);self.assertEqual(len(r['rows']),2)
  for row in r['rows']:
   self.assertEqual(row['best_ask'],.4);self.assertFalse(row['execution_eligible']);self.assertIsNone(row['net_edge_pp'])
   for flag in ['model_snapshot_stale_or_undated','research_execution_venue_mismatch','fees_not_estimated']:self.assertIn(flag,row['blockers'])
 def test_legacy_truncated_books_are_rejected(self):
  p=self.root/'clob_cfb-a-b_A.json';book=json.loads(p.read_text());book.pop('snapshot_complete');self.write(p.name,book)
  r=m.compare(self.model,self.root,self.now);self.assertEqual(r['rows'],[]);self.assertIn('truncated',r['skips'][0]['reason'])
 def test_duplicate_tokens_rejected(self):
  self.event['markets'][0]['clobTokenIds']=['1','1'];self.write('evt_cfb-a-b.json',[self.event])
  r=m.compare(self.model,self.root,self.now);self.assertEqual(r['rows'],[])
 def test_expired_quote_is_flagged_even_if_report_is_fresh(self):
  p=self.root/'clob_cfb-a-b_A.json';book=json.loads(p.read_text());book['retrieved_at']='2026-09-14T19:59:00Z';self.write(p.name,book)
  r=m.compare(self.model,self.root,self.now);self.assertIn('book_timestamp_stale_or_missing',r['rows'][0]['blockers'])

if __name__=='__main__':unittest.main()
