import pathlib
import sys
import unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import strict_rank as sr


def game(h,a,hp,ap,week=1,**kw):
    return dict(home=h,away=a,home_points=hp,away_points=ap,week=week,neutral=True,**kw)

class StrictRankingTests(unittest.TestCase):
    def setUp(self):
        self.games=[game('A','B',28,14,1),game('B','C',28,14,2),game('C','A',14,28,3)]
        self.teams=['A','B','C','Idle']
        self.cal=dict(margin_cap=28,shrinkage=1.,coefficients=dict(production=1.,sor_score=2.))

    def test_exact_record_probability_and_schedule_difficulty(self):
        self.assertAlmostEqual(sr.win_tail([.5,.5],2),.25)
        self.assertAlmostEqual(sr.win_tail([.5,.5],1),.75)
        self.assertLess(sr.win_tail([.3,.4],2),sr.win_tail([.8,.9],2))

    def test_scoring_production_distinguishes_identical_records(self):
        close=[game('A','B',21,20)]
        big=[game('A','B',35,7)]
        x,_=sr.rank(close,['A','B'],self.cal)
        y,_=sr.rank(big,['A','B'],self.cal)
        self.assertAlmostEqual(x[0]['sor_score'],y[0]['sor_score'])
        self.assertGreater(y[0]['rating'],x[0]['rating'])

    def test_blowout_padding_stops_at_cap(self):
        a,_=sr.rank([game('A','B',35,7)],['A','B'],self.cal)
        b,_=sr.rank([game('A','B',77,7)],['A','B'],self.cal)
        self.assertAlmostEqual(a[0]['rating'],b[0]['rating'])

    def test_schedule_quality_changes_credit_for_same_record_and_margin(self):
        games=[game('A','B',28,14),game('C','D',28,14),
               game('B','E',35,7,2),game('D','E',7,35,3)]
        rows,_=sr.rank(games,list('ABCDE'),self.cal)
        d={r['team']:r for r in rows}
        self.assertGreater(d['A']['production'],d['C']['production'])
        self.assertGreater(d['A']['sor_score'],d['C']['sor_score'])
        self.assertGreater(d['A']['rating'],d['C']['rating'])

    def test_no_game_no_rank_and_no_poll_or_preseason_inputs(self):
        rows,_=sr.rank(self.games,self.teams,self.cal)
        self.assertEqual([r['team'] for r in rows],list('ABC'))
        self.assertTrue(all(r['from_results']==1 and r['preseason_rank'] is None for r in rows))
        self.assertTrue(all(np.isfinite(r['rating']) for r in rows))

    def test_unfinished_and_unrelated_lower_division_games_have_no_effect(self):
        before,_=sr.rank(self.games,self.teams,self.cal)
        after,_=sr.rank(self.games+[game('A','C',70,0,4,completed=False),
                                    game('X','Y',70,0,5)],self.teams,self.cal)
        self.assertEqual(before,after)

    def test_duplicate_results_fail_closed(self):
        with self.assertRaises(ValueError): sr.rank(self.games+self.games,self.teams,self.cal)

    def test_fcs_games_count_and_are_labeled(self):
        rows,_=sr.rank([game('A','FCS',49,0)],['A'],self.cal)
        self.assertEqual(rows[0]['record'],'1-0')
        self.assertEqual(rows[0]['fcs_games'],1)

    def test_renaming_teams_does_not_change_scores(self):
        before,_=sr.rank(self.games,self.teams,self.cal)
        rename={'A':'Z','B':'Y','C':'X','Idle':'Unused'}
        games=[dict(g,home=rename[g['home']],away=rename[g['away']]) for g in self.games]
        after,_=sr.rank(games,list(rename.values()),self.cal)
        self.assertEqual([rename[r['team']] for r in before],[r['team'] for r in after])
        for a,b in zip(before,after): self.assertAlmostEqual(a['rating'],b['rating'],places=5)

if __name__=='__main__': unittest.main()
