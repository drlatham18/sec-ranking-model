import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import accuracy

CAL = {"K": 5, "scale_b1": 0.9, "scale_hfa": 3.0, "rating_diff_coef": 0.9,
       "home_field_advantage": 2.0, "residual_sd": 16.0, "min_week_validated": 2}
PRIOR = {"intercept": 0.0, "rating_diff_coef": 0.9, "home_field_advantage": 2.0,
         "residual_sd": 17.0, "neutral_residual_sd": 16.0}
RATINGS = {"A": 20.0, "B": 0.0, "C": 10.0, "D": -5.0}
AUDITED = {"provider": "cfbd", "audit": {"performed": True, "disagreed": 0}}


def game(week, home, away, hp=None, ap=None, neutral=False):
    return {"week": week, "home": home, "away": away, "home_points": hp,
            "away_points": ap, "neutral": neutral, "completed": hp is not None}


class WeeklyAccuracyTests(unittest.TestCase):
    def test_week_is_scored_without_its_own_results(self):
        """A week-2 upset must not leak into the ratings that predicted it."""
        games = [game(1, "A", "D", 30, 10), game(1, "C", "B", 24, 20),
                 game(2, "B", "A", 40, 0), game(2, "D", "C", 3, 28)]
        leaky = [g for g in games if g["week"] == 1]
        week2 = accuracy.weekly(games, RATINGS, CAL, PRIOR, source_info=AUDITED)[0]
        # Same prediction as scoring week 2 with only week-1 results available.
        clean = accuracy.weekly(leaky + [game(2, "B", "A", 40, 0), game(2, "D", "C", 3, 28)],
                                RATINGS, CAL, PRIOR, source_info=AUDITED)[0]
        self.assertEqual(week2, clean)
        self.assertEqual(week2["model"]["n"], 2)
        self.assertEqual(week2["model"]["correct"], 1)
        self.assertEqual(week2["misses"][0]["favorite"], "A")
        self.assertEqual(week2["flags"], [])

    def test_weeks_before_validation_are_not_scored(self):
        out = accuracy.weekly([game(1, "A", "B", 30, 0)], RATINGS, CAL, PRIOR,
                              source_info=AUDITED)
        self.assertEqual(out, [])

    def test_missing_data_is_flagged_not_hidden(self):
        games = [game(1, "A", "B", 30, 0), game(2, "A", "FCS U", 50, 0),
                 game(2, "C", "B"), game(2, "B", "C", 10, 7)]
        week = accuracy.weekly(games, RATINGS, CAL, PRIOR,
                               source_info={"provider": "espn", "audit": {"performed": False}})[0]
        flags = " ".join(week["flags"])
        self.assertIn("not cross-checked", flags)
        self.assertIn("not final", flags)
        self.assertIn("unrated", flags)
        self.assertIn("no game in the feed", flags)
        self.assertEqual(week["idle_teams"], ["D"])
        self.assertFalse(week["complete"])
        self.assertEqual(week["model"]["n"], 1)

    def test_score_conflicts_are_flagged(self):
        games = [game(1, "A", "B", 30, 0), game(2, "C", "D", 20, 10)]
        week = accuracy.weekly(games, RATINGS, CAL, PRIOR,
                               source_info={"provider": "cfbd",
                                            "audit": {"performed": True, "disagreed": 2}})[0]
        self.assertTrue(any("conflict" in f for f in week["flags"]))

    def test_conference_subset(self):
        games = [game(1, "A", "B", 30, 0), game(2, "A", "C", 30, 0), game(2, "B", "D", 7, 3)]
        week = accuracy.weekly(games, RATINGS, CAL, PRIOR, conference_of={"A": "SEC"},
                               conference="SEC", source_info=AUDITED)[0]
        self.assertEqual(week["conference"]["model"]["n"], 1)
        self.assertEqual(week["model"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
