import pathlib
import sys
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import schedule_data


def cfbd_game(i, week, home, away, hp, ap, completed=True):
    return dict(id=i, week=week, start_date="2026-09-05", home_team=home,
                away_team=away, completed=completed, home_points=hp, away_points=ap,
                home_conference="SEC", away_conference="SEC",
                neutral_site=False, conference_game=True)


def frame(rows):
    return pd.DataFrame([dict(id=str(i), week=w, start_date="2026-09-05",
                              home_team=h, away_team=a, home_conference="SEC",
                              away_conference="SEC", home_points=hp, away_points=ap,
                              completed=c, neutral=False, conference_game=True)
                         for i, w, h, a, hp, ap, c in rows])


class ScorelessGuardTests(unittest.TestCase):
    def test_a_completed_game_with_no_score_is_treated_as_not_played(self):
        data = [cfbd_game(1, 1, "Georgia", "Clemson", 30, 10),
                cfbd_game(2, 1, "LSU", "Tulane", None, None)]   # cancelled, still "completed"
        with patch.object(schedule_data.BD, "safe", return_value=data):
            out, _ = schedule_data.fetch_schedule(2026, "cfbd")
        self.assertTrue(bool(out[out.id == "1"].completed.iloc[0]))
        self.assertFalse(bool(out[out.id == "2"].completed.iloc[0]))

    def test_a_feed_full_of_scoreless_finals_still_fails_loudly(self):
        data = [cfbd_game(i, 1, "H%d" % i, "A%d" % i, None, None) for i in range(40)]
        with patch.object(schedule_data.BD, "safe", return_value=data):
            with self.assertRaises(ValueError) as caught:
                schedule_data.fetch_schedule(2026, "cfbd")
        self.assertIn("no final score", str(caught.exception))


class CrossCheckTests(unittest.TestCase):
    def test_matching_finals_are_kept_and_counted_as_agreed(self):
        a = frame([(1, 1, "Georgia", "Clemson", 30, 10, True)])
        b = frame([(9, 1, "Georgia", "Clemson", 30, 10, True)])
        out, audit = schedule_data.cross_check(a, b)
        self.assertEqual(audit["agreed"], 1)
        self.assertEqual(audit["disagreed"], 0)
        self.assertTrue(bool(out.completed.iloc[0]))

    def test_a_disputed_final_is_unplayed_rather_than_published(self):
        a = frame([(1, 1, "Georgia", "Clemson", 30, 10, True)])
        b = frame([(9, 1, "Georgia", "Clemson", 31, 10, True)])
        out, audit = schedule_data.cross_check(a, b)
        self.assertEqual(audit["disagreed"], 1)
        self.assertFalse(bool(out.completed.iloc[0]))
        self.assertTrue(pd.isna(out.home_points.iloc[0]))
        self.assertEqual(audit["conflicts"][0]["home"], "Georgia")

    def test_differently_named_schools_are_reported_unverified_not_dropped(self):
        a = frame([(1, 1, "Vanderbilt", "Austin Peay", 45, 3, True)])
        b = frame([(9, 1, "Vanderbilt", "Austin Peay St", 45, 3, True)])
        out, audit = schedule_data.cross_check(a, b)
        self.assertEqual(audit["unverified"], 1)
        self.assertEqual(audit["agreed"], 0)
        self.assertTrue(bool(out.completed.iloc[0]))   # kept, just not audited
        self.assertEqual(len(out), 1)                  # and never duplicated


class ConsensusTests(unittest.TestCase):
    def test_cfbd_is_the_spine_and_costs_exactly_one_request(self):
        """The key is capped monthly, so a build must not spend more than one."""
        calls = []

        def safe(endpoint, **kw):
            calls.append(endpoint)
            return [cfbd_game(1, 1, "Georgia", "Clemson", 30, 10)]

        with patch.object(schedule_data.BD, "safe", side_effect=safe), \
                patch.object(schedule_data, "espn_week_rows", return_value=[]), \
                patch.dict("os.environ", {"CFBD_API_KEY": "x"}):
            out, info = schedule_data.fetch_consensus(2026, "auto")
        self.assertEqual(calls, ["games"])
        self.assertEqual(info["provider"], "cfbd")
        self.assertEqual(len(out), 1)

    def test_a_cfbd_failure_falls_through_to_espn_instead_of_killing_the_build(self):
        def boom(*a, **k):
            raise RuntimeError("CFBD 500")

        espn = [dict(id="9", week=1, start_date="2026-09-05", home_team="Georgia",
                     away_team="Clemson", home_conference="SEC", away_conference="SEC",
                     home_points=30, away_points=10, completed=True,
                     neutral=False, conference_game=True)]
        with patch.object(schedule_data.BD, "safe", side_effect=boom), \
                patch.object(schedule_data, "espn_week_rows", return_value=espn), \
                patch.dict("os.environ", {"CFBD_API_KEY": "x"}):
            out, info = schedule_data.fetch_consensus(2026, "auto")
        self.assertEqual(info["provider"], "espn")
        self.assertFalse(info["sources"]["cfbd"]["ok"])
        self.assertTrue(info["sources"]["espn"]["ok"])
        self.assertEqual(len(out), 1)

    def test_both_sources_failing_is_not_survivable(self):
        def boom(*a, **k):
            raise RuntimeError("down")
        with patch.object(schedule_data.BD, "safe", side_effect=boom), \
                patch.object(schedule_data, "espn_week_rows", side_effect=boom), \
                patch.dict("os.environ", {"CFBD_API_KEY": "x"}):
            with self.assertRaises(ValueError):
                schedule_data.fetch_consensus(2026, "auto")

    def test_without_a_key_espn_is_used_and_cfbd_is_never_called(self):
        def safe(*a, **k):
            raise AssertionError("must not spend a CFBD request without a key")
        espn = [dict(id="9", week=1, start_date="2026-09-05", home_team="Georgia",
                     away_team="Clemson", home_conference="SEC", away_conference="SEC",
                     home_points=30, away_points=10, completed=True,
                     neutral=False, conference_game=True)]
        with patch.object(schedule_data.BD, "safe", side_effect=safe), \
                patch.object(schedule_data, "espn_week_rows", return_value=espn), \
                patch.dict("os.environ", {}, clear=True), \
                patch.object(schedule_data.pathlib.Path, "home",
                             return_value=pathlib.Path("/nonexistent")):
            out, info = schedule_data.fetch_consensus(2026, "auto")
        self.assertEqual(info["provider"], "espn")


if __name__ == "__main__":
    unittest.main()
