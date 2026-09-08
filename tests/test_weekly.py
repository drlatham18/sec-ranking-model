import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import schedule_data
import weekly
import export_data


def game(home="A", away="B", week=1, p=.7, played=False, hp=None, ap=None, conf=True):
    return dict(home=home, away=away, week=week, p_home=p, played=played,
                home_points=hp, away_points=ap, conf_game=conf, unrated=False)


class WeeklyResultsTests(unittest.TestCase):
    def test_only_explicit_finals_lock_scores(self):
        payload = json.loads((ROOT / "tests/fixtures/espn_week1.json").read_text())
        final = schedule_data.espn_rows(payload, 2026)
        self.assertEqual(final[0]["home_points"], 54)
        self.assertTrue(final[0]["completed"])
        for state in ["pre", "in"]:
            changed = copy.deepcopy(payload)
            changed["events"][0]["status"]["type"].update(completed=False, state=state)
            row = schedule_data.espn_rows(changed, 2026)[0]
            self.assertFalse(row["completed"])
            self.assertIsNone(row["home_points"])
            self.assertIsNone(row["away_points"])
        self.assertEqual(schedule_data.espn_rows(payload, 2025), [])

    def test_cfbd_uses_completion_flag_and_bypasses_cache(self):
        data = [dict(id=1, week=1, start_date="2026-09-05", home_team="A", away_team="B",
                     completed=False, home_points=10, away_points=3)]
        with patch.object(schedule_data.BD, "safe", return_value=data) as fetch:
            frame, metadata = schedule_data.fetch_schedule(2026, "cfbd", True)
        self.assertFalse(frame.iloc[0].completed)
        self.assertIsNone(frame.iloc[0].home_points)
        self.assertTrue(fetch.call_args.kwargs["refresh"])
        self.assertTrue(fetch.call_args.kwargs["_required"])

    def test_weekly_deltas_use_finals_without_future_results(self):
        games = [game(played=True, hp=7, ap=10),
                 game(home="A", away="FCS", week=2, p=.9, played=True, hp=30, ap=0, conf=False),
                 game(home="A", away="B", week=3, p=.6)]
        history = weekly.history(games, ["A", "B"], 1000)
        self.assertEqual([s["week"] for s in history], [-1, 1, 2])
        rows = [{t["team"]: t for t in s["teams"]} for s in history]
        self.assertAlmostEqual(rows[0]["A"]["proj_wins"], 2.2)
        self.assertAlmostEqual(rows[1]["A"]["proj_wins"], 1.5)
        self.assertEqual(rows[1]["A"]["wins"], 0)
        self.assertEqual(rows[1]["A"]["losses"], 1)
        self.assertAlmostEqual(rows[2]["A"]["proj_wins"], 1.6)
        self.assertEqual(rows[2]["A"]["wins"], 1)
        self.assertEqual(history, weekly.history(games, ["A", "B"], 1000))

    def test_nonconference_result_does_not_change_conference_odds(self):
        games = [game(away="FCS", conf=False), game(week=2)]
        before = weekly.simulate(games, ["A", "B"], 1000)
        games[0].update(played=True, home_points=14, away_points=7)
        after = weekly.simulate(games, ["A", "B"], 1000)
        for old, new in zip(before, after):
            self.assertEqual(old["p_conf_leader"], new["p_conf_leader"])
            self.assertEqual(old["proj_conf_wins"], new["proj_conf_wins"])

    def test_partial_week_never_replaces_previous_full_week(self):
        games = [game(played=True, hp=14, ap=7),
                 game(week=2, played=True, hp=7, ap=10),
                 game(home="C", away="D", week=2)]
        snapshots = weekly.history(games, ["A", "B", "C", "D"], 100)
        self.assertEqual(snapshots[-2]["label"], "Week 1")
        self.assertFalse(snapshots[-1]["complete"])
        self.assertEqual(snapshots[-1]["label"], "Current (partial week)")

    def test_unrated_final_is_recorded_as_actual_loss(self):
        g = game(away="FCS", p=.92, played=True, hp=7, ap=14, conf=False)
        g["unrated"] = True
        team = weekly.simulate([g], ["A"], 100)[0]
        self.assertEqual(team["wins"], 0)
        self.assertEqual(team["losses"], 1)
        self.assertEqual(team["proj_wins"], 0)
        self.assertEqual(team["win_distribution"][0], 1)

    def test_incomplete_feed_preserves_published_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            out = pathlib.Path(directory)
            for name in ("ratings_2026.csv", "selection.json", "calibration.json", "validation.csv", "app_data.json"):
                (out / name).write_bytes((ROOT / "output" / name).read_bytes())
            original = (out / "app_data.json").read_bytes()
            partial = pd.DataFrame([dict(home_team="Alabama", away_team="Georgia")])
            with patch.object(export_data, "OUT", out), patch.object(schedule_data, "fetch_schedule", return_value=(partial, {})):
                with self.assertRaisesRegex(ValueError, "schedule"):
                    export_data.build()
            self.assertEqual(original, (out / "app_data.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
