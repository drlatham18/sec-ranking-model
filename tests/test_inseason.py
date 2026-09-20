import json
import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import inseason
import schedule_data

CAL = {"K": 5, "scale_b1": 0.9, "scale_hfa": 3.0,
       "rating_diff_coef": 0.9, "home_field_advantage": 2.0,
       "residual_sd": 16.0, "playable_threshold": 0.8}


def game(home, away, hp, ap, neutral=False):
    return dict(home=home, away=away, home_points=hp, away_points=ap, neutral=neutral)


class ImpliedRatingTests(unittest.TestCase):
    def test_inverts_the_game_model_exactly(self):
        """A team that performs exactly to the model implies its own rating back."""
        ratings = {"A": 10.0, "B": 2.0}
        # model margin for A at home = 0.9*(10-2) + 3.0 = 10.2
        rows = inseason.implied_ratings([game("A", "B", 30, 20)], ratings, 0.9, 3.0)
        # actual margin 10 vs modelled 10.2 -> A implied a shade under its rating
        self.assertAlmostEqual(rows["A"][0], 2.0 + (10 - 3.0) / 0.9, places=9)
        self.assertAlmostEqual(rows["B"][0], 10.0 + (-10 + 3.0) / 0.9, places=9)

    def test_neutral_site_carries_no_home_field(self):
        ratings = {"A": 0.0, "B": 0.0}
        home = inseason.implied_ratings([game("A", "B", 10, 0)], ratings, 1.0, 3.0)
        neut = inseason.implied_ratings([game("A", "B", 10, 0, True)], ratings, 1.0, 3.0)
        self.assertAlmostEqual(home["A"][0], 7.0)
        self.assertAlmostEqual(neut["A"][0], 10.0)

    def test_unrated_opponents_and_unplayed_games_are_skipped(self):
        ratings = {"A": 5.0}
        rows = inseason.implied_ratings(
            [game("A", "FCS School", 50, 3), game("A", "A", None, None)], ratings, .9, 3.)
        self.assertEqual(rows, {})


class BlendTests(unittest.TestCase):
    def test_no_games_leaves_the_preseason_rating_untouched(self):
        out = inseason.blend({"A": 7.5}, {}, 5)
        self.assertEqual(out["A"]["blended"], 7.5)
        self.assertEqual(out["A"]["n_games"], 0)
        self.assertEqual(out["A"]["weight"], 0.0)
        self.assertIsNone(out["A"]["in_season"])

    def test_weight_follows_n_over_n_plus_k_and_moves_toward_results(self):
        out = inseason.blend({"A": 0.0}, {"A": [10.0, 10.0, 10.0]}, 3)
        self.assertAlmostEqual(out["A"]["weight"], 0.5)
        self.assertAlmostEqual(out["A"]["blended"], 5.0)
        self.assertAlmostEqual(out["A"]["in_season"], 10.0)

    def test_more_games_shrink_less(self):
        few = inseason.blend({"A": 0.0}, {"A": [20.0]}, 5)["A"]["blended"]
        many = inseason.blend({"A": 0.0}, {"A": [20.0] * 10}, 5)["A"]["blended"]
        self.assertLess(few, many)
        self.assertLess(many, 20.0)          # never fully abandons the prior

    def test_rejects_a_non_positive_shrinkage_constant(self):
        with self.assertRaises(ValueError):
            inseason.blend({"A": 1.0}, {}, 0)


class CalibrationWiringTests(unittest.TestCase):
    def test_missing_calibration_fails_loudly_instead_of_guessing(self):
        with self.assertRaises(FileNotFoundError):
            inseason.load_calibration(ROOT / "output" / "does_not_exist.json")

    def test_shipped_calibration_has_everything_the_product_reads(self):
        c = inseason.load_calibration()
        for key in ("K", "scale_b1", "scale_hfa", "rating_diff_coef",
                    "home_field_advantage", "residual_sd", "holdout_metrics",
                    "preseason_baseline_holdout", "confidence_tiers"):
            self.assertIn(key, c)
        self.assertGreater(c["K"], 0)
        # The blend must actually beat the preseason baseline it replaces.
        self.assertLess(c["holdout_metrics"]["brier"],
                        c["preseason_baseline_holdout"]["brier"])
        self.assertGreater(c["holdout_metrics"]["accuracy"],
                           c["preseason_baseline_holdout"]["accuracy"])

    def test_playable_threshold_is_backed_by_its_confidence_interval(self):
        c = inseason.load_calibration()
        if c["playable_threshold"] is None:
            self.skipTest("no tier reaches the target")
        tier = next(t for t in c["confidence_tiers"]
                    if t["min_confidence"] == c["playable_threshold"])
        # the claim rests on the lower bound, not the point estimate
        self.assertGreaterEqual(tier["ci_low"], c["target_accuracy"])

    def test_current_ratings_end_to_end(self):
        out = inseason.current_ratings(
            [game("A", "B", 40, 10)], {"A": 5.0, "B": 0.0}, CAL)
        self.assertGreater(out["A"]["blended"], 5.0)
        self.assertLess(out["B"]["blended"], 0.0)
        self.assertEqual(out["A"]["n_games"], 1)


class FbsResultsTests(unittest.TestCase):
    def test_espn_pull_covers_every_fbs_conference_and_keeps_only_finals(self):
        calls = []

        def fake(url, params=None, timeout=None):
            calls.append((params["group"] if "group" in params else params["groups"],
                          params["week"]))
            g, w = params["groups"], params["week"]
            return FakeResp({"events": [{
                "id": "g%s-%s" % (g, w), "date": "2026-09-05T16:00Z",
                "season": {"year": 2026, "type": 2}, "week": {"number": w},
                "status": {"type": {"completed": w == 1, "state": "post" if w == 1 else "pre"}},
                "competitions": [{"neutralSite": False, "conferenceCompetition": True,
                    "competitors": [
                        {"homeAway": "home", "score": "21", "team": {"location": "A", "conferenceId": "8"}},
                        {"homeAway": "away", "score": "7", "team": {"location": "B", "conferenceId": "8"}}]}],
            }]})

        with patch.object(schedule_data.requests, "get", side_effect=fake):
            frame = schedule_data.fbs_results(2026, 2, "espn")
        groups = {g for g, _ in calls}
        self.assertEqual(groups, set(schedule_data.ESPN_FBS_GROUPS))
        self.assertEqual({w for _, w in calls}, {1, 2})
        self.assertTrue(frame.completed.all())


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status
        self.url = "https://site.api.espn.com/scoreboard"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise schedule_data.requests.HTTPError(str(self.status_code))

    def json(self):
        return self._p


if __name__ == "__main__":
    unittest.main()
