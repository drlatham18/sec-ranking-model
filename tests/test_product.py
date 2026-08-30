import json
import pathlib
import unittest

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PROC = ROOT / "data" / "processed"


class ProductIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = pd.read_csv(PROC / "panel.csv")
        cls.selection = json.loads((OUT / "selection.json").read_text())
        cls.app = json.loads((OUT / "app_data.json").read_text())

    def test_panel_contains_only_unique_teams(self):
        self.assertFalse(self.panel.duplicated(["season", "team"]).any())
        names = self.panel.team.astype(str).str.lower()
        self.assertNotIn("nationalaverages", set(names))
        self.assertGreaterEqual(self.panel.groupby("season").team.nunique().min(), 100)

    def test_accuracy_claim_uses_untouched_holdout(self):
        dev = set(self.selection["development_seasons"])
        hold = set(self.selection["holdout_seasons"])
        self.assertFalse(dev & hold)
        self.assertEqual(self.selection["performance_claim"]["split"],
                         "untouched holdout")
        self.assertEqual(hold, set(self.selection["performance_claim"]["seasons"]))

    def test_matchup_probabilities_have_holdout_metrics(self):
        metrics = self.selection["calibration"]["holdout_metrics"]
        self.assertGreater(metrics["n_games"], 1000)
        self.assertGreater(metrics["brier"], 0)
        self.assertLess(metrics["brier"], 0.25)
        self.assertGreater(len(metrics["calibration_bins"]), 5)

    def test_product_payload_is_clean_and_explainable(self):
        ratings = self.app["all_ratings"]
        names = {r["team"].lower() for r in ratings}
        self.assertNotIn("nationalaverages", names)
        self.assertEqual(names, {n.lower() for n in self.app["explanations"]})
        self.assertIn(self.app["snapshot_type"], {"preseason", "live"})
        self.assertTrue(self.app["generated_at"].endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
