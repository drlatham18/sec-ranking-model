"""Tests for timestamped ATS export and missing-line behavior."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import export_predictions as p

class PredictionExportTests(unittest.TestCase):
    def test_cover_probability_uses_spread_not_win_probability(self):
        self.assertAlmostEqual(p.normal_cdf(0), 0.5)
        self.assertGreater(p.normal_cdf(1), 0.5)

    def test_snapshot_and_missing_spread(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "output"
            out.mkdir()
            spreads = root / "spreads.json"
            spreads.write_text(json.dumps([{"home": "Florida", "away": "Oklahoma",
                                            "home_spread": -6.5, "source": "test"}]))
            model = {
                "season": 2026, "conference": "SEC",
                "generated_at": "2026-11-07T13:00:00+00:00",
                "data_source": {"checked_at": "2026-11-07T12:59:00Z"},
                "games": [
                    {"id": "1", "kickoff": "2026-11-07T19:00Z",
                     "home": "Florida", "away": "Oklahoma", "neutral": False,
                     "played": False, "margin_home_current": 2.0,
                     "sd_current": 15.0, "p_home_current": 0.55,
                     "forecast_basis": "current results blend"},
                    {"id": "2", "kickoff": "2026-11-07T20:00Z",
                     "home": "Georgia", "away": "Ole Miss", "neutral": False,
                     "played": False, "margin_home_current": 3.0,
                     "sd_current": 15.0, "p_home_current": 0.58,
                     "forecast_basis": "current results blend"},
                ],
            }
            (out / "app_data.json").write_text(json.dumps(model))
            with patch.object(p, "OUT", out), patch.object(p, "SPREADS", spreads):
                result = p.export()
            self.assertEqual(len(result["games"]), 2)
            self.assertLess(result["games"][0]["home_cover_probability"], 0.5)
            self.assertEqual(result["games"][0]["ats_pick"], "Oklahoma")
            self.assertIsNone(result["games"][1]["home_cover_probability"])
            self.assertTrue((out / "predictions_latest.json").exists())
            self.assertEqual(len(list((out / "prediction_snapshots").glob("*.json"))), 1)

if __name__ == "__main__":
    unittest.main()
