import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import national


def g(home, away, hp, ap):
    return dict(home=home, away=away, home_points=hp, away_points=ap, neutral=False)


class RecordTests(unittest.TestCase):
    def test_counts_wins_losses_and_ties(self):
        rows = national.records(
            [g("A", "B", 21, 7), g("B", "A", 14, 28), g("A", "C", 10, 10)],
            ["A", "B", "C"])
        self.assertEqual(rows["A"]["record"], "2-0-1")   # a tie is not a win
        self.assertEqual(rows["A"]["wins"], 2)
        self.assertEqual(rows["A"]["ties"], 1)
        self.assertEqual(rows["B"]["record"], "0-2")

    def test_games_against_unrated_teams_still_count_toward_the_record(self):
        rows = national.records([g("A", "Some FCS School", 49, 0)], ["A"])
        self.assertEqual(rows["A"]["record"], "1-0")

    def test_unplayed_games_are_ignored(self):
        rows = national.records([g("A", "B", None, None)], ["A", "B"])
        self.assertEqual(rows["A"]["record"], "0-0")


class TableTests(unittest.TestCase):
    def setUp(self):
        self.current = {"A": 30.0, "B": 20.0, "C": 10.0}
        self.pre = {"A": 10.0, "B": 20.0, "C": 30.0}       # exactly reversed
        self.prev = {"A": 20.0, "B": 30.0, "C": 10.0}

    def test_ordered_by_rating_descending(self):
        rows = national.table(self.current, self.pre)
        self.assertEqual([r["team"] for r in rows], ["A", "B", "C"])
        self.assertEqual([r["rank"] for r in rows], [1, 2, 3])

    def test_positive_movement_means_climbed(self):
        rows = {r["team"]: r for r in national.table(self.current, self.pre, self.prev)}
        # A was preseason 3rd, now 1st -> +2
        self.assertEqual(rows["A"]["change_since_preseason"], 2)
        self.assertEqual(rows["C"]["change_since_preseason"], -2)
        # A was 2nd last week, now 1st -> +1
        self.assertEqual(rows["A"]["change_since_last_week"], 1)
        self.assertEqual(rows["B"]["change_since_last_week"], -1)

    def test_movement_is_absent_rather_than_zero_without_a_prior_week(self):
        rows = national.table(self.current, self.pre)
        self.assertTrue(all(r["change_since_last_week"] is None for r in rows))

    def test_limit_truncates_without_renumbering(self):
        rows = national.table(self.current, self.pre, limit=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["rank"], 2)

    def test_blend_detail_and_record_are_carried_through(self):
        rows = national.table(
            self.current, self.pre,
            record_table=national.records([g("A", "B", 20, 0)], ["A", "B"]),
            detail={"A": {"n_games": 3, "weight": 0.375}})
        top = rows[0]
        self.assertEqual(top["record"], "1-0")
        self.assertEqual(top["games_rated"], 3)
        self.assertEqual(top["from_results"], 0.375)

    def test_a_team_with_no_games_reports_no_results_share(self):
        rows = national.table(self.current, self.pre, detail={})
        self.assertIsNone(rows[0]["from_results"])


if __name__ == "__main__":
    unittest.main()
