import pathlib
import sys
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import polls

PAYLOAD = {"rankings": [
    {"type": "ap", "shortHeadline": "2026 AP Poll: Week 3",
     "occurrence": {"number": 3}, "date": "2026-09-14",
     "ranks": [
         {"current": 1, "team": {"location": "Texas"}, "recordSummary": "2-0", "previous": 4},
         {"current": 2, "team": {"location": "Georgia"}, "recordSummary": "2-0", "previous": 2},
         {"current": 3, "team": {"location": "Nowhere State"}, "recordSummary": "2-0"},
     ]},
    {"type": "fcs", "occurrence": {"number": 3}, "ranks": []},
]}
RATED = ["Texas", "Georgia", "Alabama", "Ohio State"]


class ParseTests(unittest.TestCase):
    def test_only_the_polls_we_track_are_kept(self):
        out = polls.parse(PAYLOAD, RATED)
        self.assertEqual(set(out), {"ap"})

    def test_a_week_n_poll_reflects_games_through_week_n_minus_1(self):
        """The trap: the poll is voted before its labelled week is played."""
        ap = polls.parse(PAYLOAD, RATED)["ap"]
        self.assertEqual(ap["poll_week"], 3)
        self.assertEqual(ap["information_through"], 2)

    def test_an_unrated_school_is_reported_not_silently_dropped(self):
        ap = polls.parse(PAYLOAD, RATED)["ap"]
        self.assertEqual(ap["unmatched"], ["Nowhere State"])
        self.assertEqual([e["team"] for e in ap["entries"]], ["Texas", "Georgia"])


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.poll = polls.parse(PAYLOAD, RATED)["ap"]

    def test_positive_delta_means_the_model_likes_them_more(self):
        top = polls.model_top({"Georgia": 30.0, "Texas": 25.0}, 2)
        cmp = polls.compare(top, self.poll)
        rows = {r["team"]: r for r in cmp["rows"]}
        self.assertEqual(rows["Georgia"]["model_rank"], 1)
        self.assertEqual(rows["Georgia"]["poll_rank"], 2)
        self.assertEqual(rows["Georgia"]["delta"], 1)     # AP 2 -> us 1
        self.assertEqual(rows["Texas"]["delta"], -1)      # AP 1 -> us 2

    def test_teams_only_one_side_ranks_are_listed_both_ways(self):
        top = polls.model_top({"Alabama": 40.0, "Georgia": 30.0}, 2)
        cmp = polls.compare(top, self.poll)
        self.assertEqual(cmp["we_rank_they_do_not"], ["Alabama"])
        self.assertEqual([x["team"] for x in cmp["they_rank_we_do_not"]], ["Texas"])
        self.assertEqual(cmp["overlap"], 1)
        self.assertIsNone({r["team"]: r for r in cmp["rows"]}["Alabama"]["delta"])

    def test_model_top_is_ordered_by_rating_and_truncated(self):
        top = polls.model_top({"A": 1.0, "B": 9.0, "C": 5.0}, 2)
        self.assertEqual([e["team"] for e in top], ["B", "C"])
        self.assertEqual(top[0]["rank"], 1)


class FetchTests(unittest.TestCase):
    def test_fetch_raises_on_a_bad_response_rather_than_returning_junk(self):
        class Bad:
            status_code = 500
            def raise_for_status(self):
                raise polls.requests.HTTPError("500")
        with patch.object(polls.requests, "get", return_value=Bad()):
            with self.assertRaises(polls.requests.HTTPError):
                polls.fetch(2026)


if __name__ == "__main__":
    unittest.main()
