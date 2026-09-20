"""Human polls, for comparison against the model's own ranking.

ESPN publishes AP and Coaches polls on a keyless endpoint, so this costs
nothing against the CFBD monthly key quota.

A poll and a rating are not the same measurement and are not expected to
agree. `sp_overall` estimates points per game against an average opponent;
a poll is voters weighing record, opponent quality and reputation. Their
disagreements are the interesting output, not an error in either.

One timing trap this module makes explicit: the poll labelled "Week N" is
voted BEFORE week N is played, so it reflects results through week N-1.
Comparing it to a rating that already includes week N would flatter the
rating. `information_through` records the poll's real cutoff.
"""
import re

import requests

ESPN_RANKINGS = ("https://site.api.espn.com/apis/site/v2/sports/"
                 "football/college-football/rankings")
WANTED = {"ap": "AP Top 25", "usa": "Coaches Poll"}
TIMEOUT = 45


def normalize(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def fetch(season=None, week=None, timeout=TIMEOUT):
    """Raw ESPN rankings payload."""
    params = {}
    if season:
        params.update(season=season, seasontype=2)
    if week:
        params["week"] = week
    response = requests.get(ESPN_RANKINGS, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse(payload, rated_teams):
    """Extract the polls we care about, mapped onto our own team names.

    rated_teams: iterable of the team names the model rates. A poll entry that
    does not map to one is reported rather than dropped silently.
    """
    lookup = {normalize(t): t for t in rated_teams}
    out = {}
    for block in payload.get("rankings", []):
        kind = block.get("type")
        if kind not in WANTED:
            continue
        occurrence = (block.get("occurrence") or {}).get("number")
        entries, unmatched = [], []
        for rank in block.get("ranks", []):
            team = rank.get("team", {})
            name = lookup.get(normalize(team.get("location")))
            if name is None:
                unmatched.append(team.get("location"))
                continue
            entries.append({
                "rank": int(rank.get("current")),
                "team": name,
                "record": rank.get("recordSummary"),
                "previous": rank.get("previous"),
            })
        out[kind] = {
            "name": WANTED[kind],
            "poll_week": occurrence,
            # Voted before the labelled week is played.
            "information_through": (occurrence - 1) if occurrence else None,
            "released": block.get("date"),
            "headline": block.get("shortHeadline"),
            "entries": sorted(entries, key=lambda e: e["rank"]),
            "unmatched": unmatched,
        }
    return out


def model_top(ratings, n=25, key="rating"):
    """Our own ranking: the top n teams by rating, highest first."""
    ordered = sorted(ratings.items(), key=lambda kv: -kv[1])
    return [{"rank": i, "team": t, "rating": round(r, 2)}
            for i, (t, r) in enumerate(ordered[:n], start=1)]


def compare(model_ranking, poll):
    """Line up our ranking against a poll and describe the disagreement."""
    ours = {e["team"]: e["rank"] for e in model_ranking}
    theirs = {e["team"]: e["rank"] for e in poll["entries"]}
    rows = []
    for entry in model_ranking:
        team = entry["team"]
        their_rank = theirs.get(team)
        rows.append({
            "team": team,
            "model_rank": entry["rank"],
            "rating": entry["rating"],
            "poll_rank": their_rank,
            # positive = the model likes the team more than the voters do
            "delta": (their_rank - entry["rank"]) if their_rank else None,
        })
    return {
        "rows": rows,
        "we_rank_they_do_not": [e["team"] for e in model_ranking if e["team"] not in theirs],
        "they_rank_we_do_not": [
            {"team": e["team"], "poll_rank": e["rank"]}
            for e in poll["entries"] if e["team"] not in ours],
        "overlap": len(set(ours) & set(theirs)),
        "biggest_disagreements": sorted(
            [r for r in rows if r["delta"] is not None],
            key=lambda r: -abs(r["delta"]))[:5],
    }
