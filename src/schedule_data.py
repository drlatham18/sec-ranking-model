"""Fetch final scores without refitting the preseason model.

CFBD remains the preferred source when its server-side key is configured.
ESPN's public scoreboard provides a keyless SEC schedule/results fallback.
"""
from datetime import datetime, timezone
import pathlib
import os
import re
import time

import pandas as pd
import requests

import build_dataset as BD

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
# ESPN stopped serving multi-day `dates=A-B` ranges on this endpoint; any range
# now answers 400. Single dates and explicit season/week selectors still work,
# so the regular season is pulled one week at a time and merged.
ESPN_WEEKS = range(1, 17)
ESPN_ATTEMPTS = 3
# ESPN caps a scoreboard response at 25 events regardless of `limit`, so an
# all-FBS pull must be split. Asking per conference keeps every response well
# under the cap; games between two conferences arrive twice and are deduped.
ESPN_FBS_GROUPS = (1, 4, 5, 8, 9, 12, 15, 17, 18, 20, 151)
# A handful of cancelled/unrecorded games is normal; a feed full of them is not.
MAX_SCORELESS_GAMES = 5
MAX_SCORELESS_SHARE = 0.02


def has_cfbd_key():
    return bool(os.environ.get("CFBD_API_KEY")
                or (pathlib.Path.home() / ".cfbd_key").exists())


def espn_rows(payload, season):
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("ESPN response has no events list")
    rows = []
    for event in payload["events"]:
        event_season = event.get("season", {})
        if event_season.get("year") != season or event_season.get("type") != 2:
            continue
        for game in event.get("competitions", []):
            competitors = {c["homeAway"]: c for c in game["competitors"]}
            home, away = competitors["home"], competitors["away"]
            status = game.get("status", event.get("status", {})).get("type", {})
            completed = status.get("completed") is True and status.get("state") == "post"
            rows.append({
                "id": event["id"], "week": event["week"]["number"],
                "start_date": event["date"],
                "home_team": home["team"]["location"],
                "away_team": away["team"]["location"],
                "home_conference": "SEC" if home["team"].get("conferenceId") == "8" else None,
                "away_conference": "SEC" if away["team"].get("conferenceId") == "8" else None,
                # Pregame/in-progress scores must never become locked results.
                "home_points": int(home["score"]) if completed else None,
                "away_points": int(away["score"]) if completed else None,
                "completed": completed,
                "neutral": bool(game.get("neutralSite", False)),
                "conference_game": bool(game.get("conferenceCompetition", False)),
            })
    return rows


def espn_get(season, week, group=8):
    """One week of the scoreboard, retried briefly for transient failures.

    A full pull is 16 requests, so a single flaky response must not discard an
    otherwise healthy refresh. A persistent error still raises: publishing a
    partial schedule would silently drop games.
    """
    last = None
    for attempt in range(ESPN_ATTEMPTS):
        try:
            response = requests.get(ESPN_URL, params={
                "dates": season, "seasontype": 2, "week": week,
                "groups": group, "limit": 1000,
            }, timeout=45)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last = error
            if attempt + 1 < ESPN_ATTEMPTS:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"ESPN week {week} (group {group}) of {season} failed: {last}") from last


def espn_week_rows(season, groups=(8,), weeks=ESPN_WEEKS):
    """Merge the requested conference groups and weeks into one schedule."""
    rows = []
    for group in groups:
        for week in weeks:
            rows.extend(espn_rows(espn_get(season, week, group), season))
    return rows


def fbs_results(season, through_week, source="auto", refresh=False):
    """Every completed FBS game through `through_week`.

    In-season strength needs the whole FBS, not just one conference: a team's
    rating is only as good as the opponents it is measured against.
    """
    if source == "auto":
        source = "cfbd" if (os.environ.get("CFBD_API_KEY") or
                            (pathlib.Path.home() / ".cfbd_key").exists()) else "espn"
    if source == "cfbd":
        frame, _ = fetch_schedule(season, "cfbd", refresh)
    else:
        weeks = range(1, max(1, int(through_week)) + 1)
        rows = espn_week_rows(season, ESPN_FBS_GROUPS, weeks)
        if not rows:
            raise ValueError(f"espn returned no {season} results through week {through_week}")
        frame = pd.DataFrame(rows).drop_duplicates("id")
    return frame[frame.completed].copy()


def fetch_schedule(season, source="auto", refresh=False, weeks=None):
    if source == "auto":
        source = "cfbd" if has_cfbd_key() else "espn"
    if source == "cfbd":
        rows = []
        for game in BD.safe("games", year=season, seasonType="regular",
                            refresh=refresh, _required=True):
            # CFBD explicitly exposes completed; never infer finality from scores.
            completed = game.get("completed") is True
            rows.append({
                "id": str(game["id"]), "week": game["week"],
                "start_date": game["start_date"],
                "home_team": game["home_team"], "away_team": game["away_team"],
                "home_conference": game.get("home_conference"),
                "away_conference": game.get("away_conference"),
                "home_points": game.get("home_points") if completed else None,
                "away_points": game.get("away_points") if completed else None,
                "completed": completed,
                "neutral": bool(game.get("neutral_site", False)),
                "conference_game": bool(game.get("conference_game", False)),
            })
        url = "https://api.collegefootballdata.com/games"
    elif source == "espn":
        rows = espn_week_rows(season, ESPN_FBS_GROUPS, weeks or ESPN_WEEKS)
        url = (f"{ESPN_URL}?dates={season}&seasontype=2"
               f"&week=1-{ESPN_WEEKS[-1]}&groups={','.join(map(str, ESPN_FBS_GROUPS))}")
    else:
        raise ValueError(f"Unknown schedule source: {source}")
    if not rows:
        raise ValueError(f"{source} returned an empty {season} regular-season schedule")
    frame = pd.DataFrame(rows).drop_duplicates("id")
    # A feed can flag a cancelled or unrecorded game "completed" with no score.
    # We will not invent a result, so such a game is simply not played. A feed
    # where this is widespread is broken rather than quirky, and still fails.
    missing = frame.completed & (frame.home_points.isna() | frame.away_points.isna())
    if missing.any():
        allowed = max(MAX_SCORELESS_GAMES, int(MAX_SCORELESS_SHARE * len(frame)))
        listed = ", ".join("%s vs %s (wk %s)" % (r.away_team, r.home_team, r.week)
                           for _, r in frame[missing].head(10).iterrows())
        if int(missing.sum()) > allowed:
            raise ValueError("%s reports %d completed games with no final score "
                             "(max tolerated %d): %s"
                             % (source, int(missing.sum()), allowed, listed))
        print("[schedule] %s: %d completed game(s) with no score, treated as "
              "not played: %s" % (source, int(missing.sum()), listed))
        frame.loc[missing, "completed"] = False
    return frame, {"provider": source, "url": url,
                   "checked_at": datetime.now(timezone.utc).isoformat(),
                   "cache_bypassed": source == "espn" or refresh or os.environ.get("CFBD_REFRESH") == "1"}


def _key(frame):
    """Match key that survives the two feeds naming schools differently."""
    def n(v):
        return re.sub(r"[^a-z0-9]", "", str(v).lower())
    return [(int(w) if pd.notna(w) else -1, n(h), n(a))
            for w, h, a in zip(frame.week, frame.home_team, frame.away_team)]


def cross_check(spine, other):
    """Verify the spine's finals against a second feed.

    The two sources name FCS schools differently, so merging them would
    duplicate games. The spine stays authoritative for WHICH games exist; the
    other feed only audits the ones that match by week and both team names.

    A final score the two feeds disagree on is not a rounding difference -- one
    of them is wrong, and a wrong score silently corrupts every rating built on
    it. Such a game is un-completed rather than published.
    """
    spine = spine.copy()
    lookup = {k: r for k, r in zip(_key(other), other.itertuples())}
    agreed = disagreed = unmatched = 0
    conflicts = []
    for i, key in zip(spine.index, _key(spine)):
        if not spine.at[i, "completed"]:
            continue
        match = lookup.get(key)
        if match is None or not match.completed:
            unmatched += 1
            continue
        if (match.home_points == spine.at[i, "home_points"]
                and match.away_points == spine.at[i, "away_points"]):
            agreed += 1
        else:
            disagreed += 1
            conflicts.append({
                "week": key[0], "home": spine.at[i, "home_team"],
                "away": spine.at[i, "away_team"],
                "spine": [spine.at[i, "home_points"], spine.at[i, "away_points"]],
                "other": [match.home_points, match.away_points]})
            spine.loc[i, ["completed", "home_points", "away_points"]] = [False, None, None]
    return spine, {"agreed": agreed, "disagreed": disagreed,
                   "unverified": unmatched, "conflicts": conflicts[:20]}


def fetch_consensus(season, source="auto", refresh=False, verify=True):
    """Full-season schedule, cross-checked against a second feed when possible.

    CFBD returns the entire season in ONE request, so it costs one call against
    the monthly key quota and is preferred as the spine. ESPN needs a request
    per conference per week, but is keyless and unmetered, so it audits.

    Either source failing is survivable; both failing is not.
    """
    attempts = ["cfbd", "espn"] if source == "auto" else [source]
    if source == "auto" and not has_cfbd_key():
        attempts = ["espn"]
    frames, status = {}, {}
    for name in attempts:
        try:
            # As auditor, ESPN only needs the weeks that already have finals --
            # a full-season sweep is 11 groups x 16 weeks for no extra signal.
            weeks = None
            if name == "espn" and "cfbd" in frames:
                done = frames["cfbd"][frames["cfbd"].completed]
                weeks = range(1, (int(done.week.max()) if len(done) else 1) + 1)
            frames[name], status[name] = fetch_schedule(season, name, refresh, weeks)
            status[name]["ok"] = True
        except Exception as error:                     # noqa: BLE001 - any feed fault
            status[name] = {"provider": name, "ok": False, "error": str(error)[:300]}
            print("[schedule] %s unavailable: %s" % (name, str(error)[:200]))
    if not frames:
        raise ValueError("no schedule source succeeded for %d: %s"
                         % (season, {k: v.get("error") for k, v in status.items()}))
    spine_name = "cfbd" if "cfbd" in frames else next(iter(frames))
    frame = frames[spine_name]
    audit = {"performed": False}
    if verify and len(frames) > 1:
        other = next(n for n in frames if n != spine_name)
        frame, audit = cross_check(frame, frames[other])
        audit.update(performed=True, against=other)
        if audit["disagreed"]:
            print("[schedule] %d final score(s) disputed between %s and %s; "
                  "left unplayed: %s" % (audit["disagreed"], spine_name, other,
                                         audit["conflicts"]))
    return frame, {"provider": spine_name, "sources": status, "audit": audit,
                   "checked_at": datetime.now(timezone.utc).isoformat(),
                   "url": status[spine_name].get("url")}


def validate_schedule(schedule, teams):
    """Reject partial feeds before replacing a healthy published snapshot."""
    for team in teams:
        count = int(((schedule.home_team == team) | (schedule.away_team == team)).sum())
        if count != 12:
            raise ValueError(f"Incomplete/unexpected schedule: {team} has {count} games, expected 12")
