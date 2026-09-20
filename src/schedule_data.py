"""Fetch final scores without refitting the preseason model.

CFBD remains the preferred source when its server-side key is configured.
ESPN's public scoreboard provides a keyless SEC schedule/results fallback.
"""
from datetime import datetime, timezone
import pathlib
import os
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


def fetch_schedule(season, source="auto", refresh=False):
    if source == "auto":
        source = "cfbd" if (os.environ.get("CFBD_API_KEY") or
                            (pathlib.Path.home() / ".cfbd_key").exists()) else "espn"
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
        rows = espn_week_rows(season)
        url = (f"{ESPN_URL}?dates={season}&seasontype=2"
               f"&week=1-{ESPN_WEEKS[-1]}&groups=8&limit=1000")
    else:
        raise ValueError(f"Unknown schedule source: {source}")
    if not rows:
        raise ValueError(f"{source} returned an empty {season} regular-season schedule")
    frame = pd.DataFrame(rows).drop_duplicates("id")
    if frame.loc[frame.completed, ["home_points", "away_points"]].isna().any().any():
        raise ValueError("A completed game is missing its final score")
    return frame, {"provider": source, "url": url,
                   "checked_at": datetime.now(timezone.utc).isoformat(),
                   "cache_bypassed": source == "espn" or refresh or os.environ.get("CFBD_REFRESH") == "1"}


def validate_schedule(schedule, teams):
    """Reject partial feeds before replacing a healthy published snapshot."""
    for team in teams:
        count = int(((schedule.home_team == team) | (schedule.away_team == team)).sum())
        if count != 12:
            raise ValueError(f"Incomplete/unexpected schedule: {team} has {count} games, expected 12")
