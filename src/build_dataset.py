"""Assemble a team-season panel from CFBD.

Output: data/processed/panel.csv  (one row per team-season, national FBS)
        data/processed/games.csv  (one row per game, used for calibration)

Key normalisation: CFBD has shipped both snake_case and camelCase payloads.
Every dict is normalised to snake_case on ingest so either generation works.
"""
from datetime import datetime, timezone
import json
import re
import sys
import pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cfbd_client as api

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

_c1 = re.compile(r"(.)([A-Z][a-z]+)")
_c2 = re.compile(r"([a-z0-9])([A-Z])")


def _snake_key(k):
    """camelCase -> snake_case, correctly handling runs of capitals (percentPPA)."""
    return _c2.sub(r"\1_\2", _c1.sub(r"\1_\2", k)).lower()


def snake(obj):
    if isinstance(obj, dict):
        return {_snake_key(k): snake(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [snake(v) for v in obj]
    return obj


def safe(endpoint, **params):
    required = params.pop("_required", False)
    try:
        return snake(api.get(endpoint, **params))
    except Exception as e:
        print("  ! %s %s -> %s: %s" % (endpoint, params, type(e).__name__, str(e)[:140]))
        if required:
            raise
        return []


def g(d, *names, **kw):
    default = kw.get("default")
    for n in names:
        if isinstance(d, dict) and d.get(n) is not None:
            return d[n]
    return default


# ---------------------------------------------------------------- per-year pulls
def sp_ratings(year):
    rows = []
    for r in safe("ratings/sp", year=year, _required=True):
        # The endpoint includes a synthetic nationalAverages record alongside
        # actual teams. Product surfaces and model training must never treat it
        # as a school.
        if not g(r, "team") or not g(r, "conference") \
                or str(g(r, "team")).lower() == "nationalaverages":
            continue
        rows.append({
            "season": year,
            "team": r["team"],
            "conference": g(r, "conference"),
            "sp_overall": g(r, "rating"),
            "sp_offense": g(r.get("offense") or {}, "rating"),
            "sp_defense": g(r.get("defense") or {}, "rating"),
            "sp_special": g(r.get("special_teams") or {}, "rating"),
            "sp_sos": g(r, "sos"),
            "second_order_wins": g(r, "second_order_wins"),
        })
    return rows


def records(year):
    rows = []
    for r in safe("records", year=year):
        tot = r.get("total") or {}
        conf = r.get("conference_games") or {}
        rows.append({
            "season": year,
            "team": g(r, "team"),
            "wins": g(tot, "wins"),
            "losses": g(tot, "losses"),
            "games": g(tot, "games"),
            "conf_wins": g(conf, "wins"),
            "conf_losses": g(conf, "losses"),
            "expected_wins": g(r, "expected_wins"),
        })
    return [r for r in rows if r["team"]]


def talent(year):
    out = []
    for r in safe("talent", year=year):
        t = g(r, "school", "team")
        if not t:
            continue
        try:
            val = float(g(r, "talent", default="nan"))
        except (TypeError, ValueError):
            val = float("nan")
        out.append({"season": year, "team": t, "talent": val})
    return out


def recruiting_team(year):
    return [{"season": year, "team": g(r, "team"),
             "recruit_points": g(r, "points"), "recruit_rank": g(r, "rank")}
            for r in safe("recruiting/teams", year=year) if g(r, "team")]


def recruiting_bluechip(year):
    """Blue-chip = 4- or 5-star high-school signees in this class."""
    counts, five, sizes = {}, {}, {}
    for p in safe("recruiting/players", year=year, classification="HighSchool"):
        t = g(p, "committed_to", "team")
        s = g(p, "stars", default=0) or 0
        if not t:
            continue
        sizes[t] = sizes.get(t, 0) + 1
        if s >= 4:
            counts[t] = counts.get(t, 0) + 1
        if s >= 5:
            five[t] = five.get(t, 0) + 1
    return [{"season": year, "team": t, "class_size": n,
             "bluechips": counts.get(t, 0), "fivestars": five.get(t, 0),
             "bluechip_ratio": (counts.get(t, 0) / n) if n else None}
            for t, n in sizes.items()]


def returning_production(year):
    rows = []
    for r in safe("player/returning", year=year):
        rows.append({
            "season": year,
            "team": g(r, "team"),
            "ret_ppa": g(r, "percent_ppa"),
            "ret_pass_ppa": g(r, "percent_passing_ppa"),
            "ret_rush_ppa": g(r, "percent_rushing_ppa"),
            "ret_recv_ppa": g(r, "percent_receiving_ppa"),
            "ret_usage": g(r, "usage"),
            "ret_pass_usage": g(r, "passing_usage"),
        })
    return [r for r in rows if r["team"]]


def portal(year):
    """Net incoming-minus-outgoing transfer volume and talent value."""
    inc, out, inc_r, out_r = {}, {}, {}, {}
    for p in safe("player/portal", year=year):
        dest, orig = g(p, "destination"), g(p, "origin")
        rating = g(p, "rating", default=0) or 0
        stars = g(p, "stars", default=0) or 0
        val = rating if rating else (stars / 5.0)
        if dest:
            inc[dest] = inc.get(dest, 0) + 1
            inc_r[dest] = inc_r.get(dest, 0) + val
        if orig:
            out[orig] = out.get(orig, 0) + 1
            out_r[orig] = out_r.get(orig, 0) + val
    teams = set(inc) | set(out)
    return [{"season": year, "team": t,
             "portal_in": inc.get(t, 0), "portal_out": out.get(t, 0),
             "portal_net": inc.get(t, 0) - out.get(t, 0),
             "portal_in_val": inc_r.get(t, 0), "portal_out_val": out_r.get(t, 0),
             "portal_net_val": inc_r.get(t, 0) - out_r.get(t, 0)}
            for t in teams]


def coaches(year):
    rows = []
    for c in safe("coaches", year=year):
        name = ("%s %s" % (g(c, "first_name", default=""),
                           g(c, "last_name", default=""))).strip()
        for s in (c.get("seasons") or []):
            if g(s, "year") == year and g(s, "school"):
                rows.append({"season": year, "team": s["school"], "coach": name})
    return rows


def games(year, season_status=None):
    rows = []
    schedule = safe("games", year=year, seasonType="regular", _required=True)
    schedule = [gm for gm in schedule if g(gm, "home_team") and g(gm, "away_team")]
    for gm in schedule:
        ht, at = g(gm, "home_team"), g(gm, "away_team")
        hp, ap = g(gm, "home_points"), g(gm, "away_points")
        if not ht or not at or hp is None or ap is None or gm.get("completed") is False:
            continue
        rows.append({
            "season": year, "week": g(gm, "week"),
            "home_team": ht, "away_team": at,
            "home_points": hp, "away_points": ap,
            "neutral": bool(g(gm, "neutral_site", default=False)),
            "conference_game": bool(g(gm, "conference_game", default=False)),
            "home_conference": g(gm, "home_conference"),
            "away_conference": g(gm, "away_conference"),
        })
    if season_status is not None:
        season_status.append({"season": year, "scheduled": len(schedule),
                              "completed": len(rows),
                              "assembled_at": datetime.now(timezone.utc).isoformat()})
    return rows


# ---------------------------------------------------------------- assembly
def build(start=2005, end=2025):
    season_status = []
    frames = {k: [] for k in
              ["sp", "rec", "tal", "rct", "blue", "ret", "por", "cch", "gms"]}
    for y in range(start, end + 1):
        print("[build] %d" % y)
        frames["sp"] += sp_ratings(y)
        frames["rec"] += records(y)
        frames["tal"] += talent(y)
        frames["rct"] += recruiting_team(y)
        frames["blue"] += recruiting_bluechip(y)
        frames["ret"] += returning_production(y)
        frames["por"] += portal(y)
        frames["cch"] += coaches(y)
        frames["gms"] += games(y, season_status)

    def df(k):
        d = pd.DataFrame(frames[k])
        return d if len(d) else pd.DataFrame(columns=["season", "team"])

    panel = df("sp")
    for k in ["rec", "tal", "rct", "blue", "ret", "por", "cch"]:
        d = df(k)
        if len(d):
            d = d.drop_duplicates(subset=["season", "team"])
            panel = panel.merge(d, on=["season", "team"], how="left")

    gms = pd.DataFrame(frames["gms"])
    gms.to_csv(OUT / "games.csv", index=False)

    # margin / close-game columns derived from actual results
    if len(gms):
        long = pd.concat([
            gms.assign(team=gms.home_team, opp=gms.away_team,
                       pf=gms.home_points, pa=gms.away_points),
            gms.assign(team=gms.away_team, opp=gms.home_team,
                       pf=gms.away_points, pa=gms.home_points),
        ])
        long["margin"] = long.pf - long.pa
        agg = long.groupby(["season", "team"]).agg(
            ppg=("pf", "mean"), papg=("pa", "mean"),
            margin_pg=("margin", "mean"), n_games=("pf", "size"),
            close_games=("margin", lambda s: int((s.abs() <= 7).sum())),
            close_wins=("margin", lambda s: int(((s > 0) & (s <= 7)).sum())),
        ).reset_index()
        agg["close_win_pct"] = agg.close_wins / agg.close_games.replace(0, float("nan"))
        panel = panel.merge(agg, on=["season", "team"], how="left")

    panel = panel.sort_values(["team", "season"]).reset_index(drop=True)

    # Fail closed on malformed core data. Optional feature endpoints may be
    # sparse by era, but ratings must be unique and team-like.
    dupes = int(panel.duplicated(["season", "team"]).sum())
    invalid = panel.team.astype(str).str.lower().eq("nationalaverages")
    if dupes or invalid.any():
        raise RuntimeError("invalid panel: %d duplicate keys, %d aggregate rows"
                           % (dupes, int(invalid.sum())))
    team_counts = panel.groupby("season").team.nunique()
    thin = team_counts[team_counts < 100]
    if len(thin):
        raise RuntimeError("ratings coverage below 100 teams: %s"
                           % thin.to_dict())

    panel.to_csv(OUT / "panel.csv", index=False)
    pd.DataFrame(season_status).to_csv(OUT / "season_status.csv", index=False)
    quality = {
        "start_season": int(start), "end_season": int(end),
        "panel_rows": int(len(panel)), "game_rows": int(len(gms)),
        "duplicate_team_seasons": dupes,
        "team_counts_by_season": {str(int(k)): int(v)
                                  for k, v in team_counts.items()},
        "missingness": {c: round(float(panel[c].isna().mean()), 4)
                        for c in panel.columns if c not in ("team", "season")},
    }
    (OUT.parent.parent / "output" / "data_quality.json").parent.mkdir(exist_ok=True)
    (OUT.parent.parent / "output" / "data_quality.json").write_text(
        json.dumps(quality, indent=2))
    print("[build] panel: %d team-seasons, %d cols -> %s"
          % (panel.shape[0], panel.shape[1], OUT / "panel.csv"))
    print("[build] games: %d rows -> %s" % (len(gms), OUT / "games.csv"))
    return panel


if __name__ == "__main__":
    a = sys.argv[1:]
    build(int(a[0]) if a else 2005, int(a[1]) if len(a) > 1 else 2025)
