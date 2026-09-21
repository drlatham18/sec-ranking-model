"""Bundle everything the UI needs into one JSON payload.

Pulls the target season's full schedule (including unplayed games), applies the
fitted game model to every SEC game, runs a Monte Carlo season simulation, and
writes output/app_data.json.
"""
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_dataset as BD
import schedule_data
import weekly
import inseason
import national

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
N_SIMS = 20000


def fcs_win_rate():
    """Empirical win rate of a rated (FBS) team against an unrated (FCS) one.

    Estimated from every such game in the panel -- not assumed to be 100%.
    """
    gm = pd.read_csv(ROOT / "data" / "processed" / "games.csv")
    panel = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    rated = set(zip(panel.season, panel.team))
    h = [(s, t) in rated for s, t in zip(gm.season, gm.home_team)]
    a = [(s, t) in rated for s, t in zip(gm.season, gm.away_team)]
    gm = gm.assign(h_rated=h, a_rated=a)
    one = gm[gm.h_rated != gm.a_rated]
    won = ((one.h_rated & (one.home_points > one.away_points))
           | (one.a_rated & (one.away_points > one.home_points)))
    return float(won.mean()), int(len(one))


def build(season=2026, conference="SEC", source="auto", refresh=False):
    ratings = pd.read_csv(OUT / ("ratings_%d.csv" % season))
    sel = json.loads((OUT / "selection.json").read_text())
    cal = json.loads((OUT / "calibration.json").read_text())
    val = pd.read_csv(OUT / "validation.csv")
    proj_rmse = float(sel["performance_claim"]["rmse"])

    rmap = dict(zip(ratings.team, ratings.projected_rating))
    sec = ratings[ratings.conference == conference].copy()
    sec_teams = set(sec.team)

    # ONE fetch serves both the conference schedule and the all-FBS results the
    # in-season blend needs. CFBD returns the whole season per request and the
    # key is capped monthly, so fetching twice would double the quota spend for
    # identical data.
    full, source_info = schedule_data.fetch_consensus(season, source, refresh)
    sched = full[full.home_team.isin(sec_teams) | full.away_team.isin(sec_teams)]

    schedule_data.validate_schedule(sched, sec_teams)
    p_fcs, n_fcs = fcs_win_rate()

    # --- current (in-season) strength ------------------------------------
    # Frozen preseason ratings stay the basis of the weekly comparison, which
    # must not be rewritten by later results. Forward-looking probabilities use
    # strength that includes the season actually being played.
    played = sched[sched.completed]
    through_week = int(played.week.max()) if len(played) else 0
    strength = None
    try:
        cal_in = inseason.load_calibration()
        if through_week:
            fbs = full[full.completed]
            results = [{"home": r.home_team, "away": r.away_team,
                        "home_points": r.home_points, "away_points": r.away_points,
                        "neutral": bool(r.neutral)} for _, r in fbs.iterrows()]
            strength = inseason.current_ratings(results, rmap, cal_in)
    except (FileNotFoundError, ValueError) as error:
        print("[export] in-season blend unavailable (%s); preseason only" % error)
        cal_in = None
    cmap = {t: v["blended"] for t, v in strength.items()} if strength else {}

    game_sd_n = cal["neutral_residual_sd"]
    game_sd_h = cal["residual_sd"]
    b1 = cal["rating_diff_coef"]
    extra = (0.0 if cal.get("method") == "walk-forward preseason predictions"
             else 2.0 * (b1 * proj_rmse) ** 2)

    from scipy.stats import norm
    games = []
    for _, r in sched.iterrows():
        rh, ra = rmap.get(r.home_team), rmap.get(r.away_team)
        unrated = rh is None or ra is None
        hf = 0.0 if r.neutral else 1.0
        margin = (None if unrated else
                  cal["intercept"] + b1 * (rh - ra) + cal["home_field_advantage"] * hf)
        sd = float(np.sqrt((game_sd_n if r.neutral else game_sd_h) ** 2 + extra))
        p_home = (p_fcs if rh is not None else 1 - p_fcs) if unrated else float(norm.cdf(margin / sd))
        # Current-strength view of the same game (None until the blend exists).
        ch, ca = cmap.get(r.home_team), cmap.get(r.away_team)
        if cal_in and ch is not None and ca is not None:
            m_cur = cal_in["rating_diff_coef"] * (ch - ca) + cal_in["home_field_advantage"] * hf
            sd_cur = cal_in["residual_sd"]
            p_cur = float(norm.cdf(m_cur / sd_cur))
        else:
            m_cur = sd_cur = None
            p_cur = p_home if unrated else None
        playable_at = (cal_in or {}).get("playable_threshold")
        confidence = max(p_cur, 1 - p_cur) if p_cur is not None else None
        games.append({
            "id": str(r.id),
            "week": int(r.week) if pd.notna(r.week) else None,
            "date": str(r.start_date)[:10] if r.start_date else None,
            "kickoff": str(r.start_date) if r.start_date else None,
            "home": r.home_team, "away": r.away_team,
            "home_conf": r.home_conference, "away_conf": r.away_conference,
            "neutral": bool(r.neutral), "conf_game": bool(r.conference_game),
            "home_rating": round(rh, 2) if rh is not None else None,
            "away_rating": round(ra, 2) if ra is not None else None,
            "margin_home": round(margin, 2) if margin is not None else None,
            "margin_home_current": round(m_cur, 2) if m_cur is not None else None,
            "p_home_current": round(p_cur, 4) if p_cur is not None else None,
            "sd_current": round(sd_cur, 2) if sd_cur is not None else None,
            "confidence_current": round(confidence, 4) if confidence is not None else None,
            # Validated tier: holdout accuracy at/above this confidence clears
            # the target with its 95% lower bound, not just its point estimate.
            "playable": (None if confidence is None or playable_at is None
                         else bool(confidence >= playable_at)),
            "unrated": unrated, "p_home": round(p_home, 4),
            "sd": round(sd, 2),
            "played": bool(r.completed),
            "home_points": None if pd.isna(r.home_points) else int(r.home_points),
            "away_points": None if pd.isna(r.away_points) else int(r.away_points),
        })

    # Each weekly comparison uses identical ratings, probabilities, and random
    # draws. Only games that are final by that week become fixed outcomes.
    teams = sorted(sec_teams)
    sim = weekly.simulate(games, teams, N_SIMS)
    weekly_history = weekly.history(games, teams, N_SIMS)
    current = weekly_history[-1]
    previous = weekly_history[-2] if len(weekly_history) > 1 else None
    previous_teams = {t["team"]: t for t in previous["teams"]} if previous else {}
    for team in sim:
        old = previous_teams.get(team["team"])
        team["change_proj_wins"] = round(team["proj_wins"] - old["proj_wins"], 4) if old else None
        team["change_outlook_rank"] = old["outlook_rank"] - team["outlook_rank"] if old else None
        team["change_conf_leader"] = round(team["p_conf_leader"] - old["p_conf_leader"], 4) if old else None

    simdf = pd.DataFrame(sim)
    sec = sec.merge(simdf, on="team", how="left")
    sec["conf_rank"] = np.arange(1, len(sec) + 1)

    # neutral-field grid
    grid = {}
    for a in teams:
        grid[a] = {b: round(float(norm.cdf(
            (cal["intercept"] + b1 * (rmap[a] - rmap[b]))
            / float(np.sqrt(game_sd_n ** 2 + extra)))), 4)
            for b in teams if b != a}

    if strength:
        ratings = ratings.assign(
            current_rating=ratings.team.map(
                lambda t: round(strength[t]["blended"], 2) if t in strength else None),
            games_rated=ratings.team.map(lambda t: strength.get(t, {}).get("n_games")),
            blend_weight=ratings.team.map(lambda t: strength.get(t, {}).get("weight")))
        sec = sec.merge(ratings[["team", "current_rating", "games_rated",
                                 "blend_weight"]], on="team", how="left")
        current_grid = {a: {b: round(float(norm.cdf(
            (cal_in["rating_diff_coef"] * (cmap[a] - cmap[b]))
            / cal_in["residual_sd"])), 4) for b in teams if b != a} for a in teams}
    else:
        current_grid = None

    # --- the model's own national ranking --------------------------------
    # A poll is votes; this is the rating, ordered. Movement is measured
    # against the model's own earlier state, not anyone else's ballot.
    rank_basis = cmap if cmap else rmap
    all_results = [{"home": r.home_team, "away": r.away_team,
                    "home_points": r.home_points, "away_points": r.away_points,
                    "neutral": bool(r.neutral)}
                   for _, r in full[full.completed].iterrows()]
    record_table = national.records(all_results, rmap.keys())
    prior_ratings = None
    if strength is not None and through_week and through_week > 1:
        earlier = full[full.completed & (full.week < through_week)]
        prior_rows = [{"home": r.home_team, "away": r.away_team,
                       "home_points": r.home_points, "away_points": r.away_points,
                       "neutral": bool(r.neutral)} for _, r in earlier.iterrows()]
        prior_ratings = {t: v["blended"]
                         for t, v in inseason.current_ratings(prior_rows, rmap, cal_in).items()}
    detail = strength or {}
    national_table = national.table(rank_basis, rmap, prior_ratings, record_table,
                                    detail)
    top25 = national_table[:25]

    oos = pd.read_csv(OUT / "oos_predictions.csv")
    explanation_path = OUT / ("explanations_%d.json" % season)
    explanations = json.loads(explanation_path.read_text()) \
        if explanation_path.exists() else {}
    payload = {
        "season": season,
        "conference": conference,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_type": "live" if any(g["played"] for g in games)
                         else "preseason",
        "data_source": source_info,
        "completed_games": sum(g["played"] for g in games),
        "current_period": current["label"],
        "comparison_period": previous["label"] if previous else None,
        "weekly_history": weekly_history,
        "history_method": "Reconstructed from fixed preseason ratings and each week's final results; future results are excluded.",
        "rating_basis": "Frozen preseason strength; results update records and season outlook, not strength ratings.",
        "generated_from": {
            "target": sel["target"],
            "winner": sel["winner"],
            "trained_through": sel["trained_through_season"],
            "n_features": len(sel["selected_features"]),
            "ridge_alpha": sel["ridge_alpha"],
        },
        "validation": {
            "by_config": {k: {kk: vv for kk, vv in v.items() if kk != "features"}
                          for k, v in sel["configurations"].items()},
            "by_season": val[val.config == sel["winner"]]
                [["season", "rmse", "mae", "r2", "spearman", "n_test", "split"]]
                .round(3).to_dict("records"),
            "oos_rmse": round(proj_rmse, 3),
            "split": sel["performance_claim"]["split"],
            "holdout_seasons": sel["holdout_seasons"],
        },
        "coefficients": sel["standardized_coefficients"],
        "block_ablation": sel["block_ablation"],
        "calibration": cal,
        "teams": sec.round(3).to_dict("records"),
        "all_ratings": ratings.round(2).to_dict("records"),
        "games": games,
        "top25": top25,
        "national_ranking": national_table,
        "top25_basis": "current_rating" if cmap else "preseason_rating",
        "ranking_through_week": through_week,
        "neutral_grid": grid,
        "current_grid": current_grid,
        "inseason": ({"calibration": cal_in,
                      "through_week": through_week,
                      "rated_teams": len(cmap)} if strength else None),
        "explanations": explanations,
        "oos_scatter": oos[oos.season >= sel["test_seasons"][0]]
            .round(2).to_dict("records"),
        "n_sims": N_SIMS,
        "fcs_win_rate": {"p": round(p_fcs, 4), "n_games": n_fcs},
    }
    path = OUT / "app_data.json"
    # NaN is not valid JSON -- json.dumps emits a bare NaN token that
    # JSON.parse rejects. Convert every non-finite value to null first.
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if o is not None and o != o:  # pandas NaT / NaN objects
            return None
        return o
    # A failed fetch/validation/simulation leaves the previous published data intact.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean(payload), allow_nan=False))
    temporary.replace(path)
    print("[export] %d teams, %d games, %d KB -> %s"
          % (len(sec), len(games), path.stat().st_size // 1024, path))
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("season", nargs="?", type=int, default=2026)
    parser.add_argument("conference", nargs="?", default="SEC")
    parser.add_argument("--source", choices=("auto", "cfbd", "espn"), default="auto")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    build(args.season, args.conference, args.source, args.refresh)
