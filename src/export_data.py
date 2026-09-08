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

    sched, source_info = schedule_data.fetch_schedule(season, source, refresh)
    sched = sched[sched.home_team.isin(sec_teams) | sched.away_team.isin(sec_teams)]

    schedule_data.validate_schedule(sched, sec_teams)
    p_fcs, n_fcs = fcs_win_rate()

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
        games.append({
            "id": str(r.id),
            "week": int(r.week) if pd.notna(r.week) else None,
            "date": str(r.start_date)[:10] if r.start_date else None,
            "home": r.home_team, "away": r.away_team,
            "home_conf": r.home_conference, "away_conf": r.away_conference,
            "neutral": bool(r.neutral), "conf_game": bool(r.conference_game),
            "home_rating": round(rh, 2) if rh is not None else None,
            "away_rating": round(ra, 2) if ra is not None else None,
            "margin_home": round(margin, 2) if margin is not None else None,
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
        "neutral_grid": grid,
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
