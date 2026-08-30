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
import matchup as MU

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
N_SIMS = 20000


def fetch_schedule(season):
    """All regular-season games for `season`, including ones not yet played."""
    rows = []
    for gm in BD.safe("games", year=season, seasonType="regular"):
        ht, at = BD.g(gm, "home_team"), BD.g(gm, "away_team")
        if not ht or not at:
            continue
        rows.append({
            "week": BD.g(gm, "week"),
            "start_date": BD.g(gm, "start_date"),
            "home_team": ht, "away_team": at,
            "home_conference": BD.g(gm, "home_conference"),
            "away_conference": BD.g(gm, "away_conference"),
            "home_points": BD.g(gm, "home_points"),
            "away_points": BD.g(gm, "away_points"),
            "neutral": bool(BD.g(gm, "neutral_site", default=False)),
            "conference_game": bool(BD.g(gm, "conference_game", default=False)),
        })
    return pd.DataFrame(rows)


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


def build(season=2026, conference="SEC"):
    ratings = pd.read_csv(OUT / ("ratings_%d.csv" % season))
    sel = json.loads((OUT / "selection.json").read_text())
    cal = json.loads((OUT / "calibration.json").read_text())
    val = pd.read_csv(OUT / "validation.csv")
    proj_rmse = float(sel["performance_claim"]["rmse"])

    rmap = dict(zip(ratings.team, ratings.projected_rating))
    sec = ratings[ratings.conference == conference].copy()
    sec_teams = set(sec.team)

    sched = fetch_schedule(season)
    sched = sched[sched.home_team.isin(sec_teams) | sched.away_team.isin(sec_teams)]

    game_sd_n = cal["neutral_residual_sd"]
    game_sd_h = cal["residual_sd"]
    b1 = cal["rating_diff_coef"]
    extra = (0.0 if cal.get("method") == "walk-forward preseason predictions"
             else 2.0 * (b1 * proj_rmse) ** 2)

    from scipy.stats import norm
    games = []
    for _, r in sched.iterrows():
        rh, ra = rmap.get(r.home_team), rmap.get(r.away_team)
        if rh is None or ra is None:
            continue  # FCS or unrated opponent
        hf = 0.0 if r.neutral else 1.0
        margin = cal["intercept"] + b1 * (rh - ra) + cal["home_field_advantage"] * hf
        sd = float(np.sqrt((game_sd_n if r.neutral else game_sd_h) ** 2 + extra))
        p_home = float(norm.cdf(margin / sd))
        games.append({
            "week": int(r.week) if pd.notna(r.week) else None,
            "date": str(r.start_date)[:10] if r.start_date else None,
            "home": r.home_team, "away": r.away_team,
            "home_conf": r.home_conference, "away_conf": r.away_conference,
            "neutral": bool(r.neutral), "conf_game": bool(r.conference_game),
            "home_rating": round(rh, 2), "away_rating": round(ra, 2),
            "margin_home": round(margin, 2), "p_home": round(p_home, 4),
            "sd": round(sd, 2),
            "played": bool(pd.notna(r.home_points) and pd.notna(r.away_points)),
            "home_points": None if pd.isna(r.home_points) else int(r.home_points),
            "away_points": None if pd.isna(r.away_points) else int(r.away_points),
        })

    # ---- Monte Carlo season simulation (SEC teams only)
    teams = sorted(sec_teams)
    idx = {t: i for i, t in enumerate(teams)}
    rng = np.random.default_rng(20260829)
    wins = np.zeros((N_SIMS, len(teams)))
    cwins = np.zeros((N_SIMS, len(teams)))
    for g in games:
        h, a = g["home"], g["away"]
        if g["played"]:
            draw = np.full(N_SIMS, g["home_points"] > g["away_points"],
                           dtype=bool)
        else:
            draw = rng.random(N_SIMS) < g["p_home"]
        if h in idx:
            wins[:, idx[h]] += draw
            if g["conf_game"]:
                cwins[:, idx[h]] += draw
        if a in idx:
            wins[:, idx[a]] += ~draw
            if g["conf_game"]:
                cwins[:, idx[a]] += ~draw

    # Unrated (FCS) opponents: simulated at the historically observed FBS win
    # rate against them, so those games are neither ignored nor assumed certain.
    p_fcs, n_fcs = fcs_win_rate()
    unrated = {t: 0 for t in teams}
    for _, r in sched.iterrows():
        for me, opp in ((r.home_team, r.away_team), (r.away_team, r.home_team)):
            if me in unrated and opp not in rmap:
                unrated[me] += 1
                if pd.notna(r.home_points) and pd.notna(r.away_points):
                    me_home = me == r.home_team
                    won = ((r.home_points > r.away_points) if me_home
                           else (r.away_points > r.home_points))
                    wins[:, idx[me]] += won
                else:
                    wins[:, idx[me]] += rng.random(N_SIMS) < p_fcs

    sim = []
    for t in teams:
        i = idx[t]
        w = wins[:, i]
        cw = cwins[:, i]
        dist = np.bincount(w.astype(int), minlength=14)[:14] / N_SIMS
        sim.append({
            "team": t,
            "proj_wins": round(float(w.mean()), 2),
            "proj_conf_wins": round(float(cw.mean()), 2),
            "wins_p10": int(np.percentile(w, 10)),
            "wins_p90": int(np.percentile(w, 90)),
            "unrated_opponents": unrated[t],
            "win_distribution": [round(float(x), 4) for x in dist],
            "p_conf_leader": 0.0,
        })
    # share of sims in which each team has the outright-or-tied best conf record
    best = cwins.max(axis=1, keepdims=True)
    share = (cwins == best)
    share = share / share.sum(axis=1, keepdims=True)
    for s in sim:
        s["p_conf_leader"] = round(float(share[:, idx[s["team"]]].mean()), 4)

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
    path.write_text(json.dumps(clean(payload), allow_nan=False))
    print("[export] %d teams, %d games, %d KB -> %s"
          % (len(sec), len(games), path.stat().st_size // 1024, path))
    return payload


if __name__ == "__main__":
    a = sys.argv[1:]
    build(int(a[0]) if a else 2026, a[1] if len(a) > 1 else "SEC")
