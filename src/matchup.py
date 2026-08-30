"""Head-to-head matchup engine.

Turns two projected ratings into a projected margin and win probability using
a game-level model fitted on real results (fit.calibrate_games):

    margin(A vs B) = intercept + b1 * (rating_A - rating_B) + hfa * home_flag

Uncertainty combines the single-game residual SD (estimated from historical
games) with the model's own projection error (walk-forward OOS RMSE), so the
win probability widens appropriately for a preseason projection.

Usage:
  python src/matchup.py Georgia Alabama                 # neutral field
  python src/matchup.py Georgia Alabama --home Georgia  # Georgia hosting
  python src/matchup.py --matrix                        # full SEC neutral grid
  python src/matchup.py --season 2026 Georgia Alabama
"""
import argparse
import json
import math
import pathlib
import sys

import pandas as pd
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def _load(season=None):
    calib = json.loads((OUT / "calibration.json").read_text())
    sel = json.loads((OUT / "selection.json").read_text())
    if season is None:
        files = sorted(OUT.glob("ratings_*.csv"))
        if not files:
            raise SystemExit("No ratings file. Run: python src/rank.py")
        path = files[-1]
        season = int(path.stem.split("_")[1])
    else:
        path = OUT / ("ratings_%d.csv" % season)
        if not path.exists():
            raise SystemExit("Run: python src/rank.py %d" % season)
    ratings = pd.read_csv(path)
    proj_rmse = float(sel["performance_claim"]["rmse"])
    return season, ratings, calib, proj_rmse


def _rating(ratings, team):
    m = ratings[ratings.team.str.lower() == team.lower()]
    if not len(m):
        near = ratings[ratings.team.str.lower().str.contains(team.lower())]
        if len(near) == 1:
            m = near
        else:
            raise SystemExit("Unknown team '%s'. Candidates: %s"
                             % (team, ", ".join(near.team.tolist()[:10]) or "none"))
    return float(m.iloc[0].projected_rating), m.iloc[0].team


def matchup(team_a, team_b, home=None, season=None, include_projection_error=True):
    """home: None => neutral field; else the team name playing at home."""
    season, ratings, c, proj_rmse = _load(season)
    ra, na = _rating(ratings, team_a)
    rb, nb = _rating(ratings, team_b)

    if home is None:
        home_flag, site = 0.0, "neutral field"
    elif home.lower() in (na.lower(), team_a.lower()):
        home_flag, site = 1.0, "%s hosting" % na
    elif home.lower() in (nb.lower(), team_b.lower()):
        home_flag, site = -1.0, "%s hosting" % nb
    else:
        raise SystemExit("--home must be one of the two teams")

    margin = (c["intercept"] + c["rating_diff_coef"] * (ra - rb)
              + c["home_field_advantage"] * home_flag)

    game_sd = c["neutral_residual_sd"] if home_flag == 0 else c["residual_sd"]
    sd = game_sd
    # New calibrations are fit directly on historical walk-forward preseason
    # ratings, so their residual already contains projection uncertainty.
    if include_projection_error and c.get("method") != "walk-forward preseason predictions":
        # two independently projected ratings, propagated through b1
        sd = math.sqrt(game_sd ** 2
                       + 2.0 * (c["rating_diff_coef"] * proj_rmse) ** 2)

    p_a = float(norm.cdf(margin / sd))
    return {
        "season": season, "site": site,
        "team_a": na, "team_b": nb,
        "rating_a": round(ra, 2), "rating_b": round(rb, 2),
        "projected_margin_a": round(margin, 2),
        "win_prob_a": round(p_a, 4), "win_prob_b": round(1 - p_a, 4),
        "sd_used": round(sd, 2), "game_sd": round(game_sd, 2),
        "projection_rmse": round(proj_rmse, 2),
        "calibration_method": c.get("method", "final-season ratings"),
    }


def fmt(r):
    fav, dog = (r["team_a"], r["team_b"]) if r["projected_margin_a"] >= 0 \
        else (r["team_b"], r["team_a"])
    line = abs(r["projected_margin_a"])
    p = max(r["win_prob_a"], r["win_prob_b"])
    return (
        "%s vs %s  (%s, %d season)\n"
        "  projected: %s by %.1f\n"
        "  win prob : %s %.1f%%   |   %s %.1f%%\n"
        "  ratings  : %s %.1f, %s %.1f\n"
        "  spread SD: %.1f pts (calibrated directly from preseason forecasts)"
        % (r["team_a"], r["team_b"], r["site"], r["season"],
           fav, line, fav, p * 100, dog, (1 - p) * 100,
           r["team_a"], r["rating_a"], r["team_b"], r["rating_b"],
           r["sd_used"])
    )


def matrix(conference="SEC", season=None):
    season, ratings, _, _ = _load(season)
    teams = ratings[ratings.conference == conference].team.tolist()
    rows = []
    for a in teams:
        row = {"team": a}
        for b in teams:
            row[b] = "" if a == b else round(matchup(a, b, None, season)["win_prob_a"], 3)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("team")
    path = OUT / ("%s_neutral_matrix_%d.csv" % (conference.lower(), season))
    df.to_csv(path)
    print("Neutral-field win probability for the ROW team vs the COLUMN team "
          "(%s, %d):" % (conference, season))
    print(df.to_string())
    print("\nwrote %s" % path)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("teams", nargs="*")
    ap.add_argument("--home", default=None)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--conference", default="SEC")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.matrix:
        matrix(a.conference, a.season)
    elif len(a.teams) == 2:
        r = matchup(a.teams[0], a.teams[1], a.home, a.season)
        print(json.dumps(r, indent=2) if a.json else fmt(r))
    else:
        ap.error("give two team names, or --matrix")
