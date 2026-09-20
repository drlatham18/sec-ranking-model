"""Fit and validate the in-season blend. Writes output/inseason_calibration.json.

Strict walk-forward, matching the rest of the project:
  * the preseason rating for season S comes from oos_predictions.csv, which was
    produced by a model trained only on seasons < S;
  * a game in season S week W is predicted using only results from S before W;
  * the shrinkage constant K is chosen by leave-one-season-out log loss on the
    DEVELOPMENT seasons, then the holdout seasons are scored once.

Nothing here is hand-set: K, the rating-difference coefficient, home-field
advantage and the residual SD are all estimated.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta, norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import inseason as IS

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
K_GRID = (2, 3, 4, 5, 6, 8, 10, 14, 20)
MIN_WEEK = 3          # weeks 1-2 have too little in-season evidence to compare
# A straight-up pick on every game cannot reach a high accuracy: see the oracle
# bound in the README. Accuracy on the games the model is CONFIDENT about is a
# different and much higher number, so the fit reports the whole curve and
# marks the lowest threshold whose 95% lower bound clears TARGET_ACCURACY.
CONFIDENCE_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
TARGET_ACCURACY = 0.85
MIN_TIER_GAMES = 100


def load():
    games = pd.read_csv(ROOT / "data" / "processed" / "games.csv")
    preds = pd.read_csv(OUT / "oos_predictions.csv")
    base = json.loads((OUT / "calibration.json").read_text())
    pre = {(s, t): p for s, t, p in zip(preds.season, preds.team, preds.pred)}
    games = games.dropna(subset=["home_points", "away_points"])
    games = games[games.season.isin(preds.season.unique())].copy()
    games["rh"] = [pre.get((s, t)) for s, t in zip(games.season, games.home_team)]
    games["ra"] = [pre.get((s, t)) for s, t in zip(games.season, games.away_team)]
    games = games.dropna(subset=["rh", "ra"]).sort_values(["season", "week"])
    games["hf"] = np.where(games.neutral, 0.0, 1.0)
    games["margin"] = games.home_points - games.away_points
    return games.reset_index(drop=True), pre, base


def walk_forward(games, pre, k, b1, hfa):
    """Blended home/away ratings for every game, using only earlier weeks."""
    rh = np.empty(len(games))
    ra = np.empty(len(games))
    for season in games.season.unique():
        rows = games[games.season == season]
        totals, counts = {}, {}
        for week in sorted(rows.week.unique()):
            index = rows.index[rows.week == week]
            for i in index:                       # predict before folding week in
                g = games.loc[i]
                for team, slot in ((g.home_team, rh), (g.away_team, ra)):
                    n = counts.get(team, 0)
                    base = pre[(season, team)]
                    slot[i] = base if not n else (
                        (1 - n / (n + k)) * base + (n / (n + k)) * totals[team] / n)
            for i in index:
                g = games.loc[i]
                for team, opponent, sign, flag in (
                        (g.home_team, g.ra, 1, g.hf), (g.away_team, g.rh, -1, -g.hf)):
                    totals[team] = totals.get(team, 0.0) + opponent + (
                        sign * g.margin - hfa * flag) / b1
                    counts[team] = counts.get(team, 0) + 1
    return rh, ra


def fit_game_model(frame, rh, ra):
    """OLS of margin on (rating difference, home flag). No intercept: margin is
    antisymmetric, and a free constant would absorb home-field."""
    x = np.column_stack([rh - ra, frame.hf.values])
    beta, *_ = np.linalg.lstsq(x, frame.margin.values, rcond=None)
    residual = frame.margin.values - x @ beta
    return float(beta[0]), float(beta[1]), float(residual.std(ddof=2))


def metrics(frame, rh, ra, b1, hfa, sd):
    pred = b1 * (rh - ra) + hfa * frame.hf.values
    actual = frame.margin.values
    y = (actual > 0).astype(float)
    p = np.clip(norm.cdf(pred / sd), 1e-9, 1 - 1e-9)
    return {
        "n_games": int(len(frame)),
        "accuracy": float(((p >= .5) == y).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "margin_mae": float(np.abs(pred - actual).mean()),
        "margin_rmse": float(np.sqrt(((pred - actual) ** 2).mean())),
    }


def confidence_curve(frame, rh, ra, b1, hfa, sd):
    """Accuracy and coverage as a function of how confident the model is.

    The 95% interval is Clopper-Pearson, so a tier is only reported as meeting
    the target when its LOWER bound clears it -- not merely its point estimate.
    """
    pred = b1 * (rh - ra) + hfa * frame.hf.values
    y = (frame.margin.values > 0).astype(float)
    p = np.clip(norm.cdf(pred / sd), 1e-9, 1 - 1e-9)
    confidence = np.maximum(p, 1 - p)
    correct = (p >= .5) == y
    tiers = []
    for threshold in CONFIDENCE_GRID:
        mask = confidence >= threshold
        n = int(mask.sum())
        if n < MIN_TIER_GAMES:
            continue
        k = int(correct[mask].sum())
        lo = float(beta.ppf(.025, k, n - k + 1)) if k else 0.0
        hi = float(beta.ppf(.975, k + 1, n - k)) if k < n else 1.0
        tiers.append({"min_confidence": threshold, "n_games": n,
                      "coverage": round(n / len(frame), 4),
                      "accuracy": round(k / n, 4),
                      "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                      "meets_target": lo >= TARGET_ACCURACY})
    qualifying = [t for t in tiers if t["meets_target"]]
    return tiers, (qualifying[0]["min_confidence"] if qualifying else None)


def main():
    games, pre, base = load()
    dev_seasons = base["development_seasons"]
    hold_seasons = base["holdout_seasons"]
    b1_scale, hfa_scale = base["rating_diff_coef"], base["home_field_advantage"]
    late = games.week >= MIN_WEEK
    dev = games.season.isin(dev_seasons) & late
    hold = games.season.isin(hold_seasons) & late

    # --- choose K on development seasons only -------------------------------
    selection = {}
    for k in K_GRID:
        rh, ra = walk_forward(games, pre, k, b1_scale, hfa_scale)
        total, n = 0.0, 0
        for season in dev_seasons:
            tr = dev & (games.season != season)
            te = dev & (games.season == season)
            b1, hfa, sd = fit_game_model(games[tr], rh[tr.values], ra[tr.values])
            m = metrics(games[te], rh[te.values], ra[te.values], b1, hfa, sd)
            total += m["log_loss"] * m["n_games"]
            n += m["n_games"]
        selection[k] = total / n
        print("  K=%-3d dev LOSO log loss = %.4f" % (k, selection[k]))
    k_star = min(selection, key=selection.get)
    print("  -> K* = %d (holdout seasons untouched)" % k_star)

    # --- fit on development, score the holdout once -------------------------
    rh, ra = walk_forward(games, pre, k_star, b1_scale, hfa_scale)
    b1, hfa, sd = fit_game_model(games[dev], rh[dev.values], ra[dev.values])
    blend_hold = metrics(games[hold], rh[hold.values], ra[hold.values], b1, hfa, sd)
    pb1, phfa, psd = fit_game_model(games[dev], games[dev].rh.values, games[dev].ra.values)
    pre_hold = metrics(games[hold], games[hold].rh.values, games[hold].ra.values,
                       pb1, phfa, psd)

    tiers, playable = confidence_curve(
        games[hold], rh[hold.values], ra[hold.values], b1, hfa, sd)

    payload = {
        "method": "preseason rating shrunk toward opponent-adjusted in-season results",
        "K": k_star,
        "K_selection": {"criterion": "leave-one-season-out log loss on development seasons",
                        "development_seasons": dev_seasons,
                        "log_loss_by_K": {str(a): b for a, b in selection.items()}},
        "scale_b1": b1_scale,
        "scale_hfa": hfa_scale,
        "rating_diff_coef": b1,
        "home_field_advantage": hfa,
        "residual_sd": sd,
        "min_week_validated": MIN_WEEK,
        "holdout_seasons": hold_seasons,
        "holdout_metrics": blend_hold,
        "preseason_baseline_holdout": pre_hold,
        "target_accuracy": TARGET_ACCURACY,
        "confidence_tiers": tiers,
        "playable_threshold": playable,
        "preseason_baseline_coefficients": {"rating_diff_coef": pb1,
                                            "home_field_advantage": phfa,
                                            "residual_sd": psd},
    }
    (OUT / "inseason_calibration.json").write_text(json.dumps(payload, indent=2))
    print("\n  holdout %s, week >= %d, n=%d" % (hold_seasons, MIN_WEEK, blend_hold["n_games"]))
    for key in ("accuracy", "brier", "log_loss", "margin_mae"):
        print("    %-12s preseason %.4f -> blend %.4f" % (key, pre_hold[key], blend_hold[key]))
    print("\n  accuracy by confidence (untouched holdout):")
    for t in tiers:
        flag = "  <- meets %.0f%% target" % (TARGET_ACCURACY * 100) if t["meets_target"] else ""
        print("    >=%.2f  n=%-5d cover=%5.1f%%  acc=%.3f  95%% CI [%.3f, %.3f]%s"
              % (t["min_confidence"], t["n_games"], t["coverage"] * 100,
                 t["accuracy"], t["ci_low"], t["ci_high"], flag))
    print("  playable threshold: %s" % (playable if playable else "none reaches target"))
    print("\n  wrote %s" % (OUT / "inseason_calibration.json"))


if __name__ == "__main__":
    main()
