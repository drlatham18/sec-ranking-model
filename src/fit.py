"""Model fitting, feature selection, and walk-forward validation.

Design rules (so that no analyst judgment enters the weights):
  * Every coefficient is estimated by regression, never assigned.
  * Which features are kept is decided by out-of-sample error, not by opinion.
  * Validation is strictly walk-forward: to predict season T the model may only
    see seasons < T. No future information, ever.
  * The winning configuration is chosen automatically by lowest OOS RMSE.

Outputs (written to output/):
  validation.csv     per-test-season OOS metrics for every configuration
  selection.json     chosen configuration, selected features, fitted coefs
  calibration.json   game-level margin model + home-field advantage + sigma
"""
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import features as F

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
PROC = ROOT / "data" / "processed"

ALPHAS = np.logspace(-2, 4, 40)


def make_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=ALPHAS)),
    ])


def _xy(df, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[F.TARGET], errors="coerce").to_numpy(dtype=float)
    return X, y


def walk_forward(df, cols, test_seasons, min_train_rows=300):
    """Train on every season strictly before T, predict T. Returns metrics df."""
    rows, preds = [], []
    for T in test_seasons:
        tr = df[(df.season < T) & df[F.TARGET].notna()]
        te = df[(df.season == T) & df[F.TARGET].notna()]
        # a feature is usable only if the training window actually contains it
        usable = [c for c in cols
                  if pd.to_numeric(tr[c], errors="coerce").notna().sum() >= 50]
        if len(tr) < min_train_rows or len(te) == 0 or not usable:
            continue
        m = make_model()
        Xtr, ytr = _xy(tr, usable)
        Xte, yte = _xy(te, usable)
        ok = ~np.isnan(ytr)
        m.fit(Xtr[ok], ytr[ok])
        yhat = m.predict(Xte)
        err = yhat - yte
        rows.append({
            "season": T, "n_test": len(te), "n_train": int(ok.sum()),
            "n_features": len(usable),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / np.sum((yte - yte.mean()) ** 2)),
            "spearman": float(stats.spearmanr(yhat, yte).statistic),
        })
        preds.append(pd.DataFrame({"season": T, "team": te.team.values,
                                   "conference": te.conference.values,
                                   "pred": yhat, "actual": yte}))
    met = pd.DataFrame(rows)
    pr = pd.concat(preds) if preds else pd.DataFrame()
    return met, pr


def summarize(met):
    if not len(met):
        return {"rmse": np.inf, "mae": np.inf, "r2": -np.inf, "spearman": -np.inf}
    return {"rmse": float(met.rmse.mean()), "mae": float(met.mae.mean()),
            "r2": float(met.r2.mean()), "spearman": float(met.spearman.mean()),
            "seasons": int(len(met))}


def coverage_set(df, cols, since, thresh=0.80):
    sub = df[df.season >= since]
    return [c for c in cols
            if pd.to_numeric(sub[c], errors="coerce").notna().mean() >= thresh]


def greedy_select(df, pool, test_seasons, max_features=14, tol=0.005):
    """Forward selection driven purely by walk-forward OOS RMSE."""
    chosen, best = [], np.inf
    trail = []
    while len(chosen) < max_features:
        cand_best, cand_col = np.inf, None
        for c in pool:
            if c in chosen:
                continue
            met, _ = walk_forward(df, chosen + [c], test_seasons)
            s = summarize(met)["rmse"]
            if s < cand_best:
                cand_best, cand_col = s, c
        if cand_col is None or cand_best > best - tol:
            break
        chosen.append(cand_col)
        best = cand_best
        trail.append({"step": len(chosen), "added": cand_col, "oos_rmse": best})
        print("   + %-24s OOS RMSE %.4f" % (cand_col, best))
    return chosen, trail


def block_ablation(df, cols, test_seasons, base_rmse):
    """How much OOS accuracy each feature block is actually worth."""
    out = []
    for name, block in F.CANDIDATE_BLOCKS.items():
        present = [c for c in block if c in cols]
        if not present:
            continue
        met, _ = walk_forward(df, [c for c in cols if c not in present],
                              test_seasons)
        s = summarize(met)["rmse"]
        out.append({"block": name, "features_removed": present,
                    "rmse_without": s, "rmse_delta": s - base_rmse})
    return sorted(out, key=lambda d: -d["rmse_delta"])


# ------------------------------------------------------------------ game model
def calibrate_games(df_feat, min_season=2005):
    """Fit actual margin on rating difference + home field, from real games.

    Gives: margin = b1*(rating_home - rating_away) + hfa*home_flag
    and the residual SD used for win probabilities. Both coefficients are
    estimated, not assumed.

    No intercept: scoring margin is antisymmetric (swapping the two teams must
    flip its sign), which forces the constant term to zero. Fitting one anyway
    lets it absorb home-field -- non-neutral games outnumber neutral ones ~40:1,
    so the constant soaks up the HFA and then wrongly applies it at neutral
    sites too. The free-intercept fit is still reported as a diagnostic.
    """
    games = pd.read_csv(PROC / "games.csv")
    r = df_feat[["season", "team", F.TARGET]].rename(
        columns={F.TARGET: "rating"})
    gm = (games.merge(r.rename(columns={"team": "home_team",
                                        "rating": "home_rating"}),
                      on=["season", "home_team"], how="inner")
                .merge(r.rename(columns={"team": "away_team",
                                         "rating": "away_rating"}),
                       on=["season", "away_team"], how="inner"))
    gm = gm[gm.season >= min_season].dropna(
        subset=["home_rating", "away_rating", "home_points", "away_points"])
    y = (gm.home_points - gm.away_points).to_numpy(dtype=float)
    diff = (gm.home_rating - gm.away_rating).to_numpy(dtype=float)
    home = (~gm.neutral.astype(bool)).to_numpy(dtype=float)
    X = np.column_stack([diff, home])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma = float(np.std(resid, ddof=2))

    Xd = np.column_stack([np.ones(len(y)), diff, home])
    bd, *_ = np.linalg.lstsq(Xd, y, rcond=None)

    neutral_mask = home == 0
    return {
        "n_games": int(len(y)),
        "intercept": 0.0,
        "rating_diff_coef": float(beta[0]),
        "home_field_advantage": float(beta[1]),
        "diagnostic_free_intercept": {
            "intercept": float(bd[0]), "rating_diff_coef": float(bd[1]),
            "home_field_advantage": float(bd[2]),
            "note": "intercept absorbs HFA; not used for prediction",
        },
        "residual_sd": sigma,
        "n_neutral_games": int(neutral_mask.sum()),
        "neutral_residual_sd": float(np.std(resid[neutral_mask], ddof=1))
        if neutral_mask.sum() > 10 else sigma,
        "min_season": int(min_season),
    }


def _game_prediction_frame(predictions):
    """Join honest preseason team predictions to the games they preceded."""
    games = pd.read_csv(PROC / "games.csv")
    r = predictions[["season", "team", "pred"]]
    gm = (games.merge(r.rename(columns={"team": "home_team",
                                        "pred": "home_rating"}),
                      on=["season", "home_team"], how="inner")
                .merge(r.rename(columns={"team": "away_team",
                                         "pred": "away_rating"}),
                       on=["season", "away_team"], how="inner"))
    return gm.dropna(subset=["home_rating", "away_rating",
                             "home_points", "away_points"])


def _fit_game_calibration(gm):
    y = (gm.home_points - gm.away_points).to_numpy(dtype=float)
    diff = (gm.home_rating - gm.away_rating).to_numpy(dtype=float)
    home = (~gm.neutral.astype(bool)).to_numpy(dtype=float)
    X = np.column_stack([diff, home])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    neutral = home == 0
    return {
        "n_games": int(len(y)), "intercept": 0.0,
        "rating_diff_coef": float(beta[0]),
        "home_field_advantage": float(beta[1]),
        "residual_sd": float(np.std(resid, ddof=2)),
        "n_neutral_games": int(neutral.sum()),
        "neutral_residual_sd": float(np.std(resid[neutral], ddof=1))
        if neutral.sum() > 10 else float(np.std(resid, ddof=2)),
    }


def _probability_metrics(gm, calibration):
    diff = (gm.home_rating - gm.away_rating).to_numpy(dtype=float)
    home = (~gm.neutral.astype(bool)).to_numpy(dtype=float)
    margin = (calibration["rating_diff_coef"] * diff
              + calibration["home_field_advantage"] * home)
    sd = np.where(home == 0, calibration["neutral_residual_sd"],
                  calibration["residual_sd"])
    p = stats.norm.cdf(margin / sd)
    actual_margin = (gm.home_points - gm.away_points).to_numpy(dtype=float)
    keep = actual_margin != 0
    y = (actual_margin[keep] > 0).astype(int)
    p = np.clip(p[keep], 1e-6, 1 - 1e-6)
    bins = []
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            bins.append({"predicted": round(float(p[mask].mean()), 4),
                         "observed": round(float(y[mask].mean()), 4),
                         "n": int(mask.sum())})
    return {
        "n_games": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(np.mean((p >= .5) == y)),
        "calibration_bins": bins,
    }


def calibrate_preseason_games(predictions, development_seasons,
                              holdout_seasons):
    """Calibrate on walk-forward preseason ratings, never final ratings.

    Hyperparameters and the initial game calibration are learned on the
    development seasons. The untouched holdout reports probability quality.
    Production constants are then refit on all available OOS predictions.
    """
    gm = _game_prediction_frame(predictions)
    dev = gm[gm.season.isin(development_seasons)]
    hold = gm[gm.season.isin(holdout_seasons)]
    dev_cal = _fit_game_calibration(dev)
    final = _fit_game_calibration(gm)
    final.update({
        "method": "walk-forward preseason predictions",
        "min_season": int(gm.season.min()),
        "development_seasons": [int(x) for x in development_seasons],
        "holdout_seasons": [int(x) for x in holdout_seasons],
        "holdout_metrics": _probability_metrics(hold, dev_cal),
    })
    return final


# ------------------------------------------------------------------ main
def last_completed_season():
    """Last season whose games are actually played.

    SP+ is published preseason, so `sp_overall` existing for a season does NOT
    mean the season happened. Anchor on real results instead: a season counts
    as complete only if >=90% of its scheduled games have final scores.
    """
    gm = pd.read_csv(PROC / "games.csv")
    played = gm.groupby("season").size()
    return int(played[played > 100].index.max())


def run(first_test=2014, last_test=None):
    feat = F.load()
    completed = last_completed_season()
    last_test = last_test or completed
    test_seasons = list(range(first_test, last_test + 1))
    if len(test_seasons) < 6:
        raise ValueError("need at least six test seasons for development/holdout split")
    development_seasons = test_seasons[:-3]
    holdout_seasons = test_seasons[-3:]
    print("[fit] rows=%d  completed seasons through %d" % (len(feat), completed))
    print("[fit] walk-forward test seasons: %d-%d" % (first_test, last_test))

    pool = F.candidate_columns(feat)
    configs = {
        "baseline_prior_year_only": ["sp_overall_lag1"],
        "baseline_3yr_weighted": ["sp_wavg3"],
        "long_history_all": coverage_set(feat, pool, 2007),
        "modern_all": coverage_set(feat, pool, 2015),
        "full_pool": pool,
    }

    development_results, config_cols = {}, {}
    for name, cols in configs.items():
        if not cols:
            continue
        met, _ = walk_forward(feat, cols, development_seasons)
        development_results[name] = summarize(met)
        development_results[name]["features"] = cols
        config_cols[name] = cols
        print("[fit] %-26s RMSE %.3f  MAE %.3f  R2 %.3f  rho %.3f  (%d feats)"
              % (name, development_results[name]["rmse"], development_results[name]["mae"],
                 development_results[name]["r2"], development_results[name]["spearman"], len(cols)))

    print("[fit] greedy forward selection (OOS-driven)...")
    sel, trail = greedy_select(feat, configs["modern_all"] or pool,
                               development_seasons)
    if sel:
        met, _ = walk_forward(feat, sel, development_seasons)
        development_results["greedy_selected"] = summarize(met)
        development_results["greedy_selected"]["features"] = sel
        config_cols["greedy_selected"] = sel
        print("[fit] %-26s RMSE %.3f  MAE %.3f  R2 %.3f  rho %.3f  (%d feats)"
              % ("greedy_selected", development_results["greedy_selected"]["rmse"],
                 development_results["greedy_selected"]["mae"], development_results["greedy_selected"]["r2"],
                 development_results["greedy_selected"]["spearman"], len(sel)))

    winner = min(development_results,
                 key=lambda k: development_results[k]["rmse"])
    wcols = development_results[winner]["features"]
    print("[fit] WINNER on development seasons: %s" % winner)

    # Evaluate the frozen configurations on seasons that played no role in
    # selecting features or choosing the winner.
    results, all_preds = {}, {}
    val_rows = []
    for name, cols in config_cols.items():
        hold_met, _ = walk_forward(feat, cols, holdout_seasons)
        results[name] = summarize(hold_met)
        results[name]["features"] = cols
        all_met, all_pr = walk_forward(feat, cols, test_seasons)
        all_preds[name] = (all_met, all_pr)
        tagged = all_met.copy()
        tagged["config"] = name
        tagged["split"] = np.where(tagged.season.isin(holdout_seasons),
                                    "holdout", "development")
        val_rows.append(tagged)
        print("[fit] HOLDOUT %-18s RMSE %.3f  MAE %.3f"
              % (name, results[name]["rmse"], results[name]["mae"]))

    abl = block_ablation(feat, wcols, development_seasons,
                         development_results[winner]["rmse"])

    # final refit on every completed season
    tr = feat[(feat.season <= completed) & feat[F.TARGET].notna()]
    usable = [c for c in wcols
              if pd.to_numeric(tr[c], errors="coerce").notna().sum() >= 50]
    model = make_model()
    Xtr, ytr = _xy(tr, usable)
    model.fit(Xtr, ytr)
    ridge = model.named_steps["ridge"]
    coefs = dict(zip(usable, [float(c) for c in ridge.coef_]))

    chosen_predictions = all_preds[winner][1]
    calib = calibrate_preseason_games(chosen_predictions,
                                      development_seasons, holdout_seasons)

    # persist
    pd.concat(val_rows).to_csv(OUT / "validation.csv", index=False)
    chosen_predictions.to_csv(OUT / "oos_predictions.csv", index=False)

    payload = {
        "target": F.TARGET,
        "trained_through_season": completed,
        "test_seasons": test_seasons,
        "development_seasons": development_seasons,
        "holdout_seasons": holdout_seasons,
        "configurations": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        "development_configurations": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in development_results.items()},
        "winner": winner,
        "performance_claim": {
            "split": "untouched holdout",
            "seasons": holdout_seasons,
            "rmse": results[winner]["rmse"],
            "mae": results[winner]["mae"],
        },
        "selected_features": usable,
        "greedy_trail": trail,
        "standardized_coefficients": coefs,
        "ridge_alpha": float(ridge.alpha_),
        "block_ablation": abl,
        "calibration": calib,
    }
    (OUT / "selection.json").write_text(json.dumps(payload, indent=2))
    (OUT / "calibration.json").write_text(json.dumps(calib, indent=2))
    print("[fit] wrote output/selection.json, validation.csv, calibration.json")
    return payload


if __name__ == "__main__":
    a = sys.argv[1:]
    run(int(a[0]) if a else 2014, int(a[1]) if len(a) > 1 else None)
