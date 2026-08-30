"""Produce the ranking for a target season using the fitted model.

Refits the winning configuration on every completed season, then applies it to
the target season's preseason features. Nothing here re-weights or overrides
the model output; it only formats it.

Outputs: output/ratings_<season>.csv   (all FBS)
         output/sec_ranking_<season>.csv
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import features as F
import fit as FIT

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def fit_final(feat, cols, through):
    tr = feat[(feat.season <= through) & feat[F.TARGET].notna()]
    usable = [c for c in cols
              if pd.to_numeric(tr[c], errors="coerce").notna().sum() >= 50]
    m = FIT.make_model()
    X, y = FIT._xy(tr, usable)
    m.fit(X, y)
    return m, usable


def project(season=None, conference="SEC"):
    sel = json.loads((OUT / "selection.json").read_text())
    feat = F.load()
    completed = int(sel["trained_through_season"])
    season = season or int(feat.season.max())
    if season <= completed:
        print("[rank] note: season %d is already complete; producing a "
              "retrodiction using only pre-%d information." % (season, season))
        completed = season - 1

    model, cols = fit_final(feat, sel["selected_features"], completed)

    tgt = feat[feat.season == season].copy()
    if not len(tgt):
        raise SystemExit("No panel rows for season %d -- rebuild the dataset "
                         "with that year included." % season)
    X, _ = FIT._xy(tgt, cols)
    tgt["projected_rating"] = model.predict(X)

    # Per-team standardized contribution breakdown for the product's "why"
    # view. Contributions sum to the projected rating with the fitted intercept.
    imp = model.named_steps["impute"].transform(X)
    scaled = model.named_steps["scale"].transform(imp)
    ridge = model.named_steps["ridge"]
    contribution = scaled * ridge.coef_
    explanations = {}
    for i, team in enumerate(tgt.team):
        drivers = [{"feature": c, "points": round(float(v), 3)}
                   for c, v in zip(cols, contribution[i])]
        drivers.sort(key=lambda d: abs(d["points"]), reverse=True)
        explanations[team] = {
            "intercept": round(float(ridge.intercept_), 3),
            "drivers": drivers[:6],
        }
    (OUT / ("explanations_%d.json" % season)).write_text(
        json.dumps(explanations, indent=2))

    # uncertainty = walk-forward OOS RMSE of the winning configuration
    val = pd.read_csv(OUT / "validation.csv")
    rmse = float(sel["performance_claim"]["rmse"])
    tgt["lo_68"] = tgt.projected_rating - rmse
    tgt["hi_68"] = tgt.projected_rating + rmse
    tgt["lo_95"] = tgt.projected_rating - 1.96 * rmse
    tgt["hi_95"] = tgt.projected_rating + 1.96 * rmse

    tgt = tgt.sort_values("projected_rating", ascending=False)
    tgt["national_rank"] = np.arange(1, len(tgt) + 1)

    keep = ["national_rank", "team", "conference", "projected_rating",
            "lo_68", "hi_68", "lo_95", "hi_95"]
    allr = tgt[keep].round(2)
    allr.to_csv(OUT / ("ratings_%d.csv" % season), index=False)

    conf = tgt[tgt.conference == conference].copy()
    conf["conf_rank"] = np.arange(1, len(conf) + 1)
    conf = conf[["conf_rank"] + keep].round(2)
    conf.to_csv(OUT / ("%s_ranking_%d.csv" % (conference.lower(), season)),
                index=False)

    print("\n=== %s projected ranking, %d season ===" % (conference, season))
    print("model=%s  features=%d  walk-forward OOS RMSE=%.2f pts"
          % (sel["winner"], len(cols), rmse))
    print(conf.to_string(index=False))
    print("\nwrote %s" % (OUT / ("%s_ranking_%d.csv" % (conference.lower(), season))))
    return tgt


if __name__ == "__main__":
    a = sys.argv[1:]
    project(int(a[0]) if a else None, a[1] if len(a) > 1 else "SEC")
