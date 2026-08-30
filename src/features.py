"""Feature engineering.

Every feature is constructed to be knowable BEFORE season t kicks off:
  - lagged on-field results        (seasons t-1, t-2, t-3)
  - recruiting classes             (signed Feb of year t or earlier)
  - roster talent composite        (published for year t preseason)
  - returning production           (computed from year t-1 roster attrition)
  - transfer portal for cycle t
  - coaching continuity through year t

No feature uses any in-season information from season t.
Target: sp_overall (SP+ points-per-game vs average) in season t.
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

TARGET = "sp_overall"

# columns lagged from prior seasons
LAGGABLE = ["sp_overall", "sp_offense", "sp_defense", "sp_special",
            "margin_pg", "ppg", "papg", "wins", "expected_wins",
            "close_win_pct", "talent", "recruit_points", "bluechip_ratio"]


def _lag(panel, k):
    cols = ["team", "season"] + [c for c in LAGGABLE if c in panel.columns]
    d = panel[cols].copy()
    d["season"] = d["season"] + k
    ren = {c: "%s_lag%d" % (c, k) for c in cols if c not in ("team", "season")}
    return d.rename(columns=ren)


def build_features(panel):
    p = panel.copy()
    p["season"] = p["season"].astype(int)

    # --- conference strength in t-1 (objective, from the panel itself)
    conf_prev = (p.groupby(["season", "conference"])[TARGET].mean()
                 .reset_index().rename(columns={TARGET: "conf_sp_mean"}))
    conf_prev["season"] = conf_prev["season"] + 1
    p = p.merge(conf_prev, on=["season", "conference"], how="left")

    # --- coaching continuity
    p = p.sort_values(["team", "season"])
    prev_coach = p.groupby("team")["coach"].shift(1)
    prev_season = p.groupby("team")["season"].shift(1)
    contiguous = (p["season"] - prev_season) == 1
    p["coach_change"] = np.where(
        contiguous, (p["coach"] != prev_coach).astype(float), np.nan)

    tenure, last_team, last_coach, run = [], None, None, 0
    for _, r in p.iterrows():
        if r["team"] != last_team or r["coach"] != last_coach:
            run = 1
        else:
            run += 1
        tenure.append(run)
        last_team, last_coach = r["team"], r["coach"]
    p["coach_tenure"] = tenure

    # --- lags
    for k in (1, 2, 3):
        p = p.merge(_lag(panel, k), on=["team", "season"], how="left")

    # --- derived composites
    p["sp_wavg3"] = (0.5 * p.sp_overall_lag1 + 0.3 * p.sp_overall_lag2
                     + 0.2 * p.sp_overall_lag3)
    p["sp_mean3"] = p[["sp_overall_lag1", "sp_overall_lag2",
                       "sp_overall_lag3"]].mean(axis=1)
    p["sp_trend"] = p.sp_overall_lag1 - p.sp_overall_lag2
    p["luck_lag1"] = p.wins_lag1 - p.expected_wins_lag1
    p["recruit_mean4"] = p[["recruit_points", "recruit_points_lag1",
                            "recruit_points_lag2",
                            "recruit_points_lag3"]].mean(axis=1)
    p["bluechip_mean4"] = p[["bluechip_ratio", "bluechip_ratio_lag1",
                             "bluechip_ratio_lag2",
                             "bluechip_ratio_lag3"]].mean(axis=1)
    p["talent_delta"] = p.talent - p.talent_lag1
    p["talent_vs_recent_sp"] = p.talent - p.sp_mean3 * 10  # scale-mismatched proxy
    p["portal_val_per_in"] = p.portal_in_val / p.portal_in.replace(0, np.nan)

    return p


# Candidate pool. Nothing here is pre-weighted; fit.py decides what survives
# out-of-sample. Grouped only so that selection can report block contributions.
CANDIDATE_BLOCKS = {
    "prior_results": ["sp_overall_lag1", "sp_overall_lag2", "sp_overall_lag3",
                      "sp_wavg3", "sp_mean3", "sp_trend",
                      "sp_offense_lag1", "sp_defense_lag1",
                      "margin_pg_lag1", "wins_lag1"],
    "luck_regression": ["luck_lag1", "close_win_pct_lag1", "expected_wins_lag1"],
    "recruiting": ["recruit_points", "recruit_points_lag1", "recruit_mean4",
                   "bluechip_ratio", "bluechip_mean4", "fivestars"],
    "roster_talent": ["talent", "talent_lag1", "talent_delta"],
    "returning_production": ["ret_ppa", "ret_pass_ppa", "ret_rush_ppa",
                             "ret_recv_ppa", "ret_usage", "ret_pass_usage"],
    "transfer_portal": ["portal_in", "portal_out", "portal_net",
                        "portal_in_val", "portal_out_val", "portal_net_val",
                        "portal_val_per_in"],
    "coaching": ["coach_change", "coach_tenure"],
    "context": ["conf_sp_mean"],
}


def candidate_columns(df):
    cols = []
    for block in CANDIDATE_BLOCKS.values():
        cols += [c for c in block if c in df.columns]
    # drop all-NaN / zero-variance columns -- purely mechanical, no judgment
    keep = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0 and s.std(skipna=True) not in (0, None) \
                and not np.isnan(s.std(skipna=True)):
            keep.append(c)
    return keep


def load():
    panel = pd.read_csv(PROC / "panel.csv")
    return build_features(panel)


if __name__ == "__main__":
    f = load()
    cc = candidate_columns(f)
    print("rows=%d  candidate features=%d" % (len(f), len(cc)))
    cov = f[f.season >= 2015][cc].notna().mean().sort_values()
    print(cov.to_string())
