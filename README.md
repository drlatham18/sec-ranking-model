# SEC Independent Ranking Model

A preseason team-strength model for SEC football. Every weight in it is
estimated from historical data by regression; none is assigned by hand.

## Design rules

1. **No hand-set weights.** Coefficients come from ridge regression. The only
   human choice is which *candidate* variables get offered to the model.
2. **The data decides what's predictive.** Feature sets are compared by
   walk-forward RMSE on development seasons. Greedy forward selection adds a
   variable only if it measurably reduces error. The frozen winner is then
   scored on three untouched holdout seasons; only that holdout score is used
   in product accuracy claims and forecast intervals.
3. **Strict walk-forward validation.** To predict season *T*, the model may
   only train on seasons < *T*. No in-season information from *T* is ever a
   feature.
4. **The winning configuration is chosen by code**, as the minimum OOS RMSE
   among all configurations tested — including the naive baselines it has to
   beat.

## Target

`sp_overall` — SP+ overall rating (points per game vs. an average opponent).
It is opponent-adjusted, so it measures team quality rather than record, and
it is on a points scale, which makes the matchup engine straightforward.

## Candidate variables (all preseason-knowable)

| Block | Variables |
|---|---|
| prior_results | SP+ in t-1/t-2/t-3, weighted and simple 3-yr averages, trend, prior offense/defense SP+, prior scoring margin, prior wins |
| luck_regression | wins minus expected wins, one-score game win %, expected wins |
| recruiting | 247-composite class points (current + lags + 4-yr mean), blue-chip ratio, five-star count |
| roster_talent | CFBD talent composite, its lag, year-over-year change |
| returning_production | returning % of PPA (total/pass/rush/receiving) and usage |
| transfer_portal | in/out counts, net count, in/out talent value, net value |
| coaching | head-coach change flag, coach tenure |
| context | prior-year mean SP+ of the team's conference |

Availability differs by era (portal ~2021+, returning production ~2014+). The
walk-forward loop drops any variable the training window doesn't actually
contain, so early seasons still contribute.

## Matchup engine

Game-level model fitted on games joined to historical walk-forward preseason
predictions:

```
margin = b1 * (rating_home - rating_away) + hfa * home_flag
```

`b1`, `hfa` (home-field advantage) and the residual standard deviation are all
**estimated from those games**, not assumed. There is no constant term: margin is
antisymmetric, and a free intercept would absorb home-field and then wrongly
apply it at neutral sites too. Neutral-field games are flagged in
the source data and get their own residual SD.

Win probability is the normal CDF of the projected margin over the directly
observed preseason game-error distribution. Brier score, log loss, accuracy,
and reliability bins are reported on the untouched holdout seasons.

## Layout

```
src/cfbd_client.py    CollegeFootballData API client, disk-cached
src/build_dataset.py  pulls raw endpoints -> data/processed/panel.csv, games.csv
src/features.py       lagged / preseason feature construction
src/fit.py            selection, walk-forward validation, game calibration
src/rank.py           applies the winning model to a target season
src/matchup.py        head-to-head margins and win probabilities
run_all.py            end-to-end: build -> fit -> rank
```

## Usage

```bash
# one-time: key from https://collegefootballdata.com/key
echo "YOUR_KEY" > ~/.cfbd_key

python run_all.py                       # complete data-to-product rebuild
python src/build_dataset.py 2004 2026    # rebuild panel (cached; re-run is cheap)
python src/fit.py 2014                   # refit + revalidate from test season 2014
python src/rank.py 2026 SEC              # ranking for a season / conference

python src/matchup.py Georgia Alabama                  # neutral field
python src/matchup.py Georgia Alabama --home Georgia   # home game
python src/matchup.py --matrix                         # full SEC neutral grid
python src/matchup.py Georgia Alabama --json           # machine-readable
python src/build_ui.py                                 # rebuild the web UI
```

## Outputs (`output/`)

| File | Contents |
|---|---|
| `selection.json` | winning configuration, selected features, standardized coefficients, ridge alpha, block ablation, game calibration |
| `validation.csv` | per-test-season OOS RMSE / MAE / R² / Spearman for every configuration |
| `oos_predictions.csv` | every out-of-sample team-season prediction vs. actual |
| `ratings_<year>.csv` | projected rating for all FBS teams, with 68% / 95% bands |
| `sec_ranking_<year>.csv` | the SEC ranking |
| `sec_neutral_matrix_<year>.csv` | neutral-field win probability grid |
| `calibration.json` | fitted HFA, rating-difference coefficient, residual SDs |
| `data_quality.json` | source coverage, missingness and integrity checks |

## Reading the validation output

`validation.csv` labels each row as `development` or `holdout` and includes two
deliberately naive baselines. `selection.json.performance_claim` is the frozen
winner's untouched holdout result; do not substitute the development score in
marketing copy.

## Reproducibility and checks

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Raw CFBD responses are private build inputs and are excluded by `.gitignore`.
See `NOTICE.md` before distributing or commercializing the product.

## Web UI

`ui/template.html` is the page; `src/build_ui.py` injects `output/app_data.json`
into it and writes `ui/index.html` — a single self-contained file that opens in
any browser, mobile included.

```bash
python src/export_data.py 2026 SEC   # schedule pull + Monte Carlo -> app_data.json
python src/build_ui.py               # -> ui/index.html
```

Tabs: **Ranking** (rating, projected record, win-total distribution, odds of the
best SEC record), **Matchup Lab** (any two of the 139 rated FBS teams, neutral or
home, with the fitted margin distribution), **Neutral Grid** (16×16 win-probability
heatmap), **Schedule** (every 2026 game with its line and win probability), and
**Model Card** (validation vs. baselines, per-season accuracy, fitted weights,
block ablation, game-model constants).

Rebuild the whole thing end to end with `python run_all.py`, then the two
commands above.
