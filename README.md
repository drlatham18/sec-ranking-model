# SEC Independent Ranking Model

Public site: https://drlatham18.github.io/sec-ranking-model/

## Best Bets

Open `?tab=bets` for read-only SEC full-game moneyline comparisons against
**Polymarket Global** (not Polymarket US). Public Gamma/CLOB endpoints require
no copied trading credentials. `src/best_bets.py` runs after the results export
and the build embeds `output/best_bets.json` in the page.

Straight mismatch means our current in-season model favors the market underdog,
with at least 5 percentage points of edge after entry fees. Good buy adds positive
24-hour momentum of at least 1 cent and a hypothetical exit halfway from the ask
to model fair value, with at least 2 cents/share left after entry and exit fees.
This is a transparent screening heuristic, not a validated price-movement model.
Both screens require a spread at most 4 cents and at least 100 shares on each side
of the top of book. Thresholds do not alter the fitted ranking model.

Exact teams and kickoff must match the model schedule. Started games, unrated
teams, unknown fees, stale books, and partial/prop contracts are excluded.
No market-feed failure reuses old candidates as fresh; the page shows an unavailable
state. Candidate display expires after 7 hours normally or 75 minutes on game days,
and at kickoff. The page reports coverage even when no candidates qualify.

The existing Pages workflow checks every 30 minutes and deploys every six hours
normally, or every half hour on SEC game days in America/New_York. Its gate checks
the stored schedule plus ESPN for reschedules. GitHub schedules are best effort
and can be delayed. Results and market prices refresh; the preseason fit does not
retrain. No second scheduler or trading runner is created.

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
src/inseason.py       blends preseason ratings with results to date
src/polls.py          AP / Coaches polls and the comparison to our own Top 25
src/fit_inseason.py   fits + validates that blend, writes inseason_calibration.json
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
| `inseason_calibration.json` | shrinkage constant K, blend game coefficients, holdout metrics vs. the preseason baseline, accuracy-by-confidence tiers |

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

## Scheduled refresh and completed-season checks

Historical data updates and model fitting stay on Body 1. Mac checkouts support
backup and manual heavy workloads. `python run_all.py 2004 2026 --refresh`
explicitly bypasses every API cache entry and requires the configured CFBD key.
Without the flag, changing-season data expires after six hours and historical
data after 30 days. Legacy cache files without a fetch timestamp refresh once.

Dataset builds now save `data/processed/season_status.csv`, with the total
regular-season schedule and final-score count. Fitting requires at least 90%
completion, agreement with the games dataset, and February 1 after the season.
Existing datasets must be rebuilt to create this provenance before refitting;
fitting will fail clearly rather than guess from the number of played games.
Committed model outputs are snapshots and are not refreshed by installing code.

## Top 25 vs the AP poll

`src/polls.py` pulls the AP and Coaches polls from ESPN's keyless rankings
endpoint, so this costs nothing against the CFBD key quota, and lines them up
against our own top 25 by current rating.

One timing trap is made explicit rather than glossed over: **the poll labelled
"Week N" is voted before week N is played**, so it reflects results through
week N-1. Comparing it against a rating that already includes week N would
flatter the rating. The UI therefore offers two views:

- **Like for like** - our ranking recomputed using only the games the voters
  had seen, so neither side has an information advantage.
- **Our latest vs AP** - our current ranking, which knows more than the poll did.

A rating and a poll are not the same measurement. `sp_overall` estimates points
per game against an average opponent; a poll is voters weighing record,
opponent quality and reputation. Their disagreements are the output, not an
error in either.

## In-season strength

The preseason model cannot see the season being played. `src/inseason.py` folds
completed games back into each rating: every result against a rated opponent
yields an opponent- and venue-adjusted implied rating, and preseason is shrunk
toward the mean of those by `weight = n / (n + K)`.

`K` is not hand-set. It is chosen by leave-one-season-out log loss on the
development seasons, then the holdout seasons are scored once. On 1,979
untouched holdout games (2023-2025, week 3 on):

| | preseason only | + in-season blend |
|---|---|---|
| Straight-up accuracy | 0.658 | **0.721** |
| Brier | 0.209 | **0.182** |
| Log loss | 0.601 | **0.538** |
| Margin MAE | 13.87 | **12.53** |

The gain holds in every week window and grows late in the year (week 13+:
0.631 -> 0.727). A simultaneous ridge/SRS opponent solve was also tested and did
*not* beat this simpler blend on development log loss, so it was not adopted.

### How accurate can this get?

An **oracle** fitted on the full season *including the game it predicts* --
maximum leakage, impossible in practice -- reaches only **0.791** straight-up on
this same game set. No honest team-strength model can beat that, and a target
above it is not reachable by improving the model.

What does clear a high bar is accuracy on the games the model is confident
about. `fit_inseason.py` reports the whole accuracy-vs-coverage curve and marks
the lowest confidence tier whose **95% lower bound** clears the target:

| min confidence | games | coverage | accuracy | 95% CI |
|---|---|---|---|---|
| >=0.50 (every game) | 1979 | 100% | 0.721 | [0.700, 0.740] |
| >=0.65 | 1222 | 62% | 0.813 | [0.790, 0.835] |
| >=0.75 | 752 | 38% | 0.874 | [0.848, 0.897] |
| **>=0.80** | **563** | **28%** | **0.895** | **[0.867, 0.919]** |
| >=0.90 | 229 | 12% | 0.965 | [0.932, 0.985] |

The shipped `playable_threshold` is the first tier meeting the target on its
lower bound. Every game in `app_data.json` carries `confidence_current` and a
`playable` flag derived from it. Coverage is the cost: the tier that clears 85%
covers about a quarter of the slate.

Sample size is the thing to watch. Ten games cannot establish reliability -- a
9/11 result has a 95% interval of roughly [0.48, 0.98]. The numbers above rest
on 1,979 games precisely so they mean something.

## Results and week-to-week comparisons

The **Week to week** tab compares projected win totals, actual records, outlook
rank (ordered by projected total wins), and best-SEC-record odds against the
previous completed week. An unfinished week is explicitly labeled partial.
Strength ratings remain the fitted preseason forecast; this does not introduce
an unvalidated in-season strength model. Weekly history is reconstructed with
the same preseason ratings and only the results final by each week's cutoff.
Expected wins are calculated directly from probabilities; stable per-game
simulation draws prevent refresh noise from appearing as weekly movement.

```bash
# Lightweight results refresh; no historical pulls or model fitting:
python src/export_data.py 2026 SEC --refresh
python src/build_ui.py

# Explicit keyless results source:
python src/export_data.py 2026 SEC --source espn --refresh
```

Both feeds are consulted every run. CFBD returns the whole season in **one**
request, so it is the spine when `CFBD_API_KEY` or `~/.cfbd_key` is configured;
ESPN is keyless but needs a request per conference per week, so it audits.

**Cross-checking.** The two sources name FCS schools differently, so merging
them would duplicate games. The spine stays authoritative for which games
exist; the other feed audits only the games matching on week and both team
names. A final score the feeds *disagree* on is not published at all -- one of
them is wrong, and a wrong score silently corrupts every rating built on it, so
that game is marked unplayed and the conflict is recorded in
`data_source.audit`. Games that cannot be matched are counted `unverified`.

Either source failing is survivable and logged; both failing fails the build. A
game flagged complete with no final score (a cancellation) is treated as not
played rather than killing the run, but a feed *full* of them still fails.

**API quota.** CFBD keys allow 1000 requests/month. A build spends **exactly
one** -- the schedule and the all-FBS in-season results come from the same
response, and a test pins that count so it cannot regress. At the shipped
cadence (daily year round, plus six-hourly in August-December) that is about
155 requests in a peak month, leaving room for manual runs. Both adapters require a final-game
flag before locking a score. All 120 games are included, including FCS opponents;
conference record denominators come from the actual nine-game 2026 schedule.
Incomplete schedules are rejected before replacing the previous output.

GitHub Pages now fetches results and checks integrity before every push/manual
deployment. A failed refresh leaves the previously deployed site available.
The page shows the source, check time, final-game count, and a stale-results
notice after 48 hours. GitHub Actions refreshes results every six hours during
every day year round at 11:20 UTC (~07:20 ET), and additionally every six hours
during August–December (00:17, 06:17, 12:17, 18:17 UTC). Each run refreshes
results and recomputes the in-season blend from them. Model fitting stays on Body 1. To manually refresh without a code change,
run **Deploy GitHub Pages** from the repository's Actions tab.

A scheduled refresh that fails opens (or comments on) a `refresh-failure` issue
instead of going unnoticed: the deployed site keeps serving its last good data,
so a broken feed shows up only as scores that stop advancing.

The ESPN adapter pulls one regular-season week at a time. ESPN no longer serves
multi-day `dates=A-B` ranges on the scoreboard endpoint — a range answers HTTP
400 — which silently stopped every scheduled refresh until it was replaced with
per-week requests.

Stable public links always open the latest deployed data:

- Main ranking: https://drlatham18.github.io/sec-ranking-model/
- Weekly changes: https://drlatham18.github.io/sec-ranking-model/?tab=weekly
- Results and schedule: https://drlatham18.github.io/sec-ranking-model/?tab=sched
- Team example: https://drlatham18.github.io/sec-ranking-model/?tab=team&team=Tennessee

Use the Share button to copy the current view. A weekly link with an explicit
`week` query value selects that historical comparison; omit `week` to follow the
latest week automatically. The season rolls over when a new preseason model is
built and the workflow's export season is updated; scheduled score refreshes do
not refit ratings or silently switch seasons.
