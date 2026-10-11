# Historical training and evaluation — Prototype 10

The new NFL add-on layers and standalone ATTD probabilities are editable
strategy assumptions. The existing NFL historical training/reference fit still
uses the unchanged core score model; its validation does not validate these
new modifiers. See `nfl_layers_guide.md` for the default-on controls, data
requirements and callable wrappers. Compare archived NFL forecasts with layers
on/off before adopting their coefficients as calibrated probabilities.

## Upload location and data choice

Use **Setup & Saved Data → CFB historical data, training and trained-model restore** in the Streamlit dashboard. It accepts multiple
CFBD CSV/JSON files or a ZIP, then reports recognized files, rejected records,
conflicting duplicates and season coverage. Exported checked datasets can be
restored through the same uploader. Importing raw data does not immediately
activate a trained model; build, fit, review, then apply it.

Download or fetch these for 2016–2025:

- `/games`: scores, IDs, season/week, teams, actual kickoff and neutral venue.
- `/games/teams`: both teams’ box scores, requested by year and regular-season
  week. Retain passing attempts/completion-attempts, rushing attempts, yards,
  turnovers and team names.
- `/stats/game/advanced`: game-level offensive/defensive plays, successRate and
  explosiveness. In the exporter this is **Returns advanced team statistics
  aggregated by game**. Whole-season aggregates are not accepted for training.
- `/lines`: optional historical provider spreads/totals, verified with the
  formatted team spread. This endpoint does not certify decision/closing time.

Direct JSON retains nested `teams`, `stats`, `offense`, `defense` and `lines`.
CSV fields can be flattened (`offense.successRate`) or contain valid JSON.
Exports serializing objects as `[object Object]` have lost data: re-export as
JSON or use the built-in API download. Field joins use game ID and exact team
name; ambiguous/conflicting data is withheld rather than guessed.

Sources: [Games](https://api.collegefootballdata.com/api/games),
[Stats](https://api.collegefootballdata.com/api/stats),
[Betting](https://api.collegefootballdata.com/api/betting),
[exporter](https://collegefootballdata.com/exporter).

## API key and free access

A key authorizes data access. Entering it in the current-slate refresh does not
train the historical model. In the training tab open **Fetch history with my
CFBD API key**, select seasons and fetch explicitly. The app requests one
regular-season schedule, one advanced-game response, AP polls, optional lines and one
box-score response for each discovered week. A provider failure stops the loop
and preserves fetched files. Export your checked history as a durable backup.

[Current free access](https://collegefootballdata.com/api-tiers), checked October
10, 2026, includes these core historical categories with 1,000 calls/month;
academic `.edu` registration has 3,000 calls/month. Data availability and access
may change. The cache conserves requests but is not permanent cloud storage.

## Feature chronology

For target week W in a season:

1. Fix the feature cutoff at that week’s earliest scheduled regular-season game.
2. Include only completed same-season games in weeks below W and before cutoff.
3. Require at least two prior games and complete prior box/advanced records for
   both target teams. Missing data excludes that game from training.
4. Aggregate offense yards/plays, opponent allowed yards/plays, success rates
   weighted by offensive/defensive plays, explosiveness weighted by successful
   plays, average offensive plays and net takeaways/giveaways.
5. Fit Colley and Massey from eligible prior results. Never use target-game or
   same-week scores. Withhold cross-component Massey comparisons.
6. Attach target final margin/total only as labels. Optional historical line
   medians are comparison benchmarks, never prediction features.

Reconstructed features can use later provider revisions to earlier game data.
They are not claimed to be archived pregame observations. Actual publication
history would need immutable source snapshots and a database. Do not fabricate
an earlier forecast timestamp to pass the separate archive calibration checks.

## Ratings

Colley solves `Cr=b`, with `Cii=2+games_i`, `Cij=−meetings_ij`, and
`bi=1+(wins_i−losses_i)/2`. Ties contribute no net win/loss change. It does not
use victory margin and its rating is not a spread or a calibrated win probability.

Massey solves least squares for `Xr=y`, where a row has +1 for the visitor and
−1 for home; `y=away_score−home_score+home_field` for non-neutral games. Minimum
norm identifies a zero-mean rating within each connected schedule component.
Cross-component point predictions are withheld. Massey is the default CFB
point-rating method. Optional ridge ratings solve `(XᵀX+λI)r=Xᵀy` instead.

[Colley’s original paper](https://www.colleyrankings.com/matrate.pdf) and
[Massey’s theory](https://masseyratings.com/theory/ls.htm) describe the methods.

## Gradient boosting

This release uses `GradientBoostingRegressor`, not the XGBoost package.
Fifteen features cover matched YPP/success/explosiveness gaps and sums, each
team’s pace, turnover gap, core neutral margin/total, Massey and Colley gaps,
and each team’s historical sample size. Missing rating features use training-only
medians; core-stat gaps are not silently imputed.

Two models learn corrections to the original statistical core, one for margin
and one for total. Four candidates use depth 2/3 and 80/160 trees, with learning
rate .04, minimum leaf size 20 and seed 7. Training sets fit the trees and missing
rating medians. Validation RMSE chooses candidates and determines whether each
correction is enabled. Correction caps are fixed at ±10 margin and ±15 total
points before model evaluation. The test set chooses none of these settings.

Default periods:

| Period | Seasons | Purpose |
| --- | --- | --- |
| Training | 2016–2022 when imported | Fit trees and imputation |
| Validation | 2023–2024 | Choose candidates and estimate residual uncertainty |
| Test | 2025 onward | Report final performance after selection |

At least 200/100/100 eligible games are required. If later test results lead to
new design choices, that test season becomes development data; reserve another
untouched future period. Repeat rolling-season evaluation before making strong
claims about stability across seasons.

The trained correction is applied to the CFB score core only. Historical FEI,
roster, injury, ten-year ATS, head-to-head, weather/rest and discretionary context
features are not trained by these four exports. Their live effects remain
separate assumptions. Applying optional modifications after fitting can change
full-model performance and probability calibration.

The artifact stores numeric trees in checked JSON, not an executable pickle.
Live inference matches the fitted estimator’s tree rules. A core-settings hash
prevents applying corrections after the scoring core or point-rating method
changes. The adjustable blend changes only the CFB correction. Export the full
snapshot to retain settings and the trained model together.

Official estimator reference:
<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html>.

## Calibration and CLV

Reports show margin/total MAE, RMSE and bias, and a same-covered-games comparison
with historical reference lines. Normal win/push/loss probabilities use residual
standard deviation from validation, not final test. The held-out test report
includes three-outcome Brier score, log loss and probability reliability bins.
This estimates and evaluates uncertainty; it is not a flexible learned
probability-calibration curve. Scoring clusters at 3 and 7 remain a limitation.

Historical medians of multiple provider lines can fall on quarter points (for
example, the median of 3 and 3.5 is 3.25). Probability checks exclude these
non-whole/half-point references and report coverage counts. Those games still
enter fitting, and their finite medians remain in point-error comparisons.
References are not rounded into invented quotes; live quote validation is unchanged.

After downloading/importing CFB history, train once and apply a qualifying fit.
When private storage is connected, checked history, the applied fit, training
results and controls save automatically. Download **Setup & Saved Data →
Complete workspace backup** as a recovery copy. On later visits, restore is
automatic; no historical training fetch or refit is required. For older exports,
restore the snapshot and checked-history JSON once in Setup. See
[personal_cloud_setup.md](personal_cloud_setup.md). The weekly slate refresh
separately loads current inputs and reuses cached historical ATS/head-to-head
schedules/lines. A disabled Apply button is explained by validation/settings;
fetching the same history again does not resolve that condition.

**Calibration & CLV** also supports genuinely timestamped pregame archives.
At least 80 distinct finished games with the same settings/model signature are
required for scale/intercept calibration, using older 75% and later 25% games.
Changing a fitted boosting model changes that signature. Calibration and its
subsequent optional modifiers still need evaluation on future unseen games.

Capture top-two pregame quotes and import the same book/side/market’s closing
snapshot in the final hour before kickoff. Use actual quote publication times.
Spread CLV is recorded handicap minus closing handicap; over reverses the total
subtraction and under retains it. A closing snapshot is an observed quote, not
necessarily the exact last quote. Positive CLV alone does not prove profitability.
Closing information never enters the predictive training features.

## NFL historical training

Open **Setup & Saved Data → NFL historical data, training and trained-model restore**. Download selected years for free, upload
`games.csv` and `play_by_play_YEAR.csv.gz`, or load the included 2016–2025 compact
dataset. This is separate from the CFBD importer and uses NFL-specific definitions.
The same tab includes a reference fit and its full validation/test report.

Eligible run/pass plays exclude no-play penalties, kneels and spikes. Game-level
sums reconstruct offensive and defensive EPA, success, explosive-pass rates,
dropback and designed-rush EPA, CPOE, sacks and turnovers. Only completed previous
weeks enter each row. The entire forecast week and its own results are withheld.
Unknown kickoff/venue and missing prior core statistics withhold forecasts.

The NFL model uses its own feature list, settings signature and `nfl-boost-v1`
safe JSON format. Margin/total corrections are capped at ±8/±12 points.
The validation and test splits remain 2023 and 2025 by default. Historical closing
lines are evaluation-only; actual current sportsbook odds still need the Odds API.
Manual injuries, weather, ATS, matchup history and discretionary context remain
separate adjustments. Export the history and model to restore them after restarts.

## College preferences and ATS/total trends

The initial college profile raises offensive influence and applies ranked-opponent
letdown/look-ahead penalties. Controls explain the exact score and selection effects.
The AP detector uses conservative prior-week polls and respects supplied publication
timestamps; current rankings are not retroactively assigned to old opponents.

Both slate tabs contain an ATS/total trend finder. It screens predefined situations
from historical games and lines. The older 70% choose the direction and newer 30%
evaluate it. It reports coverage, counts, rates and illustrative returns at your
selected assumed odds, with a conservative multiple-pattern confidence adjustment.
Trends are research evidence for selections and do not automatically change scores.

Free public downloads:

- [Play-by-play CSV.gz assets](https://github.com/nflverse/nflverse-data/releases/tag/pbp)
- [Schedules, results and lines CSV](https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv)
- [nflverse official data guidance](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)

Retain a complete workspace backup. Streamlit sessions alone are temporary;
private R2 or an explicitly durable directory keeps the saved workspace.
API access alone is not a trained predictive model.


## Prototype 10 saved setup

All uploads, including reviewed injury and CLV files, are in **Setup & Saved
Data**. Cloud restore includes checked histories and applied models. The NFL
ATTD section remains separate from ATS/total picks inside NFL. Free automatic
injury checks and additive availability scenarios are now included; use Setup
for their switches, coverage, freshness and optional player-value overrides.
See [personal_cloud_setup.md](personal_cloud_setup.md).
