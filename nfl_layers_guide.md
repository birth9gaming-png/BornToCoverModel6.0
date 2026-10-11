# NFL add-on layers — Prototype 10

All requested layers start **on**. Their controls live in the NFL slate tab.
**Anytime Touchdown Scorers** is its own section below the ATS/total slate, with
separate rules, rankings and downloads. College Football behavior is unchanged.

`model.py`, its score architecture and its trained feature definitions remain
unchanged. `nfl_layers.project` calls the original projector, applies NFL-only
adjustments, reconciles both scores, and recalculates quoted probabilities/EV.
The wrapper returns the core result directly for CFB.

These new coefficients express the requested strategy. The included NFL fit
validated its original features and total correction; it **has not validated
these add-on assumptions or ATTD probabilities**. The toggles let you compare
the original forecast with the requested profile.

## Feature engineering

`nfl_features_ext.engineer_efficiency(pbp, through_week)` uses completed prior
weeks, excludes no-play/kneel/spike events, and separates QB dropbacks from
designed runs. Sacks and scrambles belong to dropbacks. Success means positive
EPA; 100 reference-rate plays shrink small success-rate samples.

The weighted NFL efficiency signal is:

```python
weighted_epa = (2.5 * matched_pass_epa + matched_rush_epa) / 3.5
# matched means own offense and the opposing defense's allowed EPA, averaged.
epa_score = baseline_points + reference_plays * weighted_epa
success_score = baseline_points + 25 * (smoothed_success - reference_success)
target = 0.75 * epa_score + 0.25 * success_score
score_adjustment = 0.25 * (target - existing_efficiency_score)
```

Pace also scales the target. The final efficiency change is capped at 5 points
per team. Existing direct EPA contribution is included in
`existing_efficiency_score`, so the wrapper blends toward the new signal rather
than adding the full EPA forecast a second time. Your learned correction stays.

Dated opponent-adjusted offensive and defensive DVOA modify the scoring
baseline: `factor = 1 + (own_offensive_dvoa + opposing_defensive_dvoa) / 200`.
The weight is 0.35 through week 8, declines linearly, and is 0.15 by week 13.
Defensive DVOA keeps its provider sign: a negative opposing value lowers the
offense's baseline. Each change is capped at 3 points. A DVOA snapshot must have
the forecast season, an eligible through-week, and a publication timestamp.

Non-blitz pressure requires actual `was_pressure` and zero `n_blitzers` (or
explicit `is_blitz=0`) tags. The denominator includes only charted, non-blitz
dropbacks. Sacks/hits alone do not identify non-blitz pressure. A defense above
the 30% reference pressure rate suppresses its opponent's passing component.

Provider **Adjusted Line Yards** is used when supplied. Otherwise the automatic
run-blocking fallback is labeled **unadjusted PBP line-yards proxy**: negative
yards × 1.2; yards 0–4 fully; yards 5–10 at half weight; yards above 10 add
nothing. This proxy omits official situation/opponent/formation adjustments.
Pressure plus run-blocking changes are capped at 3 points per team.

## 1. Division and rest variables

```python
import pandas as pd
from nfl_features_ext import engineer_game_context

games = pd.DataFrame([
    dict(id="forecast", season=2026, week=6, away="BUF", home="NE",
         kickoff="2026-10-11T17:00:00Z", away_rest=7, home_rest=14)
])
# full_schedule must include each team's entire regular-season schedule.
features = engineer_game_context(games, full_schedule)
# features: divisional_matchup, away_division, home_division,
# away_rest_days, home_rest_days, rest_differential_home,
# away_off_bye, home_off_bye, and explanations of bye status.
```

The signed differential is **home rest minus away rest**. When supplied rest
days are missing, the function subtracts the local dates of the teams' previous
games. A 14-day interval alone does not prove an official bye. An automatic bye
bonus requires a complete team regular-season schedule and no scheduled game
in the immediately previous league week. Explicit Boolean `away_off_bye` or
`home_off_bye` flags can supply verified provider status.

The automatic home margin effect is `clip(0.12 * differential, -1.5, 1.5)` plus
0.6 for a confirmed home bye, minus 0.6 for an away bye. This replaces the old
optional direct rest-days term; manual `rest_margin_home` remains a separate
scenario. Venue/home-field logic is unchanged.

Division identity uses canonical franchise codes and the eight divisions from
the 2002 realignment onward. `LAR/STL→LA`, `OAK→LV`, `SD→LAC` and other known
aliases are normalized. Older seasons are not silently assigned current
division membership.

## 2. Final division and weather overrides

```python
from nfl_layers import apply_divisional_modifiers

predictions = pd.DataFrame([
    dict(margin_away=7.0, fair_total=49.0, divisional_matchup=True),
    dict(margin_away=-7.0, fair_total=49.0, divisional_matchup=True),
    dict(margin_away=7.0, fair_total=49.0, divisional_matchup=False),
])
adjusted = apply_divisional_modifiers(
    predictions, margin_shrink=0.12, total_drop=1.5
)
# Divisional: margins become +6.16/-6.16; totals become 47.5.
# Non-divisional: margin +7.0 and total 49.0 remain.
# fair_line_away = -margin_away; a positive margin means the away team wins.
```

The full wrapper applies compression after the other NFL adjustments. It
reconstructs score changes with `away = (margin_change + total_change)/2` and
`home = (total_change - margin_change)/2` so the scores, spread and total agree.
It then reruns the existing market evaluator using unchanged uncertainty.

Divisional underdogs receive a 0.01 ranking bonus; a home underdog of +0.5 to
+3.5 gets another 0.015. The ranking score is EV plus this preference. The
probability-derived EV is still displayed separately. Positive EV is still
required by the top-picks filter; the preference cannot turn a negative EV into
a qualifying selection. Neutral venues do not receive the home-dog bonus.

At **sustained wind ≥15 MPH**, exposed venues receive a passing-efficiency
penalty starting at 5%, increasing 1 percentage point per additional MPH,
capped at 30%. Positive passing EPA is reduced proportionally; already
negative passing EPA becomes less efficient rather than moving toward zero.
The total falls 1.5 points at 15 MPH plus 0.3 per additional MPH, capped at 8.
An existing manual weather-total change is replaced when this automatic wind
constraint triggers, avoiding two applications of the same weather effect.

Only a dated forecast published before the forecast cutoff/kickoff is eligible.
The automatic refresh uses hourly Open-Meteo sustained 10m wind, in MPH, near
kickoff at the approximate home-venue city. Gusts and recorded game weather are
not substituted. Dome/closed venues skip wind. Unknown roof exposure and
neutral/international venues require a specific forecast import.

## 3. Standalone ATTD scorer, including bye skipping

```python
from nfl_layers import DEFAULTS
from nfl_attd import last_active_games, rate_anytime_touchdowns

player_games = pd.DataFrame([
    dict(player_id="rb1", game_id="week4", week=4,
         kickoff="2026-09-27T17:00:00Z", active=True, completed=True,
         touchdowns=2, carries=16, targets=3, xTD=0.7),
    dict(player_id="rb1", game_id="bye5", week=5,
         kickoff="2026-10-04T17:00:00Z", active=False, completed=False,
         touchdowns=0, carries=0, targets=0, xTD=0.0),
])
as_of = "2026-10-10T18:00:00Z"
last = last_active_games(player_games, as_of)
assert last.iloc[0]["week"] == 4
assert last.iloc[0]["touchdowns"] == 2  # Sell-high rule, NOT bounce-back.

players = pd.DataFrame([
    dict(player_id="rb1", name="Example player", team="BUF", position="RB",
         depth_rank=1, projected_opportunities=19, recency_verified=True)
])
games = pd.DataFrame([
    dict(id="week6", away="BUF", home="NE", league="NFL", total=48.5,
         kickoff="2026-10-11T17:00:00Z")
])
scorers = rate_anytime_touchdowns(
    players, player_games, games, DEFAULTS, as_of,
    opponent_metrics={"NE": {"redzone_defensive_dvoa": 10.0}},
    quotes=[dict(game_id="week6", player_id="rb1", book="Example",
                 odds=150, quoted_at=as_of)],
)
```

The exact sequence is:

1. Keep games with **market/Vegas total ≥47.5**, including the boundary.
2. Keep RB1/2, WR1/2 and TE1; exclude known out/inactive/IR/PUP/doubtful players.
   Default high-volume floors are 10 RB, 5 WR and 4 TE carries + targets/game.
   The NFL injury reports you reviewed in the app also override roster status
   when they are current, published before the cutoff, and match the player.
3. Keep known active, completed games before forecast/kickoff; sort by actual
   kickoff and take the latest active row. Byes/inactive weeks never enter that
   lookup. A zero-touch game with actual snaps **does** enter it.
4. Use supplied projected xTD, or mean xTD over the latest four active games.
   Start with `P(TD>=1) = 1 - exp(-xTD)`. Automatic xTD is a labeled, smoothed
   league frequency proxy by rush/target opportunity and field-position bucket.
5. Add `0.25 * max((window_xTD - actual_TDs)/games, 0)` to the expected-TD
   intensity. Apply modest red-zone share and opponent red-zone DVOA modifiers.
   RBs use touch share; WRs/TEs use target share. Positive defensive red-zone
   DVOA means a weaker opposing defense and raises the intensity.
6. Last active game **0 TD**: add 0.03 probability. Last active game **2+ TD**:
   multiply probability by 0.88. These rules express the requested strategy;
   they do not establish that the player is statistically due for a TD.
7. Bound probabilities to 0–0.95. Compute EV only with a valid American-odds
   ATTD quote, timestamped before the cutoff and no older than six hours.

Passing TDs are never counted as scorer TDs. Automatic logs use `td_player_id`
and actual snaps. This includes zero-touch active games and can include return
TDs. Incomplete active-game coverage skips the automatic recency adjustment.
Provider logs must mark actual participation explicitly; an unknown game must
not be filled with a zero-TD active row.

Automatic role ranks come from recent usage, not a verified official depth
chart. Import `depth_rank` and `projected_opportunities` to override that
ranking. If supplied xTD is missing, no invented scorer probability appears.

## Hook the wrapper into an existing pipeline

```python
# Previous: from model import run, project
from nfl_layers import run, project

# The original model is still called internally; CFB returns unchanged.
results = run(snapshot)
one_game = project(snapshot, snapshot["games"][0])
```

The Streamlit app already makes this replacement. The historical trainer still
uses the unchanged core projector and original trained inputs. Layer settings
are exported, recorded in NFL CLV logs, and included in calibration signatures.

## Data contracts and availability

All `player_id` and `game_id` values must match the forecast/player stream.
Timestamps require `Z`, a UTC offset, or the app's supported New York suffix.
Rates are fractions (`0.30`), while DVOA is percentage points (`15.0` = +15%).
`xTD` is expected scored TDs, not expected passing TDs.

Import provider data in **NFL → NFL data coverage and optional provider
inputs**. The downloadable `nfl_layer_inputs_example.json` has empty input
arrays and inert examples, so it will never load fabricated statistics.
Supported arrays: `team_metrics`, `weather`, `players`, `player_games`, and
`attd_quotes`. Each imported row is checked before it can change the forecast.
Imports merge by player/game/provider snapshot identity and survive refresh.

Refresh automatically tries rosters/snaps, pressure/blitz tags, wind forecasts,
and ATTD quotes. ATTD quotes use your existing Odds API key and one additional
event request per qualifying upcoming game. Matching requires a unique player
full name across the two teams; ambiguous names are skipped.

Official DVOA, red-zone DVOA and ALY require an eligible provider import when
the connected feeds do not supply them. No paywall is bypassed and no substitute
is silently labeled DVOA. Missing inputs are visible in the NFL breakdown;
the remaining available layers continue to work.

Source definitions:

- [nflfastR EPA and field descriptions](https://www.nflfastr.com/reference/field_descriptions.html)
- [nflverse participation pressure tags](https://nflreadr.nflverse.com/articles/dictionary_participation.html)
- [FTN charting blitz tags](https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html)
- [FTN DVOA](https://ftnfantasy.com/nfl/dvoa-explainer)
- [FTN adjusted line yards](https://ftnfantasy.com/dvoa/nfl/adjusted-line-yards)
- [Open-Meteo forecast API](https://open-meteo.com/en/docs)
- [Odds API anytime-TD market](https://the-odds-api.com/sports-odds-data/betting-markets.html)


## Prototype 10 saved setup

All uploads, including reviewed injury and CLV files, are in **Setup & Saved
Data**. Cloud restore includes checked histories and applied models. The NFL
ATTD section remains separate from ATS/total picks inside NFL. Free automatic
injury checks and additive availability scenarios are now included; use Setup
for their switches, coverage, freshness and optional player-value overrides.
See [personal_cloud_setup.md](personal_cloud_setup.md).
