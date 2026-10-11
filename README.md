# Football Model Lab — Prototype 11

A personal Streamlit NFL/college football workspace. The existing scoring and
trained-model architecture is retained, with separate league controls.
Python 3.11 or 3.12 is recommended.

## Update your app

1. Extract `football_dashboard_prototype11.zip`.
2. Open the GitHub folder containing your deployed `app.py`.
3. Upload **every file inside the extracted `football_dashboard` folder**, replacing
   existing files. Include `requirements.txt` and all new Python modules. Upload
   the contents, rather than the ZIP or another nested `football_dashboard` folder.
4. Commit to the branch Streamlit uses. Allow dependencies to install, then
   use **Manage app → Reboot app** if needed.
5. Confirm the app displays **Prototype 11**.

This ZIP updates your deployment when you upload it. It does not itself change
GitHub, your live app, your provider accounts or your cloud storage.

## Startup fix for existing Prototype 10 apps

The live ATTD injury join previously scanned every roster repeatedly for every
report and scorer. This could delay the later tabs for minutes after host keys
activated full player and injury feeds. Injury names are now resolved once;
ambiguous names, report age and prior-active-game rules retain their guards.

Automatic and manual slate downloads now run on a detached copy in a background
thread. The page shows the current feed and elapsed time while the saved inputs
remain usable. Completed downloads apply in the main app thread, preserve the
latest settings/model, and trigger a page refresh. Failed/cancelled jobs retain
saved inputs. Setup, Calibration and Method render before the optional trend
and ATTD panels. The core scoring model is unchanged.

For an existing Prototype 10 deployment, only six Python files need updating in
the existing GitHub folder containing `app.py`:

- `app.py`
- `nfl_attd.py`
- `workspace_runtime.py`
- `data_pipeline.py`
- `setup_ui.py`
- `background_work.py` (new)

Commit the upload, then reboot the same Streamlit app. Keep its current Secrets
and the same private R2 bucket. Existing workspace JSON and saved history remain
compatible. Uploading the whole package is also supported.

## What changed

- **Setup & Saved Data** is the third main tab. It contains a readiness table
  and **all file uploaders**: snapshots, complete backups, histories, trained
  models, NFL provider inputs, CFB FEI/roster inputs, injury reports, player
  values, CLV ledgers/closing quotes and calibration archives. Each input has
  a loaded/missing status, purpose and next action. Optional inputs can stay absent.
- **Private save/restore** through Cloudflare R2, or an explicitly configured
  durable directory. It saves both raw checked histories, applied models,
  unapplied training results, parameters, NFL switches, player inputs, injuries,
  manual overrides, calibration data and CLV records. Keys stay in host Secrets.
  Complete workspace JSON exports provide an independent backup.
- **Automatic upcoming-slate refresh**, on visits and while the browser is
  open, with an editable interval (60 minutes initially). It uses the next seven
  days in America/New_York. It loads current statistics/odds and reuses saved
  trained models; it never automatically repeats the multi-season training
  download or refits a model. Past-season provider downloads use a 90-day cache
  in private storage so host restarts do not erase that cache.
- **Free NFL and CFB injury checks**, enabled by default, normally hourly.
  NFL uses ESPN's public injury listing, with Covers as a fallback; CFB uses
  Covers. Reports retain publication date precision, source and observation
  time. Missing, old, unmatched and post-kickoff reports do not become healthy
  coverage. Listings are partial and their schemas may change.
- **Additive injury score scenarios** and availability-aware NFL touchdown
  rankings. NFL score impacts require recent actual snap coverage or an explicit
  player-value/baseline override, reducing double counting of existing absences.
  CFB uses a conservative role proxy and skips continuing long-term absences.
  Uncertain status probabilities/player values are unfitted assumptions,
  adjustable in Setup; score caps initially are 4 NFL / 1.5 CFB points per team.
- The included compact **2016–2025 NFL history and qualifying reference fit**
  preload on a new workspace. An existing saved workspace takes precedence.
  CFB needs your own checked history and qualifying fit once; those were not
  supplied in this deployment package.
- The trained-model column reads **Active / Inactive** as text.

Start with [personal_cloud_setup.md](personal_cloud_setup.md) for the one-time
connection and exact restore steps. Your authenticated R2 account must still be
connected and checked in the app; development did not use your credentials.

## NFL features retained

All NFL add-on layers remain on by default and can be switched off in NFL:
2.5× passing/rushing EPA weighting, smoothed success, dated opponent-adjusted
DVOA, non-blitz pressure, line yards, 15+ MPH sustained wind, dynamic rest,
confirmed bye bonus, divisional margin compression/lower total, and an underdog
ranking preference. Official DVOA/red-zone DVOA/adjusted line yards remain
optional dated provider imports. Automatic line yards are an unadjusted PBP
proxy; pressure needs actual pressure and blitz tags.

**Anytime Touchdown Scorers stays inside NFL**, below ATS and Total Picks.
It uses a market total of at least 47.5, high-volume RB1/2, WR1/2 and TE1,
red-zone opportunity shares, a labeled xTD proxy/actual TD gap, optional opposing
red-zone DVOA, and the last actual active game for bounce-back/sell-high rules.
Byes and inactive games are skipped, including after a multi-TD game.
Current out/doubtful players are excluded; questionable status reduces the
unconditional game scoring scenario. Price EV is conditional on participation;
the export also reports availability-weighted EV assuming inactive bets are void.
Book settlement rules must be checked separately.

See [nfl_layers_guide.md](nfl_layers_guide.md) for callable pandas/numpy wrappers,
feature definitions, data schemas and bye-skipping logic. These add-on/injury
weights have not been validated by the included core reference fit.

## College profile and training

The saved college profile retains 65% offense / 35% opposing defense, full
efficiency strength, 40% FEI blend, ranked-win letdown −2 margin points,
ranked look-ahead −2, and a combined context cap of 3. Its selection setting can
prioritize opponents, avoid these spots or rank all eligible games by EV.
Preferences remain separate from learned coefficients and never alter NFL.

CFB and NFL train separately on prior-week statistics, with chronological
training, validation and untouched test periods. Missing first-week/incomplete
histories are withheld. Quarter-point multi-book reference medians stay in
point-error training comparisons; probability checks exclude them without
aborting training. Actual bookmaker quotes still require whole/half-point lines.

Training is in **Setup & Saved Data → CFB/NFL historical data, training and
trained-model restore**. A trained model alone is sufficient to reuse its
learned correction. Checked history is separate and supports retraining/trends.
See [training_guide.md](training_guide.md).

## Included NFL evaluation

The compact regular-season dataset has 2,639 games. Prior-week feature building
retains 2,290 eligible rows: 1,570 training (2016–2022), 480 validation
(2023–2024), and 240 test games (2025).

| Target | Validation baseline RMSE | Validation selected RMSE | 2025 selected RMSE | 2025 closing-market RMSE |
| --- | ---: | ---: | ---: | ---: |
| Away winning margin | 13.50 | 13.50 | 13.36 | 12.57 |
| Combined score | 13.88 | 13.42 | 13.50 | 12.97 |

The total correction qualified; the spread correction stayed disabled. Both
2025 model errors were larger than the closing-market benchmark. Historical
provider revisions and reference lines are retrospective; this does not
establish profitable performance for the modified live model.

Both slate tabs retain the ATS/total trend finder with fixed screens, earlier
selection/later evaluation, coverage, sample sizes and an adjustment for testing
multiple patterns. Trends inform selection and do not automatically add points.
Historical trend returns use an assumed price, not archived executable odds.

## Run and verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows activation: `.venv\Scripts\Activate.ps1`. For a local durable workspace,
set `MODEL_DATA_DIR` to a directory you keep. A hosted ephemeral directory alone
is not persistent storage. Streamlit theme settings are in `.streamlit/config.toml`.

```bash
python -m unittest discover -s tests -v
```

152 automated checks passed, including the existing score/price/chronology pipeline, free-feed parsing and
expiry, incremental injury baselines, touchdown bye/availability logic, complete
backup restoration, conditional cloud saves, corrupted-save protection and
provider failure preservation. Public injury pages and current NFL snap sheets
were checked live. R2 protocol checks use an S3-compatible test double; the actual
bucket connection requires your one-time setup. The app's frozen example remains
available when no live keys are configured.
