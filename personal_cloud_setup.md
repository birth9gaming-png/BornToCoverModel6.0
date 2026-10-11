# Your personal saved football app — one-time setup

Use **Streamlit Community Cloud + a private Standard Cloudflare R2 bucket**.
Keep using your existing app and model code. Streamlit stores the keys; R2 stores
history, models and settings. After setup, open the app and let it load upcoming
projections. You do not have to upload your checked seasons or fit every week.

## 1. Update the existing app

Before replacing an older build, download its **Export checked historical
dataset** and, after applying your trained fit, **Export model inputs and settings**.
Keep these files on your computer. A snapshot alone does not contain checked raw
CFB history, although it can contain the applied trained CFB model.

Extract `football_dashboard_prototype11.zip`. In your GitHub repository, replace
all files in the folder containing the deployed `app.py` with the contents of the
extracted `football_dashboard` folder. Include all new `.py` modules and
`requirements.txt`. Keep the same Streamlit entrypoint and branch.

Wait for the updated dependencies to install, then reboot the app if needed.
The NFL/CFB version tiles should say **Prototype 11**. Uploading only `app.py`,
only the ZIP or an extra nested folder does not complete the update.

## 2. Make it personal

In Streamlit **app Settings → Sharing**, set **Only specific people can view this
app**. Keep access limited to your own developer/account login. Check that you
have not granted viewers access. Community Cloud supports one private app.

You can keep a private GitHub repository, too. Keys go in Secrets, never in
GitHub. Do this privacy step before connecting your personal saved workspace.

Official instructions: [Streamlit sharing](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).

## 3. Create the private saved-data bucket

1. Sign into or create a Cloudflare account and open **R2 Object Storage**.
2. Activate R2 if prompted. Cloudflare may request billing details even though
   usage within the free allowance costs $0.
3. Create a bucket named `football-model`. Select **Standard** storage and the
   default jurisdiction/location selection. Leave public access disabled. No
   public `r2.dev` URL or custom domain is needed.
4. On the R2 overview, under **Account Details → API Tokens**, click **Manage**.
5. Create an R2 API token with **Object Read & Write** permission, scoped to this
   bucket. Use a name such as `personal-football-app`.
6. Copy the **Access Key ID** and **Secret Access Key**. The secret is displayed
   only at creation. Also copy your 32-character **Cloudflare Account ID**.
   The API token value itself is not the app's `R2_SECRET_ACCESS_KEY` field.

Official instructions: [R2 authentication](https://developers.cloudflare.com/r2/api/tokens/).

## 4. Store the keys once

Open your Streamlit app's **Settings → Secrets**. Add this TOML, replacing the
placeholders with your own values. Preserve unrelated existing secrets.

```toml
CFBD_API_KEY = "your-college-football-data-key"
ODDS_API_KEY = "your-odds-api-key"
R2_ACCOUNT_ID = "your-32-character-cloudflare-account-id"
R2_ACCESS_KEY_ID = "your-r2-access-key-id"
R2_SECRET_ACCESS_KEY = "your-r2-secret-access-key"
R2_BUCKET = "football-model"
```

Click **Save**, then **Reboot app**. Open **Setup & Saved Data**. The storage row
should say **Connected**, followed by **Saved automatically** once the page
finishes. Correct any displayed bucket/permission error before considering the
workspace saved. No R2 keys need to be pasted into the model UI.

Official instructions: [Streamlit Secrets settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings).

## 5. Move your existing history and trained CFB model in once

If you have an older build's two exports:

1. In **Setup & Saved Data → Import or export a snapshot**, select your
   `football_model_snapshot.json` and click **Load uploaded snapshot**. If that
   export was made after applying your CFB fit, the fit comes with it, together
   with the matching controls.
2. In the same tab's **CFB historical data, training and trained-model restore**
   section, upload `cfb_historical_dataset.json`. Click **Import and check
   historical files**. Review the seasons/coverage table. You do not need to
   click Fetch history again or train again to reuse an already applied fit.
3. If the snapshot did not contain your fit, use **Import trained model JSON →
   Load trained CFB model** in that section. A saved model requires the same
   core controls it was fitted with; restoring its original full snapshot is
   the easiest way to retain those controls. The fit report restores uncertainty.
4. If no qualifying CFB fit exists yet, download/import the seasons once,
   build and train, review validation, then click **Apply validated CFB model
   and validation uncertainty** when enabled. A fit that fails validation will
   remain inactive; downloading history repeatedly will not make it qualify.
5. Click **Save everything now**. Confirm success and download **Complete
   workspace backup** for your own copy. The complete backup includes checked
   history, unlike the older snapshot export.

For future Prototype 11 backups, **Restore complete workspace backup → Restore
complete backup** restores both history and models in one action.

A new workspace preloads the included NFL history and qualifying reference fit.
An older imported snapshot can replace that fit; the NFL history section also
has **Load included NFL fit results** and **Apply validated NFL model and
uncertainty** if you want to apply it again. Its spread correction did not
qualify, and the core reference test results do not validate the new add-on weights.

## After that

Open the app and sign in. It restores saved data before loading the bundled
example. Automatic loading uses the next seven days in New York time, checks
whether a refresh is due and reuses the saved trained models. The default
full refresh interval is 60 minutes; injury checks have their own 60-minute
interval. Both switches are already on.

While the page is open, a five-minute timer checks whether those intervals have
elapsed. When Streamlit is asleep or no browser is open, this build does not run
a separate scheduled worker. The next visit resumes checks. Community Cloud can
sleep after 12 hours without traffic; use its wake button if shown. A cold start
or a new NFL play-by-play download may take longer than loading an existing save.

The readiness table states what is loaded, absent or optional. No current
bookmaker quotes means no current price-based value. Official DVOA/red-zone
DVOA/adjusted line yards, returning starters, manual injury overrides, CLV closes
and calibration archives remain optional. Check these rows only if you want
those extra terms. A loaded fit can still be inactive after core controls change
or required live statistics are missing; each projection explains why.

Public injury listings are free, but incomplete. NFL uses ESPN with Covers as
fallback; CFB uses Covers. They need no additional key. Older/mismatched reports
are withheld. Missing reports mean unknown coverage. Estimated injury player
values and uncertain-status probabilities are assumptions, not trained effects.

## Cost and limits

- [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud): free hosting, with resource/sleep limits.
- [R2 Standard free allowance](https://developers.cloudflare.com/r2/pricing/):
  10 GB-month storage, 1 million Class A and 10 million Class B operations monthly.
  Charges apply above the allowance; it is not a hard $0 spending cap. Keep an
  eye on Cloudflare usage. Infrequent Access does not have this free allowance.
- [CFBD free plan](https://collegefootballdata.com/api-tiers): 1,000 monthly
  calls for core/historical endpoints. Existing access/quota still applies.
- [The Odds API free plan](https://the-odds-api.com/): 500 monthly credits. NFL
  ATTD requests add event-level usage. Leaving the app open for many hours with
  frequent refreshes can exhaust the free allowance. Choose a longer full
  refresh interval in Setup if you need to conserve credits; injury checking
  can retain its shorter interval.

The app does not automatically buy a plan. Quota/access failures are displayed.
Past-season requests are cached for 90 days, including across host restarts;
weekly refreshes do not repeat the multi-season training process. Stored raw
history and fitted models stay available independently of that request cache.

## If a save fails

Your current browser state remains available. Download a complete backup before
closing it. Fix host Secrets/permissions and reboot. If another browser saved
newer data, the app pauses replacement; download this session first and then
**Reload saved workspace**. A corrupt cloud workspace is also protected from
being overwritten by the example automatically.

R2 keeps the latest manifest, one previous manifest and shared history objects.
The app's complete backup is the convenient recovery file. This release was
checked with a compatible S3 test double and a durable local save/restore;
your actual authenticated R2 connection still needs the check in step 4.
