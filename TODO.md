# TODO

Each item says why it is deferred and what "done" means.

## Blocking phase 1 acceptance on real data

- **Owner: export NQ 1-minute per-contract files from NinjaTrader for 2025-01-01 to today** (ADR 0006).
  Why: the files on hand end 2025-12-26, so validation (Jan–Apr 2026) and holdout (May 2026 onward) are
  empty. The 2025 files also have a 2025-06-20 to 2025-07-28 hole: NQU25 must be re-exported from about
  2025-06-10 so the M25 → U25 roll can be measured. Each contract file must overlap the next one by at
  least the roll week. Done: the export is added as a `sources` entry (with its own `timezone` and
  `timestamp_label`) and `foundry data build` exits 0.

## Data (phase 1 follow-ups)

- **Verify the hand-entered calendar** (`config/calendar_cme_equity.yaml`), especially entries marked
  "verify": 2018-12-05, the 2021 and 2023 Good Fridays, 2022-12-26, 2023-01-02, 2025-01-09 and
  2026-04-03. Why: typed in from memory of NYSE and CME schedules, with no primary source to hand.
  Done: every entry checked against a CME holiday notice, and on the full export the report shows no
  `outside_session` bars on those dates that the calendar can't explain.
- **Calibrate `bad_tick_atr_multiple`** (20 × ATR(60) for now). Why: needs real data to see how many
  flags are real bad ticks and how many are news bars. Done: flag counts on 2018–2024 reviewed, and the
  threshold and a sample of flagged bars recorded in an ADR.
- **Initial git commit.** Why: the owner hasn't asked for commits yet, so every build records
  `git.dirty = true` and `commit = null`. Done: a commit exists, and build manifests carry its hash.
- **ET-local sources: fall-back hour.** Ambiguous wall times are resolved by order of appearance, so an
  exact duplicate row inside that hour would be read as the second (EST) occurrence. Why: it only matters
  for ET-stamped exports with duplicates in the 01:00 hour on the first Sunday of November, when the
  market is closed. Done: close it if the owner's new export is UTC, or add a check that the hour holds
  exactly two ordered copies.
- **16:00–17:00 ET (`POST`) is not tradeable by the DSL.** Why: the spec defines ETH as 18:00–09:30 only.
  Done: the owner confirms, or `POST` is added as a DSL session option.

## Evaluator (phase 2 follow-ups)

- **CPI and payrolls dates in `config/events.yaml`.** Why: only FOMC dates are listed. 2025 CPI and
  payrolls releases moved around the government shutdown, and I won't type them in without a primary
  source. Done: every 2025–2026 CPI and payrolls release date added from BLS schedules, with the source
  noted in the file.
- **Calibrate MNQ commission.** Why: $1.50 per round turn is a placeholder until the owner confirms their
  broker's rate. Done: `config/v0.yaml` holds the owner's actual all-in rate.
- **Profile a real-data evaluation.** Why: synthetic runs take about 10 ms after JIT, but real data has
  about 1.6 times as many bars and the search will call the evaluator thousands of times. Done: the
  per-evaluation time on the real search split is measured and recorded in the phase 4 plan.

## Critics (phase 3 follow-ups)

- **White's Reality Check / Hansen's SPA across the surviving set.** Why deferred: design.md lists it as
  optional, and it only means something once a search run leaves several survivors. Done: a bootstrap
  SPA p-value for the survivor set, added to the report and tested on planted-edge vs noise families.
- **Calibrate critic thresholds against real data.** Why: the defaults come straight from design.md
  (DSR 0.95, PBO 0.2, CPCV 60%). With about 16 months of history (ADR 0006) they may be very harsh. Done:
  the owner has reviewed the funnel from the first real search run. Any threshold change needs a new
  ADR, and never happens as part of the meta layer.

## Search (phase 4 follow-ups)

- **Remove the revoked DeepSeek key from the Windows user environment.** Why: it overrides the working
  key in `.env`, so `foundry search` would fail authentication. Done:
  `[Environment]::SetEnvironmentVariable('DEEPSEEK_API_KEY', $null, 'User')`, then restart the editor.
- **Tune engine v0 on real data.** Why: batch size, mix and surrogate settings were set by judgment
  and checked only on synthetic data. Done: the phase 5 meta layer compares v0 against v1 over
  several seeds.

## Meta layer (phase 5 follow-ups)

- **First real tournament.** Why: it needs the real data build, and it spends validation (each
  tournament uses a validation fold, ADR 0010). Done: the owner approves the cost (up to $3.25 LLM
  spend and about 3 hours with the defaults), and `foundry meta status` shows the result.

## Platforms and data (ADR 0006)

- **Exporters to NinjaScript and Pine Script v6**, plus `foundry parity`. Why: survivors must run on both
  platforms, and generated code is trustworthy only when the platform's own backtest matches ours. Done:
  deterministic templates for every DSL construct, and a parity test on the baseline ORB strategy that
  matches trade by trade against a strategy-tester export from each platform.
- **TradingView as a cross-check source.** Why: TradingView usually exports a continuous `NQ1!` whose
  back-adjustment differs from ours. Done: a `contract_from: continuous` source in config, and a report
  section comparing its daily returns with ours.
- **Level 2 data (NinjaTrader).** Why deferred: the v0 simulator is bar-based and the DSL has no
  order-book features. Done: an ADR on which L2 features (for example top-of-book imbalance) would be
  worth their data cost, with a separate tick/L2 loader and backtester. Not before v0 is done.

## Later phases (not started; phases 2-5 are done, see docs/plans/)

- Phase 6: HTML reports and `foundry run`.
