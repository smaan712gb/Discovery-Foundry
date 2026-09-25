# Phase 1 plan: data pipeline

Goal: `foundry data build --config config/v0.yaml` produces processed Parquet, a sealed holdout and a
data report, then verifies its own output and exits non-zero on any failure.

## Owner answers (2026-09-24)

| Question | Answer |
| --- | --- |
| Export path | `C:\Projects\favio_NQ_lucid\data\historical\NQ MM-YY.Last-Minute.txt` (5 contracts, 2025 only). A 2018–2025 export will follow. |
| Timezone | UTC with bar-end timestamps. Confirmed from the data, not assumed (ADR 0002). |
| Bar size | 1 minute. Tick files exist but are out of scope for v0. |
| MNQ | Yes: same price series, contract specs selected by config. |

## Steps

1. **Config models** (`foundry/core/config.py`, `foundry/domains/nq/config.py`): Pydantic, `extra=forbid`,
   canonical hashing.
2. **Loaders** (`domains/nq/loaders.py`): NinjaTrader text format (`yyyyMMdd HHmmss;o;h;l;c;v`) plus
   generic CSV/Parquet with a column map. Each source declares its timezone and whether timestamps mark
   bar start or bar end. Output: bar-start timestamps in America/New_York.
3. **Calendar and sessions** (`calendar.py`, `sessions.py`): trading date, session label
   (RTH / ETH / POST / HOLIDAY / OUTSIDE), early closes and holidays from `config/calendar_cme_equity.yaml`.
4. **Quality checks** (`quality.py`): duplicates, gaps, zero volume, bad ticks (ATR multiple), OHLC
   consistency, off-grid prices, bars outside sessions, session-alignment check (catches a wrong
   timezone). Everything is reported and flagged, never filled.
5. **Continuous contract** (`rolls.py`): volume-crossover roll rule, effective the trading day after the
   crossover; additive adjustment anchored to the last pre-holdout contract (ADR 0003). Roll table
   stored; continuity verified.
6. **Splits and holdout** (`splits.py`, `core/holdout.py`): search / validation to `data/processed/`,
   holdout to `data/holdout_sealed/` with a SHA-256 manifest and access log.
7. **Build and report** (`build.py`, `report.py`): orchestrate, write `build_manifest.json` (seed,
   git, config/data/output hashes, versions) and `data_report.md`, then self-verify.
8. **CLI** (`foundry/cli.py`): `foundry data build`.

## Tests

DST transitions (UTC sources and ET-local sources, including the ambiguous fall-back hour), session
boundaries, holidays and early closes, roll timing and no-jump continuity (hypothesis), no-overlap roll
failure, quality detectors, holdout seal/tamper/once-only/access log, a static check that nothing else
touches the holdout directory, and an end-to-end synthetic build that is byte-for-byte reproducible.

## Acceptance

- The synthetic end-to-end build passes and verifies.
- On the owner's 2025 files the build runs and reports honestly. Under the default splits all 2025 data
  is holdout, so the build must fail its "search split non-empty" check until the 2018–2024 export
  arrives. That failure is correct behaviour, not a bug.
