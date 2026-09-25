# ADR 0003: Continuous contract construction

- Status: accepted
- Date: 2026-09-24

## Context

Strategies run on one continuous NQ series built from per-contract files. Back-adjustment has two traps:

1. **Roll-timing lookahead.** "Roll when the next contract's daily volume exceeds the current one's" can
   only be known once that day has closed.
2. **Holdout leakage through adjustments.** Classic back-adjustment anchors on the latest contract, so
   every historical price depends on roll gaps in the most recent (holdout) period. Rebuilding with new
   data would silently change search-split prices.

## Decision

- **Roll rule** (configurable, default `volume_crossover`): on trading date d, if the next listed contract's
  volume exceeds the current contract's, the continuous series switches from the first bar of trading date
  d+1. A roll is forced when the current contract has no bars on d+1 and the next one does. Rolls only move
  forward in expiry order.
- **Adjustment** is additive in index points, since PnL is points × multiplier. The gap is measured at the
  last minute of day d at which both contracts have a bar: `gap = close_next − close_current`.
- **Anchor**: the contract that is front at the last pre-holdout bar has adjustment 0. Earlier contracts are
  back-adjusted and later (holdout-period) contracts are forward-adjusted. So search and validation prices
  never depend on holdout-period data, and the holdout segment joins the processed series with no offset.
- **Continuity check**: at every roll, the adjusted series' move across the boundary must equal the new
  contract's own raw move from the gap-measurement minute `t*` to its first bar after the switch, minus
  the old contract's own move from `t*` to its last bar before the switch (zero when `t*` is that last
  bar, which is the usual case). The build verifies this exactly (prices are on a 0.25 grid, so
  adjusted prices stay on it too) and fails otherwise.
- **No-overlap rolls** (the next contract has no bar in common with the current one on day d) have no
  measurable gap. The build records them as errors and fails. It never assumes a gap of zero.
- The processed series keeps `contract`, `raw_*` prices and `adjustment` columns, so any bar can be traced to
  its source.

## Consequences

- Adjusted historical prices can drift far from traded levels over long histories. Features must use
  differences or ratios of adjusted prices, never absolute levels. The DSL feature library (phase 2) is
  built that way.
- The owner's 2025 files have a 06-25 → 09-25 no-overlap gap (2025-06-20 to 2025-07-28). The build flags
  it, and the new export must cover it.
