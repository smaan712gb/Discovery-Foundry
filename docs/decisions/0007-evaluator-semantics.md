# ADR 0007: Evaluator semantics: fills, exits, costs, regimes

- Status: accepted
- Date: 2026-09-24

## Context

The kill tests, the search and the NinjaTrader/Pine exporters (ADR 0006) all depend on one precise definition
of how a strategy spec turns into trades. Everything below is implemented in `domains/nq/backtest.py` and
covered by the golden tests in `tests/test_backtest.py`.

## Decision

### Timing

- Features and rules are evaluated on the **close of bar t**, using only bars up to and including t.
- An entry or exit decided at bar t is an order for **bar t+1**. Nothing fills on the bar that produced the
  signal.
- A position may be opened only if bar t+1 belongs to the same session instance (same trading date, same
  tradeable window) and is not that window's flat bar.

### Order fills (long side; short mirrors it)

| Order | Trigger in bar i | Fill price | Slippage |
| --- | --- | --- | --- |
| Market | always, at the open | `open[i]` | + `slippage_ticks.market` |
| Stop (buy) at level L | `high[i] >= L` | `max(open[i], L)`, so a gap through the stop fills at the open | + `slippage_ticks.stop` |
| Limit (buy) at level L | `low[i] <= L - limit_trade_through_ticks * tick` | `min(open[i], L)` | `slippage_ticks.limit` (default 0) |

- Entry stop and limit orders stay live for `valid_bars` bars, then are cancelled. They never outlive the
  session window.
- One position at a time. There is no pyramiding and no same-bar reversal.

### Exits, in the order they are checked inside a bar

1. A pending **market exit** (from an exit rule or a time stop decided on the previous bar) fills at the open.
2. **Protective stop** (fixed points, ATR multiple, or trailing), then **target** (a limit order, with the
   trade-through rule). If both could fill in the same bar, the **stop is assumed to fill first**. With
   1-minute bars the order inside the bar is unknown, so the conservative assumption wins.
3. **Session flat**: at the close of the window's flat bar (the bar starting one bar length before the
   scheduled window end), the position exits at `close ∓ slippage_ticks.market`. The flat bar comes from the
   exchange calendar, not from which bars happen to exist, so the rule has no lookahead. If that bar is
   missing, the position exits at the next bar's open, gap risk included.
4. At the close: update the trailing stop from completed bars, then evaluate the exit rule and the time
   stop. Either one creates a market exit for the next bar.

Stops and targets are active on the entry bar too, after the open.

### Costs (always on)

- Commission per round turn per contract, from config for each contract (NQ $4.50 by default).
- Slippage as in the table above.
- Gross PnL is recorded next to net PnL only so the cost test can prove `net < gross`. No report headline
  uses gross.

### Sessions a spec can trade

`RTH` is 09:30 to the RTH close. `ETH` is 18:00 to 09:30. `both` is 18:00 to the RTH close. Each tradeable
window is one session instance, and `session_flat: true` (the default) flattens at its end.

### Regimes (computed per trading date from earlier days only)

- **Volatility bucket**: the previous trading day's realized volatility (std of 1-minute log returns over
  its RTH bars), ranked against the `vol_lookback_days` days before it. Terciles give `low`, `mid` and
  `high`. With too little history the bucket is `unknown`, and any spec with a volatility filter excludes
  those days.
- **Trend state**: the previous RTH close compared with the average of the previous `trend_days` RTH
  closes, giving `up` or `down`.
- **Event days**: dates from `config/events.yaml`.

### Data quality

Entry signals are suppressed on bars whose `quality_flags` intersect `block_signals_on_flags` (default
`bad_tick` and `ohlc_invalid`). Prices are never altered.

## Consequences

- The exporters must reproduce this table exactly. Pine's `process_orders_on_close=false` and NinjaTrader's
  `Calculate.OnBarClose` give next-bar-open fills. Stop-before-target and the trade-through limit rule must be
  coded explicitly in both, and the parity test enforces that.
- Stop-first is pessimistic for strategies with tight targets. That's intended.
