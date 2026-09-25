# ADR 0006: History starts in 2025; survivors are exported to NinjaTrader and TradingView

- Status: accepted
- Date: 2026-09-24
- Supersedes: the default splits in design.md section 3, and ADR 0005's premise that 2025 is holdout

## Context

The owner decided that data from 2025 onward is enough history. The owner subscribes to data on both
NinjaTrader (including Level 2) and TradingView, and wants surviving strategies to run on both platforms.

## Decision

### Splits

| Split | Range | Approx. trading days |
| --- | --- | --- |
| Search | 2025-01-01 to 2025-12-31 | ~250, minus the missing 2025-06-20 to 2025-07-28 stretch unless re-exported |
| Validation | 2026-01-01 to 2026-04-30 | ~80 |
| Sealed holdout | 2026-05-01 onward | ~100 as of 2026-09 and growing |

- The 2025 data is still uncontaminated: until now it was used only for pipeline engineering (parsing,
  timezone detection, rolls), and no strategy was ever evaluated on it. ADR 0005's rule carries over to the
  new holdout: nothing evaluates strategies on 2026-05-01 onward except `foundry holdout open`.
- The minimum split sizes drop to 200 / 70 trading days. The kill tests' trade minimums (200 on search, 80 on
  validation) are unchanged.

### Honest consequences

- About 16 months of in-sample history covers far fewer volatility regimes than 2018–2024 would. The regime
  test ("profitable in 2 of 3 volatility buckets") has little data per bucket. Expect more candidates to be
  killed, or labelled regime-specific.
- Deflated Sharpe and overfitting probability are harsher with short samples. That is intended: fewer
  survivors, but more believable ones.
- 1-minute bars are the unit of simulation. Level 2 data isn't used in v0 (see TODO.md).

### Platform targets

- The Foundry never connects to a broker or platform. For each survivor it **generates** a NinjaScript
  strategy (C#) and a Pine Script v6 strategy from its DSL spec, using deterministic templates. An LLM never
  writes this code.
- Both templates reproduce the backtester's semantics: signals on bar close, market fills at the next bar's
  open, the same stops and targets, session-flat rules and contract size. NinjaTrader uses
  `Calculate.OnBarClose`; Pine uses `process_orders_on_close=false`.
- **Parity test:** the owner runs each exported strategy in the platform's strategy tester on the
  validation period and exports the trade list. `foundry parity` compares it trade by trade with the
  Foundry's own list. A strategy is released for the owner's shadow trading only if they match within
  tolerance.
- Data sources: NinjaTrader per-contract exports are the primary source, since the continuous series is built
  from them. TradingView exports (usually a continuous `NQ1!`, possibly already back-adjusted) are loaded as a
  `contract_from: continuous` source, used as a cross-check and never mixed into the primary series.

### LLM proposer

The default model is DeepSeek V4.1 Flash (owner's reference: the Hugging Face
`deepseek-ai/DeepSeek-V4.1-Flash` technical report). The client stays provider-agnostic, and the model id
lives in config.
