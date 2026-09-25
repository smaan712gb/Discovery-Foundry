You are proposing new trading-strategy hypotheses for NQ (Nasdaq-100 futures, 1-minute bars).
Each hypothesis is a JSON spec in the DSL below. Your specs will be backtested on a search period and
then attacked by strict kill tests (profit after costs, 2x slippage, profit concentration, volatility
regimes, deflated Sharpe for the number of trials, parameter robustness, overfitting probability and
purged cross-validation). Most hypotheses die. Propose ideas with a plausible market reason
(auction/session structure, opening range, overnight inventory, VWAP reversion, volatility regimes),
not curve fits. Keep them simple: 1 to 3 entry conditions. Parameter bounds should be wide and
sensible, because the search tunes within them.

{{DSL}}

## What has happened so far (search-period results only; later tests are aggregate counts)

{{FEEDBACK}}

## Task

Propose {{N}} new, diverse specs that differ from what already died. Give each a one-line
"rationale". Reply with a single JSON object of the form:
{"specs": [ {spec}, {spec}, ... ]}
Every spec must be valid against the schema: "dsl_version": 1, "session", "params", "features",
"entry", "exit" (and optional "filters", "rationale"). Output JSON only.
