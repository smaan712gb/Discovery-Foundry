# ADR 0004: Session labels and data-quality policy

- Status: accepted
- Date: 2026-09-24

## Session labels

The spec defines RTH (09:30–16:00 ET) and ETH (18:00 to 09:30 ET next day), with no bars in the
17:00–18:00 maintenance break. NQ also trades 16:00–17:00 ET, which falls in neither session. Each bar,
keyed by its start time, gets exactly one label:

| Label | Rule | Tradeable by DSL |
| --- | --- | --- |
| `RTH` | 09:30 ≤ t < RTH close (16:00, or the calendar's early close) on a normal trading date | yes (`RTH`, `both`) |
| `ETH` | 18:00 ≤ t, or t < 09:30, within the trading date's Globex session | yes (`ETH`, `both`) |
| `POST` | RTH close ≤ t < Globex close (normally 16:00–17:00) | no |
| `HOLIDAY` | Daytime bars on an exchange holiday with an abbreviated Globex session (no RTH) | no |
| `OUTSIDE` | Maintenance break, weekend, after an early Globex halt, or a fully closed date | no (and a quality issue) |

**Trading date**: bars at or after 18:00 ET belong to the next weekday that is not a full closure. All other
bars belong to their own calendar date.

Holiday and early-close dates live in `config/calendar_cme_equity.yaml`. They were entered by hand from NYSE and
CME schedules and must be checked against CME notices (see TODO.md).

## Quality policy: flag, never fill

- Every issue is recorded in `quality_issues.parquet` (bar, type, detail) and summarised in the report.
- Problem bars stay in the data with a `quality_flags` bitmask. No bar is filled, interpolated or
  forward-filled. Whether flagged bars are excluded from trading is a separate, explicit evaluator setting.
- Removed rows, always counted in the report:
  - Exact duplicate rows are collapsed to one.
  - ET wall-clock times that don't exist (DST spring-forward gap) are dropped.
- Fatal (the build fails):
  - Duplicate timestamps with different values
  - Failed session-alignment check (ADR 0002)
  - Rolls without overlap (ADR 0003)
  - Failed roll-continuity check
  - An empty search or validation split
- Gaps (expected session minutes with no bar) are reported per session type, with the largest gaps
  listed. Thin overnight minutes with no trades are normal in 1-minute NQ data and are not an error.
- A bad tick is a bar whose range or close-to-close move exceeds `bad_tick_atr_multiple` × the ATR of
  the preceding `bad_tick_atr_period` bars (the current bar is excluded, so the check itself has no
  lookahead).
