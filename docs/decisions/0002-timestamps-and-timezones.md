# ADR 0002: Timestamps, timezones and bar labels

- Status: accepted
- Date: 2026-09-24

## Context

The owner's 2025 minute files (`NQ MM-YY.Last-Minute.txt`) have no timezone marker. The owner first
answered "America/New_York", but the data contradicts that:

| Evidence | If UTC | If ET |
| --- | --- | --- |
| Hour with about 1/5 of normal bars (NQ 03-25 file) | 22:00 UTC is 17:00 ET, the CME maintenance break. Matches. | Break would be at 17:00; hour 17 is full (3,301 bars). Does not match. |
| Last bar on expiry days (2025-03-21, 2025-06-20) is `133000` | 09:30 ET, the final-settlement cutoff. Matches. | 13:30 ET. Does not match. |
| First bar `20250101 230100` | 18:01 ET, the Globex open. Matches. | 23:01 ET, with no bars in the first five hours. Does not match. |

The owner confirmed UTC after seeing this evidence. The `230100` stamp for the 18:00 ET open also shows
that timestamps mark the **end** of the bar (NinjaTrader's convention).

## Decision

- Each data source in config declares `timezone` (IANA name) and `timestamp_label` (`start` or `end`).
  Neither has a default, so the owner must state them.
- Internally every bar is keyed by its **start** time in `America/New_York`, converted with `zoneinfo`
  via Polars. Bar-end sources are shifted back by one bar length.
- For ET-local sources, a wall-clock time in the fall-back hour is resolved by order of appearance
  (first occurrence is EDT, second is EST). Wall-clock times that don't exist (spring-forward gap) are
  reported as quality issues and dropped, with the count in the report.
- The data build runs a **session-alignment check**: if more than a configured fraction of bars land in
  the 17:00–18:00 ET maintenance window, the build fails with a message pointing at the timezone setting.
  A wrong timezone therefore fails loudly instead of shifting every session label.

## Consequences

- The coming 2018–2025 export may use a different timezone. It is a per-source setting, so both kinds of
  export can be mixed.
- "Bar at 09:30" always means the bar covering 09:30:00–09:30:59 ET.
