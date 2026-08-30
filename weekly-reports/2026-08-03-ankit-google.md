# Weekly Report — Ankit Ranjan — Week of 2026-08-03

**Company:** Google  ·  **This week's role:** Google specialist (company track)

> Compiled in Task 10 from this repository's commit record for the week. It was
> not written during the week itself, and it claims nothing the record does not
> show. Team-wide role rotation is the Scrum Leader's to record in
> [`meeting-minutes/`](../meeting-minutes/); none is asserted here.

**In focus:** Task 05 — Hiring Trend Analysis, and the corrections register

## Done this week
- Shared trend module with a publisher-panel control, plus the runner that
  produces 19 trend tables and 8 figures; 214 tests.
- Ran the panel check on every breakdown rather than only on job function,
  which is what turned up the first correction.
- Opened [`docs/corrections.md`](../docs/corrections.md) so a superseded claim
  is never silently rewritten: the original wording stays and gains a marker.
- Registered C1 (February's trough and August's peak are collection artefacts),
  C2 (only four of Task 04's ten headline skill movers survive the panel), and
  C3 (`posting_date` is an aggregator first-seen date, not a hiring date).

## In progress
- Team standard for trend analysis, so the other company tracks run the same
  panel control.

## Blockers
- The raw monthly series is confounded by publishers entering and leaving the
  sample. Resolved by fixing the publisher panel, at the cost of a smaller n.
- The collection covers no complete annual cycle, so month-of-year seasonality
  cannot be estimated at all. It becomes a refusal, not a caveat.

## Next week
- Task 06: the competitor comparison, on the shared panel.

## Links (commits / notebooks / data)

The list runs to the commit at which this file was written.

- `42f7785` 2026-08-03 — Task 05: shared hiring-trend module with publisher-panel control
- `7963553` 2026-08-03 — Task 05: trend runner producing 17 tables, 8 figures and an evidence report
- `6650fcd` 2026-08-03 — Task 05 (Google): trend tables, figures and evidence report
- `b59b144` 2026-08-03 — Task 05: team standard for hiring-trend analysis
- `086b320` 2026-08-03 — Task 05: run the panel check on every breakdown, not just job function
- `f5a214d` 2026-08-03 — Task 05 (Google): panel checks for job category and country
- `bf55233` 2026-08-03 — Task 05 (Google): hiring trend report
- `c09a462` 2026-08-03 — Mark Task 05 complete in the task tables
- `c5733a2` 2026-08-03 — Add a corrections register so a superseded claim is never silently rewritten
- `2d39b2c` 2026-08-03 — Correct C1: Feb's trough and Aug's peak are collection artefacts, not hiring
- `17a626a` 2026-08-03 — Correct C3: posting_date is a first-seen date on aggregator sources
- `a25d035` 2026-08-03 — Correct C2: only four of Task 04's ten headline skill movers survive
- `5e80a27` 2026-08-03 — Guard the corrections register against silent drift
- `7f70282` 2026-08-03 — Make the corrections register reachable from the READMEs and the Task 05 standard
