# Weekly Report — Ankit Ranjan — Week of 2026-08-10

**Company:** Google  ·  **This week's role:** Google specialist (company track)

> Compiled in Task 10 from this repository's commit record for the week. It was
> not written during the week itself, and it claims nothing the record does not
> show. Team-wide role rotation is the Scrum Leader's to record in
> [`meeting-minutes/`](../meeting-minutes/); none is asserted here.

**In focus:** Tasks 06 and 07 — Competitor Comparison, Demand Forecasting

## Done this week
- Employer registry and competitor-set builder, then the comparison layer:
  six companies, 33 comparison tables, 9 figures, 359 tests.
- Registered C4: Google is 846 postings for comparison, not the 848 the screen
  reports — two rows fall outside the comparability window.
- Stopped Task 06 from rewriting Task 02's raw backfill, which it had been
  doing silently.
- Built the forecasting layer and its outputs: 15 tables, 8 figures, 425 tests.
- Registered C5: Task 06's H1 panel was counting February.

## In progress
- Writing up why the forecast is published as unsupported rather than
  quietly omitted.

## Blockers
- No model beats persistence at any horizon, and the h=1 interval spans 3.15x.
  The maximum useful horizon is 0. The deliverable exists; the forecast in it
  is marked unsupported.

## Next week
- Task 08: company similarity scoring, against a null rather than against
  intuition.

## Links (commits / notebooks / data)

The list runs to the commit at which this file was written.

- `5f86b31` 2026-08-10 — Add the Task 06 employer registry and competitor-set builder
- `8cb258b` 2026-08-10 — Add the cross-company comparison layer
- `1fd2f23` 2026-08-10 — Build the Task 06 comparison tables and figures
- `3aaba53` 2026-08-10 — Stop Task 06 from rewriting Task 02's raw backfill
- `e911194` 2026-08-10 — Register C4: Google is 846 postings, not 848
- `63b4c9a` 2026-08-10 — Document the Task 06 comparison method as the team standard
- `bae36ec` 2026-08-10 — docs(task-06): Google specialist comparison report
- `b22a10b` 2026-08-10 — docs: mark Task 06 complete in the README checklists
- `b20f85c` 2026-08-16 — Add the demand-forecasting layer and its tests
- `097cf45` 2026-08-16 — Build the Task 07 forecast tables and figures
- `aa4002d` 2026-08-16 — Document the Task 07 forecasting method and the Google forecast
- `22cf72a` 2026-08-16 — Register C5: Task 06's H1 panel counts February
