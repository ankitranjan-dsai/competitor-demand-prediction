# Weekly Report — Ankit Ranjan — Week of 2026-08-17

**Company:** Google  ·  **This week's role:** Google specialist (company track)

> Compiled in Task 10 from this repository's commit record for the week. It was
> not written during the week itself, and it claims nothing the record does not
> show. Team-wide role rotation is the Scrum Leader's to record in
> [`meeting-minutes/`](../meeting-minutes/); none is asserted here.

**In focus:** Tasks 08 and 09 — Company Similarity, Insight Generation

## Done this week
- Similarity layer and runner; five metrics against two nulls; 16 tables,
  8 figures, 37 new tests (478 in the suite).
- Recalibrated the closure null at the panel's own dispersion and regenerated
  every output, then checked the statistics against scipy and scikit-learn.
- Registered C6 (concepts and rare skills dominate a cosine), and C7 — the two
  corrections about the project rather than the data: Task 08's name, and a
  handover document written for a task that never existed.
- Built the Task 09 insight compiler and its four gates, generated 436 claims
  from the committed verdict tables, and published 120 — a 27.5% yield.
- Registered C8: a unanimity count is not a robustness statistic, which is why
  every relative-share claim now carries a floor-dependence clause.

## In progress
- The Google insight report, and the machine-readable claim ledger the deck
  will later be built from.

## Blockers
- 13 of the 15 similarity pairs are unresolved; only Google–Meta is robust.
- Trajectory similarity is refused outright: one of 15 pairs is eligible.
- Salary benchmarking is refused at identification. Google discloses a salary
  in 4.02% of its 2023 postings, so a median is a median over whoever chose to
  disclose.

## Next week
- Task 10: the final presentation and the workspace tidy-up.

## Links (commits / notebooks / data)

The list runs to the commit at which this file was written.

- `2bde50b` 2026-08-18 — Add the Task 08 similarity layer
- `89b74f1` 2026-08-18 — Add the Task 08 runner
- `9009b9e` 2026-08-18 — Commit the Task 08 tables, figures and evidence report
- `b463633` 2026-08-18 — Run the closure null at the panel's own dispersion
- `15e79c6` 2026-08-18 — Regenerate Task 08 outputs at the calibrated closure null
- `63a9e32` 2026-08-18 — Validate the Task 08 statistics against scipy and scikit-learn
- `3d4fd04` 2026-08-18 — Pin the Task 08 similarity rules with 37 tests
- `b45078f` 2026-08-18 — docs: register C6 — concepts and rare skills cannot dominate a cosine
- `da48c10` 2026-08-18 — fix: calibration does reorder pairs — 6 of 15, max 3 places
- `e2656bf` 2026-08-18 — docs: Task 08 methods — company similarity scoring, team standard
- `ff13692` 2026-08-18 — docs: Task 08 Google report — one identified finding of five pairs
- `ca08b7a` 2026-08-18 — docs: mark Task 08 complete in both READMEs
- `b65569a` 2026-08-18 — docs: register C7 — Task 08's name, and a handover written for a task that never existed
- `1c3f598` 2026-08-21 — Add the Task 09 insight compiler and its four gates
- `665a5d9` 2026-08-21 — Test the gates, and pin eight defects the first draft shipped
- `411b556` 2026-08-21 — Register C8: a unanimity count is not a robustness statistic
- `a538cc2` 2026-08-21 — Add the Task 09 method standard for the team
- `3793d3e` 2026-08-21 — Add Google's Task 09 tables, figures and machine-readable report
- `1036285` 2026-08-21 — Add Google's Task 09 insight report
- `5cc87f4` 2026-08-21 — Mark Task 09 done and point the Task 06 summary at C8
