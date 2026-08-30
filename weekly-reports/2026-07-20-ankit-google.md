# Weekly Report — Ankit Ranjan — Week of 2026-07-20

**Company:** Google  ·  **This week's role:** Google specialist (company track)

> Compiled in Task 10 from this repository's commit record for the week. It was
> not written during the week itself, and it claims nothing the record does not
> show. Team-wide role rotation is the Scrum Leader's to record in
> [`meeting-minutes/`](../meeting-minutes/); none is asserted here.

**In focus:** Tasks 02, 03 and 04 — Collection, NLP preprocessing, Skill extraction

## Done this week
- Built the collector and ran the Google backfill: 848 postings, with the legal
  checklists and their evidence committed alongside.
- Wrote the preprocessing pipeline and the team's method-selection standard;
  848 rows cleaned, 89 tests.
- Built the shared skill taxonomy and the extraction layer: 2,340 skill rows
  over 91 skills, 157 tests.
- Recorded two things that constrain everything downstream: Adzuna truncates
  descriptions at 500 characters, and Reed stays behind its approval gate.

## In progress
- Deciding the skill denominator. `share_of_all` confounds demand with
  description coverage; `share_of_skilled` is the candidate replacement.

## Blockers
- The 500-character truncation means a long description contributes fewer
  skills than a short one. It is a measurement property, not a bug to fix, so
  it goes in the report and later into the linter.
- Row-level postings stay out of the repository; only aggregates are committed.

## Next week
- Task 05: hiring trends, with a control for publishers entering and leaving
  the sample.

## Links (commits / notebooks / data)

The list runs to the commit at which this file was written.

- `f541a39` 2026-07-20 — Task 02: Google data collection — collector, legal checklists, 848-posting backfill + report
- `0bd7210` 2026-07-25 — Task 03: NLP preprocessing pipeline, method-selection standard + Google report
- `264c0c4` 2026-07-25 — Task 03 report: record the Adzuna 500-char truncation and Reed approval gate
- `28f737c` 2026-07-25 — Task 04: skill extraction & feature engineering — shared taxonomy + Google report
