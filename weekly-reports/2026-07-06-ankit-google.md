# Weekly Report — Ankit Ranjan — Week of 2026-07-06

**Company:** Google  ·  **This week's role:** Google specialist (company track)

> Compiled in Task 10 from this repository's commit record for the week. It was
> not written during the week itself, and it claims nothing the record does not
> show. Team-wide role rotation is the Scrum Leader's to record in
> [`meeting-minutes/`](../meeting-minutes/); none is asserted here.

**In focus:** Task 01 — Understanding Data Sources & Legal

## Done this week
- Scaffolded the shared repository: `src/` for company-agnostic code, `members/`
  for per-company work, `docs/` for team standards, `tests/` beside them.
- Screened candidate sources API-first and wrote the §4 legal checklist that a
  source has to pass before a single request is sent.
- Approved three sources with committed checklists and evidence: Adzuna, The
  Muse, and the Hugging Face data-jobs dataset.
- Rejected Google Careers on its robots.txt, and recorded the rejection rather
  than quietly dropping it → [`docs/legal/rejected-sources.md`](../docs/legal/rejected-sources.md).
- Put `.env` in `.gitignore` before any key existed.

## In progress
- Collection plan for the Google track: which endpoints, which fields, and
  which fields are deliberately not collected.

## Blockers
- No live Adzuna key yet, so the first Google pull will have to come from the
  approved backfill dataset rather than the API.
- A Reed key exists but Reed has no approved §4 checklist, so it cannot be
  called. It never was.

## Next week
- Task 02: collect the Google postings and write the collection report.

## Links (commits / notebooks / data)

The list runs to the commit at which this file was written.

- `5a90491` 2026-07-10 — Task 01: repo scaffold + data-sources & legal deliverable (Google specialist)
