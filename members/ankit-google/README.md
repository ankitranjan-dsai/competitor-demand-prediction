# Ankit Ranjan — Google (Specialist)

My working folder for Google job-posting analysis. Shared, reusable code goes in
the top-level `src/`; company-specific exploration and notes live here.

## Data-source plan (Google)
- **Live postings:** Adzuna API + The Muse API (filter to Google).
- **History / backfill:** Kaggle & Hugging Face tech-job datasets.
- **Google Careers:** verify-only unless `docs/legal/` check comes back APPROVED.

See `docs/task-01-data-sources-and-legal.md` for the full legal rationale.

## Task progress
- [x] Task 01 — Data sources & legal (drafted)
- [x] Task 02 — Data collection → `task-02-data-collection-report.md`
      (848 postings backfill 2023; Adzuna live pull ready, pending free API keys)
- [x] Task 03 — NLP preprocessing → `task-03-preprocessing-report.md`
      (848 rows → 44 columns; method standard in `docs/task-03-preprocessing-methods.md`;
      89 tests; Layer B validated on real HTML, idle until Adzuna text lands)
- [x] Task 04 — Skill extraction & feature engineering → `task-04-skill-extraction-report.md`
      (601/848 postings skilled, 2,340 posting-skill rows, 91 skills; shared taxonomy in
      `docs/task-04-skill-taxonomy.md`; 157 tests; aggregate tables in `task-04-tables/`)
- [x] Task 05 — Hiring trend analysis → `task-05-trend-report.md`
      (velocity, publisher-panel control, spike attribution; standard in
      `docs/task-05-trend-analysis-methods.md`; 214 tests; 19 tables +
      8 figures. Volume direction is **not identified** — Dec vs Jan reads
      91 raw / 115 balanced / 185 chained; 8 of Task 04's skill trends do not
      survive stratification, Looker first)
- [ ] Task 06 — Competitor comparison
- [ ] Task 07 — Demand forecasting
- [ ] Task 08 — Company similarity scoring
- [ ] Task 09 — Insight generation & reporting
- [ ] Task 10 — Final presentation
