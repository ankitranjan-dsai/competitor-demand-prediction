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
      survive stratification, Looker first). Corrections it forced on Tasks
      01–04 are registered in [`docs/corrections.md`](../../docs/corrections.md)
- [x] Task 06 — Competitor comparison → `task-06-comparison-report.md`
      (six-company set — Google, Meta, Microsoft, Snowflake, Databricks, NVIDIA;
      OpenAI and Anthropic fail the feasibility screen on 14 and 9 postings.
      Standard in `docs/task-06-competitor-comparison-methods.md`; 359 tests;
      33 tables + 9 figures. Cross-company **levels are not identified** —
      Snowflake's common-panel share is 23.5% against a 40% floor. What is:
      share of the seven-publisher common pool, where **Google's fell 4.84 pp
      H1→H2**, holding in 4 of 6 publishers. Google has the lowest skill
      coverage of the six and it survives standardising (70.8% → 76.4%); only
      33 of its 182 pair-skill comparisons survive stratification. Raised
      [C4](../../docs/corrections.md#c4--googles-posting-count-is-846-not-848):
      Google is 846 postings, not 848)
- [ ] Task 07 — Demand forecasting
- [ ] Task 08 — Company similarity scoring
- [ ] Task 09 — Insight generation & reporting
- [ ] Task 10 — Final presentation
