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
      H1→H2** (−6.56 pp after [C5](../../docs/corrections.md#c5--task-06s-h1-aggregate-counts-february-on-a-panel-that-does-not-exist-in-february)),
      holding in 4 of 6 publishers — a count later withdrawn as a robustness
      statistic by [C8](../../docs/corrections.md#c8--a-unanimity-count-is-not-a-robustness-statistic-when-the-number-of-tests-moves-with-the-threshold). Google has the lowest skill
      coverage of the six and it survives standardising (70.8% → 76.4%); only
      33 of its 182 pair-skill comparisons survive stratification. Raised
      [C4](../../docs/corrections.md#c4--googles-posting-count-is-846-not-848):
      Google is 846 postings, not 848)
- [x] Task 07 — Demand forecasting → `task-07-forecast-report.md`
      (forecastability gate, rolling-origin backtest, Diebold–Mariano selection,
      order-statistic intervals. Standard in
      `docs/task-07-demand-forecasting-methods.md`; 425 tests; 15 tables +
      8 figures, cross-checked against scipy/statsmodels in
      `docs/task-07-forecast-validation.md`. All six series carry real signal
      (61–86%), only Google and Meta clear the 5-posting cell floor, and
      **no model beats persistence** — the h=1 interval spans 3.15×, so the
      **maximum useful horizon is 0** and the forecast ships marked
      unsupported. The collection is easier to predict than any company's
      demand (naive RMSE 0.293 on the pool against 0.534 on shares). Raised
      [C5](../../docs/corrections.md#c5--task-06s-h1-aggregate-counts-february-on-a-panel-that-does-not-exist-in-february):
      Task 06's H1 base counts February; every sign survives, Google's decline
      deepens to −6.56 pp)
- [x] Task 08 — Company similarity scoring → `task-08-similarity-report.md`
      (five metrics in two families, identical/unrelated nulls per pair,
      posting-level bootstrap, rank tiers, four published sensitivities.
      Standard in `docs/task-08-company-similarity-methods.md`; 478 tests;
      16 tables + 8 figures, cross-checked against scipy in
      `docs/task-08-similarity-validation.md`. **Google–Meta is the most
      similar pair** (0.9174 raw, 0.9123 calibrated) and the only `robust`
      one — rank 1 in 100% of resamples, under all five metrics, after
      dropping own products and after standardising role mix. Only **2 of 15
      ranks are identified**; 11 pairs are one tier. Google has the largest
      own-product vocabulary of the six, and Google–Microsoft rises 0.67 → 0.94
      when both stacks are dropped. Trajectory similarity refused: 1 eligible
      pair, mean r inside the closure null at the panel's own dispersion.
      Raised [C6](../../docs/corrections.md#c6--neither-concept-skills-nor-rare-skills-dominate-a-similarity-score):
      concept skills carry 0.36% of a cosine numerator and single-posting
      skills exactly 0%, against Task 04's prediction that both would dominate.
      Raised [C7](../../docs/corrections.md#c7--task-08-is-company-similarity-scoring-not-visualisation-and-not-evaluation): Task 06's §11 named this task
      "Visualisation" / "Evaluation" and handed it two instructions written for
      a task that does not exist — a handover section is a prediction, not an
      instruction)
- [x] Task 09 — Insight generation & reporting → `task-09-insight-report.md`
      (four gates in a fixed order — evidence, lint, identification,
      consistency — over **436 mechanically generated candidate claims**, so
      the yield denominator is not the author's shortlist. Standard in
      `docs/task-09-insight-generation-methods.md`; 557 tests; 25 tables +
      8 figures. **120 publish (27.5%)**, 102 of them about Google; **306 of
      the 316 refusals fail on identification**, not on phrasing. Position is
      available as a *profile* — 48 distinctive skills, nearest neighbour Meta
      at 0.9174 — and refused as a level and as a trajectory. Of those 48, only
      14 are skills Google asks for **more** than the other five, and two of
      its own products (GCP, Kubernetes) sit on the negative side. Google leads
      just **2 of 33** stratified pair-skill gaps. **Salary benchmarking is
      refused**: disclosure is a publisher behaviour (4.02% of Google's 846
      postings, and the disclosed subset is 76.5% US against 27.0% overall),
      and the one pair that excluded zero has no testable job-function stratum.
      **Investors get 0 claims** — every investor question is a level, a
      trajectory or a forecast. Raised
      [C8](../../docs/corrections.md#c8--a-unanimity-count-is-not-a-robustness-statistic-when-the-number-of-tests-moves-with-the-threshold):
      Task 06's publisher-agreement floor applies to the publisher's total, not
      the company's own cell, so three of six companies *gain* `confirmed` by
      dropping tests and NVIDIA's 6/6 runs p 0.0312 → 1.0000 as the floor rises)
- [ ] Task 10 — Final presentation
