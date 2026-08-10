# Legal checklist — Hugging Face `lukebarousse/data_jobs`

```
Source name:              lukebarousse/data_jobs (Hugging Face Datasets)
URL / API base:           https://huggingface.co/datasets/lukebarousse/data_jobs
Access method:            [ ] official API   [x] licensed dataset   [ ] scrape
ToS reviewed (date):      2026-07-20   Automated access allowed?   [x] yes [ ] no
Redistribution allowed?   [x] yes [ ] no [ ] n/a   (Apache-2.0)
robots.txt checked (date):2026-07-20   Target path Disallowed?     [ ] yes [x] no
Licence:                  Apache-2.0 (verified via HF API `license:apache-2.0` tag)
Personal data present?    [x] no   (company/title/location/skills only — no names,
                                    emails, or any individual-level data)
Attribution required?     [ ] yes [x] no   → how: courtesy credit to Luke Barousse
                                             kept in dataset `source` column
Decision:                 [x] APPROVED   [ ] REJECTED   Reviewer: Ankit Ranjan
```

## Notes

- 785,741 data-role job postings from 2023, collected by the dataset author via
  serp results; published under Apache-2.0, which permits use, modification and
  redistribution. Downloaded through the official HF parquet API.
- Verified 2026-07-20: licence tag `license:apache-2.0`, last modified
  2025-06-03, 10k+ downloads.
- **Fields:** job_title_short, job_title, job_location, job_via, schedule type,
  work-from-home flag, posted date, country, salary (yearly/hourly avg),
  company_name, job_skills, job_type_skills.
- **Limitations (documented in the Task 02 report):** no full description text,
  no job URL, data-analytics roles only, 2023 window. Used as *historical
  backfill*; live full-text postings come from the Adzuna API.

## Scope extension (Task 06)

**No new data source, no new collection.** Task 06 compares Google against five
competitors. Every competitor posting comes from the parquet file this
checklist already approved and Task 02 already downloaded
(`data/raw/google/_hf_data_jobs_full.parquet`, all 785,741 rows — the file is
the whole dataset, not a Google subset). The only thing Task 06 does that
Task 02 did not is select a different set of employer strings out of it.

Re-checked against the four questions that decide whether a scope change needs
its own review:

| Question | Answer |
| --- | --- |
| New request to a third-party server? | No. The file is on disk from Task 02. |
| New fields read? | No. Same twelve columns, same shared schema. |
| Personal data now in scope? | No. Employer name, title, location, publisher, skills. The employer-matching audit is built from `company_name`, which is an organisation, and `tests/test_companies.py::test_matching_audit_carries_no_personal_data_columns` re-runs Task 01's check on it. |
| Licence permits it? | Yes. Apache-2.0 places no per-employer restriction; selecting Microsoft rows is the same permission as selecting Google rows. |

Two consequences worth recording rather than assuming:

- **Nothing row-level is committed.** `src/build_competitor_set.py` writes
  competitor postings only to `data/processed/<company>/`, which is
  git-ignored. The committed Task 06 output is aggregate tables plus three
  audit tables (`employer-matching-audit.csv`,
  `company-feasibility-screen.csv`, `competitor-set-manifest.csv`).
- **No competitor is contacted, profiled or scraped.** The comparison is over
  postings a third party published in 2023 and licensed for redistribution.
  Task 01's standing rules are unchanged: API-first, no scraping without a
  passed §4 checklist, Reed still not approved.

Reviewer: Ankit Ranjan. Scope extension recorded 2026-08-10; the approval above
is unchanged and was not re-opened, because nothing it turns on changed.
