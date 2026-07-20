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
