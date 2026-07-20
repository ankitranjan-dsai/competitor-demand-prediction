# Legal checklist — Adzuna official API

```
Source name:              Adzuna Jobs API
URL / API base:           https://api.adzuna.com/v1/api/jobs/{country}/search/{page}
Access method:            [x] official API   [ ] licensed dataset   [ ] scrape
ToS reviewed (date):      2026-07-20   Automated access allowed?   [x] yes [ ] no
Redistribution allowed?   [ ] yes [ ] no [x] n/a  (we analyse, never republish text)
robots.txt checked (date):2026-07-20   Target path Disallowed?     [ ] yes [x] no
                          (n/a — sanctioned API endpoint, not crawling)
Licence:                  Adzuna developer terms (free tier)
Personal data present?    [x] no
Attribution required?     [x] yes [ ] no   → how: "Jobs data by Adzuna" credit in
                                             reports/visuals that use this data
Decision:                 [x] APPROVED   [ ] REJECTED   Reviewer: Ankit Ranjan
```

## Notes

- Access is via documented, credentialed API endpoints — the sanctioned route.
  Free registration at <https://developer.adzuna.com> issues an `app_id` +
  `app_key`; the collector reads them from `.env` (git-ignored).
- Provides exactly what the HF backfill lacks: **full description text**,
  **redirect URL** (`job_url`), created date, salary min/max, category.
- Company filter is fuzzy, so the collector re-filters results to genuine
  Google-family employers after the pull.
- Etiquette enforced in `src/collect_google_jobs.py`: honest User-Agent,
  ≤1 request/sec, incremental pulls with stable de-dup ids.

**Status 2026-07-20:** APPROVED, pending credentials — Ankit needs to register
for the free keys before the live pull runs.
