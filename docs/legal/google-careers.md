# Legal checklist — Google Careers (direct scraping)

```
Source name:              Google Careers
URL / API base:           https://www.google.com/about/careers/applications/jobs/results/
Access method:            [ ] official API   [ ] licensed dataset   [x] scrape
ToS reviewed (date):      2026-07-20   Automated access allowed?   [ ] yes [x] no
Redistribution allowed?   [ ] yes [x] no [ ] n/a
robots.txt checked (date):2026-07-20   Target path Disallowed?     [x] yes [ ] no
Licence:                  n/a — Google Terms of Service
Personal data present?    [x] no
Attribution required?     [ ] yes [x] no   → how: n/a
Decision:                 [ ] APPROVED   [x] REJECTED   Reviewer: Ankit Ranjan
```

## Evidence

`https://www.google.com/robots.txt` fetched 2026-07-20 and archived at
[`evidence/google.com-robots.txt-2026-07-20.txt`](evidence/google.com-robots.txt-2026-07-20.txt).
For `User-agent: *` it explicitly disallows the paginated job-results listing —
the exact path bulk collection would need:

```
Disallow: /about/careers/applications/candidate-prep
Disallow: /about/careers/applications/connect-with-a-googler
Disallow: /about/careers/applications/jobs/results?page=
Disallow: /about/careers/applications/jobs/results/?page=
Disallow: /about/careers/applications/jobs/results?*&page=
Disallow: /about/careers/applications/jobs/results/?*&page=
```

## Decision rationale

Task 01 required **both** ToS and `robots.txt` to permit collection. `robots.txt`
disallows crawling the paginated results, so the source fails the check
regardless of any ToS reading. **REJECTED for automated collection.** The site
remains fine for *manual verification* of individual postings (e.g. spot-checking
that an aggregator record is real).

Google postings are instead collected via the **Adzuna official API**
([`adzuna.md`](adzuna.md)) and the **Apache-2.0 Hugging Face backfill dataset**
([`huggingface-data-jobs.md`](huggingface-data-jobs.md)).
