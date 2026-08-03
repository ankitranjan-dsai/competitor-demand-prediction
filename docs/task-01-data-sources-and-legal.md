# Task 01 — Understanding Data Sources & Legal

**Specialist:** Ankit Ranjan · **Company:** Google
**Status:** Draft for team review · **Task owner deliverables:** approved source
list · legal understanding · Task 2 collection plan

> **Golden rule for the whole team:** prefer **official APIs** and **openly
> licensed datasets** over scraping. Only scrape when a site's Terms of Service
> *and* `robots.txt` both permit it, and never collect personal or sensitive
> information.

---

## 1. Where job-posting data comes from

Five source categories, from most to least preferred on legal/effort grounds:

| # | Category | Examples | Why / caveats |
| --- | --- | --- | --- |
| 1 | **Official job-board APIs** | Adzuna, The Muse, Remotive, Arbeitnow, Jobicy, Findwork, USAJobs, Reed | Sanctioned access, structured JSON, ToS-clean. **Best option.** |
| 2 | **Open / licensed datasets** | Kaggle, Hugging Face Datasets, Google Dataset Search, EU/gov open-data portals | Pre-collected, licence stated up front. Great for backfill/history. |
| 3 | **Web-archive corpora** | Common Crawl, Internet Archive | Legal to use crawled pages; heavy to process. |
| 4 | **Company career pages** | `google.com/about/careers` and peers | Freshest & most on-target, but ToS/`robots.txt` usually restrict scraping. Use only if permitted; otherwise reach them via an aggregator that has rights. |
| 5 | **Aggregators that prohibit scraping** | LinkedIn, Indeed, Glassdoor | ❌ **Do not scrape.** ToS forbid it; legal risk. Use their official API only if one exists and is granted. |

## 2. Approved source list

### 2a. Team-wide sources (all four companies)

These are API-first and safe to standardise on. Each covers multiple employers,
so every specialist can filter to their own company.

| Source | Access | Auth | Licence / ToS note | Covers Google? |
| --- | --- | --- | --- | --- |
| **Adzuna API** | REST API | Free app id + key | Developer ToS; attribution required | ✅ filter `what_company=google` |
| **The Muse API** | REST API | Optional key | Public API, generous free tier | ✅ company filter |
| **Remotive API** | REST API | None | Public remote-jobs feed | ⚠️ mostly remote-first cos |
| **Arbeitnow API** | REST API | None | Public feed, no auth | ⚠️ EU-heavy |
| **Findwork.dev API** | REST API | Free token | Public API | ⚠️ partial |
| **USAJobs API** | REST API | Free key + email | US federal (public sector benchmark) | ❌ (benchmark only) |
| **Kaggle datasets** | Download | Kaggle account | Per-dataset licence — check each | ✅ several tech-jobs sets |
| **Hugging Face Datasets** | Download | Optional | Per-dataset licence | ✅ some job-posting corpora |
| **Common Crawl** | S3 / index | None | Open crawl data | ✅ career pages appear |

### 2b. Google-specific notes (my company)

- **Google Careers** (`https://www.google.com/about/careers/applications/jobs/results/`)
  is the authoritative source but must be treated as **read/verify only** unless
  `robots.txt` + ToS clearly allow automated collection. Action: I will record the
  live `robots.txt` and the relevant ToS clause in `docs/legal/` before Task 2 and
  make a documented allow/deny decision (see checklist §4).
- **Primary plan for Google data:** pull Google postings through the **Adzuna** and
  **The Muse** APIs (both let you filter by company), and backfill history from
  **Kaggle / Hugging Face** tech-job datasets. This gives clean, ToS-safe coverage
  without scraping Google properties directly.
- **Google Dataset Search** (`datasetsearch.research.google.com`) is a discovery
  tool for finding additional openly licensed job datasets — not a data source itself.

## 3. Legal & ethical requirements

Four pillars every specialist must satisfy and evidence in `docs/legal/`:

1. **Terms of Service (ToS).** Read each source's ToS. Confirm automated access /
   redistribution is allowed. If a site forbids scraping or bulk access, we don't
   use it — no exceptions. Save the relevant clause + date accessed.
2. **`robots.txt`.** Fetch `https://<domain>/robots.txt` and honour every
   `Disallow` for our user-agent. If the target path is disallowed, don't collect
   it. Save the file text + date.
3. **Public vs copyrighted data.** Facts (company name, title, location, date) are
   generally not protected; the **prose of a job description is copyrightable
   text**. We collect it for *analysis/NLP*, store minimally, don't republish it
   verbatim, and rely on licensed/API sources that grant us the right to process it.
4. **No personal or sensitive information — ever.** No names of recruiters/hiring
   managers, no emails/phone numbers, no applicant data, nothing that identifies an
   individual. Job postings are about *roles*, not people. (Aligns with GDPR/UK GDPR
   principles for the team's UK/EU context.)

**Operational etiquette (when any HTTP access is used):**
- Send a clear, honest `User-Agent`.
- Rate-limit (e.g. ≤ 1 request/sec) and cache; never hammer a host.
- Prefer incremental pulls over re-downloading everything.
- Store the source URL (`job_url`) for every record for traceability.

## 4. Per-source legal checklist (fill one per source in `docs/legal/`)

```
Source name:
URL / API base:
Access method:            [ ] official API   [ ] licensed dataset   [ ] scrape
ToS reviewed (date):      ____   Automated access allowed?   [ ] yes [ ] no
Redistribution allowed?   [ ] yes [ ] no [ ] n/a
robots.txt checked (date):____   Target path Disallowed?     [ ] yes [ ] no
Licence:                  (e.g. CC-BY, custom API ToS, dataset licence)
Personal data present?    [ ] no  (if yes → exclude those fields)
Attribution required?     [ ] yes [ ] no   → how:
Decision:                 [ ] APPROVED   [ ] REJECTED   Reviewer: ____
```

## 5. Shared plan for data collection (feeds Task 2)

1. **Standardise on the API-first sources in §2a** so all four companies are
   collected the same way and stay comparable.
2. **Agree a shared schema now** (matches the brief's dataset schema) so everyone's
   raw output lines up:
   `job_id, company_name, job_title, job_description, posting_date, location, job_url`
   (required) + recommended fields where the source provides them
   (`employment_type, department/job_category, seniority_level, salary_range,
   employment/remote flag, scraped_date`).

   > **`posting_date` does not mean the same thing in every source**
   > *(added by Task 05, [C3](corrections.md#c3--posting_date-is-an-aggregator-first-seen-date-not-a-publication-date))*.
   > From an employer's own careers API it is a publication date. From a job
   > board or aggregator it is usually the date that aggregator **first saw**
   > the posting — which is why the Google backfill shows a crawler's working
   > week (21.3% weekend against 28.6% uniform) and 25 postings sharing one
   > day. Record which of the two you have in your source's `docs/legal/`
   > checklist, and never read a sub-weekly pattern as employer behaviour
   > unless the source is the employer.
3. **Each specialist**: complete the §4 checklist for every source they use, save
   evidence in `docs/legal/`, then pull their company's postings into
   `data/raw/<company>/` as CSV/Parquet.
4. **Google (me):** Adzuna + The Muse for live postings, Kaggle/HF for history;
   Google Careers only if the §4 check comes back APPROVED.
5. **Data Quality Lead** validates that every raw file has the required fields, a
   populated `job_url`, and **zero personal-data columns** before Task 3.
6. **Cadence:** initial bulk pull in Task 2, then a light weekly refresh so the
   time series stays current for forecasting (Task 7).

## 6. Task 1 submission checklist

- [x] Approved list of data sources (§2)
- [x] Clear statement of legal requirements (§3) + reusable checklist (§4)
- [x] Shared plan for Task 2 data collection (§5)
- [ ] Team review at next Scrum; save agreed version + per-source `docs/legal/` sheets
- [ ] Submit repo link in the CadetX portal

---

_Sources listed are examples of legal, API-first / openly licensed options; each
must still pass the §4 checklist before use. This document is the team's shared
reference — other specialists should append their company-specific notes._
