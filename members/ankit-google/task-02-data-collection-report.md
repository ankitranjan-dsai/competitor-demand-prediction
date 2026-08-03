# Task 02 — Data Collection Report (Google)

**Specialist:** Ankit Ranjan · **Company:** Google · **Date:** 2026-07-20
**Deliverables:** this report + raw collected dataset in `data/raw/google/`
(reproducible via `python src/collect_google_jobs.py hf-backfill`)

---

## 1. Sources used

| Source | Status | What it provides | Result |
| --- | --- | --- | --- |
| **Hugging Face `lukebarousse/data_jobs`** (Apache-2.0) | ✅ Collected | 785,741 data-role postings from 2023; filtered to the Google family | **848 postings**, Jan–Dec 2023 |
| **Adzuna official API** | ⏳ Approved, pending free keys | Live postings with **full description text + job URL** | Collector ready: `collect_google_jobs.py adzuna` |
| **The Muse public API** | ✅ Checked | Company-filtered postings with full text | **0 Google postings** — Google is not a Muse client |

Every source has a completed §4 legal checklist in `docs/legal/`
(`huggingface-data-jobs.md`, `adzuna.md`, `themuse.md`), and rejected options
are recorded in `docs/legal/rejected-sources.md`.

## 2. Legal checks performed

- **Google Careers direct scraping — REJECTED.** `google.com/robots.txt`
  (fetched 2026-07-20, archived in `docs/legal/evidence/`) explicitly disallows
  the paginated results path (`Disallow: /about/careers/applications/jobs/results?page=`).
  Task 01 requires ToS *and* robots.txt to both permit; this fails immediately.
  Full decision: `docs/legal/google-careers.md`.
- **LinkedIn/Indeed-derived datasets — REJECTED on provenance.** Permissive
  licence tags on re-uploads don't cure scraping that the original ToS forbade.
- **HF backfill dataset — APPROVED.** Apache-2.0 verified via the HF API;
  downloaded through the official parquet endpoint.
- **No personal data:** the dataset contains company/title/location/skills
  only. Verified no name, email, or individual-level columns exist.
- **Etiquette:** collector sends an honest User-Agent with contact email,
  rate-limits to ≤1 req/sec, and uses stable hash ids so re-pulls de-duplicate.

## 3. Collected dataset

`data/raw/google/google_jobs_hf_backfill_2023.parquet` (+ same-name CSV).
Raw data is git-ignored by design; `data-manifest.json` (committed) records
row counts, columns, date range, and the rebuild command.

- **848 postings**, 2023-01-01 → 2023-12-31, every month represented
  (min 22 in Feb, max 102 in Aug — ~~already a usable hiring-velocity
  signal~~).

  > **Corrected by Task 05 → [C1](../../docs/corrections.md#c1--raw-monthly-counts-are-not-a-hiring-velocity-signal).**
  > Neither extreme is a hiring signal. February's 22 postings arrive from
  > **6 publishers** against 23 in January and 21 in March; August's 102
  > include 24 from The Muse in its **first month** in the feed, 21 of them
  > on `2023-08-23`. All three weekly spikes in the series are
  > single-publisher batches. Velocity is only readable once a publisher
  > panel is declared, and on this data the direction of 2023 volume is
  > **not identified**.

- **Company mix:** Google/Google Inc./regional Google entities (~740), plus
  Alphabet family: Waymo (49), Verily (31), DeepMind, YouTube, Google Fiber.
  Kept the family split so Task 5/6 can analyse Alphabet-wide vs core-Google.
- **Role mix:** Data Engineer (210), Data Scientist (119), Cloud Engineer
  (116), Data Analyst (108), Software Engineer (95), ML Engineer (33)…

### Fields (normalised to the shared Task 01 schema)

| Shared-schema field | Populated? | Note |
| --- | --- | --- |
| `job_id` | ✅ | stable SHA-1 hash of source+company+title+date |
| `company_name` | ✅ | raw employer string (normalisation → Task 3) |
| `job_title` | ✅ | plus dataset's `job_title_short` role bucket |
| `job_description` | ❌ empty | **dataset limitation — see §4** |
| `posting_date` | ✅ | daily granularity, full 2023 — but **first-seen, not published**; see [C3](../../docs/corrections.md#c3--posting_date-is-an-aggregator-first-seen-date-not-a-publication-date) |
| `location` | ✅ | city/region + `country` column |
| `job_url` | ❌ empty | dataset limitation — Adzuna pull will fill this |

Recommended-field extras carried over: `employment_type`, `remote` flag,
`country`, `salary_year_avg`, **`extracted_skills`** + `skill_categories`
(pre-extracted by the dataset author — a head start for Task 4 to validate
against our own extraction), `job_via` (posting platform), `source`,
`scraped_date`.

## 4. Limitations

1. **No full description text or URLs in the backfill.** The HF dataset ships
   pre-extracted skills instead of raw text. Impact: Task 3's NLP preprocessing
   needs live-text postings → that's exactly what the approved Adzuna API
   provides. **Action:** register free Adzuna keys, then
   `python src/collect_google_jobs.py adzuna` fills the gap.
2. **Data-analytics roles only.** The dataset was collected around data/ML
   searches, so Google's non-data hiring (hardware, sales, legal…) is invisible.
   Fine for our tech/AI/data sector focus; must be stated in Task 9 insights.
3. **2023 window.** Historical backfill only; the time series needs the weekly
   Adzuna refresh (Task 01 §5 cadence) to stay current for Task 7 forecasting.
4. **Employer-name variants** (`Google`, `Google Inc.`, `GOOGLE ASIA PACIFIC
   PTE. LTD.`…) need canonicalisation in Task 3 preprocessing.
5. **The Muse dead end for Google** (0 postings) — kept in the team toolkit
   since other specialists' companies may be covered.
6. **The publisher panel is unbalanced** *(added by Task 05, [C1](../../docs/corrections.md#c1--raw-monthly-counts-are-not-a-hiring-velocity-signal))*.
   The 848 postings reach us through **96 publishers**, of which exactly
   **one** appears in all twelve months and **56** appear in a single month
   only. The feed's field of view changes month to month, so a raw count
   change confounds Google's hiring with our coverage. Any volume claim must
   name the panel treatment it was computed on.

## 5. Next steps (feeds Task 3)

- [ ] Register Adzuna keys → run live pull → first full-text Google postings.
- [ ] Team scrum: confirm other specialists adopt the same collector pattern
      (`src/collect_google_jobs.py` is written to be copy-adaptable).
- [ ] Data Quality Lead check per Task 01 §5: required fields present,
      `job_url` populated (Adzuna slice), zero personal-data columns. ✅ for
      the backfill on fields present + no personal data; `job_url` pending
      the Adzuna pull.
