# Task 03 — NLP Preprocessing Report (Google)

**Specialist:** Ankit Ranjan · **Company:** Google (Alphabet) · **Date:** 2026-07-25

Input: 848 Google-family postings collected in Task 02
(`data/raw/google/google_jobs_hf_backfill_2023.parquet`, calendar year 2023).
Output: 848 rows × 44 columns in `data/processed/google/google_jobs_clean.{parquet,csv}`.

- **Method rationale (team standard):** [`docs/task-03-preprocessing-methods.md`](../../docs/task-03-preprocessing-methods.md)
- **Code:** [`src/preprocess.py`](../../src/preprocess.py) · **Tests:** 89 passing
- **Quality report:** [`task-03-quality-report.json`](task-03-quality-report.json)
- **Text-pipeline evidence:** [`docs/task-03-text-pipeline-validation.md`](../../docs/task-03-text-pipeline-validation.md)

```bash
python src/preprocess.py --company google
python -m pytest tests/ -q
python src/validate_text_pipeline.py
```

---

## 1. What I ran

The preprocessing module is company-agnostic and lives in shared `src/`, not in
this folder, because Tasks 06 and 08 compare companies directly — if each
specialist cleaned their data differently, those differences would surface later
as fake differences between companies. My Google-specific contribution is the
**rules and the tests**, derived from real Google titles and now pinned so
nobody's later edit can silently break them.

Two layers: Layer A derives structure from fields that are always present
(title, company, location, date); Layer B is the free-text pipeline for
`job_description`. Full method-selection rationale — including why no external
NLP corpora and why stemming is off — is in the team doc.

## 2. Four defects I found in my own data

These are the substance of this task. Each one would have corrupted the
forecast in Task 07, and each is now locked by a test.

### 2.1 Data-centre facilities roles counted as data-science demand

**95 of 848 postings (11%)** are data-centre *facilities* roles: mechanical and
electrical engineers, HVAC controls, technicians, construction. A substring
match on "data" files every one of them as data demand.

Google is unusual here — it runs its own data centres, so its posting feed mixes
physical-infrastructure hiring with software hiring in a way that a
software-only competitor's feed does not. Left uncorrected this inflates
Google's apparent data-science demand by roughly an eighth and would have made
Google look like it was scaling analytics when it was actually building
buildings. `Data Center / Facilities` is now the **first** classification rule
tested, ahead of every rule containing "data".

### 2.2 A fifth of postings are go-to-market, not capacity

Found while auditing the `Data Analytics / BI` bucket, the largest category at
224 postings: **120 of those 224 (54%) are go-to-market roles**, not analytics
roles. *"Data Analytics Sales Specialist, Google Cloud"* is someone **selling**
analytics products, not analytics capacity being built.

Rule reordering could not fix this, because the domain (analytics) and the type
of work (sales) are both true and independent. So I added a second orthogonal
axis, `job_function`, instead of destroying one of the two facts:

| | count | share |
| --- | --- | --- |
| Engineering | 231 | 27% |
| Science / Research | 166 | 20% |
| Analytics | 155 | 18% |
| **Technical Sales** | **138** | **16%** |
| Facilities / Operations | 95 | 11% |
| **Sales** | **35** | **4%** |
| Other / Product / Support / Marketing | 28 | 3% |

**173 postings (20%) are go-to-market.** Google Cloud's `Customer Engineer` is
a pre-sales role, and there are a lot of them. **Task 05 must filter on
`job_function` before reading any category count as capacity** — otherwise
Google's engineering and analytics build-out is overstated by a fifth.

### 2.3 The "Other" bucket was hiding taxonomy gaps

26 rows initially fell through to `Other`. Reading them individually — rather
than accepting the bucket — showed four missing rules and one genuine data
defect:

| Found in "Other" | Fix |
| --- | --- |
| `RF Hardware Engineer`, `Power Management Silicon Validation Engineer` | new `Hardware / Silicon` category |
| `Senior clinical informatics data architect` | `data architect`, `informatics` → Data Engineering |
| `Software Test Engineer, FitbitOS Release Testing` | `test engineer`, `sdet`, `qa` → Software Engineering |
| `Student Researcher, PhD` | new `Research / Academic` category |
| `2023 summer intern data` | not a taxonomy gap — a malformed title, now flagged |

`Other` fell from 26 to 14. The remaining 14 are mostly `Application Engineer`
variants — a genuinely ambiguous Google title that I have deliberately left
unclassified rather than forcing into a bucket.

### 2.4 The United States had disappeared from the country ranking

US postings arrive as `"Sunnyvale, CA"`, and the original two-part split filed
the trailing token as the **country**. The result: the US fragmented into 17
pseudo-countries (`CA` 111, `GA` 19, `TX` 18, `NY` 16, …) and never appeared in
a country ranking, while `CA` silently collided with Canada's ISO code. Three
smaller defects sat in the same field: `UK` and `United Kingdom` counted as two
countries, `"Riyadh Saudi Arabia"` and `"Dubai - United Arab Emirates"` have no
comma so the whole string became the country, and the source's multi-location
marker turned `"Atlanta, GA (+5 others)"` into a country called `GA (+5 others)`.

I had already written up this dataset as under-sampling the US before checking
the field. It does the opposite:

| | before | after |
| --- | --- | --- |
| distinct "countries" | 83 | 46 |
| United States | absent from ranking | **229 (27%) — largest market** |
| United Kingdom | `UK` 14 + `United Kingdom` 2 | 16 |

The US is Google's largest hiring market in this data by a factor of three over
the next country. Any Task 06 geographic comparison built on the old field would
have been not just imprecise but backwards. `location_multi` is now a separate
boolean (21 rows) because a posting open in six offices is a different demand
signal from one in a single city.

The lesson I am carrying forward: a field that *parses* without error is not a
field that is *correct*. This one produced clean-looking output for 848 rows and
was wrong for 226 of them.

## 3. Other Google-specific findings

**Employer names.** 848 rows carry **30 distinct employer strings**, which
canonicalise to 9. Regional legal entities (`Google Taiwan`, `GOOGLE ASIA
PACIFIC PTE. LTD.`, `Google Germany GmbH`, `Google Czech Republic, s.r.o.`)
fold into `Google` = 742 rows, while
distinct Alphabet brands stay separate: Waymo 51, Verily 38, YouTube 6, Google
Operations Center 3, Google Fiber 3, DeepMind 3, Mandiant 1. I kept them
distinct because Waymo's autonomy hiring and Google Cloud's hiring answer
different questions, and Task 06 needs to be able to ask either.

**One row is not Google at all.** `Geoambiente - Google Cloud Premier Partner`
is a reseller whose name merely mentions Google. It is flagged
`is_alphabet = False` rather than deleted, so the exclusion is visible and
reversible.

**The manager trap.** 60 titles contain "Manager", but **17 of them are
individual-contributor titles** (`Program Manager`, `Technical Program
Manager`, `Partner Technology Manager`). Only 43 are people-management. Reading
all 60 as headcount ownership would have overstated Google's management layer
by 40%.

**Truncated titles — found by tightening the QA flag.** My first
`title_suspect` rule fired on 14 rows, but ~6 were legitimate roles (`Head of
Product Data Science`, `Data Analytics Apprenticeship`). A flag with a 40%
false-positive rate is a flag the Data Quality Lead learns to ignore, so I
tightened the role-noun list and added a truncation check. The flag now fires on
21 rows and every one is genuinely defective — including **16 postings whose
titles are cut off mid-phrase** by the aggregator (`Customer Engineer, Data
Infrastructure, Google Cloud (Ukrainian...`). That defect was invisible before.
It does not usually break classification (enough of the title survives), but it
does break grouping: two truncations of the same role read as two different
roles. The other 5 are one aggregator-spam listing (`Google Recruitment 2023 -
Work From Home - Data Analysis`), two mangled rows (`2023 summer intern data`),
one non-English title (`Ingénieur Data Sciences`), and one with no role noun at
all (`Marketing Data`).

**Seniority.** Mid 683 · Senior 69 · Intern 31 · Lead 27 · Staff 19 ·
Director+ 8 · Senior Staff 7 · Junior 3 · Principal 1. The `Mid` bucket is
large because unmarked titles default to it — treat it as a floor, not a
measurement.

**Geography (after the fix in §2.4).** 46 countries. United States 229,
Singapore 73, India 65, Chile 55, Brazil 36, Ireland 34, Taiwan 27, Poland 27,
South Korea 22. 29 postings are remote, 21 are open in multiple locations, and 4
have no location at all. The long tail of small-count countries is genuine —
Google posts one or two roles in many markets — so Task 06 should aggregate
below a threshold rather than compare 46 countries directly.

**Skills.** 572 of 848 rows (67%) carry a pre-extracted skill list, mean 4.0
skills per posting after de-duplication. Top: python 373, sql 255, r 215, java
133, c++ 103, go 95, javascript 74, tensorflow 71, looker 71, bigquery 62,
hadoop 58, scala 54, linux 54. De-duplication mattered — the source repeats
entries (`['r','python','sas','sas','sql']`), which would double-count in Task
04 features. These are the *source's* skills; Task 04 will extract my own from
description text and can then be compared against these as a baseline.

## 4. Privacy check

`personal_data_columns_present` is **empty**, as required. The check runs on
every single execution rather than being asserted once in a document, so the
Task 01 commitment — never collect personal or sensitive information — is
re-verified continuously. No candidate, applicant, recruiter, email, or phone
column exists anywhere in the 44-column output.

## 5. Layer B status — validated, not yet exercised

`rows_with_description = 0`. The Task 02 backfill carries no description text,
so **every text-derived column is currently empty by construction** and
`has_description` is `False` for all 848 rows.

Rather than ship an untested code path, I validated Layer B against **real
job-posting HTML** from the Arbeitnow public API (no auth, test input only —
none of it enters the Google dataset). Five automated checks pass: no HTML tags
remain, no unescaped entities, no non-breaking spaces, output non-empty, output
fully ASCII-normalised.

That run earned its keep — it failed first. Real postings contain em dashes,
`€`, and emoji in benefits lists (`🌎` remote, `🏖` leave) that accent
normalisation alone does not touch, so cleaned text was leaking non-ASCII
characters that would have produced two grouping keys for one value. The
`normalize_typography` step exists because of that failure.

## 6. Limitations carried into Task 04

1. **No description text yet.** Text features wait on the Adzuna live pull
   (blocked on free API keys — see the Task 02 report). Skill extraction in Task
   04 therefore starts from the source's `extracted_skills` for the 67% of rows
   that have them.
2. **Single year, single source.** All 848 rows are 2023 from one Apache-2.0
   dataset. Enough for method development; not enough for a trend claim. Task
   05 conclusions must be labelled as within-2023.
3. **14 rows remain unclassified** (`Application Engineer` variants). Left
   deliberately in `Other` rather than force-fitted.
4. **16 truncated titles** will fragment grouping slightly. Cannot be repaired
   from the source; flagged so Task 05 can exclude them if it groups on title.
5. **Rules are Google-flavoured.** `Customer Engineer` is pre-sales at Google;
   another specialist's company may use it differently. They should add a test
   with a real title from their own data before changing a shared rule.

## 7. Next

- Task 04 — skill extraction & feature engineering, on the cleaned dataset.
- Register free Adzuna keys (`ADZUNA_APP_ID` / `ADZUNA_APP_KEY` in `.env`), then
  `python src/collect_google_jobs.py adzuna` to obtain description text and
  posting URLs; re-run preprocessing to activate Layer B on real Google rows.
- Submit the repo link in the CadetX portal for Task 03.
