# Task 03 — NLP Preprocessing & Method Selection

**Team standard.** Every specialist runs the same preprocessing module
(`src/preprocess.py`) over their own company's raw dataset. Task 06
(Competitor Comparison) and Task 08 (Similarity Scoring) compare companies
directly, so any difference in preprocessing between members shows up later as
a *fake* difference between companies. This document records which methods we
chose, which we rejected, and why — so the choices are auditable and identical
across all four datasets.

- **Code:** [`src/preprocess.py`](../src/preprocess.py)
- **Tests:** [`tests/test_preprocess.py`](../tests/test_preprocess.py) (89 tests)
- **Text-pipeline evidence:** [`task-03-text-pipeline-validation.md`](task-03-text-pipeline-validation.md)

```bash
python src/preprocess.py --company google
```

Writes `data/processed/<company>/<company>_jobs_clean.{parquet,csv}` and a
quality report to `members/<member>-<company>/task-03-quality-report.json`.

---

## 1. Three decisions that shape everything else

### 1.1 No external NLP corpora

Stopword lists and all classification rules live **inside the repo**. There is
no `nltk.download()` step and no spaCy model download.

**Why:** a downloaded corpus is a hidden input. NLTK ships several stopword
list versions, spaCy models change between releases, and a member on a
different machine or a fresh CI runner can silently produce different tokens
from the same posting. With the rules in version control, the same input row
produces byte-identical output for everyone, and any change to the vocabulary
appears as a reviewable diff.

`spacy`/`nltk` stay in `requirements.txt` — Task 04 may use them for the
optional fine-tuned skill extractor — but **the Task 03 pipeline does not
import them at runtime.**

### 1.2 Technical tokens are preserved; stemming is off

Tokenisation keeps `c++`, `c#`, `.net`, `node.js`, `scikit-learn`, `ci/cd`,
`power bi`, `pl/sql` intact, and short real skills (`r`, `go`, `ai`, `bi`,
`gcp`, `sql`, `etl`) are exempt from the minimum-length filter.

**Why we rejected stemming/lemmatising:** this project's whole downstream value
is *which named technologies* a company is hiring for. A Porter stemmer turns
`kubernetes` into `kubernete` and `analytics` into `analyt`, which breaks the
join between a posting and a skill taxonomy in Task 04. The generalisation
benefit stemming normally buys (matching `model`/`modeling`/`models`) is worth
less here than keeping product names exact.

A naive tokeniser is actively destructive on this data. Two defects the tests
now lock in permanently:

| Input | Naive result | Our result |
| --- | --- | --- |
| `C#` | dropped (length 2 < min_len) | `c#` |
| `.NET` | `net` (leading dot stripped) | `.net` |

Both are caught by `test_tokenizer_preserves_technical_tokens`.

### 1.3 Two layers, because structure and free text have different risks

| | Layer A — structured | Layer B — free text |
| --- | --- | --- |
| Input | `job_title`, `company_name`, `location`, `posting_date` | `job_description` |
| Always present? | Yes | No (the Google 2023 backfill has none) |
| Main risk | **silent misclassification** | dirty markup |
| Method | ordered rule lists + explicit suspect flags | HTML strip → entity unescape → ASCII normalise → boilerplate strip |

Layer A is where the real analytical danger lives, so it gets the most
attention below. Layer B is mechanical, but because the Google backfill carries
no description text it would otherwise ship untested — so it is validated
against **real third-party posting HTML** (see §5).

---

## 2. Layer A — structured extraction

Every rule list is **ordered**, and the order is part of the method. First match
wins, most specific rule first.

### 2.1 `company_canonical` — employer names

Regional legal entities fold together (`Google Taiwan`,
`GOOGLE ASIA PACIFIC PTE. LTD.`, `Google Germany GmbH`, `Google Czech
Republic, s.r.o.` → `Google`), but **distinct Alphabet brands stay distinct**
(Waymo, Verily, DeepMind, YouTube, Google Fiber, Mandiant, Google Operations
Center).

**Why not fold everything into one bucket:** "is Alphabet hiring more ML
engineers" and "is *Google* hiring more ML engineers" are different questions,
and Waymo's autonomy hiring has nothing to do with Google Cloud's. Collapsing
them would make the company-comparison task meaningless. Each row carries
`company_canonical`, `is_alphabet`, and `company_field_suspect`, so Task 05/06
can slice either way.

Third parties that merely *mention* the company are excluded from the dataset,
not renamed: `Geoambiente - Google Cloud Premier Partner` is a reseller, and
its hiring is not Google's demand signal.

### 2.2 `job_category` vs `job_function` — two axes, not one

This is the most important modelling decision in Task 03.

- `job_category` = **which technology domain** (AI/ML, Data Engineering, Cloud…)
- `job_function` = **what kind of work** (Engineering, Analytics, Sales…)

They are orthogonal, and a single taxonomy cannot represent both. The trigger
was a real row: *"Data Analytics Sales Specialist, Google Cloud."* A one-axis
scheme must call this either analytics (wrong — nobody is building analytics
capacity) or sales (wrong — it loses the fact that the product is analytics).
Two axes keep both facts:

```
job_category = "Data Analytics / BI"   # domain
job_function = "Sales"                 # work type
```

**Consequence for later tasks: Task 05 must filter on `job_function` before
reading a category count as capacity.** In the Google dataset, 173 of 848
postings (20%) are go-to-market roles (`Sales` + `Technical Sales`) — and inside
the largest category, `Data Analytics / BI`, it is 120 of 224 (54%). Reading
category counts alone would overstate Google's engineering and analytics
build-out by a fifth overall, and by more than double in analytics.

### 2.3 The data-centre rule must come first

`Data Center / Facilities` is tested **before any rule containing "data"**.

**Why:** ~11% of Google's postings are data-centre *facilities* roles —
mechanical engineers, electrical engineers, HVAC controls, technicians,
construction. A substring match on "data" files all 95 of them as data-science
demand. That single ordering bug would have inflated the headline signal by
roughly an eighth and pointed the forecast the wrong way. Five real titles are
pinned in `test_data_center_roles_are_not_data_roles`.

### 2.4 Seniority, and the manager trap

`SENIORITY_RULES` is ordered most-specific-first so `Senior Staff` is not
swallowed by `Staff`, and `Intern` outranks everything (an intern title may
also say "Senior" in its team name). Titles with no marker default to `Mid`
rather than `Unknown`, because on this data an unmarked title genuinely means a
mid-level IC opening — but the numeric ladder marker (`I`, `II`, `III`, `IV`) is
kept in a separate `level_marker` column rather than mixed into the label.

**The manager trap:** `Program Manager`, `Technical Program Manager`, `Product
Manager`, `Partner Technology Manager` are **individual-contributor** titles at
Google, not people-management. Treating every "Manager" as headcount ownership
is the single most common error in job-title parsing, and it would corrupt any
org-shape inference. `manager_type` is therefore one of
`people_manager` / `ic_manager_title` / `""`.

### 2.5 Flag, don't fix

Where the source data is genuinely malformed we record it rather than guess:

| Column | Meaning |
| --- | --- |
| `title_suspect` | title has no recognisable role noun, or is all-lowercase (e.g. `2023 summer intern data`) |
| `company_field_suspect` | a job title appears to have leaked into the employer column |

**Why:** silently repairing a broken row hides a collection defect and makes
the dataset look healthier than it is. Both flags are counted in the quality
report so the rotating Data Quality Lead can see them, and downstream tasks can
exclude those rows deliberately.

### 2.6 Titles, locations, dates

- **Titles** are normalised for grouping (accents, whitespace, zero-width
  characters) and trailing intake-cohort noise is dropped — `", Summer 2023"`
  and a bare `", 2023"`. Without this, one recurring role fragments into a new
  title every intake and its trend line disappears.
- **Locations** split into `city` / `region` / `location_country` plus
  `location_is_remote` and `location_multi` booleans. Accents are normalised so
  `Zürich` and `Zurich` group as one city instead of two. Four source shapes are
  handled explicitly, each of which produced a wrong country before it was
  handled (see §2.7): `"City, Region, Country"`, `"City, ST"` (US), `"City -
  Country"` / `"City Country"` with no comma, and a `"(+N others)"`
  multi-location suffix.
- **Dates** derive `posting_month`, `posting_week` (ISO), `posting_quarter`,
  `posting_dow`. Task 07 forecasts on monthly and weekly counts, so these are
  computed once here rather than re-derived per notebook.
- **Skills** (where the source provides them) are parsed from the stringified
  list and **de-duplicated** — the source repeats entries
  (`['r','python','sas','sas','sql']`), which would double-count features.

### 2.7 Country resolution is not a string split

`"Sunnyvale, CA"` is the single most dangerous location string in this data. A
plain two-part split files `CA` as the **country**, which does three bad things
at once: it fragments the United States into as many pseudo-countries as there
are states, it removes the US from any country ranking entirely, and `CA`
collides with Canada's ISO-3166 code so the two silently merge.

In the Google dataset this affected 226 of 848 rows and made the country
ranking not merely imprecise but **inverted** — the US looked absent when it is
in fact the largest market at 27% of postings.

The module therefore resolves countries rather than splitting strings: a
trailing two-letter token in `US_STATES` means `location_country = "United
States"` with the state in `region`; `COUNTRY_ALIASES` unifies spellings
(`UK` → `United Kingdom`) so one country cannot appear twice in a comparison;
and `COUNTRY_SUFFIXES` recovers the country from un-delimited strings
(`"Riyadh Saudi Arabia"`). The `(+N others)` marker is stripped into the
`location_multi` boolean, because a posting open in six offices is a different
demand signal from one open in a single city.

**Each specialist must check their own `top_countries` output.** These rules are
tuned to a US-formatted source; a European-formatted feed may need different
handling, and a two-letter token that is a US state code in one feed may be a
province code in another.

---

## 3. Layer B — free-text pipeline

Applied to `job_description` in a fixed order:

1. **Block tags → space.** `<br>`, `</p>`, `</li>`, `</div>`, `</hN>` become
   spaces *before* the general tag strip. Otherwise `<p>one</p><p>two</p>`
   collapses to `onetwo` and invents a token that was never in the posting.
2. **Strip remaining tags** (including framework attributes such as
   `<div data-testid=...>`).
3. **Unescape HTML entities** — `&amp;`, `&nbsp;`, `&#39;`.
4. **Normalise to ASCII.** Typographic punctuation (em dash, curly quotes,
   ellipsis, bullets), non-breaking and zero-width spaces, accents, and emoji
   are mapped to ASCII or to a space. Currency symbols become codes
   (`€` → `EUR`) so salary text survives instead of being deleted. Non-ASCII
   characters become a **space**, never nothing, so they cannot glue two words
   together.
5. **Strip equal-opportunity / EEO boilerplate** from the first match to the
   end of the text.

Step 5 deserves its own note. EEO blocks are legally required, near-identical
across every posting, and full of words (`gender`, `orientation`,
`regardless`, `identity`) that any TF-IDF or embedding model will happily treat
as signal. Left in, they add a large constant block of text to every document —
depressing the weight of the terms we actually care about — and they are the one
part of a posting that discusses protected characteristics. Removing them is
both better modelling and better ethics.

The stopword list extends the usual function words with job-posting boilerplate
that carries no demand signal: `requirements`, `qualifications`,
`responsibilities`, `candidate`, `benefits`, `apply`, `experience`, `years`.

**Vectorisation is deliberately deferred to Task 04.** Task 03's contract is
clean text plus tokens; choosing TF-IDF vs embeddings belongs with the skill
extraction it feeds.

---

## 4. Quality report

Every run writes a JSON quality report — the Data Quality Lead's evidence for
that week, and the artefact to diff when a rule changes:

`rows_in` · `rows_alphabet` · `rows_excluded_non_alphabet` ·
`excluded_companies` · `company_field_suspect_rows` · `rows_with_description` ·
`rows_missing_location` · `rows_remote` · `rows_multi_location` ·
`personal_data_columns_present` · `title_suspect_rows` ·
`company_canonical_counts` · `job_category_counts` · `job_function_counts` ·
`seniority_counts` · `top_countries` · `rows_with_skills` ·
`mean_skills_per_posting`

`personal_data_columns_present` is a **standing privacy check**, not a
statistic. It scans the output columns for anything matching
`name|email|phone|candidate|applicant|recruiter` (excluding the employer-name
columns) and must stay empty. It exists so that the Task 01 commitment —
*never collect personal or sensitive information* — is re-verified on every
single run instead of being asserted once in a document.

---

## 5. How we know the rules are right

**89 unit tests**, written from titles and locations that actually appear in the
collected data. They are regression locks, not coverage decoration: each one
pins a defect that was found and fixed, so a later refactor cannot quietly
reintroduce it. The data-centre trap, the IC-manager trap, seniority
precedence, `.NET`/`C#` survival, EEO removal, the two-axis category/function
split, US-state-as-country, country-spelling unification, and skill
de-duplication are all pinned.

```bash
python -m pytest tests/ -q
```

**Text-pipeline validation on real HTML.** Because the Google backfill has no
description text, `src/validate_text_pipeline.py` runs Layer B against genuine
job-posting HTML from the Arbeitnow public API (no auth) and writes before/after
evidence plus five automated checks — no HTML tags remain, no unescaped
entities, no non-breaking spaces, non-empty output, fully ASCII-normalised. The
script exits non-zero if any check fails. **None of that data enters any
company dataset**; it is test input only.

That run is what surfaced the ASCII defect: real postings contain em dashes,
`€`, and emoji in benefits lists (`🌎` remote, `🏖` leave), which accent
normalisation alone does not touch. The `normalize_typography` step exists
because of it.

---

## 6. Known limitations

1. **No description text for the Google 2023 backfill.** Layer B is validated
   but idle on those rows; `has_description` is `False` for all 848.
   Text-derived features arrive with the Adzuna live pull.
2. **Rule-based classification has a ceiling.** 14 of 848 titles land in
   `Other`. Rules are transparent and reviewable, which is why we start here;
   the optional fine-tuned classifier is the upgrade path if `Other` grows.
3. **English-only.** Non-English postings would tokenise poorly. The current
   collection is English; a non-English source needs a language filter first.
4. **Seniority defaults to `Mid`** for unmarked titles. Reasonable for this
   data, but it means `Mid` is a mixture of true mid-level roles and unlabelled
   ones — treat that bucket as a floor, not a measurement.
5. **`job_function` rules are Google-flavoured.** `Customer Engineer` is
   pre-sales *at Google*; another company may use the title differently. Each
   specialist should confirm the function rules against their own titles and
   add a test rather than editing a rule silently.
6. **Location rules assume a US-formatted source.** `US_STATES` and
   `COUNTRY_SUFFIXES` cover the shapes this feed emits. A feed that writes
   `"Ontario, ON"` or postal codes needs additional rules — check
   `top_countries` for anything that is not a country before trusting it.

---

## 7. What other specialists need to do

1. Collect your raw dataset to the shared Task 02 schema.
2. Run `python src/preprocess.py --company <yourcompany>
   --infile data/raw/<yourcompany>/<file>.parquet`.
3. Read your `task-03-quality-report.json`. Check `job_category_counts` and
   `job_function_counts` for anything implausible — a large `Other` bucket
   means the taxonomy is missing a rule for your company, not that your data is
   bad. Check `top_countries` for entries that are not countries; that is how
   the US-state bug was caught.
4. **If a rule is wrong for your company, add a test with a real title from
   your data, then change the rule.** The test is what stops the fix from
   regressing when someone else edits the same list.
5. Never edit the stopword list or a rule without running
   `python -m pytest tests/ -q` — the tests exist to protect everyone else's
   already-processed output.
