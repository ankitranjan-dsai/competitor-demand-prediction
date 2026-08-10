# Task 04 — Skill Extraction & Feature Engineering Report (Google)

**Specialist:** Ankit Ranjan · **Company:** Google (Alphabet) · **Date:** 2026-07-25

Input: 848 Google-family postings, cleaned in Task 03
(`data/processed/google/google_jobs_clean.parquet`, calendar year 2023) — two
of which are not Google; see
[C4](../../docs/corrections.md#c4--googles-posting-count-is-846-not-848).
Output: an extracted-skills dataset of **2,340 posting-skill rows covering 91
distinct skills**, plus per-posting features and eight aggregate tables.

- **Method rationale (team standard):** [`docs/task-04-skill-taxonomy.md`](../../docs/task-04-skill-taxonomy.md)
- **Code:** [`src/skills.py`](../../src/skills.py) · [`src/build_features.py`](../../src/build_features.py)
- **Tests:** [`tests/test_skills.py`](../../tests/test_skills.py) — 68 new, **157 passing** in the suite
- **Free-text evidence:** [`docs/task-04-skill-extraction-validation.md`](../../docs/task-04-skill-extraction-validation.md)
- **Machine-readable report:** [`task-04-feature-report.json`](task-04-feature-report.json)
- **Tables:** [`task-04-tables/`](task-04-tables/)

```bash
python src/build_features.py --company google
python -m pytest tests/ -q
python src/validate_skill_extraction.py
```

---

## 1. What I ran

Three extraction paths over the Task 03 output, unioned with provenance kept:

| Path | Rows with ≥1 skill | Note |
| --- | --- | --- |
| `source` — the collector's own skill list | 568 | 67% of rows; derived from full text I never received |
| `title` — `job_title_clean` | 80 | precise but sparse; **recovered 33 rows the source left empty** |
| `text` — `cleaned_description` | 0 | no description text exists in this dataset (see §7) |

**601 of 848 postings (70.9%)** end up with at least one skill, mean **3.89**
skills each. All **92** distinct source skill terms map onto the shared
taxonomy — **0 unmapped**.

The taxonomy, the ambiguity guard and the employer-brand rule are shared code
in `src/`, not company-private lists, for the same reason as Task 03: Tasks 06
and 08 compare the four companies directly, so any difference in *our* method
becomes a fake difference between *them*. My Google-specific contribution is
the rules, the defects found in real Google data, and the tests that pin them.

## 2. The defect that mattered most: Google's own org names in its titles

**256 of 848 Google titles (30%)** carry a business-unit label:

> "Customer Engineer, Data Management Practice, **Google Cloud**"
> "Scaled Abuse Analyst, Trust and Safety, **YouTube**"
> "Staff Software Engineer, **Fitbit**"

A keyword matcher reads "Google Cloud" as the GCP skill. On the raw titles that
happens **196 times — on 23% of all Google postings**.

It is not a skill requirement. It is the name of the org doing the hiring.

`strip_org_segments()` drops title segments that *start with* one of the
employer's own unit names before extraction. **GCP title hits went 196 → 0**, while the
genuine title signal survived intact (Machine Learning 62, Looker 15, SAP 3,
Computer Vision 1).

**Why this is a team-level finding, not a Google one.** It is worst where the
company *is* the product — Snowflake's titles say "Snowflake", Databricks'
say "Databricks", Microsoft's say "Azure". Uncorrected, every specialist's
company would appear to lead in its own product, which is exactly the
difference Task 06 exists to measure. Each member must fill in their own
`TITLE_ORG_SEGMENTS` entry before their first run.

**What the corrected data actually says:** GCP is named as a requirement in
only **25 postings (4.2% of skilled postings)** — far below BigQuery (62) and
Looker (71). Google names its *products* in requirements, not its umbrella
brand. The uncorrected number would have been roughly eight times too high.

## 3. Skill coverage is missing-not-at-random — for two different reasons

29% of postings yield no skills. That is not random, and the two causes need
opposite treatment.

### 3.1 Role — a true negative

| job_function | postings | coverage |
| --- | --- | --- |
| Facilities / Operations | 95 | **9.5%** |
| Technical Sales | 138 | 68.1% |
| Engineering | 231 | 72.3% |
| Analytics | 155 | 83.9% |
| Science / Research | 166 | 92.8% |

The data-centre facilities roles found in Task 03 (HVAC, electrical,
technicians) genuinely have no software skills. **This absence is data**, and
excluding those rows is correct — but it must be done by `job_function`, not by
dropping every empty-skill row.

### 3.2 Publisher — a false negative

Holding role fixed by excluding facilities:

| | postings | coverage |
| --- | --- | --- |
| via Google Careers | 47 | **19.1%** |
| every other publisher | 706 | **82.6%** |

Postings republished through Google's own careers site carry almost no skill
data (mean 1.00 skills where any exist), and they are not junk roles — the mix
is 18 Engineering, 13 Analytics, 8 Technical Sales. These are **core data roles
whose skills we are missing**, a collection artefact of how that publisher's
text was captured upstream.

**Consequence:** any "Google demand for skill X" figure is also a measurement
of publisher mix. `coverage-by-publisher.csv` is shipped so Task 05/06 can
weight or stratify rather than assume.

## 4. Why the denominator rule is not pedantry

Monthly coverage swings between **58.0%** (April) and **86.4%** (February). Take
Python, March → April 2023:

| | March | April | change |
| --- | --- | --- | --- |
| `share_of_all` (naive) | 52.6% | 40.6% | **−22.9%** |
| `share_of_skilled` (correct) | 63.8% | 70.0% | **+9.7%** |

The naive series says Python demand fell by a fifth; the corrected series says
it rose. The entire difference is April's coverage. A forecast fitted to the
naive series would be modelling the collector, not the market — so
`skill-by-month.csv` ships both columns and **Task 05/07 must use
`share_of_skilled`**.

## 5. Within-2023 trends, and a stratified check

`skill-trend.csv` compares H1 vs H2 (min 10 postings support, ±25% relative
change). Headline movers:

| Emerging *(pooled)* | H1 → H2 share | | Declining *(pooled)* | H1 → H2 share |
| --- | --- | --- | --- | --- |
| SQL | 31.8% → 50.6% | | Hadoop | 13.8% → 6.5% |
| Looker | 9.2% → 13.8% | | Go | 22.2% → 11.2% |
| BigQuery | 8.4% → 11.8% | | Scala | 12.3% → 6.5% |
| GCP | 2.7% → 5.3% | | NoSQL | 10.7% → 5.6% |
| PyTorch | 1.5% → 4.1% | | Java | 26.1% → 19.1% |

> **Corrected by Task 05 → [C2](../../docs/corrections.md#c2--five-of-task-04s-ten-headline-skill-movers-do-not-survive-stratification).**
> The shares above are correct as *pooled* shares; the labels "emerging" and
> "declining" are not. Stratified within `job_function`, **four of these ten
> survive, one reverses outright, and five have no within-segment support**:
>
> | Verdict | Skills |
> | --- | --- |
> | ✅ confirmed in every supported segment | SQL (5/5 up), Go, Java, Scala (2/2 down) |
> | ❌ **reversed** | **Looker** — pooled +4.6 pp, but down in all 3 supported functions |
> | ⚠️ no within-segment support | BigQuery (`mix_dependent`), GCP, PyTorch, Hadoop, NoSQL (one segment each) |
>
> Looker is the instructive one. It rises pooled only because the mix moves
> underneath it: Sales goes from 2.3% to 6.5% of skilled postings and
> Analytics from 17.2% to 25.6%, while Science / Research — which barely
> mentions Looker — falls from 29.7% to 23.2%. Inside Analytics it goes
> 13.6% → 11.6%, inside Sales 83.3% → 77.3%, inside Technical Sales
> 25.0% → 24.0%. Google was posting more Looker-shaped jobs, not asking for
> Looker more often.

**These are not safe on their own.** The composition of the skilled postings
shifts between halves — Analytics rises 16.9% → 25.3% of rows, Data Engineering
falls 14.6% → 7.4%, and the publisher mix changes sharply (The Muse contributes
0% of H1 and 16.7% of H2). A half-over-half move can be pure mix.

So I re-ran the headline movers **within** `job_function`:

| | Engineering H1 → H2 | Analytics H1 → H2 | Science / Research H1 → H2 |
| --- | --- | --- | --- |
| SQL | 9.3% → 16.3% | 61.4% → 75.6% | 43.4% → 73.1% |
| Hadoop | 40.0% → 15.2% | — | — |
| Go | 50.7% → 19.6% | — | — |

SQL rises in *every* function and Hadoop/Go fall sharply *within Engineering*,
so those three survive the mix explanation. The rest of the trend table should
be treated as a candidate list, not a result, until Task 05 stratifies it.

**Task 05 stratified it.** Across the 29 skills with enough support,
**8 pooled directions were overturned**
([`skill-stratified-verdicts.csv`](task-05-tables/skill-stratified-verdicts.csv)):
3 rise in every segment, 5 fall in every segment, 7 are `mix_dependent`, and 14
clear only one segment. The check above was right about SQL and Go; it was too
generous to Hadoop, which moves the way this section says but clears a single
segment, and it never reached Looker. From Task 06 onward every skill claim
travels with its stratified verdict.

**Every trend label here is within-2023, single-source.** One calendar year from
one backfill cannot distinguish a market trend from a sampling change.

## 6. Skill profile

Top skills across the 601 skilled postings (`share_of_skilled`):

| Rank | Skill | Postings | Share |
| --- | --- | --- | --- |
| 1 | Python | 373 | 62.1% |
| 2 | SQL | 255 | 42.4% |
| 3 | R | 215 | 35.8% |
| 4 | Java | 133 | 22.1% |
| 5 | C++ | 103 | 17.1% |
| 6 | Go | 96 | 16.0% |
| 7 | JavaScript | 74 | 12.3% |
| 8= | Looker | 71 | 11.8% |
| 8= | TensorFlow | 71 | 11.8% |
| 10= | Machine Learning | 62 | 10.3% |
| 10= | BigQuery | 62 | 10.3% |

By kind of work, the stacks are clearly distinct — which is what makes
`job_function` a useful split for Task 06:

| job_function | leading skills |
| --- | --- |
| Engineering | Python 105, Java 91, C++ 58, Go 56, Hadoop 44 |
| Analytics | SQL 92, Python 73, R 55, Looker 16, BigQuery 14 |
| Science / Research | Python 138, R 106, SQL 90, SAS 18, TensorFlow 18 |
| Technical Sales | Python 45, R 41, Linux 34, Java 32, Go 31 |

Strongest co-occurrences by Jaccard (`skill-cooccurrence.csv`, the Task 08
input): Ansible–Jenkins 0.83, CSS–HTML 0.78, Perl–Shell 0.55, Python–R 0.53,
Go–Java 0.53, Hadoop–NoSQL 0.52. Jaccard rather than raw counts, because raw
counts just re-rank the popular skills — Python co-occurs with everything.

## 7. Limitations

1. **The `text` path extracts nothing on this dataset.** The 2023 backfill has
   no `job_description`, and the Adzuna search endpoint truncates descriptions
   to ~500 chars of marketing intro — it never reaches the "Minimum
   qualifications" block. The extractor is therefore validated against **real**
   third-party posting prose (Arbeitnow public API, no auth, test input only,
   none of it enters the Google dataset): 15 distinct skills correctly pulled
   from 3 real postings, all 7 held-out ambiguity decoys rejected. Evidence:
   [`docs/task-04-skill-extraction-validation.md`](../../docs/task-04-skill-extraction-validation.md).
2. **The source skill list is a black box.** 2,277 of 2,340 skill rows come
   from a list the upstream dataset produced with a method I cannot inspect.
   Its recall is unknown, and §3.2 shows it varies by publisher.
3. **A rule-based extractor only knows what the taxonomy knows** — it cannot
   catch a skill that is described rather than named. Optional Task 11
   (fine-tuned BERT/spaCy) now has a measurable baseline to beat.
4. **Single year, single source.** Every trend statement is within-2023.
5. **The skill matrix covers 599 of 601 skilled postings** — two carry only
   skills below the `min_postings=5` floor and drop out of the matrix (they
   remain in the long table).

## 8. What Task 05 must do with this

1. Use `share_of_skilled`, not `share_of_all` (§4).
2. Exclude `job_function == "Facilities / Operations"` before reading skill
   demand, and say so (§3.1).
3. Stratify or weight by publisher, or state the caveat explicitly (§3.2).
4. Re-check any trend within `job_function` before calling it demand (§5).
5. Label every trend within-2023.

## 9. Deliverables

| File | Contents |
| --- | --- |
| `task-04-tables/skill-frequency.csv` | 91 skills, counts, both denominators, rank |
| `task-04-tables/skill-by-month.csv` | monthly frequency, coverage-normalised |
| `task-04-tables/skill-trend.csv` | H1/H2 shares + `emerging_skill_flag` |
| `task-04-tables/skill-cooccurrence.csv` | skill pairs + Jaccard (Task 08 input) |
| `task-04-tables/skill-by-job-function.csv` | skills by kind of work |
| `task-04-tables/coverage-by-{month,job-function,publisher}.csv` | the denominators |
| `task-04-feature-report.json` | run evidence incl. `personal_data_columns_present` |
| `data/processed/google/google_skills_long.*` | 2,340 posting-skill rows (git-ignored) |
| `data/processed/google/google_features.*` | 848 rows + skill features (git-ignored) |
| `data/processed/google/google_skill_matrix.*` | 599 × 54 binary matrix (git-ignored) |

Row-level output stays git-ignored under the Task 01 data-handling rules;
`personal_data_columns_present` is re-checked on every run and is empty.
