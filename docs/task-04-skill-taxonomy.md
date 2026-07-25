# Task 04 — Skill Extraction & Feature Engineering

**Team standard.** Every specialist runs the same extraction module
(`src/skills.py`) against their own company's Task 03 output. Task 06
(Competitor Comparison) and Task 08 (Similarity Scoring) compare the four
companies skill-by-skill, so a skill named `Python` by one member, `python3`
by another and `Py` by a third makes those tasks measure our vocabulary
instead of the market. This document records the taxonomy, the extraction
methods, the traps we hit in real data, and the rules the other three members
must follow.

- **Code:** [`src/skills.py`](../src/skills.py) · [`src/build_features.py`](../src/build_features.py)
- **Tests:** [`tests/test_skills.py`](../tests/test_skills.py) (68 tests; 157 in the suite)
- **Free-text evidence:** [`task-04-skill-extraction-validation.md`](task-04-skill-extraction-validation.md)
- **Google findings:** [`members/ankit-google/task-04-skill-extraction-report.md`](../members/ankit-google/task-04-skill-extraction-report.md)

```bash
python src/build_features.py --company google
python -m pytest tests/ -q
python src/validate_skill_extraction.py     # network; writes the evidence file
```

---

## 1. Three decisions that shape everything else

### 1.1 One shared taxonomy in the repo, not four keyword lists

`SKILLS` in `src/skills.py` is the single place a skill's canonical name and
category are decided: **176 skills, 108 aliases, 13 categories**. Nobody
maintains a private list.

**Why:** the deliverable of Task 06 is the *difference* between companies. Any
naming difference between members becomes a fake company difference that no
later task can distinguish from a real one. A shared table also means a
correction ("SAS is an analytics tool, not a programming language") lands for
all four datasets in one reviewable diff.

The module refuses to import if two skills claim the same alias, or if a skill
uses a category that isn't declared — a silent merge of two different skills is
worse than a crash.

### 1.2 Rules, not models — for now

No spaCy model, no embeddings, no BERT. The taxonomy is regex over a curated
table, exactly as Task 03 decided for preprocessing (§1.1 there).

**Why:** the same reproducibility argument. A downloaded model is a hidden
input that changes between releases and between members' machines, and here it
would change the *headline numbers* — the skill counts everything downstream is
built on. Rules give byte-identical output for all four datasets and every
decision is auditable in a diff.

**What we give up, honestly:** rules cannot catch a skill that is described
rather than named ("experience building large-scale distributed storage
systems"), and the taxonomy only knows what we put in it. The optional Task 11
fine-tuned extractor (BERT/RoBERTa/spaCy) is the right place to revisit this —
by then we have a rule-based baseline to measure it against, which is the only
way to know whether the model is actually better.

### 1.3 Coverage is a first-class output, not a diagnostic

Every aggregate table ships with its denominator, and
`coverage-by-*.csv` is a deliverable, not a QA artefact.

**Why:** a posting with no extracted skills is a posting we know *nothing*
about — it is not a posting without Python. Dividing by all postings silently
treats "unknown" as "no", and coverage is not constant: it varies by month, by
role and (worst) by which job board published the posting. See §6.

**Rule for Tasks 05–08: use `share_of_skilled`, never `share_of_all`.** Both
are shipped so the difference is visible, and §6.3 shows a month where the two
give opposite signs.

---

## 2. The taxonomy

### 2.1 Shape

```python
Skill(name="Spark", category="Data Engineering",
      aliases=("apache spark", "pyspark"),
      context_required=True,           # collides with ordinary English
      qualifiers=(r"\bapache\s+spark\b", r"\bpy\s?spark\b",
                  r"\bspark\s+sql\b"),
      is_concept=False)                # a named tool, not a practice
```

`name` is what appears in every output table. `key` (`spark`, `sql_server`,
`c++` → `c++`) is the stable identifier used for matrix column names, so a
display-name change doesn't break a saved matrix.

### 2.2 Categories

| Category | n | Category | n |
| --- | --- | --- | --- |
| ML / AI | 31 | Cloud Platform | 11 |
| Programming Language | 24 | Data Science Libraries | 9 |
| Database | 17 | OS / Shell | 8 |
| DevOps / Infrastructure | 16 | Facilities / Data Centre | 6 |
| Analytics / BI | 15 | Office / Productivity | 6 |
| Data Engineering | 14 | Governance / Compliance | 5 |
| Web / Frontend | 14 | | |

The order of `CATEGORIES` is meaningful: it breaks ties when a posting's
`primary_skill_category` is ambiguous, and it puts the modelling-relevant
categories first.

Three categories are excluded from `tech_stack_tags` (`NON_STACK_CATEGORIES`):
**Office / Productivity**, **Governance / Compliance**, **Facilities / Data
Centre**. Knowing a posting mentions Outlook or GDPR says nothing about the
technology being built. They stay in the long table — dropping data is worse
than labelling it — but never reach a stack feature.

### 2.3 Concepts vs tools

`is_concept=True` marks a practice rather than a product — 12 of them: Machine
Learning, Deep Learning, NLP, Computer Vision, Reinforcement Learning,
Generative AI, LLM, RAG, Recommender Systems, ETL, Data Modelling, CI/CD. They
feed `method_tags`, not `tech_stack_tags`.

**Why the split matters:** "Machine Learning" and "PyTorch" are not comparable
evidence. Almost every ML posting says "machine learning"; only the ones with a
committed stack say "PyTorch". Mixing them makes the concept dominate every
similarity score in Task 08 while carrying almost no discriminating
information.

### 2.4 Corrections to the source's own categories

The Hugging Face backfill ships its own `job_skills` / `job_type_skills`
labelling. We do not inherit it — it has real defects, and the taxonomy is
where they are repaired. Each correction is pinned by a test.

| Skill | Source category | Ours | Why |
| --- | --- | --- | --- |
| SAS | `programming` **and** `analyst_tools` | Analytics / BI | double-filed, so it double-counted |
| MongoDB, NoSQL | `programming` | Database | they are not languages |
| GDPR | `libraries` | Governance / Compliance | a regulation, not a library |
| Colocation | `cloud` | Facilities / Data Centre | a data-centre contract, not a cloud skill |
| PowerPoint, Word, Outlook | `analyst_tools` | Office / Productivity | not analysis capability |
| SQL | (with databases) | Programming Language | a language; the database is what it queries |
| SAP | — | Analytics / BI | in a *data* posting this means BW/HANA/Analytics Cloud |

**Result on the Google data: all 92 distinct source skill terms map to the
taxonomy — 0 unmapped.** `unmapped_source_terms` is written into the report
on every run, so the next member's unmapped terms show up as a number rather
than as silence.

### 2.5 Deliberately excluded terms

`EXCLUDED_TERMS` drops two source terms with a written reason:

| Term | Reason |
| --- | --- |
| `flow` | irreducibly ambiguous — Power Automate (ex-Flow), Facebook's Flow type checker, NiFi flows and the English noun all collapse to this token (8 Google postings) |
| `terminal` | an interface, not a skill; the mention says nothing about required capability |

Excluded terms are **not** counted as unmapped — they were a decision, not a
gap.

---

## 3. Three extraction paths, one output

| Path | Input | Trust | Google coverage |
| --- | --- | --- | --- |
| `source` | the collector's own skill list | high — derived from full text we never saw | 67% of rows |
| `title` | `job_title_clean` (Task 03) | high precision, low recall | 100% of rows readable |
| `text` | `cleaned_description` (Task 03 Layer B) | ours to control | 0% today — no description text |

The paths are unioned into `skills_final`, and `skill_provenance_map` records
which path(s) produced each skill.

**Why keep provenance:** the paths are not equally trustworthy, and a later
task may want to weight them or drop one. It also makes the repair measurable:
on the Google data the title path recovered 33 postings that had no source
skills at all, and added skills to 47 more.

Ambiguous names are treated differently per path. In a curated source list
`"r"` means the language, so the source path trusts it. In free text `"r"` is
almost always noise, so the guard in §4 applies. That asymmetry is deliberate
and tested.

---

## 4. The ambiguity guard

The highest-value skills have short, common-English names. Every one of these
lines is real job-posting English, and every one becomes a false skill under a
naive keyword match:

| Posting English | Would become |
| --- | --- |
| "ability to **excel** in a fast-paced environment" | Excel |
| "you will **spark** innovation across teams" | Spark |
| "**react** to changing customer needs" | React |
| "own the **go**-to-market strategy" | Go |
| "**swift** decision-making under pressure" | Swift |
| "partner with the **chef** and facilities team" | Chef |

34 skills are marked `context_required`. A free-text mention is accepted only
when one of three things holds:

1. **a skill-specific qualifier** in a ±60-char window — `golang`, `R Studio`,
   `Apache Spark`, `PySpark`;
2. **a generic frame** — `proficiency in <X>`, `<X> developer`, `<X> cluster`,
   `written in <X>`;
3. **list adjacency** — the mention sits next to a delimiter (`,` `;` `/` `&`
   `+`, or "and"/"or") **and** an unambiguous skill is nearby: `Python, R, SQL`.

Both conditions are required for rule 3. Proximity alone accepts "go to the
next level, Python"; punctuation alone accepts "we go, we ship".

Phrases that are never a skill are handled in the pattern itself rather than
the guard — `\bgo\b(?!\s*[-\s]?to[-\s]market)` — because there is nothing to
adjudicate. That matters on this data: 20% of Google's postings are
go-to-market roles.

**Rejections are returned, not discarded.** `extract_with_audit()` gives back
the ambiguous mentions it refused, which is what makes the guard measurable —
a guard whose misses nobody can see is a guard nobody can trust.

Finally, **longest match wins**: `SQL Server` never also emits `SQL`, and `C++`
never also emits `C`.

---

## 5. The employer-brand trap (read this before running your company)

Every company names its own business unit in its job titles, and for a tech
company that unit is usually also a product in our taxonomy:

> "Customer Engineer, Data Management Practice, **Google Cloud**"

That is an org label, not a statement that the candidate needs GCP. A keyword
matcher reads it as a GCP skill on **196 of Google's 848 titles**. After the
fix: **0**.

This is not a Google quirk, and it is worst where the company *is* the product —
Snowflake's postings say "Snowflake", Databricks' say "Databricks", Microsoft's
say "Azure". Left alone it manufactures exactly the differences Task 06 exists
to measure: **each company would appear to lead in its own product.**

`strip_org_segments()` splits the title on the delimiters real titles use
(`,` `/` `|` ` - `) and drops any segment that *starts with* one of that
company's org names. A segment that merely mentions a product survives — "Data
Analyst, Looker" is a real role signal. Only the **title** path is filtered: in
description prose the same word is a genuine requirement, and there the source
and text paths keep it.

> **Action for each specialist:** fill in your company's entry in
> `TITLE_ORG_SEGMENTS` before your first run, and add a test like
> `test_own_product_is_stripped_for_the_company_that_sells_it`. Placeholders
> exist for Microsoft, Amazon, Meta, NVIDIA, Snowflake, Databricks and OpenAI;
> they are starting points, not verified lists. Check
> `org_segments_stripped_from_titles` in your report — 0 usually means the
> entry is wrong, not that your company is disciplined.

---

## 6. Features produced

### 6.1 Row-level (git-ignored, `data/processed/<company>/`)

| Output | Shape | Contents |
| --- | --- | --- |
| `<company>_features.parquet` | 1 row per posting | Task 03 columns + the Task 04 skill columns |
| `<company>_skills_long.parquet` | 1 row per (posting, skill) | the extracted-skills dataset |
| `<company>_skill_matrix.parquet` | postings × skills | binary `skill_frequency_vector` |

Per-posting skill columns: `skills_source/text/title/final`,
`skill_provenance`, `skill_provenance_map`, `skill_count_final`,
`tech_stack_tags`, `method_tags`, `n_tech_stack_skills`, `n_method_skills`,
`skill_category_count`, `skill_categories_final`, `primary_skill_category`,
one `n_skills_<category>` per category, `has_ml_skill` / `has_cloud_skill` /
`has_programming_skill` / `has_facilities_skill`, `has_any_skill`,
`skill_recovered_by_title`, `n_emerging_skills`, `has_emerging_skill`.

Task 03's `skill_count` / `has_skills` (the collector's own count) are **kept
alongside** `skill_count_final` / `has_any_skill` rather than overwritten, so
the before/after of this task stays visible in the data.

Long form was chosen for the extracted-skills dataset because every downstream
task groups differently — Task 05 by month, Task 06 by company, Task 08 by
skill pair. A single wide matrix forces one of those choices on everyone.

The matrix drops skills below `min_postings=5`: a column that is 1 for a single
posting is an identifier, not a feature, and it would dominate cosine
similarity in Task 08.

### 6.2 Aggregate tables (committed, `members/<member>-<company>/task-04-tables/`)

| File | Purpose |
| --- | --- |
| `skill-frequency.csv` | counts + **both** denominators + rank |
| `skill-by-month.csv` | monthly frequency, coverage-normalised (Task 05) |
| `skill-trend.csv` | H1 vs H2 share, `relative_change`, `emerging_skill_flag` |
| `skill-cooccurrence.csv` | skill pairs + Jaccard (Task 08) |
| `skill-by-job-function.csv` | skills split by kind of work |
| `coverage-by-month.csv`, `-job-function.csv`, `-publisher.csv` | the denominators |

These are committed because they *are* the Task 04 deliverable and contain no
posting prose. Row-level output stays git-ignored under the Task 01
data-handling rules, and `personal_data_columns_present` is re-checked on every
run and written into `task-04-feature-report.json`.

Co-occurrence uses **Jaccard**, not raw pair counts: raw counts just re-rank the
popular skills, because Python co-occurs with everything.

### 6.3 Why the denominator rule is not pedantry

Google's monthly skill coverage ranges from **58.0% to 86.4%**. Take Python,
March → April 2023:

| | March | April | change |
| --- | --- | --- | --- |
| `share_of_all` (naive) | 52.6% | 40.6% | **−22.9%** |
| `share_of_skilled` (correct) | 63.8% | 70.0% | **+9.7%** |

The naive read says Python demand fell by a fifth. The corrected read says it
rose. The difference is entirely that April's coverage was 58% against March's
82%. **A forecast built on the naive series would be modelling the collector,
not the market.**

### 6.4 Trend labelling is deliberately coarse

`skill_trend_table` compares first half vs second half, requires
`min_postings=10` total support, and labels ±25% relative change as
`emerging` / `declining`; everything else is `stable` or
`insufficient_support`.

**Why not a monthly slope:** these datasets are a single calendar year from a
single source. A month-on-month regression would fit sampling noise and report
it as a trend with a confidence interval attached. Halves with a support floor
is the strongest claim this data carries, and every label is within-year by
construction — Task 05 must say so.

Half-over-half composition also shifts (see the Google report §5), so a trend
that matters should be re-checked **within** `job_function` before it is
reported as demand.

---

## 7. Checklist for each specialist

1. Fill in your company's `TITLE_ORG_SEGMENTS` entry and add a test for it.
2. `python src/build_features.py --company <you>`.
3. Check `unmapped_source_terms` in your report — add real skills to the shared
   taxonomy (with a test), and add a written reason to `EXCLUDED_TERMS` for
   anything you deliberately drop. Don't add a company-private skill list.
4. Check `org_segments_stripped_from_titles` and your title-path skill counts
   for your own product name.
5. Read `coverage-by-publisher.csv` before reporting any share. If one
   publisher's coverage is far below the rest, your "skill demand" numbers are
   partly a publisher-mix measurement.
6. Commit `task-04-tables/` and `task-04-feature-report.json`; leave
   `data/processed/` git-ignored.
7. `python -m pytest tests/ -q` must pass before you push.
