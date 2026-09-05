# Competitor Demand Prediction Using Job Postings

An NLP + forecasting system that turns public job-posting data into competitive
intelligence — predicting competitor demand, skill trends, and market direction
across the tech / AI / data sector.

> **One sector · four companies · one unified model.**
> Job postings are a *forward-looking* signal — they reveal future intent, not
> past performance. This project forecasts competitor demand and emerging tech
> trends from that signal.

---

## Team & Company Ownership

Four data scientists, each the specialist for one company. All four datasets feed
one shared hiring-intelligence model.

| Specialist | Company | Status |
| --- | --- | --- |
| **Ankit Ranjan** | **Google** | Active |
| _TBD_ | _TBD (unique)_ | Open |
| _TBD_ | _TBD (unique)_ | Open |
| _TBD_ | _TBD (unique)_ | Open |

_Companies to choose from (examples): Google, Microsoft, OpenAI, Snowflake,
Databricks, NVIDIA, Meta, Anthropic. Each member must pick a **unique** company._

> While the other three seats are open, Task 06 builds the comparison set
> itself — Meta, Microsoft, Snowflake, Databricks and NVIDIA alongside Google.
> Those five have no specialist, so their brand ladders and exclusions were
> written by one person and are published as an auditable table
> (`members/ankit-google/task-06-tables/employer-matching-audit.csv`) for the
> incoming specialists to overrule. OpenAI and Anthropic were screened out:
> this dataset holds 14 and 9 of their 2023 postings.

## Rotating Weekly Roles

Roles rotate every week; decisions/blockers/progress are documented in
`meeting-minutes/` each week.

- **Scrum Leader** — runs the weekly Teams meeting, coordinates tasks, main point of contact.
- **Data Quality Lead** — datasets clean, consistent, validated across all four companies.
- **Documentation Lead** — meeting minutes and project notes.
- **Technical Lead** — code structure and technical decisions.

## The 10 Tasks

| # | Task | Deliverable | Status |
| --- | --- | --- | --- |
| 01 | Understanding Data Sources & Legal | Approved source list + legal review + collection plan | ✅ Drafted → `docs/task-01-data-sources-and-legal.md` |
| 02 | Data Collection | Data Collection Report + raw dataset | ✅ Google done → `members/ankit-google/task-02-data-collection-report.md` (848 backfill postings; Adzuna live pull pending keys) |
| 03 | NLP Preprocessing & Method Selection | Cleaned text + documented workflow | ✅ Google done → `docs/task-03-preprocessing-methods.md` (team standard) + `members/ankit-google/task-03-preprocessing-report.md` (848 rows cleaned, 89 tests) |
| 04 | Skill Extraction & Feature Engineering | Extracted-skills dataset + feature tables | ✅ Google done → `docs/task-04-skill-taxonomy.md` (shared taxonomy + method) + `members/ankit-google/task-04-skill-extraction-report.md` (2,340 skill rows, 91 skills, 157 tests) |
| 05 | Hiring Trend Analysis | Trend tables + visual summaries | ✅ Google done → `docs/task-05-trend-analysis-methods.md` (team standard) + `members/ankit-google/task-05-trend-report.md` (19 trend tables, 8 figures, 214 tests) |
| 06 | Competitor Comparison | Comparison tables + visuals | ✅ Google done → `docs/task-06-competitor-comparison-methods.md` (team standard) + `members/ankit-google/task-06-comparison-report.md` (six companies, 33 comparison tables, 9 figures, 359 tests) |
| 07 | Demand Forecasting | Forecast outputs + plots | ✅ Google done → `docs/task-07-demand-forecasting-methods.md` (team standard) + `members/ankit-google/task-07-forecast-report.md` (15 tables, 8 figures, 425 tests; no model beats persistence, **max useful horizon 0**, forecast published marked unsupported) |
| 08 | Company Similarity Scoring | Similarity tables + heatmaps/network graphs | ✅ Google done → `docs/task-08-company-similarity-methods.md` (team standard) + `members/ankit-google/task-08-similarity-report.md` (16 tables, 8 figures, 478 tests; five metrics against two nulls, **2 of 15 ranks identified**, Google–Meta the only robust pair, trajectory similarity refused) |
| 09 | Insight Generation & Reporting | Insight report + visuals | ✅ Google done → `docs/task-09-insight-generation-methods.md` (team standard) + `members/ankit-google/task-09-insight-report.md` (25 tables, 8 figures, 557 tests; 436 claims generated and gated, **120 publishable — a 27.5% yield**, salary benchmarking refused, investors get nothing) |
| 10 | Final Presentation & Mentor Review | Slides + finalised repo | ✅ Google done → `docs/task-10-final-presentation-methods.md` (team standard) + `members/ankit-google/task-10-presentation-report.md` (19-slide deck compiled from the claim ledger, 11 tables, 8 figures, 647 tests; 26 rules over the deck and a 21-question mentor bank in which **11 answers are refusals**; a workspace audit — 37 checks, all passing — that failed four of them on its first run and drove the repair) |
| 11 | _(Optional)_ Automated Pipeline | Scheduled end-to-end pipeline | ✅ Google done → `docs/task-11-automated-pipeline-methods.md` (team standard) + `members/ankit-google/task-11-pipeline-report.md` (16 stages declared as data, run order derived not written; 13 rules over 42 checks, 11 tables, 5 figures, 747 tests; found **nine committed paths with two writers apiece** — one of them the Google feature frame, where the wrong stage order silently reverses C4 across every downstream table without failing anything — and a drift check that reads the deck's 18 registered facts at three refs, so a stale committed number can be named rather than merely detected; GitHub Actions `check` on push and `verify` weekly, neither of which commits — and the first run on a machine that was not the author's found an **undeclared Python floor**, four Task 09 lines that need 3.12 to parse, now pinned by `PYTHON_FLOOR` and a scan that tokenises without importing) |
| 12 | _(Optional)_ Fine-Tune Skill Extraction Model | Fine-tuned model + metrics | ✅ Google done → `docs/task-12-skill-model-methods.md` (team standard) + `members/ankit-google/task-12-skill-model-report.md` (a learned title→skill model, deterministic offline scikit-learn, held to the 176-skill rule taxonomy on the only contest the data allows — predict the collector's source skills from the job title, out of fold on the same 554 postings; 6 tables, 4 figures, 23 tests (775 tests in the suite); it **wins by +0.67 micro-F1** (0.6874 vs 0.0175) and is **refused adoption** on three facts read off the data — it learns **32 of 176** skills, its labels are the collector's source list not posting text, and **0 of 848** Google postings carry description text, so the text path a fine-tune would improve has no input; title literal coverage is 0.145, so the recall is earned by association ("data scientist" → Python) not literal matching, and the model is published as association weights, not a pickle; deliberately **not** a registered pipeline stage because it is not adopted, and `src/skills.py` is retained) |

## Repository Layout

```
competitor-demand-prediction/
├── data/
│   ├── raw/          # untouched collected postings (git-ignored if large)
│   ├── processed/    # cleaned / feature-engineered datasets
│   └── external/     # third-party / public datasets
├── src/              # shared, reusable Python modules
├── notebooks/        # exploratory & per-task analysis notebooks (2)
├── docs/             # task deliverables & documentation
│   ├── corrections.md  # claims a later task disproved, and the evidence
│   └── legal/        # ToS / robots.txt evidence per source
├── members/          # per-specialist working folders (e.g. ankit-google/)
├── weekly-reports/   # each member's own weekly notes (6)
└── meeting-minutes/  # Scrum Leader's weekly meeting notes
```

## Ways of Working

- **Duration:** 3 months, flexible scheduling. One task per week, or one every two weeks.
- **Meetings:** weekly on Microsoft Teams; attendance mandatory.
- **Everything lives in this repo** — data, code, notes, weekly reports, docs, minutes.
- **Submission:** at the end of each task, the repo link is submitted in the CadetX portal.
- **Corrections are recorded, not overwritten.** A task is submitted the moment
  it closes, so when a later task disproves an earlier claim the original
  wording stays put and gains a pointer to
  [`docs/corrections.md`](docs/corrections.md). Ten claims have been
  corrected so far — three by Task 05, one by Task 06, one by Task 07, two by
  Task 08, one by Task 09, one by Task 10, one by Task 11 — and each one is
  checked against its evidence by `tests/test_corrections.py`. Seven are claims
  about the data; two are claims about the project — one where a handover
  section named the next task wrong and handed it instructions written for a
  task that does not exist, and one where that same section predicted its own
  successor could not be checked by a test. The last is a third kind, and it is
  not about whether a number is right but about **when** it is right: a seed
  pins the resampling and not the frame being resampled, so a file with two
  writers has no content of its own until something fixes the order.

## Getting Started

**Python 3.11 or newer** — 3.11 is the floor the CI installs and the suite is
asserted against. It is declared in `PYTHON_FLOOR` (`src/pipeline.py`) and
enforced by a scan that tokenises every source without importing it, so a
developer on 3.11 is told *before* pushing — not after — that four Task 09 lines
use 3.12-only syntax and are guarded accordingly. Library versions are pinned in
`requirements.txt` for the same reason: the committed artefacts only rebuild
byte-identically against the versions that built them.

```bash
git clone <repo-url>
cd competitor-demand-prediction
python3.11 -m venv .venv && source .venv/bin/activate   # 3.11 is the floor
pip install -r requirements.txt
```

Assert the suite on the floor without a local 3.11 install:

```bash
uv run --python 3.11 --with-requirements requirements.txt python -m pytest   # 747 passing
```

## Legal & Ethical First

Public data only. Review Terms of Service, honour `robots.txt`, distinguish public
vs copyrighted data, and **never collect personal or sensitive information.** See
`docs/task-01-data-sources-and-legal.md`.
