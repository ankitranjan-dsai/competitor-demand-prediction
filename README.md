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
| 05 | Hiring Trend Analysis | Trend tables + visual summaries | ⬜ |
| 06 | Competitor Comparison | Comparison tables + visuals | ⬜ |
| 07 | Demand Forecasting | Forecast outputs + plots | ⬜ |
| 08 | Company Similarity Scoring | Similarity tables + heatmaps/network graphs | ⬜ |
| 09 | Insight Generation & Reporting | Insight report + visuals | ⬜ |
| 10 | Final Presentation & Mentor Review | Slides + finalised repo | ⬜ |
| +  | _(Optional)_ Automated Pipeline | Scheduled end-to-end pipeline | ⬜ |
| +  | _(Optional)_ Fine-Tune Skill Extraction Model | Fine-tuned model + metrics | ⬜ |

## Repository Layout

```
competitor-demand-prediction/
├── data/
│   ├── raw/          # untouched collected postings (git-ignored if large)
│   ├── processed/    # cleaned / feature-engineered datasets
│   └── external/     # third-party / public datasets
├── src/              # shared, reusable Python modules
├── notebooks/        # exploratory & per-task analysis notebooks
├── docs/             # task deliverables & documentation
│   └── legal/        # ToS / robots.txt evidence per source
├── members/          # per-specialist working folders (e.g. ankit-google/)
├── weekly-reports/   # each member's own weekly notes
└── meeting-minutes/  # Scrum Leader's weekly meeting notes
```

## Ways of Working

- **Duration:** 3 months, flexible scheduling. One task per week, or one every two weeks.
- **Meetings:** weekly on Microsoft Teams; attendance mandatory.
- **Everything lives in this repo** — data, code, notes, weekly reports, docs, minutes.
- **Submission:** at the end of each task, the repo link is submitted in the CadetX portal.

## Getting Started

```bash
git clone <repo-url>
cd competitor-demand-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Legal & Ethical First

Public data only. Review Terms of Service, honour `robots.txt`, distinguish public
vs copyrighted data, and **never collect personal or sensitive information.** See
`docs/task-01-data-sources-and-legal.md`.
