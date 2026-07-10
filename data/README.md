# Data

Actual data files are **git-ignored** (see repo `.gitignore`) — share large files
via the team's agreed storage. This folder documents the schema and layout.

```
data/
├── raw/         # untouched collected postings, per company: raw/google/, raw/openai/…
├── processed/   # cleaned & feature-engineered datasets
└── external/    # third-party / openly-licensed datasets (Kaggle, HF, …)
```

## Shared schema (agree before Task 2)

### Required fields — every posting, every company
| Field | Purpose |
| --- | --- |
| `job_id` | Unique identifier for each posting |
| `company_name` | Competitor comparison |
| `job_title` | Role classification & seniority detection |
| `job_description` | Main NLP input (full text) |
| `posting_date` | Hiring velocity, time-series forecasting |
| `location` | Regional patterns; remote vs onsite |
| `job_url` | Traceability & validation |

### Recommended fields — richer NLP + forecasting
`employment_type` · `department/job_category` · `seniority_level` ·
`extracted_skills` · `skill_categories` · `tech_stack_tags` ·
`cleaned_description` · `posting_month/week` · `scraped_date` · `job_status` ·
`salary_range` · `company_industry/sector` · `role_cluster_id` ·
`embedding_vector` · `skill_frequency_vector` · `emerging_skill_flag` ·
`hiring_velocity_bucket`

> **Never** store personal data: no recruiter/manager names, emails, phone numbers,
> or applicant information.
