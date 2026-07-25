# Task 04 — Skill extraction validation (evidence)

_Generated 2026-07-25 by `src/validate_skill_extraction.py`._

The Google dataset has no description text (Task 02 backfill) and the
Adzuna search endpoint truncates descriptions to ~500 chars, so the
`text` extraction path in `src/skills.py` finds 0 skills on Google
rows today. That is a data limitation, not a working extractor — so
it is exercised here on **real job-posting prose** from the Arbeitnow
public API (no auth). None of this data enters the Google dataset.

Text is run through the Task 03 `clean_text` first, exactly as the
pipeline will when description text lands.

## Sample 1 — Engineering Manager - Edge AI (SumUp)

- cleaned text: **4,297** chars
- skills extracted: **9** — LLM (ML / AI), Machine Learning (ML / AI), RAG (ML / AI), LangChain (ML / AI), Python (Programming Language), PyTorch (ML / AI), Hugging Face (ML / AI), Transformers (ML / AI), CI/CD (DevOps / Infrastructure)

## Sample 2 — Senior Full-Stack Engineer - SuperApp (SumUp)

- cleaned text: **3,919** chars
- skills extracted: **5** — React (Web / Frontend), JavaScript (Programming Language), TypeScript (Programming Language), Go (Programming Language), Node.js (Web / Frontend)

## Sample 3 — Senior Data Science/ML Engineer - Financial Crime (SumUp)

- cleaned text: **5,694** chars
- skills extracted: **4** — Machine Learning (ML / AI), Python (Programming Language), CI/CD (DevOps / Infrastructure), Spark (Data Engineering)

## Held-out ambiguity cases

Ordinary posting English that a keyword matcher reads as a skill.
Each line must extract **nothing**.

| phrase | word that collides | extracted |
| --- | --- | --- |
| Ability to excel in a fast-paced environment | Excel | — |
| You will spark innovation across the organisation | Spark | — |
| React quickly to changing customer needs | React | — |
| Own the go-to-market strategy for the region | Go | — |
| Swift decision-making under pressure | Swift | — |
| Work with the chef to plan on-site catering | Chef | — |
| We are looking for a self-starter who can scale a team | Scala | — |

## Automated checks

- PASS — extractor finds skills in real posting prose
- PASS — every extracted name is a canonical taxonomy name
- PASS — context guard rejects every held-out ambiguity case
- PASS — no substring artefacts (C inside C++, SQL inside SQL Server)
- PASS — no personal data field is read from the API payload
- PASS — taxonomy loaded intact

**Distinct skills found across 3 real postings:** CI/CD, Go, Hugging Face, JavaScript, LLM, LangChain, Machine Learning, Node.js, PyTorch, Python, RAG, React, Spark, Transformers, TypeScript

**Ambiguous mentions rejected across all samples:** none
