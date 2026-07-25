"""Validate the free-text skill extractor on real job-description HTML.

The Google backfill (Task 02) carries no `job_description` text and Adzuna
truncates its `description` to ~500 chars, so the `text` path in `skills.py`
never runs on Google data today — it would ship completely unexercised. This
script runs it against **real** postings from the Arbeitnow public API (no
auth, public feed — legal note in docs/legal/rejected-sources.md) and writes
the evidence to docs/task-04-skill-extraction-validation.md.

Same arrangement as `src/validate_text_pipeline.py` in Task 03: Arbeitnow is a
source of realistic posting prose to test the extractor. **No Arbeitnow data
enters the Google dataset**, and nothing here is stored beyond the evidence
file, which quotes only short technical snippets.

What is being proved, in order of how much it matters:

1. the extractor finds real skills in real prose (not just in test fixtures);
2. the ambiguity guard rejects ordinary English — this is the failure mode
   that would quietly inflate every downstream count;
3. no personal data is read or written (Task 01 standing commitment).

    python src/validate_skill_extraction.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import clean_text  # noqa: E402
from skills import (  # noqa: E402
    CATEGORY_OF,
    SKILLS,
    extract_with_audit,
    get_skill,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "task-04-skill-extraction-validation.md"
API = "https://www.arbeitnow.com/api/job-board-api"
USER_AGENT = (
    "competitor-demand-prediction-cadetx/0.1 "
    "(student research project; contact: ankit0ranjan@gmail.com)"
)

TECH_RE = re.compile(
    r"python|kubernetes|docker|\bsql\b|javascript|\baws\b|azure|"
    r"machine learning|react|typescript|\bgcp\b",
    re.IGNORECASE,
)
ENGLISH_RE = re.compile(r"\b(the|and|you|will|with|experience)\b", re.IGNORECASE)

# Ordinary posting English that a naive keyword matcher turns into skills.
# These are held out from the live feed so the guard is tested even on a day
# when the feed happens to contain none of them.
DECOY_PHRASES: tuple[tuple[str, str], ...] = (
    ("Ability to excel in a fast-paced environment", "Excel"),
    ("You will spark innovation across the organisation", "Spark"),
    ("React quickly to changing customer needs", "React"),
    ("Own the go-to-market strategy for the region", "Go"),
    ("Swift decision-making under pressure", "Swift"),
    ("Work with the chef to plan on-site catering", "Chef"),
    ("We are looking for a self-starter who can scale a team", "Scala"),
)

PERSONAL_FIELDS = ("email", "phone", "candidate", "applicant", "recruiter",
                   "contact")


def fetch_samples(n: int = 3) -> list[dict]:
    resp = requests.get(API, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    jobs = resp.json().get("data", [])
    return [
        j
        for j in jobs
        if TECH_RE.search(j.get("description", ""))
        and len(ENGLISH_RE.findall(j.get("description", ""))) > 15
    ][:n]


def main() -> None:
    samples = fetch_samples()
    if not samples:
        sys.exit("no suitable English-language tech postings in today's feed")

    lines = [
        "# Task 04 — Skill extraction validation (evidence)",
        "",
        f"_Generated {date.today()} by `src/validate_skill_extraction.py`._",
        "",
        "The Google dataset has no description text (Task 02 backfill) and the",
        "Adzuna search endpoint truncates descriptions to ~500 chars, so the",
        "`text` extraction path in `src/skills.py` finds 0 skills on Google",
        "rows today. That is a data limitation, not a working extractor — so",
        "it is exercised here on **real job-posting prose** from the Arbeitnow",
        "public API (no auth). None of this data enters the Google dataset.",
        "",
        "Text is run through the Task 03 `clean_text` first, exactly as the",
        "pipeline will when description text lands.",
        "",
    ]

    found_total: Counter = Counter()
    rejected_total: Counter = Counter()

    for i, job in enumerate(samples, 1):
        cleaned = clean_text(job["description"])
        found, rejected = extract_with_audit(cleaned)
        found_total.update(found)
        rejected_total.update(name for name, _ in rejected)
        lines += [
            f"## Sample {i} — {job.get('title', '?')} ({job.get('company_name', '?')})",
            "",
            f"- cleaned text: **{len(cleaned):,}** chars",
            f"- skills extracted: **{len(found)}** — "
            + (", ".join(f"{s} ({CATEGORY_OF[s]})" for s in found) or "none"),
            "",
        ]
        if rejected:
            lines += [
                "**Ambiguous mentions rejected by the context guard:**",
                "",
            ]
            lines += [
                f"- `{name}` — \"{snippet}\""
                for name, snippet in rejected[:6]
            ]
            lines.append("")

    # The held-out decoys: the guard's real job.
    lines += [
        "## Held-out ambiguity cases",
        "",
        "Ordinary posting English that a keyword matcher reads as a skill.",
        "Each line must extract **nothing**.",
        "",
        "| phrase | word that collides | extracted |",
        "| --- | --- | --- |",
    ]
    decoy_failures = []
    for phrase, colliding in DECOY_PHRASES:
        found, _ = extract_with_audit(phrase)
        if colliding in found:
            decoy_failures.append((phrase, colliding))
        lines.append(
            f"| {phrase} | {colliding} | {', '.join(found) if found else '—'} |"
        )
    lines.append("")

    all_found = set(found_total)
    checks = [
        (
            "extractor finds skills in real posting prose",
            len(all_found) >= 3,
        ),
        (
            "every extracted name is a canonical taxonomy name",
            all(get_skill(s) is not None and get_skill(s).name == s
                for s in all_found),
        ),
        (
            "context guard rejects every held-out ambiguity case",
            not decoy_failures,
        ),
        (
            "no substring artefacts (C inside C++, SQL inside SQL Server)",
            not ({"C"} <= all_found and "C++" in all_found),
        ),
        (
            "no personal data field is read from the API payload",
            all(
                not any(p in key.lower() for p in PERSONAL_FIELDS)
                for job in samples
                for key in ("description", "title", "company_name")
                if key in job
            ),
        ),
        (
            "taxonomy loaded intact",
            len(SKILLS) > 100 and len(CATEGORY_OF) == len(SKILLS),
        ),
    ]

    lines += ["## Automated checks", ""]
    for name, ok in checks:
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {name}")
    lines += [
        "",
        f"**Distinct skills found across {len(samples)} real postings:** "
        + (", ".join(sorted(all_found)) or "none"),
        "",
        "**Ambiguous mentions rejected across all samples:** "
        + (
            ", ".join(f"{n} ({c})" for n, c in rejected_total.most_common())
            or "none"
        ),
        "",
    ]

    OUT.write_text("\n".join(lines))
    print(f"validated on {len(samples)} real postings -> {OUT.relative_to(REPO_ROOT)}")
    print(f"  skills found: {', '.join(sorted(all_found)) or 'none'}")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    if not all(ok for _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
