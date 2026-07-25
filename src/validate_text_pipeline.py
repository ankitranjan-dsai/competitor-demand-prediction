"""Validate the Layer B free-text pipeline on real job-description HTML.

The Google backfill (Task 02) carries no `job_description` text, so the text
path in `preprocess.py` would otherwise ship untested until the Adzuna pull
lands. This script exercises it against **real** postings from the Arbeitnow
public API (no auth, public feed — legal note in
docs/legal/rejected-sources.md) and writes the before/after evidence to
docs/task-03-text-pipeline-validation.md.

Arbeitnow is used here purely as a source of realistic posting HTML to test
the cleaner. No Arbeitnow data enters the Google dataset.

    python src/validate_text_pipeline.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import clean_text, tokenize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "task-03-text-pipeline-validation.md"
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
        "# Task 03 — Text pipeline validation (evidence)",
        "",
        f"_Generated {date.today()} by `src/validate_text_pipeline.py`._",
        "",
        "The Google backfill has no `job_description` text, so the Layer B",
        "text pipeline is validated here against **real job-posting HTML**",
        "from the Arbeitnow public API (no auth). This proves the cleaner and",
        "tokeniser work on genuine posting prose before the Adzuna pull lands.",
        "None of this data enters the Google dataset.",
        "",
        "**What the pipeline must survive:** HTML tags and framework attributes",
        "(`<div data-testid=...>`), HTML entities (`&amp;`, `&nbsp;`), block",
        "tags that must not glue words together, accented characters, and",
        "technical tokens that must stay intact.",
        "",
    ]

    for i, job in enumerate(samples, 1):
        raw = job["description"]
        cleaned = clean_text(raw)
        tokens = tokenize(cleaned)
        top = Counter(tokens).most_common(12)
        lines += [
            f"## Sample {i} — {job.get('title', '?')} ({job.get('company_name', '?')})",
            "",
            f"- raw length: **{len(raw):,}** chars · cleaned: **{len(cleaned):,}** chars "
            f"(**{100 - round(100 * len(cleaned) / max(len(raw), 1))}%** markup removed)",
            f"- tokens after stopword removal: **{len(tokens):,}** "
            f"({len(set(tokens)):,} unique)",
            "",
            "**Raw (first 300 chars):**",
            "",
            "```html",
            raw[:300].replace("`", "'"),
            "```",
            "",
            "**Cleaned (first 300 chars):**",
            "",
            "```text",
            cleaned[:300].replace("`", "'"),
            "```",
            "",
            f"**Top tokens:** {', '.join(f'{w} ({c})' for w, c in top)}",
            "",
        ]

    # Assertions on the aggregate result — a failure here is a real defect.
    all_clean = " ".join(clean_text(j["description"]) for j in samples)
    checks = [
        ("no HTML tags remain", "<" not in all_clean and ">" not in all_clean),
        ("no unescaped entities", "&amp;" not in all_clean and "&nbsp;" not in all_clean),
        ("no non-breaking spaces", " " not in all_clean),
        ("text is non-empty", len(all_clean.strip()) > 100),
        ("ascii-normalised", all(ord(c) < 128 for c in all_clean)),
    ]
    lines += ["## Automated checks", ""]
    for name, ok in checks:
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {name}")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"validated {len(samples)} real postings -> {OUT.relative_to(REPO_ROOT)}")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    if not all(ok for _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
