"""Task 06 — build the competitor set through the same pipeline as Google.

Runs the shortlisted competitors from the approved Hugging Face backfill
(`lukebarousse/data_jobs`, Apache-2.0 — see docs/legal/huggingface-data-jobs.md)
through Task 03's preprocessor and Task 04's skill extractor, unchanged, so
Task 06 compares companies rather than comparing pipelines.

    python src/build_competitor_set.py                # screen + build all
    python src/build_competitor_set.py --company meta # one company
    python src/build_competitor_set.py --screen-only  # feasibility table only

Why this is a separate runner
-----------------------------
`collect_google_jobs.py`, `preprocess.py` and `build_features.py` are the
Google specialist's Task 02–04 runners and they write into
`members/ankit-google/`. A competitor is *not* a specialist's own company: no
one owns Microsoft in this team, its per-task reports are not deliverables, and
writing them under a member folder would say otherwise. So this runner calls
the same library functions and writes only to `data/processed/<company>/`,
which stays git-ignored. The committed output of Task 06 is the comparison,
plus the three audit tables that make the company selection reviewable.

**No new data source.** Every posting here comes from the parquet Task 02
already downloaded and Task 01 already approved; the only new thing is which
employer strings are selected out of it. The legal position is recorded in
docs/legal/huggingface-data-jobs.md §"Scope extension (Task 06)".
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

import companies as co
import preprocess as pp
import skills as sk

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The full backfill parquet, cached by `collect_google_jobs.py hf-backfill`.
#: It is the whole 785k-row dataset, not a Google subset, which is why Task 06
#: can select competitors from it without a new download.
SOURCE = REPO_ROOT / "data" / "raw" / "google" / "_hf_data_jobs_full.parquet"

TABLES = REPO_ROOT / "members" / "ankit-google" / "task-06-tables"

SOURCE_LABEL = "hf:lukebarousse/data_jobs (Apache-2.0)"


def stable_job_id(source: str, *parts: str) -> str:
    """Task 02's id, reproduced exactly so ids are comparable across tasks."""
    digest = hashlib.sha1("|".join([source, *[p or "" for p in parts]]).encode())
    return f"{source}-{digest.hexdigest()[:12]}"


def to_shared_schema(block: pd.DataFrame) -> pd.DataFrame:
    """HF columns -> the team's shared schema.

    Mirrors `collect_google_jobs.collect_hf_backfill`; a test asserts this
    produces the committed Google backfill row-for-row rather than trusting the
    two copies to stay in step.
    """
    out = pd.DataFrame(
        {
            "job_id": [
                stable_job_id("hf", c, t, str(d))
                for c, t, d in zip(block.company_name, block.job_title,
                                   block.job_posted_date)
            ],
            "company_name": block.company_name,
            "job_title": block.job_title,
            "job_description": "",   # dataset limitation, see Task 02 report
            "posting_date": pd.to_datetime(block.job_posted_date).dt.date.astype(str),
            "location": block.job_location,
            "job_url": "",
            "employment_type": block.job_schedule_type,
            "remote": block.job_work_from_home,
            "country": block.job_country,
            "salary_year_avg": block.salary_year_avg,
            "extracted_skills": block.job_skills,
            "skill_categories": block.job_type_skills,
            "job_via": block.job_via,
            "source": SOURCE_LABEL,
            "scraped_date": str(date.today()),
        }
    )
    return out.drop_duplicates(subset="job_id").reset_index(drop=True)


def build_company(full: pd.DataFrame, key: str) -> dict:
    """Select, preprocess and skill-extract one company. Returns a manifest row."""
    company = co.get_company(key)
    selected = co.select_company(full, key)
    raw = to_shared_schema(selected)
    # Re-derived from the name rather than carried across the dedup, so a
    # dropped duplicate cannot shift the brand labels by a row.
    raw["company_brand"] = raw.company_name.map(
        lambda n: co.classify_employer(n, key)["brand"]
    )
    raw["company_key"] = key

    raw_dir = REPO_ROOT / "data" / "raw" / key
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_dir / f"{key}_jobs_hf_backfill_2023.parquet", index=False)

    clean = pp.preprocess(raw)
    features, unmapped = sk.build_posting_features(clean)
    long = sk.skills_long_table(features)

    out_dir = REPO_ROOT / "data" / "processed" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / f"{key}_features.parquet", index=False)
    long.to_parquet(out_dir / f"{key}_skills_long.parquet", index=False)

    dates = pd.to_datetime(features.posting_date, errors="coerce")
    row = {
        "company": key,
        "display": company.display,
        "postings_matched": int(len(selected)),
        "postings_after_dedup": int(len(raw)),
        "postings_with_skills": int(features.has_any_skill.sum()),
        "skill_coverage": round(float(features.has_any_skill.mean()), 4),
        "distinct_skills": int(long.skill.nunique()) if len(long) else 0,
        "publishers": int(features.job_via.nunique()),
        "countries": int(features.location_country.replace("", pd.NA).nunique()),
        "first_posting": str(dates.min().date()),
        "last_posting": str(dates.max().date()),
        "brands": ", ".join(
            f"{b} ({n})" for b, n in raw.company_brand.value_counts().items()
        ),
        "unmapped_source_terms": len(unmapped),
    }
    print(f"{key:11s} {row['postings_after_dedup']:5d} postings  "
          f"{row['skill_coverage']:.1%} skilled  "
          f"{row['publishers']:3d} publishers")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", default=None,
                        help="one registry key; default is every company that "
                             "passes the feasibility screen")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--source", default=str(SOURCE))
    args = parser.parse_args()

    full = pd.read_parquet(args.source)
    # The screen reads `posting_date`; the source calls it `job_posted_date`.
    full["posting_date"] = pd.to_datetime(full.job_posted_date).dt.date.astype(str)
    print(f"source: {len(full):,} rows <- {Path(args.source).name}")

    TABLES.mkdir(parents=True, exist_ok=True)

    screen = co.feasibility_screen(full)
    screen.to_csv(TABLES / "company-feasibility-screen.csv", index=False)
    audit = co.matching_audit(full)
    audit.to_csv(TABLES / "employer-matching-audit.csv", index=False)
    print(screen.to_string(index=False))
    print(f"\naudit: {len(audit)} distinct employer strings, "
          f"{int(audit.loc[audit.decision == 'excluded', 'postings'].sum())} "
          f"postings excluded as third parties")

    if args.screen_only:
        return

    keys = [args.company] if args.company else co.included_companies(screen)
    manifest = [build_company(full, key) for key in keys]
    man = pd.DataFrame(manifest)
    man.to_csv(TABLES / "competitor-set-manifest.csv", index=False)

    meta = {
        "source": SOURCE_LABEL,
        "source_rows": int(len(full)),
        "shortlist": list(co.SHORTLIST),
        "included": keys,
        "excluded": screen.loc[screen.verdict != "included", "company"].tolist(),
        "min_postings": co.MIN_POSTINGS,
        "min_months": co.MIN_MONTHS,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (TABLES / "competitor-set-manifest.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(f"\nmanifest -> {(TABLES / 'competitor-set-manifest.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
