"""Task 04 — build the extracted-skills dataset and feature tables.

Reads the Task 03 cleaned table, runs the shared taxonomy in `src/skills.py`
over it, and writes:

    data/processed/<company>/            (git-ignored — row-level)
        <company>_skills_long.parquet|csv    one row per (posting, skill)
        <company>_features.parquet|csv       one row per posting + skill features
        <company>_skill_matrix.parquet|csv   binary posting x skill matrix

    members/ankit-<company>/task-04-tables/   (committed — aggregate only)
        skill-frequency.csv          skill counts, both denominators
        skill-by-month.csv           monthly frequency, coverage-normalised
        skill-trend.csv              H1 vs H2 share + emerging_skill_flag
        skill-cooccurrence.csv       skill pairs + Jaccard (feeds Task 08)
        skill-by-job-function.csv    skills split by kind of work
        coverage-by-month.csv        denominators for every share above
        coverage-by-job-function.csv
        coverage-by-publisher.csv

    members/ankit-<company>/task-04-feature-report.json   quality evidence

Aggregate tables are committed because they *are* the Task 04 deliverable and
carry no posting prose; row-level output stays git-ignored under the Task 01
data-handling rules.

    python src/build_features.py --company google
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import skills as sk

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build(df: pd.DataFrame, company: str) -> dict:
    features, unmapped = sk.build_posting_features(df)
    long = sk.skills_long_table(features)

    n_total = len(features)
    n_skilled = int(features.has_any_skill.sum())

    cov_month = sk.coverage_table(features, "posting_month")
    cov_function = (
        sk.coverage_table(features, "job_function")
        if "job_function" in features.columns
        else pd.DataFrame()
    )
    cov_publisher = (
        sk.coverage_table(features, "job_via").sort_values(
            "postings", ascending=False
        )
        if "job_via" in features.columns
        else pd.DataFrame()
    )

    freq = sk.skill_frequency_table(long, n_skilled, n_total)
    by_month = sk.skill_by_month_table(long, cov_month)
    trend = sk.skill_trend_table(long, features)
    cooc = sk.skill_cooccurrence_table(long)
    matrix = sk.skill_matrix(long)

    by_function = (
        long.groupby(["job_function", "skill", "skill_category"], as_index=False)
        .agg(n_postings=("job_id", "nunique"))
        .sort_values(["job_function", "n_postings"], ascending=[True, False])
        if "job_function" in long.columns and not long.empty
        else pd.DataFrame()
    )

    # Emerging-skill count per posting, joined back so the row-level table
    # carries the brief's `emerging_skill_flag` signal too.
    emerging = set(trend.loc[trend.emerging_skill_flag == "emerging", "skill"]) \
        if not trend.empty else set()
    features["n_emerging_skills"] = features.skills_final.apply(
        lambda names: sum(1 for n in names if n in emerging)
    )
    features["has_emerging_skill"] = features.n_emerging_skills > 0

    out_dir = REPO_ROOT / "data" / "processed" / company
    out_dir.mkdir(parents=True, exist_ok=True)
    # Parquet preserves the list columns; CSV is the readable copy.
    features.to_parquet(out_dir / f"{company}_features.parquet", index=False)
    features.drop(columns=["skill_categories_parsed", "skill_provenance_map"],
                  errors="ignore").to_csv(
        out_dir / f"{company}_features.csv", index=False
    )
    long.to_parquet(out_dir / f"{company}_skills_long.parquet", index=False)
    long.to_csv(out_dir / f"{company}_skills_long.csv", index=False)
    matrix.to_parquet(out_dir / f"{company}_skill_matrix.parquet", index=False)
    matrix.to_csv(out_dir / f"{company}_skill_matrix.csv", index=False)

    tables = REPO_ROOT / "members" / f"ankit-{company}" / "task-04-tables"
    _write(freq, tables / "skill-frequency.csv")
    _write(by_month, tables / "skill-by-month.csv")
    _write(trend, tables / "skill-trend.csv")
    _write(cooc, tables / "skill-cooccurrence.csv")
    _write(cov_month, tables / "coverage-by-month.csv")
    if not by_function.empty:
        _write(by_function, tables / "skill-by-job-function.csv")
    if not cov_function.empty:
        _write(cov_function, tables / "coverage-by-job-function.csv")
    if not cov_publisher.empty:
        _write(cov_publisher, tables / "coverage-by-publisher.csv")

    top = freq.head(20)[["skill", "skill_category", "n_postings",
                         "share_of_skilled"]].to_dict("records")
    per_path = {
        "source": int(features.skills_source.apply(len).gt(0).sum()),
        "title": int(features.skills_title.apply(len).gt(0).sum()),
        "text": int(features.skills_text.apply(len).gt(0).sum()),
    }
    report = {
        "company": company,
        "rows_in": n_total,
        "rows_with_any_skill": n_skilled,
        "skill_coverage": round(n_skilled / max(n_total, 1), 4),
        "rows_with_skills_by_path": per_path,
        "rows_recovered_by_title": int(features.skill_recovered_by_title.sum()),
        "distinct_skills": int(long.skill.nunique()) if not long.empty else 0,
        "mean_skills_when_present": round(
            float(features.loc[features.has_any_skill, "skill_count_final"].mean()), 2
        )
        if n_skilled
        else 0.0,
        "taxonomy_size": len(sk.SKILLS),
        "taxonomy_categories": list(sk.CATEGORIES),
        "unmapped_source_terms": dict(unmapped),
        "excluded_terms_seen": {
            term: int(
                df.get("skills_parsed", pd.Series(dtype=object))
                .apply(lambda lst, t=term: bool(lst is not None and t in lst))
                .sum()
            )
            for term in sk.EXCLUDED_TERMS
        },
        "org_segments_stripped_from_titles": int(
            df.apply(
                lambda r: sk.strip_org_segments(
                    r.get("job_title_clean"), r.get("company_canonical")
                )
                != r.get("job_title_clean"),
                axis=1,
            ).sum()
        ),
        "skill_matrix_shape": list(matrix.shape),
        "coverage_by_month": cov_month.set_index("posting_month").coverage.to_dict(),
        "coverage_min_max_by_month": [
            float(cov_month.coverage.min()), float(cov_month.coverage.max())
        ],
        "top_skills": top,
        "emerging_skills": trend.loc[
            trend.emerging_skill_flag == "emerging", "skill"
        ].tolist()
        if not trend.empty
        else [],
        "declining_skills": trend.loc[
            trend.emerging_skill_flag == "declining", "skill"
        ].tolist()
        if not trend.empty
        else [],
        # Standing Task 01 commitment — re-checked on every run, never assumed.
        "personal_data_columns_present": [
            c
            for c in features.columns
            if any(
                k in c.lower()
                for k in ("email", "phone", "candidate", "applicant", "recruiter")
            )
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    qa = REPO_ROOT / "members" / f"ankit-{company}" / "task-04-feature-report.json"
    qa.write_text(json.dumps(report, indent=2) + "\n")

    print(f"postings         {n_total:,}  ({n_skilled:,} with >=1 skill, "
          f"{100 * n_skilled / max(n_total, 1):.1f}%)")
    print(f"skills long      {len(long):,} rows, {report['distinct_skills']} "
          f"distinct skills")
    print(f"skill matrix     {matrix.shape[0]:,} x {matrix.shape[1] - 1}")
    print(f"unmapped terms   {report['unmapped_source_terms'] or 'none'}")
    print(f"tables           -> {tables.relative_to(REPO_ROOT)}/")
    print(f"report           -> {qa.relative_to(REPO_ROOT)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", default="google")
    parser.add_argument("--infile", default=None,
                        help="defaults to the Task 03 cleaned parquet")
    args = parser.parse_args()

    infile = args.infile or (
        f"data/processed/{args.company}/{args.company}_jobs_clean.parquet"
    )
    df = pd.read_parquet(REPO_ROOT / infile)
    print(f"read {len(df):,} rows <- {infile}")
    build(df, args.company)


if __name__ == "__main__":
    main()
