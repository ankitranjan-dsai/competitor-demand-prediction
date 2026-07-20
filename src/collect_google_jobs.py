"""Task 02 — Data collection for the Google specialist track.

Collects Google job postings from the legally approved sources in
docs/task-01-data-sources-and-legal.md and normalises everything to the
team's shared schema:

    job_id, company_name, job_title, job_description, posting_date,
    location, job_url

plus recommended fields where the source provides them. Raw output lands in
data/raw/google/ (git-ignored); a small committed manifest records what was
collected so the pull is reproducible.

Sources
-------
hf-backfill   lukebarousse/data_jobs on Hugging Face (Apache-2.0).
              785k data-role postings from 2023; filtered to the Google
              family. No description text / URLs (dataset limitation) but
              includes pre-extracted skills, salary and remote flags.
adzuna        Live postings via the official Adzuna API. Requires free
              credentials in the environment or a .env file:
                  ADZUNA_APP_ID=...  ADZUNA_APP_KEY=...
              (register at https://developer.adzuna.com — takes minutes).
themuse       The Muse public API. Kept for team-wide reuse; as of
              2026-07-20 it lists no Google postings (documented in the
              Task 02 report).

Usage
-----
    python src/collect_google_jobs.py hf-backfill
    python src/collect_google_jobs.py adzuna --pages 10
    python src/collect_google_jobs.py themuse
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "google"
MANIFEST = REPO_ROOT / "members" / "ankit-google" / "data-manifest.json"

USER_AGENT = (
    "competitor-demand-prediction-cadetx/0.1 "
    "(student research project; contact: ankit0ranjan@gmail.com)"
)

HF_PARQUET_URL = (
    "https://huggingface.co/api/datasets/lukebarousse/data_jobs"
    "/parquet/default/train/0.parquet"
)
GOOGLE_FAMILY_PATTERN = r"google|alphabet|deepmind|youtube|waymo|verily"

SHARED_SCHEMA = [
    "job_id",
    "company_name",
    "job_title",
    "job_description",
    "posting_date",
    "location",
    "job_url",
]


def stable_job_id(source: str, *parts: str) -> str:
    """Deterministic id so re-pulls don't duplicate rows."""
    digest = hashlib.sha1("|".join([source, *[p or "" for p in parts]]).encode())
    return f"{source}-{digest.hexdigest()[:12]}"


def polite_get(url: str, params: dict | None = None) -> requests.Response:
    resp = requests.get(
        url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30
    )
    resp.raise_for_status()
    time.sleep(1.0)  # ≤1 req/sec, per the Task 01 etiquette rules
    return resp


# --------------------------------------------------------------------------
# Source: Hugging Face backfill (lukebarousse/data_jobs, Apache-2.0)
# --------------------------------------------------------------------------

def collect_hf_backfill() -> pd.DataFrame:
    cache = RAW_DIR / "_hf_data_jobs_full.parquet"
    if not cache.exists():
        print(f"downloading {HF_PARQUET_URL} …")
        resp = requests.get(
            HF_PARQUET_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=600,
            allow_redirects=True,
        )
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    df = pd.read_parquet(cache)
    print(f"full dataset: {len(df):,} rows")

    g = df[
        df.company_name.str.contains(GOOGLE_FAMILY_PATTERN, case=False, na=False)
    ].copy()
    print(f"google-family rows: {len(g):,}")

    out = pd.DataFrame(
        {
            "job_id": [
                stable_job_id("hf", c, t, str(d))
                for c, t, d in zip(g.company_name, g.job_title, g.job_posted_date)
            ],
            "company_name": g.company_name,
            "job_title": g.job_title,
            # Dataset limitation: no description text — see Task 02 report.
            "job_description": "",
            "posting_date": pd.to_datetime(g.job_posted_date).dt.date.astype(str),
            "location": g.job_location,
            "job_url": "",
            "employment_type": g.job_schedule_type,
            "remote": g.job_work_from_home,
            "country": g.job_country,
            "salary_year_avg": g.salary_year_avg,
            "extracted_skills": g.job_skills,
            "skill_categories": g.job_type_skills,
            "job_via": g.job_via,
            "source": "hf:lukebarousse/data_jobs (Apache-2.0)",
            "scraped_date": str(date.today()),
        }
    )
    return out.drop_duplicates(subset="job_id")


# --------------------------------------------------------------------------
# Source: Adzuna official API (requires free ADZUNA_APP_ID / ADZUNA_APP_KEY)
# --------------------------------------------------------------------------

def collect_adzuna(pages: int, country: str = "gb") -> pd.DataFrame:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    app_id, app_key = os.getenv("ADZUNA_APP_ID"), os.getenv("ADZUNA_APP_KEY")
    if not (app_id and app_key):
        sys.exit(
            "Adzuna credentials missing. Register (free) at "
            "https://developer.adzuna.com and put ADZUNA_APP_ID / "
            "ADZUNA_APP_KEY in the environment or a .env file."
        )

    rows = []
    for page in range(1, pages + 1):
        data = polite_get(
            f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}",
            params={
                "app_id": app_id,
                "app_key": app_key,
                "company": "Google",
                "results_per_page": 50,
                "content-type": "application/json",
            },
        ).json()
        results = data.get("results", [])
        if not results:
            break
        for r in results:
            rows.append(
                {
                    "job_id": stable_job_id("adzuna", str(r.get("id"))),
                    "company_name": (r.get("company") or {}).get(
                        "display_name", ""
                    ),
                    "job_title": r.get("title", ""),
                    "job_description": r.get("description", ""),
                    "posting_date": (r.get("created") or "")[:10],
                    "location": (r.get("location") or {}).get(
                        "display_name", ""
                    ),
                    "job_url": r.get("redirect_url", ""),
                    "employment_type": r.get("contract_time", ""),
                    "salary_min": r.get("salary_min"),
                    "salary_max": r.get("salary_max"),
                    "category": (r.get("category") or {}).get("label", ""),
                    "source": "adzuna-api",
                    "scraped_date": str(date.today()),
                }
            )
        print(f"page {page}: +{len(results)} postings")
    df = pd.DataFrame(rows)
    # Adzuna's company filter is fuzzy; keep genuine Google-family employers.
    if not df.empty:
        df = df[
            df.company_name.str.contains(
                GOOGLE_FAMILY_PATTERN, case=False, na=False
            )
        ]
    return df.drop_duplicates(subset="job_id")


# --------------------------------------------------------------------------
# Source: The Muse public API
# --------------------------------------------------------------------------

def collect_themuse() -> pd.DataFrame:
    data = polite_get(
        "https://www.themuse.com/api/public/jobs",
        params={"company": "Google", "page": 1},
    ).json()
    rows = [
        {
            "job_id": stable_job_id("muse", str(r.get("id"))),
            "company_name": (r.get("company") or {}).get("name", ""),
            "job_title": r.get("name", ""),
            "job_description": r.get("contents", ""),
            "posting_date": (r.get("publication_date") or "")[:10],
            "location": "; ".join(
                loc.get("name", "") for loc in r.get("locations", [])
            ),
            "job_url": (r.get("refs") or {}).get("landing_page", ""),
            "source": "themuse-api",
            "scraped_date": str(date.today()),
        }
        for r in data.get("results", [])
    ]
    print(f"the muse: {len(rows)} Google postings")
    return pd.DataFrame(rows, columns=SHARED_SCHEMA + ["source", "scraped_date"])


# --------------------------------------------------------------------------

def save(df: pd.DataFrame, name: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = RAW_DIR / f"{name}.parquet"
    csv_path = RAW_DIR / f"{name}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"wrote {len(df):,} rows -> {parquet_path.relative_to(REPO_ROOT)}")

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[name] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "file": str(parquet_path.relative_to(REPO_ROOT)),
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date_min": str(df.posting_date.min()) if len(df) else None,
        "date_max": str(df.posting_date.max()) if len(df) else None,
        "rebuild": f"python src/collect_google_jobs.py {name.split('_')[-1] if name.split('_')[-1] in ('adzuna', 'themuse') else 'hf-backfill'}",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest updated -> {MANIFEST.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["hf-backfill", "adzuna", "themuse"])
    parser.add_argument("--pages", type=int, default=10, help="adzuna pages")
    parser.add_argument("--country", default="gb", help="adzuna country code")
    args = parser.parse_args()

    if args.source == "hf-backfill":
        save(collect_hf_backfill(), "google_jobs_hf_backfill_2023")
    elif args.source == "adzuna":
        save(collect_adzuna(args.pages, args.country), "google_jobs_adzuna")
    else:
        save(collect_themuse(), "google_jobs_themuse")


if __name__ == "__main__":
    main()
