"""Task 03 — NLP preprocessing for job-posting data.

Turns a raw collected dataset (Task 02 schema) into a modelling-ready table.
Written to be company-agnostic so all four specialists run the *same*
preprocessing and cross-company comparisons stay fair (brief, Task 03).

Design decisions (rationale in docs/task-03-preprocessing-methods.md)
--------------------------------------------------------------------
1. **No external corpora.** Stopwords and rules live in this file, so every
   team member and any CI run gets byte-identical output with no
   `nltk.download()` / spaCy model step to drift between machines.
2. **Technical tokens are preserved.** `c++`, `c#`, `.net`, `node.js`,
   `scikit-learn`, `ci/cd` survive tokenisation, and short tech names
   (`r`, `go`, `ai`, `bi`) are exempt from the min-length filter. Generic
   stemming/lemmatising is *off by default* because it corrupts skill names
   (`kubernetes` -> `kubernete`), and Task 04 skill extraction depends on
   these tokens being intact.
3. **Two layers.** Layer A derives structure from fields we always have
   (title, company, location, date). Layer B is the free-text pipeline for
   `job_description`. The Google backfill carries no description text, so
   Layer B currently no-ops on those rows — it is validated separately
   against real posting HTML by `src/validate_text_pipeline.py`.

Usage
-----
    python src/preprocess.py --company google
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Layer B — free-text cleaning
# ---------------------------------------------------------------------------

# Multi-character technical tokens that must survive tokenisation intact.
TECH_LITERALS = [
    "c++", "c#", "f#", ".net", "asp.net", "node.js", "next.js", "vue.js",
    "react.js", "scikit-learn", "ci/cd", "objective-c", "t-sql", "pl/sql",
    "power bi", "google cloud", "big query", "d3.js", "socket.io",
]

# Short tokens that are real skills and must escape the min-length filter.
SHORT_TECH_TOKENS = {
    "r", "c", "go", "ai", "ml", "bi", "ux", "ui", "qa", "os", "db", "js",
    "gcp", "aws", "sql", "etl", "nlp", "api", "cv", "rl", "3d",
}

# Curated stopword list — in-repo for reproducibility. English function words
# plus job-posting boilerplate that carries no signal about skills or demand.
STOPWORDS = {
    # function words
    "a", "about", "above", "after", "again", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "did", "do", "does",
    "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
    "most", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours",
    # job-posting boilerplate (no signal for skills / demand)
    "ability", "applicant", "apply", "benefit", "benefits", "candidate",
    "candidates", "career", "company", "culture", "employer", "equal",
    "experience", "gender", "identity", "including", "job", "know", "level",
    "location", "minimum", "opportunity", "orientation", "per", "please",
    "plus", "position", "preferred", "qualification", "qualifications",
    "regardless", "relevant", "requirement", "requirements", "responsibility",
    "responsibilities", "role", "salary", "sexual", "skill", "skills", "team",
    "us", "work", "working", "year", "years",
}

TECH_LITERAL_SET = set(TECH_LITERALS)

_TECH_ALT = "|".join(
    re.escape(t) for t in sorted(TECH_LITERALS, key=len, reverse=True)
)
# Tech literals first, then general words allowing internal . _ - / (aws/s3).
TOKEN_RE = re.compile(
    rf"(?:{_TECH_ALT})|[a-z][a-z0-9]*(?:[.\-_/][a-z0-9]+)*", re.IGNORECASE
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# Equal-opportunity / legal boilerplate: strip whole trailing block.
_EEO_RE = re.compile(
    r"(google is proud to be an equal opportunity|"
    r"we are an equal opportunity employer|"
    r"equal employment opportunity).*",
    re.IGNORECASE | re.DOTALL,
)


def strip_accents(text: str) -> str:
    """Zürich -> Zurich, São Paulo -> Sao Paulo, so grouping keys match."""
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


# Real postings are full of typographic characters and emoji (benefits lists
# use 🌎 remote / 💶 salary / 🏖 leave). Output is forced to ASCII so grouping
# keys, CSV interchange and tokenisation behave identically on all four
# team members' machines regardless of OS locale.
_TYPOGRAPHY = {
    "—": "-", "–": "-", "‒": "-", "−": "-",   # dashes
    "‘": "'", "’": "'", "‚": "'", "′": "'",   # quotes
    "“": '"', "”": '"', "„": '"', "″": '"',
    "…": "...", "•": " ", "·": " ", "→": " ",  # misc
    " ": " ", "​": " ", "﻿": " ",                   # spaces
    # Currency kept as ASCII codes so salary text survives cleaning.
    "€": " EUR ", "£": " GBP ", "¥": " JPY ",
    "₹": " INR ", "₩": " KRW ",
}


def normalize_typography(text: str) -> str:
    """Map typographic characters to ASCII and drop emoji / pictographs."""
    for src, dst in _TYPOGRAPHY.items():
        text = text.replace(src, dst)
    text = strip_accents(text)
    # Anything still non-ASCII (emoji, variation selectors, CJK) becomes a
    # space rather than being deleted, so it can't glue two words together.
    return "".join(c if ord(c) < 128 else " " for c in text)


def clean_text(raw: str | float | None) -> str:
    """HTML -> readable plain text. Idempotent; safe on empty input."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    text = raw
    # <br>/</p>/</li> mark real breaks — keep them as spaces, not word joins.
    text = re.sub(r"(?i)<\s*(br|/p|/li|/div|/h[1-6])\s*/?>", " ", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)          # &amp; &nbsp; &#39;
    text = normalize_typography(text)
    text = _EEO_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str, remove_stopwords: bool = True, min_len: int = 3) -> list[str]:
    """Lowercase tokens with technical names preserved.

    `min_len` is waived for SHORT_TECH_TOKENS so `r`, `go`, `bi` survive.
    """
    if not text:
        return []
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    out = []
    for tok in tokens:
        # Known technical names bypass punctuation-stripping and the length
        # filter — otherwise ".net" becomes "net" and "c#" is dropped entirely.
        if tok not in TECH_LITERAL_SET:
            tok = tok.strip("._-/")
            if not tok:
                continue
            if len(tok) < min_len and tok not in SHORT_TECH_TOKENS:
                continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def preprocess_description(raw: str | float | None) -> tuple[str, int]:
    """Returns (cleaned_description, token_count)."""
    cleaned = clean_text(raw)
    return cleaned, len(tokenize(cleaned))


# ---------------------------------------------------------------------------
# Layer A — company canonicalisation
# ---------------------------------------------------------------------------

# Alphabet brands kept distinct so Task 05/06 can compare core Google vs
# Alphabet-wide. Regional legal entities (Google Taiwan, Google Germany
# GmbH, ...) all fold into "Google".
ALPHABET_BRANDS = [
    ("Waymo", r"\bwaymo\b"),
    ("Verily", r"\bverily\b"),
    ("DeepMind", r"deep\s*mind"),
    ("YouTube", r"\byoutube\b"),
    ("Google Fiber", r"google\s+fiber"),
    ("Mandiant", r"\bmandiant\b"),
    ("Google Operations Center", r"google\s+operations\s+center"),
    ("Google", r"\bgoogle\b|\balphabet\b"),
]

# Third parties whose name merely mentions Google (partners, resellers).
NON_ALPHABET_MARKERS = re.compile(
    r"premier partner|reseller|partner\b.*\bgoogle|consultancy|geoambiente",
    re.IGNORECASE,
)


def canonicalize_company(raw: str | float | None) -> dict:
    """Map a messy employer string to a canonical brand + provenance flags."""
    if not isinstance(raw, str) or not raw.strip():
        return {
            "company_canonical": "",
            "is_alphabet": False,
            "company_field_suspect": True,
        }
    name = _WS_RE.sub(" ", strip_accents(raw)).strip(" -,")

    # A comma-heavy string that looks like a job title leaked into the
    # company column (observed once in the Google backfill).
    suspect = bool(
        re.search(r"\b(engineer|scientist|analyst|manager|developer)\b", name, re.I)
        and "," in name
    )

    if NON_ALPHABET_MARKERS.search(name):
        return {
            "company_canonical": name,
            "is_alphabet": False,
            "company_field_suspect": suspect,
        }

    for brand, pattern in ALPHABET_BRANDS:
        if re.search(pattern, name, re.IGNORECASE):
            return {
                "company_canonical": brand,
                "is_alphabet": True,
                "company_field_suspect": suspect,
            }
    return {
        "company_canonical": name,
        "is_alphabet": False,
        "company_field_suspect": suspect,
    }


# ---------------------------------------------------------------------------
# Layer A — seniority
# ---------------------------------------------------------------------------

# Order matters: most specific first ("Senior Staff" before "Staff").
SENIORITY_RULES = [
    ("Intern", r"\b(intern|internship|apprentice|co-?op|trainee)\b"),
    ("Director+", r"\b(director|head of|vice president|vp|chief)\b"),
    ("Principal", r"\b(principal|distinguished|fellow)\b"),
    ("Senior Staff", r"\b(senior|sr\.?)\s+staff\b"),
    ("Staff", r"\bstaff\b"),
    ("Lead", r"\b(lead|leader)\b"),
    ("Senior", r"\b(senior|sr\.?)\b"),
    ("Junior", r"\b(junior|jr\.?|associate|entry[\s-]level)\b"),
]

# "Product/Program Manager" is an individual-contributor title, NOT people
# management — the single most common misclassification in job-title parsing.
IC_MANAGER_RE = re.compile(
    r"\b(product|program|project|account|marketing|engagement|partner|"
    r"category|relationship|technology|technical program|supply chain|"
    r"community|content)\s+manager\b",
    re.IGNORECASE,
)
LEVEL_MARKER_RE = re.compile(r"(?<![A-Za-z])(IV|III|II|I)(?![A-Za-z])")


def extract_seniority(title: str | float | None) -> dict:
    if not isinstance(title, str) or not title.strip():
        return {"seniority_level": "Unknown", "level_marker": "", "manager_type": ""}
    t = strip_accents(title)

    level = "Mid"  # default when the title carries no seniority marker
    for label, pattern in SENIORITY_RULES:
        if re.search(pattern, t, re.IGNORECASE):
            level = label
            break

    marker = LEVEL_MARKER_RE.search(t)
    manager_type = ""
    if re.search(r"\bmanager\b", t, re.IGNORECASE):
        manager_type = "ic_manager_title" if IC_MANAGER_RE.search(t) else "people_manager"

    return {
        "seniority_level": level,
        "level_marker": marker.group(1) if marker else "",
        "manager_type": manager_type,
    }


# ---------------------------------------------------------------------------
# Layer A — role category
# ---------------------------------------------------------------------------

# Order is critical. "Data Center" MUST be tested before any "data" rule:
# ~11% of Google's postings are data-centre facilities roles (mechanical /
# electrical engineers, technicians) that would otherwise be miscounted as
# data-science demand and corrupt the whole forecast.
CATEGORY_RULES = [
    ("Data Center / Facilities",
     r"data\s*cent(er|re)|\b(mechanical|electrical|hvac|plant|facilities|"
     r"technician|controls|construction)\b"),
    ("AI / ML",
     r"machine learning|\bml\b|\bai\b|artificial intelligence|deep learning|"
     r"\bnlp\b|computer vision|research scientist|generative|\bllm\b"),
    ("Data Science",
     r"data scien|\bstatistic|quantitative|biostatistic|\bmodel(l)?ing\b"),
    ("Data Engineering",
     r"data engineer|data architect|\betl\b|data pipeline|big data|"
     r"data platform|data warehouse|database|\bdba\b|data management|"
     r"informatics"),
    ("Data Analytics / BI",
     r"analyst|analytics|business intelligence|\bbi\b|insight|reporting"),
    ("Cloud / Infrastructure",
     r"cloud|customer engineer|solutions? (architect|engineer)|\bdevops\b|"
     r"\bsre\b|site reliability|infrastructur|network|systems engineer|"
     r"security engineer"),
    ("Hardware / Silicon",
     r"\bsilicon\b|hardware|\brf\b|\basic\b|firmware|circuit|semiconductor|"
     r"\bchip\b|soc\b"),
    ("Software Engineering",
     r"software engineer|\bswe\b|developer|full[\s-]stack|front[\s-]?end|"
     r"back[\s-]?end|android|\bios\b|mobile engineer|test engineer|\bsdet\b|"
     r"\bqa\b"),
    ("Research / Academic",
     r"student researcher|research (intern|assistant|program)|\bphd\b"),
    ("Product / Program",
     r"product manager|program manager|\btpm\b|product owner"),
    ("Sales / Marketing / GTM",
     r"sales|marketing|account (executive|manager)|business development|"
     r"partnership|go[\s-]to[\s-]market"),
    ("Business / Strategy / Ops",
     r"business|strategy|operations|finance|legal|recruit|administrative|"
     r"people\b"),
]


def extract_job_category(title: str | float | None) -> str:
    if not isinstance(title, str) or not title.strip():
        return "Unknown"
    t = strip_accents(title)
    for label, pattern in CATEGORY_RULES:
        if re.search(pattern, t, re.IGNORECASE):
            return label
    return "Other"


# `job_category` says *which technology area*; `job_function` says *what kind
# of work*. They are orthogonal, and conflating them corrupts demand signals:
# an "Analytics Sales Specialist" is someone selling analytics products, not
# analytics capacity being built. Task 05 should filter on job_function to
# separate capability-building from go-to-market hiring.
FUNCTION_RULES = [
    ("Facilities / Operations",
     r"data\s*cent(er|re)|\b(mechanical|electrical|hvac|plant|facilities|"
     r"technician|controls|construction)\b"),
    ("Technical Sales",
     r"customer engineer|sales engineer|solutions? (architect|consultant)|"
     r"pre[\s-]?sales|technical account manager"),
    ("Sales",
     r"\bsales\b|account executive|business development|\bquota\b"),
    ("Analytics",
     r"analyst|analytics|business intelligence|\bbi\b|insight|reporting|"
     r"metrics"),
    ("Science / Research",
     r"scientist|research|\bstatistic|quantitative|\bphd\b"),
    ("Engineering",
     r"engineer|developer|architect|\bsre\b|\bdevops\b|programmer"),
    ("Product / Program",
     r"product manager|program manager|\btpm\b|product owner"),
    ("Marketing / Communications", r"marketing|brand|communications"),
    ("Support / Admin",
     r"administrative|business partner|recruit|\bhr\b|legal|finance|"
     r"specialist|coordinator"),
]


def extract_job_function(title: str | float | None) -> str:
    if not isinstance(title, str) or not title.strip():
        return "Unknown"
    t = strip_accents(title)
    for label, pattern in FUNCTION_RULES:
        if re.search(pattern, t, re.IGNORECASE):
            return label
    return "Other"


# Malformed titles observed in the source (e.g. "2023 summer intern data") —
# flagged for the Data Quality Lead rather than silently analysed.
_ROLE_NOUN_RE = re.compile(
    r"engineer|scientist|analyst|manager|developer|architect|specialist|"
    r"technician|researcher|lead|director|intern|consultant|designer|"
    r"representative|partner|negotiator|coordinator|administrator|advisor|"
    r"apprentice|associate|instructor|owner|strategist|head of|chief|"
    r"president|officer|counsel|trainee",
    re.IGNORECASE,
)
# Aggregator feeds truncate long titles; the tail is lost, not just the words.
_TRUNCATED_RE = re.compile(r"(\.\.\.|…)\s*$")


def is_title_suspect(title: str | float | None) -> bool:
    """True when a title is unusable as-is for grouping or classification.

    Three signals, all observed in real collected rows: no recognisable role
    noun ("Marketing Data"), an all-lowercase title suggesting a mangled feed
    ("2023 summer intern data"), and a truncated title from an aggregator.
    """
    if not isinstance(title, str) or not title.strip():
        return True
    t = title.strip()
    if _TRUNCATED_RE.search(t):
        return True
    if not _ROLE_NOUN_RE.search(t):
        return True
    letters = [c for c in t if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def clean_job_title(title: str | float | None) -> str:
    """Normalise a title for grouping: accents, whitespace, zero-width chars.

    Also drops trailing intake-season noise ("..., Summer 2023") that would
    otherwise fragment identical roles across cohorts.
    """
    if not isinstance(title, str):
        return ""
    t = strip_accents(title).replace("​", " ")
    t = _WS_RE.sub(" ", t).strip(" ,-")
    # ", Summer 2023" / ", 2023" — intake cohort, not part of the role.
    t = re.sub(
        r"(?:,\s*|\s+)(?:(?:summer|fall|winter|spring)\s*/?\s*)?\d{4}\s*$",
        "",
        t,
        flags=re.I,
    )
    return t.strip(" ,-")


# ---------------------------------------------------------------------------
# Layer A — location, skills, time
# ---------------------------------------------------------------------------

REMOTE_RE = re.compile(r"\b(anywhere|remote|work from home|wfh)\b", re.IGNORECASE)

# US postings arrive as "Sunnyvale, CA" — the trailing token is a *state*, not a
# country. Without this the US fragments into 17 pseudo-countries and vanishes
# from any country ranking, while "CA" collides with Canada's ISO code.
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR",
}

# One country, one spelling — otherwise cross-company grouping splits it.
COUNTRY_ALIASES = {
    "uk": "United Kingdom", "u.k.": "United Kingdom",
    "great britain": "United Kingdom", "england": "United Kingdom",
    "usa": "United States", "us": "United States",
    "u.s.": "United States", "u.s.a.": "United States",
    "uae": "United Arab Emirates", "korea": "South Korea",
    "czech republic": "Czechia", "turkey": "Turkiye",
}

# Multi-word countries that appear in postings *without* a comma
# ("Riyadh Saudi Arabia", "Dubai - United Arab Emirates").
COUNTRY_SUFFIXES = sorted(
    [
        "Saudi Arabia", "United Arab Emirates", "United States",
        "United Kingdom", "South Korea", "South Africa", "New Zealand",
        "Dominican Republic", "Costa Rica", "Hong Kong", "Sri Lanka",
        "Czech Republic", "Singapore", "Israel", "Ireland", "India",
    ],
    key=len,
    reverse=True,
)

# The source marks postings open in several places as "NY (+1 other)".
_MULTI_LOC_RE = re.compile(r"\s*\(\+\s*\d+\s+others?\s*\)\s*$", re.IGNORECASE)


def _resolve_country(raw: str) -> tuple[str, str]:
    """Return (city_prefix, country) for a fragment with no comma separator."""
    for country in COUNTRY_SUFFIXES:
        if raw.lower().endswith(country.lower()):
            prefix = raw[: -len(country)].strip(" ,-")
            return prefix, country
    return "", COUNTRY_ALIASES.get(raw.lower(), raw)


def parse_location(loc: str | float | None) -> dict:
    """Split a posting location into city / region / country.

    Handles the four shapes this source actually emits: "City, Region,
    Country", "City, ST" (US), "City - Country" / "City Country" with no
    comma, and a "(+N others)" multi-location suffix.
    """
    empty = {"city": "", "region": "", "location_country": "",
             "location_is_remote": False, "location_multi": False,
             "location_clean": ""}
    if not isinstance(loc, str) or not loc.strip():
        return empty
    clean = _WS_RE.sub(" ", strip_accents(loc)).strip()
    if REMOTE_RE.search(clean):
        return {**empty, "location_is_remote": True, "location_clean": clean}

    # Strip the multi-location marker but record that it was there — a posting
    # open in six offices is a different demand signal from one in a single city.
    stripped = _MULTI_LOC_RE.sub("", clean)
    multi = stripped != clean

    parts = [p.strip() for p in stripped.replace(" - ", ", ").split(",") if p.strip()]
    if not parts:
        return {**empty, "location_multi": multi, "location_clean": clean}

    last = parts[-1]
    if last.upper() in US_STATES:
        # "Sunnyvale, CA" / "Mountain View, CA, USA" -> state is the region.
        city = parts[0] if len(parts) >= 2 else ""
        region, country = last.upper(), "United States"
    elif len(parts) >= 3:
        city, region, country = parts[0], parts[1], _resolve_country(last)[1]
    elif len(parts) == 2:
        city, region, country = parts[0], "", _resolve_country(last)[1]
    else:
        # Single fragment: a city-state ("Singapore") or an un-delimited
        # "City Country" ("Riyadh Saudi Arabia").
        city, country = _resolve_country(parts[0])
        region = ""

    return {"city": city, "region": region, "location_country": country,
            "location_is_remote": False, "location_multi": multi,
            "location_clean": clean}


def parse_skill_list(raw: str | float | None) -> list[str]:
    """Parse the stringified Python list the source ships (`"['python']"`).

    De-duplicates while preserving order — the source repeats skills
    (`['r','python','sas','sas','sql']`), which would double-count features.
    """
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            items = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            items = [s for s in re.split(r"[,;|]", raw) if s.strip()]
    else:
        return []
    if not isinstance(items, (list, tuple)):
        return []
    seen, out = set(), []
    for item in items:
        s = str(item).strip().lower()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def parse_skill_categories(raw: str | float | None) -> dict:
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            data = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    else:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(k): sorted({str(v).strip().lower() for v in vals})
        for k, vals in data.items()
        if isinstance(vals, (list, tuple))
    }


def derive_time_fields(date_str: str | float | None) -> dict:
    ts = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(ts):
        return {"posting_month": "", "posting_week": "", "posting_quarter": "",
                "posting_dow": ""}
    iso = ts.isocalendar()
    return {
        "posting_month": f"{ts.year}-{ts.month:02d}",
        "posting_week": f"{iso[0]}-W{iso[1]:02d}",
        "posting_quarter": f"{ts.year}-Q{(ts.month - 1) // 3 + 1}",
        "posting_dow": ts.day_name(),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    company = out.company_name.apply(canonicalize_company).apply(pd.Series)
    seniority = out.job_title.apply(extract_seniority).apply(pd.Series)
    location = out.location.apply(parse_location).apply(pd.Series)
    times = out.posting_date.apply(derive_time_fields).apply(pd.Series)
    out = pd.concat([out, company, seniority, location, times], axis=1)

    out["job_title_clean"] = out.job_title.apply(clean_job_title)
    out["job_category"] = out.job_title.apply(extract_job_category)
    out["job_function"] = out.job_title.apply(extract_job_function)
    out["title_suspect"] = out.job_title.apply(is_title_suspect)

    desc = out.job_description.apply(preprocess_description)
    out["cleaned_description"] = [d[0] for d in desc]
    out["description_token_count"] = [d[1] for d in desc]
    out["has_description"] = out.description_token_count > 0

    if "extracted_skills" in out.columns:
        out["skills_parsed"] = out.extracted_skills.apply(parse_skill_list)
        out["skill_count"] = out.skills_parsed.apply(len)
        out["has_skills"] = out.skill_count > 0
    if "skill_categories" in out.columns:
        out["skill_categories_parsed"] = out.skill_categories.apply(
            parse_skill_categories
        )

    out["is_internship"] = (out.seniority_level == "Intern") | out.get(
        "employment_type", pd.Series("", index=out.index)
    ).fillna("").str.contains("intern", case=False)

    return out


def quality_report(df: pd.DataFrame) -> dict:
    """Data Quality Lead checks (Task 01 §5) — recorded, not silently fixed."""
    report = {
        "rows_in": int(len(df)),
        "rows_alphabet": int(df.is_alphabet.sum()),
        "rows_excluded_non_alphabet": int((~df.is_alphabet).sum()),
        "excluded_companies": sorted(
            df.loc[~df.is_alphabet, "company_name"].unique().tolist()
        ),
        "company_field_suspect_rows": int(df.company_field_suspect.sum()),
        "rows_with_description": int(df.has_description.sum()),
        "rows_missing_location": int((df.location_clean == "").sum()),
        "rows_remote": int(df.location_is_remote.sum()),
        "rows_multi_location": int(df.location_multi.sum()),
        "personal_data_columns_present": [
            c for c in df.columns
            if re.search(r"name|email|phone|candidate|applicant|recruiter", c, re.I)
            and c not in {"company_name", "company_canonical"}
        ],
        "title_suspect_rows": int(df.title_suspect.sum()),
        "company_canonical_counts": df.company_canonical.value_counts().to_dict(),
        "job_category_counts": df.job_category.value_counts().to_dict(),
        "job_function_counts": df.job_function.value_counts().to_dict(),
        "seniority_counts": df.seniority_level.value_counts().to_dict(),
        "top_countries": df.loc[df.location_country != "", "location_country"]
        .value_counts()
        .head(15)
        .to_dict(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if "skill_count" in df.columns:
        report["rows_with_skills"] = int(df.has_skills.sum())
        report["mean_skills_per_posting"] = round(
            float(df.loc[df.has_skills, "skill_count"].mean()), 2
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", default="google")
    parser.add_argument(
        "--infile",
        default="data/raw/google/google_jobs_hf_backfill_2023.parquet",
    )
    args = parser.parse_args()

    src = REPO_ROOT / args.infile
    df = pd.read_parquet(src)
    print(f"read {len(df):,} rows <- {args.infile}")

    clean = preprocess(df)
    report = quality_report(clean)

    out_dir = REPO_ROOT / "data" / "processed" / args.company
    out_dir.mkdir(parents=True, exist_ok=True)
    # Parquet keeps list/dict columns; CSV is the human-readable copy.
    clean.to_parquet(out_dir / f"{args.company}_jobs_clean.parquet", index=False)
    clean.drop(columns=["skill_categories_parsed"], errors="ignore").to_csv(
        out_dir / f"{args.company}_jobs_clean.csv", index=False
    )

    qa_path = (
        REPO_ROOT / "members" / f"ankit-{args.company}" / "task-03-quality-report.json"
    )
    qa_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"wrote {len(clean):,} rows -> data/processed/{args.company}/")
    print(f"quality report -> {qa_path.relative_to(REPO_ROOT)}")
    print(json.dumps({k: v for k, v in report.items()
                      if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
