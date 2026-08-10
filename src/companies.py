"""Task 06 — who counts as which company, and which companies can be compared.

Shared, company-agnostic employer registry for all four specialists. The team
standard and its rationale live in docs/task-06-competitor-comparison-methods.md.

Why this module exists
----------------------
Task 06 compares companies. Before a single number is comparable, two questions
have to be answered the same way for every company in the set:

1. **Which postings belong to the company?** The employer field is free text
   written by whichever job board syndicated the posting. Substring matching —
   the obvious implementation — is wrong in both directions:

   ===================================  ==================================
   ``Metasys Technologies, Inc.``       a staffing agency, not Meta
   ``We Are META`` (64 postings)        a recruitment brand, not Meta
   ``Oculus Search Partners``           a search firm, not Meta's Oculus
   ``ShortHills Tech (Gold Microsoft    a Microsoft partner, not Microsoft
   Partner)``
   ``Geoambiente - Google Cloud         a Google Cloud reseller, not Google
   Premier Partner``
   ``OpenAirlines``                     not OpenAI
   ``Facebook App`` (45 postings)       *is* Meta
   ``CN05 NVIDIA Shanghai WFOE``        *is* NVIDIA
   ===================================  ==================================

   A naive ``company_name.str.contains("meta")`` inflates Meta by 8.1% with
   agency postings, and every one of those postings carries a different
   publisher mix, a different country mix and a different skill mix to Meta's
   own. So each company gets an explicit include pattern, an explicit exclude
   pattern, and — the part that makes it reviewable — ``matching_audit()``
   emits every distinct employer string with the decision and the reason.

2. **Does the company have enough data to be compared at all?** Two of the
   eight companies the team shortlisted have 14 and 9 postings in the approved
   source. Comparing them to Google's 848 would produce differences that are
   entirely sampling noise. ``feasibility_screen()`` applies a stated floor
   *before* any comparison is computed, and companies that fail are reported
   as excluded-with-a-reason rather than quietly dropped.

Nothing here collects data. It classifies employer strings in data that
Task 02's approved sources already produced.

Usage
-----
    from companies import REGISTRY, resolve_employers, feasibility_screen
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# The shortlist
# ---------------------------------------------------------------------------

#: The eight companies named in the project README as the team's choice set.
#: Registered here whether or not they survive the feasibility screen — a
#: company that fails is a *result*, and it can only be reported as a result if
#: it was screened rather than never considered.
SHORTLIST = (
    "google", "microsoft", "meta", "nvidia",
    "snowflake", "databricks", "openai", "anthropic",
)


# ---------------------------------------------------------------------------
# Third-party markers shared by every company
# ---------------------------------------------------------------------------

#: Words that mark an employer string as a *third party mentioning* the company
#: rather than the company: resellers, implementation partners, staffing firms.
#: Applied to every company, so no specialist has to rediscover them.
#:
#: Deliberately short. Every term here was put in by a name observed in the
#: 2023 backfill, and a term that excludes a real subsidiary costs more than
#: one agency posting: ``solutions``, ``technologies`` and ``careers`` were all
#: considered and rejected for that reason (``Snowflake Computing``,
#: ``Meta Careers`` are real).
AGENCY_MARKERS = r"""
    \bpartners?\b | \breseller\b | \brecruit(ing|ment)?\b | \bstaffing\b
  | \bconsultanc(y|ies)\b | \bbusiness\s+group\b | \btalent\s+(solutions|acquisition)\b
"""

_AGENCY_RE = re.compile(AGENCY_MARKERS, re.IGNORECASE | re.VERBOSE)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Company:
    """One comparable employer: what belongs to it, and what only mentions it.

    ``include`` and ``exclude`` are matched against the raw employer string.
    ``exclude`` wins — a name that matches both is a third party, because the
    third-party markers are more specific than the brand name they contain.

    ``brands`` splits the family into reportable sub-brands, most specific
    first, so a comparison can be run on the core brand or on the whole group.
    It mirrors ``preprocess.ALPHABET_BRANDS``, which Task 03 established for
    Google; ``tests/test_companies.py`` pins the two together.
    """

    key: str
    display: str
    include: str
    exclude: str = ""
    brands: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    note: str = ""

    def __post_init__(self) -> None:
        re.compile(self.include, re.IGNORECASE)
        if self.exclude:
            re.compile(self.exclude, re.IGNORECASE)

    def brand_of(self, name: str) -> str:
        """Sub-brand label for an employer string already known to belong."""
        for label, pattern in self.brands:
            if re.search(pattern, name, re.IGNORECASE):
                return label
        return self.display


REGISTRY: dict[str, Company] = {
    # --- Google -----------------------------------------------------------
    # `include` is Task 02's GOOGLE_FAMILY_PATTERN verbatim and `exclude`
    # carries Task 03's NON_ALPHABET_MARKERS, so Task 06 selects exactly the
    # postings the committed Google pipeline already selected. A test asserts
    # the three stay identical.
    "google": Company(
        key="google",
        display="Google",
        include=r"google|alphabet|deepmind|youtube|waymo|verily",
        exclude=r"premier partner|reseller|partner\b.*\bgoogle|consultancy|geoambiente",
        brands=(
            ("Waymo", r"\bwaymo\b"),
            ("Verily", r"\bverily\b"),
            ("DeepMind", r"deep\s*mind"),
            ("YouTube", r"\byoutube\b"),
            ("Google Fiber", r"google\s+fiber"),
            ("Mandiant", r"\bmandiant\b"),
            ("Google Operations Center", r"google\s+operations\s+center"),
            ("Google", r"\bgoogle\b|\balphabet\b"),
        ),
        note="Alphabet group. Matches Task 02 collection exactly (848 postings, "
             "847 after the one Cloud partner is excluded).",
    ),
    # --- Microsoft --------------------------------------------------------
    # LinkedIn is a Microsoft subsidiary and is deliberately NOT in the family.
    # `job_via` — the publisher column every panel in Task 05 depends on — is
    # "via LinkedIn" for 214 Google postings alone, and the employer column in
    # this source demonstrably receives publisher strings: "LinkedIn Job
    # Wrapping", "LinkedIn Limited Listings", "Our Linkedin group" and two
    # bare linkedin.com URLs all appear as company names. A posting whose
    # employer is "LinkedIn" cannot be separated from that leakage, so all 46
    # are excluded. The cost is recorded in the matching audit rather than
    # absorbed silently.
    "microsoft": Company(
        key="microsoft",
        display="Microsoft",
        include=r"\bmicrosoft\b|\bgithub\b|\bnuance\b|\bxbox\b",
        exclude=r"shorthills|octaware|cognizant|^delivery\b|\bd365\b",
        brands=(
            ("GitHub", r"\bgithub\b"),
            ("Nuance", r"\bnuance\b"),
            ("Xbox", r"\bxbox\b"),
            ("Microsoft", r"\bmicrosoft\b"),
        ),
        note="LinkedIn excluded from the family — the name is indistinguishable "
             "from publisher leakage in this source (46 postings).",
    ),
    # --- Meta -------------------------------------------------------------
    # The worst case in the shortlist: "meta" is a common word-stem and a
    # popular agency prefix. 84 of 1,117 substring matches are third parties.
    "meta": Company(
        key="meta",
        display="Meta",
        include=r"\bmeta\b|\bfacebook\b|\binstagram\b|\bwhatsapp\b|oculus\s*vr",
        exclude=(
            r"we\s+are\s+meta|meta\s+soft|meta\s+force|meta\s+it\b|meta\s+science"
            r"|meta[-\s]?sistem|meta\s+system|meta\s+materials|nocap\s+meta"
            r"|deep\.meta|meta\s+virus|metasys|oculus\s+search"
        ),
        brands=(
            ("Instagram", r"\binstagram\b"),
            ("WhatsApp", r"\bwhatsapp\b"),
            ("Oculus", r"oculus"),
            ("Facebook", r"\bfacebook\b"),
            ("Meta", r"\bmeta\b"),
        ),
        note="Bare 'Oculus' (1 posting) is excluded because 'Oculus Search "
             "Partners' exists in the same source; only 'Oculus VR' is matched.",
    ),
    # --- NVIDIA -----------------------------------------------------------
    # Clean: all 11 distinct names are NVIDIA legal entities, including the
    # site-coded ones ("CN05 NVIDIA Shanghai WFOE", "AM01 NVIDIA Armenia LLC").
    "nvidia": Company(
        key="nvidia",
        display="NVIDIA",
        include=r"\bnvidia\b",
        brands=(("NVIDIA", r"\bnvidia\b"),),
    ),
    "snowflake": Company(
        key="snowflake",
        display="Snowflake",
        include=r"\bsnowflake\b",
        brands=(("Snowflake", r"\bsnowflake\b"),),
    ),
    "databricks": Company(
        key="databricks",
        display="Databricks",
        include=r"\bdatabricks\b",
        brands=(("Databricks", r"\bdatabricks\b"),),
    ),
    # --- The two that fail the screen ------------------------------------
    # Registered so the screen can report them as excluded on the evidence.
    # `\bopenai\b` and not `open\s*ai`: the source contains "OpenAirlines"
    # (4 postings) and "開源智造 Open AI Fab Inc." (1), neither of which is OpenAI.
    "openai": Company(
        key="openai",
        display="OpenAI",
        include=r"\bopenai\b",
        brands=(("OpenAI", r"\bopenai\b"),),
    ),
    "anthropic": Company(
        key="anthropic",
        display="Anthropic",
        include=r"\banthropic\b",
        brands=(("Anthropic", r"\banthropic\b"),),
    ),
}


def get_company(key: str) -> Company:
    try:
        return REGISTRY[key]
    except KeyError:
        raise ValueError(
            f"unknown company {key!r}; registered: {sorted(REGISTRY)}"
        ) from None


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

#: Reasons a name that mentions a company is not that company. Reported per
#: name so the exclusion list is reviewable instead of authoritative.
EXCLUSION_REASONS = {
    "third_party_marker": "partner / reseller / staffing marker in the name",
    "named_third_party": "named in the company's own exclusion list",
    "role_string": "looks like a job title leaked into the employer field",
}

_ROLE_LEAK_RE = re.compile(
    r"\b(engineer|scientist|analyst|manager|developer|architect)\b", re.IGNORECASE
)


def classify_employer(name: str | float | None, key: str) -> dict:
    """Does this employer string belong to ``key``, and if not, why not?

    Returns ``matched``, ``brand`` and ``reason``. ``reason`` is empty on a
    match; on a rejection it is one of ``EXCLUSION_REASONS`` or
    ``no_family_match``.
    """
    company = get_company(key)
    if not isinstance(name, str) or not name.strip():
        return {"matched": False, "brand": "", "reason": "empty_employer_field"}
    text = " ".join(name.split())

    if not re.search(company.include, text, re.IGNORECASE):
        return {"matched": False, "brand": "", "reason": "no_family_match"}

    # A comma-heavy string carrying a role noun is Task 03's leaked-title case
    # ("Customer Engineer, Machine Learning, Google Cloud - Doha").
    if "," in text and _ROLE_LEAK_RE.search(text):
        return {"matched": False, "brand": "", "reason": "role_string"}
    if company.exclude and re.search(company.exclude, text, re.IGNORECASE):
        return {"matched": False, "brand": "", "reason": "named_third_party"}
    if _AGENCY_RE.search(text):
        return {"matched": False, "brand": "", "reason": "third_party_marker"}

    return {"matched": True, "brand": company.brand_of(text), "reason": ""}


def select_company(df: pd.DataFrame, key: str,
                   column: str = "company_name") -> pd.DataFrame:
    """Rows of ``df`` whose employer belongs to ``key``, with the brand attached.

    Classifies distinct employer strings, not rows: the source has 785k rows
    and ~140k distinct employers, and the eight registry passes are otherwise
    the slowest step in the whole pipeline.
    """
    if df.empty:
        return df.assign(company_key=pd.Series(dtype=str),
                         company_brand=pd.Series(dtype=str))
    names = df[column].fillna("")
    verdicts = {name: classify_employer(name, key) for name in names.unique()}
    keep = names.map(lambda n: verdicts[n]["matched"])
    out = df[keep.to_numpy()].copy()
    out["company_key"] = key
    out["company_brand"] = names[keep.to_numpy()].map(
        lambda n: verdicts[n]["brand"]
    ).to_numpy()
    return out


def resolve_employers(df: pd.DataFrame, keys=SHORTLIST,
                      column: str = "company_name") -> pd.DataFrame:
    """Long table of every (company, matched rows) pair in ``keys``.

    A posting can in principle match two registry entries; none do in the 2023
    backfill, and ``matching_audit`` reports the count so the assumption is
    checked rather than assumed.
    """
    frames = []
    for key in keys:
        block = select_company(df, key, column)
        if len(block):
            frames.append(block)
    if not frames:
        return df.iloc[0:0].assign(company_key="", company_brand="")
    return pd.concat(frames, ignore_index=True)


def matching_audit(df: pd.DataFrame, keys=SHORTLIST,
                   column: str = "company_name") -> pd.DataFrame:
    """Every distinct employer string that mentions a shortlisted company.

    One row per (company, employer string): the posting count, the decision and
    the reason. This is the deliverable that makes the include/exclude patterns
    reviewable — a reader disagreeing with a call can point at the row.
    """
    rows = []
    counts = df[column].fillna("").value_counts()
    for key in keys:
        company = get_company(key)
        for name, n in counts.items():
            if not re.search(company.include, str(name), re.IGNORECASE):
                continue
            verdict = classify_employer(name, key)
            rows.append({
                "company": key,
                "employer_string": name,
                "postings": int(n),
                "decision": "matched" if verdict["matched"] else "excluded",
                "brand": verdict["brand"],
                "reason": verdict["reason"],
            })
    out = pd.DataFrame(rows, columns=["company", "employer_string", "postings",
                                      "decision", "brand", "reason"])
    if out.empty:
        return out
    return out.sort_values(["company", "decision", "postings"],
                           ascending=[True, True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feasibility screen
# ---------------------------------------------------------------------------

#: A company enters the comparison only above this floor.
#:
#: 250 postings so each half-year carries ~125 and a 10%-share estimate has a
#: 95% half-width under 6 percentage points — the smallest set on which the
#: share differences Task 06 reports can be told from noise. 10 of 12 months so
#: a company missing a stretch of the year cannot masquerade as a level
#: difference. Both are arbitrary in the sense that any threshold is; they are
#: declared before the comparison rather than chosen after seeing it.
MIN_POSTINGS = 250
MIN_MONTHS = 10


def feasibility_screen(df: pd.DataFrame, keys=SHORTLIST,
                       column: str = "company_name",
                       date_col: str = "posting_date",
                       min_postings: int = MIN_POSTINGS,
                       min_months: int = MIN_MONTHS) -> pd.DataFrame:
    """Which shortlisted companies carry enough data to be compared.

    Runs *before* any comparison, on the collected source, and reports the
    verdict for every candidate — including the failures, which are a finding
    about the source rather than about the companies.
    """
    rows = []
    for key in keys:
        block = select_company(df, key, column)
        dates = pd.to_datetime(block[date_col], errors="coerce").dropna() \
            if len(block) and date_col in block.columns else pd.Series(dtype="datetime64[ns]")
        months = int(dates.dt.to_period("M").nunique()) if len(dates) else 0
        publishers = (
            int(block["job_via"].nunique()) if "job_via" in block.columns else 0
        )
        reasons = []
        if len(block) < min_postings:
            reasons.append(f"under {min_postings} postings")
        if months < min_months:
            reasons.append(f"present in {months}/12 months")
        rows.append({
            "company": key,
            "display": get_company(key).display,
            "postings": int(len(block)),
            "months_present": months,
            "publishers": publishers,
            "first_posting": str(dates.min().date()) if len(dates) else "",
            "last_posting": str(dates.max().date()) if len(dates) else "",
            "verdict": "included" if not reasons else "excluded_low_support",
            "reason": "; ".join(reasons),
        })
    return pd.DataFrame(rows).sort_values(
        ["verdict", "postings"], ascending=[True, False]
    ).reset_index(drop=True)


def included_companies(screen: pd.DataFrame) -> list[str]:
    """Companies that passed the screen, in a stable order."""
    return sorted(screen.loc[screen.verdict == "included", "company"].tolist())
