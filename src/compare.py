"""Task 06 — competitor comparison: the shared cross-company structure.

Shared, company-agnostic comparison layer for all four specialists. The team
standard and its rationale live in docs/task-06-competitor-comparison-methods.md.

Why this module exists
----------------------
Task 05 established that a *single* company's posting volume is observed
through a shifting set of publishers, so its direction of travel can be an
artefact of collection. Comparing companies makes that worse in a specific
way: the artefact is now **different for each company**, and it does not cancel.

Three confounders sit between "postings we observed" and "hiring we want to
compare". Every function here exists to control one of them:

1. **Channel.** Each company is syndicated by a different set of job boards,
   and some publish heavily through their own careers site. Snowflake reaches
   us through 37 publishers with 45% of its postings on its own channel;
   Google through 95. A raw count comparison is largely a syndication
   comparison. -> ``common_publishers`` / ``common_panel_table``.

2. **Role mix.** Companies are not hiring for the same jobs. 70% of
   Snowflake's postings are Engineering; 44% of Meta's are Analytics; 11% of
   Google's are data-centre Facilities. Any pooled rate — skill coverage,
   skill share, seniority — is a weighted average over that mix, so a pooled
   difference can be pure composition. -> ``standardise`` (direct
   standardisation to a pooled mix) and ``stratified_company_verdict``.

3. **Measurement.** Skill coverage runs 70.8% (Google) to 100.0%
   (Databricks). That is upstream extraction quality varying by channel and by
   role, not a fact about the companies. It is why every skill share uses
   ``share_of_skilled`` and why coverage itself is reported as a comparability
   diagnostic rather than a finding. -> ``comparability_table``.

The gate comes before the comparison
------------------------------------
``comparability_table`` runs first and is a committed deliverable in its own
right. A pair of companies that fails it does not get a number with a caveat
attached; it gets ``identified = False`` and the comparison is reported as
unavailable. Task 05's lesson was that four defensible treatments of the same
data spanned -9% to +91%; the response is to say so, not to pick one.

What this module deliberately does not do
-----------------------------------------
* **No country comparison.** Only 21 of 48 countries kept their direction on
  Google's balanced panel (Task 05 §7), and country coverage differs by
  publisher, so a cross-company country split compares crawlers.
* **No ``share_of_all``.** Every skill rate is over *skilled* postings, with
  ``Facilities / Operations`` excluded from the denominator (Task 04 §3.1).
* **No scipy.** Wilson, Newcombe and Benjamini-Hochberg are implemented here
  in numpy so the core modules keep one dependency story.

Usage
-----
    import compare as cmp
    frames = cmp.load_frames(["google", "meta"])
    gate = cmp.comparability_table(frames)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import trends as tr

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"

PUBLISHER_COL = tr.PUBLISHER_COL
SKILL_EXCLUDED_FUNCTIONS = tr.SKILL_EXCLUDED_FUNCTIONS

#: Columns no Task 06 table may contain. `country` because the panel does not
#: support it; `share_of_all` because it is the denominator Task 04 retired.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset({
    "country", "location_country", "share_of_all",
})

#: A company must reach the common publisher panel with at least this share of
#: its postings before a *level* comparison is even attempted. Set from the
#: observed spread (23.5%-67.5%): at 40% the panel is carrying a minority of
#: the thinnest company's postings and the level is a channel statistic.
MIN_COMMON_PANEL_SHARE = 0.40

#: Minimum postings in a cell before a rate is reported. Matches Task 05's
#: `min_postings` so a within-function comparison here and a within-function
#: trend there are supported by the same evidence floor.
MIN_CELL = 10

#: Skills each company **sells or created**. Task 06's own trap: Databricks
#: leads the sector on "Databricks" (421 of 2,373 rival postings mention it,
#: 0 of Google's), and reporting that as a demand signal would be reporting a
#: tautology. Two relations, because they read differently:
#:
#: ``own_product``  the company builds and sells it. A posting mentioning it
#:                  is near-certain to be self-reference, so the skill is
#:                  uninformative about that company's *demand*.
#: ``origin``       the company created it and it is now independently
#:                  governed (Kubernetes -> CNCF, Beam/Keras -> ASF/Keras
#:                  team). Still a bias, but a much weaker one — the wider
#:                  market genuinely adopted these.
#:
#: The list is a judgement call and therefore committed, versioned and
#: reviewable rather than applied silently: `skill_distinctiveness` flags
#: these rows, it never drops them.
VENDOR_SKILLS: dict[str, dict[str, str]] = {
    "google": {
        "BigQuery": "own_product", "Bigtable": "own_product",
        "GCP": "own_product", "Google Analytics": "own_product",
        "Google Sheets": "own_product", "Data Studio": "own_product",
        "Looker": "own_product", "Spanner": "own_product",
        "Firebase": "own_product", "Vertex AI": "own_product",
        "TensorFlow": "origin", "JAX": "origin", "Keras": "origin",
        "Kubernetes": "origin", "Kubeflow": "origin", "Apache Beam": "origin",
        "Go": "origin", "Angular": "origin", "Dart": "origin",
        "Flutter": "origin", "Bazel": "origin",
    },
    "microsoft": {
        "Azure": "own_product", "Power BI": "own_product",
        "Excel": "own_product", "PowerPoint": "own_product",
        "Microsoft Word": "own_product", "Microsoft Outlook": "own_product",
        "Microsoft Access": "own_product", "SharePoint": "own_product",
        "SQL Server": "own_product", "VBA": "own_product",
        "PowerShell": "own_product", "Windows": "own_product",
        "GitHub": "own_product", ".NET": "own_product",
        "C#": "origin", "TypeScript": "origin", "LightGBM": "origin",
        "ONNX": "origin",
    },
    "meta": {
        "PyTorch": "own_product", "React": "own_product",
        "Presto": "origin", "Cassandra": "origin", "Hive": "origin",
    },
    "nvidia": {"CUDA": "own_product", "TensorRT": "own_product"},
    "snowflake": {"Snowflake": "own_product"},
    "databricks": {
        "Databricks": "own_product", "Delta Lake": "own_product",
        "MLflow": "own_product", "Spark": "origin",
    },
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_frames(keys, root: Path = PROCESSED) -> dict[str, pd.DataFrame]:
    """Read each company's posting-level feature table.

    Row-level data stays git-ignored, so this only works after
    ``python src/build_competitor_set.py`` has been run locally. Every
    committed table is derived from these frames, never shipped with them.
    """
    out = {}
    for key in keys:
        path = Path(root) / key / f"{key}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing - run src/build_competitor_set.py first"
            )
        out[key] = pd.read_parquet(path)
    return out


def load_long(keys, root: Path = PROCESSED) -> dict[str, pd.DataFrame]:
    """Read each company's posting-skill long table."""
    return {
        key: pd.read_parquet(Path(root) / key / f"{key}_skills_long.parquet")
        for key in keys
    }


def stack(frames: dict[str, pd.DataFrame], company_col: str = "company") -> pd.DataFrame:
    """One long frame with a ``company`` column, for pooled calculations."""
    parts = []
    for key, df in frames.items():
        part = df.copy()
        part[company_col] = key
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Uncertainty, without scipy
# ---------------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Score interval for a proportion.

    Wilson rather than Wald because these cells are small and the shares are
    near 0 or 1 exactly where the interesting skills live: Wald gives Meta's
    3/3 an interval of [1.0, 1.0], which would let a three-posting cell
    outrank a three-hundred-posting one.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe_diff_interval(k1: int, n1: int, k2: int, n2: int,
                           z: float = 1.96) -> tuple[float, float]:
    """Interval for ``p1 - p2`` (Newcombe's method 10).

    Built from the two Wilson intervals rather than from a pooled standard
    error, so it stays sane when one company's cell is tiny — which is the
    normal case here, not the exception.
    """
    if n1 <= 0 or n2 <= 0:
        return (float("nan"), float("nan"))
    l1, u1 = wilson_interval(k1, n1, z)
    l2, u2 = wilson_interval(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value for ``p1 == p2``, pooled-variance z test.

    Only ever used as an input to `bh_fdr`; the reported uncertainty is the
    Newcombe interval, because an interval says how big the difference might
    be and a p-value does not.
    """
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def bh_fdr(pvalues, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg q-values and rejections.

    Task 06 tests every skill against every company. At 91 skills and 6
    companies an uncorrected 5% threshold manufactures roughly 27 "findings"
    from noise alone, which is more than the report has room for. BH controls
    the false *discovery* rate, which is the right error to control when the
    output is a ranked shortlist rather than a single decision.
    """
    p = pd.Series(pvalues, dtype=float).reset_index(drop=True)
    ok = p.notna()
    m = int(ok.sum())
    q = pd.Series(np.nan, index=p.index, dtype=float)
    if m:
        order = p[ok].sort_values().index
        ranked = p.loc[order].to_numpy() * m / np.arange(1, m + 1)
        # Monotone from the largest rank down, so a small p can never get a
        # larger q than a bigger one.
        q.loc[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return pd.DataFrame({"p_value": p, "q_value": q,
                         "significant": (q <= alpha).fillna(False)})


# ---------------------------------------------------------------------------
# Comparability — the gate
# ---------------------------------------------------------------------------


def own_channel_publishers(df: pd.DataFrame, key: str) -> list[str]:
    """Publishers that are the company itself (``via Snowflake Careers``).

    An own-channel posting is observed because the employer published it, not
    because an aggregator picked it up. It is the least comparable channel
    there is: its volume tracks the company's own site policy, and Task 04
    measured its skill coverage at 19.1% against 82.6% elsewhere.
    """
    import companies as co  # local import: keeps compare.py usable standalone

    pubs = sorted(df[PUBLISHER_COL].dropna().unique())
    return [p for p in pubs if co.classify_employer(p, key)["matched"]]


def comparability_table(frames: dict[str, pd.DataFrame],
                        period: str = "month",
                        min_share: float = 0.75) -> pd.DataFrame:
    """Per-company diagnostics that decide what may be compared at all.

    Read the columns as a checklist, not a description: ``common_panel_share``
    gates level comparisons, ``treatments_agree`` gates trend comparisons, and
    ``skill_coverage`` gates any comparison whose denominator is skilled
    postings.
    """
    common = common_publishers(frames)
    rows = []
    for key, df in sorted(frames.items()):
        panel = tr.panel_publishers(df, period, min_share=min_share)
        own = own_channel_publishers(df, key)
        sens = tr.panel_sensitivity_table(df, period, min_share=min_share)
        verdict = tr.panel_verdict(sens)
        months = df.posting_month.nunique()
        on_panel = df[PUBLISHER_COL].isin(panel).mean() if panel else 0.0
        skilled = df[~df.job_function.isin(SKILL_EXCLUDED_FUNCTIONS)]
        top = df[PUBLISHER_COL].value_counts(normalize=True)
        rows.append({
            "company": key,
            "postings": int(len(df)),
            "publishers": int(df[PUBLISHER_COL].nunique()),
            "months_observed": int(months),
            "balanced_panel_publishers": len(panel),
            "balanced_panel_share": round(float(on_panel), 4),
            "common_panel_share": round(
                float(df[PUBLISHER_COL].isin(common).mean()), 4),
            "own_channel_publishers": len(own),
            "own_channel_share": round(
                float(df[PUBLISHER_COL].isin(own).mean()), 4),
            "top_publisher": top.index[0] if len(top) else "",
            "top_publisher_share": round(float(top.iloc[0]), 4) if len(top) else 0.0,
            "trend_direction": verdict.direction,
            "treatments_agree": bool(verdict.agrees),
            "skill_coverage": round(float(df.has_any_skill.mean()), 4),
            "skill_coverage_ex_facilities": round(
                float(skilled.has_any_skill.mean()), 4) if len(skilled) else 0.0,
            "facilities_share": round(
                float(df.job_function.isin(SKILL_EXCLUDED_FUNCTIONS).mean()), 4),
        })
    out = pd.DataFrame(rows)
    out["level_comparable"] = out.common_panel_share >= MIN_COMMON_PANEL_SHARE
    return out


@dataclass(frozen=True)
class LevelVerdict:
    """Whether *relative posting volume* between companies is identified."""

    identified: bool
    reason: str
    detail: dict


def level_verdict(gate: pd.DataFrame,
                  min_share: float = MIN_COMMON_PANEL_SHARE) -> LevelVerdict:
    """Can we say company A posts more than company B?

    Almost certainly not, and this function's job is to say so in a way the
    report cannot quietly skip. The check is deliberately blunt: if any
    company reaches the common panel with less than ``min_share`` of its
    postings, the panel is measuring channel choice for that company and the
    level ranking is not identified.
    """
    failing = gate.loc[gate.common_panel_share < min_share, "company"].tolist()
    detail = dict(zip(gate.company, gate.common_panel_share.round(3)))
    if failing:
        return LevelVerdict(
            False,
            "common-panel coverage below the floor for "
            + ", ".join(f"{c} ({detail[c]:.0%})" for c in failing),
            detail,
        )
    return LevelVerdict(True, "all companies clear the common-panel floor", detail)


# ---------------------------------------------------------------------------
# The common publisher panel — like-for-like channels
# ---------------------------------------------------------------------------


def common_publishers(frames: dict[str, pd.DataFrame],
                      min_postings: int = 1) -> list[str]:
    """Publishers that carry **every** company.

    The cross-company analogue of Task 05's balanced panel. Comparing two
    companies over publishers that only one of them uses compares
    distribution deals, so the intersection is the only channel-neutral view
    available — and it is small, which is itself a reportable finding.
    """
    sets = []
    for df in frames.values():
        counts = df[PUBLISHER_COL].value_counts()
        sets.append(set(counts.index[counts >= min_postings]))
    return sorted(set.intersection(*sets)) if sets else []


def common_panel_table(frames: dict[str, pd.DataFrame],
                       publishers: list[str] | None = None) -> pd.DataFrame:
    """Postings per company on each shared publisher, long form."""
    pubs = common_publishers(frames) if publishers is None else publishers
    rows = []
    for key, df in sorted(frames.items()):
        on = df[df[PUBLISHER_COL].isin(pubs)]
        for pub in pubs:
            block = on[on[PUBLISHER_COL] == pub]
            rows.append({
                "company": key,
                "publisher": pub,
                "postings": int(len(block)),
                "share_of_company_postings": round(len(block) / len(df), 4)
                if len(df) else 0.0,
                "skill_coverage": round(float(block.has_any_skill.mean()), 4)
                if len(block) else float("nan"),
            })
    return pd.DataFrame(rows)


def common_panel_by_period(frames: dict[str, pd.DataFrame],
                           period: str = "month") -> pd.DataFrame:
    """How many publishers are shared by all companies *within* each period.

    The window-level intersection flatters the data: a publisher can serve
    Google in March and Meta in September and still count as "common" over the
    year. Within a month the intersection collapses, and a period where it
    hits zero cannot support any like-for-like comparison at all.
    """
    col = tr.period_col(period)
    keys = sorted(set().union(*[set(df[col].dropna()) for df in frames.values()]))
    rows = []
    for k in keys:
        sets = [set(df.loc[df[col] == k, PUBLISHER_COL].dropna())
                for df in frames.values()]
        shared = set.intersection(*sets) if sets else set()
        rows.append({
            "period": k,
            "n_common_publishers": len(shared),
            "n_companies_present": int(sum(bool(s) for s in sets)),
            "publishers": ", ".join(sorted(shared)),
        })
    return pd.DataFrame(rows)


def restrict(frames: dict[str, pd.DataFrame],
             publishers: list[str]) -> dict[str, pd.DataFrame]:
    """Every frame cut to a publisher set — the channel-controlled replicate."""
    return {k: df[df[PUBLISHER_COL].isin(publishers)].copy()
            for k, df in frames.items()}


# ---------------------------------------------------------------------------
# Volume: trends, not levels
# ---------------------------------------------------------------------------


def volume_panel_table(frames: dict[str, pd.DataFrame], period: str = "month",
                       min_share: float = 0.75) -> pd.DataFrame:
    """Task 05's four-treatment sensitivity table, stacked over companies."""
    parts = []
    for key, df in sorted(frames.items()):
        sens = tr.panel_sensitivity_table(df, period, min_share=min_share)
        sens.insert(0, "company", key)
        parts.append(sens)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def volume_verdict_table(frames: dict[str, pd.DataFrame], period: str = "month",
                         min_share: float = 0.75,
                         flat_band: float = 10.0) -> pd.DataFrame:
    """One row per company: does its own direction survive all four treatments?

    This is the only volume comparison Task 06 is entitled to make. It
    compares *verdicts* — "identified growth" vs "not identified" — not
    magnitudes, because each company's index is built on its own publisher
    panel and the indices are not on a common scale.
    """
    rows = []
    for key, df in sorted(frames.items()):
        sens = tr.panel_sensitivity_table(df, period, min_share=min_share)
        v = tr.panel_verdict(sens, flat_band=flat_band)
        row = {"company": key, "direction": v.direction, "treatments_agree": v.agrees}
        row.update({f"{k}_index_final": val for k, val in v.detail.items()})
        row["spread"] = (round(max(v.detail.values()) - min(v.detail.values()), 2)
                         if v.detail else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def half_over_half(frames: dict[str, pd.DataFrame],
                   min_share: float = 0.75) -> pd.DataFrame:
    """H2 vs H1 on raw and balanced postings.

    Task 05 found this to be the one volume statement Google's data supports
    (H2 > H1 on both treatments) precisely because a half-year aggregates over
    the publisher churn that wrecks the month-to-month index. It is reported
    per company and only called a finding where both treatments agree.
    """
    rows = []
    for key, df in sorted(frames.items()):
        panel = tr.panel_publishers(df, "month", min_share=min_share)
        half = df.posting_month.map(tr.half_of)
        bal = df[df[PUBLISHER_COL].isin(panel)]
        bal_half = bal.posting_month.map(tr.half_of)
        raw_h1, raw_h2 = int((half == "h1").sum()), int((half == "h2").sum())
        bal_h1, bal_h2 = int((bal_half == "h1").sum()), int((bal_half == "h2").sum())
        raw_chg = (raw_h2 / raw_h1 - 1) * 100 if raw_h1 else float("nan")
        bal_chg = (bal_h2 / bal_h1 - 1) * 100 if bal_h1 else float("nan")
        rows.append({
            "company": key,
            "raw_h1": raw_h1, "raw_h2": raw_h2, "raw_pct_change": round(raw_chg, 1),
            "balanced_h1": bal_h1, "balanced_h2": bal_h2,
            "balanced_pct_change": round(bal_chg, 1),
            "agrees": bool(np.sign(raw_chg) == np.sign(bal_chg)),
        })
    return pd.DataFrame(rows)


def collection_artefact_table(frames: dict[str, pd.DataFrame],
                              period: str = "month") -> pd.DataFrame:
    """Is a shared monthly pattern the sector, or the crawler?

    Six independent employers cannot make job boards disappear. So if a month
    is thin *and* the number of distinct publishers observed across all six
    collapses with it, the thinness is collection; if a month is thick while
    the publisher count falls, it cannot be collection expansion.

    That second case is what makes Task 06's half-over-half result usable.
    Every company shows H2 > H1, which on its own is exactly what a crawler
    ramping up its coverage would produce — but the pooled publisher count
    *falls* from 66 in January to 51 in December while postings per observed
    publisher rise from 5.0 to 7.6. The rise is more postings per board, not
    more boards, which is the one direction a coverage artefact cannot fake.
    """
    col = tr.period_col(period)
    stacked = stack(frames)
    counts = stacked.groupby([col, "company"]).size().unstack(fill_value=0)
    median = counts.median()
    out = pd.DataFrame({
        "period": counts.index,
        "postings": counts.sum(axis=1).to_numpy(),
        "distinct_publishers": stacked.groupby(col)[PUBLISHER_COL]
                                      .nunique().reindex(counts.index).to_numpy(),
        "companies_below_own_median": (counts < median).sum(axis=1).to_numpy(),
        "n_companies": len(counts.columns),
    })
    out["postings_per_publisher"] = (out.postings / out.distinct_publishers).round(2)
    # A month every company is thin in, with a thin publisher roster, is the
    # collector's month rather than the sector's.
    out["shared_thin_month"] = (
        (out.companies_below_own_median >= len(counts.columns) - 1)
        & (out.distinct_publishers < out.distinct_publishers.median())
    )
    return out


def company_correlation(frames: dict[str, pd.DataFrame],
                        period: str = "month") -> pd.DataFrame:
    """Pairwise correlation of the companies' period counts.

    A common collection factor would push every pair towards 1. The observed
    mean is 0.32 with two negative pairs, so whatever is shared is not
    dominating the series — which is the other half of the argument that the
    relative-share estimator has something left to measure.
    """
    col = tr.period_col(period)
    counts = stack(frames).groupby([col, "company"]).size().unstack(fill_value=0)
    return counts.corr().round(3).rename_axis(index="company", columns="company")


def relative_share_table(frames: dict[str, pd.DataFrame],
                         publishers: list[str] | None = None) -> pd.DataFrame:
    """Each company's **share of the shared-publisher panel**, H1 vs H2.

    This is Task 06's headline volume estimator, and the reasoning behind it
    is the whole reason the module exists.

    All six companies are observed through the same crawler. Whatever that
    crawler did to its coverage during 2023 multiplies every company's counts
    by roughly the same publisher-period factor — which is why all six show
    H2 > H1 and why none of that is evidence about hiring. Take shares of the
    same publisher-period pool and the common factor divides out. What remains
    is *relative* movement: whose postings grew faster than the sector's on
    channels every company actually used.

    The identifying assumption is stated rather than hidden: a company's
    propensity to syndicate onto these shared publishers is constant between
    the halves. It is not testable here, but it is checkable in one direction
    — ``relative_share_by_publisher`` recomputes the same quantity inside each
    publisher separately, and a company whose sign flips between publishers
    has no relative claim worth reporting.

    Levels stay unidentified. This says Snowflake fell *relative to the panel*,
    never that Snowflake posts fewer jobs than Google.
    """
    pubs = common_publishers(frames) if publishers is None else publishers
    cut = restrict(frames, pubs)
    rows = []
    for key, df in sorted(cut.items()):
        half = df.posting_month.map(tr.half_of)
        rows.append({"company": key,
                     "h1_postings": int((half == "h1").sum()),
                     "h2_postings": int((half == "h2").sum())})
    out = pd.DataFrame(rows)
    t1, t2 = out.h1_postings.sum(), out.h2_postings.sum()
    out["h1_share"] = (out.h1_postings / t1).round(4) if t1 else float("nan")
    out["h2_share"] = (out.h2_postings / t2).round(4) if t2 else float("nan")
    out["share_change_pp"] = ((out.h2_share - out.h1_share) * 100).round(2)
    # Log ratio, so a halving and a doubling are the same size with opposite
    # signs; the +0.5 keeps an empty half from returning -inf.
    out["log_share_change"] = np.log(
        ((out.h2_postings + 0.5) / (t2 + 0.5 * len(out)))
        / ((out.h1_postings + 0.5) / (t1 + 0.5 * len(out)))
    ).round(3)
    out["panel_h1_total"], out["panel_h2_total"] = int(t1), int(t2)
    out.attrs["publishers"] = pubs
    return out


def relative_share_by_publisher(frames: dict[str, pd.DataFrame],
                                publishers: list[str] | None = None,
                                min_half: int = 10) -> pd.DataFrame:
    """`relative_share_table` recomputed inside each shared publisher.

    A publisher whose own half carries fewer than ``min_half`` postings across
    all companies is dropped: its share change is one or two postings moving.
    """
    pubs = common_publishers(frames) if publishers is None else publishers
    cut = restrict(frames, pubs)
    rows = []
    for pub in pubs:
        block = {k: df[df[PUBLISHER_COL] == pub] for k, df in cut.items()}
        halves = {k: df.posting_month.map(tr.half_of) for k, df in block.items()}
        h1 = {k: int((v == "h1").sum()) for k, v in halves.items()}
        h2 = {k: int((v == "h2").sum()) for k, v in halves.items()}
        t1, t2 = sum(h1.values()), sum(h2.values())
        if t1 < min_half or t2 < min_half:
            continue
        n = len(block)
        for key in sorted(block):
            rows.append({
                "publisher": pub, "company": key,
                "h1_postings": h1[key], "h2_postings": h2[key],
                "publisher_h1_total": t1, "publisher_h2_total": t2,
                "log_share_change": round(float(np.log(
                    ((h2[key] + 0.5) / (t2 + 0.5 * n))
                    / ((h1[key] + 0.5) / (t1 + 0.5 * n)))), 3),
            })
    return pd.DataFrame(rows)


def relative_share_verdict(by_publisher: pd.DataFrame,
                           pooled: pd.DataFrame) -> pd.DataFrame:
    """Does a company's relative move hold on every shared publisher?

    Same rule as Task 05's panel verdict, moved sideways: agreement on
    **sign** across independent channels, not agreement on magnitude.
    ``confirmed`` requires every qualifying publisher to point the same way as
    the pooled estimate.
    """
    if by_publisher.empty:
        return pd.DataFrame()
    pooled_sign = dict(zip(pooled.company, np.sign(pooled.log_share_change)))
    rows = []
    for key, block in by_publisher.groupby("company"):
        signs = np.sign(block.log_share_change)
        agree = int((signs == pooled_sign.get(key, 0)).sum())
        n = int(len(block))
        rows.append({
            "company": key,
            "pooled_log_share_change": float(
                pooled.loc[pooled.company == key, "log_share_change"].iloc[0]),
            "publishers_tested": n, "publishers_agreeing": agree,
            "verdict": ("confirmed" if agree == n else
                        "mixed" if agree > n / 2 else "not supported"),
            "direction": ("gaining share" if pooled_sign.get(key, 0) > 0
                          else "losing share" if pooled_sign.get(key, 0) < 0
                          else "flat"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Role mix and direct standardisation
# ---------------------------------------------------------------------------


def mix_table(frames: dict[str, pd.DataFrame], by: str = "job_function",
              z: float = 1.96) -> pd.DataFrame:
    """Share of each company's postings in each stratum, with Wilson intervals.

    The mix is a finding *and* the nuisance parameter for everything else. It
    is reported first so a reader can see which later differences are already
    explained by it.
    """
    rows = []
    for key, df in sorted(frames.items()):
        n = len(df)
        counts = df[by].fillna("Unknown").value_counts()
        for stratum, k in counts.items():
            lo, hi = wilson_interval(int(k), n, z)
            rows.append({
                "company": key, by: stratum, "postings": int(k),
                "company_postings": n, "share": round(k / n, 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            })
    return pd.DataFrame(rows).sort_values(["company", "share"],
                                          ascending=[True, False]).reset_index(drop=True)


def mix_distance(mix: pd.DataFrame, by: str = "job_function") -> pd.DataFrame:
    """Pairwise total-variation distance between companies' mixes.

    Half the sum of absolute share differences: 0 means identical role mixes,
    1 means disjoint. It quantifies how much standardisation is doing — a pair
    at 0.5 cannot be compared on any pooled rate without it.
    """
    wide = mix.pivot_table(index=by, columns="company", values="share",
                           fill_value=0.0)
    keys = list(wide.columns)
    out = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for a in keys:
        for b in keys:
            out.loc[a, b] = round(float((wide[a] - wide[b]).abs().sum() / 2), 4)
    return out.rename_axis(index="company", columns="company")


def pooled_weights(frames: dict[str, pd.DataFrame],
                   by: str = "job_function") -> pd.Series:
    """The standard population: the pooled mix over all compared companies.

    Pooled rather than one company's mix, so no company is the implicit
    reference and the standardised rates stay symmetric. Weights are recomputed
    whenever the compared set changes, which is why every standardised table
    ships the weights it used.
    """
    pooled = pd.concat([df[[by]] for df in frames.values()], ignore_index=True)
    return (pooled[by].fillna("Unknown").value_counts(normalize=True)
            .rename("weight").rename_axis(by))


def standardise(df: pd.DataFrame, value_col: str, weights: pd.Series,
                by: str = "job_function", min_cell: int = MIN_CELL) -> dict:
    """Direct standardisation of a 0/1 rate to a common stratum mix.

    Computes the rate within each stratum, then reweights to ``weights``:
    the rate this company would show **if it hired the same mix of roles as
    the sector**. Strata below ``min_cell`` are dropped and the remaining
    weights renormalised, so a 2-posting cell cannot swing the answer; the
    share of the standard population actually covered comes back as
    ``weight_covered`` and should be read alongside the rate.
    """
    grp = df.groupby(df[by].fillna("Unknown"))[value_col]
    stats = pd.DataFrame({"n": grp.size(), "k": grp.sum()})
    stats = stats[stats.n >= min_cell]
    if stats.empty:
        return {"crude": float(df[value_col].mean()) if len(df) else float("nan"),
                "standardised": float("nan"), "weight_covered": 0.0,
                "strata_used": 0}
    w = weights.reindex(stats.index).fillna(0.0)
    covered = float(w.sum())
    rate = float((stats.k / stats.n * w).sum() / covered) if covered else float("nan")
    return {
        "crude": float(df[value_col].mean()),
        "standardised": rate,
        "weight_covered": round(covered, 4),
        "strata_used": int(len(stats)),
    }


def standardised_table(frames: dict[str, pd.DataFrame], value_col: str,
                       by: str = "job_function",
                       min_cell: int = MIN_CELL) -> pd.DataFrame:
    """`standardise` for every company, crude and adjusted side by side.

    The gap between the two columns is the part of a raw difference that was
    role mix. Reporting both is the point: a difference that vanishes under
    standardisation was never about the companies.
    """
    w = pooled_weights(frames, by)
    rows = []
    for key, df in sorted(frames.items()):
        r = standardise(df, value_col, w, by, min_cell)
        rows.append({"company": key, "metric": value_col,
                     "crude": round(r["crude"], 4),
                     "standardised": round(r["standardised"], 4),
                     "mix_effect": round(r["crude"] - r["standardised"], 4),
                     "weight_covered": r["weight_covered"],
                     "strata_used": r["strata_used"],
                     "postings": int(len(df))})
    out = pd.DataFrame(rows)
    out.attrs["weights"] = w
    return out


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def skill_denominator(features: pd.DataFrame) -> pd.DataFrame:
    """Postings eligible to be a skill denominator.

    Skilled postings, with ``Facilities / Operations`` removed. Both filters
    are inherited, not chosen here: Task 04 §3.1 showed facilities roles are
    true negatives that drag every share down, and Task 04's denominator flip
    established ``share_of_skilled``. Because the facilities share differs
    tenfold across companies (11.2% Google, 0% Snowflake), skipping either
    filter would make the cross-company comparison worse than the
    within-company one.
    """
    keep = features.has_any_skill & ~features.job_function.isin(SKILL_EXCLUDED_FUNCTIONS)
    return features[keep]


def skill_share_table(longs: dict[str, pd.DataFrame],
                      frames: dict[str, pd.DataFrame],
                      min_postings: int = MIN_CELL,
                      z: float = 1.96) -> pd.DataFrame:
    """Per company per skill: ``share_of_skilled`` with a Wilson interval.

    Skills below ``min_postings`` for a company still appear, with their
    interval, rather than being dropped — an absent skill and a rare one are
    different claims, and the ``supported`` flag keeps them apart without
    hiding either.
    """
    denom = {k: len(skill_denominator(df)) for k, df in frames.items()}
    eligible = {k: set(skill_denominator(df).job_id) for k, df in frames.items()}
    rows = []
    for key, long in sorted(longs.items()):
        sub = long[long.job_id.isin(eligible[key])]
        n = denom[key]
        counts = sub.groupby("skill").job_id.nunique()
        cats = sub.drop_duplicates("skill").set_index("skill").skill_category
        for skill, k in counts.items():
            lo, hi = wilson_interval(int(k), n, z)
            rows.append({
                "company": key, "skill": skill,
                "skill_category": cats.get(skill, ""),
                "postings_with_skill": int(k), "n_skilled": n,
                "share_of_skilled": round(k / n, 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "supported": bool(k >= min_postings),
            })
    return pd.DataFrame(rows)


def self_reference_table(longs: dict[str, pd.DataFrame],
                         frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How much of each company's skill signal is its own name coming back.

    Task 04 found the employer's brand leaking into *titles* and wrote
    `strip_org_segments` to remove it. The same leak exists one level up, in
    the skills the upstream dataset extracted from descriptions, and it is
    severe exactly where the company name is itself a token in the taxonomy:
    "Snowflake" and "Databricks" are both employers and skills, so 99% of
    those companies' postings carry their own name as a skill. "Google" and
    "NVIDIA" are not skill tokens, so they show 14% and 0.4%.

    Two columns matter and they answer different questions.
    ``coverage_excl_own_product`` asks whether the leak manufactured the
    company's skill coverage — it did not, it is worth 1.5 to 3.2 points.
    ``own_product_share`` asks whether the leak dominates its skill profile —
    for two companies it entirely does.
    """
    rows = []
    for key in sorted(frames):
        f, long = frames[key], longs[key]
        own = [s for s, rel in VENDOR_SKILLS.get(key, {}).items()
               if rel == "own_product"]
        with_own = set(long.loc[long.skill.isin(own), "job_id"])
        with_other = set(long.loc[~long.skill.isin(own), "job_id"])
        n = len(f)
        hits = long[long.skill.isin(own)]
        rows.append({
            "company": key, "postings": n,
            "own_product_skills": len(own),
            "postings_with_own_product": len(with_own),
            "own_product_share": round(len(with_own) / n, 4) if n else 0.0,
            "coverage": round(float(f.has_any_skill.mean()), 4),
            "coverage_excl_own_product": round(len(with_other) / n, 4) if n else 0.0,
            "coverage_inflation_pp": round(
                (len(with_own - with_other)) / n * 100, 2) if n else 0.0,
            "top_own_product": (hits.skill.value_counts().index[0]
                                if len(hits) else ""),
            "provenance": ", ".join(
                f"{p} ({c})" for p, c in hits.provenance.value_counts().items()),
        })
    return pd.DataFrame(rows)


def skill_matrix(share: pd.DataFrame, value: str = "share_of_skilled",
                 min_companies: int = 2) -> pd.DataFrame:
    """Skill x company wide table, zero-filled.

    A skill absent from a company is a genuine 0 here — the denominator is
    that company's skilled postings, which exist — so zero-filling is a
    measurement, not an imputation.
    """
    wide = share.pivot_table(index="skill", columns="company", values=value,
                             fill_value=0.0)
    present = (wide > 0).sum(axis=1)
    return wide[present >= min_companies].sort_index()


def skill_distinctiveness(share: pd.DataFrame, company: str,
                          min_postings: int = MIN_CELL,
                          alpha: float = 0.05, z: float = 1.96) -> pd.DataFrame:
    """How much more (or less) a company asks for each skill than its rivals.

    The comparison is against the **pooled rest of the sector**, not against
    one chosen rival, so the answer does not depend on who else happens to be
    in the set. ``log_lift`` is reported because ratios of small shares are
    unreadable on a linear axis; the Newcombe interval and the BH q-value do
    the inferential work, and ``q_value`` is what a shortlist should be cut on.
    """
    if company not in set(share.company):
        return pd.DataFrame()
    # Denominators are one per company, so read them once rather than per skill.
    denoms = share.drop_duplicates("company").set_index("company").n_skilled
    n1 = int(denoms.loc[company])
    others_n = int(denoms.drop(index=company).sum())
    mine = share[share.company == company].set_index("skill")
    rest = (share[share.company != company]
            .groupby("skill").postings_with_skill.sum())
    cats = share.drop_duplicates("skill").set_index("skill").skill_category
    # Union, not `mine` alone: a skill the whole sector asks for and this
    # company does not is a finding, and it has k1 = 0 rather than no row.
    rows = []
    for skill in sorted(set(mine.index) | set(rest.index)):
        k1 = int(mine.postings_with_skill.get(skill, 0))
        k2 = int(rest.get(skill, 0))
        p1, p2 = k1 / n1, (k2 / others_n if others_n else float("nan"))
        lo, hi = newcombe_diff_interval(k1, n1, k2, others_n, z)
        rows.append({
            "company": company, "skill": skill,
            "skill_category": cats.get(skill, ""),
            "postings_with_skill": k1, "n_skilled": n1, "share": round(p1, 4),
            "rest_with_skill": k2, "rest_n_skilled": others_n,
            "rest_share": round(p2, 4) if others_n else float("nan"),
            "diff": round(p1 - p2, 4) if others_n else float("nan"),
            "diff_ci_low": round(lo, 4), "diff_ci_high": round(hi, 4),
            # Haldane-Anscombe correction, not an epsilon: a zero cell has to
            # produce a finite, comparable lift. With a bare 1e-9 the absent
            # skills all collapse onto -23 to -27 and the axis becomes a list
            # of denominators rather than a measurement.
            "log_lift": round(float(np.log2(
                ((k1 + 0.5) / (n1 + 1)) / ((k2 + 0.5) / (others_n + 1)))), 3)
            if others_n else float("nan"),
            "p_value": two_proportion_p(k1, n1, k2, others_n),
            "supported": bool(k1 >= min_postings or k2 >= min_postings),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    fdr = bh_fdr(out.p_value, alpha)
    out["q_value"] = fdr.q_value.round(5)
    out["significant"] = fdr.significant & out.supported
    out = annotate_self_reference(out)
    return out.sort_values("log_lift", ascending=False).reset_index(drop=True)


def annotate_self_reference(table: pd.DataFrame, company_col: str = "company",
                            skill_col: str = "skill") -> pd.DataFrame:
    """Flag rows where the skill is the company's own product.

    Flag, never filter. "Snowflake asks for Snowflake" is a tautology and
    "NVIDIA asks for CUDA" is half of one, but both are true and a reader is
    entitled to see them; what they are not entitled to is a top-ten list
    silently led by them.
    """
    out = table.copy()
    out["vendor_relation"] = [
        VENDOR_SKILLS.get(str(c), {}).get(str(s), "")
        for c, s in zip(out[company_col], out[skill_col])
    ]
    out["self_referential"] = out.vendor_relation.ne("")
    return out


def pairwise_skill_gap(share: pd.DataFrame, a: str, b: str,
                       min_postings: int = MIN_CELL,
                       alpha: float = 0.05, z: float = 1.96) -> pd.DataFrame:
    """``share(a) - share(b)`` per skill, with a Newcombe interval and BH q.

    The interval is the deliverable. "Google asks for SQL 8 points more often
    than Meta, 95% CI [2, 14]" is a usable sentence; "Google 41% vs Meta 33%"
    invites a reader to treat 8 points from 600 postings as a fact.
    """
    wide = share.pivot_table(index="skill", columns="company",
                             values="postings_with_skill", fill_value=0)
    if a not in wide.columns or b not in wide.columns:
        return pd.DataFrame()
    na = int(share.loc[share.company == a, "n_skilled"].iloc[0])
    nb = int(share.loc[share.company == b, "n_skilled"].iloc[0])
    cats = share.drop_duplicates("skill").set_index("skill").skill_category
    rows = []
    for skill in wide.index:
        ka, kb = int(wide.loc[skill, a]), int(wide.loc[skill, b])
        if max(ka, kb) < min_postings:
            continue
        lo, hi = newcombe_diff_interval(ka, na, kb, nb, z)
        rows.append({
            "skill": skill, "skill_category": cats.get(skill, ""),
            "company_a": a, "company_b": b,
            "a_with_skill": ka, "a_n": na, "a_share": round(ka / na, 4),
            "b_with_skill": kb, "b_n": nb, "b_share": round(kb / nb, 4),
            "diff": round(ka / na - kb / nb, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "p_value": two_proportion_p(ka, na, kb, nb),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    fdr = bh_fdr(out.p_value, alpha)
    out["q_value"] = fdr.q_value.round(5)
    # A gap is only called when the interval clears zero *and* survives BH.
    out["significant"] = fdr.significant & ((out.ci_low > 0) | (out.ci_high < 0))
    # Both sides get a vendor flag: "google minus databricks on Databricks" is
    # -100pp and means nothing, and the reader should not have to know why.
    out["a_vendor_relation"] = [VENDOR_SKILLS.get(a, {}).get(s, "")
                                for s in out.skill]
    out["b_vendor_relation"] = [VENDOR_SKILLS.get(b, {}).get(s, "")
                                for s in out.skill]
    out["self_referential"] = (out.a_vendor_relation.ne("")
                               | out.b_vendor_relation.ne(""))
    return out.sort_values("diff", ascending=False).reset_index(drop=True)


def standardised_skill_share(long: pd.DataFrame, features: pd.DataFrame,
                             skill: str, weights: pd.Series,
                             by: str = "job_function",
                             min_cell: int = MIN_CELL) -> dict:
    """One skill's share, reweighted to the pooled role mix."""
    denom = skill_denominator(features)
    has = set(long.loc[long.skill == skill, "job_id"])
    frame = denom[[by]].copy()
    frame["hit"] = denom.job_id.isin(has).astype(int)
    return standardise(frame, "hit", weights, by, min_cell)


def standardised_skill_table(longs: dict[str, pd.DataFrame],
                             frames: dict[str, pd.DataFrame],
                             skills, by: str = "job_function",
                             min_cell: int = MIN_CELL) -> pd.DataFrame:
    """Crude vs mix-standardised share for a set of skills, all companies.

    The cross-company answer to Task 05's Simpson's paradox. Looker was flagged
    as emerging for Google and reversed inside every job function; the same
    trap here would read Snowflake's 70%-Engineering mix as a preference for
    engineering skills.
    """
    # Weights come from the *skill denominator*, not from all postings, so the
    # standard population is the population the shares are actually measured on.
    w = pooled_weights({k: skill_denominator(df) for k, df in frames.items()}, by)
    rows = []
    for key in sorted(frames):
        for skill in skills:
            r = standardised_skill_share(longs[key], frames[key], skill, w, by,
                                         min_cell)
            std = r["standardised"]
            rows.append({
                "company": key, "skill": skill,
                "crude_share_of_skilled": round(r["crude"], 4),
                "standardised_share": round(std, 4),
                "mix_effect": (round(r["crude"] - std, 4)
                               if not math.isnan(std) else float("nan")),
                "weight_covered": r["weight_covered"],
                "strata_used": r["strata_used"],
            })
    out = pd.DataFrame(rows)
    out.attrs["weights"] = w
    return out


# ---------------------------------------------------------------------------
# Stratified verdicts and panel robustness
# ---------------------------------------------------------------------------


def within_stratum_shares(longs: dict[str, pd.DataFrame],
                          frames: dict[str, pd.DataFrame], skill: str,
                          by: str = "job_function",
                          min_cell: int = MIN_CELL) -> pd.DataFrame:
    """One skill's share per company *within* each stratum."""
    rows = []
    for key in sorted(frames):
        denom = skill_denominator(frames[key])
        has = set(longs[key].loc[longs[key].skill == skill, "job_id"])
        grp = denom.assign(hit=denom.job_id.isin(has).astype(int)) \
                   .groupby(denom[by].fillna("Unknown"))
        for stratum, block in grp:
            if len(block) < min_cell:
                continue
            k, n = int(block.hit.sum()), int(len(block))
            lo, hi = wilson_interval(k, n, 1.96)
            rows.append({"company": key, by: stratum, "skill": skill,
                         "postings_with_skill": k, "n_skilled": n,
                         "share": round(k / n, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    return pd.DataFrame(rows)


def stratified_company_verdict(within: pd.DataFrame, skill: str, a: str, b: str,
                               by: str = "job_function",
                               min_strata: int = 2) -> dict:
    """Does ``a > b`` on this skill hold inside every shared stratum?

    The cross-company Simpson's check. A pooled gap that reverses in some
    strata is a mix artefact, and Task 05 found eight of Google's twenty-nine
    supported skill trends were exactly that. ``verdict`` is one of
    ``confirmed`` / ``reversed`` / ``mixed`` / ``unsupported``.
    """
    sub = within[within.skill == skill]
    wide = sub.pivot_table(index=by, columns="company", values="share")
    if a not in wide.columns or b not in wide.columns:
        return {"skill": skill, "company_a": a, "company_b": b,
                "verdict": "unsupported", "n_strata": 0, "n_agree": 0,
                "detail": "one company has no supported stratum"}
    both = wide[[a, b]].dropna()
    if len(both) < min_strata:
        return {"skill": skill, "company_a": a, "company_b": b,
                "verdict": "unsupported", "n_strata": int(len(both)), "n_agree": 0,
                "detail": f"only {len(both)} shared stratum/strata"}
    pooled_a = sub.loc[sub.company == a, "postings_with_skill"].sum() / \
        max(sub.loc[sub.company == a, "n_skilled"].sum(), 1)
    pooled_b = sub.loc[sub.company == b, "postings_with_skill"].sum() / \
        max(sub.loc[sub.company == b, "n_skilled"].sum(), 1)
    pooled_sign = np.sign(pooled_a - pooled_b)
    signs = np.sign(both[a] - both[b])
    agree = int((signs == pooled_sign).sum())
    if agree == len(both):
        verdict = "confirmed"
    elif agree == 0:
        verdict = "reversed"
    else:
        verdict = "mixed"
    return {
        "skill": skill, "company_a": a, "company_b": b,
        "pooled_diff": round(float(pooled_a - pooled_b), 4),
        "verdict": verdict, "n_strata": int(len(both)), "n_agree": agree,
        "detail": "; ".join(f"{s}: {both.loc[s, a]:.3f} vs {both.loc[s, b]:.3f}"
                            for s in both.index),
    }


def panel_robustness(longs: dict[str, pd.DataFrame],
                     frames: dict[str, pd.DataFrame], skills,
                     publishers: list[str] | None = None,
                     min_postings: int = MIN_CELL) -> pd.DataFrame:
    """Recompute skill shares on the common publisher panel and compare signs.

    Channel is the other half of the confounding, and standardisation does not
    touch it. A gap that holds on all publishers but flips on the shared ones
    was a syndication difference wearing a skill's name.
    """
    pubs = common_publishers(frames) if publishers is None else publishers
    cut = restrict(frames, pubs)
    keep = {k: set(df.job_id) for k, df in cut.items()}
    cut_longs = {k: longs[k][longs[k].job_id.isin(keep[k])] for k in frames}
    full = skill_share_table(longs, frames, min_postings)
    panel = skill_share_table(cut_longs, cut, min_postings)
    merged = full.merge(panel, on=["company", "skill"], how="left",
                        suffixes=("_all", "_panel"))
    merged = merged[merged.skill.isin(list(skills))]
    out = merged[[
        "company", "skill", "skill_category_all",
        "postings_with_skill_all", "n_skilled_all", "share_of_skilled_all",
        "postings_with_skill_panel", "n_skilled_panel", "share_of_skilled_panel",
    ]].rename(columns={"skill_category_all": "skill_category"}).reset_index(drop=True)
    # A skill that vanishes entirely from the panel produces no row there, and
    # a missing row is not missing evidence: the panel denominator exists, so
    # the share is a measured zero. Left as NaN this reads as "not channel
    # sensitive", which is backwards — it is the most channel-sensitive case
    # there is.
    panel_n = {k: len(skill_denominator(df)) for k, df in cut.items()}
    observed = out.company.map(panel_n).fillna(0).astype(int)
    out["n_skilled_panel"] = out.n_skilled_panel.fillna(observed)
    missing = out.share_of_skilled_panel.isna() & (observed > 0)
    out.loc[missing, ["postings_with_skill_panel", "share_of_skilled_panel"]] = 0.0
    out["share_delta"] = (out.share_of_skilled_panel
                          - out.share_of_skilled_all).round(4)
    # A share that moves by more than 10 points when the channel is held
    # fixed was partly a channel statistic; the flag is what the report cuts on.
    out["channel_sensitive"] = out.share_delta.abs() > 0.10
    return out


# ---------------------------------------------------------------------------
# Standing checks
# ---------------------------------------------------------------------------


def forbidden_columns(table: pd.DataFrame) -> list[str]:
    """Task 06 columns that must never reach a committed table."""
    return sorted(set(table.columns) & FORBIDDEN_COLUMNS)


def personal_data_columns_present(table: pd.DataFrame) -> list[str]:
    """Task 01's standing privacy check, re-run on every emitted table.

    Re-run every task rather than trusted from the last one: the check is
    cheap, and the failure it guards against — a name or contact column
    arriving through a new join — is exactly the kind that appears when the
    pipeline gains a stage.
    """
    banned = ("name_first", "first_name", "last_name", "surname", "email",
              "phone", "candidate", "applicant", "recruiter", "contact",
              "person", "author", "profile_url", "linkedin_url")
    cols = [c for c in table.columns
            if any(b in str(c).lower() for b in banned)
            and str(c).lower() not in {"company_name"}]
    return sorted(cols)
