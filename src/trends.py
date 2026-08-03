"""Task 05 — hiring trend analysis: the shared time-series structure.

Shared, company-agnostic trend engine for all four specialists. The team
standard and its rationale live in docs/task-05-trend-analysis-methods.md.

Why this module exists
----------------------
Task 06 compares the four companies' hiring velocity and Task 07 forecasts it.
Both break if one specialist counts postings per calendar month, another per
ISO week, and a third quietly drops the months their collector was asleep. So
the period keys, the velocity definition and the panel rule are **one shared
implementation**, not four notebooks.

The one thing that has to be controlled: the publisher panel
------------------------------------------------------------
Job-posting volume is not observed directly. It is observed *through* a set of
publishers (``job_via`` — LinkedIn, BeBee, the employer's own careers site, …)
that enter and leave the collection window. In the Google data, 96 publishers
produce 848 postings and **exactly one of them is present in all 12 months**;
56 appear in a single month. A raw monthly count therefore moves for two
reasons that look identical in a line chart:

1. the company posted more jobs — the signal;
2. a different set of aggregators was being captured — the artefact.

Every series this module produces carries the publisher count that generated
it, and a panel treatment has to be named explicitly:

=================  =========================================================
``raw``            every posting. Honest volume, uncontrolled composition.
``balanced``       only publishers present in >= ``min_periods`` periods.
                   Controlled, but throws away the rest of the data.
``matched``        chain-linked index; each period-to-period link uses only
                   the publishers present in *both* of its periods. Uses all
                   the data, at the cost of chain drift on thin links.
=================  =========================================================

None of the three is "correct" on its own. ``panel_sensitivity_table`` puts
them side by side, and the rule is that a trend claim survives only if the
treatments agree on its **sign**.

Velocity, not counts
--------------------
A "hiring velocity" comparison across February (28 days) and a truncated first
ISO week is a comparison of calendars. Velocity here is always
**postings per 7 observed days**, where observed days are clipped to the
dataset's own collection window, and any period only partly inside that window
is flagged ``is_partial`` so it can be dropped from a slope.

Usage
-----
    from trends import volume_series, panel_sensitivity_table
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Period grammar — shared with Task 03, which already wrote these columns
# ---------------------------------------------------------------------------

# key            column written by Task 03      display
PERIODS: dict[str, str] = {
    "week": "posting_week",        # ISO year-week, e.g. 2023-W07
    "month": "posting_month",      # 2023-07
    "quarter": "posting_quarter",  # 2023-Q3
}

PERIOD_DAYS: dict[str, int] = {"week": 7, "month": 30, "quarter": 91}

#: Publisher column. Named once so a company whose collector calls it something
#: else changes one line rather than every call site.
PUBLISHER_COL = "job_via"

#: `job_function` values excluded from *skill* demand series. Task 04 §3.1:
#: data-centre facilities roles genuinely carry no software skills, so leaving
#: them in the denominator makes every skill share look like it is falling.
#: They stay in the *volume* series — they are real hiring.
SKILL_EXCLUDED_FUNCTIONS: frozenset[str] = frozenset({"Facilities / Operations"})

PANEL_TREATMENTS = ("raw", "balanced", "matched")


# ---------------------------------------------------------------------------
# Period helpers
# ---------------------------------------------------------------------------


def period_col(period: str) -> str:
    """Column name for a period key, validated."""
    try:
        return PERIODS[period]
    except KeyError:
        raise ValueError(
            f"unknown period {period!r}; expected one of {sorted(PERIODS)}"
        ) from None


def period_bounds(key: str, period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last calendar day of a period key.

    Weeks are ISO weeks, so ``2023-W01`` starts on the Monday — which is why
    2023-01-01 (a Sunday) belongs to ``2022-W52`` and the weekly series has 53
    buckets for one calendar year. That boundary is a real trap: two of those
    53 buckets are partial and must not be read as a slowdown.
    """
    key = str(key)
    if period == "week":
        year, week = key.split("-W")
        start = pd.Timestamp.fromisocalendar(int(year), int(week), 1)
        return start, start + pd.Timedelta(days=6)
    if period == "month":
        start = pd.Timestamp(f"{key}-01")
        return start, start + pd.offsets.MonthEnd(0)
    if period == "quarter":
        quarter = pd.Period(key, freq="Q")
        return quarter.start_time.normalize(), quarter.end_time.normalize()
    raise ValueError(f"unknown period {period!r}")


def observation_window(df: pd.DataFrame,
                       date_col: str = "posting_date") -> tuple[pd.Timestamp,
                                                                pd.Timestamp]:
    """The dataset's own first and last observed day.

    Everything is clipped to this rather than to the calendar, because a
    collection window that starts mid-month must not report that month as a
    hiring dip.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        raise ValueError(f"no parseable dates in {date_col!r}")
    return dates.min().normalize(), dates.max().normalize()


# ---------------------------------------------------------------------------
# Volume and velocity
# ---------------------------------------------------------------------------


def volume_series(df: pd.DataFrame, period: str = "month",
                  window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
                  date_col: str = "posting_date",
                  min_observed_frac: float = 0.5) -> pd.DataFrame:
    """Postings per period, with the exposure needed to make them comparable.

    Columns: ``period``, ``postings``, ``n_publishers``, ``days_in_period``,
    ``days_observed``, ``is_partial``, ``postings_per_week``.

    ``postings_per_week`` is the velocity every downstream task should use.
    Raw counts across months compare February with March and call the calendar
    a trend; across ISO weeks they compare a 3-day stub with a full week.

    A rate is left blank where less than ``min_observed_frac`` of the period
    was observed. The Google window opens on a Sunday, which ISO calls
    ``2022-W52``: one observed day carrying 7 postings scales to "49 per week"
    and would be the largest value in the series. That is not a busy week, it
    is a divide-by-one.

    Periods with no postings at all inside the observation window are emitted
    as zeros rather than skipped — an empty month is information, and silently
    dropping it turns a collection gap into a straight line.
    """
    col = period_col(period)
    if col not in df.columns:
        raise KeyError(f"{col!r} missing; run the Task 03 preprocessor first")

    start, end = window or observation_window(df, date_col)
    keys = _period_keys_between(start, end, period)

    grouped = df.groupby(col)
    counts = grouped.size().reindex(keys, fill_value=0)
    pubs = (
        grouped[PUBLISHER_COL].nunique().reindex(keys, fill_value=0)
        if PUBLISHER_COL in df.columns
        else pd.Series(np.nan, index=keys)
    )

    rows = []
    for key in keys:
        p_start, p_end = period_bounds(key, period)
        observed = (min(p_end, end) - max(p_start, start)).days + 1
        full = (p_end - p_start).days + 1
        rows.append(
            {
                "period": key,
                "postings": int(counts[key]),
                "n_publishers": _as_int(pubs[key]),
                "days_in_period": full,
                "days_observed": max(observed, 0),
                "is_partial": observed < full,
            }
        )
    out = pd.DataFrame(rows)
    exposure = out.days_observed.where(
        out.days_observed >= min_observed_frac * out.days_in_period
    ).replace(0, np.nan)
    out["postings_per_week"] = (7 * out.postings / exposure).round(3)
    return out


def _period_keys_between(start: pd.Timestamp, end: pd.Timestamp,
                         period: str) -> list[str]:
    """Every period key touched by [start, end], including empty ones."""
    days = pd.date_range(start, end, freq="D")
    if period == "week":
        iso = days.isocalendar()
        keys = [f"{y}-W{w:02d}" for y, w in zip(iso.year, iso.week)]
    elif period == "month":
        keys = days.strftime("%Y-%m").tolist()
    else:
        keys = [f"{d.year}-Q{d.quarter}" for d in days]
    return list(dict.fromkeys(keys))


def add_velocity(series: pd.DataFrame, value_col: str = "postings_per_week",
                 window: int = 3, drop_partial: bool = True) -> pd.DataFrame:
    """Growth, momentum and acceleration on a velocity series.

    ``index_base_100`` is rebased on the first **complete** period, so a
    truncated first week cannot inflate every later value.

    Growth is reported as both a percentage change and a log difference. The
    log difference is the one Task 07 should model: percentage changes are
    asymmetric (a −50% month followed by a +50% month is not flat), which
    matters here because the series swings by more than half from month to
    month.
    """
    out = series.copy()
    usable = ~out.get("is_partial", pd.Series(False, index=out.index)) \
        if drop_partial else pd.Series(True, index=out.index)
    values = out[value_col].where(usable)

    base_candidates = values.dropna()
    base = base_candidates.iloc[0] if len(base_candidates) else np.nan
    out["index_base_100"] = (100 * values / base).round(2) if base else np.nan

    # fill_method=None on purpose: a blanked partial period is a gap in the
    # series, and padding across it would report a growth rate that spans two
    # periods as though it spanned one.
    out["growth_pct"] = (100 * values.pct_change(fill_method=None)).round(2)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_growth"] = np.log(values.replace(0, np.nan)).diff().round(4)
    out["rolling_mean"] = values.rolling(window, min_periods=1).mean().round(3)
    out["acceleration"] = out.growth_pct.diff().round(2)
    return out


# ---------------------------------------------------------------------------
# Publisher panel control
# ---------------------------------------------------------------------------


def publisher_presence(df: pd.DataFrame, period: str = "month") -> pd.DataFrame:
    """Publisher x period posting counts — the panel, made visible.

    This table is a deliverable, not a diagnostic. It is the evidence for
    whether a volume trend is a hiring trend at all.
    """
    col = period_col(period)
    if PUBLISHER_COL not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df[PUBLISHER_COL], df[col])


def publisher_panel_table(df: pd.DataFrame, period: str = "month") -> pd.DataFrame:
    """Per-publisher entry, exit and continuity."""
    presence = publisher_presence(df, period)
    if presence.empty:
        return pd.DataFrame()
    active = presence > 0
    n_periods = presence.shape[1]
    out = pd.DataFrame(
        {
            "publisher": presence.index,
            "postings": presence.sum(axis=1).to_numpy(),
            "periods_present": active.sum(axis=1).to_numpy(),
            "first_period": [active.columns[r.argmax()] for _, r in active.iterrows()],
            "last_period": [
                active.columns[len(r) - 1 - r[::-1].argmax()]
                for _, r in active.iterrows()
            ],
        }
    )
    out["coverage_of_periods"] = (out.periods_present / n_periods).round(4)
    out["is_continuous"] = out.periods_present == n_periods
    return out.sort_values("postings", ascending=False).reset_index(drop=True)


def panel_publishers(df: pd.DataFrame, period: str = "month",
                     min_periods: int | None = None,
                     min_share: float = 0.75) -> list[str]:
    """Publishers stable enough to carry a trend.

    ``min_periods`` defaults to ``min_share`` of the observed periods (75%),
    which keeps a publisher that missed a month or two but drops the 56
    single-month arrivals that otherwise dominate the Google series.
    """
    presence = publisher_presence(df, period)
    if presence.empty:
        return []
    n_periods = presence.shape[1]
    threshold = min_periods if min_periods is not None else int(
        np.ceil(min_share * n_periods)
    )
    active = (presence > 0).sum(axis=1)
    return sorted(active.index[active >= threshold].tolist())


def balanced_volume_series(df: pd.DataFrame, period: str = "month",
                           min_periods: int | None = None,
                           min_share: float = 0.75,
                           **kwargs) -> pd.DataFrame:
    """`volume_series` restricted to the stable publisher panel.

    Same-store sales, applied to job boards: only compare the publishers that
    were open in both periods.
    """
    keep = panel_publishers(df, period, min_periods, min_share)
    subset = df[df[PUBLISHER_COL].isin(keep)] if keep else df.iloc[0:0]
    out = volume_series(subset, period, **kwargs) if len(subset) else \
        volume_series(df, period, **kwargs).assign(postings=0, n_publishers=0,
                                                   postings_per_week=0.0)
    out.attrs["panel_publishers"] = keep
    return out


def matched_index(df: pd.DataFrame, period: str = "month",
                  method: str = "chained", base: str | None = None) -> pd.DataFrame:
    """Publisher-matched volume index — an unbalanced panel, handled properly.

    ``chained``    each link compares consecutive periods over the publishers
                   present in both, then multiplies the links. Uses every
                   publisher somewhere, but a thin link (few shared
                   publishers) propagates its noise into every later value —
                   classic chain drift, so ``n_matched_publishers`` ships
                   alongside and a link built on a handful of publishers
                   should be read as a break, not a measurement.
    ``bilateral``  every period compared directly against ``base`` over the
                   publishers present in both. No drift, but the matched set
                   thins out the further from base you go.

    Both are reported because they fail differently. Where they agree, the
    move is not a panel artefact.
    """
    if method not in {"chained", "bilateral"}:
        raise ValueError(f"unknown method {method!r}")
    presence = publisher_presence(df, period)
    if presence.empty or presence.shape[1] < 2:
        return pd.DataFrame(columns=["period", "index", "n_matched_publishers",
                                     "base_postings", "period_postings", "method"])
    keys = list(presence.columns)
    base = base or keys[0]

    rows = [{"period": base, "index": 100.0,
             "n_matched_publishers": int((presence[base] > 0).sum()),
             "base_postings": int(presence[base].sum()),
             "period_postings": int(presence[base].sum()), "method": method}]

    level = 100.0
    for prev, cur in zip(keys, keys[1:]):
        left = prev if method == "chained" else base
        shared = presence.index[(presence[left] > 0) & (presence[cur] > 0)]
        n_left = presence.loc[shared, left].sum()
        n_cur = presence.loc[shared, cur].sum()
        link = (n_cur / n_left) if n_left else np.nan
        level = level * link if method == "chained" else 100.0 * link
        rows.append(
            {
                "period": cur,
                "index": round(level, 2) if pd.notna(level) else np.nan,
                "n_matched_publishers": len(shared),
                "base_postings": int(n_left),
                "period_postings": int(n_cur),
                "method": method,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PanelVerdict:
    """Whether a direction survives every panel treatment."""

    direction: str          # "growth" | "decline" | "flat" | "unresolved"
    agrees: bool
    detail: dict[str, float]


def panel_sensitivity_table(df: pd.DataFrame, period: str = "month",
                            min_share: float = 0.75) -> pd.DataFrame:
    """The four treatments side by side, indexed to their own first period.

    The headline table of Task 05. If a company's volume rises under ``raw``
    and falls under ``balanced``, the honest report is that its hiring
    direction is **not identified by this data** — not whichever number is
    more flattering.
    """
    raw = volume_series(df, period)
    bal = balanced_volume_series(df, period, min_share=min_share)
    chained = matched_index(df, period, "chained")
    bilateral = matched_index(df, period, "bilateral")

    out = raw[["period", "postings", "n_publishers", "is_partial"]].rename(
        columns={"postings": "raw_postings", "n_publishers": "raw_publishers"}
    )
    out = out.merge(
        bal[["period", "postings"]].rename(columns={"postings": "balanced_postings"}),
        on="period", how="left",
    )
    out["raw_index"] = _index_100(out.raw_postings)
    out["balanced_index"] = _index_100(out.balanced_postings)
    out = out.merge(
        chained[["period", "index", "n_matched_publishers"]].rename(
            columns={"index": "chained_index",
                     "n_matched_publishers": "chained_matched_publishers"}),
        on="period", how="left",
    ).merge(
        bilateral[["period", "index", "n_matched_publishers"]].rename(
            columns={"index": "bilateral_index",
                     "n_matched_publishers": "bilateral_matched_publishers"}),
        on="period", how="left",
    )
    out.attrs["panel_publishers"] = bal.attrs.get("panel_publishers", [])
    return out


def panel_verdict(sensitivity: pd.DataFrame,
                  flat_band: float = 10.0) -> PanelVerdict:
    """Do the treatments agree on the direction of travel?

    Compares each treatment's last value against its own base of 100. A move
    inside +/- ``flat_band`` percent counts as flat. ``agrees`` is False when
    the treatments disagree on sign — the case that has to be reported rather
    than resolved.
    """
    cols = ["raw_index", "balanced_index", "chained_index", "bilateral_index"]
    detail: dict[str, float] = {}
    for c in cols:
        if c not in sensitivity.columns:
            continue
        vals = sensitivity[c].dropna()
        if len(vals):
            detail[c.replace("_index", "")] = float(round(vals.iloc[-1], 2))
    if not detail:
        return PanelVerdict("unresolved", False, {})

    def label(v: float) -> str:
        if v > 100 + flat_band:
            return "growth"
        if v < 100 - flat_band:
            return "decline"
        return "flat"

    labels = {label(v) for v in detail.values()}
    if len(labels) == 1:
        return PanelVerdict(labels.pop(), True, detail)
    if labels <= {"growth", "flat"}:
        return PanelVerdict("growth", False, detail)
    if labels <= {"decline", "flat"}:
        return PanelVerdict("decline", False, detail)
    return PanelVerdict("unresolved", False, detail)


def _index_100(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    first = vals.dropna()
    if first.empty or first.iloc[0] == 0:
        return pd.Series(np.nan, index=s.index)
    return (100 * vals / first.iloc[0]).round(2)


def _as_int(value) -> int:
    return 0 if pd.isna(value) else int(value)


# ---------------------------------------------------------------------------
# Spikes — and whether they are hiring events at all
# ---------------------------------------------------------------------------


def robust_spikes(series: pd.DataFrame, value_col: str = "postings",
                  window: int = 9, z_threshold: float = 3.5) -> pd.DataFrame:
    """Flag periods that stand out from their local level.

    Median/MAD rather than mean/sd, because the spikes we are hunting would
    otherwise inflate the very statistics used to detect them. Where MAD is
    zero (a flat local window) the test falls back to a Poisson-scale
    deviation, since counts of this size have variance about equal to their
    mean.
    """
    out = series.copy()
    values = pd.to_numeric(out[value_col], errors="coerce")
    med = values.rolling(window, center=True, min_periods=3).median()
    mad = (values - med).abs().rolling(window, center=True, min_periods=3).median()

    with np.errstate(divide="ignore", invalid="ignore"):
        robust_z = 0.6745 * (values - med) / mad.replace(0, np.nan)
        poisson_z = (values - med) / np.sqrt(med.clip(lower=1))

    out["local_median"] = med.round(2)
    out["robust_z"] = robust_z.round(2)
    out["poisson_z"] = poisson_z.round(2)
    out["is_spike"] = (
        (robust_z >= z_threshold) | (robust_z.isna() & (poisson_z >= 3.0))
    ).fillna(False)
    out["is_trough"] = (
        (robust_z <= -z_threshold) | (robust_z.isna() & (poisson_z <= -3.0))
    ).fillna(False)
    return out


def batch_table(df: pd.DataFrame, min_batch: int = 3,
                date_col: str = "posting_date") -> pd.DataFrame:
    """Publisher-day clusters: one publisher dumping N postings on one date.

    An employer does not publish 21 roles in a single day and nothing either
    side of it; an aggregator backfilling its archive does exactly that. Batch
    postings inherit the *batch's* date, not the job's, so they are the main
    way a collection event becomes a fake hiring spike.
    """
    if PUBLISHER_COL not in df.columns:
        return pd.DataFrame(columns=["publisher", "date", "postings"])
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.date
    agg = (
        df.assign(_d=dates)
        .groupby([PUBLISHER_COL, "_d"], as_index=False)
        .size()
        .rename(columns={PUBLISHER_COL: "publisher", "_d": "date",
                         "size": "postings"})
    )
    agg = agg[agg.postings >= min_batch]
    return agg.sort_values("postings", ascending=False).reset_index(drop=True)


def attribute_spikes(df: pd.DataFrame, spikes: pd.DataFrame,
                     period: str = "week", date_col: str = "posting_date",
                     excess_threshold: float = 0.5,
                     day_threshold: float = 0.4) -> pd.DataFrame:
    """For every flagged spike, name the publisher that produced it.

    Attribution is on **excess**, not raw share. Asking "which publisher has
    the most postings this week?" mostly returns whichever publisher is
    biggest overall, so the discriminator is instead: of the postings above
    the local median, how many come from one publisher posting above *its*
    own usual level? A publisher that has never appeared before contributes
    its entire batch as excess, which is exactly the case worth catching.

    A genuine hiring surge is spread across the publishers already covering
    the company. A collection artefact is one publisher, usually on one day.
    """
    col = period_col(period)
    flagged = spikes.loc[spikes.get("is_spike", False), "period"].tolist()
    empty = pd.DataFrame(columns=["period", "postings", "local_median",
                                  "excess", "top_publisher",
                                  "top_publisher_postings",
                                  "top_publisher_typical",
                                  "excess_explained", "largest_single_day",
                                  "largest_day_share_of_excess",
                                  "distinct_publishers", "verdict"])
    if not flagged or PUBLISHER_COL not in df.columns:
        return empty

    presence = publisher_presence(df, period).reindex(
        columns=spikes.period.tolist(), fill_value=0
    ).fillna(0)
    typical = presence.median(axis=1)
    medians = spikes.set_index("period").local_median.to_dict()
    batches = batch_table(df, min_batch=1, date_col=date_col)

    rows = []
    for key in flagged:
        block = df[df[col] == key]
        n = len(block)
        local_med = medians.get(key, np.nan)
        excess = n - local_med if pd.notna(local_med) else np.nan
        counts = presence[key] if key in presence.columns else pd.Series(dtype=float)
        surplus = (counts - typical).clip(lower=0).sort_values(ascending=False)
        top = surplus.index[0] if len(surplus) and surplus.iloc[0] > 0 else \
            block[PUBLISHER_COL].value_counts().index[0]

        day = batches[
            batches.date.isin(
                pd.to_datetime(block[date_col], errors="coerce").dt.date.unique())
            & batches.publisher.isin(block[PUBLISHER_COL].unique())
        ]
        biggest = int(day.postings.max()) if len(day) else 0
        explained = float(surplus.get(top, 0) / excess) if excess and excess > 0 \
            else np.nan
        day_share = float(biggest / excess) if excess and excess > 0 else np.nan

        rows.append(
            {
                "period": key,
                "postings": n,
                "local_median": local_med,
                "excess": excess,
                "top_publisher": top,
                "top_publisher_postings": int(counts.get(top, 0)),
                "top_publisher_typical": float(typical.get(top, 0)),
                "excess_explained": round(explained, 4)
                if pd.notna(explained) else np.nan,
                "largest_single_day": biggest,
                "largest_day_share_of_excess": round(day_share, 4)
                if pd.notna(day_share) else np.nan,
                "distinct_publishers": int(block[PUBLISHER_COL].nunique()),
                "verdict": "publisher_batch"
                if (pd.notna(explained) and explained >= excess_threshold)
                or (pd.notna(day_share) and day_share >= day_threshold)
                else "broad_based",
            }
        )
    return pd.DataFrame(rows, columns=empty.columns)


# ---------------------------------------------------------------------------
# Composition: which parts of the company are growing
# ---------------------------------------------------------------------------


def half_of(period_key: str) -> str:
    """H1/H2 label, matching `skills.skill_trend_table` so the two agree."""
    key = str(period_key)
    try:
        if "-W" in key:
            month = pd.Timestamp.fromisocalendar(
                int(key[:4]), int(key.split("-W")[1]), 1).month
        elif "-Q" in key:
            month = int(key.split("-Q")[1]) * 3
        else:
            month = int(key[5:7])
    except (ValueError, TypeError, IndexError):
        return ""
    return "h1" if month <= 6 else "h2"


def segment_series(df: pd.DataFrame, by: str, period: str = "month",
                   min_postings: int = 10) -> pd.DataFrame:
    """Volume per segment per period, with the segment's share of the period.

    Both columns matter and they answer different questions. The count answers
    "is this team hiring more?"; the share answers "is this team taking a
    bigger slice?". A segment can rise on one and fall on the other, and only
    the share is robust to the collection volume moving underneath it.
    """
    col = period_col(period)
    if by not in df.columns:
        return pd.DataFrame()
    keep = df[by].value_counts()
    keep = keep[keep >= min_postings].index
    block = df[df[by].isin(keep)]
    agg = (
        block.groupby([col, by], as_index=False).size()
        .rename(columns={col: "period", "size": "postings"})
    )
    totals = block.groupby(col).size().rename("period_postings")
    agg = agg.merge(totals, left_on="period", right_index=True, how="left")
    agg["share_of_period"] = (agg.postings / agg.period_postings).round(4)
    return agg.sort_values(["period", "postings"],
                           ascending=[True, False]).reset_index(drop=True)


def segment_trend_table(df: pd.DataFrame, by: str, period: str = "month",
                        min_postings: int = 10, threshold: float = 0.25,
                        panel: str = "raw", min_share: float = 0.75) -> pd.DataFrame:
    """H1 vs H2 per segment, in both counts and share of that half.

    Halves, not a monthly slope, for the same reason as Task 04: one calendar
    year from one source cannot support a 12-point regression without reading
    sampling noise as trend.

    ``panel="balanced"`` re-runs the whole thing on the stable publishers, so
    a segment that only "grew" because an aggregator specialising in it
    arrived in July is visible as a treatment disagreement.
    """
    block = df
    if panel == "balanced":
        keep = panel_publishers(df, period, min_share=min_share)
        block = df[df[PUBLISHER_COL].isin(keep)]
    elif panel != "raw":
        raise ValueError(f"unknown panel {panel!r}")

    col = period_col(period)
    if by not in block.columns or block.empty:
        return pd.DataFrame()

    block = block.assign(half=block[col].map(half_of))
    block = block[block.half.isin({"h1", "h2"})]
    denom = block.half.value_counts().to_dict()
    if not denom.get("h1") or not denom.get("h2"):
        return pd.DataFrame()

    counts = (
        block.groupby([by, "half"]).size().unstack("half", fill_value=0)
        .reindex(columns=["h1", "h2"], fill_value=0).reset_index()
        .rename(columns={by: "segment", "h1": "n_h1", "h2": "n_h2"})
    )
    counts["share_h1"] = (counts.n_h1 / denom["h1"]).round(4)
    counts["share_h2"] = (counts.n_h2 / denom["h2"]).round(4)
    counts["share_delta"] = (counts.share_h2 - counts.share_h1).round(4)
    counts["count_change_pct"] = (
        100 * (counts.n_h2 / counts.n_h1.replace(0, np.nan) - 1)
    ).astype(float).round(1)
    counts["share_change_pct"] = (
        100 * counts.share_delta / counts.share_h1.replace(0, np.nan)
    ).astype(float).round(1)

    supported = (counts.n_h1 + counts.n_h2) >= min_postings
    counts["direction"] = "insufficient_support"
    counts.loc[supported, "direction"] = "stable"
    counts.loc[supported & (counts.share_change_pct >= 100 * threshold),
               "direction"] = "growth"
    counts.loc[supported & (counts.share_change_pct <= -100 * threshold),
               "direction"] = "decline"
    counts["dimension"] = by
    counts["panel"] = panel
    return counts.sort_values("share_delta", ascending=False).reset_index(drop=True)


def compare_panels(df: pd.DataFrame, by: str, **kwargs) -> pd.DataFrame:
    """`segment_trend_table` under both panels, joined, with an agreement flag.

    A segment trend counts as a finding only when ``directions_agree``.
    """
    raw = segment_trend_table(df, by, panel="raw", **kwargs)
    bal = segment_trend_table(df, by, panel="balanced", **kwargs)
    if raw.empty:
        return raw
    if bal.empty:
        return raw.assign(direction_balanced="no_panel", directions_agree=False)
    merged = raw.merge(
        bal[["segment", "n_h1", "n_h2", "share_change_pct", "direction"]].rename(
            columns={"n_h1": "n_h1_balanced", "n_h2": "n_h2_balanced",
                     "share_change_pct": "share_change_pct_balanced",
                     "direction": "direction_balanced"}),
        on="segment", how="left",
    )
    merged["direction_balanced"] = merged.direction_balanced.fillna("no_panel")
    merged["directions_agree"] = merged.direction == merged.direction_balanced
    return merged


# ---------------------------------------------------------------------------
# Seasonality — and what this data can and cannot say about it
# ---------------------------------------------------------------------------


def seasonality_table(df: pd.DataFrame, date_col: str = "posting_date",
                      min_cycles: int = 2) -> pd.DataFrame:
    """Day-of-week and month-of-year profiles, each with an identifiability flag.

    The brief asks for seasonal spikes. With a single calendar year, an annual
    seasonal factor has **one observation per month** and is perfectly
    confounded with the trend and with the collection ramp — it is not
    identifiable, and this table says so in a column rather than in a footnote
    nobody reads. Day-of-week has ~52 observations per level and *is*
    estimable, which turns out to be the more informative of the two here:
    it measures whether ``posting_date`` behaves like a publication date at
    all.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return pd.DataFrame()
    span_years = (dates.max() - dates.min()).days / 365.25

    rows = []
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                 "Saturday", "Sunday"]
    dow_counts = dates.dt.dayofweek.value_counts()
    n_weeks = max(len(pd.unique(dates.dt.to_period("W"))), 1)
    for i, name in enumerate(dow_names):
        n = int(dow_counts.get(i, 0))
        rows.append({
            "cycle": "day_of_week", "level": name, "postings": n,
            "share": round(n / len(dates), 4),
            "expected_share": round(1 / 7, 4),
            "index_vs_uniform": round(7 * n / len(dates), 3),
            "observations_per_level": n_weeks,
            "identifiable": n_weeks >= min_cycles,
        })

    month_counts = dates.dt.month.value_counts()
    for m in range(1, 13):
        n = int(month_counts.get(m, 0))
        rows.append({
            "cycle": "month_of_year", "level": f"{m:02d}", "postings": n,
            "share": round(n / len(dates), 4),
            "expected_share": round(1 / 12, 4),
            "index_vs_uniform": round(12 * n / len(dates), 3),
            "observations_per_level": int(np.floor(span_years)),
            "identifiable": np.floor(span_years) >= min_cycles,
        })
    return pd.DataFrame(rows)


def weekend_share(df: pd.DataFrame, date_col: str = "posting_date") -> float:
    """Fraction of postings dated on a Saturday or Sunday.

    A publication-date field should be near zero here. Anything close to the
    uniform 2/7 says the field is really "first seen by the aggregator", which
    is a different thing to forecast and a caveat Task 07 inherits.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return float("nan")
    return round(float((dates.dt.dayofweek >= 5).mean()), 4)


# ---------------------------------------------------------------------------
# Skill demand over time — under the Task 04 rules
# ---------------------------------------------------------------------------


def skill_velocity_table(long: pd.DataFrame, features: pd.DataFrame,
                         period: str = "month", min_postings: int = 10,
                         exclude_functions: frozenset[str] = SKILL_EXCLUDED_FUNCTIONS
                         ) -> pd.DataFrame:
    """Skill demand per period as `share_of_skilled`, facilities excluded.

    Task 04 §4 showed the denominator flips signs: on ``share_of_all`` Python
    "fell" 23% from March to April 2023 while on ``share_of_skilled`` it rose
    10%, and the whole difference was April's extraction coverage. So this
    function will not emit ``share_of_all`` at all — the wrong denominator
    should not be one column away from the right one.

    Facilities/Operations rows are dropped first (Task 04 §3.1): they are real
    hiring but genuinely skill-free, so they only dilute the denominator.
    """
    col = period_col(period)
    if long.empty or features.empty:
        return pd.DataFrame(columns=["period", "skill", "skill_category",
                                     "n_postings", "postings_with_skills",
                                     "share_of_skilled"])

    feats = features
    lng = long
    if exclude_functions and "job_function" in features.columns:
        feats = features[~features.job_function.isin(exclude_functions)]
        if "job_function" in long.columns:
            lng = long[~long.job_function.isin(exclude_functions)]
        else:
            lng = long[long.job_id.isin(feats.job_id)]

    denom = (
        feats.groupby(col).has_any_skill.sum()
        .rename("postings_with_skills").reset_index().rename(columns={col: "period"})
    )
    agg = (
        lng.groupby([col, "skill", "skill_category"], as_index=False)
        .agg(n_postings=("job_id", "nunique"))
        .rename(columns={col: "period"})
    )
    agg = agg.merge(denom, on="period", how="left")
    agg["share_of_skilled"] = (
        agg.n_postings / agg.postings_with_skills.replace(0, np.nan)
    ).astype(float).round(4)

    support = agg.groupby("skill").n_postings.sum()
    agg["meets_support"] = agg.skill.map(support >= min_postings).fillna(False)
    return agg.sort_values(["period", "n_postings"],
                           ascending=[True, False]).reset_index(drop=True)


def skill_trend_within_segment(long: pd.DataFrame, features: pd.DataFrame,
                               by: str = "job_function", min_postings: int = 10,
                               exclude_functions: frozenset[str] =
                               SKILL_EXCLUDED_FUNCTIONS) -> pd.DataFrame:
    """H1 vs H2 skill share computed *inside* each segment.

    Task 04 §5 left this as the outstanding job: a half-over-half skill move
    can be pure mix, because the composition of skilled postings shifts
    between halves. Stratifying holds the mix fixed — a skill that rises in
    every segment is not a composition effect.
    """
    if long.empty or by not in long.columns:
        return pd.DataFrame()
    lng = long[~long[by].isin(exclude_functions)] if exclude_functions else long
    feats = features
    if by in features.columns and exclude_functions:
        feats = features[~features[by].isin(exclude_functions)]

    lng = lng.assign(half=lng.posting_month.map(half_of))
    feats = feats.assign(half=feats.posting_month.map(half_of))

    denom = (
        feats[feats.half.isin({"h1", "h2"})]
        .groupby([by, "half"]).has_any_skill.sum().rename("denom").reset_index()
    )
    counts = (
        lng[lng.half.isin({"h1", "h2"})]
        .groupby([by, "skill", "half"], as_index=False)
        .agg(n=("job_id", "nunique"))
    )
    wide = counts.pivot_table(index=[by, "skill"], columns="half", values="n",
                              fill_value=0).reset_index()
    wide.columns.name = None
    for h in ("h1", "h2"):
        if h not in wide.columns:
            wide[h] = 0
    wide = wide.rename(columns={by: "segment", "h1": "n_h1", "h2": "n_h2"})

    d = denom.pivot_table(index=by, columns="half", values="denom", fill_value=0)
    wide["denom_h1"] = wide.segment.map(d.get("h1", pd.Series(dtype=float)))
    wide["denom_h2"] = wide.segment.map(d.get("h2", pd.Series(dtype=float)))
    wide["share_h1"] = (wide.n_h1 / wide.denom_h1.replace(0, np.nan)).round(4)
    wide["share_h2"] = (wide.n_h2 / wide.denom_h2.replace(0, np.nan)).round(4)
    wide["share_delta"] = (wide.share_h2 - wide.share_h1).round(4)
    wide["meets_support"] = (wide.n_h1 + wide.n_h2) >= min_postings
    wide["dimension"] = by
    return wide.sort_values(["segment", "share_delta"],
                            ascending=[True, False]).reset_index(drop=True)


def stratified_verdict(within: pd.DataFrame, skill: str,
                       min_segments: int = 2) -> dict:
    """Does a skill move the same way inside every segment that supports it?

    This is the test Task 04 §5 asked for and could not finish. A skill passes
    only if it moves in one direction in at least ``min_segments`` segments
    and never contradicts itself in a supported one.
    """
    block = within[(within.skill == skill) & within.meets_support]
    block = block.dropna(subset=["share_delta"])
    if block.empty:
        return {"skill": skill, "verdict": "insufficient_support",
                "n_segments": 0, "up": 0, "down": 0}
    up = int((block.share_delta > 0).sum())
    down = int((block.share_delta < 0).sum())
    n = len(block)
    if up >= min_segments and down == 0:
        verdict = "rising_in_all_segments"
    elif down >= min_segments and up == 0:
        verdict = "falling_in_all_segments"
    elif n < min_segments:
        verdict = "insufficient_support"
    else:
        verdict = "mix_dependent"
    return {"skill": skill, "verdict": verdict, "n_segments": n,
            "up": up, "down": down,
            "segments": block.segment.tolist()}
