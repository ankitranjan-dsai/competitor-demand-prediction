"""Task 05 — build the hiring-trend tables, figures and evidence report.

Reads the Task 04 feature table and skills-long table, runs the shared
time-series layer in `src/trends.py` over them, and writes:

    members/ankit-<company>/task-05-tables/     (committed — aggregate only)
        volume-by-month.csv            counts, exposure, velocity, growth
        volume-by-week.csv
        volume-by-quarter.csv
        publisher-panel.csv            entry/exit per publisher
        publisher-presence.csv         publisher x month matrix
        panel-sensitivity.csv          raw vs balanced vs matched index
        spikes-weekly.csv              robust z-scores per week
        spike-attribution.csv          which publisher explains each spike
        publisher-batches.csv          same-publisher same-day bulk loads
        trend-by-job-function.csv      count growth vs share growth
        trend-by-job-category.csv
        trend-by-country.csv
        trend-by-seniority.csv
        panel-check-job-function.csv   does the direction survive rebalancing
        seasonality.csv                day-of-week and month-of-year + support
        skill-velocity-by-month.csv    share_of_skilled only, Facilities out
        skill-trend-within-function.csv
        skill-stratified-verdicts.csv  pooled trend vs within-segment trend

    members/ankit-<company>/task-05-figures/    (committed)
        01-monthly-volume-and-velocity.png
        02-panel-sensitivity.png
        03-publisher-panel.png
        04-weekly-spikes.png
        05-job-function-mix.png
        06-skill-velocity.png
        07-simpsons-paradox-looker.png
        08-day-of-week.png

    members/ankit-<company>/task-05-trend-report.json   quality evidence

Nothing row-level is written: Task 05 adds no new posting-level columns, so
`data/processed/` is left exactly as Task 04 built it.

    python src/build_trends.py --company google
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display in CI or on a headless run
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

import trends as tr               # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# One palette for every Task 05 figure, so the four specialists' charts stack
# up in Task 06 without a legend clash.
INK = "#1f2933"
GREY = "#9aa5b1"
BLUE = "#2f6fb5"
ORANGE = "#e08a1e"
RED = "#c0392b"
GREEN = "#2e7d5b"
PURPLE = "#7d5ba6"


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _style(ax, title: str, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.grid(axis="y", color=GREY, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_volume(monthly: pd.DataFrame, path: Path) -> str:
    """Counts and exposure-normalised velocity, with partial months marked."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    partial = monthly.is_partial.fillna(False).astype(bool)

    ax1.bar(monthly.period, monthly.postings,
            color=[ORANGE if p else BLUE for p in partial])
    caption = ("orange months are only partly observed" if partial.any()
               else "every month fully observed")
    _style(ax1, f"Postings per month — {caption}", "postings")

    ax2.plot(monthly.period, monthly.postings_per_week, marker="o",
             color=BLUE, linewidth=1.8, markersize=4, label="postings / week")
    ax2.plot(monthly.period, monthly.rolling_mean, color=PURPLE,
             linewidth=1.2, linestyle="--", label="3-month mean")
    _style(ax2, "Hiring velocity — postings per 7 observed days",
           "postings / week", "month")
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=45)
    return _save(fig, path)


def fig_panel(sens: pd.DataFrame, verdict: tr.PanelVerdict, path: Path) -> str:
    """The headline chart: the same year under four panel treatments."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, colour, label in (
        ("raw_index", GREY, "raw (every publisher)"),
        ("balanced_index", BLUE, "balanced panel (present in >=75% of months)"),
        ("chained_index", GREEN, "matched-model, chained"),
        ("bilateral_index", ORANGE, "matched-model, bilateral vs base"),
    ):
        if col in sens.columns:
            ax.plot(sens.period, sens[col], marker="o", markersize=3,
                    linewidth=1.8, color=colour, label=label)
    ax.axhline(100, color=INK, linewidth=0.8, alpha=0.4)
    _style(ax,
           f"Same postings, four panel treatments — direction is "
           f"{'NOT resolved' if not verdict.agrees else verdict.direction}",
           "index (first period = 100)", "month")
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, path)


def fig_publishers(panel: pd.DataFrame, presence: pd.DataFrame,
                   path: Path) -> str:
    """How many publishers are present, and how few of them persist."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    active = presence.gt(0).sum(axis=0)
    ax1.bar(active.index.astype(str), active.values, color=BLUE)
    _style(ax1, "Publishers active per month", "publishers", "month")
    ax1.tick_params(axis="x", rotation=45)

    counts = panel.periods_present.value_counts().sort_index()
    ax2.bar(counts.index.astype(str), counts.values, color=PURPLE)
    _style(ax2, "Publishers by number of months present — the panel is "
                "mostly one-offs", "publishers", "months present")
    return _save(fig, path)


def fig_spikes(weekly: pd.DataFrame, spikes: pd.DataFrame,
               attributed: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(range(len(weekly)), weekly.postings, color=BLUE, linewidth=1.2)
    ax.plot(range(len(spikes)), spikes.local_median, color=GREY,
            linewidth=1.0, linestyle="--", label="local median")

    hit = spikes.index[spikes.is_spike].tolist()
    ax.scatter(hit, spikes.loc[hit, "postings"], color=RED, zorder=5, s=45,
               label="flagged spike")
    labels = dict(zip(attributed.period, attributed.top_publisher)) \
        if not attributed.empty else {}
    for i in hit:
        key = spikes.loc[i, "period"]
        who = labels.get(key, "")
        ax.annotate(f"{key}\n{who}", (i, spikes.loc[i, "postings"]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=7, color=RED)

    ax.set_ylim(0, float(weekly.postings.max()) * 1.25)   # headroom for labels
    step = max(len(weekly) // 14, 1)
    ax.set_xticks(range(0, len(weekly), step))
    ax.set_xticklabels(weekly.period[::step], rotation=45)
    _style(ax, "Weekly postings — every flagged spike is one publisher's "
               "batch, not a hiring event", "postings", "ISO week")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def fig_mix(seg: pd.DataFrame, path: Path) -> str:
    """Count growth and share growth side by side — they disagree."""
    top = seg.nlargest(10, "n_h1").sort_values("share_change_pct")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(top))
    ax.barh(y - 0.2, top.count_change_pct, height=0.38, color=BLUE,
            label="change in postings (%)")
    ax.barh(y + 0.2, top.share_change_pct, height=0.38, color=ORANGE,
            label="change in share of postings (%)")
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(top.segment, fontsize=8)
    _style(ax, "Job function, H1 vs H2 2023 — growing in count is not the "
               "same as growing in share", "", "% change")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GREY, alpha=0.25, linewidth=0.6)
    return _save(fig, path)


def fig_skill_velocity(sv: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    top = (sv.groupby("skill").n_postings.sum().nlargest(6).index)
    palette = [BLUE, ORANGE, GREEN, PURPLE, RED, GREY]
    for colour, skill in zip(palette, top):
        block = sv[sv.skill == skill].sort_values("period")
        ax.plot(block.period, 100 * block.share_of_skilled, marker="o",
                markersize=3, linewidth=1.6, color=colour, label=skill)
    _style(ax, "Top skills by month — share of *skilled* postings, "
               "Facilities/Operations excluded", "% of skilled postings",
           "month")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, path)


def fig_paradox(within: pd.DataFrame, pooled: pd.DataFrame, skill: str,
                path: Path) -> str:
    """The Simpson's-paradox chart: pooled up, every segment flat or down."""
    block = within[(within.skill == skill) & within.meets_support]
    fig, ax = plt.subplots(figsize=(9, 5))

    pooled_row = pooled[pooled.skill == skill]
    if not pooled_row.empty:
        ax.plot(["H1", "H2"],
                [100 * pooled_row.iloc[0].share_h1,
                 100 * pooled_row.iloc[0].share_h2],
                marker="o", markersize=8, linewidth=3, color=RED,
                label="pooled (all functions)  <- Task 04 called this emerging")

    palette = [BLUE, GREEN, PURPLE, ORANGE, GREY]
    for colour, row in zip(palette, block.itertuples()):
        ax.plot(["H1", "H2"], [100 * row.share_h1, 100 * row.share_h2],
                marker="o", markersize=5, linewidth=1.6, color=colour,
                linestyle="--",
                label=f"{row.segment} (n={int(row.n_h1 + row.n_h2)})")

    _style(ax, f"{skill}: rises pooled, flat or falling inside every job "
               f"function that supports it",
           "% of skilled postings in that group", "half of 2023")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def fig_dow(seasonality: pd.DataFrame, path: Path) -> str:
    dow = seasonality[seasonality.cycle == "day_of_week"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    weekend = dow.level.isin(["Saturday", "Sunday"])
    ax.bar(dow.level, dow.share, color=[ORANGE if w else BLUE for w in weekend])
    ax.axhline(1 / 7, color=INK, linestyle="--", linewidth=1,
               label="uniform (1/7)")
    _style(ax, "Postings by day of week — weekends are far too busy for a "
               "publication date", "share of postings", "")
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(features: pd.DataFrame, long: pd.DataFrame, company: str) -> dict:
    member = REPO_ROOT / "members" / f"ankit-{company}"
    tables = member / "task-05-tables"
    figures = member / "task-05-figures"

    # --- volume, velocity ---------------------------------------------------
    start, end = tr.observation_window(features)
    monthly = tr.add_velocity(tr.volume_series(features, "month"))
    weekly = tr.add_velocity(tr.volume_series(features, "week"))
    quarterly = tr.add_velocity(tr.volume_series(features, "quarter"))

    _write(monthly, tables / "volume-by-month.csv")
    _write(weekly, tables / "volume-by-week.csv")
    _write(quarterly, tables / "volume-by-quarter.csv")

    # --- publisher panel ----------------------------------------------------
    panel = tr.publisher_panel_table(features, "month")
    presence = tr.publisher_presence(features, "month")
    sensitivity = tr.panel_sensitivity_table(features, "month")
    verdict = tr.panel_verdict(sensitivity)
    keep = tr.panel_publishers(features, "month")

    _write(panel, tables / "publisher-panel.csv")
    _write(presence.reset_index(), tables / "publisher-presence.csv")
    _write(sensitivity, tables / "panel-sensitivity.csv")

    # --- spikes -------------------------------------------------------------
    spikes = tr.robust_spikes(weekly)
    attributed = tr.attribute_spikes(features, spikes, "week")
    batches = tr.batch_table(features, min_batch=5)

    _write(spikes, tables / "spikes-weekly.csv")
    _write(attributed, tables / "spike-attribution.csv")
    _write(batches, tables / "publisher-batches.csv")

    # --- composition --------------------------------------------------------
    segments = {}
    for dim, fname in (("job_function", "trend-by-job-function.csv"),
                       ("job_category", "trend-by-job-category.csv"),
                       ("country", "trend-by-country.csv"),
                       ("seniority", "trend-by-seniority.csv")):
        if dim not in features.columns:
            continue
        seg = tr.segment_trend_table(features, dim)
        if seg.empty:
            continue
        segments[dim] = seg
        _write(seg, tables / fname)

    panel_check = (tr.compare_panels(features, "job_function")
                   if "job_function" in features.columns else pd.DataFrame())
    if not panel_check.empty:
        _write(panel_check, tables / "panel-check-job-function.csv")

    seasonality = tr.seasonality_table(features)
    _write(seasonality, tables / "seasonality.csv")

    # --- skills, under the Task 04 rules ------------------------------------
    skill_velocity = tr.skill_velocity_table(long, features, "month")
    within = tr.skill_trend_within_segment(long, features, "job_function")
    _write(skill_velocity, tables / "skill-velocity-by-month.csv")
    _write(within, tables / "skill-trend-within-function.csv")

    # Pooled halves, for the pooled-vs-stratified comparison. Same denominator
    # discipline: skilled postings only, Facilities excluded.
    pooled = tr.skill_trend_within_segment(
        long.assign(_all="all"), features.assign(_all="all"), "_all")
    tested = sorted(within.loc[within.meets_support, "skill"].unique()) \
        if not within.empty else []
    verdicts = pd.DataFrame(
        [tr.stratified_verdict(within, s) for s in tested])
    if not verdicts.empty and not pooled.empty:
        pooled_map = pooled.set_index("skill")
        verdicts["pooled_share_h1"] = verdicts.skill.map(pooled_map.share_h1)
        verdicts["pooled_share_h2"] = verdicts.skill.map(pooled_map.share_h2)
        verdicts["pooled_direction"] = np.where(
            verdicts.pooled_share_h2 > verdicts.pooled_share_h1, "up",
            np.where(verdicts.pooled_share_h2 < verdicts.pooled_share_h1,
                     "down", "flat"))
        verdicts["pooled_delta"] = (verdicts.pooled_share_h2
                                    - verdicts.pooled_share_h1).round(4)
        # A reversal is when every supported segment moves *against* the
        # pooled direction; mix_dependent is the weaker case where the
        # segments simply do not agree and the pooled number picks a side.
        reversal = (
            ((verdicts.pooled_direction == "up")
             & (verdicts.verdict == "falling_in_all_segments"))
            | ((verdicts.pooled_direction == "down")
               & (verdicts.verdict == "rising_in_all_segments"))
        )
        verdicts["contradiction"] = np.where(
            reversal, "reversed",
            np.where(verdicts.verdict == "mix_dependent", "unsupported", ""))
        verdicts["overturned_by_stratification"] = verdicts.contradiction != ""
        # Order the table the way it should be read: outright reversals first,
        # then by how big a pooled move the stratification just overturned.
        rank = verdicts.contradiction.map(
            {"reversed": 0, "unsupported": 1}).fillna(2)
        verdicts = (verdicts.assign(_rank=rank,
                                    _size=verdicts.pooled_delta.abs())
                    .sort_values(["_rank", "_size"], ascending=[True, False])
                    .drop(columns=["_rank", "_size"]))
        _write(verdicts, tables / "skill-stratified-verdicts.csv")

    # --- figures ------------------------------------------------------------
    written = [
        fig_volume(monthly, figures / "01-monthly-volume-and-velocity.png"),
        fig_panel(sensitivity, verdict, figures / "02-panel-sensitivity.png"),
        fig_publishers(panel, presence, figures / "03-publisher-panel.png"),
        fig_spikes(weekly, spikes, attributed, figures / "04-weekly-spikes.png"),
    ]
    if "job_function" in segments:
        written.append(fig_mix(segments["job_function"],
                               figures / "05-job-function-mix.png"))
    if not skill_velocity.empty:
        written.append(fig_skill_velocity(skill_velocity,
                                          figures / "06-skill-velocity.png"))
    overturned = verdicts[verdicts.overturned_by_stratification] \
        if "overturned_by_stratification" in verdicts.columns else pd.DataFrame()
    if not overturned.empty:
        headline = overturned.iloc[0].skill
        written.append(fig_paradox(
            within, pooled, headline,
            figures / f"07-simpsons-paradox-{headline.lower()}.png"))
    written.append(fig_dow(seasonality, figures / "08-day-of-week.png"))

    # --- evidence report ----------------------------------------------------
    complete = monthly[~monthly.is_partial.astype(bool)]
    first, last = (complete.iloc[0], complete.iloc[-1]) if len(complete) >= 2 \
        else (monthly.iloc[0], monthly.iloc[-1])

    report = {
        "company": company,
        "rows_in": int(len(features)),
        "observation_window": [str(start.date()), str(end.date())],
        "periods": {
            "months": int(len(monthly)),
            "weeks": int(len(weekly)),
            "quarters": int(len(quarterly)),
            "partial_months": monthly.loc[monthly.is_partial.astype(bool),
                                          "period"].tolist(),
        },
        "velocity": {
            "first_complete_period": str(first.period),
            "last_complete_period": str(last.period),
            "postings_per_week_first": float(first.postings_per_week),
            "postings_per_week_last": float(last.postings_per_week),
            "median_postings_per_week": round(
                float(monthly.postings_per_week.median()), 3),
            "min_month": str(monthly.loc[monthly.postings.idxmin(), "period"]),
            "max_month": str(monthly.loc[monthly.postings.idxmax(), "period"]),
        },
        "publisher_panel": {
            "n_publishers": int(len(panel)),
            "present_in_all_months": int(
                (panel.periods_present == len(monthly)).sum()),
            "present_in_one_month_only": int((panel.periods_present == 1).sum()),
            "panel_publishers_at_75pct": keep,
            "share_of_postings_in_panel": round(
                float(features[tr.PUBLISHER_COL].isin(keep).mean()), 4)
            if keep else 0.0,
        },
        # The Task 05 headline: the direction of the trend is a function of the
        # panel you declare, so the panel is reported before the trend is.
        "panel_verdict": {
            "direction": verdict.direction,
            "treatments_agree": bool(verdict.agrees),
            "detail": verdict.detail,
            "final_index": {
                col.replace("_index", ""): float(sensitivity.iloc[-1][col])
                for col in ("raw_index", "balanced_index", "chained_index",
                            "bilateral_index")
                if col in sensitivity.columns
                and pd.notna(sensitivity.iloc[-1][col])
            },
        },
        "spikes": {
            "n_flagged": int(spikes.is_spike.sum()),
            "attributed": attributed.to_dict("records")
            if not attributed.empty else [],
            "n_batches": int(len(batches)),
            "postings_in_batches": int(batches.postings.sum())
            if not batches.empty else 0,
            "share_of_postings_in_batches": round(
                float(batches.postings.sum() / max(len(features), 1)), 4)
            if not batches.empty else 0.0,
        },
        "seasonality": {
            "month_of_year_identifiable": bool(
                seasonality.loc[seasonality.cycle == "month_of_year",
                                "identifiable"].all()),
            "day_of_week_identifiable": bool(
                seasonality.loc[seasonality.cycle == "day_of_week",
                                "identifiable"].all()),
            "weekend_share": round(float(tr.weekend_share(features)), 4),
            "uniform_weekend_share": round(2 / 7, 4),
        },
        "composition": {
            dim: seg.loc[seg.direction != "stable",
                         ["segment", "n_h1", "n_h2", "count_change_pct",
                          "share_change_pct", "direction"]].to_dict("records")
            for dim, seg in segments.items()
        },
        "job_function_directions_agree_across_panels": (
            int(panel_check.directions_agree.sum()) if not panel_check.empty
            else None),
        "job_function_segments_checked": (
            int(len(panel_check)) if not panel_check.empty else 0),
        "skills": {
            "rule_share_of_all_emitted": "share_of_all" in skill_velocity.columns,
            "excluded_job_functions": sorted(tr.SKILL_EXCLUDED_FUNCTIONS),
            "skills_with_support": len(tested),
            "verdict_counts": (verdicts.verdict.value_counts().to_dict()
                               if not verdicts.empty else {}),
            "overturned_by_stratification": (
                overturned.skill.tolist() if not overturned.empty else []),
        },
        "figures": written,
        # Standing Task 01 commitment — re-checked on every run, never assumed.
        "personal_data_columns_present": sorted({
            c
            for table in (monthly, weekly, panel, sensitivity, spikes,
                          seasonality, skill_velocity, within)
            for c in table.columns
            if any(k in c.lower() for k in ("email", "phone", "candidate",
                                            "applicant", "recruiter"))
        }),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    qa = member / "task-05-trend-report.json"
    qa.write_text(json.dumps(report, indent=2) + "\n")

    print(f"window           {start.date()} -> {end.date()} "
          f"({len(monthly)} months, {len(weekly)} ISO weeks)")
    print(f"velocity         {report['velocity']['postings_per_week_first']} "
          f"-> {report['velocity']['postings_per_week_last']} postings/week")
    print(f"publishers       {len(panel)} total, "
          f"{report['publisher_panel']['present_in_one_month_only']} in one "
          f"month only, {len(keep)} in the 75% panel")
    print(f"panel verdict    {verdict.direction} "
          f"(treatments agree: {verdict.agrees})")
    print(f"                 {report['panel_verdict']['final_index']}")
    print(f"spikes           {int(spikes.is_spike.sum())} flagged, "
          f"{len(batches)} publisher batches "
          f"({100 * report['spikes']['share_of_postings_in_batches']:.1f}% of "
          f"postings)")
    print(f"skill verdicts   {report['skills']['verdict_counts']}")
    print("overturned       "
          f"{report['skills']['overturned_by_stratification'] or 'none'}")
    print(f"tables           -> {tables.relative_to(REPO_ROOT)}/")
    print(f"figures          -> {figures.relative_to(REPO_ROOT)}/ "
          f"({len(written)} files)")
    print(f"report           -> {qa.relative_to(REPO_ROOT)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", default="google")
    parser.add_argument("--features", default=None,
                        help="defaults to the Task 04 feature parquet")
    parser.add_argument("--skills", default=None,
                        help="defaults to the Task 04 skills-long parquet")
    args = parser.parse_args()

    company = args.company
    features_path = args.features or \
        f"data/processed/{company}/{company}_features.parquet"
    skills_path = args.skills or \
        f"data/processed/{company}/{company}_skills_long.parquet"

    features = pd.read_parquet(REPO_ROOT / features_path)
    long = pd.read_parquet(REPO_ROOT / skills_path)
    print(f"read {len(features):,} postings <- {features_path}")
    print(f"read {len(long):,} skill rows <- {skills_path}")
    build(features, long, company)


if __name__ == "__main__":
    main()
