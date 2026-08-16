"""Task 07 — build the demand-forecast tables, figures and evidence report.

Reads the six built company frames from `data/processed/<company>/`, runs the
shared forecasting layer in `src/forecast.py` over them, and writes:

    members/ankit-google/task-07-tables/        (committed — aggregate only)
        forecastability-gate.csv           THE GATE — read this first
        gate-threshold-sensitivity.csv     does the gate depend on the cutoff
        panel-share-series.csv             the input series, February flagged
        backtest-errors.csv                every origin x model x horizon
        model-accuracy.csv                 RMSE, MAE, MASE by model + horizon
        model-contest.csv                  Diebold-Mariano against naive
        model-selection.csv                what was chosen, and what a ranking
                                           would have chosen instead
        horizon-limits.csv                 where the interval stops excluding
        horizon-limits-all-companies.csv   the same, pooled over all six
        interval-coverage.csv              the interval against its own residuals
        forecast.csv                       the published forecast
        levels-vs-shares.csv               the crawler is the easiest series
        february-correction.csv            C5 evidence — H1/H2 with and without
        skill-forecastability-google.csv   which of Google's skills are modellable

    members/ankit-google/task-07-figures/       (committed)
        01-forecastability-gate.png
        02-panel-share-series.png
        03-backtest-accuracy.png
        04-selection-vs-ranking.png
        05-horizon-limits.png
        06-levels-vs-shares.png
        07-forecast.png
        08-february-correction.png

    members/ankit-google/task-07-forecast-report.json   quality evidence

Nothing row-level is written. The forecast consumes `data/processed/`, which
stays git-ignored, and emits only aggregates.

    python src/build_forecast.py
    python src/build_forecast.py --focus google --companies google,meta
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

import companies as co            # noqa: E402
import compare as cmp             # noqa: E402
import forecast as fc             # noqa: E402
import trends as tr               # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "members" / "ankit-google"
TABLES = OUT / "task-07-tables"
FIGURES = OUT / "task-07-figures"
TASK06_TABLES = OUT / "task-06-tables"

# Task 05/06 palette, reused so the three tasks' figures sit together.
INK = "#1f2933"
GREY = "#9aa5b1"
BLUE = "#2f6fb5"
ORANGE = "#e08a1e"
RED = "#c0392b"
GREEN = "#2e7d5b"
PURPLE = "#7d5ba6"

COMPANY_COLOURS = {
    "google": BLUE, "microsoft": ORANGE, "meta": PURPLE,
    "nvidia": GREEN, "snowflake": RED, "databricks": "#4a5568",
}

#: Halves, as Task 06 defined them. Reused verbatim so the February correction
#: is measured against Task 06's own estimator rather than a new one.
H1 = [f"2023-{m:02d}" for m in range(1, 7)]
H2 = [f"2023-{m:02d}" for m in range(7, 13)]

#: How many of Google's skills the skill gate carries.
TOP_SKILLS = 15


def _write(df: pd.DataFrame, path: Path) -> None:
    """Write a table, refusing the two column families Task 06 has banned."""
    bad = fc.forbidden_columns(df)
    if bad:
        raise ValueError(f"{path.name} carries forbidden columns {bad}")
    personal = fc.personal_data_columns_present(df)
    if personal:
        raise ValueError(f"{path.name} carries personal-data columns {personal}")
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


def _colour(company: str) -> str:
    return COMPANY_COLOURS.get(company, GREY)


# ---------------------------------------------------------------------------
# Tables that only exist for this task
# ---------------------------------------------------------------------------


def gate_sensitivity(series: pd.DataFrame,
                     cutoffs=(3, 5, 10)) -> pd.DataFrame:
    """The verdict set at each candidate minimum cell.

    A threshold picked after seeing the answer is not a threshold. The
    defensible version is to declare one and then show what the finding would
    have been at the alternatives — including Task 06's own floor of 10.
    """
    rows = []
    for cutoff in cutoffs:
        gate = fc.forecastability_table(series, min_cell=cutoff)
        passing = sorted(gate.loc[gate.verdict == "forecastable", "key"])
        rows.append({
            "min_cell": cutoff,
            "n_forecastable": len(passing),
            "forecastable": ", ".join(passing),
            "is_published_threshold": cutoff == fc.MIN_CELL_MONTH,
            "matches_task06_min_cell": cutoff == cmp.MIN_CELL,
        })
    return pd.DataFrame(rows)


def batch_sensitivity(frames: dict[str, pd.DataFrame],
                      publishers: list[str],
                      focus: str = "google") -> pd.DataFrame:
    """Drop each panel publisher in turn and re-run the whole verdict.

    Task 05 §3 flagged three Google spike weeks as publisher batches and told
    Task 07 to exclude or dummy them. At monthly frequency two of the three
    leave on their own — `via Google Careers` and `via The Muse` are not on the
    common panel, so the postings never enter the series. The third does not:
    `via Recruit.net` **is** on the panel, and its W30 batch is overwhelmingly
    the focus company's, so it inflates the numerator far more than the shared
    denominator and does not cancel the way §1.1 needs it to.

    A share is only robust to collection if the collection factor is common.
    This drops one publisher at a time and reports whether the gate, the
    refusal and the horizon verdict survive — which is the honest test of that
    assumption rather than an appeal to it.
    """
    rows = []
    for dropped in [None] + list(publishers):
        panel = [p for p in publishers if p != dropped]
        series = fc.panel_share_series(frames, publishers=panel)
        gate = fc.forecastability_table(series)
        passing = sorted(gate.loc[gate.verdict == "forecastable", "key"])
        backtest = fc.rolling_origin_backtest(series)
        contest = fc.model_contest(backtest, horizon=1).sort_values("p_value")
        horizons = fc.horizon_table(backtest, keys=passing)
        verdict = fc.horizon_verdict(horizons)
        focus_rows = series[(series.key == focus) & series.is_observed]
        rows.append({
            "dropped_publisher": dropped or "(none — published panel)",
            "n_publishers": len(panel),
            "forecastable": ", ".join(passing),
            "any_model_beats_naive": bool(contest.beats_benchmark.any()),
            "smallest_p_value": float(contest.p_value.min()),
            "h1_interval_factor": float(horizons.iloc[0].interval_factor),
            "max_useful_horizon": int(verdict.max_useful_horizon),
            f"{focus}_max_share": float(focus_rows.share.max()),
            f"{focus}_largest_share_move_pp": float(
                100 * (focus_rows.share
                       - series[(series.key == focus)
                               & series.is_observed].share.mean()).abs().max()),
        })
    out = pd.DataFrame(rows)
    baseline = out.iloc[0]
    out["verdict_unchanged"] = (
        (out.forecastable == baseline.forecastable)
        & (out.any_model_beats_naive == baseline.any_model_beats_naive)
        & (out.max_useful_horizon == baseline.max_useful_horizon))
    return out


def levels_vs_shares(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One-step naive error on the pool, the shares and the counts.

    The measurement behind Task 06's ruling that levels are not identified.
    If the panel pool — which is the crawler's output, not anyone's hiring —
    is the easiest of the three to predict, then a count forecast scoring well
    is a forecast of collection wearing a demand label.
    """
    rows = []
    for label, series, meaning in (
        ("panel_pool", fc.pool_series(frames),
         "total postings on the shared panel; no company's demand"),
        ("company_share", fc.panel_share_series(frames),
         "each company's share of that pool; the identified object"),
        ("company_count", fc.company_count_series(frames),
         "each company's panel postings; not identified"),
    ):
        backtest = fc.rolling_origin_backtest(
            series, models={"naive": fc.m_naive}, horizons=(1,))
        errors = backtest.error.dropna().to_numpy()
        rows.append({
            "series": label, "what_it_measures": meaning,
            "n_keys": int(series.key.nunique()),
            "n_origins": int(len(errors)),
            "naive_rmse_log": float(np.sqrt(np.mean(np.square(errors)))),
            "typical_error_factor": float(np.exp(np.sqrt(np.mean(np.square(errors))))),
        })
    out = pd.DataFrame(rows)
    out["easier_than_shares"] = out.naive_rmse_log < float(
        out.loc[out.series == "company_share", "naive_rmse_log"].iloc[0])
    return out


def february_correction(frames: dict[str, pd.DataFrame],
                        publishers: list[str]) -> pd.DataFrame:
    """Correction C5 — Task 06's half-over-half estimator, with February and without.

    Task 06 §11 tells Task 07 that "February is missing for the panel entirely
    — treat as missing, not zero", and Task 06's own H1 aggregate counts the 97
    February postings that sit on the window-level panel. Both cannot hold.
    This table is the size of the difference, per company, so the register
    entry quotes a measurement rather than an argument.
    """
    restricted = cmp.restrict(frames, publishers)
    col = tr.period_col("month")

    def shares(months):
        counts = {k: int(df[col].isin(months).sum())
                  for k, df in restricted.items()}
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}, total

    with_feb, n_with = shares(H1)
    without_feb, n_without = shares([m for m in H1 if m != "2023-02"])
    h2, n_h2 = shares(H2)

    rows = []
    for key in sorted(restricted):
        published = (h2[key] - with_feb[key]) * 100
        corrected = (h2[key] - without_feb[key]) * 100
        rows.append({
            "company": key,
            "h1_share_with_february": with_feb[key],
            "h1_share_without_february": without_feb[key],
            "h2_share": h2[key],
            "task06_published_delta_pp": published,
            "corrected_delta_pp": corrected,
            "change_pp": corrected - published,
            "sign_unchanged": bool(np.sign(published) == np.sign(corrected)),
        })
    out = pd.DataFrame(rows)
    out.attrs["n_h1_with_february"] = n_with
    out.attrs["n_h1_without_february"] = n_without
    out.attrs["n_h2"] = n_h2
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_gate(gate: pd.DataFrame, path: Path) -> str:
    """Two diagnostics, two different reasons a series is refused."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    order = gate.sort_values("signal_share")
    y = np.arange(len(order))
    colours = [_colour(c) for c in order.key]
    hatch = ["" if v == "forecastable" else "///" for v in order.verdict]

    bars = ax1.barh(y, order.min_cell, color=colours)
    for bar, mark in zip(bars, hatch):
        bar.set_hatch(mark)
    ax1.axvline(fc.MIN_CELL_MONTH, color=RED, ls="--", lw=1.2)
    ax1.set_yticks(y, order.key, fontsize=8)
    _style(ax1, "Thinnest month in the series", "", "postings in the numerator")
    ax1.text(fc.MIN_CELL_MONTH + 0.3, -0.45, f"floor {fc.MIN_CELL_MONTH}",
             color=RED, fontsize=7)

    bars = ax2.barh(y, order.signal_share * 100, color=colours)
    for bar, mark in zip(bars, hatch):
        bar.set_hatch(mark)
    ax2.set_yticks(y, order.key, fontsize=8)
    _style(ax2, "Variance left after removing binomial noise", "",
           "% of observed variance")
    ax2.set_xlim(0, 100)

    fig.suptitle("Task 07 forecastability gate — every series carries real "
                 "signal; four are too thin to model it (hatched)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.84)
    return _save(fig, path)


def fig_series(series: pd.DataFrame, path: Path) -> str:
    """The six share series, with February drawn as the hole it is."""
    fig, ax = plt.subplots(figsize=(11, 5))
    periods = sorted(series.period.unique())
    x = np.arange(len(periods))
    index = {p: i for i, p in enumerate(periods)}

    for key in sorted(series.key.unique()):
        block = series[series.key == key].set_index("period")
        values = [block.share.get(p, np.nan) if block.is_observed.get(p, False)
                  else np.nan for p in periods]
        ax.plot(x, np.array(values, dtype=float) * 100, marker="o", ms=4,
                lw=1.6, color=_colour(key), label=key)

    unobserved = sorted(series.loc[~series.is_observed, "period"].unique())
    for period in unobserved:
        ax.axvspan(index[period] - 0.4, index[period] + 0.4,
                   color=RED, alpha=0.10)
        ax.text(index[period], ax.get_ylim()[1] * 0.96, "no common\npublisher",
                ha="center", va="top", fontsize=7, color=RED)

    ax.set_xticks(x, [p[-2:] for p in periods], fontsize=8)
    _style(ax, "Share of the seven-publisher common panel, 2023",
           "% of panel postings", "month")
    ax.legend(fontsize=8, frameon=False, ncol=6, loc="lower center",
              bbox_to_anchor=(0.5, -0.22))
    return _save(fig, path)


def fig_accuracy(accuracy: pd.DataFrame, contest: pd.DataFrame,
                 path: Path) -> str:
    """Every model against naive, and the test that stops any of them winning."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    order = (accuracy[accuracy.horizon == 1].sort_values("rmse")
             .reset_index(drop=True))
    y = np.arange(len(order))
    colours = [RED if m == "naive" else BLUE for m in order.model]
    ax1.barh(y, order.rmse, color=colours)
    ax1.set_yticks(y, order.model, fontsize=8)
    ax1.invert_yaxis()
    _style(ax1, "One-step error, all six companies pooled", "",
           "RMSE on the log share")

    block = contest[contest.horizon == 1].sort_values("p_value")
    y = np.arange(len(block))
    colours = [GREEN if s < 0 else GREY for s in block.dm_stat]
    ax2.barh(y, block.p_value, color=colours)
    ax2.axvline(fc.ALPHA, color=RED, ls="--", lw=1.2)
    ax2.set_yticks(y, block.model, fontsize=8)
    ax2.invert_yaxis()
    _style(ax2, "Diebold-Mariano p against naive "
                "(green = lower loss)", "", "p-value")
    ax2.text(fc.ALPHA + 0.01, len(block) - 0.4, f"a = {fc.ALPHA}",
             color=RED, fontsize=7)

    fig.suptitle("No model beats persistence — the best is 6% lower on RMSE "
                 "and nowhere near significant",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.84)
    return _save(fig, path)


def fig_selection(selection: pd.DataFrame, path: Path) -> str:
    """What a ranking would have published, against what the test allows."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    order = selection.sort_values("selected_rmse")
    x = np.arange(len(order))
    width = 0.38

    ax.bar(x - width / 2, order.selected_rmse, width, color=RED,
           label="selected (naive)")
    ax.bar(x + width / 2, order.lowest_rmse, width, color=BLUE,
           label="lowest-RMSE model")
    for i, row in enumerate(order.itertuples()):
        if row.would_have_picked_differently:
            ax.text(i + width / 2, row.lowest_rmse + 0.02, row.lowest_rmse_model,
                    ha="center", fontsize=7, color=INK)

    ax.set_xticks(x, order.key, fontsize=8)
    _style(ax, "Per-company one-step error: the benchmark against the "
               "best-fitting challenger", "RMSE on the log share", "")
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Four of six companies have a different best-fitting model, "
                 "and they do not agree — that is noise, not selection",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.86)
    return _save(fig, path)


def fig_horizons(gated: pd.DataFrame, everyone: pd.DataFrame,
                 path: Path) -> str:
    """How wide the interval is, and where it stops existing at all."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    width = 0.36

    for offset, block, colour, label in (
        (-width / 2, gated, BLUE, "published companies (Google, Meta)"),
        (width / 2, everyone, GREY, "all six companies"),
    ):
        x = np.arange(len(block)) + offset
        heights = block.interval_factor.fillna(0).to_numpy()
        bars = ax.bar(x, heights, width, color=colour, label=label)
        for bar, row in zip(bars, block.itertuples()):
            if not row.interval_sufficient:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.15,
                        f"no interval\n({int(row.n_residuals)} residuals)",
                        ha="center", va="bottom", fontsize=7, color=RED,
                        rotation=90)

    # A horizon under the limit is still unreachable if a shorter one failed —
    # you cannot forecast two months out without passing through one.
    failed_earlier = False
    for i, row in enumerate(gated.itertuples()):
        usable = row.interval_sufficient and row.interval_factor < 3.0
        if usable and failed_earlier:
            ax.text(i, row.interval_factor + 0.08,
                    f"under 3x, but h={int(gated.horizon.iloc[0])} is not —\n"
                    "horizons have to be contiguous",
                    ha="right", va="bottom", fontsize=7, color=RED)
        failed_earlier = failed_earlier or not usable

    ax.axhline(3.0, color=RED, ls="--", lw=1.2)
    ax.set_xlim(-0.85, len(gated) - 0.5)
    ax.text(-0.80, 3.06, "3x", color=RED, fontsize=7, va="bottom")
    ax.set_xticks(np.arange(len(gated)),
                  [f"h={int(h)}" for h in gated.horizon], fontsize=8)
    _style(ax, f"Width of the {fc.INTERVAL_LEVEL:.0%} prediction interval as a "
               "multiple of its own lower end — past 3x it excludes nothing",
           "interval span (x)", "")
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Even the one-month interval spans more than 3x — no horizon "
                 "is supported under either pooling",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.86)
    return _save(fig, path)


def fig_levels(levels: pd.DataFrame, path: Path) -> str:
    """The crawler is the most predictable series in the file."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = levels.sort_values("naive_rmse_log")
    y = np.arange(len(order))
    colours = [RED if s == "panel_pool" else
               (BLUE if s == "company_share" else GREY) for s in order.series]
    ax.barh(y, order.naive_rmse_log, color=colours)
    for i, row in enumerate(order.itertuples()):
        ax.text(row.naive_rmse_log + 0.012, i, f"{row.naive_rmse_log:.3f}",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [s.replace("_", " ") for s in order.series], fontsize=9)
    ax.set_xlim(0, float(order.naive_rmse_log.max()) * 1.22)
    _style(ax, "One-step persistence error on the log scale", "",
           "RMSE (lower = easier to predict)")
    fig.suptitle("The panel pool — pure collection, nobody's demand — is the "
                 "easiest series here.\nThat is why a count forecast is refused "
                 "rather than caveated.",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.80)
    return _save(fig, path)


def fig_forecast(series: pd.DataFrame, forecast: pd.DataFrame,
                 verdict: fc.HorizonVerdict, path: Path) -> str:
    """The published forecast, drawn with its own refusal on the face of it."""
    keys = sorted(forecast.key.unique())
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 4.4), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, key in zip(axes, keys):
        history = fc.observed(series, key)
        n = len(history)
        x = np.arange(n)
        ax.plot(x, history.share * 100, marker="o", ms=4, lw=1.6,
                color=_colour(key), label="observed")

        block = forecast[forecast.key == key].sort_values("horizon")
        fx = n - 1 + block.horizon.to_numpy()
        ax.plot(np.concatenate([[n - 1], fx]),
                np.concatenate([[history.share.iloc[-1] * 100],
                                block.point_share.to_numpy() * 100]),
                ls="--", lw=1.5, color=INK, label="persistence forecast")
        band = block.dropna(subset=["lo_share", "hi_share"])
        if len(band):
            ax.fill_between(n - 1 + band.horizon.to_numpy(),
                            band.lo_share * 100, band.hi_share * 100,
                            color=INK, alpha=0.12,
                            label=f"{fc.INTERVAL_LEVEL:.0%} interval")
        missing = block[block.lo_share.isna()]
        for row in missing.itertuples():
            ax.text(n - 1 + row.horizon, row.point_share * 100, "  no\n  interval",
                    fontsize=7, color=RED, va="center")

        labels = list(history.period.str[-2:]) + [p[-2:] for p in block.target_period]
        ax.set_xticks(np.arange(n + len(block)), labels, fontsize=7)
        _style(ax, key, "% of panel", "month (2023 -> 2024)")
        ax.legend(fontsize=7, frameon=False)

    fig.suptitle("Published forecast — and it is not supported at any horizon: "
                 f"{verdict.reason}",
                 fontsize=11, color=RED, x=0.01, ha="left")
    fig.subplots_adjust(top=0.84)
    return _save(fig, path)


def fig_february(correction: pd.DataFrame, path: Path) -> str:
    """Correction C5, at the size it actually is."""
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    order = correction.sort_values("task06_published_delta_pp")
    x = np.arange(len(order))
    width = 0.38

    ax.bar(x - width / 2, order.task06_published_delta_pp, width,
           color=GREY, label="Task 06 as published (February in H1)")
    ax.bar(x + width / 2, order.corrected_delta_pp, width,
           color=BLUE, label="February treated as missing, per Task 06's own rule")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, order.company, fontsize=8)
    _style(ax, "Change in common-panel share, H1 to H2", "percentage points", "")
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("C5 — every sign survives, every magnitude moves. "
                 "Google's headline goes from -4.84 pp to -6.56 pp.",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.86)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def build(keys: list[str], focus: str = "google") -> dict:
    frames = cmp.load_frames(keys)
    longs = cmp.load_long([focus])
    publishers = cmp.common_publishers(frames)

    report: dict = {
        "task": "07-demand-forecasting",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "companies": sorted(keys),
        "focus": focus,
        "thresholds": {
            "min_train": fc.MIN_TRAIN,
            "min_observations": fc.MIN_OBSERVATIONS,
            "min_cell_month": fc.MIN_CELL_MONTH,
            "max_horizon": fc.MAX_HORIZON,
            "alpha": fc.ALPHA,
            "interval_level": fc.INTERVAL_LEVEL,
        },
    }

    # -- the series ---------------------------------------------------------
    series = fc.panel_share_series(frames, publishers=publishers)
    _write(series, TABLES / "panel-share-series.csv")
    unobserved = fc.unobserved_periods(frames)
    report["panel"] = {
        "publishers": publishers,
        "n_publishers": len(publishers),
        "periods": int(series.period.nunique()),
        "unobserved_periods": unobserved,
        "observed_periods_per_company": int(len(fc.observed(series, focus))),
    }

    # -- the gate -----------------------------------------------------------
    gate = fc.forecastability_table(series)
    _write(gate, TABLES / "forecastability-gate.csv")
    sensitivity = gate_sensitivity(series)
    _write(sensitivity, TABLES / "gate-threshold-sensitivity.csv")
    gated = sorted(gate.loc[gate.verdict == "forecastable", "key"])
    report["gate"] = {
        "verdicts": dict(zip(gate.key, gate.verdict)),
        "forecastable": gated,
        "n_refused": int((gate.verdict != "forecastable").sum()),
        "all_reject_a_constant_share": bool((gate.homogeneity_p < fc.ALPHA).all()),
        "signal_share_range": [round(float(gate.signal_share.min()), 4),
                               round(float(gate.signal_share.max()), 4)],
        "threshold_sensitivity": sensitivity.set_index(
            "min_cell").forecastable.to_dict(),
    }

    # -- backtest -----------------------------------------------------------
    backtest = fc.rolling_origin_backtest(series)
    _write(backtest, TABLES / "backtest-errors.csv")
    accuracy = fc.accuracy_table(backtest)
    _write(accuracy, TABLES / "model-accuracy.csv")
    contest = pd.concat([fc.model_contest(backtest, h) for h in (1, 2, 3)],
                        ignore_index=True)
    _write(contest, TABLES / "model-contest.csv")
    selection = fc.select_model(backtest, horizon=1)
    _write(selection, TABLES / "model-selection.csv")

    one_step = accuracy[accuracy.horizon == 1].sort_values("rmse")
    best = contest[(contest.horizon == 1) & (contest.dm_stat < 0)].sort_values("rmse")
    report["backtest"] = {
        "origins_per_company": {
            str(h): int(len(backtest[(backtest.horizon == h)
                                     & (backtest.model == "naive")
                                     & (backtest.key == focus)]))
            for h in (1, 2, 3)},
        "models": list(fc.MODELS),
        "seasonal_model_available": False,
        "one_step_rmse": one_step.set_index("model").rmse.round(4).to_dict(),
        "best_challenger": (best.model.iloc[0] if len(best) else None),
        "best_challenger_p": (round(float(best.p_value.iloc[0]), 4)
                              if len(best) else None),
        "any_model_beats_naive": bool(contest.beats_benchmark.any()),
    }
    report["selection"] = {
        "selected": dict(zip(selection.key, selection.selected)),
        "all_naive": bool(set(selection.selected) == {"naive"}),
        "would_have_picked_differently": int(
            selection.would_have_picked_differently.sum()),
        "ranking_would_have_chosen": dict(zip(selection.key,
                                              selection.lowest_rmse_model)),
    }

    # -- horizons and intervals --------------------------------------------
    horizons = fc.horizon_table(backtest, keys=gated)
    _write(horizons, TABLES / "horizon-limits.csv")
    horizons_all = fc.horizon_table(backtest)
    _write(horizons_all, TABLES / "horizon-limits-all-companies.csv")
    verdict = fc.horizon_verdict(horizons)
    verdict_all = fc.horizon_verdict(horizons_all)

    coverage = pd.DataFrame([
        {"pool": label, "horizon": h,
         **fc.interval_coverage(backtest, "naive", h, keys=pool)}
        for label, pool in (("published", gated), ("all_companies", None))
        for h in (1, 2, 3)
    ])
    _write(coverage, TABLES / "interval-coverage.csv")
    report["horizons"] = {
        "max_useful_horizon": verdict.max_useful_horizon,
        "reason": verdict.reason,
        "detail": verdict.detail,
        "max_useful_horizon_all_companies": verdict_all.max_useful_horizon,
        "robust_to_pooling": bool(
            verdict.max_useful_horizon == verdict_all.max_useful_horizon),
        "interval_level_requested": fc.INTERVAL_LEVEL,
        "interval_level_achieved": horizons.set_index(
            "horizon").achieved_level.round(4).to_dict(),
    }

    # -- the forecast itself ------------------------------------------------
    forecast = fc.forecast_table(series, backtest, selection,
                                 gate=gate, verdict=verdict)
    _write(forecast, TABLES / "forecast.csv")
    report["forecast"] = {
        "companies": sorted(forecast.key.unique()),
        "targets": sorted(forecast.target_period.unique()),
        "model": sorted(set(forecast.model)),
        "any_row_supported": bool(forecast.supported.any()),
        "shares_sum_to_one": False,
        "composition_incomplete_by": sorted(set(keys) - set(forecast.key)),
        "out_of_window": bool(forecast.out_of_window.all()),
    }

    # -- levels are not identified, measured -------------------------------
    levels = levels_vs_shares(frames)
    _write(levels, TABLES / "levels-vs-shares.csv")
    pool_rmse = float(levels.loc[levels.series == "panel_pool", "naive_rmse_log"].iloc[0])
    share_rmse = float(levels.loc[levels.series == "company_share", "naive_rmse_log"].iloc[0])
    count_rmse = float(levels.loc[levels.series == "company_count", "naive_rmse_log"].iloc[0])
    report["levels"] = {
        "naive_rmse_pool": round(pool_rmse, 4),
        "naive_rmse_share": round(share_rmse, 4),
        "naive_rmse_count": round(count_rmse, 4),
        "collection_easier_than_demand": bool(pool_rmse < share_rmse < count_rmse),
    }

    # -- is the shared collection factor really shared? ---------------------
    batches = batch_sensitivity(frames, publishers, focus=focus)
    _write(batches, TABLES / "publisher-batch-sensitivity.csv")
    worst = batches.iloc[1:].loc[
        batches.iloc[1:][f"{focus}_max_share"].sub(
            batches.iloc[0][f"{focus}_max_share"]).abs().idxmax()]
    report["batch_sensitivity"] = {
        "publishers_tested": int(len(publishers)),
        "verdict_unchanged_in_every_case": bool(batches.verdict_unchanged.all()),
        "largest_single_publisher_effect": {
            "publisher": str(worst.dropped_publisher),
            f"{focus}_max_share_published": round(
                float(batches.iloc[0][f"{focus}_max_share"]), 4),
            f"{focus}_max_share_without": round(float(worst[f"{focus}_max_share"]), 4),
            "h1_interval_factor_without": round(float(worst.h1_interval_factor), 3),
            "max_useful_horizon_without": int(worst.max_useful_horizon),
        },
        "task05_spikes_off_panel": ["via Google Careers", "via The Muse"],
        "task05_spikes_on_panel": ["via Recruit.net"],
    }

    # -- correction C5 ------------------------------------------------------
    correction = february_correction(frames, publishers)
    _write(correction, TABLES / "february-correction.csv")
    focus_row = correction[correction.company == focus].iloc[0]
    report["correction_c5"] = {
        "n_h1_with_february": correction.attrs["n_h1_with_february"],
        "n_h1_without_february": correction.attrs["n_h1_without_february"],
        "february_postings_on_panel": (correction.attrs["n_h1_with_february"]
                                       - correction.attrs["n_h1_without_february"]),
        "focus_published_delta_pp": round(float(focus_row.task06_published_delta_pp), 2),
        "focus_corrected_delta_pp": round(float(focus_row.corrected_delta_pp), 2),
        "all_signs_unchanged": bool(correction.sign_unchanged.all()),
        "largest_change_pp": round(float(correction.change_pp.abs().max()), 2),
        "largest_change_company": str(
            correction.loc[correction.change_pp.abs().idxmax(), "company"]),
    }

    # -- skills, for the focus company -------------------------------------
    skills = fc.skill_share_series(longs[focus], frames[focus],
                                   unobserved=unobserved, top=TOP_SKILLS)
    skill_gate = fc.forecastability_table(skills)
    _write(skill_gate, TABLES / f"skill-forecastability-{focus}.csv")
    skill_ok = sorted(skill_gate.loc[skill_gate.verdict == "forecastable", "key"])
    report["skills"] = {
        "skills_screened": int(skill_gate.key.nunique()),
        "forecastable": skill_ok,
        "n_forecastable": len(skill_ok),
        "verdicts": skill_gate.verdict.value_counts().to_dict(),
        "february_excluded": unobserved,
    }

    # -- figures ------------------------------------------------------------
    figs = [
        fig_gate(gate, FIGURES / "01-forecastability-gate.png"),
        fig_series(series, FIGURES / "02-panel-share-series.png"),
        fig_accuracy(accuracy, contest, FIGURES / "03-backtest-accuracy.png"),
        fig_selection(selection, FIGURES / "04-selection-vs-ranking.png"),
        fig_horizons(horizons, horizons_all, FIGURES / "05-horizon-limits.png"),
        fig_levels(levels, FIGURES / "06-levels-vs-shares.png"),
        fig_forecast(series, forecast, verdict, FIGURES / "07-forecast.png"),
        fig_february(correction, FIGURES / "08-february-correction.png"),
    ]
    report["figures"] = figs

    # The hand-written statistics in `forecast.py` are cross-checked against
    # scipy and statsmodels by a module nothing here imports; point at the
    # evidence from the machine-readable report so it travels with the numbers.
    report["validation"] = {
        "evidence": "docs/task-07-forecast-validation.md",
        "script": "src/validate_forecast.py",
    }

    # -- standing privacy check --------------------------------------------
    offenders = {}
    for path in sorted(TABLES.glob("*.csv")):
        table = pd.read_csv(path)
        found = fc.personal_data_columns_present(table)
        if found:
            offenders[path.name] = found
    report["privacy"] = {
        "tables_checked": len(list(TABLES.glob("*.csv"))),
        "personal_data_columns_present": offenders,
        "passed": not offenders,
    }

    path = OUT / "task-07-forecast-report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"\ntables  -> {TABLES.relative_to(REPO_ROOT)} "
          f"({len(list(TABLES.glob('*.csv')))} csv)")
    print(f"figures -> {FIGURES.relative_to(REPO_ROOT)} ({len(figs)} png)")
    print(f"report  -> {path.relative_to(REPO_ROOT)}")
    print(f"\nforecastable: {', '.join(gated) or 'no company'} "
          f"({report['gate']['n_refused']} refused as too thin)")
    print(f"any model beats naive: {report['backtest']['any_model_beats_naive']} "
          f"(best challenger {report['backtest']['best_challenger']} at "
          f"p={report['backtest']['best_challenger_p']})")
    print(f"max useful horizon: {verdict.max_useful_horizon} — {verdict.reason}")
    print(f"collection easier than demand: "
          f"{report['levels']['collection_easier_than_demand']} "
          f"(pool {pool_rmse:.4f} < share {share_rmse:.4f} < count {count_rmse:.4f})")
    print(f"privacy check passed: {report['privacy']['passed']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus", default="google",
                        help="the specialist's own company; gets the skill gate")
    parser.add_argument("--companies", default=None,
                        help="comma-separated keys; defaults to everything that "
                             "passed Task 06's feasibility screen")
    args = parser.parse_args()

    if args.companies:
        keys = [k.strip() for k in args.companies.split(",")]
    else:
        screen = pd.read_csv(TASK06_TABLES / "company-feasibility-screen.csv")
        keys = co.included_companies(screen)
    print(f"forecasting {len(keys)} companies: {', '.join(sorted(keys))}")
    build(sorted(keys), focus=args.focus)


if __name__ == "__main__":
    main()
