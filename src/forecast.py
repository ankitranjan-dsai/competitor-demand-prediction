"""Task 07 — demand forecasting: the shared forecasting layer.

Shared, company-agnostic forecasting engine for all four specialists. The team
standard and its rationale live in docs/task-07-demand-forecasting-methods.md.

Why this module is mostly gates
-------------------------------
Tasks 05 and 06 spent their length establishing what this data cannot measure,
and Task 07 is where that bill comes due. The brief asks for a demand forecast.
The inputs are:

* **one year**, 2023, from one upstream dataset;
* **twelve monthly periods**, one of which (February) has *no* publisher
  carrying all six companies, so it is unobserved rather than low;
* a **level that is not identified** — Task 06 §1.3 gated it, because a
  company's posting count is a joint statement about its hiring and about how
  many boards syndicate it;
* pooled monthly cells between 1 and 76 postings, so a visible part of every
  wiggle is binomial sampling noise rather than demand.

Eleven observations do not support model selection, and this module is built
so that fact is enforced rather than described. Three rules do the work:

1. ``forecastability_table`` runs **before** any model is fitted and can
   return ``too_thin`` or ``noise_only``. A series that is statistically
   indistinguishable from a constant share does not get a trend line.
2. ``select_model`` returns **``naive`` unless a challenger beats it** on a
   rolling-origin backtest at ``ALPHA`` on a Diebold-Mariano test. Winning on
   average is not enough; on six origins the average is noise.
3. ``horizon_table`` publishes the error at every horizon, so a reader can see
   the point at which the interval stops excluding anything.

What is forecast, and why it is a share
---------------------------------------
The forecast object is each company's **share of the common publisher panel**,
not its posting count. Task 06 §2 showed that share is the one cross-company
volume quantity this data identifies: anything that scales the whole pool — a
crawler indexing harder in H2, a board joining mid-year — multiplies numerator
and denominator alike and cancels.

Forecasting the count instead would mean forecasting the pool, and the pool is
the collection. ``pool_series`` exists so that comparison can be published
rather than asserted: in this data the pool is by some distance the *easiest*
series in the file to predict, which is exactly why a level forecast must not
be published as a demand forecast. A model that scores well there has learned
the crawler.

Models
------
Seven, all of them small, all fitted on the log share:

=================  =========================================================
``naive``          last observed value. The benchmark, and the default.
``mean``           historical mean. Wins whenever the series is noise.
``drift``          random walk with drift through the first and last points.
``ses``            simple exponential smoothing, alpha on a grid.
``holt_damped``    damped local trend, three parameters on a grid.
``loglinear``      OLS trend — Task 05's ``log_growth``, fitted per company.
``ar1``            mean reversion at the estimated lag-1 coefficient.
=================  =========================================================

``seasonal_naive`` is deliberately **not** here. 2023 is a single year, so
there are zero complete seasonal cycles and a month-of-year term is not
identified (Task 05 §6). ``seasonal_naive`` exists only to raise, so that a
specialist who reaches for it gets the reason rather than a plausible number.

No scipy, no statsmodels, no Prophet
------------------------------------
Same dependency story as Tasks 05 and 06: pandas, numpy and ``math``. The
p-values come from ``math.erfc``, the smoothing parameters from a declared
grid. ``src/validate_forecast.py`` cross-checks this module against
statsmodels and reports the agreement, so the choice costs no accuracy and
buys a reviewer the ability to rebuild every published number.

Usage
-----
    import forecast as fc
    series = fc.panel_share_series(frames)
    gate = fc.forecastability_table(series)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import compare as cmp
import trends as tr

PUBLISHER_COL = tr.PUBLISHER_COL
SKILL_EXCLUDED_FUNCTIONS = tr.SKILL_EXCLUDED_FUNCTIONS

#: Inherited unchanged from Task 06 — a committed table may not carry a
#: cross-company country split or a ``share_of_all`` denominator.
FORBIDDEN_COLUMNS = cmp.FORBIDDEN_COLUMNS

# ---------------------------------------------------------------------------
# Declared thresholds. All of these are set here, before the data is seen,
# because a threshold chosen after the answer is not a threshold (Task 06 §1.2).
# ---------------------------------------------------------------------------

#: Smallest training window a model may be fitted on. Below five points a
#: three-parameter model interpolates its training set exactly.
MIN_TRAIN = 5

#: Fewest observed periods a series needs before it is eligible at all.
MIN_OBSERVATIONS = 8

#: Smallest monthly numerator a series may have in *any* observed period. A
#: month with two postings in it contributes a share whose binomial standard
#: error is larger than most of the movement being modelled.
MIN_CELL_MONTH = 5

#: Longest horizon published. Not a judgement that three months is useful —
#: `horizon_table` shows it is not — but the point past which the empirical
#: interval is built on fewer than a dozen residuals.
MAX_HORIZON = 3

#: Significance level for the Diebold-Mariano test that a model beats naive,
#: and for the homogeneity test in the gate.
ALPHA = 0.05

#: Prediction-interval coverage. 80 rather than 95: with 36 pooled residuals
#: the 2.5th percentile is one observation, and an interval whose endpoint is
#: a single backtest error is not an interval.
INTERVAL_LEVEL = 0.80

#: Grid for the exponential-smoothing parameters. Declared rather than
#: optimised continuously so that a rebuild on another machine returns the
#: same number to the last digit.
SES_ALPHA_GRID = tuple(round(a, 2) for a in np.linspace(0.05, 0.95, 19))
HOLT_ALPHA_GRID = tuple(round(a, 2) for a in np.linspace(0.10, 0.90, 9))
HOLT_BETA_GRID = tuple(round(b, 2) for b in np.linspace(0.05, 0.50, 10))
HOLT_PHI_GRID = (0.80, 0.90, 0.98)

#: Canonical column order for a demand series. Every function here consumes
#: and returns this shape, so a company series and a skill series are the
#: same object and the gate does not need two implementations.
SERIES_COLUMNS = ["key", "period", "numerator", "denominator", "share",
                  "is_observed"]


# ---------------------------------------------------------------------------
# Period arithmetic
# ---------------------------------------------------------------------------


def period_ordinal(keys, period: str = "month") -> np.ndarray:
    """Map period keys to evenly spaced integers on the calendar.

    The forecast index has to be **calendar** time, not position in the
    observed sequence. February is dropped from the Google panel, so the
    January row and the March row are adjacent in the array and two months
    apart in the world. A drift model indexed by position would spread
    January's move over one step instead of two and quietly inflate the slope.
    """
    keys = list(keys)
    if not keys:
        return np.array([], dtype=float)
    if period == "month":
        vals = [int(k[:4]) * 12 + int(k[5:7]) for k in keys]
    elif period == "quarter":
        vals = [int(k[:4]) * 4 + int(k[-1]) for k in keys]
    elif period == "week":
        vals = [int(k[:4]) * 53 + int(k.split("W")[-1]) for k in keys]
    else:
        raise ValueError(f"unknown period {period!r}")
    return np.asarray(vals, dtype=float) - float(min(vals))


def next_periods(last_key: str, n: int, period: str = "month") -> list[str]:
    """The ``n`` period keys following ``last_key``.

    These are the forecast targets, and for this dataset every one of them is
    outside the collection window. Nothing in this repo can ever check them,
    which is why `forecast_table` labels them ``out_of_window``.
    """
    if period != "month":
        raise ValueError("only monthly forecast targets are supported")
    year, month = int(last_key[:4]), int(last_key[5:7])
    out = []
    for _ in range(n):
        month += 1
        if month > 12:
            month, year = 1, year + 1
        out.append(f"{year:04d}-{month:02d}")
    return out


def _safe_log(share: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Log share with a Haldane-Anscombe half-count floor.

    A zero cell has log ``-inf``, which would take every downstream mean and
    every model fit with it. Task 06 §8 already uses the ½ correction on log
    lifts; the same correction is used here so a zero month shrinks toward the
    floor its denominator supports rather than leaving the real line.
    """
    share = np.asarray(share, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    floor = np.where(denominator > 0, 0.5 / np.maximum(denominator, 1.0), 0.5)
    return np.log(np.maximum(share, floor))


# ---------------------------------------------------------------------------
# Building the series
# ---------------------------------------------------------------------------


def unobserved_periods(frames: dict[str, pd.DataFrame],
                       period: str = "month") -> list[str]:
    """Periods in which no single publisher carries every company.

    **This is where February leaves the data, and it is derived rather than
    hard-coded.** Task 06 §1.3 measured the within-period intersection of the
    publisher sets and found it is zero in February: the six companies do
    share a panel across the window, but inside that month there is no
    publisher on which any two of them can be compared like for like.

    A share computed there is a ratio of counts drawn from disjoint channels.
    That is not a low month; it is an unobserved one, and every function
    downstream treats it as missing rather than as data.
    """
    by_period = cmp.common_panel_by_period(frames, period)
    return by_period.loc[by_period.n_common_publishers == 0, "period"].tolist()


def panel_share_series(frames: dict[str, pd.DataFrame],
                       period: str = "month",
                       publishers: list[str] | None = None) -> pd.DataFrame:
    """Each company's share of the common publisher panel, per period.

    The identified object (Task 06 §2). ``numerator`` is the company's panel
    postings, ``denominator`` the whole panel pool for that period, so the
    share is a composition over companies and sums to one within a period.
    """
    pubs = cmp.common_publishers(frames) if publishers is None else publishers
    restricted = cmp.restrict(frames, pubs)
    col = tr.period_col(period)
    unobserved = set(unobserved_periods(frames, period))

    counts = {k: df.groupby(col).size() for k, df in restricted.items()}
    periods = sorted(set().union(*[set(s.index) for s in counts.values()]))
    wide = pd.DataFrame({k: s.reindex(periods).fillna(0) for k, s in counts.items()},
                        index=periods).astype(int)
    pool = wide.sum(axis=1)

    rows = []
    for key in sorted(wide.columns):
        for p in periods:
            n = int(pool.loc[p])
            k = int(wide.loc[p, key])
            rows.append({
                "key": key, "period": p, "numerator": k, "denominator": n,
                "share": (k / n) if n else float("nan"),
                "is_observed": p not in unobserved and n > 0,
            })
    out = pd.DataFrame(rows, columns=SERIES_COLUMNS)
    out.attrs["panel_publishers"] = list(pubs)
    out.attrs["unobserved_periods"] = sorted(unobserved)
    return out


def pool_series(frames: dict[str, pd.DataFrame],
                period: str = "month",
                publishers: list[str] | None = None) -> pd.DataFrame:
    """The panel pool itself, as a series — the thing a level forecast forecasts.

    Published so that §"levels are not identified" is a measurement rather
    than an assertion. Run the same backtest on this series and on the shares:
    if the pool is the more predictable of the two, then a per-company count
    forecast is mostly a forecast of the collection, and its better score is
    the reason to refuse it rather than a reason to publish it.
    """
    pubs = cmp.common_publishers(frames) if publishers is None else publishers
    restricted = cmp.restrict(frames, pubs)
    col = tr.period_col(period)
    unobserved = set(unobserved_periods(frames, period))
    counts = {k: df.groupby(col).size() for k, df in restricted.items()}
    periods = sorted(set().union(*[set(s.index) for s in counts.values()]))
    pool = sum(s.reindex(periods).fillna(0) for s in counts.values()).astype(int)
    return pd.DataFrame([{
        "key": "panel_pool", "period": p, "numerator": int(pool.loc[p]),
        "denominator": int(pool.loc[p]), "share": float(pool.loc[p]),
        "is_observed": p not in unobserved,
    } for p in periods], columns=SERIES_COLUMNS)


def company_count_series(frames: dict[str, pd.DataFrame],
                         period: str = "month",
                         publishers: list[str] | None = None) -> pd.DataFrame:
    """Per-company panel **counts**, carried only as the refused alternative.

    ``denominator`` is set to the count itself so the frame keeps one schema.
    Nothing in the published forecast uses this; `build_forecast` backtests it
    beside the shares to show what a level forecast would have scored.
    """
    share = panel_share_series(frames, period, publishers)
    out = share.copy()
    out["denominator"] = out["numerator"]
    out["share"] = out["numerator"].astype(float)
    return out


def skill_share_series(long: pd.DataFrame, features: pd.DataFrame,
                       skills: list[str] | None = None,
                       period: str = "month",
                       unobserved: list[str] | None = None,
                       top: int = 15) -> pd.DataFrame:
    """Monthly ``share_of_skilled`` per skill, for one company.

    Denominator rules inherited whole from Task 04 §3.1 and Task 06 §4: only
    postings that carry at least one skill, with ``Facilities / Operations``
    excluded, because Google's data-centre roles genuinely carry no software
    skills and leaving them in measures extraction coverage instead of demand.
    """
    col = tr.period_col(period)
    eligible = features[
        ~features.job_function.isin(SKILL_EXCLUDED_FUNCTIONS)
        & features.has_any_skill.astype(bool)
    ]
    denominator = eligible.groupby(col).size()
    #: Period comes from ``features``, never from a period column on ``long``.
    #: The two frames both carry one, and a posting that lands in different
    #: months on each would put its skill in a numerator whose denominator is
    #: elsewhere — a share that can exceed 1. Mapping by ``job_id`` makes the
    #: numerator and denominator share one period assignment by construction.
    period_of = dict(zip(eligible.job_id, eligible[col]))
    rows_long = long[long.job_id.isin(period_of)].copy()
    rows_long["_period"] = rows_long.job_id.map(period_of)
    if skills is None:
        skills = rows_long.skill.value_counts().head(top).index.tolist()

    periods = sorted(denominator.index)
    unobserved = set(unobserved or [])
    rows = []
    for skill in skills:
        per = (rows_long[rows_long.skill == skill]
               .groupby("_period").job_id.nunique().reindex(periods).fillna(0))
        for p in periods:
            n = int(denominator.loc[p])
            k = int(per.loc[p])
            rows.append({
                "key": skill, "period": p, "numerator": k, "denominator": n,
                "share": (k / n) if n else float("nan"),
                "is_observed": p not in unobserved and n > 0,
            })
    return pd.DataFrame(rows, columns=SERIES_COLUMNS)


def observed(series: pd.DataFrame, key: str) -> pd.DataFrame:
    """One key's observed rows, in period order — the modelling input."""
    block = series[(series.key == key) & series.is_observed]
    return block.sort_values("period").reset_index(drop=True)


# ---------------------------------------------------------------------------
# The gate: is there anything here to forecast?
# ---------------------------------------------------------------------------


def homogeneity_test(numerator, denominator) -> dict:
    """Test a constant share against a moving one, and split the variance.

    The question every monthly share series has to answer before it is
    modelled: **is this moving at all, or is it one proportion observed
    through small samples?** Google's panel months hold 26 to 73 postings, so
    a share of 0.30 carries a binomial standard error near 0.05 — a third of
    the range the series actually covers.

    The statistic is Pearson's chi-square for homogeneity of proportions.
    ``overdispersion`` is chi-square per degree of freedom: 1.0 means every
    wiggle is sampling noise, and ``signal_share`` = 1 − 1/overdispersion is
    the fraction of the observed variance left once binomial noise is removed.

    The p-value uses the Wilson-Hilferty cube-root normal approximation to the
    chi-square, so this module keeps its no-scipy promise. Measured against the
    exact tail in `validate_forecast.py` §3: worst p-value error 1.9e-3 over
    the df range this gate can produce, and a critical value at ``ALPHA``
    within 0.15% of exact. The error runs slightly anti-conservative — it calls
    a series signal-bearing marginally more readily than the exact tail — which
    is why the gate refuses on cell size independently, and never on this test
    alone.
    """
    k = np.asarray(numerator, dtype=float)
    n = np.asarray(denominator, dtype=float)
    if len(k) < 2 or n.sum() <= 0:
        return {"chi2": float("nan"), "df": 0, "p_value": float("nan"),
                "overdispersion": float("nan"), "signal_share": float("nan")}
    p = k.sum() / n.sum()
    if p <= 0 or p >= 1:
        return {"chi2": 0.0, "df": len(k) - 1, "p_value": 1.0,
                "overdispersion": 0.0, "signal_share": 0.0}
    chi2 = float((((k - n * p) ** 2) / (n * p * (1 - p))).sum())
    dfree = len(k) - 1
    z = ((chi2 / dfree) ** (1 / 3) - (1 - 2 / (9 * dfree))) / math.sqrt(2 / (9 * dfree))
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    overdispersion = chi2 / dfree
    return {
        "chi2": chi2, "df": dfree, "p_value": p_value,
        "overdispersion": overdispersion,
        "signal_share": max(0.0, 1.0 - 1.0 / overdispersion) if overdispersion > 0 else 0.0,
    }


def forecastability_table(series: pd.DataFrame,
                          min_cell: int = MIN_CELL_MONTH,
                          min_observations: int = MIN_OBSERVATIONS,
                          alpha: float = ALPHA) -> pd.DataFrame:
    """The gate. One row per key, run before any model is fitted.

    Task 06's comparability gate decided whether two companies could be
    compared at all; this is its Task 07 twin, and it answers a different
    question — whether a series carries enough thickness and enough signal to
    be worth a model. Three verdicts:

    ``too_thin``
        some observed period has fewer than ``min_cell`` in the numerator, or
        there are fewer than ``min_observations`` periods. Whatever the series
        appears to do, most of it is the small cell.
    ``noise_only``
        thick enough, but homogeneity is not rejected: a constant share
        explains the series as well as anything. The honest forecast is the
        mean, and it is not a trend.
    ``forecastable``
        thick enough and moving by more than sampling noise. Note what this
        does **not** say: that a model can predict the movement. That is
        `select_model`'s question, and in this data the two answers differ.
    """
    rows = []
    for key in sorted(series.key.unique()):
        block = observed(series, key)
        n_obs = len(block)
        min_num = int(block.numerator.min()) if n_obs else 0
        test = homogeneity_test(block.numerator, block.denominator) if n_obs >= 2 else \
            {"chi2": float("nan"), "df": 0, "p_value": float("nan"),
             "overdispersion": float("nan"), "signal_share": float("nan")}

        if n_obs < min_observations or min_num < min_cell:
            verdict = "too_thin"
            reason = (f"{n_obs} observed periods (need {min_observations}), "
                      f"smallest cell {min_num} (need {min_cell})")
        elif not (test["p_value"] < alpha):
            verdict = "noise_only"
            reason = (f"constant share not rejected (p={test['p_value']:.3f}); "
                      "a mean is the honest forecast")
        else:
            verdict = "forecastable"
            reason = (f"moves by more than sampling noise "
                      f"(p={test['p_value']:.4f}, "
                      f"{test['signal_share']*100:.0f}% of variance is signal)")

        rows.append({
            "key": key, "n_observed": n_obs, "min_cell": min_num,
            "median_denominator": float(block.denominator.median()) if n_obs else float("nan"),
            "mean_share": float(block.share.mean()) if n_obs else float("nan"),
            "cv": float(block.share.std(ddof=1) / block.share.mean())
            if n_obs > 1 and block.share.mean() else float("nan"),
            "chi2": round(test["chi2"], 3), "df": test["df"],
            "homogeneity_p": test["p_value"],
            "overdispersion": round(test["overdispersion"], 3)
            if test["overdispersion"] == test["overdispersion"] else float("nan"),
            "signal_share": round(test["signal_share"], 4)
            if test["signal_share"] == test["signal_share"] else float("nan"),
            "verdict": verdict, "reason": reason,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Models. Signature is (y, t, t_target) -> float, all on the log scale.
# ---------------------------------------------------------------------------


def m_naive(y: np.ndarray, t: np.ndarray, t_target: float) -> float:
    """Last observed value. The benchmark every other model has to beat."""
    return float(y[-1])


def m_mean(y: np.ndarray, t: np.ndarray, t_target: float) -> float:
    """Historical mean — the right answer when the series is noise."""
    return float(np.mean(y))


def m_drift(y: np.ndarray, t: np.ndarray, t_target: float) -> float:
    """Random walk with drift through the first and last observations."""
    span = t[-1] - t[0]
    slope = (y[-1] - y[0]) / span if span else 0.0
    return float(y[-1] + slope * (t_target - t[-1]))


def m_ses(y: np.ndarray, t: np.ndarray, t_target: float) -> float:
    """Simple exponential smoothing, alpha chosen on `SES_ALPHA_GRID` by SSE."""
    best_sse, best_alpha = math.inf, SES_ALPHA_GRID[0]
    for alpha in SES_ALPHA_GRID:
        level, sse = y[0], 0.0
        for value in y[1:]:
            sse += (value - level) ** 2
            level = alpha * value + (1 - alpha) * level
        if sse < best_sse:
            best_sse, best_alpha = sse, alpha
    level = y[0]
    for value in y[1:]:
        level = best_alpha * value + (1 - best_alpha) * level
    return float(level)


def m_holt_damped(y: np.ndarray, t: np.ndarray, t_target: float) -> float:
    """Damped local trend. Three parameters on eleven points, on purpose.

    Kept in the comparison because it is what a specialist reaching for
    ``statsmodels.ExponentialSmoothing`` would get by default, and the
    backtest is the place to find out what that costs.
    """
    best = (math.inf, (HOLT_ALPHA_GRID[0], HOLT_BETA_GRID[0], HOLT_PHI_GRID[0]))
    for alpha in HOLT_ALPHA_GRID:
        for beta in HOLT_BETA_GRID:
            for phi in HOLT_PHI_GRID:
                level, trend, sse = y[0], 0.0, 0.0
                for value in y[1:]:
                    fitted = level + phi * trend
                    sse += (value - fitted) ** 2
                    new_level = alpha * value + (1 - alpha) * fitted
                    trend = beta * (new_level - level) + (1 - beta) * phi * trend
                    level = new_level
                if sse < best[0]:
                    best = (sse, (alpha, beta, phi))
    alpha, beta, phi = best[1]
    level, trend = y[0], 0.0
    for value in y[1:]:
        fitted = level + phi * trend
        new_level = alpha * value + (1 - alpha) * fitted
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    steps = max(1, int(round(t_target - t[-1])))
    damping = sum(phi ** i for i in range(1, steps + 1))
    return float(level + damping * trend)


def m_loglinear(y: np.ndarray, t: np.ndarray, t_target: float) -> float:
    """OLS trend on the log share — Task 05's ``log_growth``, per company.

    Fitted on calendar time, so the February gap widens the January-March step
    instead of being absorbed into the slope.
    """
    design = np.vstack([np.ones_like(t), t]).T
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0] + coef[1] * t_target)


def m_ar1(y: np.ndarray, t: np.ndarray, t_target: float) -> float:
    """Mean reversion at the estimated lag-1 coefficient, clipped to ±0.95."""
    mean = float(np.mean(y))
    dev = y - mean
    denom = float((dev[:-1] ** 2).sum())
    phi = float((dev[:-1] * dev[1:]).sum() / denom) if denom else 0.0
    phi = max(-0.95, min(0.95, phi))
    steps = max(1, int(round(t_target - t[-1])))
    return float(mean + (phi ** steps) * dev[-1])


def seasonal_naive(*_args, **_kwargs):
    """Refuses. A month-of-year term is not identified in a single year.

    The data covers 2023-01 to 2023-12: **zero complete seasonal cycles**.
    Every month-of-year effect is perfectly collinear with whatever else
    happened in that month, and Task 05 §6 already showed what that produces
    here — February's "trough" is a collection gap and August's "peak" is one
    publisher's first month. A seasonal model would fit both and call them
    seasonality.
    """
    raise NotImplementedError(
        "seasonal_naive is not available: 2023 is one year, so there are zero "
        "complete seasonal cycles and a month-of-year term is not identified "
        "(Task 05 §6, docs/task-07-demand-forecasting-methods.md §3). Use a "
        "non-seasonal model, or collect a second year."
    )


#: The model registry. Order is the tie-break order in `select_model`, so
#: `naive` is first and a tie never unseats it.
MODELS: dict[str, callable] = {
    "naive": m_naive,
    "mean": m_mean,
    "drift": m_drift,
    "ses": m_ses,
    "holt_damped": m_holt_damped,
    "loglinear": m_loglinear,
    "ar1": m_ar1,
}

BENCHMARK = "naive"


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------


def rolling_origin_backtest(series: pd.DataFrame,
                            keys: list[str] | None = None,
                            models: dict | None = None,
                            horizons=(1, 2, 3),
                            min_train: int = MIN_TRAIN,
                            period: str = "month") -> pd.DataFrame:
    """Expanding-window rolling-origin evaluation on the log share.

    The only honest accuracy statement available. A model fitted on all eleven
    points and scored on those same eleven points is measuring how many
    parameters it has, and with a three-parameter model on eleven points that
    number is flattering and meaningless.

    Every fit here sees only data before its target. The cost is severe and is
    reported rather than hidden: ``min_train`` of five leaves **six** one-step
    origins per key, and four at horizon three.
    """
    models = MODELS if models is None else models
    keys = sorted(series.key.unique()) if keys is None else keys
    rows = []
    for key in keys:
        block = observed(series, key)
        if len(block) <= min_train:
            continue
        y = _safe_log(block.share.to_numpy(), block.denominator.to_numpy())
        t = period_ordinal(block.period, period)
        periods = block.period.tolist()
        for horizon in horizons:
            for i in range(min_train, len(y) - horizon + 1):
                target = i + horizon - 1
                for name, fn in models.items():
                    try:
                        pred = float(fn(y[:i], t[:i], t[target]))
                    except Exception:      # a model that cannot fit scores nothing
                        pred = float("nan")
                    rows.append({
                        "key": key, "model": name, "horizon": horizon,
                        "origin": periods[i - 1], "target": periods[target],
                        "n_train": i, "actual": float(y[target]), "predicted": pred,
                        "error": float(y[target] - pred),
                    })
    return pd.DataFrame(rows)


def accuracy_table(backtest: pd.DataFrame) -> pd.DataFrame:
    """RMSE, MAE and the ratio to naive, per model and horizon.

    Errors are on the log share, so an RMSE of 0.53 means a typical one-step
    forecast is out by a factor of ``exp(0.53)`` — about 1.7× — in either
    direction. That translation is the reason the report quotes the factor and
    not the RMSE.
    """
    if backtest.empty:
        return pd.DataFrame()
    grouped = backtest.groupby(["horizon", "model"], as_index=False).agg(
        n=("error", "size"),
        rmse=("error", lambda e: float(np.sqrt(np.mean(np.square(e))))),
        mae=("error", lambda e: float(np.mean(np.abs(e)))),
        bias=("error", "mean"),
    )
    base = (grouped[grouped.model == BENCHMARK]
            .set_index("horizon")[["rmse", "mae"]]
            .rename(columns={"rmse": "naive_rmse", "mae": "naive_mae"}))
    out = grouped.merge(base, left_on="horizon", right_index=True, how="left")
    out["rmse_ratio_to_naive"] = out.rmse / out.naive_rmse
    out["mase"] = out.mae / out.naive_mae
    out["error_factor"] = np.exp(out.rmse)
    return out.drop(columns=["naive_rmse", "naive_mae"]).sort_values(
        ["horizon", "rmse"]).reset_index(drop=True)


def diebold_mariano(errors_a, errors_b, power: int = 2) -> dict:
    """Test whether two forecast error series differ, on paired losses.

    ``errors_a`` is the challenger, ``errors_b`` the benchmark, and the two
    must be aligned on the same targets. A negative ``dm_stat`` means the
    challenger has the smaller loss.

    Small-sample caveat, stated because it is load-bearing: the DM statistic
    is asymptotically normal and thirty-six paired errors is not asymptotic.
    It is used here in the conservative direction only — to **refuse** a model
    that has not clearly beaten naive — so an understated rejection rate costs
    nothing that matters. The p-value comes from ``math.erfc``.
    """
    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 3:
        return {"n": n, "dm_stat": float("nan"), "p_value": float("nan"),
                "mean_loss_diff": float("nan")}
    d = np.abs(a) ** power - np.abs(b) ** power
    se = d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return {"n": n, "dm_stat": 0.0, "p_value": 1.0, "mean_loss_diff": 0.0}
    stat = float(d.mean() / se)
    return {"n": n, "dm_stat": stat,
            "p_value": float(math.erfc(abs(stat) / math.sqrt(2))),
            "mean_loss_diff": float(d.mean())}


def model_contest(backtest: pd.DataFrame, horizon: int = 1,
                  benchmark: str = BENCHMARK) -> pd.DataFrame:
    """Every model against the benchmark on the same targets, with a DM test."""
    block = backtest[backtest.horizon == horizon]
    if block.empty:
        return pd.DataFrame()
    base = block[block.model == benchmark].set_index(["key", "target"]).error
    rows = []
    for name in block.model.unique():
        if name == benchmark:
            continue
        challenger = block[block.model == name].set_index(["key", "target"]).error
        aligned = base.reindex(challenger.index)
        test = diebold_mariano(challenger.to_numpy(), aligned.to_numpy())
        rows.append({
            "model": name, "benchmark": benchmark, "horizon": horizon,
            "n_pairs": test["n"], "dm_stat": test["dm_stat"],
            "p_value": test["p_value"],
            "rmse": float(np.sqrt(np.mean(np.square(challenger.dropna())))),
            "benchmark_rmse": float(np.sqrt(np.mean(np.square(aligned.dropna())))),
            "beats_benchmark": bool(test["p_value"] < ALPHA and test["dm_stat"] < 0),
        })
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def select_model(backtest: pd.DataFrame, horizon: int = 1,
                 alpha: float = ALPHA,
                 benchmark: str = BENCHMARK) -> pd.DataFrame:
    """Choose a model per key — and choose ``naive`` unless beaten.

    **The rule this whole task turns on.** Picking the lowest-RMSE model per
    company is what the backtest invites and it is wrong here: six origins per
    company means the ranking is mostly noise, and a per-company winner chosen
    that way is a description of which model happened to fit six months.

    So a challenger is selected only when it beats the benchmark on a
    Diebold-Mariano test at ``alpha``. Otherwise the selection is ``naive``,
    ``selected_by`` is ``benchmark_not_beaten``, and the report says the
    forecast is persistence. In this data no challenger clears the bar for any
    company, which is the finding rather than a failure of the search.
    """
    block = backtest[backtest.horizon == horizon]
    rows = []
    for key in sorted(block.key.unique()):
        own = block[block.key == key]
        base = own[own.model == benchmark].set_index("target").error
        best_name, best_p, best_stat = benchmark, float("nan"), float("nan")
        best_rmse = float(np.sqrt(np.mean(np.square(base.dropna()))))
        for name in own.model.unique():
            if name == benchmark:
                continue
            challenger = own[own.model == name].set_index("target").error
            test = diebold_mariano(challenger.to_numpy(),
                                   base.reindex(challenger.index).to_numpy())
            rmse = float(np.sqrt(np.mean(np.square(challenger.dropna()))))
            if test["p_value"] < alpha and test["dm_stat"] < 0 and rmse < best_rmse:
                best_name, best_rmse = name, rmse
                best_p, best_stat = test["p_value"], test["dm_stat"]
        lowest = (own.groupby("model").error
                  .apply(lambda e: float(np.sqrt(np.mean(np.square(e.dropna()))))))
        rows.append({
            "key": key, "horizon": horizon, "selected": best_name,
            "selected_by": "beats_benchmark" if best_name != benchmark
            else "benchmark_not_beaten",
            "selected_rmse": best_rmse,
            "dm_p_value": best_p, "dm_stat": best_stat,
            "lowest_rmse_model": str(lowest.idxmin()),
            "lowest_rmse": float(lowest.min()),
            "would_have_picked_differently": bool(lowest.idxmin() != best_name),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Intervals and the published forecast
# ---------------------------------------------------------------------------


def empirical_interval(backtest: pd.DataFrame, model: str, horizon: int,
                       level: float = INTERVAL_LEVEL,
                       keys: list[str] | None = None) -> dict:
    """Prediction interval from the model's own backtest residuals.

    Not a model-based standard error. A three-parameter model fitted on five
    to ten points reports an in-sample sigma that is optimistic by roughly the
    amount it overfits, and the whole point of this interval is to be honest
    about a series that nothing predicts.

    ``keys`` restricts which series contribute residuals, and it matters more
    than it looks. Pooled across all six companies the one-step naive RMSE is
    0.534; on the two companies thick enough to publish it is 0.372 and 0.229.
    Building Google's interval from Snowflake's errors — Snowflake having
    months of one posting — would widen it by half for a reason that has
    nothing to do with Google. So the caller passes the keys it is publishing,
    and ``n_residuals`` records what the endpoints rest on.

    Order statistics, not quantiles
    -------------------------------
    The endpoints are **raw order statistics of the residuals**, with indices
    chosen so that coverage is at least ``level``. The obvious implementation,
    ``np.quantile(residuals, 0.1)``, interpolates between neighbouring
    residuals, and on the twelve residuals available at one step that produced
    an interval whose measured coverage was 67% while it was labelled 80%.
    Interpolation invents an endpoint the sample does not contain, and at this
    n the invention is most of the interval.

    The distribution-free construction instead takes the ``j``-th smallest and
    ``j``-th largest residual, where ``j = floor((1-level)/2 * (n+1))``, giving
    exact coverage ``(n + 1 - 2j) / (n + 1)`` for an exchangeable future
    error. It is conservative by construction, and ``achieved_level`` reports
    what was actually bought.

    **And it refuses.** When ``j`` computes to zero the required endpoint lies
    outside the sample: there are not enough residuals to bound an interval at
    this level by any distribution-free means, so this returns ``sufficient =
    False`` and no numbers. At three-step horizon, on eight residuals, that is
    what happens — which is the honest reason the three-month forecast is not
    published, in place of a threshold chosen by hand.
    """
    block = backtest[(backtest.model == model) & (backtest.horizon == horizon)]
    if keys is not None:
        block = block[block.key.isin(keys)]
    residuals = np.sort(block.error.dropna().to_numpy())
    n = len(residuals)
    base = {"n_residuals": int(n), "level": level,
            "residual_sd": float(residuals.std(ddof=1)) if n > 1 else float("nan")}
    j = int(math.floor((1 - level) / 2 * (n + 1)))
    if n < 3 or j < 1:
        return {**base, "lo": float("nan"), "hi": float("nan"),
                "achieved_level": float("nan"), "sufficient": False,
                "reason": (f"{n} residuals cannot bound a {level:.0%} interval "
                           f"without interpolating beyond the sample "
                           f"(need at least {int(math.ceil(2/(1-level))) - 1})")}
    return {**base, "lo": float(residuals[j - 1]), "hi": float(residuals[n - j]),
            "achieved_level": float((n + 1 - 2 * j) / (n + 1)),
            "sufficient": True, "reason": ""}


def interval_coverage(backtest: pd.DataFrame, model: str, horizon: int,
                      level: float = INTERVAL_LEVEL,
                      keys: list[str] | None = None) -> dict:
    """Does the empirical interval actually cover at its nominal rate?

    In-sample by construction — the residuals that build the interval are the
    residuals it is scored on — so this is a consistency check, not
    validation. It is here because it caught a real defect: the first version
    of `empirical_interval` used ``np.quantile``, and this function measured
    67% coverage on an interval labelled 80%.

    With order-statistic endpoints ``covered`` is ``(n - 2j + 2) / n``, which
    is 1.0 whenever ``j`` is 1 and the endpoints are the sample minimum and
    maximum. That is not the interval being too wide; it is what in-sample
    means here. The invariant worth checking, and the one the tests pin, is
    ``covered >= achieved_level``: the out-of-sample guarantee may never
    exceed what the residuals themselves show.
    """
    band = empirical_interval(backtest, model, horizon, level, keys)
    block = backtest[(backtest.model == model) & (backtest.horizon == horizon)]
    if keys is not None:
        block = block[block.key.isin(keys)]
    residuals = block.error.dropna().to_numpy()
    if not len(residuals) or not band.get("sufficient"):
        return {"covered": float("nan"), "nominal": level,
                "achieved": float("nan"), "n": int(len(residuals))}
    inside = ((residuals >= band["lo"]) & (residuals <= band["hi"])).mean()
    return {"covered": float(inside), "nominal": level,
            "achieved": band["achieved_level"], "n": int(len(residuals))}


def horizon_table(backtest: pd.DataFrame, model: str = BENCHMARK,
                  level: float = INTERVAL_LEVEL,
                  keys: list[str] | None = None) -> pd.DataFrame:
    """Error and interval width by horizon — where the forecast stops saying anything.

    ``interval_factor`` is the multiplicative width of the prediction interval
    on the natural share scale. When it reaches 3, the interval spans a share
    three times its own lower end, and it has stopped excluding any outcome a
    reader would have considered plausible without it.
    """
    rows = []
    for horizon in sorted(backtest.horizon.unique()):
        block = backtest[(backtest.model == model) & (backtest.horizon == horizon)]
        if keys is not None:
            block = block[block.key.isin(keys)]
        errors = block.error.dropna().to_numpy()
        if not len(errors):
            continue
        band = empirical_interval(backtest, model, horizon, level, keys)
        width = band["hi"] - band["lo"]
        rows.append({
            "horizon": int(horizon), "model": model, "n_origins": int(len(errors)),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mae": float(np.mean(np.abs(errors))),
            "interval_lo": band["lo"], "interval_hi": band["hi"],
            "interval_width_log": width,
            "interval_factor": float(np.exp(width)) if np.isfinite(width) else float("nan"),
            "n_residuals": band["n_residuals"],
            "achieved_level": band.get("achieved_level", float("nan")),
            "interval_sufficient": bool(band.get("sufficient", False)),
            "insufficient_reason": band.get("reason", ""),
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class HorizonVerdict:
    """How far the data supports forecasting, and why it stops there."""

    max_useful_horizon: int
    reason: str
    detail: dict


def horizon_verdict(horizons: pd.DataFrame,
                    max_factor: float = 3.0) -> HorizonVerdict:
    """The longest horizon that has an interval, and whose interval says something.

    Two ways to fail, and they are different failures worth distinguishing in
    the reason string:

    * **no interval.** Too few backtest residuals to bound the level without
      interpolating outside the sample. Derived, not chosen.
    * **an interval that excludes nothing.** ``max_factor`` of 3 *is* a
      declared threshold: an interval spanning a threefold range of shares is
      compatible with almost any story a reader arrives with, and publishing a
      point beside it invites the point to be read instead.

    A horizon must also have every shorter horizon usable before it counts. A
    forecast that fails at two months and succeeds at three has found an
    accident of eight residuals, not a longer reach.
    """
    if horizons.empty:
        return HorizonVerdict(0, "no backtest available", {})
    detail = {int(r.horizon): {
        "interval_factor": round(float(r.interval_factor), 3)
        if np.isfinite(r.interval_factor) else None,
        "n_residuals": int(r.n_residuals),
        "sufficient": bool(r.interval_sufficient),
    } for r in horizons.itertuples()}

    best = 0
    blocker = ""
    for row in horizons.sort_values("horizon").itertuples():
        if not row.interval_sufficient:
            blocker = (f"h={int(row.horizon)} has no interval at all: "
                       f"{row.insufficient_reason}")
            break
        if not (row.interval_factor <= max_factor):
            blocker = (f"h={int(row.horizon)} has an interval spanning "
                       f"{row.interval_factor:.2f}x, past the {max_factor:g}x "
                       "limit, so it excludes nothing")
            break
        best = int(row.horizon)

    if best == 0:
        return HorizonVerdict(0, blocker or "no usable horizon", detail)
    factor = float(horizons.loc[horizons.horizon == best, "interval_factor"].iloc[0])
    reason = (f"h={best} is the longest usable horizon (interval spans "
              f"{factor:.2f}x on {int(horizons.loc[horizons.horizon == best, 'n_residuals'].iloc[0])} "
              f"residuals)")
    if blocker:
        reason += f"; {blocker}"
    return HorizonVerdict(best, reason, detail)


def forecast_table(series: pd.DataFrame, backtest: pd.DataFrame,
                   selection: pd.DataFrame, gate: pd.DataFrame | None = None,
                   verdict: HorizonVerdict | None = None,
                   horizons=(1, 2, 3), level: float = INTERVAL_LEVEL,
                   period: str = "month") -> pd.DataFrame:
    """The published forecast: point, interval, and what it rests on.

    Three things travel in every row because none of them can be left to a
    caption:

    ``model`` / ``selected_by``
        a persistence forecast is a different object from a fitted model's
        prediction and must not be readable as one.
    ``supported``
        False when ``verdict`` says no horizon carries an interval narrow
        enough to exclude anything. **In this dataset that is every row.** The
        numbers are still computed and still published — a refusal a reader
        cannot audit is just an assertion — but the column says the table does
        not support them, so a row lifted into a slide carries its own
        disclaimer.
    ``out_of_window``
        True everywhere. Collection stops at 2023-12-31, so no target this
        function produces can ever be checked against the data that made it.

    ``gate`` restricts the output to keys that passed `forecastability_table`,
    and its keys are what the interval is pooled over.
    """
    chosen = selection.set_index("key")
    if gate is not None:
        allowed = set(gate.loc[gate.verdict == "forecastable", "key"])
    else:
        allowed = set(series.key.unique())
    pool_keys = sorted(allowed) if gate is not None else None
    max_useful = verdict.max_useful_horizon if verdict is not None else max(horizons)

    rows = []
    for key in sorted(series.key.unique()):
        if key not in allowed:
            continue
        block = observed(series, key)
        if len(block) <= MIN_TRAIN or key not in chosen.index:
            continue
        y = _safe_log(block.share.to_numpy(), block.denominator.to_numpy())
        t = period_ordinal(block.period, period)
        model_name = str(chosen.loc[key, "selected"])
        fn = MODELS[model_name]
        targets = next_periods(block.period.iloc[-1], max(horizons), period)
        for horizon in horizons:
            point = float(fn(y, t, t[-1] + horizon))
            band = empirical_interval(backtest, model_name, horizon, level, pool_keys)
            has_band = bool(band.get("sufficient"))
            width = band["hi"] - band["lo"] if has_band else float("nan")
            rows.append({
                "key": key, "target_period": targets[horizon - 1],
                "horizon": int(horizon), "model": model_name,
                "selected_by": str(chosen.loc[key, "selected_by"]),
                "last_observed_share": float(block.share.iloc[-1]),
                "point_share": float(np.exp(point)),
                "lo_share": float(np.exp(point + band["lo"])) if has_band else float("nan"),
                "hi_share": float(np.exp(point + band["hi"])) if has_band else float("nan"),
                "interval_level": level,
                "achieved_level": band.get("achieved_level", float("nan")),
                "interval_factor": float(np.exp(width)) if has_band else float("nan"),
                "n_residuals": band["n_residuals"],
                "n_train": int(len(y)),
                "supported": bool(has_band and horizon <= max_useful),
                "out_of_window": True,
            })
    return pd.DataFrame(rows)


def compositional_normalise(forecasts: pd.DataFrame, expected_keys: list[str],
                            value_col: str = "point_share") -> pd.DataFrame:
    """Rescale each period's company shares so they sum to one.

    The shares are a composition — the companies divide one panel pool — and
    models fitted independently do not respect that. Left alone the points sum
    to anything, and a reader adding them up finds the forecast contradicts
    its own definition. Renormalising is the minimum fix, and it is not free:
    it moves every company's point by whatever the others did, so
    ``renormalisation_factor`` is published beside it. The interval is **not**
    rescaled — it comes from the marginal backtest, and rescaling it would
    imply a joint distribution this module never estimated.

    Why ``expected_keys`` is mandatory
    ----------------------------------
    **Renormalising a partial composition is not a correction, it is a
    fabrication**, and this argument exists because the unguarded version
    produced one. In this dataset the forecastability gate passes two of the
    six companies. Those two hold about 40% of the panel between them; scaling
    them to sum to 1 reported Google at 58% — a number with no referent, since
    the missing 60% of the pool did not stop existing when it failed the gate.

    So the caller must name the full composition, and a frame that does not
    carry every one of those keys raises rather than silently rescaling. For
    Task 07 that means renormalisation **does not apply**: the published
    forecast covers Google and Meta, and its shares deliberately do not sum to
    one, because they are shares of a pool that includes four companies this
    task will not forecast.
    """
    present = set(forecasts.key.unique())
    missing = sorted(set(expected_keys) - present)
    extra = sorted(present - set(expected_keys))
    if missing or extra:
        raise ValueError(
            "compositional_normalise needs the complete composition: "
            f"missing {missing}, unexpected {extra}. Renormalising a subset "
            "rescales it to a total it does not have — the omitted members "
            "still hold their share of the pool. Forecast every member, or "
            "publish the un-normalised shares and say they do not sum to 1."
        )
    out = forecasts.copy()
    totals = out.groupby(["target_period", "horizon"])[value_col].transform("sum")
    out["raw_" + value_col] = out[value_col]
    out["renormalisation_factor"] = np.where(totals > 0, 1.0 / totals, float("nan"))
    out[value_col] = out[value_col] * out["renormalisation_factor"]
    return out


# ---------------------------------------------------------------------------
# Standing guards, inherited
# ---------------------------------------------------------------------------


def forbidden_columns(table: pd.DataFrame) -> list[str]:
    """Task 06's banned column families, unchanged."""
    return cmp.forbidden_columns(table)


def personal_data_columns_present(table: pd.DataFrame) -> list[str]:
    """The standing Task 01 privacy check, re-run on every Task 07 table."""
    return cmp.personal_data_columns_present(table)
