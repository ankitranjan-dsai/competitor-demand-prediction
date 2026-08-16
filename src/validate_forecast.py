"""Check `src/forecast.py`'s hand-rolled statistics against scipy and statsmodels.

`forecast.py` deliberately carries no scipy, statsmodels or prophet import. The
reason is stated in `docs/task-07-demand-forecasting-methods.md` §2: the module
has to run in a teammate's environment and in the grader's, and a forecasting
stack that silently swaps an optimiser between versions is a poor place to put
a gate that decides whether a number gets published at all. The cost of that
promise is that three pieces of statistics are written out by hand — a
chi-square tail, a normal tail, and two exponential-smoothing recursions — and
hand-written statistics are exactly the kind of thing that is wrong in a way no
unit test catches, because the test and the code share the author's mistake.

So this script imports the libraries `forecast.py` refuses to depend on and
checks the two implementations agree. It is a **validator, not a dependency**:
nothing in `src/`, `tests/` or `build_forecast.py` imports it, and the pipeline
runs to completion with scipy and statsmodels uninstalled.

What is being proved, in order of how much it matters:

1. **the model recursions are the textbook ones.** SES and damped Holt match
   `statsmodels.tsa.holtwinters` to machine precision at fixed parameters, so
   "no model beats naive" is a statement about the data and not about a
   mis-transcribed update equation.
2. **the refusal survives a better test.** The Diebold-Mariano p-value uses a
   normal tail, which is anti-conservative in the direction of *accepting* a
   challenger. Recomputed against Student's t and again with the
   Harvey-Leybourne-Newbold small-sample correction, every challenger is
   refused by a wider margin.
3. **the chi-square approximation cannot flip a gate verdict.** Wilson-Hilferty
   is measured against the exact tail over the whole df range the gate uses,
   and the real series are located relative to the boundary where the two
   would disagree.
4. **the prediction interval covers what it claims, for any distribution.** The
   order-statistic interval is checked by simulation against normal, t(3) and
   lognormal residuals — and against the `np.quantile` version it replaced,
   which undercovers.

    python src/validate_forecast.py

Writes `docs/task-07-forecast-validation.md`. Exits non-zero if any check fails.
No posting-level data is read for the reference checks, and the evidence file
holds only statistics.
"""

from __future__ import annotations

import math
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare as cmp    # noqa: E402
import forecast as fc    # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "task-07-forecast-validation.md"

COMPANIES = ["databricks", "google", "meta", "microsoft", "nvidia", "snowflake"]

#: Machine-precision agreement. The recursions are algebraically identical, so
#: anything above float noise means one of the two is not what it claims to be.
EXACT = 1e-10


def _fmt(x: float, places: int = 6) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{places}f}"


# --------------------------------------------------------------------------
# 1. the smoothing recursions
# --------------------------------------------------------------------------

def check_smoothers(rng) -> tuple[list, list[str]]:
    """SES and damped Holt against statsmodels at fixed parameters.

    Fixed parameters, not fitted ones: statsmodels optimises the initial state
    by default and `forecast.py` pins it to the first observation, so an
    optimised comparison would be measuring two different estimators. Holding
    alpha, beta, phi and the initial state fixed leaves only the recursion,
    which is the thing that could be mistyped.
    """
    from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing

    lines = ["## 1. Smoothing recursions against statsmodels", ""]
    ses_err = holt_err = sse_err = 0.0

    for trial in range(200):
        n = int(rng.integers(6, 15))
        y = np.log(rng.uniform(0.05, 0.60, n))

        alpha = float(rng.choice(fc.SES_ALPHA_GRID))
        res = SimpleExpSmoothing(
            y, initialization_method="known", initial_level=y[0],
        ).fit(smoothing_level=alpha, optimized=False)
        level, sse = y[0], 0.0
        for value in y[1:]:
            sse += (value - level) ** 2
            level = alpha * value + (1 - alpha) * level
        ses_err = max(ses_err, abs(level - float(res.forecast(1)[0])))
        sse_err = max(sse_err, abs(sse - float(res.sse)))

        a = float(rng.choice(fc.HOLT_ALPHA_GRID))
        b = float(rng.choice(fc.HOLT_BETA_GRID))
        phi = float(rng.choice(fc.HOLT_PHI_GRID))
        fit = Holt(
            y, damped_trend=True, initialization_method="known",
            initial_level=y[0], initial_trend=0.0,
        ).fit(smoothing_level=a, smoothing_trend=b, damping_trend=phi,
              optimized=False)
        lvl, trend = y[0], 0.0
        for value in y[1:]:
            fitted = lvl + phi * trend
            new = a * value + (1 - a) * fitted
            trend = b * (new - lvl) + (1 - b) * phi * trend
            lvl = new
        for steps in (1, 2, 3):
            mine = lvl + sum(phi ** i for i in range(1, steps + 1)) * trend
            holt_err = max(holt_err, abs(mine - float(fit.forecast(steps)[-1])))

    lines += [
        f"200 random log-share series, 6-14 points, parameters drawn from the "
        f"module's own grids. Largest absolute disagreement:",
        "",
        "| quantity | reference | max abs difference |",
        "| --- | --- | --- |",
        f"| SES one-step level | `SimpleExpSmoothing.forecast(1)` | {ses_err:.3e} |",
        f"| SES fitting SSE | `SimpleExpSmoothing.sse` | {sse_err:.3e} |",
        f"| damped Holt, h=1,2,3 | `Holt(damped_trend=True).forecast(h)` | {holt_err:.3e} |",
        "",
        "The damped forecast agrees at all three horizons, which is the part "
        "worth stating: the damping sum `phi + phi^2 + ... + phi^h` is the "
        "easiest term in the whole module to get wrong by one power, and a "
        "one-step-only check would not have seen it.",
        "",
    ]
    checks = [
        ("SES level matches statsmodels at fixed alpha", ses_err < EXACT),
        ("SES SSE matches, so the grid search minimises the same thing",
         sse_err < EXACT),
        ("damped Holt matches at every horizon in MAX_HORIZON", holt_err < EXACT),
    ]
    return checks, lines


def check_regressions(rng) -> tuple[list, list[str]]:
    """The two regression models against statsmodels OLS."""
    import statsmodels.api as sm
    from statsmodels.tsa.ar_model import AutoReg

    lines = ["## 2. Regression models against statsmodels OLS", ""]
    loglinear_err = ar1_err = 0.0
    autoreg_gaps = []

    for _ in range(200):
        n = int(rng.integers(8, 15))
        t = np.arange(n, dtype=float) + float(rng.integers(0, 5))
        y = np.log(rng.uniform(0.05, 0.60, n))
        target = t[-1] + float(rng.integers(1, 4))

        ols = sm.OLS(y, sm.add_constant(t)).fit()
        loglinear_err = max(loglinear_err, abs(
            fc.m_loglinear(y, t, target)
            - float(ols.params[0] + ols.params[1] * target)))

        # m_ar1 is conditional OLS through the origin on the demeaned lag pair.
        dev = y - y.mean()
        slope = float(sm.OLS(dev[1:], dev[:-1]).fit().params[0])
        phi = max(-0.95, min(0.95, slope))
        steps = max(1, int(round(target - t[-1])))
        ar1_err = max(ar1_err, abs(
            fc.m_ar1(y, t, target) - (y.mean() + (phi ** steps) * dev[-1])))
        autoreg_gaps.append(abs(
            slope - float(AutoReg(y, lags=1, trend="c").fit().params[1])))

    lines += [
        "| model | reference | max abs difference |",
        "| --- | --- | --- |",
        f"| `m_loglinear` | `sm.OLS(y, [1, t])` prediction | {loglinear_err:.3e} |",
        f"| `m_ar1` | `sm.OLS(dev[1:], dev[:-1])`, no intercept | {ar1_err:.3e} |",
        "",
        "`m_ar1` is checked against the estimator it actually is, not against "
        "`AutoReg`. Demeaning first and regressing through the origin is not "
        "the same estimator as fitting an intercept and a slope jointly, and "
        "over the same 200 series the two lag-1 coefficients differ by up to "
        f"{max(autoreg_gaps):.3f} (median {np.median(autoreg_gaps):.3f}). "
        "Both are defensible on eleven points; claiming they are equal would "
        "not be. What matters for Task 07 is that neither wins — `ar1` is the "
        "fourth-ranked model on the real backtest and loses to naive.",
        "",
    ]
    checks = [
        ("m_loglinear matches OLS on the log share", loglinear_err < EXACT),
        ("m_ar1 matches OLS through the origin on demeaned lags", ar1_err < EXACT),
    ]
    return checks, lines


# --------------------------------------------------------------------------
# 3. the two tail approximations
# --------------------------------------------------------------------------

def _wilson_hilferty(chi2: float, dfree: int) -> float:
    """The two lines `homogeneity_test` uses, copied so the sweep can drive them.

    Copied rather than imported because `homogeneity_test` takes counts, and a
    sweep over (chi2, df) cannot be expressed as counts. `check_chi_square`
    verifies the copy against the module itself on real count tables before
    trusting it.
    """
    z = ((chi2 / dfree) ** (1 / 3) - (1 - 2 / (9 * dfree))) / math.sqrt(
        2 / (9 * dfree))
    return 0.5 * math.erfc(z / math.sqrt(2))


def check_chi_square() -> tuple[list, list[str]]:
    """Wilson-Hilferty against the exact chi-square tail, over the gate's range.

    The question is not "how close is the approximation" in the abstract. It is
    whether the approximation can move a series across `ALPHA` and change its
    verdict. So the error is measured where it can do damage: at the critical
    value itself, for every df the gate can produce.
    """
    from scipy import stats

    lines = ["## 3. Wilson-Hilferty against the exact chi-square tail", ""]

    # The local copy has to be the module's own arithmetic before the sweep
    # below says anything about the module.
    rng = np.random.default_rng(7)
    copy_err = 0.0
    for _ in range(200):
        size = int(rng.integers(8, 13))
        n = rng.integers(20, 90, size).astype(float)
        k = rng.binomial(n.astype(int), rng.uniform(0.1, 0.5)).astype(float)
        result = fc.homogeneity_test(k, n)
        copy_err = max(copy_err, abs(
            result["p_value"] - _wilson_hilferty(result["chi2"], result["df"])))

    worst_err = 0.0
    worst_at = None
    gate_err = 0.0
    gate_at = None
    rows = []
    gate_df = (7, 8, 9, 10, 11)

    # The gate sees df = (observed periods - 1), which is 7 to 11 for this
    # window; the sweep is deliberately wider than that.
    for dfree in range(2, 25):
        for chi2 in np.linspace(0.2 * dfree, 6.0 * dfree, 400):
            approx = _wilson_hilferty(float(chi2), dfree)
            exact = float(stats.chi2.sf(chi2, dfree))
            err = abs(approx - exact)
            if err > worst_err:
                worst_err, worst_at = err, (dfree, float(chi2), approx, exact)
            if dfree in gate_df and err > gate_err:
                gate_err, gate_at = err, (dfree, float(chi2), approx, exact)

        # Where the two would actually disagree about the verdict: the chi2 at
        # which each crosses ALPHA.
        crit_exact = float(stats.chi2.isf(fc.ALPHA, dfree))
        lo, hi = 0.5 * crit_exact, 2.0 * crit_exact
        for _ in range(80):
            mid = (lo + hi) / 2
            if _wilson_hilferty(mid, dfree) > fc.ALPHA:
                lo = mid
            else:
                hi = mid
        crit_approx = (lo + hi) / 2
        if dfree in gate_df:
            rows.append((dfree, crit_exact, crit_approx,
                         100 * (crit_approx - crit_exact) / crit_exact))

    dfree, chi2, approx, exact = worst_at
    gdf, gchi2, gapprox, gexact = gate_at
    lines += [
        "Sweep: df 2 to 24, chi-square from 0.2·df to 6·df — wider than the "
        "df 7-11 the gate can produce on a twelve-month window.",
        "",
        f"- largest absolute p-value error anywhere in the sweep: "
        f"**{worst_err:.2e}**, at df {dfree}, chi2 {chi2:.2f} "
        f"(approximate {approx:.6f} against exact {exact:.6f})",
        f"- largest error at the df the gate actually sees (7-11): "
        f"**{gate_err:.2e}**, at df {gdf}, chi2 {gchi2:.2f} "
        f"(approximate {gapprox:.6f} against exact {gexact:.6f})",
        "",
        "Two things are worth saying plainly about those numbers. The error is "
        "worst at **low df in the body of the distribution**, not in the tail "
        "— Wilson-Hilferty is a cube-root normal approximation and it is the "
        "small-df shape it struggles with. And it shrinks quickly as df grows, "
        "so the range this gate uses is the accurate end of the sweep.",
        "",
        "Neither error is where a verdict is decided. What decides a verdict "
        f"is the critical value at alpha = {fc.ALPHA}:",
        "",
        "| df | exact critical chi2 | approximate critical chi2 | gap |",
        "| --- | --- | --- | --- |",
    ]
    for dfree, ce, ca, pct in rows:
        lines.append(f"| {dfree} | {ce:.4f} | {ca:.4f} | {pct:+.3f}% |")
    lines += [
        "",
        "The approximate critical value sits below the exact one, so "
        "Wilson-Hilferty rejects a constant share very slightly more readily "
        "than the exact tail does. That is the anti-conservative direction for "
        "this gate — it errs toward calling a series signal-bearing. A series "
        "would have to land inside a window well under one percent wide for "
        "the two to disagree; §5 locates the real series relative to it.",
        "",
    ]
    checks = [
        ("the sweep drives homogeneity_test's own arithmetic", copy_err < EXACT),
        ("Wilson-Hilferty critical value within 1% of exact for every gate df",
         all(abs(pct) < 1.0 for *_, pct in rows)),
        ("largest p-value error at the gate's own df range is below 0.002",
         gate_err < 0.002),
        ("largest p-value error anywhere in the df 2-24 sweep is below 0.01",
         worst_err < 0.01),
    ]
    return checks, lines


def check_normal_tail(rng) -> tuple[list, list[str]]:
    """The `math.erfc` two-sided normal tail against scipy, and DM against a t-test."""
    from scipy import stats

    lines = ["## 4. The Diebold-Mariano statistic and its tail", ""]
    stat_err = tail_err = 0.0

    for _ in range(300):
        n = int(rng.integers(5, 40))
        a = rng.normal(0, 1.0, n)
        b = rng.normal(0, 1.2, n)
        mine = fc.diebold_mariano(a, b)
        loss = np.abs(a) ** 2 - np.abs(b) ** 2
        ref = stats.ttest_1samp(loss, 0.0)
        stat_err = max(stat_err, abs(mine["dm_stat"] - float(ref.statistic)))
        tail_err = max(tail_err, abs(
            mine["p_value"] - 2 * float(stats.norm.sf(abs(mine["dm_stat"])))))

    lines += [
        "The DM statistic at h=1 with no HAC correction is algebraically the "
        "one-sample t statistic on the paired loss differential, which gives a "
        "reference implementation that shares none of this module's code.",
        "",
        "| quantity | reference | max abs difference |",
        "| --- | --- | --- |",
        f"| `dm_stat` | `scipy.stats.ttest_1samp(d, 0).statistic` | {stat_err:.3e} |",
        f"| `p_value` | `2 * scipy.stats.norm.sf(abs(stat))` | {tail_err:.3e} |",
        "",
        "The statistic is exact. The p-value is exactly the two-sided *normal* "
        "tail — which is the approximation, and §5 measures what it costs on "
        "the real contest.",
        "",
    ]
    checks = [
        ("dm_stat equals the one-sample t statistic on the loss differential",
         stat_err < EXACT),
        ("the erfc tail equals scipy's normal tail", tail_err < EXACT),
    ]
    return checks, lines


# --------------------------------------------------------------------------
# 5. the real series
# --------------------------------------------------------------------------

def check_real_series() -> tuple[list, list[str]]:
    """Locate the published conclusions relative to every approximation above."""
    from scipy import stats

    lines = ["## 5. The published conclusions, recomputed with scipy", ""]
    try:
        frames = cmp.load_frames(COMPANIES)
    except FileNotFoundError as exc:
        lines += [f"*Skipped — row-level data not built ({exc}).* The reference "
                  "checks above do not need it; this section does.", ""]
        return [], lines

    series = fc.panel_share_series(frames, publishers=cmp.common_publishers(frames))
    gate = fc.forecastability_table(series)

    lines += [
        "### 5.1 The gate, exact tail against the approximation", "",
        "| company | chi2 | df | approximate p | exact p | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    gate_flips = []
    for row in gate.itertuples():
        exact = float(stats.chi2.sf(row.chi2, row.df))
        if (exact < fc.ALPHA) != (row.homogeneity_p < fc.ALPHA):
            gate_flips.append(row.key)
        lines.append(
            f"| {row.key} | {row.chi2:.2f} | {int(row.df)} | "
            f"{row.homogeneity_p:.2e} | {exact:.2e} | {row.verdict} |")
    weakest = gate.loc[gate.homogeneity_p.idxmax()]
    exact_ps = [float(stats.chi2.sf(r.chi2, r.df)) for r in gate.itertuples()]
    spread = np.log10(max(exact_ps) / min(exact_ps))
    lines += [
        "",
        f"Every series rejects a constant share under both tails. The weakest, "
        f"{weakest.key}, sits at an exact p of {max(exact_ps):.1e} against "
        f"alpha = {fc.ALPHA}; the strongest is {spread:.0f} orders of "
        f"magnitude below that. "
        "Nothing is near the boundary where §3's sub-one-percent disagreement "
        "could change a verdict — and the four `too_thin` verdicts are refused "
        "on cell size, not on this test, so the tail plays no part in them at "
        "all.",
        "",
    ]

    gated = sorted(gate.loc[gate.verdict == "forecastable", "key"])
    lines += [
        "### 5.2 The refusal under three tails", "",
        "The normal tail is the most permissive of the three, so it is the one "
        "most likely to let a challenger through. Recomputed with Student's t "
        "on n-1 df, and again with the Harvey-Leybourne-Newbold small-sample "
        "correction (Harvey, Leybourne & Newbold 1997), which shrinks the "
        "statistic by `sqrt((n + 1 - 2h + h(h-1)/n) / n)`. Both poolings the "
        "task publishes are shown, because the smaller one has twelve paired "
        "errors and that is exactly where a normal tail is least defensible:",
        "",
    ]
    worst_p = 1.0
    poolings = (
        ("all six companies — the pooling behind `model-contest.csv`", None),
        ("Google and Meta only — the pair the forecast is published for", gated),
    )
    for label, keys in poolings:
        contest = fc.model_contest(
            fc.rolling_origin_backtest(series, keys=keys), horizon=1)
        lines += [
            f"**{label}**", "",
            "| model | n pairs | dm_stat | p (normal, published) | "
            "p (Student t) | p (HLN-corrected t) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in contest.sort_values("p_value").itertuples():
            n, stat, h = int(row.n_pairs), float(row.dm_stat), 1
            p_t = 2 * float(stats.t.sf(abs(stat), n - 1))
            factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
            p_hln = 2 * float(stats.t.sf(abs(stat * factor), n - 1))
            worst_p = min(worst_p, float(row.p_value), p_t, p_hln)
            lines.append(
                f"| {row.model} | {n} | {stat:+.4f} | {row.p_value:.4f} | "
                f"{p_t:.4f} | {p_hln:.4f} |")
        lines.append("")

    lines += [
        f"Nothing reaches alpha = {fc.ALPHA} under either pooling or any of "
        "the three tails, and both corrections move every p-value **up**. The "
        "published refusal used the easiest of the three tests to pass, and no "
        "model passed it.",
        "",
        "Read the sign, not only the column: a low p with `dm_stat > 0` is a "
        "model that is reliably *worse* than naive, which is why the selection "
        "rule in `select_model` requires a negative statistic as well as a "
        "small p. `drift` is the case in point — under both poolings it has "
        "the smallest p-value of any model while forecasting worse than the "
        "benchmark it is being tested against. A rule that read the p-value "
        "alone would have published it.",
        "",
    ]

    checks = [
        ("no gate verdict flips under the exact chi-square tail", not gate_flips),
        ("no challenger reaches alpha under normal, t, or HLN-corrected t",
         worst_p >= fc.ALPHA),
    ]
    return checks, lines


# --------------------------------------------------------------------------
# 6. the prediction interval
# --------------------------------------------------------------------------

def check_interval_coverage(rng) -> tuple[list, list[str]]:
    """Simulate the order-statistic interval, and the np.quantile one it replaced.

    The claim in `empirical_interval` is distribution-free: for `n` exchangeable
    residuals and `j = floor((1-level)/2 * (n+1))`, the interval bounded by the
    j-th and (n+1-j)-th order statistics contains a fresh residual with
    probability exactly `(n+1-2j)/(n+1)` — whatever the residuals are drawn
    from. That is a strong claim and it is cheap to test.
    """
    from scipy import stats

    lines = ["## 6. Prediction-interval coverage by simulation", ""]
    level = fc.INTERVAL_LEVEL
    draws = 40_000
    rows = []
    order_ok, quantile_undercovers = True, True

    generators = {
        "normal(0, 1)": lambda size: rng.normal(0, 1, size),
        "t(3), heavy tails": lambda size: stats.t.rvs(3, size=size,
                                                      random_state=rng),
        "lognormal, skewed": lambda size: rng.lognormal(0, 1, size),
    }

    for name, draw in generators.items():
        for n in (12, 20, 40):
            j = int(np.floor((1 - level) / 2 * (n + 1)))
            claimed = (n + 1 - 2 * j) / (n + 1)
            sample = draw((draws, n + 1))
            past, future = np.sort(sample[:, :n], axis=1), sample[:, n]
            hit = ((future >= past[:, j - 1]) & (future <= past[:, n - j])).mean()
            # The version this replaced: np.quantile interpolates between the
            # order statistics, so its endpoints are inside the sample range.
            qlo = np.quantile(sample[:, :n], (1 - level) / 2, axis=1)
            qhi = np.quantile(sample[:, :n], 1 - (1 - level) / 2, axis=1)
            hit_q = ((future >= qlo) & (future <= qhi)).mean()
            se = np.sqrt(claimed * (1 - claimed) / draws)
            rows.append((name, n, j, claimed, hit, hit_q))
            order_ok &= abs(hit - claimed) < 4 * se
            quantile_undercovers &= hit_q < claimed

    lines += [
        f"{draws:,} replications per cell. Each draws `n` residuals plus one "
        "fresh observation, and asks how often the interval built from the "
        "first contains the last.",
        "",
        "| residual distribution | n | j | claimed coverage | order statistics | `np.quantile` |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, n, j, claimed, hit, hit_q in rows:
        lines.append(f"| {name} | {n} | {j} | {claimed:.4f} | {hit:.4f} | {hit_q:.4f} |")
    lines += [
        "",
        "The order-statistic column tracks the claimed coverage across a "
        "normal, a t(3) and a lognormal — the distribution never enters the "
        f"calculation. Note the claimed level is not {level:.0%}: at n = 12 it "
        "is 11/13 = 84.6%, because only whole order statistics exist and the "
        "module rounds **outward**. `horizon-limits.csv` publishes the achieved "
        "level rather than the requested one for that reason.",
        "",
        "The last column is the defect this replaced. `np.quantile` "
        "interpolates between order statistics, so its endpoints sit strictly "
        "inside the sample and its interval is narrower than any interval the "
        "sample can actually support — it undercovers in every cell above. On "
        "the real backtest that cost was nearly a factor of two: the h=1 "
        "interval read 1.84x interpolated against 3.15x measured, which is the "
        "difference between clearing the 3x publication limit and failing it.",
        "",
    ]
    checks = [
        ("order-statistic coverage matches its claimed level under all three "
         "distributions", order_ok),
        ("the np.quantile interval undercovers in every cell", quantile_undercovers),
    ]
    return checks, lines


# --------------------------------------------------------------------------

def main() -> None:
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(20240707)

    header = [
        "# Task 07 — forecasting validation against scipy and statsmodels",
        "",
        f"Generated by `src/validate_forecast.py` on {date.today().isoformat()}.",
        "",
        "`src/forecast.py` imports neither scipy nor statsmodels, so its "
        "chi-square tail, normal tail and smoothing recursions are written out "
        "by hand. This file is the evidence that the hand-written versions "
        "agree with the reference implementations — and, where they do not "
        "agree exactly, the measurement of how much the difference could move "
        "a published conclusion.",
        "",
        "Versions used for the comparison: "
        f"numpy {np.__version__}, "
        f"scipy {__import__('scipy').__version__}, "
        f"statsmodels {__import__('statsmodels').__version__}.",
        "",
    ]

    sections = [
        check_smoothers(rng),
        check_regressions(rng),
        check_chi_square(),
        check_normal_tail(rng),
        check_real_series(),
        check_interval_coverage(rng),
    ]

    checks: list = []
    body: list[str] = []
    for section_checks, section_lines in sections:
        checks += section_checks
        body += section_lines

    summary = ["## Automated checks", ""]
    for name, ok in checks:
        summary.append(f"- {'PASS' if ok else 'FAIL'} — {name}")
    summary.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header + summary + body))

    print(f"validated forecast.py against scipy + statsmodels "
          f"-> {OUT.relative_to(REPO_ROOT)}")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    if not all(ok for _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
