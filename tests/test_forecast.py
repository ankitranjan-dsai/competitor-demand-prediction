"""Tests for the Task 07 forecasting layer.

Same intent as the other suites: every case here is either a trap that was
actually hit while building Task 07 on the real six-company data, or a rule
the other three specialists depend on, locked in so a later refactor cannot
quietly undo it.

Four of these pin defects that were live in this module before the tests
existed, and they are the reason the suite is worth reading:

* **``np.quantile`` builds an interval narrower than its own label.** The
  first version of `empirical_interval` interpolated, and on the twelve
  one-step residuals actually available it produced an interval labelled 80%
  whose measured coverage was 67%. Interpolation invents an endpoint the
  sample does not contain, and at this n the invention is most of the
  interval. Pinned in ``test_interval_endpoints_are_real_residuals`` and
  ``test_interval_coverage_is_at_least_the_achieved_level``.
* **renormalising a partial composition fabricates a number.** With four of
  six companies failing the gate, scaling the survivors to sum to 1 reported
  Google at 58% of a panel it holds 23% of. `compositional_normalise` now
  demands the full composition and raises otherwise —
  ``test_partial_composition_is_refused`` pins both the refusal and the size
  of the lie it prevents.
* **a level series must carry the level.** `pool_series` initially set its own
  denominator equal to its numerator, so every value was 1.0 and the pool
  backtested at a perfect zero error. ``test_pool_series_carries_counts``.
* **calendar time, not position.** February is dropped from the panel, so
  January and March are adjacent in the array and two months apart in the
  world. A drift model indexed by position spreads January's move over one
  step instead of two. ``test_gap_is_measured_in_calendar_time``.

And the rules the task turns on:

* **naive is published unless a challenger beats it on a test**, not on a
  ranking. On six origins per company the RMSE ordering is noise, and in the
  real data the lowest-RMSE model differs from the selected one for four of
  six companies. Both directions are pinned: a challenger that genuinely wins
  is selected, and one that merely leads on RMSE is not.
* **the backtest never sees its own target.** Pinned with a recording model
  that reports the largest training period it was handed.
* **the crawler is the most predictable series in the file.** Naive one-step
  log-RMSE runs 0.29 on the panel pool, 0.53 on the shares and 0.65 on the
  counts. That ordering is the whole argument for refusing a level forecast,
  so it is a test rather than a sentence in a report.
* **the gate refuses.** Thin series, constant series, and a seasonal model in
  a single-year window all get a refusal with a reason attached.
* the two standing checks: no ``country``/``share_of_all`` column and no
  personal-data column reaches a committed table.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compare as cmp  # noqa: E402
import forecast as fc  # noqa: E402
import trends as tr  # noqa: E402

MONTH = tr.period_col("month")
COMPANIES = ["google", "meta", "microsoft", "snowflake", "databricks", "nvidia"]


# ---------------------------------------------------------------------------
# Fixtures. Synthetic unless a test is specifically about the real data.
# ---------------------------------------------------------------------------


def make_frame(counts: dict[str, dict[str, int]]) -> pd.DataFrame:
    """A posting frame from ``{publisher: {period: n}}``."""
    rows = []
    for publisher, per_period in counts.items():
        for period, n in per_period.items():
            for i in range(n):
                rows.append({"job_id": f"{publisher}-{period}-{i}",
                             fc.PUBLISHER_COL: publisher, MONTH: period})
    return pd.DataFrame(rows, columns=["job_id", fc.PUBLISHER_COL, MONTH])


def months(n: int = 12, start: int = 1) -> list[str]:
    return [f"2023-{m:02d}" for m in range(start, start + n)]


def series_from(values: dict[str, list], denominator: int = 200,
                key: str = "a", periods=None) -> pd.DataFrame:
    """A demand series straight from share values, bypassing panel logic."""
    periods = periods or months(len(next(iter(values.values()))))
    rows = []
    for name, shares in values.items():
        for period, share in zip(periods, shares):
            rows.append({"key": name, "period": period,
                         "numerator": int(round(share * denominator)),
                         "denominator": denominator, "share": float(share),
                         "is_observed": True})
    return pd.DataFrame(rows, columns=fc.SERIES_COLUMNS)


@pytest.fixture(scope="module")
def real_frames():
    """The real six-company frames, or skip if the row-level build has not run."""
    try:
        return cmp.load_frames(COMPANIES)
    except FileNotFoundError as exc:                      # pragma: no cover
        pytest.skip(f"row-level data not built: {exc}")


@pytest.fixture(scope="module")
def real_series(real_frames):
    return fc.panel_share_series(real_frames)


@pytest.fixture(scope="module")
def real_backtest(real_series):
    return fc.rolling_origin_backtest(real_series)


@pytest.fixture(scope="module")
def gated_keys(real_series):
    gate = fc.forecastability_table(real_series)
    return sorted(gate.loc[gate.verdict == "forecastable", "key"])


# ---------------------------------------------------------------------------
# Period arithmetic — the February gap
# ---------------------------------------------------------------------------


def test_period_ordinal_is_calendar_not_positional():
    """A missing month must leave a hole in the index, not close up."""
    keys = ["2023-01", "2023-03", "2023-04"]
    assert list(fc.period_ordinal(keys)) == [0.0, 2.0, 3.0]


def test_period_ordinal_crosses_a_year_boundary():
    assert list(fc.period_ordinal(["2023-11", "2023-12", "2024-01"])) == [0.0, 1.0, 2.0]


def test_gap_is_measured_in_calendar_time():
    """The trap February sets for every trend model in this task.

    A series that rises by 1.0 in log terms from January to March has risen by
    0.5 per month, not by 1.0. Indexing by position — which is what a naive
    ``range(len(y))`` gives after the missing month is dropped — reads the
    slope as twice what it is and doubles the extrapolation.
    """
    y = np.array([0.0, 1.0])
    calendar = fc.period_ordinal(["2023-01", "2023-03"])
    positional = np.arange(2, dtype=float)

    assert fc.m_drift(y, calendar, 3.0) == pytest.approx(1.5)
    assert fc.m_drift(y, positional, 2.0) == pytest.approx(2.0)


def test_next_periods_rolls_the_year_over():
    assert fc.next_periods("2023-12", 3) == ["2024-01", "2024-02", "2024-03"]
    assert fc.next_periods("2023-10", 2) == ["2023-11", "2023-12"]


def test_next_periods_refuses_a_frequency_it_cannot_forecast():
    with pytest.raises(ValueError):
        fc.next_periods("2023-W40", 1, period="week")


def test_unobserved_periods_is_derived_not_hardcoded():
    """February is found in the data, not written into the module.

    Built so a different dataset with a different empty month gets the same
    treatment. Here the gap is deliberately put in May.
    """
    frames = {
        "a": make_frame({"P1": {"2023-04": 5, "2023-06": 5}, "P2": {"2023-05": 5}}),
        "b": make_frame({"P1": {"2023-04": 5, "2023-06": 5}, "P3": {"2023-05": 5}}),
    }
    assert fc.unobserved_periods(frames) == ["2023-05"]


def test_february_is_unobserved_in_the_real_panel(real_frames):
    assert fc.unobserved_periods(real_frames) == ["2023-02"]


def test_unobserved_period_is_excluded_from_the_series(real_series):
    feb = real_series[real_series.period == "2023-02"]
    assert len(feb) == len(COMPANIES)
    assert not feb.is_observed.any()
    assert (feb.numerator > 0).any(), (
        "February must be carried as unobserved rather than dropped — it has "
        "postings on the window-level panel, and hiding that is what C5 corrects"
    )
    assert set(fc.observed(real_series, "google").period) == set(months()) - {"2023-02"}


# ---------------------------------------------------------------------------
# Series construction
# ---------------------------------------------------------------------------


def test_panel_shares_sum_to_one_within_a_period(real_series):
    totals = real_series[real_series.is_observed].groupby("period").share.sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_relative_share_ignores_a_shared_collection_factor():
    """Task 06's identifying property, carried into the forecast object.

    Every company arrives through one crawler. If the crawler indexes twice as
    hard in a month, every count doubles and no share moves. A forecast built
    on shares therefore cannot mistake a collection surge for demand — which
    is the entire reason Task 07 forecasts shares.
    """
    base = {"P": {"2023-01": 10, "2023-02": 10}}
    doubled = {"P": {"2023-01": 10, "2023-02": 20}}
    quiet = fc.panel_share_series({"a": make_frame(base), "b": make_frame(base)})
    surge = fc.panel_share_series({"a": make_frame(doubled), "b": make_frame(doubled)})
    assert np.allclose(sorted(quiet.share), sorted(surge.share))


def test_pool_series_carries_counts(real_frames):
    """A level series must hold the level.

    This was a real defect: `pool_series` set ``denominator = numerator`` and
    left ``share`` at a constant 1.0, so the pool's log series was all zeros
    and it backtested at exactly zero error — a perfect forecast of nothing.
    """
    pool = fc.pool_series(real_frames)
    assert (pool.share == pool.numerator).all()
    assert pool.share.nunique() > 1
    assert int(pool.loc[pool.period == "2023-01", "numerator"].iloc[0]) == 158


def test_company_count_series_matches_the_share_numerators(real_frames):
    share = fc.panel_share_series(real_frames)
    level = fc.company_count_series(real_frames)
    merged = share.merge(level, on=["key", "period"], suffixes=("_s", "_l"))
    assert (merged.numerator_s == merged.numerator_l).all()
    assert (merged.share_l == merged.numerator_l).all()


def test_safe_log_survives_an_empty_cell():
    """A zero month must not take every downstream mean to negative infinity."""
    out = fc._safe_log(np.array([0.0, 0.25]), np.array([200.0, 200.0]))
    assert np.isfinite(out).all()
    assert out[0] < out[1]
    assert out[0] == pytest.approx(math.log(0.5 / 200))


def test_skill_series_excludes_facilities_operations():
    """Task 04's denominator rule, inherited unchanged.

    Google's data-centre roles genuinely carry no software skills. Leaving
    them in the denominator measures extraction coverage instead of demand.
    """
    features = pd.DataFrame({
        "job_id": ["1", "2", "3"],
        MONTH: ["2023-01"] * 3,
        "job_function": ["Engineering", "Facilities / Operations", "Engineering"],
        "has_any_skill": [True, True, True],
    })
    long = pd.DataFrame({"job_id": ["1", "2", "3"], "skill": ["Python"] * 3})
    out = fc.skill_share_series(long, features, skills=["Python"])
    assert int(out.denominator.iloc[0]) == 2


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_homogeneity_finds_no_signal_in_a_constant_share():
    """Binomial noise around a fixed proportion must not read as movement."""
    rng = np.random.default_rng(11)
    n = np.full(12, 400)
    k = rng.binomial(400, 0.25, size=12)
    out = fc.homogeneity_test(k, n)
    assert out["p_value"] > 0.05
    assert out["overdispersion"] < 2.0
    assert out["signal_share"] < 0.5


def test_homogeneity_finds_signal_in_a_share_that_actually_moves():
    n = np.full(12, 400)
    k = (np.linspace(0.10, 0.40, 12) * 400).astype(int)
    out = fc.homogeneity_test(k, n)
    assert out["p_value"] < 1e-6
    assert out["signal_share"] > 0.9


def test_homogeneity_p_value_matches_scipy():
    """The Wilson-Hilferty approximation is the no-scipy promise's weak point."""
    scipy_stats = pytest.importorskip("scipy.stats")
    n = np.full(12, 300)
    k = np.array([60, 72, 55, 81, 66, 70, 58, 77, 64, 69, 73, 61])
    out = fc.homogeneity_test(k, n)
    expected = scipy_stats.chi2.sf(out["chi2"], out["df"])
    assert out["p_value"] == pytest.approx(expected, abs=2e-3)


def test_gate_refuses_a_series_with_a_one_posting_month():
    """Thin beats interesting.

    A series can look like it is moving and be measuring a cell of one. In the
    real panel Snowflake reaches a month with a single posting, and its share
    that month carries a relative standard error near 100%.
    """
    series = series_from({"thin": [0.30, 0.05, 0.30, 0.05, 0.30, 0.05,
                                   0.30, 0.05, 0.30, 0.05, 0.30, 0.005]},
                         denominator=200)
    gate = fc.forecastability_table(series)
    assert gate.verdict.iloc[0] == "too_thin"
    assert "smallest cell" in gate.reason.iloc[0]


def test_gate_refuses_a_series_that_is_only_noise():
    rng = np.random.default_rng(3)
    shares = rng.binomial(400, 0.25, size=12) / 400
    gate = fc.forecastability_table(series_from({"flat": shares}, denominator=400))
    assert gate.verdict.iloc[0] == "noise_only"
    assert "constant share not rejected" in gate.reason.iloc[0]


def test_gate_passes_a_thick_series_that_moves():
    shares = np.linspace(0.10, 0.40, 12)
    gate = fc.forecastability_table(series_from({"real": shares}, denominator=400))
    assert gate.verdict.iloc[0] == "forecastable"
    assert gate.signal_share.iloc[0] > 0.9


def test_gate_refuses_a_series_that_is_too_short():
    series = series_from({"short": [0.2, 0.3, 0.25, 0.28]}, denominator=400)
    gate = fc.forecastability_table(series)
    assert gate.verdict.iloc[0] == "too_thin"
    assert "observed periods" in gate.reason.iloc[0]


def test_only_google_and_meta_survive_the_real_gate(real_series):
    """The headline restriction, and it does not hinge on the threshold.

    Four of six companies reach a month too thin to model. The verdict set is
    identical at a minimum cell of 5 and at Task 06's floor of 10, so the
    finding is not an artefact of where the line was drawn — though at 3 it
    would widen to four companies, which the report states.
    """
    for min_cell in (5, 10):
        gate = fc.forecastability_table(real_series, min_cell=min_cell)
        passing = set(gate.loc[gate.verdict == "forecastable", "key"])
        assert passing == {"google", "meta"}, f"changed at min_cell={min_cell}"


def test_real_series_all_carry_signal_even_when_refused(real_series):
    """Refused for thinness is not the same as refused for being noise.

    All six companies reject a constant share, and between 61% and 86% of
    their variance survives the binomial correction. The series move; the task
    is that nothing predicts the movement. Both halves have to stay true or
    the report is telling a simpler story than the data.
    """
    gate = fc.forecastability_table(real_series)
    assert (gate.homogeneity_p < 0.005).all()
    assert gate.signal_share.min() > 0.60
    assert "noise_only" not in set(gate.verdict)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_naive_returns_the_last_observation():
    y = np.array([1.0, 2.0, 3.0])
    assert fc.m_naive(y, np.arange(3.0), 9.0) == 3.0


def test_every_model_reproduces_a_constant_series():
    """A series that never moves must not be forecast to move."""
    y = np.full(8, math.log(0.25))
    t = np.arange(8.0)
    for name, model in fc.MODELS.items():
        assert model(y, t, 8.0) == pytest.approx(math.log(0.25), abs=1e-6), name


def test_loglinear_recovers_a_known_slope():
    t = np.arange(10.0)
    y = 0.5 + 0.3 * t
    assert fc.m_loglinear(y, t, 12.0) == pytest.approx(0.5 + 0.3 * 12.0)


def test_ar1_reverts_toward_the_mean():
    rng = np.random.default_rng(5)
    y = 2.0 + rng.normal(0, 0.1, 40)
    y[-1] = 4.0
    out = fc.m_ar1(y, np.arange(40.0), 45.0)
    assert 2.0 <= out < 4.0


def test_damped_holt_flattens_at_long_range():
    """Damping is the point: the trend must not run away with the horizon."""
    t = np.arange(10.0)
    y = 0.5 + 0.2 * t
    near = fc.m_holt_damped(y, t, 11.0)
    far = fc.m_holt_damped(y, t, 40.0)
    undamped = 0.5 + 0.2 * 40.0
    assert far > near
    assert far < undamped


def test_seasonal_naive_refuses_and_says_why():
    """One year is zero complete cycles, so a month-of-year term is not identified."""
    with pytest.raises(NotImplementedError) as exc:
        fc.seasonal_naive(np.arange(12.0), np.arange(12.0), 12.0)
    assert "seasonal cycles" in str(exc.value)
    assert "seasonal_naive" not in fc.MODELS


def test_naive_is_first_in_the_registry():
    """Order is the tie-break order, so a tie can never unseat the benchmark."""
    assert next(iter(fc.MODELS)) == fc.BENCHMARK == "naive"


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------


def test_backtest_never_sees_its_own_target():
    """The classic forecasting bug, pinned with a recording model.

    Every fit is handed the training slice and the target's calendar index. If
    the slice ever reaches the target period the accuracy table becomes a
    measure of nothing.
    """
    seen = []

    def spy(y, t, t_target):
        seen.append((len(y), float(t[-1]), float(t_target)))
        return float(y[-1])

    series = series_from({"a": np.linspace(0.1, 0.4, 12)}, denominator=400)
    fc.rolling_origin_backtest(series, models={"spy": spy})
    assert seen
    for _, last_train_t, target_t in seen:
        assert last_train_t < target_t


def test_backtest_training_window_expands():
    series = series_from({"a": np.linspace(0.1, 0.4, 12)}, denominator=400)
    out = fc.rolling_origin_backtest(series, models={"naive": fc.m_naive},
                                     horizons=(1,))
    assert out.n_train.min() == fc.MIN_TRAIN
    assert list(out.n_train) == sorted(out.n_train)


def test_backtest_origin_counts_are_what_the_report_claims(real_backtest):
    """Six one-step origins per company, four at three steps. Eleven points buys this."""
    for horizon, expected in ((1, 6), (2, 5), (3, 4)):
        block = real_backtest[(real_backtest.horizon == horizon)
                              & (real_backtest.model == "naive")
                              & (real_backtest.key == "google")]
        assert len(block) == expected


def test_backtest_skips_a_series_too_short_to_train():
    series = series_from({"a": [0.2, 0.3, 0.25]}, denominator=400)
    assert fc.rolling_origin_backtest(series).empty


def test_a_model_that_cannot_fit_scores_nothing_rather_than_crashing():
    def broken(y, t, t_target):
        raise RuntimeError("no")

    series = series_from({"a": np.linspace(0.1, 0.4, 12)}, denominator=400)
    out = fc.rolling_origin_backtest(series, models={"broken": broken})
    assert len(out) > 0
    assert out.predicted.isna().all()


def test_accuracy_table_ratios_are_against_naive(real_backtest):
    acc = fc.accuracy_table(real_backtest)
    base = acc[acc.model == "naive"]
    assert np.allclose(base.rmse_ratio_to_naive, 1.0)
    assert np.allclose(base.mase, 1.0)
    assert np.allclose(acc.error_factor, np.exp(acc.rmse))


# ---------------------------------------------------------------------------
# Diebold-Mariano and model selection
# ---------------------------------------------------------------------------


def test_dm_is_zero_for_identical_error_series():
    errors = np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.2])
    out = fc.diebold_mariano(errors, errors)
    assert out["dm_stat"] == 0.0
    assert out["p_value"] == 1.0


def test_dm_sign_marks_the_better_forecaster():
    rng = np.random.default_rng(7)
    good = rng.normal(0, 0.1, 60)
    bad = rng.normal(0, 1.0, 60)
    out = fc.diebold_mariano(good, bad)
    assert out["dm_stat"] < 0
    assert out["p_value"] < 0.05


def test_dm_refuses_a_sample_too_small_to_speak():
    assert math.isnan(fc.diebold_mariano([0.1, 0.2], [0.3, 0.4])["p_value"])


def test_no_model_beats_naive_on_the_real_shares(real_backtest):
    """The finding, pinned so a refactor cannot quietly produce a winner.

    Simple exponential smoothing has the lowest one-step RMSE of the seven and
    still cannot clear the bar: DM p = 0.217. Every other model is worse than
    naive outright.

    Note the direction check. The DM test is two-sided, so a low p-value can
    also mean a model is reliably *worse* — drift posts the **smallest**
    p-value of any model, 0.177, while forecasting worse than the benchmark it
    is being tested against. It does not clear alpha, so nothing is selected
    either way; what the p-value column does do is rank the worse model first.
    Reading it as a ranking, or lowering alpha to 0.2, would have published it.
    That is why `select_model` requires `dm_stat < 0` as well as a small p.
    """
    contest = fc.model_contest(real_backtest, horizon=1)
    assert not contest.beats_benchmark.any()

    best = contest.iloc[0]
    assert best.model == "ses"
    assert best.rmse < best.benchmark_rmse
    assert best.p_value == pytest.approx(0.217, abs=5e-3)

    losers = contest[contest.dm_stat > 0]
    assert (losers.rmse > losers.benchmark_rmse).all()
    assert losers.p_value.min() < 0.2, (
        "a two-sided p-value alone does not identify a winner; the sign of "
        "dm_stat is what separates better from reliably worse"
    )


def test_selection_keeps_naive_when_a_challenger_only_leads_on_rmse():
    """The rule the task turns on.

    A model can post the lowest RMSE across six origins on nothing but luck.
    Selection needs a test, not a ranking, and this pins that a lower RMSE
    alone does not win — here by a wide margin. The series is pure noise
    around a fixed level, and exponential smoothing still comes out **41%
    below** naive on RMSE. A ranking would have published it.
    """
    rng = np.random.default_rng(1)
    shares = 0.25 * np.exp(rng.normal(0, 0.25, 12))
    series = series_from({"a": shares}, denominator=400)
    backtest = fc.rolling_origin_backtest(series)
    selection = fc.select_model(backtest, horizon=1)
    row = selection.iloc[0]

    assert row.selected == "naive"
    assert row.selected_by == "benchmark_not_beaten"
    assert row.lowest_rmse_model == "ses"
    assert row.lowest_rmse < 0.6 * row.selected_rmse
    assert row.would_have_picked_differently


def test_selection_does_take_a_challenger_that_genuinely_wins():
    """Otherwise the rule is vacuous rather than conservative.

    A share pinned at one level with a single large excursion is a case where
    persistence is genuinely and repeatedly wrong: naive chases the excursion
    for a step, the historical mean does not.
    """
    shares = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                       0.60, 0.25, 0.60, 0.25, 0.60, 0.25])
    series = series_from({"spiky": shares}, denominator=400)
    backtest = fc.rolling_origin_backtest(series)
    selection = fc.select_model(backtest, horizon=1)
    row = selection.iloc[0]

    assert row.selected != "naive"
    assert row.selected_by == "beats_benchmark"
    assert row.dm_p_value < fc.ALPHA
    assert row.selected_rmse < backtest[
        (backtest.model == "naive") & (backtest.horizon == 1)
    ].error.pow(2).mean() ** 0.5


def test_real_selection_is_naive_for_every_company(real_backtest):
    selection = fc.select_model(real_backtest, horizon=1)
    assert set(selection.selected) == {"naive"}
    assert set(selection.selected_by) == {"benchmark_not_beaten"}


def test_per_company_rmse_winners_disagree_on_the_real_data(real_backtest):
    """The signature of noise-chasing, and the reason selection needs a test.

    Four of six companies have a lowest-RMSE model that is not naive, and they
    do not agree on which one. A per-company pick would have published four
    different models fitted to six months each.
    """
    selection = fc.select_model(real_backtest, horizon=1)
    assert selection.would_have_picked_differently.sum() >= 4
    assert selection.lowest_rmse_model.nunique() >= 3


def test_refusal_survives_dropping_any_single_panel_publisher(real_frames):
    """The share cancels collection only if collection is common to everyone.

    Task 05 §3 flagged three Google spike weeks as publisher batches. Two leave
    on their own — `via Google Careers` and `via The Muse` are not on the common
    panel. The third, `via Recruit.net`, is on it, and its batch is almost
    entirely Google's, so it inflates the numerator far more than the shared
    denominator. That is exactly the assumption §1.1 rests on, failing.

    So drop each panel publisher in turn and re-run the whole verdict. The
    published numbers move — Google's July share falls 7.6 pp without
    Recruit.net — and both conclusions hold on every one of the seven panels.
    """
    publishers = cmp.common_publishers(real_frames)
    assert "via Recruit.net" in publishers
    assert not {"via Google Careers", "via The Muse"} & set(publishers)

    # The batch itself, before the verdict: it is the number both write-ups quote.
    published = fc.panel_share_series(real_frames, publishers=publishers)
    without_rn = fc.panel_share_series(
        real_frames, publishers=[p for p in publishers if p != "via Recruit.net"])
    july = published[(published.key == "google") & (published.period == "2023-07")]
    july_without = without_rn[(without_rn.key == "google")
                              & (without_rn.period == "2023-07")]
    moved_pp = (july.share.iloc[0] - july_without.share.iloc[0]) * 100
    assert moved_pp == pytest.approx(7.6, abs=0.05)
    assert int(july.numerator.iloc[0] - july_without.numerator.iloc[0]) == 24

    for dropped in publishers:
        panel = [p for p in publishers if p != dropped]
        series = fc.panel_share_series(real_frames, publishers=panel)
        gate = fc.forecastability_table(series)
        passing = sorted(gate.loc[gate.verdict == "forecastable", "key"])
        backtest = fc.rolling_origin_backtest(series)
        contest = fc.model_contest(backtest, horizon=1)
        verdict = fc.horizon_verdict(fc.horizon_table(backtest, keys=passing))

        assert not contest.beats_benchmark.any(), (
            f"dropping {dropped} produces a winner; the refusal in the report "
            "is then a property of one publisher rather than of the data")
        assert verdict.max_useful_horizon == 0, (
            f"dropping {dropped} makes a horizon usable")


def test_a_significantly_worse_model_is_refused_on_the_real_data(real_frames):
    """Condition 2 of `select_model`, earning its place on real numbers.

    On the published panel nothing comes near alpha, so the sign guard looks
    decorative. Drop `via BeBee` and it stops being decorative: `loglinear`
    lands at p = 0.051 with a **positive** statistic and an RMSE 24% above
    naive's. It is significantly *worse*, and a rule that read the p-value
    column alone — or ran at alpha = 0.10 — would have published it.
    """
    panel = [p for p in cmp.common_publishers(real_frames) if p != "via BeBee"]
    series = fc.panel_share_series(real_frames, publishers=panel)
    backtest = fc.rolling_origin_backtest(series)

    contest = fc.model_contest(backtest, horizon=1)
    closest = contest.loc[contest.p_value.idxmin()]
    assert closest.model == "loglinear"
    assert closest.p_value < 0.06
    assert closest.dm_stat > 0
    assert closest.rmse > closest.benchmark_rmse
    assert not closest.beats_benchmark

    selection = fc.select_model(backtest, horizon=1, alpha=0.10)
    assert set(selection.selected) == {"naive"}, (
        "even at alpha = 0.10 the sign requirement keeps a worse model out")


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def test_interval_endpoints_are_real_residuals(real_backtest, gated_keys):
    """The interpolation defect, pinned at its source.

    ``np.quantile`` returns a value between two residuals. At twelve residuals
    that invented endpoint is most of the interval, and it made an 80% band
    cover 67%. The endpoints must be observations.
    """
    band = fc.empirical_interval(real_backtest, "naive", 1, keys=gated_keys)
    residuals = real_backtest[(real_backtest.model == "naive")
                              & (real_backtest.horizon == 1)
                              & (real_backtest.key.isin(gated_keys))].error.to_numpy()
    assert band["sufficient"]
    assert band["lo"] in residuals
    assert band["hi"] in residuals


def test_interval_coverage_is_at_least_the_achieved_level(real_backtest, gated_keys):
    for horizon in (1, 2):
        out = fc.interval_coverage(real_backtest, "naive", horizon, keys=gated_keys)
        assert out["covered"] >= out["achieved"]


def test_achieved_level_meets_the_nominal_level():
    """Conservative by construction — never sell more coverage than was bought."""
    backtest = pd.DataFrame({
        "key": ["a"] * 40, "model": ["naive"] * 40, "horizon": [1] * 40,
        "error": np.linspace(-1, 1, 40),
    })
    band = fc.empirical_interval(backtest, "naive", 1, level=0.80)
    assert band["sufficient"]
    assert band["achieved_level"] >= 0.80


def test_interval_refuses_when_residuals_are_too_few():
    """Derived refusal, not a hand-set threshold.

    With eight residuals the endpoint an 80% interval needs sits outside the
    sample, so no distribution-free interval exists. This is why the real
    three-step forecast is published without a band.
    """
    backtest = pd.DataFrame({
        "key": ["a"] * 8, "model": ["naive"] * 8, "horizon": [3] * 8,
        "error": np.linspace(-1, 1, 8),
    })
    band = fc.empirical_interval(backtest, "naive", 3, level=0.80)
    assert not band["sufficient"]
    assert math.isnan(band["lo"])
    assert "cannot bound" in band["reason"]


def test_real_three_step_interval_does_not_exist(real_backtest, gated_keys):
    band = fc.empirical_interval(real_backtest, "naive", 3, keys=gated_keys)
    assert band["n_residuals"] == 8
    assert not band["sufficient"]


def test_interval_pools_only_the_keys_being_published(real_backtest, gated_keys):
    """Snowflake's errors must not widen Google's interval.

    Snowflake reaches months of one posting and backtests at a one-step RMSE
    of 1.00 against Google's 0.37. Pooling all six inflates the published
    interval by half for a reason that has nothing to do with the companies
    being forecast.
    """
    gated = fc.empirical_interval(real_backtest, "naive", 1, keys=gated_keys)
    everyone = fc.empirical_interval(real_backtest, "naive", 1)
    assert gated["n_residuals"] == 12
    assert everyone["n_residuals"] == 36
    assert (gated["hi"] - gated["lo"]) < (everyone["hi"] - everyone["lo"])


def test_horizon_verdict_refuses_an_interval_that_excludes_nothing():
    horizons = pd.DataFrame({
        "horizon": [1, 2], "interval_factor": [5.0, 6.0], "n_residuals": [30, 25],
        "interval_sufficient": [True, True], "insufficient_reason": ["", ""],
    })
    verdict = fc.horizon_verdict(horizons)
    assert verdict.max_useful_horizon == 0
    assert "excludes nothing" in verdict.reason


def test_horizon_verdict_stops_at_the_first_failure():
    """A longer horizon that happens to look better is an accident of eight points."""
    horizons = pd.DataFrame({
        "horizon": [1, 2, 3], "interval_factor": [2.0, 9.0, 2.1],
        "n_residuals": [30, 25, 20], "interval_sufficient": [True, True, True],
        "insufficient_reason": ["", "", ""],
    })
    assert fc.horizon_verdict(horizons).max_useful_horizon == 1


def test_real_horizon_verdict_is_zero_under_either_pooling(real_backtest, gated_keys):
    """The task's conclusion, and it does not depend on the pooling choice.

    On the two published companies the one-step interval spans 3.15x; on all
    six it spans 4.34x. Both are past the point where an interval excludes
    anything, so no horizon is supported either way.
    """
    for keys in (gated_keys, None):
        verdict = fc.horizon_verdict(fc.horizon_table(real_backtest, keys=keys))
        assert verdict.max_useful_horizon == 0


# ---------------------------------------------------------------------------
# The published forecast
# ---------------------------------------------------------------------------


def test_forecast_table_covers_only_keys_that_passed_the_gate(real_series, real_backtest):
    gate = fc.forecastability_table(real_series)
    selection = fc.select_model(real_backtest, horizon=1)
    out = fc.forecast_table(real_series, real_backtest, selection, gate=gate)
    assert set(out.key) == {"google", "meta"}


def test_every_real_forecast_row_is_marked_unsupported(real_series, real_backtest, gated_keys):
    """The refusal has to travel with the number.

    A row lifted out of this table and onto a slide must carry its own
    disclaimer, because the caption will not survive the copy-paste.
    """
    gate = fc.forecastability_table(real_series)
    selection = fc.select_model(real_backtest, horizon=1)
    verdict = fc.horizon_verdict(fc.horizon_table(real_backtest, keys=gated_keys))
    out = fc.forecast_table(real_series, real_backtest, selection,
                            gate=gate, verdict=verdict)
    assert not out.supported.any()
    assert out.out_of_window.all()
    assert set(out.selected_by) == {"benchmark_not_beaten"}


def test_a_persistence_forecast_repeats_the_last_observation(real_series, real_backtest):
    gate = fc.forecastability_table(real_series)
    selection = fc.select_model(real_backtest, horizon=1)
    out = fc.forecast_table(real_series, real_backtest, selection, gate=gate)
    assert np.allclose(out.point_share, out.last_observed_share)


def test_forecast_targets_are_outside_the_collection_window(real_series, real_backtest):
    gate = fc.forecastability_table(real_series)
    selection = fc.select_model(real_backtest, horizon=1)
    out = fc.forecast_table(real_series, real_backtest, selection, gate=gate)
    assert set(out.target_period) == {"2024-01", "2024-02", "2024-03"}
    assert out.out_of_window.all()


def test_partial_composition_is_refused(real_series, real_backtest):
    """Renormalising a subset does not fix a composition, it invents one.

    Google and Meta hold about 40% of the panel between them. Scaling the two
    survivors to sum to 1 put Google at 58% — of a pool where it sits near
    23%. The four companies that failed the gate did not stop holding their
    share when they failed it.
    """
    gate = fc.forecastability_table(real_series)
    selection = fc.select_model(real_backtest, horizon=1)
    out = fc.forecast_table(real_series, real_backtest, selection, gate=gate)

    with pytest.raises(ValueError) as exc:
        fc.compositional_normalise(out, expected_keys=COMPANIES)
    assert "complete composition" in str(exc.value)

    inflated = out.point_share / out.groupby(["target_period", "horizon"]
                                             ).point_share.transform("sum")
    assert inflated.max() > 0.55, "the refused rescaling is the one worth refusing"


def test_compositional_normalise_works_on_a_complete_composition():
    forecasts = pd.DataFrame({
        "key": ["a", "b"], "target_period": ["2024-01"] * 2, "horizon": [1, 1],
        "point_share": [0.30, 0.30],
    })
    out = fc.compositional_normalise(forecasts, expected_keys=["a", "b"])
    assert out.point_share.sum() == pytest.approx(1.0)
    assert np.allclose(out.raw_point_share, 0.30)


# ---------------------------------------------------------------------------
# Levels are not identified — the measurement, not the assertion
# ---------------------------------------------------------------------------


def test_the_crawler_is_the_most_predictable_series_in_the_file(real_frames):
    """Why a level forecast is refused rather than caveated.

    Naive one-step log-RMSE: 0.29 on the panel pool, 0.53 on the company
    shares, 0.65 on the company counts. The collection is easier to predict
    than the demand, so a count forecast that scores well has learned the
    crawler's indexing schedule. Its better score is the reason to refuse it.
    """
    def naive_rmse(series):
        backtest = fc.rolling_origin_backtest(series, models={"naive": fc.m_naive},
                                              horizons=(1,))
        return float(np.sqrt(np.mean(np.square(backtest.error))))

    pool = naive_rmse(fc.pool_series(real_frames))
    share = naive_rmse(fc.panel_share_series(real_frames))
    level = naive_rmse(fc.company_count_series(real_frames))

    assert pool < share < level
    assert pool == pytest.approx(0.2935, abs=5e-3)


# ---------------------------------------------------------------------------
# Standing guards
# ---------------------------------------------------------------------------


def test_no_forbidden_column_reaches_a_task_07_table(real_series, real_backtest, gated_keys):
    gate = fc.forecastability_table(real_series)
    selection = fc.select_model(real_backtest, horizon=1)
    tables = [
        real_series, gate, real_backtest, selection,
        fc.accuracy_table(real_backtest), fc.model_contest(real_backtest, 1),
        fc.horizon_table(real_backtest, keys=gated_keys),
        fc.forecast_table(real_series, real_backtest, selection, gate=gate),
    ]
    for table in tables:
        assert fc.forbidden_columns(table) == []


def test_no_personal_data_column_reaches_a_task_07_table(real_series, real_backtest):
    gate = fc.forecastability_table(real_series)
    selection = fc.select_model(real_backtest, horizon=1)
    for table in (real_series, gate, real_backtest, selection,
                  fc.accuracy_table(real_backtest)):
        assert fc.personal_data_columns_present(table) == []


def test_forbidden_column_check_still_bites():
    assert fc.forbidden_columns(pd.DataFrame({"country": ["UK"]})) == ["country"]
    assert fc.personal_data_columns_present(
        pd.DataFrame({"recruiter_email": ["a@b.c"]})) != []


def test_forecast_module_imports_no_heavy_stats_dependency():
    """The no-scipy promise, pinned at the source.

    `validate_forecast.py` cross-checks this module against scipy and
    statsmodels precisely because this module does not use them, and that
    argument collapses the moment someone adds a convenience import here. The
    check is on the source text rather than on `sys.modules`, because pytest
    itself pulls scipy in through other suites.
    """
    source = (Path(fc.__file__)).read_text()
    for banned in ("scipy", "statsmodels", "prophet", "sklearn"):
        assert f"import {banned}" not in source, (
            f"forecast.py imports {banned}; the module is meant to run in an "
            "environment that has none of them, and validate_forecast.py's "
            "cross-check is only meaningful while that stays true")
