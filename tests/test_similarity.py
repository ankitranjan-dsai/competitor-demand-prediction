"""Tests for the Task 08 company-similarity layer.

Same intent as the other suites: every case here is either a trap that was
actually hit while building Task 08 on the real six-company data, or a rule
the other three specialists depend on, locked in so a later refactor cannot
quietly undo it.

Four of these pin defects that were live while this module was being written,
and they are the reason the suite is worth reading:

* **`-1/(D-1)` is not the null this data has.** The validator first asserted
  the analytic closure expectation falls inside the simulated band; it does
  not, and the fault was the assertion. The formula is exact under
  *geometric* closure and a share table is closed *arithmetically*, which
  lands short of it by +0.018 at six parts and sigma 0.3. Pinned in
  ``test_arithmetic_closure_falls_short_of_the_analytic_value`` and
  ``test_geometric_closure_reproduces_the_analytic_value``.
* **the null's dispersion was a placeholder.** `closure_null`'s sigma
  defaulted to 0.3 while the panel's own CLR spread is 0.80, and the band
  moves with it — [-0.197, -0.165] at sigma 0.2 against [-0.168, 0.039] at
  1.2. A refusal cannot rest on a constant nobody chose. Pinned in
  ``test_calibrate_sigma_recovers_the_dispersion_it_is_given`` and
  ``test_the_closure_band_widens_with_dispersion``.
* **resampling shares instead of postings gives intervals that are wrong by
  an order of magnitude.** The sampling unit is a posting; a company's skill
  shares are not an independent draw of anything.
  ``test_bootstrap_width_tracks_posting_count``.
* **an undefined similarity must not come back as a number.** Cosine,
  Jensen – Shannon and Spearman return nan on an all-zero vector rather than
  0 or 1, because both of those are claims — "maximally dissimilar" and
  "identical" — that an empty profile does not support. Bray – Curtis is the
  principled exception, and the test says why.
  ``test_metrics_are_undefined_on_an_empty_profile``.

And the rules the task turns on:

* **similarity has no zero.** Raw cosine on these profiles runs 0.50-0.92 and
  two *unrelated* profiles score 0.10-0.19, so a raw number cannot be read
  without both nulls. The calibration formula and its two ends are pinned.
* **the ranking is metric-dependent, and `cosine` is the odd one out.** It is
  `PRIMARY_METRIC`, it correlates 0.72 with its own family and -0.03 with the
  rank metrics. Pinned against the committed concordance table.
* **only ranks that survive resampling are published.** The 0.90 stability
  floor, and the tier collapse that follows from overlapping rank intervals.
* **a dendrogram that is sample-stable can still be specification-unstable.**
  Cuts must nest, and support is reported for both vocabularies.
* **February is never filled.** `panel_wide` intersects observed periods.
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

import compare as cmp        # noqa: E402
import similarity as sim     # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "members" / "ankit-google" / "task-08-tables"
COMPANIES = ["databricks", "google", "meta", "microsoft", "nvidia", "snowflake"]


# ---------------------------------------------------------------------------
# Fixtures. Synthetic unless a test is specifically about the real data.
# ---------------------------------------------------------------------------


def make_incidence(profile: dict[str, list[float]], n: int = 200,
                   seed: int = 11) -> dict[str, np.ndarray]:
    """Posting x skill 0/1 matrices drawn from per-company skill rates."""
    rng = np.random.default_rng(seed)
    return {key: (rng.random((n, len(rates))) < np.array(rates)).astype(np.int8)
            for key, rates in profile.items()}


def make_profile(values: dict[str, list[float]]) -> pd.DataFrame:
    """A skill x company frame straight from share values."""
    n = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.Index([f"s{i}" for i in range(n)], name="skill"))


def closed_panel(sigma: float, n_periods: int = 400, n_parts: int = 6,
                 seed: int = 7) -> pd.DataFrame:
    """A synthetic share panel with a known pre-closure dispersion."""
    rng = np.random.default_rng(seed)
    x = np.exp(rng.normal(0.0, sigma, size=(n_periods, n_parts)))
    x /= x.sum(axis=1, keepdims=True)
    return pd.DataFrame(x, columns=[f"c{i}" for i in range(n_parts)])


def committed(name: str) -> pd.DataFrame:
    """A committed Task 08 table, or skip if the build has not been run."""
    path = TABLES / f"{name}.csv"
    if not path.exists():                                 # pragma: no cover
        pytest.skip(f"task-08 tables not built: {path.name}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def real_incidence():
    """The real six-company incidence matrices, or skip without row-level data."""
    try:
        frames = cmp.load_frames(COMPANIES)
        longs = cmp.load_long(COMPANIES)
    except FileNotFoundError as exc:                      # pragma: no cover
        pytest.skip(f"row-level data not built: {exc}")
    return sim.incidence(longs, frames)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_are_one_on_identical_vectors():
    v = np.array([0.4, 0.1, 0.0, 0.25, 0.9])
    assert sim.cosine(v, v) == pytest.approx(1.0)
    assert sim.jensen_shannon(v, v) == pytest.approx(1.0)
    assert sim.bray_curtis(v, v) == pytest.approx(1.0)
    assert sim.spearman(v, v) == pytest.approx(1.0)
    assert sim.jaccard({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)


def test_metrics_are_symmetric():
    a = np.array([0.4, 0.1, 0.0, 0.25, 0.9])
    b = np.array([0.1, 0.3, 0.2, 0.05, 0.4])
    for fn in (sim.cosine, sim.jensen_shannon, sim.bray_curtis, sim.spearman):
        assert fn(a, b) == pytest.approx(fn(b, a))


def test_metrics_are_undefined_on_an_empty_profile():
    """nan, not 0 and not 1 — both of those would be claims.

    A company with no skilled postings has no direction in skill space. A 0
    would say it is maximally unlike everyone and a 1 would say it matches
    everyone; the honest answer is that the question has no value here, and
    a nan is the only value that survives into a table as a visible gap.

    Bray – Curtis is the exception and belongs in this test for that reason.
    It measures shared abundance rather than direction, so an empty profile
    against a non-empty one is a genuine zero — nothing is shared — and only
    two empty profiles are undefined. The distinction is worth pinning
    because it is the one place the five metrics disagree about what an
    empty vector *means*.
    """
    zero, other = np.zeros(5), np.array([0.2, 0.4, 0.1, 0.0, 0.3])
    for fn in (sim.cosine, sim.jensen_shannon, sim.spearman):
        assert math.isnan(fn(zero, other)), fn.__name__
        assert math.isnan(fn(zero, zero)), fn.__name__
    assert sim.bray_curtis(zero, other) == 0.0
    assert math.isnan(sim.bray_curtis(zero, zero))
    assert math.isnan(sim.jaccard(set(), set()))


def test_spearman_averages_tied_ranks():
    """Most skills are zero for most companies, so ties are the common case.

    With min or first tie-breaking the zeros acquire an arbitrary order and
    the correlation reads a pattern out of the tie-breaking rule rather than
    out of the data.
    """
    a = np.array([0.0, 0.0, 0.0, 0.5, 0.9])
    b = np.array([0.0, 0.0, 0.0, 0.9, 0.5])
    assert sim.spearman(a, b) == pytest.approx(sim.spearman(b, a))
    # the three tied zeros carry no information either way
    assert sim.spearman(a, b) < 1.0
    assert sim.spearman(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 2.0])) == pytest.approx(1.0)


def test_jaccard_is_a_set_metric_not_a_share_metric():
    """Two companies can share every skill and ask for them at wildly
    different rates; Jaccard says 1.0 and cosine does not. That gap is the
    reason both families ship."""
    assert sim.jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0
    assert sim.jaccard({"a"}, {"b"}) == 0.0
    assert sim.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_every_metric_is_registered_with_a_family():
    for name in sim.METRICS:
        assert name in sim.METRIC_FAMILY, name
    assert sim.METRIC_FAMILY["jaccard_supported"] == "rank_or_set"
    assert sim.PRIMARY_METRIC in sim.METRICS


# ---------------------------------------------------------------------------
# Pair tables and calibration — similarity has no zero
# ---------------------------------------------------------------------------


def test_pair_table_has_one_row_per_unordered_pair():
    prof = make_profile({"a": [0.5, 0.2, 0.1], "b": [0.4, 0.3, 0.0],
                         "c": [0.1, 0.1, 0.7], "d": [0.0, 0.6, 0.2]})
    pt = sim.pair_table(prof)
    assert len(pt) == 6                     # 4 choose 2
    assert len(sim.pairs(prof.columns)) == 6
    assert (pt.company_a < pt.company_b).all()


def test_calibration_puts_zero_at_unrelated_and_one_at_identical():
    """The formula is the whole point of the calibration table."""
    inc = make_incidence({"a": [0.6, 0.3, 0.1, 0.05] * 6,
                          "b": [0.5, 0.35, 0.15, 0.05] * 6,
                          "c": [0.1, 0.7, 0.05, 0.4] * 6}, n=300)
    prof = sim.profile_frame(inc, [f"s{i}" for i in range(24)])
    cal = sim.calibration_table(prof, inc, draws=60, seed=sim.SEED)
    for row in cal.itertuples():
        expected = ((row.observed - row.null_unrelated)
                    / (row.null_identical - row.null_unrelated))
        assert row.calibrated == pytest.approx(expected, abs=5e-4)
        assert row.null_unrelated < row.null_identical


def test_identical_null_is_near_one_and_unrelated_null_is_not_near_zero():
    """Neither end of the ruler is where an unaided reader would put it.

    Two identical companies do not score 1.0 at these sample sizes, and two
    unrelated ones do not score 0. Reading a raw 0.87 as "87% similar"
    requires both of those to be false, which is why no raw value is
    published without its calibration.
    """
    cal = committed("similarity-calibration")
    assert cal.null_identical.min() > 0.98
    assert cal.null_identical.max() < 1.0
    assert cal.null_unrelated.min() > 0.05
    assert cal.null_unrelated.max() < 0.25
    assert (cal.observed < cal.null_identical_p2_5).all()
    assert (cal.observed > cal.null_unrelated_p97_5).all()


def test_calibration_is_monotone_in_the_observed_value():
    prof = make_profile({"a": [0.5, 0.2, 0.1, 0.4], "b": [0.45, 0.25, 0.1, 0.35],
                         "c": [0.1, 0.6, 0.3, 0.05]})
    inc = make_incidence({k: list(prof[k]) for k in prof.columns}, n=250)
    cal = sim.calibration_table(prof, inc, draws=40, seed=sim.SEED)
    ordered = cal.sort_values("observed")
    assert ordered.calibrated.is_monotonic_increasing


# ---------------------------------------------------------------------------
# The bootstrap resamples postings
# ---------------------------------------------------------------------------


def test_bootstrap_width_tracks_posting_count():
    """The defect this pins: resampling shares instead of postings.

    Share-level resampling would give an interval that barely moves with the
    number of postings behind it. Posting-level resampling has a signature:
    the width follows 1/sqrt(n), so sixteen times the postings must give
    about four times the precision. The test brackets that ratio rather than
    demanding "wider", because a one-sided check would also pass for a
    bootstrap that is merely noisy.
    """
    rates = {"a": [0.6, 0.4, 0.2, 0.05] * 8,
             "b": [0.5, 0.45, 0.25, 0.05] * 8,
             "c": [0.2, 0.1, 0.6, 0.3] * 8}
    thin = sim.bootstrap_pairs(make_incidence(rates, n=25, seed=3),
                               draws=80, seed=sim.SEED)
    thick = sim.bootstrap_pairs(make_incidence(rates, n=400, seed=3),
                                draws=80, seed=sim.SEED)
    thin_width = float((thin.ci_high - thin.ci_low).mean())
    thick_width = float((thick.ci_high - thick.ci_low).mean())
    ratio = thin_width / thick_width
    assert 2.5 < ratio < 6.0, (ratio, thin_width, thick_width)


def test_bootstrap_interval_contains_the_observed_value():
    inc = make_incidence({"a": [0.6, 0.3, 0.1] * 8, "b": [0.5, 0.35, 0.1] * 8,
                          "c": [0.1, 0.7, 0.3] * 8}, n=200)
    boot = sim.bootstrap_pairs(inc, draws=100, seed=sim.SEED)
    assert ((boot.ci_low <= boot.observed) & (boot.observed <= boot.ci_high)).all()


def test_rank_is_published_only_above_the_stability_floor():
    inc = make_incidence({"a": [0.6, 0.3, 0.1] * 8, "b": [0.55, 0.32, 0.1] * 8,
                          "c": [0.1, 0.7, 0.3] * 8}, n=150)
    boot = sim.bootstrap_pairs(inc, draws=100, seed=sim.SEED)
    assert (boot.rank_identified == (boot.rank_stability >= sim.RANK_STABILITY_FLOOR)).all()
    assert sim.RANK_STABILITY_FLOOR == 0.90


def test_overlapping_rank_intervals_collapse_into_one_tier():
    """Fifteen ranks imply fourteen distinctions; most do not survive."""
    boot = pd.DataFrame({
        "company_a": ["a", "a", "b", "b"], "company_b": ["b", "c", "c", "d"],
        "observed": [0.9, 0.8, 0.7, 0.4], "rank": [1, 2, 3, 4],
        "rank_low": [1, 2, 2, 4], "rank_high": [1, 3, 3, 4],
        "rank_stability": [1.0, 0.5, 0.5, 1.0],
    })
    tiers = sim.rank_tiers(boot)
    assert list(tiers.tier) == [1, 2, 2, 3]


# ---------------------------------------------------------------------------
# Structure: the tree and the network
# ---------------------------------------------------------------------------


def test_linkage_merges_the_closest_pair_first():
    dist = pd.DataFrame(
        [[0.0, 0.1, 0.8, 0.9],
         [0.1, 0.0, 0.7, 0.9],
         [0.8, 0.7, 0.0, 0.2],
         [0.9, 0.9, 0.2, 0.0]],
        index=list("abcd"), columns=list("abcd"))
    merges = sim.average_linkage(dist)
    assert merges[0][0] | merges[0][1] == frozenset({"a", "b"})
    heights = [h for _, _, h in merges]
    assert heights == sorted(heights), "average linkage heights must not invert"


def test_cuts_of_the_same_tree_nest():
    """A k+1 partition must refine the k partition, or the tree is not a tree."""
    dist = pd.DataFrame(
        [[0.0, 0.1, 0.8, 0.9, 0.85],
         [0.1, 0.0, 0.7, 0.9, 0.80],
         [0.8, 0.7, 0.0, 0.2, 0.30],
         [0.9, 0.9, 0.2, 0.0, 0.25],
         [0.85, 0.80, 0.30, 0.25, 0.0]],
        index=list("abcde"), columns=list("abcde"))
    keys = list("abcde")
    merges = sim.average_linkage(dist)
    for k in range(1, len(keys)):
        coarse, fine = sim.cut_tree(merges, keys, k), sim.cut_tree(merges, keys, k + 1)
        for cluster in fine:
            assert any(cluster <= parent for parent in coarse), (k, cluster)
    assert sim.cut_tree(merges, keys, 1) == frozenset({frozenset(keys)})
    assert sim.cut_tree(merges, keys, len(keys)) == frozenset(
        frozenset([k]) for k in keys)


def test_network_edges_never_increase_with_the_threshold():
    matrix = pd.DataFrame(
        [[1.0, 0.9, 0.6], [0.9, 1.0, 0.5], [0.6, 0.5, 1.0]],
        index=list("abc"), columns=list("abc"))
    sweep = sim.network_thresholds(matrix)
    assert list(sweep.edges) == sorted(sweep.edges, reverse=True)
    assert sweep.possible_edges.unique().tolist() == [3]


def test_the_published_network_has_no_plateau_to_choose():
    """The sweep is published because no single picture is defensible."""
    sweep = committed("network-thresholds")
    complete = int((sweep.edges == sweep.possible_edges).sum())
    empty = int((sweep.edges == 0).sum())
    assert complete >= 1 and empty >= 1, "the sweep must reach both extremes"
    assert int(sweep.groupby("edges").size().max()) <= 5, (
        "a long plateau would mean a defensible threshold exists, and the "
        "report says one does not")


# ---------------------------------------------------------------------------
# Closure — the correction this module took
# ---------------------------------------------------------------------------


def test_geometric_closure_reproduces_the_analytic_value():
    """`-1/(D-1)` is exact for the closure it was derived under.

    Divide each period by its geometric mean — the CLR transform — and the
    mean pairwise correlation of independent parts is the analytic value.
    """
    rng = np.random.default_rng(sim.SEED)
    for d in (4, 6, 10):
        logs = rng.normal(0.0, 0.3, size=(4000, d))
        clr = logs - logs.mean(axis=1, keepdims=True)
        r = np.mean([np.corrcoef(clr[:, i], clr[:, j])[0, 1]
                     for i in range(d) for j in range(i + 1, d)])
        assert r == pytest.approx(sim.closure_expectation(d), abs=5e-3), d


def test_arithmetic_closure_falls_short_of_the_analytic_value():
    """And a share table is closed arithmetically, which is a different map.

    The gap always runs toward zero, so quoting `-1/(D-1)` as the null for
    shares overstates how negative independence looks and makes a co-movement
    easier to claim than it should be.
    """
    for d in (4, 6, 10):
        null = sim.closure_null(600, d, sigma=0.3, draws=60, seed=sim.SEED)
        gap = null["mean"] - sim.closure_expectation(d)
        assert 0.0 < gap < 0.05, (d, gap)


def test_calibrate_sigma_recovers_the_dispersion_it_is_given():
    for true_sigma in (0.3, 0.9):
        wide = closed_panel(true_sigma)
        assert sim.calibrate_sigma(wide, seed=sim.SEED) == pytest.approx(
            true_sigma, rel=0.05)


def test_the_closure_band_widens_with_dispersion():
    """Which is why the published band is calibrated rather than defaulted."""
    narrow = sim.closure_null(11, 6, sigma=0.2, draws=200, seed=sim.SEED)
    wide = sim.closure_null(11, 6, sigma=1.2, draws=200, seed=sim.SEED)
    assert (wide["p97_5"] - wide["p2_5"]) > 4 * (narrow["p97_5"] - narrow["p2_5"])
    assert wide["mean"] > narrow["mean"], "the centre climbs toward zero too"


def test_the_published_closure_clause_holds_at_the_panels_own_dispersion():
    sens = committed("closure-sensitivity")
    calibrated = sens[sens.sigma.between(0.85, 0.95)]
    assert len(calibrated) == 1, "the calibrated sigma must appear in the sweep"
    assert bool(calibrated.inside.iloc[0])
    assert (sens.gap_to_analytic > 0).all(), "the gap always runs toward zero"


def test_aitchison_variation_is_zero_for_proportional_series():
    """Scale invariance is the property the whole log-ratio detour buys."""
    base = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    wide = pd.DataFrame({"a": base, "b": base * 3.0, "c": base ** 2})
    var = sim.aitchison_variation(wide)
    assert var.loc["a", "b"] == pytest.approx(0.0, abs=1e-12)
    assert var.loc["a", "c"] > 0.0
    assert var.loc["a", "b"] == pytest.approx(var.loc["b", "a"])


# ---------------------------------------------------------------------------
# The trajectory refusal
# ---------------------------------------------------------------------------


def test_panel_wide_never_fills_a_missing_period():
    """February is absent from the panel and must stay absent.

    The index is the intersection of observed periods, so a company missing a
    month shortens the panel for everyone rather than acquiring an
    interpolated value.
    """
    rows = []
    for key, periods in {"a": ["2023-01", "2023-02", "2023-03"],
                         "b": ["2023-01", "2023-03"]}.items():
        for period in periods:
            rows.append({"key": key, "period": period, "numerator": 10,
                         "denominator": 100, "share": 0.1, "is_observed": True})
    wide = sim.panel_wide(pd.DataFrame(rows))
    assert list(wide.index) == ["2023-01", "2023-03"]
    assert not wide.isna().any().any()


def test_trajectory_is_refused_and_says_why():
    traj = committed("trajectory-similarity")
    null = sim.closure_null(11, 6, sigma=0.89, draws=200, seed=sim.SEED)
    verdict = sim.trajectory_verdict(traj, null)
    assert verdict.identified is False
    assert verdict.reason.count(";") >= 2, (
        "the refusal has three independent reasons and reports all of them, "
        "so it cannot be read as a quirk of one threshold")
    assert verdict.detail["closure_sigma"] == pytest.approx(0.89)


def test_the_task_07_gate_applies_to_both_members_of_a_pair():
    traj = committed("trajectory-similarity")
    assert (traj.eligible == (traj.gate_a & traj.gate_b)).all()
    assert int(traj.eligible.sum()) < len(traj), (
        "if every pair were eligible the gate would not be doing anything")


# ---------------------------------------------------------------------------
# What the task actually publishes
# ---------------------------------------------------------------------------


def test_cosine_is_the_metric_least_like_the_rank_metrics():
    """`PRIMARY_METRIC` is the outlier, and the report has to say so.

    The two families are not the clean split the labels imply — the lowest
    within-family correlation and the highest cross-family one differ by
    0.007. What separates is cosine alone.
    """
    conc = committed("metric-concordance")
    idx = conc.set_index(["metric_a", "metric_b"])
    cosine_vs_ranks = [idx.loc[k, "rank_correlation"] for k in
                       [("cosine", "spearman"), ("cosine", "jaccard_supported")]]
    assert max(cosine_vs_ranks) < 0.1, cosine_vs_ranks
    others = conc[(~conc.same_family)
                  & (conc.metric_a != "cosine") & (conc.metric_b != "cosine")]
    assert others.rank_correlation.min() > 0.4, (
        "the other prevalence metrics do agree with the rank metrics, so the "
        "disagreement belongs to cosine and not to the family divide")


def test_most_pairs_are_unresolved():
    """The shape of the result, pinned so a refactor cannot quietly improve it."""
    verdicts = committed("pair-verdicts")
    counts = verdicts.verdict.value_counts()
    assert counts.get("unresolved", 0) > len(verdicts) / 2
    assert counts.get("robust", 0) >= 1
    assert (verdicts[verdicts.verdict == "unresolved"].rank_stability
            < sim.RANK_STABILITY_FLOOR).all()


def test_own_products_move_the_ranking_more_than_role_mix_does():
    """The opposite of Task 06's skill-level result, and the reason both sweeps
    are published rather than only the one Task 06 predicted would matter."""
    vend = committed("vendor-sensitivity")
    mix = committed("mix-sensitivity")
    assert vend.rank_move.abs().max() > mix.rank_move.abs().max()
    assert vend.delta_all_products.abs().max() > mix.mix_effect.abs().max()


def test_a_rare_skill_cannot_carry_a_cosine_numerator():
    """Task 04 §6.1 predicted it would; the arithmetic says otherwise.

    A skill's contribution to a cosine on share vectors is the product of two
    shares. One posting in six hundred contributes a millionth of what a
    skill in half the postings does, however many columns it occupies.
    """
    prof = make_profile({"a": [0.5, 0.4, 1 / 600], "b": [0.45, 0.35, 1 / 600]})
    contrib = sim.numerator_contribution(
        prof, {"rare": ["s2"], "common": ["s0", "s1"]}, top_k=2)
    row = contrib.iloc[0]
    assert row.share_rare < 1e-5
    assert row.share_common > 0.999


def test_vendor_skills_are_a_sensitivity_not_a_filter():
    """Task 06 §7 settled this: dropping them silently would hide the lever.

    The primary pair table is computed on the whole vocabulary; the sweep
    reports what happens without own products as a separate column, so the
    0.30 swing is visible rather than absorbed.
    """
    vend = committed("vendor-sensitivity")
    pt = committed("similarity-pairs")
    merged = pt.merge(vend, on=["company_a", "company_b"])
    assert (merged.cosine == merged.all_skills).all(), (
        "the published cosine must be the all-skills one")
    assert merged.delta_all_products.abs().max() > 0.1


# ---------------------------------------------------------------------------
# Standing guards, inherited
# ---------------------------------------------------------------------------


def test_no_forbidden_column_reaches_a_task_08_table():
    for path in sorted(TABLES.glob("*.csv")):
        assert sim.forbidden_columns(pd.read_csv(path, nrows=1)) == [], path.name


def test_no_personal_data_column_reaches_a_task_08_table():
    for path in sorted(TABLES.glob("*.csv")):
        assert sim.personal_data_columns_present(pd.read_csv(path, nrows=1)) == [], path.name


def test_forbidden_column_check_still_bites():
    assert sim.forbidden_columns(pd.DataFrame({"country": ["UK"]})) == ["country"]
    assert sim.personal_data_columns_present(
        pd.DataFrame({"recruiter_email": ["a@b.c"]})) != []


def test_similarity_module_imports_no_heavy_stats_dependency():
    """The no-scipy promise, pinned at the source.

    `validate_similarity.py` cross-checks this module against scipy and
    scikit-learn precisely because this module does not use them, and that
    argument collapses the moment someone adds a convenience import here. The
    check is on the source text rather than on `sys.modules`, because pytest
    itself pulls scipy in through other suites.
    """
    source = Path(sim.__file__).read_text()
    for banned in ("scipy", "statsmodels", "sklearn", "networkx"):
        assert f"import {banned}" not in source, (
            f"similarity.py imports {banned}; the module is meant to run in an "
            "environment that has none of them, and validate_similarity.py's "
            "cross-check is only meaningful while that stays true")


def test_the_incidence_denominator_is_inherited_not_reinvented(real_incidence):
    """Task 06's `skill_denominator`, so facilities roles stay excluded.

    Google's facilities share is ten times Snowflake's; a denominator that
    kept those postings would make the similarity matrix partly a map of who
    runs data centres.
    """
    inc, skills = real_incidence
    frames = cmp.load_frames(COMPANIES)
    for key in COMPANIES:
        assert inc[key].shape[0] == len(cmp.skill_denominator(frames[key]).job_id.unique())
        assert inc[key].shape[1] == len(skills)
        assert set(np.unique(inc[key])) <= {0, 1}
