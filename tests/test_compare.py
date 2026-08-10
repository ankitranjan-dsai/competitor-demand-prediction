"""Tests for the Task 06 cross-company comparison layer.

Same intent as the other suites: every case here is either a trap that was
actually hit in the real six-company data, or a rule the other three
specialists depend on, locked in so a later refactor cannot quietly undo it.

The ones that matter most:

* **the relative-share estimator cancels a shared collection factor.** All six
  companies come through one crawler, so all six show H2 > H1 and none of that
  is evidence. If both companies double, their shares of the same
  publisher-period pool must not move. That single property is why Task 06 has
  a volume finding at all, and it is pinned in
  ``test_relative_share_cancels_a_common_multiplicative_factor``;
* **a level comparison is not identified and must say so.** ``level_verdict``
  returns False whenever any company reaches the common panel with too few of
  its postings, and it names the company;
* **role mix can manufacture an entire difference.** Two companies with the
  same within-function rates and opposite mixes differ by 48 points crude and
  by nothing standardised;
* **Simpson's paradox across companies**, not just across time: a pooled gap
  that reverses inside every job function;
* **log-lift stays finite and informative for a skill a company never asks
  for.** An epsilon in the denominator collapses every absent skill onto the
  same number and the axis becomes a list of denominators;
* **self-reference is flagged, never filtered** — Snowflake leads the sector on
  the skill "Snowflake", and a top-ten list must not quietly be that;
* the two standing checks: no ``country``/``share_of_all`` column and no
  personal-data column reaches a committed table.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compare as cmp  # noqa: E402
from compare import (  # noqa: E402
    FORBIDDEN_COLUMNS,
    MIN_CELL,
    MIN_COMMON_PANEL_SHARE,
    PUBLISHER_COL,
    VENDOR_SKILLS,
    annotate_self_reference,
    bh_fdr,
    collection_artefact_table,
    common_panel_by_period,
    common_panel_table,
    common_publishers,
    comparability_table,
    company_correlation,
    forbidden_columns,
    half_over_half,
    level_verdict,
    mix_distance,
    mix_table,
    newcombe_diff_interval,
    own_channel_publishers,
    pairwise_skill_gap,
    panel_robustness,
    personal_data_columns_present,
    pooled_weights,
    relative_share_by_publisher,
    relative_share_table,
    relative_share_verdict,
    restrict,
    self_reference_table,
    skill_denominator,
    skill_distinctiveness,
    skill_matrix,
    skill_share_table,
    stack,
    standardise,
    standardised_skill_table,
    standardised_table,
    stratified_company_verdict,
    two_proportion_p,
    volume_verdict_table,
    wilson_interval,
    within_stratum_shares,
)

_ids = itertools.count()


# ---------------------------------------------------------------------------
# Fixtures — Task 03-shaped frames, built the way test_trends.py builds them
# ---------------------------------------------------------------------------


def make_postings(dates, publishers=None, functions=None, skilled=None,
                  job_ids=None) -> pd.DataFrame:
    """A minimal Task 03-shaped frame: dates plus the columns compare.py reads."""
    dates = pd.to_datetime(pd.Series(list(dates)))
    n = len(dates)
    iso = dates.dt.isocalendar()
    return pd.DataFrame({
        "job_id": job_ids or [f"j{next(_ids)}" for _ in range(n)],
        "posting_date": dates.dt.strftime("%Y-%m-%d"),
        "posting_month": dates.dt.strftime("%Y-%m"),
        "posting_week": [f"{y}-W{w:02d}" for y, w in zip(iso.year, iso.week)],
        "posting_quarter": [f"{d.year}-Q{(d.month - 1) // 3 + 1}" for d in dates],
        "job_via": publishers if publishers is not None else ["via A"] * n,
        "job_function": functions if functions is not None else ["Engineering"] * n,
        "has_any_skill": skilled if skilled is not None else [True] * n,
        "skill_count_final": [2] * n,
    })


def panel_frames(spec: dict) -> dict[str, pd.DataFrame]:
    """Frames from ``{company: {publisher: (h1_count, h2_count)}}``.

    Half-year counts are what the relative-share estimator reads, so the
    fixture is written in the estimator's own units rather than in dates.
    """
    frames = {}
    for company, pubs in spec.items():
        dates, publishers = [], []
        for pub, (h1, h2) in pubs.items():
            dates += ["2023-03-15"] * h1 + ["2023-09-15"] * h2
            publishers += [pub] * (h1 + h2)
        frames[company] = make_postings(dates, publishers=publishers)
    return frames


def make_long(hits: dict[str, list[str]], category: str = "analyst_tools",
              provenance: str = "source") -> pd.DataFrame:
    """Skills long table from ``{skill: [job_id, ...]}``."""
    rows = [{"job_id": jid, "skill": skill, "skill_category": category,
             "provenance": provenance}
            for skill, ids in hits.items() for jid in ids]
    return pd.DataFrame(rows, columns=["job_id", "skill", "skill_category",
                                       "provenance"])


def monthly(company_months: dict[str, int], publisher: str = "via A",
            **kwargs) -> pd.DataFrame:
    """``{month: n}`` -> one frame, all on one publisher."""
    dates = [f"2023-{m}-15" for m, n in company_months.items() for _ in range(n)]
    return make_postings(dates, publishers=[publisher] * len(dates), **kwargs)


# ---------------------------------------------------------------------------
# Uncertainty, without scipy
# ---------------------------------------------------------------------------


def test_wilson_does_not_give_a_three_posting_cell_a_zero_width_interval():
    # Wald gives 3/3 the interval [1.0, 1.0], which would let a three-posting
    # cell outrank a three-hundred-posting one in every skill ranking.
    lo, hi = wilson_interval(3, 3)
    assert lo < 0.9 and hi == 1.0


def test_wilson_narrows_as_the_denominator_grows():
    small = wilson_interval(30, 100)
    large = wilson_interval(300, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_stays_inside_the_unit_interval():
    for k, n in ((0, 5), (5, 5), (1, 1000), (999, 1000)):
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_returns_nan_rather_than_dividing_by_an_empty_denominator():
    lo, hi = wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_newcombe_interval_contains_the_observed_difference():
    lo, hi = newcombe_diff_interval(30, 100, 10, 100)
    assert lo < 0.20 < hi
    assert -1.0 <= lo and hi <= 1.0


def test_newcombe_survives_a_tiny_cell_on_one_side():
    # The normal case here, not the exception: one company with 3 postings
    # mentioning a skill and another with 300.
    lo, hi = newcombe_diff_interval(3, 3, 150, 300)
    assert not math.isnan(lo) and not math.isnan(hi)
    assert hi - lo > 0.3   # a 3-posting cell must not look precise


def test_two_proportion_p_is_one_for_identical_shares():
    assert two_proportion_p(50, 100, 25, 50) == pytest.approx(1.0, abs=1e-9)


def test_two_proportion_p_is_small_for_a_large_clean_difference():
    assert two_proportion_p(90, 100, 10, 100) < 1e-6


def test_bh_is_stricter_than_an_uncorrected_threshold():
    # 91 skills x 6 companies makes roughly 27 "findings" from noise alone at
    # an uncorrected 5%, which is more than the report has room for.
    p = [0.001, 0.04, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    out = bh_fdr(p)
    assert sum(x <= 0.05 for x in p) == 2      # uncorrected would call two
    assert bool(out.significant.iloc[0]) is True
    assert int(out.significant.sum()) == 1     # BH calls one


def test_bh_q_values_are_monotone_in_p():
    out = bh_fdr([0.5, 0.001, 0.2, 0.04])
    ordered = out.sort_values("p_value")
    assert ordered.q_value.is_monotonic_increasing
    assert (ordered.q_value >= ordered.p_value - 1e-12).all()


def test_bh_ignores_missing_p_values_without_dropping_the_rows():
    out = bh_fdr([0.001, float("nan"), 0.9])
    assert len(out) == 3
    assert math.isnan(out.q_value.iloc[1])
    assert bool(out.significant.iloc[1]) is False


# ---------------------------------------------------------------------------
# The comparability gate
# ---------------------------------------------------------------------------


def test_own_channel_publishers_recognises_the_company_as_its_own_publisher():
    # An own-channel posting is observed because the employer published it,
    # not because an aggregator picked it up: the least comparable channel
    # there is (19.1% skill coverage against 82.6% elsewhere).
    df = make_postings(["2023-01-15"] * 3,
                       publishers=["via Snowflake Careers", "via LinkedIn",
                                   "via Indeed"])
    assert own_channel_publishers(df, "snowflake") == ["via Snowflake Careers"]


def test_comparability_gate_reports_one_row_per_company_with_the_level_flag():
    # Real registry keys: the gate asks each company which publishers are its
    # own careers site, and that question only has an answer for a registered
    # company.
    frames = {
        "google": monthly({f"{m:02d}": 10 for m in range(1, 13)}, "via A"),
        "meta": monthly({f"{m:02d}": 10 for m in range(1, 13)}, "via A"),
    }
    gate = comparability_table(frames)
    assert list(gate.company) == ["google", "meta"]
    # One shared publisher carrying everything: the level *is* comparable here.
    assert (gate.common_panel_share == 1.0).all()
    assert gate.level_comparable.all()


def test_level_comparison_is_refused_when_one_company_barely_reaches_the_panel():
    # The real case: Snowflake reaches the common panel with 23.5% of its
    # postings, so "company A posts more than company B" is a statement about
    # syndication deals.
    gate = pd.DataFrame({"company": ["google", "snowflake"],
                         "common_panel_share": [0.68, 0.235]})
    verdict = level_verdict(gate)
    assert verdict.identified is False
    assert "snowflake" in verdict.reason
    assert verdict.detail["snowflake"] == pytest.approx(0.235)


def test_level_comparison_is_allowed_only_when_every_company_clears_the_floor():
    gate = pd.DataFrame({"company": ["a", "b"],
                         "common_panel_share": [MIN_COMMON_PANEL_SHARE, 0.9]})
    assert level_verdict(gate).identified is True


def test_the_common_panel_floor_is_a_declared_constant():
    # A floor chosen after seeing which companies fail is not a floor.
    assert 0.0 < MIN_COMMON_PANEL_SHARE < 1.0


# ---------------------------------------------------------------------------
# The common publisher panel
# ---------------------------------------------------------------------------


def test_common_publishers_is_the_intersection_not_the_union():
    frames = {
        "a": make_postings(["2023-01-15"] * 3,
                           publishers=["via A", "via B", "via C"]),
        "b": make_postings(["2023-01-15"] * 3,
                           publishers=["via B", "via C", "via D"]),
        "c": make_postings(["2023-01-15"] * 2, publishers=["via C", "via E"]),
    }
    assert common_publishers(frames) == ["via C"]


def test_the_window_level_intersection_flatters_the_data():
    # A publisher can serve one company in March and another in September and
    # still count as "common" over the year. Within a month the intersection
    # collapses, and a month at zero supports no like-for-like comparison.
    frames = {
        "a": make_postings(["2023-03-15"], publishers=["via A"]),
        "b": make_postings(["2023-09-15"], publishers=["via A"]),
    }
    assert common_publishers(frames) == ["via A"]
    by_period = common_panel_by_period(frames)
    assert set(by_period.n_common_publishers) == {0}


def test_common_panel_table_reports_what_share_of_a_company_the_panel_carries():
    frames = {
        "a": make_postings(["2023-01-15"] * 4,
                           publishers=["via A", "via A", "via A", "via Z"]),
        "b": make_postings(["2023-01-15"] * 2, publishers=["via A", "via Y"]),
    }
    panel = common_panel_table(frames)
    a_row = panel[(panel.company == "a") & (panel.publisher == "via A")].iloc[0]
    assert a_row.postings == 3
    assert a_row.share_of_company_postings == pytest.approx(0.75)


def test_restrict_keeps_only_the_named_publishers():
    frames = {"a": make_postings(["2023-01-15"] * 3,
                                 publishers=["via A", "via B", "via A"])}
    cut = restrict(frames, ["via A"])
    assert len(cut["a"]) == 2
    assert set(cut["a"][PUBLISHER_COL]) == {"via A"}


def test_stack_labels_every_row_with_its_company():
    frames = {"a": make_postings(["2023-01-15"] * 2),
              "b": make_postings(["2023-01-15"] * 3)}
    stacked = stack(frames)
    assert stacked.company.value_counts().to_dict() == {"b": 3, "a": 2}


# ---------------------------------------------------------------------------
# Volume: trends, not levels
# ---------------------------------------------------------------------------


def test_volume_verdict_compares_verdicts_not_magnitudes():
    # Each company's index is built on its own publisher panel, so the indices
    # are not on a common scale and only the direction is comparable.
    frames = {
        "a": monthly({f"{m:02d}": 10 + m for m in range(1, 13)}),
        "b": monthly({f"{m:02d}": 24 - m for m in range(1, 13)}),
    }
    out = volume_verdict_table(frames)
    assert list(out.company) == ["a", "b"]
    assert set(out.columns) >= {"direction", "treatments_agree", "spread"}


def test_half_over_half_flags_whether_raw_and_balanced_agree():
    frames = {"a": monthly({f"{m:02d}": (5 if m <= 6 else 15)
                            for m in range(1, 13)})}
    out = half_over_half(frames).iloc[0]
    assert out.raw_h1 == 30 and out.raw_h2 == 90
    assert out.raw_pct_change == pytest.approx(200.0)
    assert bool(out.agrees) is True


def test_a_thick_half_with_a_shrinking_publisher_roster_is_not_coverage_growth():
    # The argument the whole half-over-half result rests on. Six independent
    # employers cannot make job boards appear; a crawler widening its coverage
    # shows *more publishers*. Here publishers fall while postings per
    # publisher rise, which is the one direction a coverage artefact cannot
    # fake.
    jan = make_postings(["2023-01-15"] * 12,
                        publishers=[f"via P{i}" for i in range(6)] * 2)
    feb = make_postings(["2023-02-15"] * 24,
                        publishers=[f"via P{i}" for i in range(3)] * 8)
    frames = {"a": pd.concat([jan, feb], ignore_index=True)}
    out = collection_artefact_table(frames).set_index("period")
    assert out.loc["2023-01", "distinct_publishers"] == 6
    assert out.loc["2023-02", "distinct_publishers"] == 3
    assert out.loc["2023-02", "postings_per_publisher"] > \
        out.loc["2023-01", "postings_per_publisher"]


def test_a_month_every_company_is_thin_in_with_a_thin_roster_is_the_collector():
    def company(publisher_offset: int) -> pd.DataFrame:
        parts = []
        for month, (n_pub, per_pub) in {"01": (4, 2), "02": (1, 1),
                                        "03": (4, 2)}.items():
            pubs = [f"via P{publisher_offset + i}" for i in range(n_pub)] * per_pub
            parts.append(make_postings([f"2023-{month}-15"] * len(pubs),
                                       publishers=pubs))
        return pd.concat(parts, ignore_index=True)

    frames = {"a": company(0), "b": company(0)}
    out = collection_artefact_table(frames).set_index("period")
    assert bool(out.loc["2023-02", "shared_thin_month"]) is True
    assert bool(out.loc["2023-01", "shared_thin_month"]) is False


def test_company_correlation_is_a_square_matrix_with_a_unit_diagonal():
    frames = {"a": monthly({f"{m:02d}": m for m in range(1, 13)}),
              "b": monthly({f"{m:02d}": 13 - m for m in range(1, 13)})}
    corr = company_correlation(frames)
    assert list(corr.columns) == list(corr.index) == ["a", "b"]
    assert corr.loc["a", "a"] == 1.0
    assert corr.loc["a", "b"] < 0   # opposite series must read as opposite


# ---------------------------------------------------------------------------
# The relative-share estimator — Task 06's one volume finding
# ---------------------------------------------------------------------------


def test_relative_share_cancels_a_common_multiplicative_factor():
    # The property the estimator exists for. Both companies double between the
    # halves — exactly what one crawler ramping up produces — and neither may
    # be reported as gaining share.
    frames = panel_frames({"alpha": {"via A": (20, 40)},
                           "beta": {"via A": (10, 20)}})
    out = relative_share_table(frames).set_index("company")
    assert out.loc["alpha", "share_change_pp"] == pytest.approx(0.0, abs=0.05)
    assert out.loc["beta", "share_change_pp"] == pytest.approx(0.0, abs=0.05)
    assert abs(out.loc["alpha", "log_share_change"]) < 0.02
    assert abs(out.loc["beta", "log_share_change"]) < 0.02


def test_relative_share_reports_the_company_that_outgrew_the_pool():
    # alpha flat, beta quadrupled: beta gains share, alpha loses it, and the
    # two moves are equal and opposite because they are shares of one pool.
    frames = panel_frames({"alpha": {"via A": (20, 20)},
                           "beta": {"via A": (10, 40)}})
    out = relative_share_table(frames).set_index("company")
    assert out.loc["beta", "log_share_change"] > 0
    assert out.loc["alpha", "log_share_change"] < 0
    assert out.loc["alpha", "share_change_pp"] == \
        pytest.approx(-out.loc["beta", "share_change_pp"], abs=0.05)


def test_relative_share_survives_a_company_that_vanishes_in_one_half():
    # +0.5 rather than an epsilon: an empty half has to give a finite number,
    # or the figure's axis is set by whichever company disappeared.
    frames = panel_frames({"alpha": {"via A": (20, 20)},
                           "beta": {"via A": (10, 0)}})
    out = relative_share_table(frames).set_index("company")
    assert np.isfinite(out.loc["beta", "log_share_change"])
    assert out.loc["beta", "log_share_change"] < 0


def test_relative_share_is_computed_on_shared_publishers_only():
    # A publisher only one company uses is a distribution deal, not a channel.
    frames = panel_frames({"alpha": {"via A": (10, 10), "via Solo": (0, 100)},
                           "beta": {"via A": (10, 10)}})
    out = relative_share_table(frames).set_index("company")
    assert out.attrs["publishers"] == ["via A"]
    assert out.loc["alpha", "h2_postings"] == 10


def test_a_relative_move_is_confirmed_only_when_every_publisher_agrees():
    spec = {pub: (5, 15) for pub in ("via A", "via B", "via C")}
    frames = panel_frames({
        "alpha": spec,
        "beta": {pub: (10, 5) for pub in ("via A", "via B", "via C")},
    })
    pooled = relative_share_table(frames)
    by_pub = relative_share_by_publisher(frames)
    verdict = relative_share_verdict(by_pub, pooled).set_index("company")
    assert verdict.loc["alpha", "publishers_tested"] == 3
    assert verdict.loc["alpha", "verdict"] == "confirmed"
    assert verdict.loc["alpha", "direction"] == "gaining share"
    assert verdict.loc["beta", "direction"] == "losing share"


def test_a_relative_move_that_flips_on_one_publisher_is_only_mixed():
    # The identifying assumption — constant syndication propensity between the
    # halves — is not testable, but it is checkable in this one direction.
    frames = panel_frames({
        "alpha": {"via A": (5, 15), "via B": (5, 15), "via C": (15, 5)},
        "beta": {"via A": (10, 5), "via B": (10, 5), "via C": (5, 15)},
    })
    pooled = relative_share_table(frames)
    by_pub = relative_share_by_publisher(frames)
    verdict = relative_share_verdict(by_pub, pooled).set_index("company")
    assert verdict.loc["alpha", "publishers_agreeing"] == 2
    assert verdict.loc["alpha", "verdict"] == "mixed"


def test_a_publisher_too_thin_to_carry_a_half_is_dropped():
    # Its share change is one or two postings moving.
    frames = panel_frames({"alpha": {"via A": (20, 20), "via Thin": (2, 2)},
                           "beta": {"via A": (20, 20), "via Thin": (1, 1)}})
    by_pub = relative_share_by_publisher(frames)
    assert set(by_pub.publisher) == {"via A"}


# ---------------------------------------------------------------------------
# Role mix and direct standardisation
# ---------------------------------------------------------------------------


def test_mix_distance_is_zero_for_identical_mixes_and_one_for_disjoint_ones():
    same = {"a": make_postings(["2023-01-15"] * 2,
                               functions=["Engineering", "Analytics"]),
            "b": make_postings(["2023-01-15"] * 4,
                               functions=["Engineering", "Engineering",
                                          "Analytics", "Analytics"])}
    assert mix_distance(mix_table(same)).loc["a", "b"] == pytest.approx(0.0)

    apart = {"a": make_postings(["2023-01-15"] * 2,
                                functions=["Engineering"] * 2),
             "b": make_postings(["2023-01-15"] * 2, functions=["Analytics"] * 2)}
    assert mix_distance(mix_table(apart)).loc["a", "b"] == pytest.approx(1.0)


def test_standardisation_removes_a_difference_that_was_entirely_role_mix():
    # Same rate inside every function, opposite mixes. The crude gap is 48
    # points and it is not about the companies at all.
    def company(engineering: int, analytics: int) -> pd.DataFrame:
        functions = ["Engineering"] * engineering + ["Analytics"] * analytics
        skilled = ([True] * int(engineering * 0.8)
                   + [False] * (engineering - int(engineering * 0.8))
                   + [True] * int(analytics * 0.2)
                   + [False] * (analytics - int(analytics * 0.2)))
        return make_postings(["2023-01-15"] * len(functions),
                             functions=functions, skilled=skilled)

    frames = {"a": company(90, 10), "b": company(10, 90)}
    out = standardised_table(frames, "has_any_skill").set_index("company")
    assert out.loc["a", "crude"] - out.loc["b", "crude"] == pytest.approx(0.48)
    assert out.loc["a", "standardised"] == pytest.approx(out.loc["b", "standardised"])
    assert out.loc["a", "mix_effect"] > 0.2


def test_pooled_weights_are_the_pooled_mix_so_no_company_is_the_reference():
    frames = {"a": make_postings(["2023-01-15"] * 3, functions=["Engineering"] * 3),
              "b": make_postings(["2023-01-15"], functions=["Analytics"])}
    w = pooled_weights(frames)
    assert w["Engineering"] == pytest.approx(0.75)
    assert w.sum() == pytest.approx(1.0)


def test_standardisation_drops_thin_strata_and_says_how_much_weight_it_kept():
    # A 2-posting stratum must not be allowed to swing the answer, and the
    # reader has to be told what fraction of the standard population survived.
    functions = ["Engineering"] * 40 + ["Tiny"] * 2
    df = make_postings(["2023-01-15"] * 42, functions=functions,
                       skilled=[True] * 40 + [False] * 2)
    weights = pd.Series({"Engineering": 0.6, "Tiny": 0.4}).rename_axis("job_function")
    out = standardise(df, "has_any_skill", weights)
    assert out["strata_used"] == 1
    assert out["weight_covered"] == pytest.approx(0.6)
    assert out["standardised"] == pytest.approx(1.0)


def test_standardise_reports_nan_rather_than_guessing_when_no_stratum_qualifies():
    df = make_postings(["2023-01-15"] * 3)
    out = standardise(df, "has_any_skill", pooled_weights({"a": df}))
    assert math.isnan(out["standardised"])
    assert out["weight_covered"] == 0.0


# ---------------------------------------------------------------------------
# Skills — denominators first
# ---------------------------------------------------------------------------


def test_the_skill_denominator_is_skilled_postings_without_facilities():
    # Both filters are inherited, not chosen here, and the facilities share
    # differs tenfold across companies (11.2% Google, 0% Snowflake), so
    # skipping either makes the cross-company comparison worse than the
    # within-company one.
    df = make_postings(["2023-01-15"] * 5,
                       functions=["Engineering", "Engineering",
                                  "Facilities / Operations",
                                  "Facilities / Operations", "Analytics"],
                       skilled=[True, False, True, True, True])
    denom = skill_denominator(df)
    assert len(denom) == 2
    assert set(denom.job_function) == {"Engineering", "Analytics"}


def test_skill_shares_are_over_skilled_postings_and_never_over_all_postings():
    df = make_postings(["2023-01-15"] * 10,
                       functions=["Engineering"] * 6 + ["Facilities / Operations"] * 4,
                       skilled=[True] * 5 + [False] + [True] * 4)
    long = make_long({"SQL": list(df.job_id[:5])})
    share = skill_share_table({"a": long}, {"a": df}, min_postings=1)
    row = share.iloc[0]
    assert row.n_skilled == 5              # not 10, and not 9
    assert row.share_of_skilled == pytest.approx(1.0)
    assert "share_of_all" not in share.columns


def test_a_rare_skill_is_reported_as_unsupported_rather_than_dropped():
    # An absent skill and a rare one are different claims.
    df = make_postings(["2023-01-15"] * 50)
    long = make_long({"SQL": list(df.job_id[:40]), "Rust": list(df.job_id[:2])})
    share = skill_share_table({"a": long}, {"a": df}).set_index("skill")
    assert bool(share.loc["Rust", "supported"]) is False
    assert bool(share.loc["SQL", "supported"]) is True


def test_skill_matrix_treats_an_absent_skill_as_a_genuine_zero():
    # The denominator — that company's skilled postings — exists, so a zero is
    # a measurement rather than an imputation.
    a = make_postings(["2023-01-15"] * 20)
    b = make_postings(["2023-01-15"] * 20)
    longs = {"a": make_long({"SQL": list(a.job_id), "Rust": list(a.job_id[:5])}),
             "b": make_long({"SQL": list(b.job_id)})}
    share = skill_share_table(longs, {"a": a, "b": b}, min_postings=1)
    wide = skill_matrix(share, min_companies=1)
    assert wide.loc["Rust", "b"] == 0.0
    assert wide.loc["SQL", "b"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Distinctiveness — the log-lift trap
# ---------------------------------------------------------------------------


def _distinctiveness_fixture():
    a = make_postings(["2023-01-15"] * 20)
    b = make_postings(["2023-01-15"] * 40)
    longs = {
        "a": make_long({"SQL": list(a.job_id[:10])}),
        "b": make_long({"SQL": list(b.job_id[:10]),
                        "Python": list(b.job_id),
                        "Rust": list(b.job_id[:2])}),
    }
    frames = {"a": a, "b": b}
    return skill_share_table(longs, frames, min_postings=1)


def test_a_skill_the_sector_asks_for_and_this_company_never_does_gets_a_row():
    # Iterating only the focus company's own skills silently drops the most
    # interesting finding there is.
    out = skill_distinctiveness(_distinctiveness_fixture(), "a")
    python = out[out.skill == "Python"].iloc[0]
    assert python.postings_with_skill == 0
    assert python.rest_with_skill == 40
    assert python["diff"] < 0


def test_log_lift_stays_finite_and_informative_for_a_zero_cell():
    # With a bare epsilon in the denominator every absent skill collapses onto
    # -23 to -27 and the axis becomes a list of denominators. The
    # Haldane-Anscombe correction keeps the two absences distinguishable:
    # Python is absent from a company whose rivals all ask for it, Rust from
    # one whose rivals barely do.
    out = skill_distinctiveness(_distinctiveness_fixture(), "a").set_index("skill")
    python, rust = out.loc["Python", "log_lift"], out.loc["Rust", "log_lift"]
    assert np.isfinite(python) and np.isfinite(rust)
    assert python > -10 and rust > -5
    assert rust - python > 1.0


def test_distinctiveness_compares_against_the_pooled_rest_not_a_chosen_rival():
    # Otherwise the answer depends on who else happens to be in the set.
    out = skill_distinctiveness(_distinctiveness_fixture(), "a")
    assert set(out.rest_n_skilled) == {40}
    assert set(out.n_skilled) == {20}


def test_distinctiveness_of_an_uncompared_company_is_empty_not_an_exception():
    assert skill_distinctiveness(_distinctiveness_fixture(), "nobody").empty


# ---------------------------------------------------------------------------
# Self-reference — the vendor tautology
# ---------------------------------------------------------------------------


def test_the_vendor_list_covers_every_company_that_is_its_own_skill_token():
    # "Snowflake" and "Databricks" are both employers and skills in the
    # upstream taxonomy, which is why those two companies lead the sector on
    # themselves.
    assert VENDOR_SKILLS["snowflake"]["Snowflake"] == "own_product"
    assert VENDOR_SKILLS["databricks"]["Databricks"] == "own_product"
    # A skill the company created but no longer governs is a weaker bias and
    # is labelled differently, because it reads differently.
    assert VENDOR_SKILLS["google"]["Kubernetes"] == "origin"
    assert VENDOR_SKILLS["google"]["BigQuery"] == "own_product"


def test_self_reference_is_flagged_and_never_filtered():
    table = pd.DataFrame({"company": ["snowflake", "snowflake"],
                          "skill": ["Snowflake", "SQL"]})
    out = annotate_self_reference(table)
    assert len(out) == 2                     # nothing dropped
    assert out.self_referential.tolist() == [True, False]
    assert out.vendor_relation.tolist() == ["own_product", ""]


def test_self_reference_separates_profile_domination_from_coverage_inflation():
    # The leak dominates Snowflake's skill *profile* (99% of postings) while
    # adding almost nothing to its *coverage*, and those are different claims.
    df = make_postings(["2023-01-15"] * 10)
    long = make_long({"Snowflake": list(df.job_id),
                      "SQL": list(df.job_id[:9])})
    out = self_reference_table({"snowflake": long}, {"snowflake": df}).iloc[0]
    assert out.own_product_share == pytest.approx(1.0)
    assert out.coverage_excl_own_product == pytest.approx(0.9)
    assert out.coverage_inflation_pp == pytest.approx(10.0)


def test_a_pairwise_gap_flags_the_vendor_side_so_the_reader_need_not_know_why():
    # "google minus databricks on Databricks" is -100pp and means nothing.
    g = make_postings(["2023-01-15"] * 50)
    d = make_postings(["2023-01-15"] * 50)
    longs = {"google": make_long({"SQL": list(g.job_id[:30])}),
             "databricks": make_long({"SQL": list(d.job_id[:20]),
                                      "Databricks": list(d.job_id)})}
    share = skill_share_table(longs, {"google": g, "databricks": d},
                              min_postings=1)
    gap = pairwise_skill_gap(share, "google", "databricks").set_index("skill")
    assert gap.loc["Databricks", "b_vendor_relation"] == "own_product"
    assert bool(gap.loc["Databricks", "self_referential"]) is True
    assert bool(gap.loc["SQL", "self_referential"]) is False


def test_a_pairwise_gap_is_called_only_when_the_interval_clears_zero():
    g = make_postings(["2023-01-15"] * 200)
    d = make_postings(["2023-01-15"] * 200)
    longs = {"google": make_long({"SQL": list(g.job_id[:180]),
                                  "Python": list(g.job_id[:100])}),
             "databricks": make_long({"SQL": list(d.job_id[:20]),
                                      "Python": list(d.job_id[:98])})}
    share = skill_share_table(longs, {"google": g, "databricks": d})
    gap = pairwise_skill_gap(share, "google", "databricks").set_index("skill")
    assert bool(gap.loc["SQL", "significant"]) is True
    assert bool(gap.loc["Python", "significant"]) is False   # 100 vs 98 of 200


# ---------------------------------------------------------------------------
# Simpson's paradox, across companies
# ---------------------------------------------------------------------------


def _simpsons_fixture():
    """``a`` leads ``b`` pooled on Looker and trails it inside every function.

    The cross-company form of the trap Task 05 caught in time: Looker was
    flagged as emerging for Google and reversed inside every job function that
    supported it.
    """
    a = make_postings(["2023-01-15"] * 60,
                      functions=["Engineering"] * 50 + ["Analytics"] * 10)
    b = make_postings(["2023-01-15"] * 60,
                      functions=["Engineering"] * 10 + ["Analytics"] * 50)
    eng_a, ana_a = list(a.job_id[:50]), list(a.job_id[50:])
    eng_b, ana_b = list(b.job_id[:10]), list(b.job_id[10:])
    longs = {"a": make_long({"Looker": eng_a[:44] + ana_a[:1]}),
             "b": make_long({"Looker": eng_b[:9] + ana_b[:10]})}
    return longs, {"a": a, "b": b}


def test_a_pooled_lead_that_reverses_inside_every_function_is_reported_reversed():
    longs, frames = _simpsons_fixture()
    within = within_stratum_shares(longs, frames, "Looker")
    verdict = stratified_company_verdict(within, "Looker", "a", "b")
    assert verdict["pooled_diff"] > 0          # a leads pooled: 75% vs 32%
    assert verdict["verdict"] == "reversed"    # b leads in both functions
    assert verdict["n_strata"] == 2 and verdict["n_agree"] == 0


def test_a_stratified_verdict_needs_shared_strata_to_say_anything():
    longs, frames = _simpsons_fixture()
    within = within_stratum_shares(longs, frames, "Looker")
    only_one = within[within.job_function == "Engineering"]
    verdict = stratified_company_verdict(only_one, "Looker", "a", "b")
    assert verdict["verdict"] == "unsupported"


def test_within_stratum_shares_drop_cells_below_the_evidence_floor():
    df = make_postings(["2023-01-15"] * 42,
                       functions=["Engineering"] * 40 + ["Tiny"] * 2)
    long = make_long({"SQL": list(df.job_id[:20])})
    within = within_stratum_shares({"a": long}, {"a": df}, "SQL")
    assert set(within.job_function) == {"Engineering"}
    assert MIN_CELL == 10


def test_standardised_skill_shares_use_the_skill_denominator_as_the_population():
    # The standard population has to be the population the shares are actually
    # measured on, or the weights and the rates disagree about who is counted.
    longs, frames = _simpsons_fixture()
    out = standardised_skill_table(longs, frames, ["Looker"])
    w = out.attrs["weights"]
    assert w.sum() == pytest.approx(1.0)
    assert w["Engineering"] == pytest.approx(0.5)
    assert set(out.company) == {"a", "b"}
    assert out.standardised_share.between(0, 1).all()


# ---------------------------------------------------------------------------
# Channel robustness
# ---------------------------------------------------------------------------


def test_a_skill_share_that_moves_when_the_channel_is_held_fixed_is_flagged():
    # Standardisation does not touch channel. A gap that holds on all
    # publishers but flips on the shared ones was a syndication difference
    # wearing a skill's name.
    a = make_postings(["2023-01-15"] * 40,
                      publishers=["via Shared"] * 20 + ["via Own"] * 20)
    b = make_postings(["2023-01-15"] * 20, publishers=["via Shared"] * 20)
    # SQL is on every own-channel posting and none of the shared ones.
    longs = {"a": make_long({"SQL": list(a.job_id[20:])}),
             "b": make_long({"SQL": list(b.job_id[:10])})}
    out = panel_robustness(longs, {"a": a, "b": b}, ["SQL"]).set_index("company")
    assert out.loc["a", "share_of_skilled_all"] == pytest.approx(0.5)
    # A skill that vanishes from the panel produces no row there. That is a
    # measured zero against a denominator that exists, not missing evidence:
    # left as NaN it would read as "not channel sensitive", which is backwards.
    assert out.loc["a", "n_skilled_panel"] == 20
    assert out.loc["a", "share_of_skilled_panel"] == pytest.approx(0.0)
    assert bool(out.loc["a", "channel_sensitive"]) is True
    assert bool(out.loc["b", "channel_sensitive"]) is False


# ---------------------------------------------------------------------------
# Standing checks — the two that run on every emitted table
# ---------------------------------------------------------------------------


def test_the_forbidden_column_list_is_the_two_things_task_06_ruled_out():
    # `country` because only 21 of 48 countries kept their direction on the
    # balanced panel and country coverage differs by publisher, so a
    # cross-company country split compares crawlers. `share_of_all` because
    # Task 04 retired that denominator.
    assert FORBIDDEN_COLUMNS == {"country", "location_country", "share_of_all"}


def test_forbidden_columns_catches_each_banned_name():
    for col in FORBIDDEN_COLUMNS:
        assert forbidden_columns(pd.DataFrame(columns=["company", col])) == [col]


def test_personal_data_check_catches_a_contact_column_arriving_through_a_join():
    assert personal_data_columns_present(
        pd.DataFrame(columns=["company", "recruiter_email"])) == ["recruiter_email"]
    assert personal_data_columns_present(
        pd.DataFrame(columns=["company", "postings"])) == []


def test_the_writer_refuses_a_table_carrying_a_banned_column(tmp_path):
    import build_comparison as bc

    with pytest.raises(ValueError, match="forbidden columns"):
        bc._write(pd.DataFrame({"country": ["UK"]}), tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="personal-data columns"):
        bc._write(pd.DataFrame({"candidate_name": ["x"]}), tmp_path / "bad.csv")
    bc._write(pd.DataFrame({"company": ["google"]}), tmp_path / "ok.csv")
    assert (tmp_path / "ok.csv").exists()


# ---------------------------------------------------------------------------
# Against the committed tables, when they have been built
# ---------------------------------------------------------------------------

TABLES = (Path(__file__).resolve().parents[1] / "members" / "ankit-google" /
          "task-06-tables")
COMMITTED = sorted(TABLES.glob("*.csv"))


@pytest.mark.skipif(not COMMITTED, reason="run src/build_comparison.py first")
@pytest.mark.parametrize("path", COMMITTED, ids=lambda p: p.name)
def test_every_committed_table_passes_both_standing_checks(path):
    table = pd.read_csv(path)
    assert forbidden_columns(table) == []
    assert personal_data_columns_present(table) == []


@pytest.mark.skipif(not COMMITTED, reason="run src/build_comparison.py first")
def test_the_comparison_actually_covers_six_companies():
    gate = pd.read_csv(TABLES / "company-comparability.csv")
    assert len(gate) == 6
    assert set(gate.columns) >= {"common_panel_share", "level_comparable",
                                 "skill_coverage", "treatments_agree"}
