"""Tests for the Task 05 hiring-trend engine.

Same intent as `tests/test_preprocess.py` and `tests/test_skills.py`: every
case here is either a trap that was actually hit in the real Google data, or a
rule the other three specialists depend on, locked in so a later refactor
cannot quietly undo it.

The ones that matter most:

* the ISO-week boundary that puts 2023-01-01 in ``2022-W52``;
* the publisher panel, which flips the sign of Google's 2023 volume trend;
* the Simpson's-paradox case, where a skill rises overall while falling inside
  every segment that supports it. Task 04 flagged Looker as emerging on the
  pooled numbers; it is not.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import trends as tr  # noqa: E402
from trends import (  # noqa: E402
    PERIODS,
    add_velocity,
    attribute_spikes,
    balanced_volume_series,
    batch_table,
    compare_panels,
    half_of,
    matched_index,
    observation_window,
    panel_publishers,
    panel_sensitivity_table,
    panel_verdict,
    period_bounds,
    period_col,
    publisher_panel_table,
    publisher_presence,
    robust_spikes,
    seasonality_table,
    segment_series,
    segment_trend_table,
    skill_trend_within_segment,
    skill_velocity_table,
    stratified_verdict,
    volume_series,
    weekend_share,
)


# ---------------------------------------------------------------------------
# Fixtures — built the way Task 03 builds its period keys
# ---------------------------------------------------------------------------


def make_postings(dates, publishers=None, functions=None, skilled=None,
                  job_ids=None) -> pd.DataFrame:
    """A minimal Task 03-shaped frame: dates plus the columns trends.py reads."""
    dates = pd.to_datetime(pd.Series(list(dates)))
    n = len(dates)
    iso = dates.dt.isocalendar()
    df = pd.DataFrame(
        {
            "job_id": job_ids or [f"j{i}" for i in range(n)],
            "posting_date": dates.dt.strftime("%Y-%m-%d"),
            "posting_month": dates.dt.strftime("%Y-%m"),
            "posting_week": [f"{y}-W{w:02d}" for y, w in zip(iso.year, iso.week)],
            "posting_quarter": [f"{d.year}-Q{(d.month - 1) // 3 + 1}" for d in dates],
            "job_via": publishers if publishers is not None else ["via A"] * n,
            "job_function": functions if functions is not None else ["Engineering"] * n,
            "has_any_skill": skilled if skilled is not None else [True] * n,
            "skill_count_final": [2] * n,
        }
    )
    return df


def daily(start: str, end: str, per_day: int = 1, publisher: str = "via A"):
    """One frame with `per_day` postings on every day in [start, end]."""
    days = pd.date_range(start, end, freq="D")
    dates = [d for d in days for _ in range(per_day)]
    return make_postings(dates, publishers=[publisher] * len(dates))


# ---------------------------------------------------------------------------
# Period grammar
# ---------------------------------------------------------------------------


def test_period_col_maps_the_three_agreed_periods():
    assert period_col("month") == "posting_month"
    assert period_col("week") == "posting_week"
    assert period_col("quarter") == "posting_quarter"
    assert set(PERIODS) == {"week", "month", "quarter"}


def test_period_col_rejects_an_unknown_period():
    # A silent fallback to months is how two specialists end up publishing
    # "weekly" series on different clocks.
    with pytest.raises(ValueError, match="unknown period"):
        period_col("fortnight")


def test_period_bounds_month_and_quarter():
    assert period_bounds("2023-02", "month") == (
        pd.Timestamp("2023-02-01"), pd.Timestamp("2023-02-28"))
    assert period_bounds("2023-Q3", "quarter") == (
        pd.Timestamp("2023-07-01"), pd.Timestamp("2023-09-30"))


def test_iso_week_boundary_puts_new_years_day_in_the_previous_year():
    # 2023-01-01 is a Sunday, so ISO files it under 2022-W52 and the Google
    # weekly series has 53 buckets for one calendar year. Two of them are
    # partial; reading them as a slow start is the trap.
    start, end = period_bounds("2022-W52", "week")
    assert start == pd.Timestamp("2022-12-26")
    assert end == pd.Timestamp("2023-01-01")


def test_observation_window_uses_the_data_not_the_calendar():
    df = daily("2023-03-15", "2023-04-10")
    assert observation_window(df) == (pd.Timestamp("2023-03-15"),
                                      pd.Timestamp("2023-04-10"))


# ---------------------------------------------------------------------------
# Volume and velocity
# ---------------------------------------------------------------------------


def test_velocity_normalises_february_against_the_calendar():
    # Same true rate of 1 posting/day. Raw counts say March is 11% busier than
    # February; it is not, February is short.
    df = daily("2023-02-01", "2023-03-31")
    out = volume_series(df, "month").set_index("period")
    assert out.loc["2023-02", "postings"] == 28
    assert out.loc["2023-03", "postings"] == 31
    assert out.loc["2023-02", "postings_per_week"] == pytest.approx(7.0)
    assert out.loc["2023-03", "postings_per_week"] == pytest.approx(7.0)


def test_days_observed_is_clipped_to_the_collection_window():
    df = daily("2023-03-20", "2023-04-30")
    out = volume_series(df, "month").set_index("period")
    assert out.loc["2023-03", "days_observed"] == 12   # 20th-31st
    assert bool(out.loc["2023-03", "is_partial"]) is True
    assert bool(out.loc["2023-04", "is_partial"]) is False


def test_a_period_seen_for_under_half_its_length_reports_no_rate():
    # 2022-W52 in the Google data is one observed day carrying 7 postings.
    # Reported as a rate that is 49/week — the largest value in the series.
    df = make_postings(["2023-01-01"] * 7 + ["2023-01-05"] * 3)
    out = volume_series(df, "week").set_index("period")
    assert out.loc["2022-W52", "postings"] == 7
    assert out.loc["2022-W52", "days_observed"] == 1
    assert pd.isna(out.loc["2022-W52", "postings_per_week"])


def test_empty_periods_are_emitted_as_zero_not_dropped():
    # A collection gap has to stay visible as a hole in the series.
    df = make_postings(["2023-01-10", "2023-03-10"])
    out = volume_series(df, "month")
    assert out.period.tolist() == ["2023-01", "2023-02", "2023-03"]
    assert out.set_index("period").loc["2023-02", "postings"] == 0


def test_volume_series_counts_publishers_per_period():
    df = make_postings(
        ["2023-01-05", "2023-01-06", "2023-02-05"],
        publishers=["via A", "via B", "via A"],
    )
    out = volume_series(df, "month").set_index("period")
    assert out.loc["2023-01", "n_publishers"] == 2
    assert out.loc["2023-02", "n_publishers"] == 1


def test_volume_series_requires_the_task_03_period_column():
    df = daily("2023-01-01", "2023-01-31").drop(columns=["posting_month"])
    with pytest.raises(KeyError, match="posting_month"):
        volume_series(df, "month")


def test_velocity_rebases_on_the_first_complete_period():
    df = daily("2023-01-15", "2023-03-31")
    out = add_velocity(volume_series(df, "month")).set_index("period")
    assert pd.isna(out.loc["2023-01", "index_base_100"])   # partial, excluded
    assert out.loc["2023-02", "index_base_100"] == pytest.approx(100.0)


def test_log_growth_is_symmetric_where_percent_change_is_not():
    # Anchored on the 1st and the 31st so all three months are fully observed;
    # a partial month is blanked and would leave nothing to difference.
    df = make_postings(
        ["2023-01-01"] * 10 + ["2023-02-10"] * 5 + ["2023-03-31"] * 10)
    out = add_velocity(volume_series(df, "month"))
    # Down then back up to the same rate: the log differences cancel exactly,
    # the percentage changes do not. Task 07 models the log series for this.
    assert out.log_growth.dropna().sum() == pytest.approx(0.0, abs=1e-3)
    assert abs(out.growth_pct.dropna().sum()) > 10


def test_growth_is_not_padded_across_a_blanked_period():
    # Collection starts on 20 February, so February's rate is blank. Pandas'
    # old default would pad it forward and report March's growth against a
    # rate that was never measured.
    df = make_postings(["2023-02-20"] * 10 + ["2023-03-31"] * 20)
    out = add_velocity(volume_series(df, "month")).set_index("period")
    assert pd.isna(out.loc["2023-02", "postings_per_week"])
    assert pd.isna(out.loc["2023-03", "growth_pct"])


def test_acceleration_is_the_change_in_growth():
    df = make_postings(
        ["2023-01-01"] * 10 + ["2023-02-10"] * 20 + ["2023-03-31"] * 25)
    out = add_velocity(volume_series(df, "month"))
    assert out.growth_pct.iloc[-1] > 0      # still growing
    assert out.acceleration.iloc[-1] < 0    # but slower than last month


# ---------------------------------------------------------------------------
# The publisher panel — the Task 05 defect
# ---------------------------------------------------------------------------


def test_publisher_panel_table_records_entry_and_exit():
    df = make_postings(
        ["2023-01-10", "2023-02-10", "2023-03-10", "2023-03-11"],
        publishers=["via A", "via A", "via A", "via B"],
    )
    panel = publisher_panel_table(df, "month").set_index("publisher")
    assert panel.loc["via A", "periods_present"] == 3
    assert bool(panel.loc["via A", "is_continuous"]) is True
    assert panel.loc["via B", "first_period"] == "2023-03"
    assert panel.loc["via B", "last_period"] == "2023-03"
    assert bool(panel.loc["via B", "is_continuous"]) is False


def test_panel_publishers_drops_the_one_month_arrivals():
    # 56 of Google's 96 publishers appear in exactly one month.
    df = make_postings(
        ["2023-01-10", "2023-02-10", "2023-03-10", "2023-03-11"],
        publishers=["via A", "via A", "via A", "via B"],
    )
    assert panel_publishers(df, "month", min_share=0.75) == ["via A"]


def test_publisher_presence_is_a_publisher_by_period_matrix():
    df = make_postings(["2023-01-10", "2023-01-11", "2023-02-10"],
                       publishers=["via A", "via B", "via A"])
    presence = publisher_presence(df, "month")
    assert presence.loc["via A", "2023-01"] == 1
    assert presence.loc["via B", "2023-02"] == 0


def test_an_entering_publisher_can_invent_a_hiring_rise():
    # The whole reason this module exists. The incumbent publisher's volume is
    # flat all year; a second aggregator arrives in July with a steady feed.
    # Raw volume says +100%; the balanced panel says flat.
    incumbent = daily("2023-01-01", "2023-12-31", per_day=1, publisher="via A")
    newcomer = daily("2023-07-01", "2023-12-31", per_day=1, publisher="via B")
    df = pd.concat([incumbent, newcomer], ignore_index=True)
    df["job_id"] = [f"j{i}" for i in range(len(df))]

    raw = volume_series(df, "month").set_index("period")
    bal = balanced_volume_series(df, "month", min_share=0.75).set_index("period")

    assert raw.loc["2023-12", "postings"] == 2 * raw.loc["2023-01", "postings"]
    assert bal.loc["2023-12", "postings"] == bal.loc["2023-01", "postings"]
    assert panel_publishers(df, "month") == ["via A"]


def test_matched_index_is_100_at_its_base():
    df = daily("2023-01-01", "2023-03-31")
    for method in ("chained", "bilateral"):
        out = matched_index(df, "month", method)
        assert out.iloc[0]["index"] == pytest.approx(100.0)
        assert (out.method == method).all()


def test_matched_index_ignores_a_publisher_absent_from_one_side_of_the_link():
    # via B exists only in February, so the Jan->Feb link must be computed on
    # via A alone: 10 -> 10, an index of 100, not 100 * (10+40)/10.
    df = make_postings(
        ["2023-01-10"] * 10 + ["2023-02-10"] * 10 + ["2023-02-11"] * 40,
        publishers=["via A"] * 20 + ["via B"] * 40,
    )
    out = matched_index(df, "month", "chained").set_index("period")
    assert out.loc["2023-02", "index"] == pytest.approx(100.0)
    assert out.loc["2023-02", "n_matched_publishers"] == 1


def test_chained_and_bilateral_agree_when_the_panel_is_stable():
    df = pd.concat(
        [daily("2023-01-01", "2023-03-31", per_day=1, publisher="via A"),
         daily("2023-01-01", "2023-03-31", per_day=1, publisher="via B")],
        ignore_index=True)
    df["job_id"] = [f"j{i}" for i in range(len(df))]
    chained = matched_index(df, "month", "chained").set_index("period")["index"]
    bilateral = matched_index(df, "month", "bilateral").set_index("period")["index"]
    assert chained.round(2).tolist() == bilateral.round(2).tolist()


def test_matched_index_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        matched_index(daily("2023-01-01", "2023-02-28"), "month", "hedonic")


def test_panel_verdict_flags_disagreement_between_treatments():
    # Raw falls while the balanced panel rises: the direction of Google's 2023
    # volume is not identified, and the report has to say so.
    sens = pd.DataFrame({
        "period": ["2023-01", "2023-12"],
        "raw_index": [100.0, 70.0],
        "balanced_index": [100.0, 140.0],
        "chained_index": [100.0, 150.0],
        "bilateral_index": [100.0, 160.0],
    })
    verdict = panel_verdict(sens)
    assert verdict.agrees is False
    assert verdict.direction == "unresolved"


def test_panel_verdict_agrees_when_every_treatment_grows():
    sens = pd.DataFrame({
        "period": ["2023-01", "2023-12"],
        "raw_index": [100.0, 130.0],
        "balanced_index": [100.0, 140.0],
        "chained_index": [100.0, 150.0],
        "bilateral_index": [100.0, 160.0],
    })
    verdict = panel_verdict(sens)
    assert verdict.agrees is True
    assert verdict.direction == "growth"


def test_panel_verdict_treats_a_small_move_as_flat():
    sens = pd.DataFrame({
        "period": ["2023-01", "2023-12"],
        "raw_index": [100.0, 104.0],
        "balanced_index": [100.0, 97.0],
        "chained_index": [100.0, 101.0],
        "bilateral_index": [100.0, 99.0],
    })
    assert panel_verdict(sens).direction == "flat"


def test_panel_sensitivity_table_carries_every_treatment_and_its_support():
    df = pd.concat(
        [daily("2023-01-01", "2023-06-30", per_day=1, publisher="via A"),
         daily("2023-04-01", "2023-06-30", per_day=2, publisher="via B")],
        ignore_index=True)
    df["job_id"] = [f"j{i}" for i in range(len(df))]
    sens = panel_sensitivity_table(df, "month")
    for col in ("raw_index", "balanced_index", "chained_index", "bilateral_index",
                "raw_publishers", "chained_matched_publishers"):
        assert col in sens.columns
    assert sens.iloc[0]["raw_index"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Spikes
# ---------------------------------------------------------------------------


def test_robust_spikes_finds_a_single_outlying_period():
    counts = [10] * 8 + [60] + [10] * 8
    df = pd.DataFrame({"period": [f"p{i}" for i in range(len(counts))],
                       "postings": counts})
    out = robust_spikes(df)
    assert out.is_spike.sum() == 1
    assert out.loc[out.is_spike, "period"].iloc[0] == "p8"


def test_spike_detection_falls_back_to_poisson_when_mad_is_zero():
    # A perfectly flat local window gives MAD 0 and a divide-by-zero z-score.
    # Without the fallback the largest spike in the series is missed entirely.
    counts = [10] * 8 + [40] + [10] * 8
    df = pd.DataFrame({"period": [f"p{i}" for i in range(len(counts))],
                       "postings": counts})
    out = robust_spikes(df)
    assert pd.isna(out.loc[8, "robust_z"])
    assert bool(out.loc[8, "is_spike"]) is True


def test_a_flat_series_has_no_spikes():
    df = pd.DataFrame({"period": [f"p{i}" for i in range(12)],
                       "postings": [10] * 12})
    assert robust_spikes(df).is_spike.sum() == 0


def test_batch_table_finds_one_publisher_dumping_on_one_day():
    df = make_postings(
        ["2023-08-23"] * 21 + ["2023-08-24"] * 2,
        publishers=["via The Muse"] * 21 + ["via LinkedIn"] * 2,
    )
    batches = batch_table(df, min_batch=5)
    assert len(batches) == 1
    assert batches.iloc[0]["publisher"] == "via The Muse"
    assert batches.iloc[0]["postings"] == 21


def test_attribute_spikes_calls_a_new_publishers_backfill_a_batch():
    # The real 2023-W34 case: The Muse enters the panel with 21 postings on a
    # single day, and it becomes the biggest "hiring week" of Google's year.
    base = daily("2023-07-03", "2023-09-24", per_day=2, publisher="via LinkedIn")
    burst = make_postings(["2023-08-23"] * 21,
                          publishers=["via The Muse"] * 21)
    df = pd.concat([base, burst], ignore_index=True)
    df["job_id"] = [f"j{i}" for i in range(len(df))]

    series = volume_series(df, "week")
    spikes = robust_spikes(series)
    attributed = attribute_spikes(df, spikes, "week")

    assert len(attributed) == 1
    row = attributed.iloc[0]
    assert row.top_publisher == "via The Muse"
    assert row.verdict == "publisher_batch"
    assert row.excess_explained >= 0.5


def test_attribute_spikes_calls_a_broad_surge_broad_based():
    # Same size of spike, but shared across the publishers already present and
    # spread over the week — this one is allowed to be a hiring event.
    frames = [daily("2023-07-03", "2023-09-24", per_day=2, publisher=f"via {p}")
              for p in "ABCDE"]
    extra = pd.concat(
        [make_postings(pd.date_range("2023-08-21", "2023-08-25", freq="D"),
                       publishers=[f"via {p}"] * 5) for p in "ABCDE"],
        ignore_index=True)
    df = pd.concat(frames + [extra], ignore_index=True)
    df["job_id"] = [f"j{i}" for i in range(len(df))]

    spikes = robust_spikes(volume_series(df, "week"))
    attributed = attribute_spikes(df, spikes, "week")
    assert (attributed.verdict == "broad_based").all()


def test_attribute_spikes_returns_an_empty_frame_when_nothing_spikes():
    df = daily("2023-01-01", "2023-03-31")
    spikes = robust_spikes(volume_series(df, "week"))
    out = attribute_spikes(df, spikes, "week")
    assert out.empty
    assert "verdict" in out.columns


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,period,expected",
    [("2023-03", "month", "h1"), ("2023-07", "month", "h2"),
     ("2023-Q2", "quarter", "h1"), ("2023-Q3", "quarter", "h2"),
     ("2023-W05", "week", "h1"), ("2023-W40", "week", "h2")],
)
def test_half_of_agrees_across_period_types(key, period, expected):
    # skills.skill_trend_table splits on the month number; the weekly and
    # quarterly keys have to land on the same side of the year or the Task 04
    # and Task 05 halves stop being comparable.
    assert half_of(key) == expected


def test_half_of_is_blank_for_an_unparseable_key():
    assert half_of("not-a-period") == ""


def test_segment_series_reports_both_count_and_share_of_period():
    df = make_postings(
        ["2023-01-05"] * 10 + ["2023-02-05"] * 20,
        functions=["Engineering"] * 5 + ["Analytics"] * 5
        + ["Engineering"] * 10 + ["Analytics"] * 10,
    )
    out = segment_series(df, "job_function", "month", min_postings=1)
    jan = out[out.period == "2023-01"].set_index("segment" if "segment" in out
                                                 else "job_function")
    assert jan.loc["Engineering", "share_of_period"] == pytest.approx(0.5)


def test_segment_trend_table_separates_count_growth_from_share_growth():
    # Engineering doubles in count but the company more than doubles around
    # it, so its share falls. Both facts are true and they answer different
    # questions; reporting only the count would call this expansion.
    df = make_postings(
        ["2023-02-05"] * 20 + ["2023-08-05"] * 60,
        functions=["Engineering"] * 10 + ["Analytics"] * 10
        + ["Engineering"] * 20 + ["Analytics"] * 40,
    )
    out = segment_trend_table(df, "job_function").set_index("segment")
    assert out.loc["Engineering", "count_change_pct"] == pytest.approx(100.0)
    assert out.loc["Engineering", "share_change_pct"] < 0
    assert out.loc["Engineering", "direction"] == "decline"


def test_segment_trend_table_rejects_an_unknown_panel():
    with pytest.raises(ValueError, match="unknown panel"):
        segment_trend_table(daily("2023-01-01", "2023-12-31"), "job_function",
                            panel="weighted")


def test_compare_panels_flags_a_segment_that_only_grows_on_the_raw_panel():
    # via B arrives in H2 publishing nothing but Analytics. On raw volume
    # Analytics "grows"; on the stable panel it does not move at all.
    incumbent = daily("2023-01-01", "2023-12-31", per_day=2, publisher="via A")
    incumbent["job_function"] = ["Engineering", "Analytics"] * (len(incumbent) // 2)
    newcomer = daily("2023-07-01", "2023-12-31", per_day=2, publisher="via B")
    newcomer["job_function"] = "Analytics"
    df = pd.concat([incumbent, newcomer], ignore_index=True)
    df["job_id"] = [f"j{i}" for i in range(len(df))]

    out = compare_panels(df, "job_function").set_index("segment")
    assert out.loc["Analytics", "direction"] == "growth"
    assert out.loc["Analytics", "direction_balanced"] == "stable"
    assert bool(out.loc["Analytics", "directions_agree"]) is False


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------


def test_annual_seasonality_is_not_identifiable_from_a_single_year():
    # The brief asks for seasonal spikes. One year gives one observation per
    # calendar month, perfectly confounded with the trend. The table has to
    # say so in a column, not in a footnote.
    out = seasonality_table(daily("2023-01-01", "2023-12-31"))
    months = out[out.cycle == "month_of_year"]
    assert not months.identifiable.any()
    weekdays = out[out.cycle == "day_of_week"]
    assert weekdays.identifiable.all()


def test_annual_seasonality_becomes_identifiable_with_enough_years():
    out = seasonality_table(daily("2021-01-01", "2023-12-31"))
    assert out[out.cycle == "month_of_year"].identifiable.all()


def test_weekend_share_detects_a_date_field_that_is_not_a_publication_date():
    # A real publication date is near zero at weekends. Google's field sits at
    # 21%, close to the uniform 2/7 — it is an aggregator first-seen date.
    business = make_postings(pd.bdate_range("2023-01-02", "2023-03-31"))
    assert weekend_share(business) == pytest.approx(0.0)
    everyday = daily("2023-01-01", "2023-12-31")
    assert weekend_share(everyday) == pytest.approx(2 / 7, abs=0.01)


# ---------------------------------------------------------------------------
# Skill demand under the Task 04 rules
# ---------------------------------------------------------------------------


def skills_long_for(df: pd.DataFrame, skill_by_id: dict) -> pd.DataFrame:
    rows = [
        {"job_id": r.job_id, "skill": s, "skill_category": "Programming Language",
         "job_function": r.job_function, "posting_month": r.posting_month,
         "posting_week": r.posting_week, "posting_quarter": r.posting_quarter}
        for r in df.itertuples()
        for s in skill_by_id.get(r.job_id, [])
    ]
    return pd.DataFrame(rows)


def test_skill_velocity_never_emits_the_wrong_denominator():
    # Task 04 §4: on share_of_all Python "fell" 23% March->April 2023 while on
    # share_of_skilled it rose 10%, and the entire difference was April's
    # extraction coverage. The wrong denominator must not be one column away
    # from the right one.
    df = make_postings(["2023-01-05"] * 4, skilled=[True, True, False, False])
    long = skills_long_for(df, {"j0": ["Python"], "j1": ["Python"]})
    out = skill_velocity_table(long, df)
    assert "share_of_all" not in out.columns
    assert out.iloc[0]["share_of_skilled"] == pytest.approx(1.0)


def test_skill_velocity_excludes_facilities_roles_from_the_denominator():
    # Task 04 §3.1: data-centre facilities roles have 9.5% skill coverage
    # because they genuinely have no software skills. Left in, they drag every
    # skill share down and manufacture a decline.
    df = make_postings(
        ["2023-01-05"] * 6,
        functions=["Engineering"] * 2 + ["Facilities / Operations"] * 4,
        skilled=[True, True, False, False, False, False],
    )
    long = skills_long_for(df, {"j0": ["Python"], "j1": ["Python"]})
    out = skill_velocity_table(long, df)
    assert out.iloc[0]["postings_with_skills"] == 2
    assert out.iloc[0]["share_of_skilled"] == pytest.approx(1.0)


def test_skill_velocity_is_empty_for_empty_input():
    out = skill_velocity_table(pd.DataFrame(), pd.DataFrame())
    assert out.empty
    assert "share_of_skilled" in out.columns


def simpsons_paradox_frame():
    """A skill that rises overall while falling inside every segment.

    This is the Looker case from the real Google data, reduced: Looker is
    heavily used in Sales and barely used in Research. Its share falls in both
    halves *within* each function, but Sales grows as a share of the company,
    so the pooled number rises. Task 04 flagged it emerging.
    """
    rows = []
    spec = [
        # (half, function, n_postings, n_with_looker)
        ("h1", "Sales", 10, 9),
        ("h2", "Sales", 40, 32),               # 90% -> 80% within Sales
        ("h1", "Science / Research", 40, 12),
        ("h2", "Science / Research", 40, 4),   # 30% -> 10% within Research
    ]
    dates = {"h1": "2023-03-05", "h2": "2023-09-05"}
    i = 0
    for half, function, n, n_skill in spec:
        for k in range(n):
            rows.append({"job_id": f"j{i}", "date": dates[half],
                         "function": function, "looker": k < n_skill})
            i += 1
    frame = pd.DataFrame(rows)
    df = make_postings(frame.date, functions=frame.function.tolist(),
                       job_ids=frame.job_id.tolist())
    long = skills_long_for(
        df, {r.job_id: ["Looker"] for r in frame.itertuples() if r.looker})
    return df, long


def test_stratification_overturns_a_pooled_emerging_flag():
    df, long = simpsons_paradox_frame()

    pooled_h1 = long[long.posting_month == "2023-03"].job_id.nunique() / 50
    pooled_h2 = long[long.posting_month == "2023-09"].job_id.nunique() / 80
    assert pooled_h2 > pooled_h1          # pooled: Looker is "emerging"

    within = skill_trend_within_segment(long, df, "job_function")
    verdict = stratified_verdict(within, "Looker")
    assert verdict["verdict"] == "falling_in_all_segments"
    assert verdict["up"] == 0


def test_stratified_verdict_confirms_a_skill_that_rises_everywhere():
    # The SQL case: it rises inside all five supported functions, so the
    # pooled rise is not a mix effect.
    df = make_postings(
        ["2023-03-05"] * 40 + ["2023-09-05"] * 40,
        functions=(["Engineering"] * 20 + ["Analytics"] * 20) * 2,
    )
    ids = df.job_id.tolist()
    rising = ids[0:5] + ids[20:25] + ids[40:55] + ids[60:75]
    long = skills_long_for(df, {i: ["SQL"] for i in rising})
    verdict = stratified_verdict(skill_trend_within_segment(long, df,
                                                            "job_function"), "SQL")
    assert verdict["verdict"] == "rising_in_all_segments"
    assert verdict["n_segments"] == 2


def test_stratified_verdict_reports_insufficient_support_rather_than_guessing():
    df = make_postings(["2023-03-05", "2023-09-05"])
    long = skills_long_for(df, {"j0": ["Rust"]})
    within = skill_trend_within_segment(long, df, "job_function")
    assert stratified_verdict(within, "Rust")["verdict"] == "insufficient_support"


def test_stratified_verdict_names_a_mix_dependent_skill():
    df = make_postings(
        ["2023-03-05"] * 40 + ["2023-09-05"] * 40,
        functions=(["Engineering"] * 20 + ["Analytics"] * 20) * 2,
    )
    ids = df.job_id.tolist()
    mixed = ids[0:5] + ids[20:30] + ids[40:55] + ids[60:65]   # up in one, down in other
    long = skills_long_for(df, {i: ["Java"] for i in mixed})
    within = skill_trend_within_segment(long, df, "job_function")
    assert stratified_verdict(within, "Java")["verdict"] == "mix_dependent"


# ---------------------------------------------------------------------------
# Standing project rules
# ---------------------------------------------------------------------------


def test_no_personal_data_column_is_introduced_by_the_trend_tables():
    # The Task 01 commitment, re-checked wherever a new table is created.
    df = daily("2023-01-01", "2023-03-31")
    banned = ("email", "phone", "candidate", "applicant", "recruiter", "name")
    for table in (volume_series(df, "month"),
                  publisher_panel_table(df, "month"),
                  seasonality_table(df),
                  segment_trend_table(df, "job_function")):
        assert not [c for c in table.columns
                    if any(b in c.lower() for b in banned)]


def test_the_facilities_exclusion_is_named_once_and_shared():
    # Every specialist must exclude the same roles, or Task 06 compares
    # different denominators across companies.
    assert "Facilities / Operations" in tr.SKILL_EXCLUDED_FUNCTIONS
    assert tr.PUBLISHER_COL == "job_via"
    assert set(tr.PANEL_TREATMENTS) == {"raw", "balanced", "matched"}
