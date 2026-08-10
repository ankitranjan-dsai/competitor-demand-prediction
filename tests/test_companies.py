"""Tests for the Task 06 employer registry.

Same intent as the other suites: every case here is either a trap the real HF
backfill actually sprang, or a rule the other three specialists depend on when
they add their own company.

The ones that matter most:

* substring matching is wrong in *both* directions — ``Geoambiente - Google
  Cloud Premier Partner`` is not Google, and ``Facebook App`` is Meta;
* Task 06 must select exactly the postings Task 02 and Task 03 already
  selected for Google, or the cross-company comparison silently disagrees with
  the Google specialist's own committed numbers;
* the feasibility screen is a *published finding*, not a private filter:
  OpenAI and Anthropic fail it, and their failure is a fact about the source;
* every judgement call in the registry (LinkedIn out of Microsoft, bare
  "Oculus" out of Meta) is pinned here, so widening a pattern later has to
  argue with a test rather than slip through.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import preprocess as pp  # noqa: E402
from collect_google_jobs import GOOGLE_FAMILY_PATTERN  # noqa: E402
from companies import (  # noqa: E402
    MIN_MONTHS,
    MIN_POSTINGS,
    REGISTRY,
    SHORTLIST,
    classify_employer,
    feasibility_screen,
    get_company,
    included_companies,
    matching_audit,
    resolve_employers,
    select_company,
)
from compare import personal_data_columns_present  # noqa: E402


def make_postings(names, dates=None) -> pd.DataFrame:
    """A frame shaped like the HF source: employer string plus a posting date.

    Dates cycle through the twelve months of 2023 unless given, so a company
    built from this helper passes the months-present half of the screen and
    each test isolates the rule it is actually about.
    """
    if dates is None:
        dates = [f"2023-{(i % 12) + 1:02d}-15" for i in range(len(names))]
    return pd.DataFrame({"company_name": names, "posting_date": dates})


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_shortlist_is_exactly_the_registry():
    # The README offers eight candidate companies to four specialists. If the
    # two drift, a specialist analyses a company nobody screened.
    assert set(SHORTLIST) == set(REGISTRY)


def test_every_company_has_a_display_name_and_an_include_pattern():
    for key, company in REGISTRY.items():
        assert company.key == key
        assert company.display
        assert company.include


def test_get_company_names_the_registered_keys_when_it_fails():
    # The error is read by three specialists who did not write this module.
    with pytest.raises(ValueError, match="unknown company"):
        get_company("openai-labs")


# ---------------------------------------------------------------------------
# The Google entry has to stay identical to Tasks 02 and 03
# ---------------------------------------------------------------------------


def test_google_include_is_task_02s_collection_pattern():
    # If these diverge, Task 06 compares a different Google to the one whose
    # 848 postings every earlier task reports.
    assert REGISTRY["google"].include == GOOGLE_FAMILY_PATTERN


def test_google_exclude_is_task_03s_third_party_marker_list():
    assert REGISTRY["google"].exclude == pp.NON_ALPHABET_MARKERS.pattern


def test_google_brands_are_task_03s_brand_ladder_in_order():
    # Order is load-bearing: "Google Fiber" must be tested before "Google".
    assert REGISTRY["google"].brands == tuple(
        tuple(pair) for pair in pp.ALPHABET_BRANDS
    )


# ---------------------------------------------------------------------------
# Matching — wrong in both directions
# ---------------------------------------------------------------------------


def test_google_family_brands_are_matched():
    for name in ("Google", "Alphabet Inc", "DeepMind", "YouTube", "Waymo",
                 "Google Germany GmbH"):
        assert classify_employer(name, "google")["matched"] is True, name


def test_a_google_reseller_is_not_google():
    # Task 03 flagged this exact string. A partner's postings are the
    # partner's hiring, and they carry the partner's publisher and skill mix.
    call = classify_employer("Geoambiente - Google Cloud Premier Partner", "google")
    assert call["matched"] is False
    assert call["reason"] == "named_third_party"


def test_a_role_title_leaked_into_the_employer_field_is_not_the_employer():
    call = classify_employer(
        "Customer Engineer, Machine Learning, Google Cloud - Doha", "google")
    assert call["matched"] is False
    assert call["reason"] == "role_string"


def test_agency_markers_exclude_every_company_not_just_the_ones_that_listed_them():
    # AGENCY_MARKERS is shared, so a specialist adding a company inherits the
    # staffing-firm exclusions without rediscovering them.
    assert classify_employer("Meta Recruitment Ltd", "meta")["reason"] == \
        "third_party_marker"
    assert classify_employer("NVIDIA Staffing", "nvidia")["reason"] == \
        "third_party_marker"


def test_the_agency_marker_list_does_not_swallow_real_subsidiaries():
    # "solutions", "technologies" and "careers" were all considered as markers
    # and rejected, because these names are real.
    assert classify_employer("Snowflake Computing", "snowflake")["matched"] is True
    assert classify_employer("CN05 NVIDIA Shanghai WFOE", "nvidia")["matched"] is True
    assert classify_employer("Facebook App", "meta")["matched"] is True


def test_a_word_that_merely_starts_with_the_brand_is_not_the_brand():
    # "OpenAirlines" and "Metasys Technologies" are the two that cost the most
    # if matched loosely: 4 and 60-odd postings of somebody else's hiring.
    assert classify_employer("OpenAirlines", "openai")["matched"] is False
    assert classify_employer("Metasys Technologies, Inc.", "meta")["matched"] is False


def test_linkedin_is_not_counted_as_microsoft():
    # Microsoft owns LinkedIn, but the employer field in this source
    # demonstrably receives publisher strings ("LinkedIn Job Wrapping"), and a
    # posting whose employer is "LinkedIn" cannot be told apart from that
    # leakage. Deliberate under-count, recorded in the registry note.
    assert classify_employer("LinkedIn", "microsoft")["matched"] is False
    assert classify_employer("LinkedIn Job Wrapping", "microsoft")["matched"] is False


def test_bare_oculus_is_not_counted_as_meta():
    # "Oculus Search Partners" is a search firm. Only "Oculus VR" is Meta.
    assert classify_employer("Oculus Search Partners", "meta")["matched"] is False
    assert classify_employer("Oculus VR", "meta")["matched"] is True


def test_classify_survives_the_empty_and_missing_employer_field():
    for value in (None, float("nan"), "", "   "):
        call = classify_employer(value, "google")
        assert call["matched"] is False
        assert call["reason"] == "empty_employer_field"


def test_the_matched_brand_is_the_most_specific_one():
    assert classify_employer("YouTube LLC", "google")["brand"] == "YouTube"
    assert classify_employer("Google Fiber", "google")["brand"] == "Google Fiber"
    assert classify_employer("Google Cloud", "google")["brand"] == "Google"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_select_company_returns_only_that_company():
    df = make_postings(["Google", "Meta", "Geoambiente - Google Cloud Premier Partner",
                        "DeepMind"])
    out = select_company(df, "google")
    assert sorted(out.company_name) == ["DeepMind", "Google"]


def test_select_company_labels_every_row_it_keeps():
    df = make_postings(["Google", "YouTube", "Meta"])
    out = select_company(df, "google")
    assert set(out.company_key) == {"google"}
    assert sorted(out.company_brand) == ["Google", "YouTube"]


def test_select_company_caches_by_distinct_name_without_changing_the_answer():
    # 785k rows x 8 registry passes is minutes of regex; ~140k distinct names
    # is seconds. This pins the result so the optimisation cannot alter which
    # postings are selected, and that repeats of a name all resolve alike.
    df = make_postings(["Google"] * 50 + ["Metasys Technologies"] * 50)
    out = select_company(df, "google")
    assert len(out) == 50
    assert set(out.company_name) == {"Google"}


def test_select_company_on_an_empty_frame_still_returns_the_added_columns():
    out = select_company(make_postings([]), "google")
    assert {"company_key", "company_brand"} <= set(out.columns)
    assert out.empty


def test_resolve_employers_labels_each_posting_with_its_own_company():
    df = make_postings(["Google", "Meta", "Microsoft", "NVIDIA"])
    resolved = resolve_employers(df)
    assert sorted(resolved.company_key) == ["google", "meta", "microsoft", "nvidia"]


def test_no_posting_in_the_shortlist_belongs_to_two_companies():
    # The assumption the whole comparison rests on: company sets are disjoint,
    # so a share computed over one is not contaminated by another.
    df = make_postings(["Google", "Meta", "Microsoft", "NVIDIA", "Snowflake",
                        "Databricks", "OpenAI", "Anthropic"])
    resolved = resolve_employers(df)
    assert len(resolved) == resolved.company_name.nunique()


# ---------------------------------------------------------------------------
# Audit — the table that makes the calls reviewable
# ---------------------------------------------------------------------------


def test_matching_audit_reports_both_the_kept_and_the_dropped_strings():
    df = make_postings(["Google"] * 3 +
                       ["Geoambiente - Google Cloud Premier Partner"])
    audit = matching_audit(df, keys=("google",))
    assert set(audit.decision) == {"matched", "excluded"}
    assert int(audit.loc[audit.decision == "excluded", "postings"].sum()) == 1


def test_matching_audit_carries_a_reason_for_every_exclusion():
    # An exclusion list without reasons is an assertion; with them it is an
    # argument a reader can point at and disagree with.
    df = make_postings(["Geoambiente - Google Cloud Premier Partner",
                        "Customer Engineer, Google Cloud - Doha"])
    audit = matching_audit(df, keys=("google",))
    excluded = audit[audit.decision == "excluded"]
    assert len(excluded) == 2
    assert excluded.reason.ne("").all()


def test_matching_audit_only_lists_names_that_mention_the_company():
    df = make_postings(["Google", "Acme Corp", "Barclays"])
    audit = matching_audit(df, keys=("google",))
    assert list(audit.employer_string) == ["Google"]


def test_matching_audit_carries_no_personal_data_columns():
    # Task 01's standing check. The audit is the one Task 06 table built out of
    # a free-text field, so it is the one most likely to acquire a name.
    audit = matching_audit(make_postings(["Google", "Meta"]))
    assert personal_data_columns_present(audit) == []


# ---------------------------------------------------------------------------
# Feasibility — a finding, not a filter
# ---------------------------------------------------------------------------


def test_screen_excludes_a_company_with_too_few_postings():
    df = make_postings(["Google"] * (MIN_POSTINGS + 10) + ["Anthropic"] * 9)
    screen = feasibility_screen(df, keys=("google", "anthropic"))
    verdicts = dict(zip(screen.company, screen.verdict))
    assert verdicts["google"] == "included"
    assert verdicts["anthropic"] == "excluded_low_support"


def test_screen_excludes_a_company_present_in_too_few_months():
    # 300 postings all dated one day is not a year of hiring, it is one batch,
    # and a within-2023 trend cannot be read off it.
    df = make_postings(["OpenAI"] * 300, dates=["2023-04-15"] * 300)
    screen = feasibility_screen(df, keys=("openai",))
    row = screen.iloc[0]
    assert row.postings >= MIN_POSTINGS
    assert row.months_present < MIN_MONTHS
    assert row.verdict == "excluded_low_support"


def test_screen_gives_the_reason_it_excluded_a_company():
    df = make_postings(["Anthropic"] * 9)
    screen = feasibility_screen(df, keys=("anthropic",))
    assert "postings" in screen.iloc[0].reason
    assert "months" in screen.iloc[0].reason


def test_screen_reports_excluded_companies_rather_than_dropping_them():
    # "OpenAI has 14 postings in this source" is a finding about the source and
    # belongs in the report, so the row has to survive the screen.
    df = make_postings(["Google"] * (MIN_POSTINGS + 10) + ["OpenAI"] * 5)
    screen = feasibility_screen(df, keys=("google", "openai"))
    assert set(screen.company) == {"google", "openai"}
    assert included_companies(screen) == ["google"]


def test_included_companies_is_ordered_the_same_way_every_run():
    df = make_postings(["Meta"] * (MIN_POSTINGS + 5) +
                       ["Google"] * (MIN_POSTINGS + 5))
    screen = feasibility_screen(df, keys=("meta", "google"))
    # Sorted, not by posting count: figure legends and table column order must
    # not silently reshuffle when one company's volume changes.
    assert included_companies(screen) == ["google", "meta"]


def test_screen_thresholds_are_named_constants():
    # A floor invented per-analysis is a floor chosen after seeing the answer.
    assert isinstance(MIN_POSTINGS, int) and MIN_POSTINGS > 0
    assert isinstance(MIN_MONTHS, int) and 0 < MIN_MONTHS <= 12


# ---------------------------------------------------------------------------
# Against the committed tables, when they have been built
# ---------------------------------------------------------------------------

TABLES = (Path(__file__).resolve().parents[1] / "members" / "ankit-google" /
          "task-06-tables")


@pytest.mark.skipif(not (TABLES / "employer-matching-audit.csv").exists(),
                    reason="run src/build_competitor_set.py first")
def test_committed_audit_excludes_the_two_known_non_google_strings():
    audit = pd.read_csv(TABLES / "employer-matching-audit.csv")
    dropped = audit[(audit.company == "google") & (audit.decision == "excluded")]
    names = " | ".join(dropped.employer_string.astype(str))
    assert "Premier Partner" in names
    assert "Customer Engineer" in names


@pytest.mark.skipif(not (TABLES / "company-feasibility-screen.csv").exists(),
                    reason="run src/build_competitor_set.py first")
def test_committed_screen_excludes_openai_and_anthropic_and_keeps_six():
    screen = pd.read_csv(TABLES / "company-feasibility-screen.csv")
    excluded = set(screen.loc[screen.verdict != "included", "company"])
    assert {"openai", "anthropic"} <= excluded
    assert included_companies(screen) == [
        "databricks", "google", "meta", "microsoft", "nvidia", "snowflake",
    ]
