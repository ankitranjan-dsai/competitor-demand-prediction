"""Tests for the Task 04 skill taxonomy, extraction and feature engineering.

Same intent as `tests/test_preprocess.py`: every case here is a trap that was
actually hit in the real Google data (or a rule the other three specialists
depend on), locked in so a later refactor can't quietly undo it.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import skills as sk  # noqa: E402
from skills import (  # noqa: E402
    CATEGORIES,
    CATEGORY_OF,
    EXCLUDED_TERMS,
    NON_STACK_CATEGORIES,
    SKILLS,
    as_skill_sequence,
    build_posting_features,
    category_slug,
    coverage_table,
    extract_skills,
    extract_skills_from_text,
    extract_skills_from_title,
    extract_with_audit,
    get_skill,
    normalize_skill_list,
    skill_by_month_table,
    skill_cooccurrence_table,
    skill_features,
    skill_frequency_table,
    skill_matrix,
    skill_trend_table,
    skills_long_table,
    strip_org_segments,
)


# ---------------------------------------------------------------------------
# Taxonomy integrity — the part the other three specialists rely on
# ---------------------------------------------------------------------------


def test_every_skill_uses_a_declared_category():
    assert {s.category for s in SKILLS} <= set(CATEGORIES)


def test_canonical_names_and_keys_are_unique():
    names = [s.name.lower() for s in SKILLS]
    keys = [s.key for s in SKILLS]
    assert len(set(names)) == len(names)
    assert len(set(keys)) == len(keys)


def test_no_alias_is_claimed_by_two_skills():
    # The module raises at import time if this breaks; assert it explicitly so
    # the guarantee is visible in the test report, not just in a traceback.
    owner: dict[str, str] = {}
    for skill in SKILLS:
        for token in (skill.name.lower(), *skill.aliases):
            assert owner.setdefault(token, skill.name) == skill.name


def test_context_required_skills_are_the_short_english_words():
    # If a skill with a distinctive name gets context=True it silently stops
    # matching; if a common word gets context=False it floods the output.
    ambiguous = {s.name for s in SKILLS if s.context_required}
    assert {"R", "C", "Go", "Excel", "Spark", "React", "Swift"} <= ambiguous
    assert not {"Python", "TensorFlow", "Kubernetes", "BigQuery"} & ambiguous


def test_excluded_terms_never_resolve_to_a_skill():
    for term in EXCLUDED_TERMS:
        assert get_skill(term) is None


def test_category_slug_is_a_safe_column_name():
    assert category_slug("ML / AI") == "ml_ai"
    assert category_slug("Facilities / Data Centre") == "facilities_data_centre"


@pytest.mark.parametrize(
    "name,category",
    [
        # The source dataset files these wrong; the taxonomy is the fix, so the
        # corrections are pinned here rather than left to a code comment.
        ("SAS", "Analytics / BI"),          # source: "programming" AND "analyst_tools"
        ("MongoDB", "Database"),            # source: "programming"
        ("NoSQL", "Database"),              # source: "programming"
        ("GDPR", "Governance / Compliance"),  # source: "libraries"
        ("Colocation", "Facilities / Data Centre"),  # source: "cloud"
        ("PowerPoint", "Office / Productivity"),     # source: "analyst_tools"
        ("SQL", "Programming Language"),    # a language, not the database
        ("SQL Server", "Database"),
    ],
)
def test_source_category_defects_are_corrected(name, category):
    assert CATEGORY_OF[get_skill(name).name] == category


def test_non_stack_categories_are_all_real_categories():
    assert NON_STACK_CATEGORIES <= set(CATEGORIES)


# ---------------------------------------------------------------------------
# Path A — normalising a collector's own skill list
# ---------------------------------------------------------------------------


def test_aliases_collapse_onto_one_canonical_name():
    assert normalize_skill_list(["python", "Python 3", "python3"]) == ["Python"]
    assert normalize_skill_list(["gcp", "google cloud"]) == ["GCP"]


def test_first_seen_order_is_preserved():
    assert normalize_skill_list(["sql", "python", "sql"]) == ["SQL", "Python"]


def test_excluded_terms_are_dropped_and_are_not_counted_as_unmapped():
    unmapped: Counter = Counter()
    assert normalize_skill_list(["flow", "python"], unmapped=unmapped) == ["Python"]
    assert unmapped == Counter()


def test_unknown_terms_are_counted_so_coverage_is_measurable():
    unmapped: Counter = Counter()
    normalize_skill_list(["python", "wingdings"], unmapped=unmapped)
    assert unmapped["wingdings"] == 1


def test_missing_skill_list_is_empty_not_an_error():
    assert normalize_skill_list(None) == []


def test_source_path_trusts_the_collector_for_ambiguous_names():
    # "r" in a curated skill list means the language; the context guard applies
    # to free text only, where "r" is usually noise.
    assert normalize_skill_list(["r", "go", "c"]) == ["R", "Go", "C"]


# ---------------------------------------------------------------------------
# Paths B & C — free text, and the ambiguity guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,skill",
    [
        ("Proficiency in Excel and SQL", "Excel"),
        ("Experience with Spark clusters", "Spark"),
        ("Frontend in React, TypeScript, Node.js", "React"),
        ("Backend written in Go and Java", "Go"),
        ("Programming in Python, R, SQL", "R"),
        ("Swift developer for iOS", "Swift"),
        ("Statistical analysis in R Studio", "R"),
    ],
)
def test_ambiguous_skill_is_accepted_when_context_confirms_it(text, skill):
    assert skill in extract_skills_from_text(text)


@pytest.mark.parametrize(
    "text,skill",
    [
        # All of these are real job-posting English, and all become false
        # skills under a naive keyword match.
        ("Ability to excel in a fast-paced environment", "Excel"),
        ("You will spark innovation across teams", "Spark"),
        ("React to changing customer needs", "React"),
        ("Own the go-to-market strategy", "Go"),
        ("Swift decision-making under pressure", "Swift"),
        ("Partner with the chef and facilities team", "Chef"),
    ],
)
def test_ambiguous_word_in_plain_english_is_rejected(text, skill):
    assert skill not in extract_skills_from_text(text)


def test_rejections_are_recorded_so_the_guard_is_auditable():
    found, rejected = extract_with_audit(
        "Ability to excel in a fast-paced environment"
    )
    assert found == []
    name, snippet = rejected[0]
    assert name == "Excel"
    assert snippet in "Ability to excel in a fast-paced environment"


def test_go_to_market_never_reaches_the_guard_at_all():
    # Handled in the pattern, not the context guard: "go-to-market" is a phrase,
    # so there is nothing to adjudicate. 173 Google postings (20%) contain it.
    _, rejected = extract_with_audit("Own the go-to-market strategy")
    assert rejected == []


def test_longest_match_wins():
    assert extract_skills_from_text("Experience with SQL Server") == ["SQL Server"]
    assert "C" not in extract_skills_from_text("Strong C++ and Python skills")


def test_empty_and_non_string_text_are_handled():
    assert extract_skills_from_text(None) == []
    assert extract_skills_from_text(float("nan")) == []
    assert extract_skills_from_text("   ") == []


# ---------------------------------------------------------------------------
# The employer-brand trap
# ---------------------------------------------------------------------------


def test_employer_org_segment_is_stripped_from_the_title():
    title = "Customer Engineer, Data Management Practice, Google Cloud"
    assert "Google Cloud" not in strip_org_segments(title, "Google")
    assert extract_skills_from_title(title, "Google") == []


def test_product_named_as_a_real_requirement_survives():
    # A segment that *mentions* a product is a role signal; only segments that
    # start with the employer's own org label are dropped.
    assert "Looker" in extract_skills_from_title("Data Analyst, Looker", "Google")
    assert "Machine Learning" in extract_skills_from_title(
        "Software Engineer, Machine Learning, Google Cloud", "Google"
    )


def test_org_stripping_is_per_company():
    title = "Solutions Architect, Google Cloud"
    assert strip_org_segments(title, "Google") != title
    assert strip_org_segments(title, "Microsoft") == title  # not their org unit
    assert strip_org_segments(title, None) == title


def test_own_product_is_stripped_for_the_company_that_sells_it():
    # Snowflake's own postings say "Snowflake"; left alone every specialist
    # would appear to lead in their own product in Task 06.
    assert extract_skills_from_title("Sales Engineer, Snowflake", "Snowflake") == []
    assert "Snowflake" in extract_skills_from_title("Data Engineer, Snowflake",
                                                   "Google")


def test_missing_title_is_empty_not_an_error():
    assert strip_org_segments(None, "Google") == ""
    assert extract_skills_from_title(float("nan"), "Google") == []


# ---------------------------------------------------------------------------
# Combining the paths
# ---------------------------------------------------------------------------


def test_paths_are_unioned_and_provenance_is_recorded():
    out = extract_skills(
        source_skills=["python"],
        title="Machine Learning Engineer",
        description="Experience with Kubernetes",
        company="Google",
    )
    assert out["skills_source"] == ["Python"]
    assert out["skills_title"] == ["Machine Learning"]
    assert out["skills_text"] == ["Kubernetes"]
    assert set(out["skills_final"]) == {"Python", "Machine Learning", "Kubernetes"}
    assert out["skill_provenance"] == "source+text+title"
    assert out["skill_provenance_map"]["Python"] == "source"


def test_a_skill_found_twice_records_both_paths():
    out = extract_skills(source_skills=["python"], description="Python and SQL")
    assert out["skill_provenance_map"]["Python"] == "source+text"
    assert out["skills_final"].count("Python") == 1


def test_no_skills_anywhere_reports_provenance_none():
    out = extract_skills(source_skills=[], title="Program Manager", description="")
    assert out["skills_final"] == []
    assert out["skill_provenance"] == "none"


def test_final_skills_are_ordered_by_category_not_by_path():
    out = extract_skills(source_skills=["sql", "tensorflow"])
    # ML / AI sorts before Programming Language in CATEGORIES.
    assert out["skills_final"] == ["TensorFlow", "SQL"]


# ---------------------------------------------------------------------------
# Per-posting features
# ---------------------------------------------------------------------------


def test_tech_stack_excludes_concepts_and_non_stack_categories():
    feats = skill_features(["Python", "Machine Learning", "PowerPoint", "GDPR"])
    assert feats["tech_stack_tags"] == ["Python"]
    assert feats["method_tags"] == ["Machine Learning"]
    assert feats["n_tech_stack_skills"] == 1
    assert feats["skill_count_final"] == 4


def test_category_counts_have_one_column_per_category():
    feats = skill_features(["Python", "SQL", "TensorFlow"])
    assert feats["n_skills_programming_language"] == 2
    assert feats["n_skills_ml_ai"] == 1
    assert feats["skill_category_count"] == 2
    for category in CATEGORIES:
        assert f"n_skills_{category_slug(category)}" in feats


def test_primary_category_breaks_ties_towards_the_modelling_categories():
    # One ML skill, one language: ML / AI comes first in CATEGORIES, so it wins.
    assert skill_features(["Python", "TensorFlow"])["primary_skill_category"] == "ML / AI"
    # A clear majority beats the ordering.
    assert (
        skill_features(["Python", "SQL", "TensorFlow"])["primary_skill_category"]
        == "Programming Language"
    )


def test_empty_skill_list_gives_zeroed_features():
    feats = skill_features([])
    assert feats["skill_count_final"] == 0
    assert feats["primary_skill_category"] == ""
    assert feats["has_ml_skill"] is False


# ---------------------------------------------------------------------------
# The parquet round-trip trap
# ---------------------------------------------------------------------------


def test_numpy_array_from_parquet_is_treated_as_a_skill_list():
    # Parquet returns list columns as ndarray. An isinstance(..., list) check
    # here dropped every source skill and still "succeeded" — 9% coverage
    # instead of 71%.
    assert as_skill_sequence(np.array(["python", "sql"], dtype=object)) == [
        "python",
        "sql",
    ]
    assert as_skill_sequence(["python"]) == ["python"]


@pytest.mark.parametrize("value", [None, float("nan"), "python", b"python"])
def test_non_list_cells_are_not_mistaken_for_skill_lists(value):
    assert as_skill_sequence(value) is None


# ---------------------------------------------------------------------------
# Pipeline + aggregate tables
# ---------------------------------------------------------------------------


@pytest.fixture()
def postings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": ["a", "b", "c", "d"],
            "job_title_clean": [
                "Data Scientist, Google Cloud",       # org segment, no skill
                "Machine Learning Engineer",          # title-only skill
                "Data Analyst",
                "Data Centre Technician",
            ],
            "cleaned_description": ["", "", "", ""],
            "company_canonical": ["Google"] * 4,
            "skills_parsed": [
                np.array(["python", "sql"], dtype=object),
                None,
                np.array(["sql", "tableau"], dtype=object),
                np.array([], dtype=object),
            ],
            "posting_month": ["2023-01", "2023-02", "2023-07", "2023-08"],
            "job_function": ["Data Science", "Engineering", "Analytics",
                             "Facilities"],
        }
    )


def test_build_posting_features_uses_every_path(postings):
    out, unmapped = build_posting_features(postings)
    assert unmapped == Counter()
    assert out.loc[0, "skills_final"] == ["Python", "SQL"]      # source only
    assert out.loc[1, "skills_final"] == ["Machine Learning"]   # title recovery
    assert out.has_any_skill.tolist() == [True, True, True, False]


def test_title_recovery_is_flagged_only_when_the_source_was_empty(postings):
    out, _ = build_posting_features(postings)
    assert out.skill_recovered_by_title.tolist() == [False, True, False, False]


def test_task_03_skill_count_is_kept_alongside_the_task_04_one(postings):
    # preprocess.py already ships `skill_count`; overwriting it would erase the
    # before/after evidence, and colliding on the name breaks the concat.
    df = postings.assign(skill_count=[2, 0, 2, 0])
    out, _ = build_posting_features(df)
    assert out.skill_count.tolist() == [2, 0, 2, 0]
    assert out.skill_count_final.tolist() == [2, 1, 2, 0]


def test_long_table_is_one_row_per_posting_and_skill(postings):
    out, _ = build_posting_features(postings)
    long = skills_long_table(out)
    assert len(long) == 5
    assert set(long.columns) >= {"job_id", "skill", "skill_category", "provenance",
                                 "in_tech_stack", "posting_month"}
    assert long.loc[long.skill == "Machine Learning", "in_tech_stack"].item() is False


def test_coverage_table_reports_the_denominator_not_just_the_count(postings):
    out, _ = build_posting_features(postings)
    cov = coverage_table(out, "job_function")
    facilities = cov[cov.job_function == "Facilities"].iloc[0]
    assert facilities.postings == 1
    assert facilities.postings_with_skills == 0
    assert facilities.coverage == 0.0


def test_frequency_table_carries_both_denominators(postings):
    out, _ = build_posting_features(postings)
    long = skills_long_table(out)
    freq = skill_frequency_table(long, n_skilled=3, n_total=4)
    sql = freq[freq.skill == "SQL"].iloc[0]
    assert sql.n_postings == 2
    assert sql.share_of_skilled == pytest.approx(2 / 3, abs=1e-4)
    assert sql.share_of_all == pytest.approx(0.5, abs=1e-4)


def test_monthly_share_is_normalised_by_that_months_coverage(postings):
    out, _ = build_posting_features(postings)
    long = skills_long_table(out)
    by_month = skill_by_month_table(long, coverage_table(out, "posting_month"))
    jan = by_month[(by_month.posting_month == "2023-01") & (by_month.skill == "SQL")]
    assert jan.share_of_skilled.item() == 1.0   # 1 of 1 skilled posting
    aug = by_month[by_month.posting_month == "2023-08"]
    assert aug.empty                            # no skills extracted that month


def test_cooccurrence_uses_jaccard_and_a_support_floor():
    long = pd.DataFrame(
        {
            "job_id": ["a", "a", "b", "b", "c"],
            "skill": ["Python", "SQL", "Python", "SQL", "Python"],
        }
    )
    assert skill_cooccurrence_table(long, min_pairs=5).empty     # below support
    pair = skill_cooccurrence_table(long, min_pairs=2).iloc[0]
    assert pair.n_both == 2
    assert pair.jaccard == pytest.approx(2 / 3, abs=1e-4)        # 2 / (3 + 2 - 2)


def test_trend_flags_need_support_in_both_halves():
    df = pd.DataFrame(
        {
            "job_id": [str(i) for i in range(20)],
            "posting_month": ["2023-02"] * 10 + ["2023-09"] * 10,
            "has_any_skill": [True] * 20,
        }
    )
    long = pd.DataFrame(
        {
            "job_id": [str(i) for i in range(20)],
            "skill": ["Python"] * 20,
            "skill_category": ["Programming Language"] * 20,
            "posting_month": ["2023-02"] * 10 + ["2023-09"] * 10,
        }
    )
    trend = skill_trend_table(long, df, min_postings=10)
    row = trend.iloc[0]
    assert row.n_h1 == 10 and row.n_h2 == 10
    assert row.emerging_skill_flag == "stable"

    # Same skill, but only two postings: no claim is made.
    thin = skill_trend_table(long.head(2), df, min_postings=10)
    assert thin.iloc[0].emerging_skill_flag == "insufficient_support"


def test_trend_table_is_empty_without_both_halves():
    df = pd.DataFrame({"job_id": ["a"], "posting_month": ["2023-01"],
                       "has_any_skill": [True]})
    long = pd.DataFrame({"job_id": ["a"], "skill": ["Python"],
                         "skill_category": ["Programming Language"],
                         "posting_month": ["2023-01"]})
    assert skill_trend_table(long, df).empty


def test_skill_matrix_is_binary_and_drops_one_off_skills():
    long = pd.DataFrame(
        {
            "job_id": ["a", "b", "c", "a"],
            "skill": ["Python", "Python", "Python", "Perl"],
        }
    )
    matrix = skill_matrix(long, min_postings=3)
    assert list(matrix.columns) == ["job_id", "skill_python"]
    assert set(matrix.skill_python) == {1}


def test_pipeline_adds_no_personal_data_columns(postings):
    # Standing Task 01 commitment, re-checked in code rather than trusted.
    out, _ = build_posting_features(postings)
    banned = ("email", "phone", "candidate", "applicant", "recruiter")
    assert [c for c in out.columns if any(b in c.lower() for b in banned)] == []
