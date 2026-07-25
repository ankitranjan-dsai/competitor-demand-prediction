"""Tests for the Task 03 preprocessing rules.

These lock in the classification traps found in the real Google data, so a
later refactor can't silently reintroduce them.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import (  # noqa: E402
    canonicalize_company,
    clean_job_title,
    clean_text,
    derive_time_fields,
    extract_job_category,
    extract_job_function,
    extract_seniority,
    is_title_suspect,
    parse_location,
    parse_skill_categories,
    parse_skill_list,
    tokenize,
)


# --------------------------------------------------------------------------
# The data-centre trap: ~11% of Google postings are facilities roles.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "title",
    [
        "Senior Data Center Electrical Engineer",
        "Data Center Mechanical Engineer",
        "Data Center Technician Intern, Summer 2023",
        "Administrative Business Partner, Data Center",
        "Data Center Services Manager",
    ],
)
def test_data_center_roles_are_not_data_roles(title):
    assert extract_job_category(title) == "Data Center / Facilities"


def test_taxonomy_gaps_found_in_the_other_bucket():
    """Roles that fell through to "Other" on the first pass."""
    assert extract_job_category("RF Hardware Engineer") == "Hardware / Silicon"
    assert (
        extract_job_category("Power Management Silicon Validation Engineer")
        == "Hardware / Silicon"
    )
    assert (
        extract_job_category("Senior clinical informatics data architect")
        == "Data Engineering"
    )
    assert (
        extract_job_category("Software Test Engineer, FitbitOS Release Testing")
        == "Software Engineering"
    )
    assert extract_job_category("Student Researcher, PhD") == "Research / Academic"


# --------------------------------------------------------------------------
# job_function is orthogonal to job_category: selling analytics is not
# analytics capacity. Conflating them distorts the demand signal.
# --------------------------------------------------------------------------

def test_sales_roles_separated_from_analytics_capacity():
    title = "Data Analytics Sales Specialist, Google Cloud"
    assert extract_job_category(title) == "Data Analytics / BI"  # domain
    assert extract_job_function(title) == "Sales"                 # work type


def test_field_sales_is_sales_not_infrastructure():
    title = "Field Sales Representative, Corporate, Google Cloud"
    assert extract_job_function(title) == "Sales"


def test_marketing_analyst_is_analytics_work():
    """A Marketing Analyst does analytics; only explicit selling is Sales."""
    assert extract_job_function("Marketing Analyst, Media Lab") == "Analytics"


def test_customer_engineer_is_technical_sales():
    """Google's Customer Engineer is a pre-sales technical role."""
    assert (
        extract_job_function("Customer Engineer, Data Analytics, Google Cloud")
        == "Technical Sales"
    )


def test_core_functions():
    assert extract_job_function("Senior Data Scientist") == "Science / Research"
    assert extract_job_function("Data Center Mechanical Engineer") == "Facilities / Operations"
    assert extract_job_function("Senior Data Engineer") == "Engineering"


# --------------------------------------------------------------------------
# Malformed titles are flagged, not silently analysed.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("title", ["2023 summer intern data", "", None])
def test_malformed_titles_flagged(title):
    assert is_title_suspect(title) is True


def test_good_titles_not_flagged():
    assert is_title_suspect("Senior Data Scientist, Medical Imaging") is False


@pytest.mark.parametrize(
    "title",
    [
        "Head of Product Data Science",
        "Data Analytics Apprenticeship, July 2023",
        "Data Science Associate",
        "Senior Malware Analysis Instructor",
    ],
)
def test_real_but_unusual_titles_are_not_false_positives(title):
    """These are legitimate roles; a noisy flag wastes the Data Quality Lead's time."""
    assert is_title_suspect(title) is False


def test_truncated_title_flagged():
    """Aggregator feeds cut long titles off — the tail is lost, not just words."""
    assert is_title_suspect("gPTO - Analytics & Insights Senior Associate...") is True


def test_non_english_title_flagged():
    """The pipeline is English-only, so a French title must not pass silently."""
    assert is_title_suspect("Ingénieur Data Sciences") is True


def test_real_data_roles_still_classify_correctly():
    assert extract_job_category("Data Scientist, Ads Insight") == "Data Science"
    assert extract_job_category("Machine Learning Engineer") == "AI / ML"
    assert extract_job_category("Senior Data Engineer, Analytics") == "Data Engineering"
    assert extract_job_category("Product Analyst, Data Science") == "Data Science"
    assert (
        extract_job_category("Customer Engineer, Google Cloud")
        == "Cloud / Infrastructure"
    )


# --------------------------------------------------------------------------
# The manager trap: "Program Manager" is an IC title, not people management.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "title",
    [
        "Program Manager Data Science",
        "Technical Program Manager, Data Engineering",
        "Partner Technology Manager, Data Analytics and AI",
        "Product Manager, Cloud AI",
    ],
)
def test_ic_manager_titles_are_flagged_as_ic(title):
    assert extract_seniority(title)["manager_type"] == "ic_manager_title"


@pytest.mark.parametrize(
    "title",
    ["Software Engineering Manager I", "Data Analytics Manager", "Senior Data Scientist Manager"],
)
def test_people_manager_titles(title):
    assert extract_seniority(title)["manager_type"] == "people_manager"


def test_non_manager_title_has_no_manager_type():
    assert extract_seniority("Data Scientist")["manager_type"] == ""


# --------------------------------------------------------------------------
# Seniority ordering — most specific wins.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "title,expected",
    [
        ("Senior Staff Data Scientist, Research", "Senior Staff"),
        ("Staff Data Scientist", "Staff"),
        ("Senior Data Scientist, Medical Imaging", "Senior"),
        ("Data Analytics Lead, Google Cloud", "Lead"),
        ("Data Science Intern, 2023", "Intern"),
        ("Director, Data Centers", "Director+"),
        ("Principal Engineer", "Principal"),
        ("Associate Data Analyst", "Junior"),
        ("Data Scientist", "Mid"),  # unmarked titles default to Mid
    ],
)
def test_seniority_precedence(title, expected):
    assert extract_seniority(title)["seniority_level"] == expected


def test_level_marker_captured_separately():
    assert extract_seniority("Software Engineering Manager I")["level_marker"] == "I"
    assert extract_seniority("Data Scientist III")["level_marker"] == "III"
    assert extract_seniority("Data Scientist")["level_marker"] == ""


# --------------------------------------------------------------------------
# Company canonicalisation.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw",
    ["Google", "Google Inc.", "GOOGLE", "google", "Google LLC", "Google, LLC",
     "Google Taiwan", "GOOGLE ASIA PACIFIC PTE. LTD.", "Google Germany GmbH",
     "Google Czech Republic, s.r.o.", "Google Dubai -"],
)
def test_google_legal_entities_fold_into_google(raw):
    result = canonicalize_company(raw)
    assert result["company_canonical"] == "Google"
    assert result["is_alphabet"] is True


@pytest.mark.parametrize(
    "raw,expected",
    [("Waymo LLC", "Waymo"), ("Verily Life Sciences", "Verily"),
     ("DeepMind", "DeepMind"), ("YouTube", "YouTube"),
     ("Google Fiber", "Google Fiber"),
     ("Mandiant (now part of Google Cloud)", "Mandiant")],
)
def test_alphabet_brands_stay_distinct(raw, expected):
    """Kept separate so Task 05/06 can compare core Google vs Alphabet-wide."""
    result = canonicalize_company(raw)
    assert result["company_canonical"] == expected
    assert result["is_alphabet"] is True


def test_third_party_partner_is_not_alphabet():
    result = canonicalize_company("Geoambiente - Google Cloud Premier Partner")
    assert result["is_alphabet"] is False


def test_title_leaked_into_company_field_is_flagged():
    result = canonicalize_company(
        "Customer Engineer, Machine Learning, Google Cloud - Doha"
    )
    assert result["company_field_suspect"] is True


# --------------------------------------------------------------------------
# Text cleaning and tokenisation.
# --------------------------------------------------------------------------

def test_clean_text_strips_html_and_entities():
    raw = '<div data-testid="x"><p>Risk &amp; Compliance</p><p>ML&nbsp;systems</p></div>'
    out = clean_text(raw)
    assert "<" not in out and "&amp;" not in out and "&nbsp;" not in out
    assert "Risk & Compliance" in out
    # Block tags must become spaces, not glue words together.
    assert "ComplianceML" not in out


def test_clean_text_block_tags_do_not_join_words():
    assert "onetwo" not in clean_text("<p>one</p><p>two</p>")


def test_clean_text_handles_empty_and_non_string():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_accents_normalised():
    assert clean_text("Zürich") == "Zurich"
    assert clean_text("São Paulo") == "Sao Paulo"


def test_tokenizer_preserves_technical_tokens():
    tokens = tokenize("Experience with C++, C#, .NET, Node.js and scikit-learn")
    for expected in ["c++", "c#", ".net", "node.js", "scikit-learn"]:
        assert expected in tokens, f"{expected} was destroyed by tokenisation"


def test_tokenizer_keeps_short_tech_names():
    tokens = tokenize("Strong R, Go, BI and AI background")
    for expected in ["r", "go", "bi", "ai"]:
        assert expected in tokens


def test_tokenizer_removes_stopwords_and_boilerplate():
    tokens = tokenize("You will have experience with the requirements of this role")
    assert "you" not in tokens and "experience" not in tokens
    assert "requirements" not in tokens


def test_eeo_boilerplate_removed():
    text = clean_text(
        "<p>Build ML models.</p><p>Google is proud to be an equal opportunity "
        "employer regardless of race, gender, or sexual orientation.</p>"
    )
    assert "ML models" in text
    assert "equal opportunity" not in text.lower()


# --------------------------------------------------------------------------
# Locations.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,city,country",
    [
        ("Bengaluru, Karnataka, India", "Bengaluru", "India"),
        ("Madrid, Spain", "Madrid", "Spain"),
        ("Singapore", "", "Singapore"),
    ],
)
def test_location_split(raw, city, country):
    result = parse_location(raw)
    assert result["city"] == city
    assert result["location_country"] == country
    assert result["location_is_remote"] is False


@pytest.mark.parametrize(
    "raw,city,region",
    [
        ("Sunnyvale, CA", "Sunnyvale", "CA"),
        ("Atlanta, GA", "Atlanta", "GA"),
        ("New York, NY", "New York", "NY"),
        ("Washington, DC", "Washington", "DC"),
    ],
)
def test_us_state_is_a_region_not_a_country(raw, city, region):
    """"Sunnyvale, CA" is the US — not a country called "CA"."""
    result = parse_location(raw)
    assert result["city"] == city
    assert result["region"] == region
    assert result["location_country"] == "United States"


def test_country_spelling_unified():
    """One country, one spelling, or cross-company grouping splits it."""
    assert parse_location("London, UK")["location_country"] == "United Kingdom"
    assert (
        parse_location("London, United Kingdom")["location_country"]
        == "United Kingdom"
    )


@pytest.mark.parametrize(
    "raw,city,country",
    [
        ("Riyadh Saudi Arabia", "Riyadh", "Saudi Arabia"),
        ("Dubai - United Arab Emirates", "Dubai", "United Arab Emirates"),
    ],
)
def test_country_without_comma_separator(raw, city, country):
    result = parse_location(raw)
    assert result["city"] == city
    assert result["location_country"] == country


def test_multi_location_suffix_stripped_and_flagged():
    """"GA (+5 others)" is one posting open in six places, not a country."""
    result = parse_location("Atlanta, GA (+5 others)")
    assert result["location_country"] == "United States"
    assert result["region"] == "GA"
    assert result["location_multi"] is True
    assert parse_location("Madrid, Spain")["location_multi"] is False


def test_anywhere_is_remote():
    result = parse_location("Anywhere")
    assert result["location_is_remote"] is True


def test_location_accents_normalised():
    assert parse_location("Zürich, Switzerland")["city"] == "Zurich"


def test_missing_location_is_safe():
    assert parse_location(None)["location_clean"] == ""


# --------------------------------------------------------------------------
# Skills parsing.
# --------------------------------------------------------------------------

def test_skill_list_parsed_and_deduplicated():
    """The source repeats skills; double-counting would skew features."""
    assert parse_skill_list("['r', 'python', 'sas', 'sas', 'sql']") == [
        "r", "python", "sas", "sql"
    ]


def test_skill_list_handles_missing():
    assert parse_skill_list(None) == []
    assert parse_skill_list("") == []


def test_skill_list_survives_malformed_input():
    assert parse_skill_list("python, sql") == ["python", "sql"]


def test_skill_categories_parsed():
    parsed = parse_skill_categories(
        "{'analyst_tools': ['sas'], 'programming': ['r', 'python', 'sas', 'sql']}"
    )
    assert parsed["analyst_tools"] == ["sas"]
    assert set(parsed["programming"]) == {"r", "python", "sas", "sql"}


# --------------------------------------------------------------------------
# Titles and time fields.
# --------------------------------------------------------------------------

def test_title_cleaning_removes_intake_season():
    """Cohort noise would otherwise fragment one role across many titles."""
    assert clean_job_title("Data Science Intern, 2023") == "Data Science Intern"
    assert (
        clean_job_title("Data Scientist, Research Intern, PhD, Summer 2024")
        == "Data Scientist, Research Intern, PhD"
    )


def test_title_cleaning_strips_zero_width_chars():
    assert clean_job_title("​Data Center Mechanical Engineer").startswith("Data")


def test_time_fields():
    fields = derive_time_fields("2023-08-15")
    assert fields["posting_month"] == "2023-08"
    assert fields["posting_quarter"] == "2023-Q3"
    assert fields["posting_week"].startswith("2023-W")


def test_time_fields_handle_bad_input():
    assert derive_time_fields("not-a-date")["posting_month"] == ""
