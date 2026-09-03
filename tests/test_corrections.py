"""Tests that keep `docs/corrections.md` honest.

The register records claims a later task disproved. Prose corrections rot: a
rebuild changes a number, nobody re-reads the markdown, and the repo goes back
to asserting the thing it corrected — quietly, because nothing fails.

So every quantitative claim in the register is checked here against the
committed evidence table it cites. These are consistency tests, not unit
tests: they read committed CSVs rather than fixtures, which is why they skip
cleanly on a checkout where a specialist's Task 05 tables are absent.

    python -m pytest tests/test_corrections.py -q
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTER = REPO_ROOT / "docs" / "corrections.md"
TABLES = REPO_ROOT / "members" / "ankit-google" / "task-05-tables"

pytestmark = pytest.mark.skipif(
    not TABLES.is_dir(),
    reason="Google Task 05 tables not present in this checkout",
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _cell(text: str) -> str:
    """Strip markdown emphasis so a bolded cell compares like a plain one."""
    return text.replace("*", "").replace("`", "").strip()


def markdown_table(heading_contains: str, nth: int = 0) -> list[dict[str, str]]:
    """Return the nth markdown table appearing after a matching heading.

    Written against the register's own formatting rather than a general
    markdown parser: the register is ours, and a strict reader is the point —
    if someone restyles the table, this fails loudly instead of silently
    matching nothing.
    """
    lines = REGISTER.read_text(encoding="utf-8").splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("#") and heading_contains.lower() in line.lower()
    )
    tables: list[list[list[str]]] = []
    rows: list[list[str]] = []
    for line in lines[start + 1:]:
        stripped = line.lstrip("> ").rstrip()
        if not stripped.startswith("|"):
            if rows:
                tables.append(rows)
                rows = []
            if line.startswith("#") and tables:
                break  # next entry; stop before its tables
            continue
        cells = [_cell(c) for c in stripped.strip("|").split("|")]
        if all(set(c) <= {"-", ":"} and c for c in cells):
            continue  # separator row
        rows.append(cells)
    if rows:
        tables.append(rows)
    if len(tables) <= nth:
        raise AssertionError(
            f"no table {nth} under heading {heading_contains!r} "
            f"({len(tables)} found)"
        )
    header, *body = tables[nth]
    return [dict(zip(header, row)) for row in body]


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


# --------------------------------------------------------------------------
# the register itself
# --------------------------------------------------------------------------


def test_register_lists_every_correction_as_corrected():
    """An entry with an open status is a limitation, not a correction."""
    entries = re.findall(r"^## (C\d+) —", REGISTER.read_text(encoding="utf-8"), re.M)
    index = markdown_table("Corrections Register")
    listed = [re.match(r"\[(C\d+)\]", row["#"]).group(1) for row in index]
    assert listed == entries, "the index and the entries have drifted apart"
    for row in index:
        assert row["Status"] == "✅ corrected", row


def test_every_register_anchor_resolves():
    """The in-page links are how the task reports reach their correction."""
    text = REGISTER.read_text(encoding="utf-8")
    headings = {
        "#"
        + re.sub(r"[^a-z0-9 -]", "", line.lstrip("# ").lower()).replace(" ", "-")
        for line in text.splitlines()
        if line.startswith("##")
    }
    anchors = set(re.findall(r"\]\((#[a-z0-9-]+)\)", text))
    assert anchors, "register has no internal links"
    assert anchors <= headings, anchors - headings


def test_task_reports_point_at_the_register():
    """A correction nobody can find from the corrected document is not one."""
    corrected = [
        Path("docs/task-01-data-sources-and-legal.md"),
        Path("docs/task-03-preprocessing-methods.md"),
        Path("members/ankit-google/task-02-data-collection-report.md"),
        Path("members/ankit-google/task-03-preprocessing-report.md"),
        Path("members/ankit-google/task-04-skill-extraction-report.md"),
        Path("members/ankit-google/task-05-trend-report.md"),
        Path("docs/task-06-competitor-comparison-methods.md"),
        Path("members/ankit-google/task-06-comparison-report.md"),
        Path("docs/task-04-skill-taxonomy.md"),
        Path("docs/task-08-company-similarity-methods.md"),
    ]
    for rel in corrected:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "corrections.md#" in text, f"{rel} carries no link to the register"


# --------------------------------------------------------------------------
# C1 — the spikes are publisher batches
# --------------------------------------------------------------------------


def test_c1_every_flagged_spike_is_a_publisher_batch():
    """The register says all three; if a rebuild finds a fourth, re-read it."""
    attribution = read_table("spike-attribution.csv")
    claimed = markdown_table("C1")
    assert len(attribution) == len(claimed)
    assert set(attribution.verdict) == {"publisher_batch"}

    by_period = attribution.set_index("period")
    for row in claimed:
        got = by_period.loc[row["Week"]]
        assert row["Verdict"] == got.verdict
        assert int(row["Postings"]) == int(got.postings)
        assert row["Top publisher"] == got.top_publisher
        assert float(row["Share of excess explained"]) == pytest.approx(
            got.excess_explained, abs=0.005
        )


def test_c1_february_has_fewer_publishers_than_its_neighbours():
    """The claim that February is a collection gap, not a hiring freeze."""
    volume = read_table("volume-by-month.csv").set_index("period")
    feb = volume.loc["2023-02"]
    assert feb.n_publishers < volume.loc["2023-01"].n_publishers
    assert feb.n_publishers < volume.loc["2023-03"].n_publishers
    assert not feb.is_partial, "February is fully observed — the gap is coverage"


def test_c1_volume_direction_is_still_unidentified():
    """If the panel treatments ever agree, C1's consequence needs rewriting."""
    sensitivity = read_table("panel-sensitivity.csv")
    treatments = ["raw_index", "balanced_index", "chained_index", "bilateral_index"]
    december = sensitivity[sensitivity.period == sensitivity.period.max()]
    indices = december[treatments].iloc[0]
    assert indices.min() < 100 < indices.max(), "treatments no longer straddle flat"
    assert indices.max() - indices.min() > 50, "treatments have converged — re-read C1"


# --------------------------------------------------------------------------
# C2 — the overturned skill trends
# --------------------------------------------------------------------------


def test_c2_each_skill_matches_the_verdict_table():
    verdicts = read_table("skill-stratified-verdicts.csv").set_index("skill")
    claimed = markdown_table("C2")
    assert len(claimed) == 10, "Task 04 named ten headline movers"

    for row in claimed:
        skill = row["Skill"]
        assert skill in verdicts.index, f"{skill} is no longer in the verdict table"
        stated = row["Task 05 verdict"].split(" (")[0]
        assert stated == verdicts.loc[skill].verdict, skill


def test_c2_looker_is_the_only_reversal_among_the_ten():
    verdicts = read_table("skill-stratified-verdicts.csv").set_index("skill")
    claimed = markdown_table("C2")
    reversed_in_register = {r["Skill"] for r in claimed if "reversed" in r["Outcome"]}
    reversed_in_data = {
        r["Skill"]
        for r in claimed
        if verdicts.loc[r["Skill"]].contradiction == "reversed"
    }
    assert reversed_in_register == reversed_in_data == {"Looker"}


def test_c2_looker_falls_inside_every_supported_function():
    """The Simpson's-paradox claim, read straight off the stratified table."""
    within = read_table("skill-trend-within-function.csv")
    looker = within[(within.skill == "Looker") & within.meets_support]
    assert len(looker) == 3
    assert (looker.share_delta < 0).all()

    pooled = read_table("skill-stratified-verdicts.csv").set_index("skill").loc["Looker"]
    assert pooled.pooled_direction == "up"
    assert pooled.overturned_by_stratification


def test_c2_confirmed_skills_hold_in_every_segment():
    verdicts = read_table("skill-stratified-verdicts.csv").set_index("skill")
    claimed = markdown_table("C2")
    for row in claimed:
        if "confirmed" not in row["Outcome"]:
            continue
        verdict = verdicts.loc[row["Skill"]].verdict
        assert verdict in {"rising_in_all_segments", "falling_in_all_segments"}
        assert not verdicts.loc[row["Skill"]].overturned_by_stratification


# --------------------------------------------------------------------------
# C3 — posting_date is an ingestion date
# --------------------------------------------------------------------------


def test_c3_weekends_are_under_represented():
    seasonality = read_table("seasonality.csv")
    dow = seasonality[seasonality.cycle == "day_of_week"].set_index("level")
    weekend = dow.loc[["Saturday", "Sunday"], "share"].sum()
    assert weekend < 2 / 7, "a publication date would not avoid weekends"
    assert dow.loc["Saturday", "index_vs_uniform"] < 0.9
    assert dow.loc["Sunday", "index_vs_uniform"] < 0.9


def test_c3_day_of_week_is_identifiable_but_month_of_year_is_not():
    """C3 leans on the day-of-week index, so it has to be estimable at all."""
    seasonality = read_table("seasonality.csv")
    by_cycle = seasonality.groupby("cycle").identifiable.all()
    assert by_cycle["day_of_week"]
    assert not by_cycle["month_of_year"]


# --------------------------------------------------------------------------
# C4 — Google is 846 postings, not 848
# --------------------------------------------------------------------------

TABLES_06 = REPO_ROOT / "members" / "ankit-google" / "task-06-tables"

needs_task_06 = pytest.mark.skipif(
    not TABLES_06.is_dir(), reason="Task 06 tables not present in this checkout"
)


@needs_task_06
def test_c4_the_two_dropped_rows_are_the_ones_the_register_names():
    audit = pd.read_csv(TABLES_06 / "employer-matching-audit.csv")
    dropped = audit[(audit.company == "google") & (audit.decision == "excluded")]
    claimed = markdown_table("C4")
    assert len(claimed) == 2

    by_string = dropped.set_index("employer_string")
    for row in claimed:
        name = row["Employer string"]
        assert name in by_string.index, f"{name} is no longer excluded"
        assert row["Audit reason"] == by_string.loc[name].reason

    # And the count they explain: 848 matched by name, 846 after the rule.
    manifest = pd.read_csv(TABLES_06 / "competitor-set-manifest.csv")
    google = manifest.set_index("company").loc["google"]
    assert int(google.postings_after_dedup) == 846
    assert int(dropped.postings.sum()) == 2


@needs_task_06
def test_c4_only_the_raw_index_moves():
    """The register's claim is narrow: raw shifts, the panels do not."""
    old = read_table("panel-sensitivity.csv")
    new = pd.read_csv(TABLES_06 / "volume-panel-sensitivity.csv")
    new = new[new.company == "google"]
    december = old.period.max()
    a = old[old.period == december].iloc[0]
    b = new[new.period == december].iloc[0]

    claimed = {r["Treatment"]: r for r in markdown_table("C4", nth=1)}
    assert set(claimed) == {"raw", "balanced", "chained", "bilateral"}
    for treatment, row in claimed.items():
        column = f"{treatment}_index"
        assert float(row["Task 05 (848)"]) == pytest.approx(a[column], abs=0.005)
        assert float(row["Task 06 (846)"]) == pytest.approx(b[column], abs=0.005)

    assert a.raw_index != b.raw_index
    for treatment in ("balanced", "chained", "bilateral"):
        assert a[f"{treatment}_index"] == pytest.approx(b[f"{treatment}_index"])


@needs_task_06
def test_c4_task_05s_own_numbers_are_left_standing():
    """The register is a register: the corrected document keeps its wording."""
    report = (REPO_ROOT / "members" / "ankit-google" /
              "task-05-trend-report.md").read_text(encoding="utf-8")
    assert "848" in report
    assert "corrections.md#" in report


# --------------------------------------------------------------------------
# C5 — Task 06's H1 panel counts February
# --------------------------------------------------------------------------

TABLES_07 = REPO_ROOT / "members" / "ankit-google" / "task-07-tables"

needs_task_07 = pytest.mark.skipif(
    not TABLES_07.is_dir(), reason="Task 07 tables not present in this checkout"
)


def _number(text: str) -> float:
    """Parse a register cell: Unicode minus, trailing `pp` or `%`, bolding."""
    cleaned = _cell(text).replace("−", "-").replace("pp", "").replace("%", "")
    return float(cleaned.strip())


def _key(company: str) -> str:
    return _cell(company).lower()


@needs_task_07
def test_c5_february_is_the_share_of_h1_the_register_claims():
    """The whole correction rests on 97 of 620 — check both ends of it."""
    series = pd.read_csv(TABLES_07 / "panel-share-series.csv")
    february = series[series.period == "2023-02"]

    assert not february.is_observed.any(), "February is no longer flagged unobserved"
    assert february.denominator.nunique() == 1
    assert int(february.denominator.iloc[0]) == 97
    assert int(february.numerator.sum()) == 97

    halves = pd.read_csv(TABLES_06 / "relative-share-by-half.csv")
    assert halves.panel_h1_total.nunique() == 1
    h1_total = int(halves.panel_h1_total.iloc[0])
    assert h1_total == 620, "Task 06's H1 base changed; re-read C5"

    correction = pd.read_csv(TABLES_07 / "february-correction.csv")
    without = h1_total - int(february.numerator.sum())
    assert without == 523
    # The corrected shares have to be shares of that smaller base.
    recomputed = (
        (halves.set_index("company").h1_postings
         - february.set_index("key").numerator) / without
    )
    for company, share in recomputed.items():
        got = correction.set_index("company").loc[company]
        assert got.h1_share_without_february == pytest.approx(share, abs=5e-4), company


@needs_task_07
def test_c5_the_february_distribution_table_matches_the_data():
    """February is not a neutral month — that is why the magnitudes move."""
    series = pd.read_csv(TABLES_07 / "panel-share-series.csv")
    february = series[series.period == "2023-02"].set_index("key")
    gate = pd.read_csv(TABLES_07 / "forecastability-gate.csv").set_index("key")

    claimed = markdown_table("C5", nth=0)
    assert len(claimed) == 6, "all six companies belong in the February table"

    for row in claimed:
        key = _key(row["Company"])
        assert int(row["Feb postings"]) == int(february.loc[key].numerator), key
        assert _number(row["Feb share of the month"]) == pytest.approx(
            february.loc[key].share * 100, abs=0.05
        ), key
        assert _number(row["Mean monthly panel share"]) == pytest.approx(
            gate.loc[key].mean_share * 100, abs=0.05
        ), key

    meta = february.loc["meta"]
    assert meta.numerator == february.numerator.max(), "Meta no longer tops February"
    assert meta.share > gate.loc["meta"].mean_share, "Meta's February is no longer high"


@needs_task_07
def test_c5_every_sign_survives_excluding_february():
    """The correction's central claim: magnitudes move, conclusions do not."""
    correction = pd.read_csv(TABLES_07 / "february-correction.csv")
    assert correction.sign_unchanged.all()
    assert (
        np.sign(correction.task06_published_delta_pp)
        == np.sign(correction.corrected_delta_pp)
    ).all()


@needs_task_07
def test_c5_the_recompute_table_matches_the_committed_correction():
    correction = pd.read_csv(TABLES_07 / "february-correction.csv").set_index("company")
    halves = pd.read_csv(TABLES_06 / "relative-share-by-half.csv").set_index("company")
    february = pd.read_csv(TABLES_07 / "panel-share-series.csv")
    february = february[february.period == "2023-02"].set_index("key")

    claimed = markdown_table("C5", nth=1)
    assert len(claimed) == 6

    for row in claimed:
        key = _key(row["Company"])
        with_feb = int(halves.loc[key].h1_postings)
        assert int(row["H1 with Feb"]) == with_feb, key
        assert int(row["H1 without"]) == with_feb - int(february.loc[key].numerator), key
        # The left column is Task 06's own rounding, so allow the rounding step.
        assert _number(row["Task 06 published"]) == pytest.approx(
            correction.loc[key].task06_published_delta_pp, abs=0.011
        ), key
        assert _number(row["Corrected"]) == pytest.approx(
            correction.loc[key].corrected_delta_pp, abs=0.005
        ), key
        assert row["Sign"] == "unchanged", key


@needs_task_07
def test_c5_meta_moves_most_and_google_deepens():
    """Both named magnitudes in the register, read off the committed table."""
    correction = pd.read_csv(TABLES_07 / "february-correction.csv").set_index("company")
    moves = correction.change_pp.abs()
    assert moves.idxmax() == "meta"
    assert correction.loc["meta"].corrected_delta_pp > (
        3 * correction.loc["meta"].task06_published_delta_pp
    ), "Meta's rise no longer more than triples"
    google = correction.loc["google"]
    assert google.corrected_delta_pp < google.task06_published_delta_pp < 0
    assert google.h1_share_without_february > google.h1_share_with_february


# --------------------------------------------------------------------------
# C6 — neither concepts nor rare skills dominate a similarity score
# --------------------------------------------------------------------------

TABLES_08 = REPO_ROOT / "members" / "ankit-google" / "task-08-tables"

needs_task_08 = pytest.mark.skipif(
    not TABLES_08.is_dir(), reason="Task 08 tables not present in this checkout"
)


def _triple(text: str) -> list[tuple[float, float]]:
    """Parse a `min / mean / max` register cell into (value, tolerance) pairs.

    The register prints each figure at the precision it deserves — 0.00%, 0.36%,
    80.5% — so the check is "is it right at the precision published", half a
    unit in the last printed place, rather than one tolerance for all of them.
    """
    parts = [_cell(part).replace("−", "-").replace("%", "").strip()
             for part in _cell(text).split("/")]
    assert len(parts) == 3, text
    return [(float(part), 0.5 * 10 ** -len(part.partition(".")[2])) for part in parts]


@needs_task_08
def test_c6_the_contribution_table_matches_the_committed_shares():
    """Every cell of the register's headline table, recomputed from the CSV."""
    contrib = pd.read_csv(TABLES_08 / "numerator-contribution.csv")
    assert len(contrib) == 15, "the six-company panel has 15 pairs"

    columns = {
        "Concepts": ("share_concept", "n_concept"),
        "Skills in ≤ 1 posting": ("share_postings_le_1", "n_postings_le_1"),
        "Skills in ≤ 10 postings": ("share_postings_le_10", "n_postings_le_10"),
        "Top 5 skills of the pair": ("share_top5", None),
    }
    claimed = markdown_table("C6")
    assert len(claimed) == len(columns)

    for row in claimed:
        group = _cell(row["Group"]).split(" (")[0]
        share, count = columns[group]
        values = contrib[share] * 100
        claims = _triple(row["Share of the cosine numerator (min / mean / max)"])
        got = (values.min(), values.mean(), values.max())
        for (claim, tol), actual in zip(claims, got):
            assert claim == pytest.approx(actual, abs=tol), (group, claim, actual)
        if count is not None:
            assert contrib[count].nunique() == 1, count
            assert int(row["Skills"]) == int(contrib[count].iloc[0]), group
        else:
            assert int(row["Skills"]) == 5


@needs_task_08
def test_c6_a_single_posting_skill_contributes_exactly_zero():
    """Not "almost nothing" — zero, because the product term has a zero in it."""
    contrib = pd.read_csv(TABLES_08 / "numerator-contribution.csv")
    assert (contrib.share_postings_le_1 == 0.0).all()
    assert contrib.n_postings_le_1.iloc[0] > 0, "no single-posting skills left to check"


@needs_task_08
def test_c6_five_skills_carry_four_fifths_of_every_score():
    """The other half of the arithmetic: weight sits in the head, not the tail."""
    contrib = pd.read_csv(TABLES_08 / "numerator-contribution.csv")
    assert contrib.share_top5.min() > 0.70
    assert contrib.share_top5.mean() > 0.80
    assert (contrib.share_top5 > contrib.share_postings_le_10 * 1000).all()


@needs_task_08
def test_c6_removing_every_concept_skill_leaves_the_ranking_identical():
    """The claim that would have to fail for §2.3's prediction to be right."""
    removal = pd.read_csv(TABLES_08 / "concept-skill-removal.csv")
    assert set(removal.metric) == {"cosine"}, "C6 is a statement about cosine"
    assert (removal.rank_move == 0).all(), "a pair moved — re-read C6"
    assert removal[["rank_with", "rank_without"]].corr(method="spearman").iloc[0, 1] == 1.0
    assert removal.delta.abs().max() == pytest.approx(0.0020, abs=5e-5)
    assert removal.with_group.min() == pytest.approx(0.4961, abs=5e-5)
    assert removal.with_group.max() == pytest.approx(0.9174, abs=5e-5)


@needs_task_08
def test_c6_the_prediction_would_have_been_right_for_a_set_metric():
    """The general lesson only holds if the two metric families really differ."""
    concordance = pd.read_csv(TABLES_08 / "metric-concordance.csv")
    pair = concordance[
        (concordance.metric_a == "cosine") & (concordance.metric_b == "jaccard_supported")
    ]
    assert len(pair) == 1, "cosine vs jaccard is the comparison C6 rests on"
    assert float(pair.rank_correlation.iloc[0]) == pytest.approx(-0.04, abs=0.005)
    assert not bool(pair.same_family.iloc[0])


@needs_task_08
def test_c6_the_support_sweep_is_not_a_robustness_check():
    """C6 says the sweep asks a different question — it has to move something."""
    support = pd.read_csv(TABLES_08 / "support-sensitivity.csv")
    assert support.rank_move.abs().sum() > 0, (
        "restricting to core skills changed no ranking — then it is a no-op, "
        "not a different question"
    )
    assert support.n_core.iloc[0] < support.n_all.iloc[0]


@needs_task_08
def test_c6_the_taxonomy_keeps_its_wording():
    """A register, not an eraser: the disproved sentences stay where they were."""
    raw = (REPO_ROOT / "docs" / "task-04-skill-taxonomy.md").read_text(encoding="utf-8")
    taxonomy = " ".join(raw.split())  # the sentences are hard-wrapped
    assert "dominate every similarity score in Task 08" in taxonomy
    assert "it would dominate cosine similarity in Task 08" in taxonomy
    assert raw.count("corrections.md#c6") == 2, "both passages carry a pointer"


# --------------------------------------------------------------------------
# C7 — Task 08's name, and the handover written for a task that never existed
# --------------------------------------------------------------------------


def test_c7_the_brief_and_the_readme_agree_on_the_name():
    """C7's evidence is committed prose, so the check is that it still says it."""
    readme = " ".join((REPO_ROOT / "README.md").read_text(encoding="utf-8").split())
    assert "| 08 | Company Similarity Scoring |" in readme
    methods = REPO_ROOT / "docs" / "task-08-company-similarity-methods.md"
    assert methods.is_file(), "C7 cites the Task 08 methods document"
    assert "# Task 08 — Company Similarity Scoring" in methods.read_text(
        encoding="utf-8"
    )


def test_c7_both_task_06_documents_keep_their_wrong_name_and_gain_a_pointer():
    """A register, not an eraser — for a naming claim as much as a numeric one."""
    for rel, name in (
        ("docs/task-06-competitor-comparison-methods.md", "Task 08 (Visualisation)"),
        ("members/ankit-google/task-06-comparison-report.md", "Task 08 (Evaluation)"),
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert name in text, f"{rel} no longer carries its original wording"
        assert "corrections.md#c7" in text, f"{rel} carries no pointer to C7"


@needs_task_08
def test_c7_mix_standardisation_really_is_third_order():
    """The reason Task 06's "evaluate against the standardised shares" is wrong."""
    mix = pd.read_csv(TABLES_08 / "mix-sensitivity.csv")
    vendor = pd.read_csv(TABLES_08 / "vendor-sensitivity.csv")
    assert mix.mix_effect.abs().max() < vendor.delta_all_products.abs().max() / 3, (
        "the role-mix effect is no longer small next to the own-product lever — "
        "re-read C7"
    )
    assert mix[["rank_crude", "rank_standardised"]].corr(
        method="spearman"
    ).iloc[0, 1] == pytest.approx(0.90, abs=0.005)


@needs_task_08
def test_c7_task_08_has_no_skill_level_significance_test():
    """The other half: the unit is the pair, so there is nothing to FDR-correct."""
    skill_keyed = []
    for path in sorted(TABLES_08.glob("*.csv")):
        columns = set(pd.read_csv(path, nrows=1).columns)
        assert not columns & {"p_value", "p_adjusted", "significant"}, path.name
        if "skill" in columns:
            skill_keyed.append(path.name)

    # Exactly one table is keyed by skill, and it is a description of the
    # profiles rather than an inference about them: one column per company and
    # nothing else. Every other table is keyed by the pair.
    assert skill_keyed == ["skill-profiles.csv"], skill_keyed
    profiles = pd.read_csv(TABLES_08 / "skill-profiles.csv")
    assert list(profiles.columns)[0] == "skill"
    assert len(profiles.columns) == 7, "skill-profiles gained a non-company column"


# --------------------------------------------------------------------------
# C8 — unanimity, and the tests that vanish when the floor rises
# --------------------------------------------------------------------------

TABLES_09 = REPO_ROOT / "members" / "ankit-google" / "task-09-tables"

needs_task_09 = pytest.mark.skipif(
    not TABLES_09.is_dir(),
    reason="Google Task 09 tables not present in this checkout",
)


@needs_task_09
def test_c8_the_register_floor_table_matches_the_sweep():
    """Six rows of prose against the committed recount, cell by cell."""
    register = {row["Company"]: row for row in markdown_table("C8 —", nth=1)}
    committed = pd.read_csv(TABLES_09 / "unanimity-verdict.csv")
    assert set(register) == set(committed.company)
    for _, row in committed.iterrows():
        entry = register[row.company]
        assert entry["Verdict at floor 0"] == row.verdict_at_floor_0
        assert entry["Floors confirmed"] == row.floors_confirmed
        low, high = row.publishers_tested_range.split("-")
        assert entry["Publishers tested"] == f"{low} → {high}"


@needs_task_09
def test_c8_three_companies_gain_confirmation_by_dropping_tests():
    """The correction in one line: a verdict that improves as evidence goes."""
    committed = pd.read_csv(TABLES_09 / "unanimity-verdict.csv")
    gained = committed[committed.confirmation_gained_by_dropping_tests]
    assert sorted(gained.company) == ["google", "microsoft", "snowflake"]
    assert (gained.verdict_at_floor_0 == "mixed").all()
    assert (gained.min_tested < 6).all()


@needs_task_09
def test_c8_the_sign_test_power_table_matches_the_register():
    power = pd.read_csv(TABLES_09 / "sign-test-power.csv")
    quoted = markdown_table("C8 —", nth=2)[0]
    for _, row in power.iterrows():
        assert quoted[str(row.publishers_tested)] == f"{row.sign_test_p:.4f}"
    # Six is the smallest panel on which unanimity can reach 0.05 at all.
    assert power[power.clears_005].publishers_tested.min() == 6


@needs_task_09
def test_c8_nvidia_clears_005_only_by_admitting_one_posting_cells():
    """The verdict never flips; the evidence under it evaporates."""
    sweep = pd.read_csv(TABLES_09 / "publisher-cell-floor.csv")
    nvidia = sweep[sweep.company == "nvidia"].set_index("cell_floor")
    assert (nvidia.verdict == "confirmed").all(), "NVIDIA is confirmed at every floor"
    assert list(nvidia.clears_005) == [True, False, False, False]
    assert list(nvidia.publishers_tested) == [6, 4, 4, 1]
    assert nvidia.loc[10, "sign_test_p"] == pytest.approx(1.0)


@needs_task_09
def test_c8_the_pooled_direction_never_moves_with_the_floor():
    """What C8 does *not* correct — Task 06's directions stand."""
    sweep = pd.read_csv(TABLES_09 / "publisher-cell-floor.csv")
    for company, rows in sweep.groupby("company"):
        assert rows.pooled_direction.nunique() == 1, company
        assert rows.pooled_log_share_change.nunique() == 1, company


@needs_task_09
def test_c8_no_relative_share_claim_publishes_unqualified():
    """The consequence, enforced: Task 06 §2's unqualified sentence is gone."""
    ledger = pd.read_csv(TABLES_09 / "claim-ledger.csv")
    shares = ledger[ledger.family == "relative_share"]
    published = shares[shares.status.str.startswith("published")]
    assert len(published) == 6
    assert (published.status == "published_qualified").all()
    assert published.clause.str.contains("floor-dependent, see C8").all()


def test_c8_both_task_06_documents_keep_their_wording_and_gain_a_pointer():
    for rel, quote in (
        ("docs/task-06-competitor-comparison-methods.md",
         "the single cross-company volume finding in this"),
        ("members/ankit-google/task-06-comparison-report.md",
         "**Only NVIDIA is `confirmed`** (6/6)"),
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert quote in text, f"{rel} no longer carries its original wording"
        assert "corrections.md#c8" in text, f"{rel} carries no pointer to C8"


# --------------------------------------------------------------------------
# C9 — the prediction that a deck cannot be checked
# --------------------------------------------------------------------------
#
# C9 is a correction to a claim about testability, so a test for it is the
# only honest form the correction can take: an entry that says "this is
# checkable now" and is itself unchecked would be making its predecessor's
# mistake in the opposite direction.

TABLES_10 = REPO_ROOT / "members" / "ankit-google" / "task-10-tables"
REPORT_09 = REPO_ROOT / "members" / "ankit-google" / "task-09-insight-report.md"

needs_task_10 = pytest.mark.skipif(
    not TABLES_10.is_dir(),
    reason="Task 10 build outputs not present in this checkout",
)

#: A rule name is the fourth argument of `_v` and the second of the `here`
#: partial that wraps it. Read off the source rather than imported, because
#: the claim under test is about *wording*: the register names rules in prose,
#: and prose does not fail when a rule is renamed.
RULE_LITERAL = re.compile(
    r'here\(\s*-?\w+,\s*"([a-z_]+)"'
    r'|_v\(\s*[^,\n]+,\s*[^,\n]+,\s*-?\w+,\s*"([a-z_]+)"'
)

PREDICTION = (
    "Task 10 is the final presentation, and it is the first task in this "
    "project whose output is not checkable by a test."
)


def lint_rule_names() -> set[str]:
    """Every rule `src/present.py` can name in a violation."""
    source = (REPO_ROOT / "src" / "present.py").read_text(encoding="utf-8")
    return {first or second for first, second in RULE_LITERAL.findall(source)}


def entry_text(heading_contains: str) -> str:
    """The register entry under a heading, up to the next one."""
    lines = REGISTER.read_text(encoding="utf-8").splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("#") and heading_contains.lower() in line.lower()
    )
    end = next(
        (
            i
            for i, line in enumerate(lines[start + 1:], start + 1)
            if line.startswith("#")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def backticked_rules(text: str) -> set[str]:
    """Rule names a passage claims exist, from its right-hand table cells."""
    found = set()
    for line in text.splitlines():
        line = line.lstrip("> ").rstrip()
        if not line.startswith("|"):
            continue
        found.update(re.findall(r"`([a-z_]+)`", line.strip("|").split("|")[-1]))
    return found


def c9_marking() -> str:
    """The block Task 10 inserted into Task 09 §13, and nothing around it.

    Scoped to the one quote block rather than every ``>`` line in the report,
    because §13 is not the only passage in that file carrying a correction.
    """
    lines = REPORT_09.read_text(encoding="utf-8").splitlines()
    pointer = next(
        i for i, line in enumerate(lines) if "corrections.md#c9" in line
    )
    end = next(
        (i for i, line in enumerate(lines[pointer:], pointer)
         if not line.startswith(">")),
        len(lines),
    )
    return "\n".join(line.lstrip("> ") for line in lines[pointer:end])


def presentation_report() -> dict:
    import json

    path = TABLES_10.parent / "task-10-presentation-report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_c9_task_09_keeps_its_prediction_and_gains_a_pointer():
    """A register, not an eraser — the sentence C9 overturns stays where it was."""
    report = REPORT_09.read_text(encoding="utf-8")
    flat = " ".join(report.split())
    assert PREDICTION in flat, "Task 09 §13 no longer carries its prediction"
    assert "corrections.md#c9" in report, "§13 carries no pointer to C9"
    quoted = " ".join(
        line.lstrip("> ") for line in entry_text("C9 —").splitlines()
    ).split()
    assert PREDICTION in " ".join(quoted), (
        "the register no longer quotes the sentence it corrects"
    )


def test_c9_every_rule_the_correction_names_is_one_the_linter_can_fire():
    """C9's argument is that each §13 instruction became a named rule.

    The six-row table is the argument in full, so an unrecognised name in it
    is the correction quietly ceasing to be true. That each rule *fires* on
    the case it was written for is ``tests/test_presentation.py``'s job; this
    checks only that the register is naming rules that exist.
    """
    vocabulary = lint_rule_names()
    entry = entry_text("C9 —")
    named = backticked_rules(entry)

    instructions = [
        line for line in entry.splitlines()
        if line.startswith("| ") and "`" in line.split("|")[-2]
    ]
    assert len(instructions) == 6, "C9 maps §13's six instructions, one per row"

    unknown = named - vocabulary
    assert not unknown, f"C9 names rules src/present.py does not emit: {unknown}"
    assert {"claim_exists", "refusals_intact", "clause_travels",
            "correction_carried", "asset_exists"} <= named

    marking = c9_marking()
    in_place = set(re.findall(r"`([a-z_]+)`", marking))
    assert in_place, "the in-place marking in §13 names no rule at all"
    assert in_place <= vocabulary, (
        f"the in-place marking names rules that do not exist: "
        f"{in_place - vocabulary}"
    )
    assert in_place <= named, "§13 is marked with a rule C9 does not argue for"


@needs_task_10
def test_c9_the_deck_the_register_describes_is_the_deck_that_shipped():
    """Every count in the entry, read back off the build's own report."""
    prose = " ".join(entry_text("C9 —").split())
    report = presentation_report()
    rules = len(lint_rule_names())

    assert f"{report['deck']['slides']} slides and " \
           f"{report['deck']['bullets']} bullets" in prose
    assert f"{rules} rules run over the deck and the " \
           f"{report['bank']['questions']}-question mentor bank" in prose
    assert f"{rules} rules run over it" in " ".join(c9_marking().split()), (
        "the marking left in Task 09 §13 quotes a different number of rules"
    )

    collected = len(pd.read_csv(TABLES_10 / "deck-lint.csv"))
    assert collected == report["deck"]["violations"] == 0, (
        "the shipped deck no longer lints clean — rebuild before trusting C9"
    )
    assert len(pd.read_csv(TABLES_10 / "qa-lint.csv")) == \
        report["bank"]["violations"] == 0


@needs_task_10
def test_c9_the_test_count_it_quotes_is_the_suite_that_exists():
    """The staleness C9 caught in a README, applied to C9 itself."""
    import subprocess
    import sys

    suite = REPO_ROOT / "tests" / "test_presentation.py"
    run = subprocess.run(
        [sys.executable, "-m", "pytest", str(suite), "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    found = re.search(r"^(\d+) tests? collected", run.stdout, re.M)
    assert found, f"could not collect {suite.name}:\n{run.stdout[-800:]}"
    prose = " ".join(entry_text("C9 —").split())
    assert f"{found.group(1)} tests in" in prose, (
        f"the register quotes a stale suite size; {suite.name} now collects "
        f"{found.group(1)}"
    )


@needs_task_10
def test_c9_the_audit_still_runs_the_checks_that_found_the_four_defects():
    """C9's second half: the defects were found by checks, not by reading.

    The failures themselves are Task 10's to fix, so this pins the checks
    rather than their verdicts — the four defects are only evidence for C9 if
    something in the repository was looking for them.
    """
    audit = pd.read_csv(TABLES_10 / "workspace-audit.csv")
    assert "suite_size_current" in set(audit.check), "nothing checks README staleness"
    folders = audit[audit.check == "working_dir_used"]
    assert set(folders.subject) == {"notebooks", "weekly-reports", "meeting-minutes"}, (
        "C9 cites three directories named in the layout and never used"
    )


# --------------------------------------------------------------------------
# Task 11 build outputs
# --------------------------------------------------------------------------

TABLES_11 = REPO_ROOT / "members" / "ankit-google" / "task-11-tables"

needs_task_11 = pytest.mark.skipif(
    not TABLES_11.is_dir(),
    reason="Task 11 build outputs not present in this checkout",
)


# --------------------------------------------------------------------------
# C10 — fixing the seed does not fix the input
# --------------------------------------------------------------------------
#
# C10 is the one entry whose evidence is an *absence* in every earlier task: no
# committed file said which of two writers a reader sees. So the tests pin the
# thing that replaced the absence — the contested table and the two rules that
# read it — plus the two row counts that make the hazard real rather than
# theoretical.

CONTESTED_FRAME = "data/processed/google/google_features.parquet"


def contested_table() -> pd.DataFrame:
    return pd.read_csv(TABLES_11 / "pipeline-contested-artefacts.csv")


@needs_task_11
def test_c10_the_contested_paths_are_the_ones_the_entry_counts():
    """Nine paths, five processed frames and four Task 06 tables."""
    contested = contested_table()
    frames = contested.path.str.startswith("data/processed/google/")
    tables = contested.path.str.contains("task-06-tables/")
    assert len(contested) == 9, (
        f"C10 counts nine contested paths; the table has {len(contested)}"
    )
    assert frames.sum() == 5 and tables.sum() == 4
    assert (frames | tables).all(), "a contested path C10 does not account for"

    prose = " ".join(entry_text("C10 —").split())
    assert "**nine** committed paths have two writers apiece" in prose
    assert "five processed Google frames and four Task 06 tables" in prose


@needs_task_11
def test_c10_the_feature_frame_is_read_on_both_sides_of_the_audit():
    """The row C10 quotes: `trends` sees 848, `similarity` sees 846.

    This is the correction in one line. Both readers are legitimate, both get
    a complete frame, and the difference between them is C4.
    """
    contested = contested_table()
    row = contested[contested.path == CONTESTED_FRAME]
    assert len(row) == 1, f"{CONTESTED_FRAME} is no longer a contested path"
    row = row.iloc[0]

    assert set(row.writers.split("; ")) == {"features", "competitor-set"}
    sees = dict(pair.split("=") for pair in row.sees.split("; "))
    assert sees["trends"] == "features", (
        "trends no longer reads the pre-audit frame — Task 05's committed "
        "848-row numbers would rebuild as 846"
    )
    assert sees["similarity"] == "competitor-set", (
        "similarity no longer reads the post-audit frame — Task 08 would "
        "rebuild from 848 rows, with the same seed"
    )
    assert bool(row.writers_ordered) and bool(row.readers_pinned)


@needs_task_11
def test_c10_both_row_counts_it_names_are_committed():
    """848 and 846 are not an argument, they are two tables in this repo."""
    before = read_table("volume-by-month.csv").postings.sum()
    after = pd.read_csv(
        REPO_ROOT / "members" / "ankit-google" / "task-06-tables"
        / "company-comparability.csv"
    ).set_index("company").postings["google"]
    assert (before, after) == (848, 846), (before, after)

    prose = " ".join(entry_text("C10 —").split())
    assert f"writes **{before}** rows" in prose
    assert f"writes\n  **{after}**" in entry_text("C10 —")


@needs_task_11
def test_c10_the_two_rules_it_names_exist_and_hold():
    """The remedy, checked where the entry says it lives.

    That each rule *fires* on a constructed violation is
    ``tests/test_pipeline.py``'s job. This checks the register is naming rules
    the linter emits, and that they are currently passing — a correction whose
    remedy is failing is a limitation.
    """
    lint = pd.read_csv(TABLES_11 / "pipeline-lint.csv")
    named = {"contested_writes_ordered", "contested_reads_pinned"}
    assert named <= set(lint.rule), (
        f"C10 names rules the linter does not emit: {named - set(lint.rule)}"
    )
    rows = lint[lint.rule.isin(named)]
    assert (rows.status == "pass").all(), rows.to_dict("records")

    reads = rows[rows.rule == "contested_reads_pinned"].iloc[0].subject
    assert f"**{reads.split()[0]}** stage-reads land on one of them" in \
        " ".join(entry_text("C10 —").split()), (
        f"C10 quotes a different number of contested reads than the {reads}"
    )
    source = (REPO_ROOT / "src" / "pipeline.py").read_text(encoding="utf-8")
    assert re.search(r"^    after: tuple\[str, \.\.\.\] = \(\)", source, re.M), (
        "C10's remedy is a sequencing edge; Stage no longer declares `after`"
    )


def test_c10_task_08_keeps_its_claim_and_gains_a_pointer():
    """§2 item 3 stays as written; the scope it omitted is marked beside it."""
    methods = (REPO_ROOT / "docs"
               / "task-08-company-similarity-methods.md").read_text(encoding="utf-8")
    claim = "fixed, so the\n   committed tables rebuild bit-for-bit."
    assert claim in methods, "Task 08 §2 no longer carries the corrected claim"
    assert "corrections.md#c10" in methods, "§2 carries no pointer to C10"
    quoted = " ".join(
        line.lstrip("> ") for line in entry_text("C10 —").splitlines()
    )
    assert " ".join(claim.split()) in quoted, (
        "the register no longer quotes the sentence it corrects"
    )
