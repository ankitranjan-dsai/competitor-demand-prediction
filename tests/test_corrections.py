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
    index = markdown_table("Corrections Register")
    assert len(index) == 6
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
