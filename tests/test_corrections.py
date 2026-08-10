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
    assert len(index) == 4
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
        Path("members/ankit-google/task-04-skill-extraction-report.md"),
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
