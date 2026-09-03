"""Tests for the Task 11 pipeline layer.

Task 10 §13 made a prediction about this task:

    "There is no Task 11 in the brief; the two optional extensions are an
    automated pipeline and a fine-tuned extraction model."

The first half is wrong in a way worth recording — the brief's optional
extensions *are* numbered work, and this file is the automated pipeline. The
second half is right, and C7's standing lesson applies again: read the brief,
not the handover.

The traps below were live while `src/pipeline.py` was being written. They are
the reason the rules take the shape they do.

* **the edge nobody declared.** `build_competitor_set.py` reads the 75 MB full
  backfill cache that `collect_google_jobs.py` leaves behind as a by-product.
  On a developer's machine the file is always there, so nothing ever failed and
  nothing ever wrote the edge down. ``test_the_competitor_set_depends_on_the_
  collector`` pins it.
* **`requires_covers_reads` first demanded a direct parent, and was wrong.**
  Ordering is guaranteed by any ancestor, so demanding a direct edge forces the
  registry to spell out the full transitive closure — three spurious violations
  on the first run, and a graph nobody would maintain. The rule checks
  ancestry; the registry stays a transitive reduction. The same rule, once
  fixed, caught a *real* missing edge: `build_similarity.py` reads Task 07's
  `panel-share-series.csv`, so similarity depends on forecast, which the first
  draft of the registry did not say. ``test_similarity_depends_on_forecast``
  and ``test_a_transitive_ancestor_satisfies_the_read_rule``.
* **`git check-ignore` says nothing about a directory.** `.gitignore` carries
  `data/processed/*`, which ignores the contents without ignoring the
  directory, so asking about `data/processed/` returned nothing and the privacy
  check reported a violation that did not exist. The rule means "would a file
  written here be ignored", so it probes a file.
  ``test_a_directory_declaration_is_probed_as_a_file``.
* **the drift measurement changed its own answer.** The fact register counts
  the tables and figures in the workspace, and this task writes eight tables
  and five figures. Measuring drift before those landed gave a first run and a
  second run different answers to the same question. It is measured last, and
  anchored to a pinned commit so what the submission asserted survives the
  rebuild that fixes it. ``test_drift_is_anchored_to_a_pinned_commit``.
* **a diff is not a change.** Every JSON report stamps the wall clock, so a
  full rebuild moves bytes in ten files and content in none. A schedule that
  reported that as a change would commit noise until nobody read the diffs.
  ``test_a_clock_field_alone_is_not_a_change``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pipeline as pl        # noqa: E402

TABLES = REPO_ROOT / "members" / "ankit-google" / "task-11-tables"


@pytest.fixture(scope="module")
def lint():
    return pl.lint_dag()


@pytest.fixture(scope="module")
def audit():
    return pl.pipeline_audit()


@pytest.fixture(scope="module")
def drift():
    return pl.fact_drift()


def _toy(*stages):
    """A registry small enough to reason about, for testing a rule's teeth."""
    return tuple(stages)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_every_stage_name_is_unique():
    names = [s.name for s in pl.STAGES]
    assert len(names) == len(set(names))


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown kind"):
        pl.Stage(name="x", task=1, title="t", kind="ponder",
                 command=("python", "x.py"))


def test_an_unknown_volatility_is_rejected():
    with pytest.raises(ValueError, match="unknown volatility"):
        pl.Stage(name="x", task=1, title="t", kind="analyse",
                 command=("python", "x.py"), volatility="chaotic")


def test_a_stage_with_no_command_is_rejected():
    with pytest.raises(ValueError, match="no command"):
        pl.Stage(name="x", task=1, title="t", kind="analyse", command=())


def test_the_brief_five_steps_all_have_a_stage():
    """Collection, preprocessing, skill extraction, trends, forecasting."""
    tasks = {s.task for s in pl.STAGES}
    for task in (2, 3, 4, 5, 7):
        assert task in tasks


def test_the_competitor_set_depends_on_the_collector():
    """The by-product edge: it reads the cache the collector downloads."""
    stage = pl.STAGES_BY_NAME["competitor-set"]
    assert "collect" in stage.requires
    assert any("_hf_data_jobs_full.parquet" in r for r in stage.reads)


def test_similarity_depends_on_forecast():
    """`build_similarity.py` reads Task 07's panel series and gate."""
    stage = pl.STAGES_BY_NAME["similarity"]
    assert "forecast" in stage.requires


def test_only_one_stage_reaches_the_network_in_the_schedule():
    reaching = [s.name for s in pl.STAGES
                if s.scheduled and any(r.startswith("http") for r in s.reads)]
    assert reaching == ["collect"]


# ---------------------------------------------------------------------------
# DAG mechanics
# ---------------------------------------------------------------------------


def test_topological_order_covers_every_stage_once():
    order = pl.topological_order()
    assert sorted(order) == sorted(s.name for s in pl.STAGES)


def test_topological_order_is_deterministic():
    assert pl.topological_order() == pl.topological_order()


def test_a_cycle_is_refused_rather_than_looping():
    a = pl.Stage(name="a", task=1, title="a", kind="analyse",
                 command=("python", "a.py"), requires=("b",), writes=("a",))
    b = pl.Stage(name="b", task=1, title="b", kind="analyse",
                 command=("python", "b.py"), requires=("a",), writes=("b",))
    with pytest.raises(pl.CycleError):
        pl.topological_order(_toy(a, b))


def test_no_stage_runs_before_something_it_depends_on():
    position = {n: i for i, n in enumerate(pl.topological_order())}
    for stage in pl.STAGES:
        for parent in stage.requires:
            assert position[parent] < position[stage.name]


def test_ancestors_are_transitive():
    assert "collect" in pl.ancestors("insights")
    assert "preprocess" in pl.ancestors("presentation")


def test_a_root_has_no_ancestors():
    assert pl.ancestors("collect") == set()


def test_selecting_one_stage_pulls_in_what_it_needs():
    plan = [s.name for s in pl.plan_run(only=("forecast",))]
    assert plan[-1] == "forecast"
    assert {"collect", "preprocess", "features", "comparison"} <= set(plan)
    assert "presentation" not in plan


def test_selecting_an_unknown_stage_is_an_error():
    with pytest.raises(KeyError):
        pl.plan_run(only=("nonesuch",))


def test_optional_stages_are_out_of_the_default_plan():
    plan = {s.name for s in pl.plan_run()}
    assert "validate-text" not in plan
    assert "validate-forecast" not in plan


def test_asking_for_an_optional_stage_by_name_runs_it():
    plan = {s.name for s in pl.plan_run(only=("validate-forecast",))}
    assert "validate-forecast" in plan


def test_the_scheduled_plan_excludes_the_source_volatile_stage():
    plan = {s.name for s in pl.plan_run(scheduled_only=True,
                                        include_optional=True)}
    assert "validate-text" not in plan


def test_a_failure_blocks_its_descendants_and_nothing_else():
    blocked = pl.blocked_by({"comparison"})
    assert {"forecast", "similarity", "insights", "presentation"} <= blocked
    assert "collect" not in blocked
    assert "preprocess" not in blocked


def test_a_blocked_stage_is_not_recorded_as_failed():
    record = pl.RunRecord(stage="forecast", status="blocked")
    ledger = pl.run_ledger([record])
    assert ledger.status.iloc[0] == "blocked"
    assert pl.run_verdict([record], pd.DataFrame(
        columns=list(pl.REPRO_COLUMNS)))["state"] == "clean"


def test_an_unknown_run_status_is_rejected():
    with pytest.raises(ValueError, match="unknown run status"):
        pl.RunRecord(stage="collect", status="probably fine")


def test_an_edge_with_no_shared_artefact_is_labelled_sequencing():
    edges = pl.dag_edges()
    tests_edge = edges[edges.child == "tests"]
    assert (tests_edge.edge == "sequencing").all()


def test_an_edge_that_carries_a_file_is_labelled_data():
    edges = pl.dag_edges()
    row = edges[(edges.parent == "insights") & (edges.child == "presentation")]
    assert row.edge.iloc[0] == "data"
    assert "claim-ledger.csv" in row.via.iloc[0]


# ---------------------------------------------------------------------------
# The DAG linter — one test per rule
# ---------------------------------------------------------------------------


def test_the_declared_pipeline_lints_clean(lint):
    assert list(pl.lint_violations(lint).rule) == []


def test_every_lint_rule_reports_a_summary_row(lint):
    for rule in ("dag_acyclic", "order_covers_registry", "requires_known",
                 "order_is_topological", "contested_writes_ordered",
                 "contested_reads_pinned", "requires_covers_reads", "inputs_produced", "task_covered",
                 "optional_not_required", "schedule_closed"):
        assert rule in set(lint.rule), rule


def test_lint_catches_a_dependency_on_a_stage_that_does_not_exist():
    broken = _toy(replace(pl.STAGES_BY_NAME["collect"], requires=("ghost",)))
    failures = pl.lint_violations(pl.lint_dag(broken))
    assert "requires_known" in set(failures.rule)


def test_two_stages_may_write_one_path_when_the_dag_orders_them():
    """The rule this replaces forbade it outright, and was wrong to.

    This repository writes Google's feature frame twice on purpose: 848 rows
    from `features`, then 846 from `competitor-set` once C4's exclusions are
    applied. Both are correct, for different readers.
    """
    a = pl.Stage(name="a", task=1, title="a", kind="transform",
                 command=("python", "src/pipeline.py"), writes=("out/x.csv",))
    b = pl.Stage(name="b", task=1, title="b", kind="transform",
                 command=("python", "src/pipeline.py"), requires=("a",),
                 writes=("out/x.csv",))
    failures = pl.lint_violations(pl.lint_dag(_toy(a, b)))
    assert "contested_writes_ordered" not in set(failures.rule)


def test_lint_catches_two_unordered_writers_of_one_path():
    a = pl.Stage(name="a", task=1, title="a", kind="transform",
                 command=("python", "src/pipeline.py"), writes=("out/x.csv",))
    b = pl.Stage(name="b", task=1, title="b", kind="transform",
                 command=("python", "src/pipeline.py"), writes=("out/x.csv",))
    failures = pl.lint_violations(pl.lint_dag(_toy(a, b)))
    assert "contested_writes_ordered" in set(failures.rule)


def test_lint_catches_a_reader_that_could_land_either_side_of_a_writer():
    """The defect that reversed C4 across 116 files.

    `r` reads what `a` wrote. `b` rewrites the same path with different
    content. Nothing orders `r` against `b`, so `r` reads whichever version
    the run order happens to leave — and both runs succeed.
    """
    a = pl.Stage(name="a", task=1, title="a", kind="transform",
                 command=("python", "src/pipeline.py"), writes=("out/x.csv",))
    r = pl.Stage(name="r", task=1, title="r", kind="analyse",
                 command=("python", "src/pipeline.py"), requires=("a",),
                 reads=("out/x.csv",), writes=("out/r.csv",))
    b = pl.Stage(name="b", task=1, title="b", kind="transform",
                 command=("python", "src/pipeline.py"), requires=("a",),
                 writes=("out/x.csv",))
    failures = pl.lint_violations(pl.lint_dag(_toy(a, r, b)))
    assert "contested_reads_pinned" in set(failures.rule)

    pinned = pl.lint_violations(pl.lint_dag(
        _toy(a, r, replace(b, after=("r",)))))
    assert "contested_reads_pinned" not in set(pinned.rule)


def test_an_after_edge_orders_without_forcing_the_stage_into_the_run():
    """`competitor-set` must follow `validate-skills`, which is optional.

    A `requires` edge would break `optional_not_required`; an `after` edge
    constrains the order and leaves selection alone.
    """
    assert pl.STAGES_BY_NAME["competitor-set"].after == ("validate-skills",)
    order = pl.topological_order()
    assert order.index("validate-skills") < order.index("competitor-set")

    default = [s.name for s in pl.plan_run()]
    assert "validate-skills" not in default
    assert "competitor-set" in default
    assert "optional_not_required" not in set(
        pl.lint_violations(pl.lint_dag()).rule)


def test_the_contested_google_frame_is_read_before_it_is_overwritten():
    """Task 05 reports 848 and Task 06 reports 846, from one path.

    This is the ordering the repository's committed numbers depend on, and
    nothing declared it before Task 11. Reverse it and every Task 05 table
    rebuilds against 846 while C4 says Tasks 02-05 keep their 848.
    """
    contested = pl.contested_artefacts()
    row = contested[contested.path
                    == "data/processed/google/google_features.parquet"]
    assert len(row) == 1
    sees = dict(part.split("=") for part in row.iloc[0].sees.split("; "))
    assert sees["trends"] == "features"
    assert sees["comparison"] == "competitor-set"
    assert bool(row.iloc[0].writers_ordered)
    assert bool(row.iloc[0].readers_pinned)


def test_every_contested_path_is_ordered_and_pinned():
    contested = pl.contested_artefacts()
    assert len(contested) > 0
    assert contested.writers_ordered.all()
    assert contested.readers_pinned.all()


def test_a_stage_that_reads_what_it_writes_is_not_ambiguous_to_itself():
    """`comparison` reads the feasibility screen and rewrites it."""
    contested = pl.contested_artefacts()
    row = contested[contested.path.str.endswith(
        "company-feasibility-screen.csv")]
    sees = dict(part.split("=") for part in row.iloc[0].sees.split("; "))
    assert sees["comparison"] == "competitor-set"


def test_lint_catches_a_read_whose_producer_is_not_an_ancestor():
    a = pl.Stage(name="a", task=1, title="a", kind="transform",
                 command=("python", "src/pipeline.py"), writes=("out/a.csv",))
    b = pl.Stage(name="b", task=1, title="b", kind="analyse",
                 command=("python", "src/pipeline.py"),
                 reads=("out/a.csv",), writes=("out/b.csv",))
    failures = pl.lint_violations(pl.lint_dag(_toy(a, b)))
    assert "requires_covers_reads" in set(failures.rule)


def test_a_transitive_ancestor_satisfies_the_read_rule():
    a = pl.Stage(name="a", task=1, title="a", kind="transform",
                 command=("python", "src/pipeline.py"), writes=("out/a.csv",))
    b = pl.Stage(name="b", task=1, title="b", kind="analyse",
                 command=("python", "src/pipeline.py"), requires=("a",),
                 writes=("out/b.csv",))
    c = pl.Stage(name="c", task=1, title="c", kind="analyse",
                 command=("python", "src/pipeline.py"), requires=("b",),
                 reads=("out/a.csv",), writes=("out/c.csv",))
    failures = pl.lint_violations(pl.lint_dag(_toy(a, b, c)))
    assert "requires_covers_reads" not in set(failures.rule)


def test_lint_catches_an_input_nothing_produces():
    a = pl.Stage(name="a", task=1, title="a", kind="analyse",
                 command=("python", "src/pipeline.py"),
                 reads=("out/never-written.csv",), writes=("out/a.csv",))
    failures = pl.lint_violations(pl.lint_dag(_toy(a)))
    assert "inputs_produced" in set(failures.rule)


def test_lint_catches_a_stage_that_runs_a_script_that_is_gone():
    a = pl.Stage(name="a", task=1, title="a", kind="analyse",
                 command=("python", "src/does_not_exist.py"),
                 writes=("out/a.csv",))
    failures = pl.lint_violations(pl.lint_dag(_toy(a)))
    assert "command_resolves" in set(failures.rule)


def test_lint_catches_a_non_gate_stage_that_writes_nothing():
    a = pl.Stage(name="a", task=1, title="a", kind="analyse",
                 command=("python", "src/pipeline.py"))
    failures = pl.lint_violations(pl.lint_dag(_toy(a)))
    assert "writes_declared" in set(failures.rule)


def test_lint_catches_an_uncovered_analysis_task():
    failures = pl.lint_violations(
        pl.lint_dag(_toy(pl.STAGES_BY_NAME["collect"])))
    assert "task_covered" in set(failures.rule)


def test_lint_catches_a_default_stage_depending_on_an_optional_one():
    a = pl.Stage(name="a", task=1, title="a", kind="validate",
                 command=("python", "src/pipeline.py"), optional=True,
                 writes=("out/a.csv",))
    b = pl.Stage(name="b", task=1, title="b", kind="analyse",
                 command=("python", "src/pipeline.py"), requires=("a",),
                 writes=("out/b.csv",))
    failures = pl.lint_violations(pl.lint_dag(_toy(a, b)))
    assert "optional_not_required" in set(failures.rule)


def test_lint_catches_a_scheduled_stage_depending_on_an_unscheduled_one():
    a = pl.Stage(name="a", task=1, title="a", kind="validate",
                 command=("python", "src/pipeline.py"), scheduled=False,
                 writes=("out/a.csv",))
    b = pl.Stage(name="b", task=1, title="b", kind="analyse",
                 command=("python", "src/pipeline.py"), requires=("a",),
                 writes=("out/b.csv",))
    failures = pl.lint_violations(pl.lint_dag(_toy(a, b)))
    assert "schedule_closed" in set(failures.rule)


def test_lint_catches_a_cycle_without_hanging():
    a = pl.Stage(name="a", task=1, title="a", kind="analyse",
                 command=("python", "src/pipeline.py"), requires=("b",),
                 writes=("out/a.csv",))
    b = pl.Stage(name="b", task=1, title="b", kind="analyse",
                 command=("python", "src/pipeline.py"), requires=("a",),
                 writes=("out/b.csv",))
    failures = pl.lint_violations(pl.lint_dag(_toy(a, b)))
    assert "dag_acyclic" in set(failures.rule)


# ---------------------------------------------------------------------------
# Volatility and digests
# ---------------------------------------------------------------------------


def test_a_clock_field_alone_is_not_a_change(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps({"generated_at": "2026-01-01", "n": 4}))
    second.write_text(json.dumps({"generated_at": "2026-09-03", "n": 4}))
    assert pl.raw_digest(first) != pl.raw_digest(second)
    assert pl.stable_digest(first) == pl.stable_digest(second)


def test_a_real_change_survives_the_clock_strip(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps({"generated_at": "2026-01-01", "n": 4}))
    second.write_text(json.dumps({"generated_at": "2026-01-01", "n": 5}))
    assert pl.stable_digest(first) != pl.stable_digest(second)


def test_a_nested_clock_field_is_stripped():
    stripped = pl.strip_volatile({"run": {"generated_at": "x", "n": 1}})
    assert stripped == {"run": {"n": 1}}


def test_key_order_is_not_a_change(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text('{"a": 1, "b": 2}')
    second.write_text('{"b": 2,   "a": 1}')
    assert pl.stable_digest(first) == pl.stable_digest(second)


def test_a_csv_byte_change_is_always_a_change(tmp_path):
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    first.write_text("skill,share\nsql,0.51\n")
    second.write_text("skill,share\nsql,0.52\n")
    assert pl.stable_digest(first) != pl.stable_digest(second)


def test_a_markdown_generation_stamp_is_not_a_change(tmp_path):
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text("# Report\n\n_Generated 2026-01-01._\n\nBody.\n")
    second.write_text("# Report\n\n_Generated 2026-09-03._\n\nBody.\n")
    assert pl.stable_digest(first) == pl.stable_digest(second)


def test_unparseable_json_falls_back_to_the_bytes(tmp_path):
    broken = tmp_path / "a.json"
    broken.write_text("{not json")
    assert pl.stable_digest(broken) == pl.raw_digest(broken)


@pytest.mark.parametrize("verdict", pl.REPRO_VERDICTS)
def test_every_reproducibility_verdict_is_reachable(verdict):
    before = {"same": {"raw": "1", "stable": "s"},
              "clock": {"raw": "1", "stable": "s"},
              "real": {"raw": "1", "stable": "s"},
              "gone": {"raw": "1", "stable": "s"}}
    after = {"same": {"raw": "1", "stable": "s"},
             "clock": {"raw": "2", "stable": "s"},
             "real": {"raw": "2", "stable": "t"},
             "new": {"raw": "1", "stable": "s"}}
    audit = pl.reproducibility_audit(before, after)
    assert verdict in set(audit.verdict)


def test_volatility_classes_are_ordered_strictest_first():
    assert pl.volatility_rank("deterministic") < pl.volatility_rank("clock_stamped")
    assert pl.volatility_rank("clock_stamped") < pl.volatility_rank("source_volatile")


def test_measured_volatility_reads_the_strictest_consistent_class():
    stage = pl.STAGES_BY_NAME["preprocess"]
    audit = pd.DataFrame([
        {"path": "members/ankit-google/task-03-quality-report.json",
         "verdict": "volatile_only", "detail": ""}])
    assert pl.measured_volatility(audit, stage) == "clock_stamped"


def test_a_stage_that_moved_nothing_measures_deterministic():
    stage = pl.STAGES_BY_NAME["preprocess"]
    empty = pd.DataFrame(columns=list(pl.REPRO_COLUMNS))
    assert pl.measured_volatility(empty, stage) == "deterministic"


def test_a_directory_write_covers_a_file_inside_it():
    assert pl._covers("members/ankit-google/task-05-tables/",
                      "members/ankit-google/task-05-tables/x.csv")


def test_a_directory_read_covers_a_file_written_inside_it():
    assert pl._covers("data/processed/google/google_features.parquet",
                      "data/processed/")


def test_unrelated_paths_do_not_cover_each_other():
    assert not pl._covers("data/raw/", "data/processed/")


# ---------------------------------------------------------------------------
# Sources and the schedule
# ---------------------------------------------------------------------------


def test_every_source_verdict_is_registered():
    for source in pl.SOURCES:
        assert source.verdict in pl.SOURCE_VERDICTS


def test_an_unknown_source_verdict_is_rejected():
    with pytest.raises(ValueError, match="unknown verdict"):
        pl.Source(name="x", task=1, licence="l", window="w",
                  verdict="probably", blocker="b", falsifier="f")


def test_every_blocked_source_names_a_falsifier():
    for source in pl.SOURCES:
        if source.verdict != "refreshable":
            assert source.falsifier.strip()


def test_the_rejected_source_is_still_registered():
    """Task 01 rejected Google Careers; a schedule must not quietly retry it."""
    names = {s.name for s in pl.SOURCES}
    assert "google-careers" in names
    assert pl.SOURCES[-1].verdict == "blocked"


def test_reed_is_not_an_approved_source():
    assert "reed" not in {s.name for s in pl.SOURCES}


def test_no_source_can_refresh_today():
    assert pl.refresh_verdict()["refreshable"] == 0


def test_the_schedule_verifies_rather_than_refreshes():
    assert pl.refresh_verdict()["mode"] == "verify"


def test_a_refreshable_source_would_flip_the_mode():
    live = pl.Source(name="fake", task=2, licence="api", window="rolling",
                     verdict="refreshable", blocker="", falsifier="")
    assert pl.refresh_verdict(pl.SOURCES + (live,))["mode"] == "refresh"


def test_a_monthly_refresh_is_not_adopted_with_no_refreshable_source():
    cadence = pl.schedule_rationale()
    monthly = cadence[cadence.cadence == "monthly"]
    assert not bool(monthly.adopted.iloc[0])


def test_a_refreshable_source_would_adopt_the_monthly_cadence():
    live = pl.Source(name="fake", task=2, licence="api", window="rolling",
                     verdict="refreshable", blocker="", falsifier="")
    cadence = pl.schedule_rationale(pl.SOURCES + (live,))
    monthly = cadence[cadence.cadence == "monthly"]
    assert bool(monthly.adopted.iloc[0])


def test_the_weekly_verification_is_adopted_regardless():
    cadence = pl.schedule_rationale()
    weekly = cadence[cadence.cadence == "weekly"]
    assert bool(weekly.adopted.iloc[0])


def test_every_cadence_says_what_it_would_detect():
    assert pl.schedule_rationale().detects.str.strip().ne("").all()


# ---------------------------------------------------------------------------
# The pipeline audit
# ---------------------------------------------------------------------------


def test_the_audit_covers_every_declared_area(audit):
    assert {"privacy", "outputs", "volatility", "inputs", "sources",
            "secrets"} <= set(audit.area)


def test_row_level_output_stays_out_of_git(audit):
    rows = audit[audit.check == "row_level_ignored"]
    assert len(rows)
    assert (rows.status == "pass").all()


def test_a_directory_declaration_is_probed_as_a_file():
    """`.gitignore` ignores `data/processed/*`, not the directory itself."""
    assert "data/processed/" in pl.git_ignored(["data/processed/"])


def test_the_env_file_is_never_tracked(audit):
    row = audit[audit.check == "env_untracked"]
    assert row.status.iloc[0] == "pass"


def test_the_source_volatile_stage_is_reported(audit):
    """It commits a document a rolling feed rewrites; the audit says so."""
    row = audit[audit.check == "source_volatile_not_committed"]
    assert row.status.iloc[0] == "fail"
    assert "task-03-text-pipeline-validation.md" in row.detail.iloc[0]


def test_every_committed_output_is_actually_tracked(audit):
    rows = audit[audit.check == "committed_output_tracked"]
    assert len(rows)
    assert set(rows[rows.status == "fail"].subject) <= {
        "members/ankit-google/task-06-tables/competitor-set-manifest.json"}


def test_the_audit_never_stops_the_build(audit):
    """Task 10's precedent: the lint fails, the audit reports."""
    assert (audit.status == "fail").any()
    assert list(pl.lint_violations(pl.lint_dag()).rule) == []


# ---------------------------------------------------------------------------
# Fact drift — three readings of one number
# ---------------------------------------------------------------------------


def test_drift_covers_every_registered_fact(drift):
    import present as pr
    assert len(drift) == len(pr.FACTS)


def test_drift_is_anchored_to_a_pinned_commit(drift):
    assert pl.TASK10_SUBMISSION
    assert drift.submitted.str.strip().ne("").all()


def test_the_submitted_deck_counted_127_tables_and_49_figures(drift):
    """The fixed record: what the Task 10 submission asserted."""
    indexed = drift.set_index("key")
    assert indexed.loc["tables_committed", "submitted"] == "127"
    assert indexed.loc["figures_committed", "submitted"] == "49"
    assert indexed.loc["tasks_total", "submitted"] == "10"


def test_task_11_moved_the_decks_self_counts(drift):
    """Adding a task moves numbers the deck asserts about the repository."""
    summary = pl.drift_summary(drift)
    assert summary["moved_since_submission"] >= 2
    assert {"tables_committed", "figures_committed"} <= set(
        summary["keys_since_submission"])


def test_the_committed_deck_matches_the_repository(drift):
    """The invariant the pipeline exists to hold: rebuild after you add files.

    `committed` reads git, not the working tree, so a rebuild that has not been
    committed still fails here. That is deliberate — an uncommitted deck is
    exactly as stale to anyone cloning the repository as one never rebuilt.
    """
    stale = drift[drift.moved]
    assert list(stale.key) == [], (
        "the deck's self-counts have drifted from what git holds; run "
        "`python src/run_pipeline.py --run --only presentation` "
        "and commit the rebuilt deck")


def test_missing_history_degrades_rather_than_fails():
    assert pl.facts_at(ref="0000000000000000000000000000000000000000") == {}


# ---------------------------------------------------------------------------
# The committed tables
# ---------------------------------------------------------------------------


def test_the_stage_table_is_committed_and_current():
    committed = pd.read_csv(TABLES / "pipeline-stages.csv")
    assert sorted(committed.stage) == sorted(s.name for s in pl.STAGES)


def test_the_committed_run_order_matches_the_registry():
    committed = pd.read_csv(TABLES / "pipeline-stages.csv")
    assert list(committed.sort_values("run_order").stage) == \
        pl.topological_order()


def test_the_committed_lint_has_no_violations():
    committed = pd.read_csv(TABLES / "pipeline-lint.csv")
    assert (committed.status == "fail").sum() == 0


def test_every_failure_mode_names_what_the_pipeline_does():
    modes = pl.failure_mode_table()
    assert modes.what_the_pipeline_does.str.strip().ne("").all()
    assert "cache_absent" in set(modes["mode"])
