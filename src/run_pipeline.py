"""Task 11 (optional) — the orchestrator: run the DAG, or just check it.

Two modes, because the pipeline has two jobs and they cost three orders of
magnitude apart.

``--check`` (the default, ~2 s) never executes a stage. It lints the DAG,
audits the repository, compares the committed Task 10 fact register against a
live one, and writes every table and figure that does not need a run. This is
what a pull request runs: it catches a stage added without its edge, an output
two stages both claim, an input a fresh clone would not have, and a deck whose
numbers the repository has moved past.

``--run`` (~30 s over the cached backfill) executes the plan, timing each
stage, and — the part that matters — digests every rebuildable committed
artefact before and after so the run can answer "did anything change" without
anyone reading a diff. Ten JSON reports rewrite their wall clock on every
single run; that is `volatile_only`, not a finding, and separating the two is
the whole reason a scheduled rebuild here is worth running at all.

Neither mode commits anything. The workflow decides that, and it decides it
differently for a pull request (fail on drift) than for the weekly verification
(report drift), because on a frozen source a content change means the code or
its dependencies moved, and that is a thing to look at rather than a thing to
merge.

Usage
-----
    python src/run_pipeline.py                    # check: lint, audit, drift
    python src/run_pipeline.py --run              # execute the default plan
    python src/run_pipeline.py --run --scheduled  # what the cron executes
    python src/run_pipeline.py --run --only forecast
    python src/run_pipeline.py --plan             # print the order and stop
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # no display in CI or on a headless run
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import insights as ins
import pipeline as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "members" / "ankit-google"
TABLES = OUT / "task-11-tables"
FIGURES = OUT / "task-11-figures"
REPORT = OUT / "task-11-pipeline-report.json"

INK = "#1f2933"
GREY = "#9aa5b1"
BLUE = "#2f6fb5"
ORANGE = "#e08a1e"
RED = "#c0392b"
GREEN = "#2e7d5b"
PURPLE = "#7d5ba6"

KIND_COLOUR = {"collect": PURPLE, "transform": BLUE, "analyse": GREEN,
               "publish": ORANGE, "validate": GREY, "gate": INK}
VERDICT_COLOUR = {"identical": GREY, "volatile_only": BLUE, "changed": RED,
                  "added": GREEN, "removed": ORANGE}


def _write(df: pd.DataFrame, path: Path) -> Path:
    """Write a table, refusing anything that carries a forbidden column.

    Task 11's tables are about the pipeline rather than about postings, so the
    check should never fire. It runs anyway: the privacy guard is standing, and
    a guard that is skipped where it "obviously" cannot matter is a guard that
    stops being run at all.
    """
    bad = ins.forbidden_columns(df)
    if bad:
        raise ValueError(f"{path.name} carries forbidden columns {bad}")
    personal = ins.personal_data_columns_present(df)
    if personal:
        raise ValueError(f"{path.name} carries personal-data columns {personal}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _style(ax, title, ylabel="", xlabel=""):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute(stage: pl.Stage, *, repo_root: Path = REPO_ROOT) -> pl.RunRecord:
    """Run one stage and time it, capturing rather than streaming its output."""
    command = list(stage.command)
    if command and command[0] == "python":
        command[0] = sys.executable
    started = time.perf_counter()
    try:
        done = subprocess.run(command, cwd=repo_root, capture_output=True,
                              text=True, timeout=3600)
        code, tail = done.returncode, (done.stdout or "")[-2000:]
        if code != 0:
            tail = ((done.stderr or "")[-2000:]) or tail
    except (OSError, subprocess.SubprocessError) as exc:
        code, tail = 127, str(exc)
    seconds = time.perf_counter() - started
    return pl.RunRecord(
        stage=stage.name,
        status="ok" if code == 0 else "failed",
        returncode=code,
        seconds=seconds,
        note="" if code == 0 else tail.strip().splitlines()[-1][:200]
        if tail.strip() else "no output",
        stdout_tail=tail,
    )


def run_plan(plan: list[pl.Stage], *, repo_root: Path = REPO_ROOT,
             verbose: bool = True) -> tuple[list[pl.RunRecord], pd.DataFrame]:
    """Execute a plan, snapshotting artefacts either side of it.

    A stage whose dependency failed is recorded `blocked` and never run, so one
    broken stage produces one line to read rather than a cascade.
    """
    before = pl.snapshot(repo_root)
    records: list[pl.RunRecord] = []
    failed: set[str] = set()
    for stage in plan:
        blocked = pl.blocked_by(failed)
        if stage.name in blocked:
            records.append(pl.RunRecord(
                stage=stage.name, status="blocked",
                note="upstream stage failed; not attempted"))
            if verbose:
                print(f"  {stage.name:<20} blocked")
            continue
        record = execute(stage, repo_root=repo_root)
        records.append(record)
        if record.status == "failed":
            failed.add(stage.name)
        if verbose:
            mark = "ok " if record.status == "ok" else "FAIL"
            print(f"  {stage.name:<20} {mark} {record.seconds:6.2f}s")
    after = pl.snapshot(repo_root)
    audit = pl.reproducibility_audit(before, after)

    for record in records:
        stage = pl.STAGES_BY_NAME[record.stage]
        mine = audit[audit.path.apply(
            lambda p: any(pl._covers(w, p) for w in stage.writes))]
        record.artefacts_changed = int((mine.verdict == "changed").sum())
        record.artefacts_volatile = int((mine.verdict == "volatile_only").sum())
    return records, audit


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _depths(stages: tuple[pl.Stage, ...]) -> dict[str, int]:
    """Longest path from a root, which is the column a stage is drawn in."""
    depth: dict[str, int] = {}
    for name in pl.topological_order(stages):
        parents = pl.STAGES_BY_NAME[name].requires
        depth[name] = 1 + max((depth[p] for p in parents), default=-1)
    return depth


def figure_dag(stages: tuple[pl.Stage, ...], path: Path) -> Path:
    """The pipeline as it is declared — drawn from the registry, not by hand.

    Redrawn from `STAGES` on every build, so a diagram that disagrees with the
    code is not possible. That is the point: the diagrams that rot are the ones
    maintained separately from the thing they describe.
    """
    depth = _depths(stages)
    columns: dict[int, list[str]] = {}
    for name in pl.topological_order(stages):
        columns.setdefault(depth[name], []).append(name)

    width = max(columns) + 1
    height = max(len(c) for c in columns.values())
    fig, ax = plt.subplots(figsize=(1.9 * width + 1.5, 1.15 * height + 1.2))

    pos: dict[str, tuple[float, float]] = {}
    for col, names in columns.items():
        offset = (height - len(names)) / 2
        for row, name in enumerate(names):
            pos[name] = (col, height - (row + offset))

    for stage in stages:
        for parent in stage.requires:
            x0, y0 = pos[parent]
            x1, y1 = pos[stage.name]
            ax.annotate("", xy=(x1 - 0.34, y1), xytext=(x0 + 0.34, y0),
                        arrowprops=dict(arrowstyle="-|>", color=GREY,
                                        lw=0.9, shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3,rad=0.08"))
    for stage in stages:
        x, y = pos[stage.name]
        colour = KIND_COLOUR.get(stage.kind, GREY)
        ax.add_patch(plt.Rectangle(
            (x - 0.36, y - 0.24), 0.72, 0.48,
            facecolor="white", edgecolor=colour,
            linewidth=1.6, linestyle="--" if stage.optional else "-",
            zorder=3))
        ax.text(x, y + 0.05, stage.name, ha="center", va="center",
                fontsize=7.5, color=INK, zorder=4)
        ax.text(x, y - 0.11, f"task {stage.task} · {stage.kind}",
                ha="center", va="center", fontsize=6, color=GREY, zorder=4)

    ax.set_xlim(-0.7, width - 0.3)
    ax.set_ylim(0.2, height + 0.9)
    ax.axis("off")
    ax.set_title("The pipeline as declared — solid runs by default, dashed is "
                 "opt-in; left to right is dependency order",
                 loc="left", fontsize=10, color=INK, pad=12)
    handles = [plt.Line2D([], [], color=c, lw=3, label=k)
               for k, c in KIND_COLOUR.items()]
    ax.legend(handles=handles, loc="lower center", ncol=len(KIND_COLOUR),
              frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    return _save(fig, path)


def figure_sources(refresh: pd.DataFrame, cadence: pd.DataFrame,
                   path: Path) -> Path:
    """Why the schedule verifies instead of refreshing."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6),
                                      gridspec_kw={"width_ratios": [1, 1.15]})
    order = ["refreshable", "live_empty", "blocked", "frozen"]
    colour = {"refreshable": GREEN, "live_empty": ORANGE,
              "blocked": RED, "frozen": GREY}
    counts = refresh.verdict.value_counts().reindex(order).fillna(0)
    left.bar(range(len(order)), counts.values,
             color=[colour[v] for v in order], width=0.62)
    left.set_xticks(range(len(order)))
    left.set_xticklabels(order, rotation=20, ha="right")
    left.set_yticks(range(0, int(counts.max()) + 2))
    _style(left, "Approved sources by what a re-run could return",
           ylabel="sources")
    for i, value in enumerate(counts.values):
        if value:
            left.text(i, value + 0.06, f"{int(value)}", ha="center",
                      fontsize=8, color=INK)

    rows = cadence.iloc[::-1]
    right.barh(range(len(rows)), [1] * len(rows),
               color=[GREEN if a else GREY for a in rows.adopted],
               alpha=0.85, height=0.6)
    right.set_yticks(range(len(rows)))
    right.set_yticklabels(rows.cadence, fontsize=8)
    right.set_xticks([])
    right.spines[["top", "right", "bottom"]].set_visible(False)
    right.set_title("Cadences considered — green adopted", loc="left",
                    fontsize=11, color=INK, pad=10)
    for i, detects in enumerate(rows.detects):
        right.text(0.02, i, detects[:64], va="center", fontsize=7,
                   color="white" if rows.adopted.iloc[i] else INK)
    return _save(fig, path)


def figure_drift(drift: pd.DataFrame, path: Path) -> Path:
    """The committed deck's self-counts against the repository today."""
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(drift) + 1.6))
    rows = drift.iloc[::-1].reset_index(drop=True)
    ax.barh(range(len(rows)), [1] * len(rows),
            color=[ORANGE if m else GREY for m in rows.moved_since_submission],
            alpha=0.85, height=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows.key, fontsize=7.5)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    for i, row in rows.iterrows():
        text = (f"{row.submitted} -> {row.live}" if row.moved_since_submission
                else f"{row.live}")
        ax.text(0.015, i, text, va="center", fontsize=7.5,
                color="white" if row.moved_since_submission else INK)
    moved = int(rows.moved_since_submission.sum())
    ax.set_title(f"Task 10's self-counts, at the submission commit against "
                 f"live — {moved} of {len(rows)} have moved (C10)",
                 loc="left", fontsize=11, color=INK, pad=10)
    return _save(fig, path)


def figure_checks(lint: pd.DataFrame, audit: pd.DataFrame, path: Path) -> Path:
    """What the two rule tables found, side by side."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8))
    summary = (lint.groupby("rule").status
               .apply(lambda s: (s == "fail").sum()).sort_index())
    left.barh(range(len(summary)), [1] * len(summary),
              color=[RED if v else GREEN for v in summary.values],
              alpha=0.85, height=0.62)
    left.set_yticks(range(len(summary)))
    left.set_yticklabels(summary.index, fontsize=7.5)
    left.set_xticks([])
    left.spines[["top", "right", "bottom"]].set_visible(False)
    left.set_title("DAG lint — a failure stops the build", loc="left",
                   fontsize=11, color=INK, pad=10)

    by_area = (audit.groupby("area").status
               .value_counts().unstack(fill_value=0)
               .reindex(columns=["pass", "fail"], fill_value=0))
    idx = range(len(by_area))
    right.bar(idx, by_area["pass"], color=GREEN, width=0.6, label="pass")
    right.bar(idx, by_area["fail"], bottom=by_area["pass"], color=ORANGE,
              width=0.6, label="finding")
    right.set_xticks(list(idx))
    right.set_xticklabels(by_area.index, rotation=20, ha="right")
    _style(right, "Pipeline audit — findings are reported, not fatal",
           ylabel="checks")
    right.legend(frameon=False, fontsize=8)
    return _save(fig, path)


def figure_run(ledger: pd.DataFrame, audit: pd.DataFrame, path: Path) -> Path:
    """How long the run took, and what it actually moved."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2),
                                      gridspec_kw={"width_ratios": [1.3, 1]})
    rows = ledger[ledger.seconds > 0].iloc[::-1]
    left.barh(range(len(rows)), rows.seconds,
              color=[KIND_COLOUR.get(k, GREY) for k in rows.kind],
              height=0.62)
    left.set_yticks(range(len(rows)))
    left.set_yticklabels(rows.stage, fontsize=7.5)
    _style(left, f"Stage wall time — {rows.seconds.sum():.0f}s end to end",
           xlabel="seconds")
    left.grid(axis="x", alpha=0.25, linewidth=0.6)
    left.grid(axis="y", visible=False)

    order = [v for v in pl.REPRO_VERDICTS if v in set(audit.verdict)]
    counts = audit.verdict.value_counts().reindex(order).fillna(0)
    right.bar(range(len(order)), counts.values,
              color=[VERDICT_COLOUR[v] for v in order], width=0.6)
    right.set_xticks(range(len(order)))
    right.set_xticklabels(order, rotation=20, ha="right")
    _style(right, f"What the run moved, across {len(audit)} committed artefacts",
           ylabel="artefacts")
    for i, value in enumerate(counts.values):
        if value:
            right.text(i, value + 0.6, f"{int(value)}", ha="center",
                       fontsize=8, color=INK)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------


def build(*, execute_stages: bool = False, only: tuple[str, ...] = (),
          skip: tuple[str, ...] = (), include_optional: bool = False,
          scheduled_only: bool = False, focus: str = "google") -> dict:
    written: list[Path] = []
    figures: list[Path] = []

    lint = pl.lint_dag()
    violations = pl.lint_violations(lint)
    if len(violations):
        for row in violations.itertuples():
            print(f"  ! {row.rule}: {row.subject} — {row.detail}",
                  file=sys.stderr)
        raise SystemExit(
            f"the pipeline definition is invalid: {len(violations)} lint "
            f"violation{'s' if len(violations) != 1 else ''}")

    audit = pl.pipeline_audit()
    refresh = pl.refreshability_table()
    cadence = pl.schedule_rationale()

    written += [
        _write(pl.stage_table(), TABLES / "pipeline-stages.csv"),
        _write(pl.dag_edges(), TABLES / "pipeline-dag-edges.csv"),
        _write(lint, TABLES / "pipeline-lint.csv"),
        _write(audit, TABLES / "pipeline-audit.csv"),
        _write(refresh, TABLES / "source-refreshability.csv"),
        _write(cadence, TABLES / "schedule-rationale.csv"),
        _write(pl.failure_mode_table(), TABLES / "pipeline-failure-modes.csv"),
    ]

    # Only now — the fact register counts the files this run just wrote, so
    # measuring earlier would give a first run and a second run different
    # answers to the same question.
    drift = pl.fact_drift(REPO_ROOT, OUT, focus)
    written.append(_write(drift, TABLES / "deck-fact-drift.csv"))

    figures += [
        figure_dag(pl.STAGES, FIGURES / "pipeline-dag.png"),
        figure_sources(refresh, cadence, FIGURES / "source-refreshability.png"),
        figure_checks(lint, audit, FIGURES / "pipeline-checks.png"),
    ]
    if len(drift):
        figures.append(figure_drift(drift, FIGURES / "deck-fact-drift.png"))

    plan = pl.plan_run(only=only, skip=skip, include_optional=include_optional,
                       scheduled_only=scheduled_only)
    records: list[pl.RunRecord] = []
    repro = pd.DataFrame(columns=list(pl.REPRO_COLUMNS))
    verdict = {"state": "not_run", "stages_run": 0, "failed": [],
               "blocked": [], "artefacts_changed": 0,
               "artefacts_volatile_only": 0, "changed_paths": []}
    if execute_stages:
        print(f"running {len(plan)} stage{'s' if len(plan) != 1 else ''}:")
        records, repro = run_plan(plan)
        ledger = pl.run_ledger(records)
        verdict = pl.run_verdict(records, repro)
        written += [_write(ledger, TABLES / "run-ledger.csv"),
                    _write(repro, TABLES / "reproducibility-audit.csv")]
        figures.append(figure_run(ledger, repro, FIGURES / "pipeline-run.png"))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "run" if execute_stages else "check",
        "focus": focus,
        "stages_declared": len(pl.STAGES),
        "stages_planned": [s.name for s in plan],
        "run_order": pl.topological_order(),
        "lint": {"checks": int(len(lint)),
                 "violations": int(len(violations))},
        "audit": {"checks": int(len(audit)),
                  "findings": int((audit.status == "fail").sum()),
                  "finding_checks": sorted(set(
                      audit[audit.status == "fail"].check))},
        "sources": pl.refresh_verdict(),
        "credentials": pl.credential_state(),
        "fact_drift": pl.drift_summary(drift),
        "run": verdict,
        "tables": sorted(p.name for p in written),
        "figures": sorted(p.name for p in figures),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return {"report": report, "lint": lint, "audit": audit, "drift": drift,
            "repro": repro, "records": records, "written": written,
            "figures": figures, "plan": plan}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="execute the stages; without it, only check")
    parser.add_argument("--plan", action="store_true",
                        help="print the resolved run order and stop")
    parser.add_argument("--only", default="",
                        help="comma-separated stages, with their dependencies")
    parser.add_argument("--skip", default="",
                        help="comma-separated stages to leave out")
    parser.add_argument("--include-optional", action="store_true",
                        help="also run the scipy validators")
    parser.add_argument("--scheduled", action="store_true",
                        help="only stages the schedule is allowed to run")
    parser.add_argument("--focus", default="google")
    args = parser.parse_args()

    only = tuple(s for s in args.only.split(",") if s)
    skip = tuple(s for s in args.skip.split(",") if s)

    if args.plan:
        plan = pl.plan_run(only=only, skip=skip,
                           include_optional=args.include_optional,
                           scheduled_only=args.scheduled)
        print(f"{len(plan)} stages, in order:")
        for i, stage in enumerate(plan):
            deps = ", ".join(stage.requires) or "-"
            print(f"  {i:>2}. {stage.name:<20} {stage.kind:<10} "
                  f"after {deps}")
        return

    out = build(execute_stages=args.run, only=only, skip=skip,
                include_optional=args.include_optional,
                scheduled_only=args.scheduled, focus=args.focus)
    report = out["report"]

    print(f"tables  -> {TABLES.relative_to(REPO_ROOT)} "
          f"({len(out['written'])} csv)")
    print(f"figures -> {FIGURES.relative_to(REPO_ROOT)} "
          f"({len(out['figures'])} png)")
    print(f"report  -> {REPORT.relative_to(REPO_ROOT)}")
    print(f"\nDAG lints clean: {report['lint']['checks']} checks over "
          f"{report['stages_declared']} stages")

    sources = report["sources"]
    print(f"schedule mode: {sources['mode']} — {sources['reason']}")

    drift = report["fact_drift"]
    if drift["stale"]:
        print(f"deck facts: {drift['moved']} of {drift['facts']} have moved "
              f"since Task 10 ({', '.join(drift['keys'])})")
    else:
        print(f"deck facts: all {drift['facts']} still match the repository")

    findings = out["audit"][out["audit"].status == "fail"]
    if len(findings):
        print(f"\npipeline audit: {len(findings)} of "
              f"{report['audit']['checks']} checks report a finding")
        for row in findings.itertuples():
            print(f"  - {row.check}: {row.subject} — {row.detail}")
    else:
        print(f"\npipeline audit: all {report['audit']['checks']} checks pass")

    if report["mode"] == "run":
        run = report["run"]
        print(f"\nrun verdict: {run['state']} over {run['stages_run']} stages")
        print(f"  {run['artefacts_changed']} artefacts changed, "
              f"{run['artefacts_volatile_only']} moved only their clock")
        for path in run["changed_paths"]:
            print(f"  ~ {path}")
        if run["failed"]:
            raise SystemExit("stages failed: " + ", ".join(run["failed"]))


if __name__ == "__main__":
    main()
