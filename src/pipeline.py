"""Task 11 (optional) — the pipeline layer: the DAG, its rules, and what a schedule may claim.

The brief asks for collection, preprocessing, skill extraction, trend analysis
and forecasting to be connected into "one scheduled, end-to-end pipeline". Ten
tasks already built the stages; the work left is the *connective tissue*, and
the connective tissue is where an analysis project quietly stops being
reproducible.

Three failure modes motivate everything in this module.

**The dependency nobody declared.** `build_competitor_set.py` reads
`data/raw/google/_hf_data_jobs_full.parquet` — the 75 MB full backfill cache
that `collect_google_jobs.py` downloads as a side effect of collecting Google.
Nothing in either file says so. On a developer's machine the cache is always
there, so the edge is invisible; on a fresh clone, where `data/` is git-ignored
in its entirety, the stage fails. A pipeline that only *works* is not a
pipeline that is *understood*, so the DAG here is declared as data and its
edges are checked against what the stages actually read.

**The diff that means nothing.** Re-running all ten stages over the frozen 2023
backfill rewrites every committed table, figure and the deck — byte for byte
identically. The only bytes that move are ten clock fields in ten JSON reports.
A scheduled job that committed its output would therefore produce a commit per
run carrying ten timestamp lines and no information, and after a month of that
nobody reads the diffs, which is precisely when a real change would slip
through. So "did anything change" is answered here by a **stable digest** that
strips declared volatile fields, and the question a schedule exists to answer
becomes a yes/no rather than a diff to eyeball.

**The refresh with nothing to refresh.** A schedule implies new data. This
project has three approved sources and *none* of them can return a posting it
has not already returned: the Hugging Face backfill is a frozen 2023 snapshot,
Adzuna has no credentials and truncates descriptions to 500 characters, and The
Muse lists no Google postings. Automating a nightly refresh against that would
be theatre. `refreshability_table` says so per source, and `schedule_rationale`
derives the cadence from it instead of from habit — which is the same move
Task 07 made when the forecast horizon came out at zero and the forecast
shipped marked unsupported.

What this module does *not* do is run anything. It is the layer: records,
rules and verdicts, importable by the runner and by tests. `src/run_pipeline.py`
executes it. As with every core module since Task 03, nothing here imports
scipy — the arithmetic is small and hand-rolled, and the cross-check against a
solver library lives in a separate `validate_*.py`.

Sections
--------
A. The stage registry — the DAG as data
B. DAG mechanics — order, edges, plans
C. Volatility — what a re-run is allowed to change
D. Sources — what a schedule could possibly refresh
E. The DAG linter — rules that fail the build
F. The pipeline audit — findings that do not
G. Drift — a generated number against the repository it counted
H. The run ledger
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMBER = REPO_ROOT / "members" / "ankit-google"


# ---------------------------------------------------------------------------
# A. The stage registry — the DAG as data
# ---------------------------------------------------------------------------
#
# A stage is a record, not a shell line in a README. Everything the rules in
# section E check is a field here, which is the only reason those rules can be
# exhaustive: there is no stage that exists but is not registered, because the
# registry is what the runner iterates.

#: What a stage does. The kind decides how the scheduled workflow treats it.
STAGE_KINDS = ("collect", "transform", "analyse", "publish", "validate", "gate")

#: How much a re-run of the same stage on the same inputs may change, ordered
#: from strictest to loosest. A stage declares its class; section F measures
#: what it actually did and compares. The declaration is the claim; the
#: measurement is the evidence.
VOLATILITY = ("deterministic", "clock_stamped", "source_volatile")


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline, and everything a rule needs to check it.

    ``reads`` and ``writes`` are repository-relative paths. A trailing ``/``
    means "this directory", and a ``*`` is a glob — both are resolved against
    the working tree when a rule needs the real file list, and compared as
    strings when a rule is about the declaration itself.

    ``requires`` is the declared dependency edge. It is *not* derived from
    ``reads``, deliberately: deriving it would make ``requires_covers_reads``
    vacuous, and that rule is the one that caught the undeclared cache edge
    this module's docstring opens with.
    """

    name: str
    task: int
    title: str
    kind: str
    command: tuple[str, ...]
    requires: tuple[str, ...] = ()
    #: Stages that must finish first when they run at all, without being pulled
    #: into the run if they would not otherwise be in it. `requires` says "I
    #: need what this produced"; `after` says "if this runs, it runs before
    #: me". The distinction exists because an *optional* stage can still hold
    #: an ordering constraint: `validate-skills` reads a frame `competitor-set`
    #: overwrites, so it must precede it — but making it a `requires` would
    #: mean the default run depends on a stage the default run skips.
    after: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    volatility: str = "deterministic"
    scheduled: bool = True
    optional: bool = False
    note: str = ""

    def __post_init__(self):
        if self.kind not in STAGE_KINDS:
            raise ValueError(f"stage {self.name!r} has unknown kind {self.kind!r}")
        if self.volatility not in VOLATILITY:
            raise ValueError(
                f"stage {self.name!r} has unknown volatility {self.volatility!r}")
        if not self.command:
            raise ValueError(f"stage {self.name!r} has no command")


#: The pipeline. Order in this tuple is documentation; the order the runner
#: uses comes from `topological_order`, so adding a stage in the wrong place
#: cannot change what runs before what.
STAGES: tuple[Stage, ...] = (
    Stage(
        name="collect",
        task=2,
        title="Collect the approved backfill",
        kind="collect",
        command=("python", "src/collect_google_jobs.py", "hf-backfill"),
        reads=("https://huggingface.co/datasets/lukebarousse/data_jobs",),
        writes=("data/raw/google/_hf_data_jobs_full.parquet",
                "data/raw/google/google_jobs_hf_backfill_2023.parquet",
                "data/raw/google/google_jobs_hf_backfill_2023.csv",
                "members/ankit-google/data-manifest.json"),
        volatility="clock_stamped",
        note="The only stage that reaches the network. Its 75 MB cache is also "
             "the competitor set's input, which is the edge nothing declared "
             "before this task.",
    ),
    Stage(
        name="preprocess",
        task=3,
        title="Clean and normalise to the shared schema",
        kind="transform",
        command=("python", "src/preprocess.py", "--company", "google"),
        requires=("collect",),
        reads=("data/raw/google/google_jobs_hf_backfill_2023.parquet",),
        writes=("data/processed/google/google_jobs_clean.parquet",
                "data/processed/google/google_jobs_clean.csv",
                "members/ankit-google/task-03-quality-report.json"),
        volatility="clock_stamped",
    ),
    Stage(
        name="features",
        task=4,
        title="Extract skills and build the feature tables",
        kind="transform",
        command=("python", "src/build_features.py", "--company", "google"),
        requires=("preprocess",),
        reads=("data/processed/google/google_jobs_clean.parquet",),
        writes=("data/processed/google/google_features.parquet",
                "data/processed/google/google_skills_long.parquet",
                "data/processed/google/google_skill_matrix.parquet",
                "members/ankit-google/task-04-tables/",
                "members/ankit-google/task-04-feature-report.json"),
        volatility="clock_stamped",
    ),
    Stage(
        name="trends",
        task=5,
        title="Hiring trends on a declared publisher panel",
        kind="analyse",
        command=("python", "src/build_trends.py", "--company", "google"),
        requires=("features",),
        reads=("data/processed/google/google_features.parquet",
               "data/processed/google/google_skills_long.parquet"),
        writes=("members/ankit-google/task-05-tables/",
                "members/ankit-google/task-05-figures/",
                "members/ankit-google/task-05-trend-report.json"),
        volatility="clock_stamped",
    ),
    Stage(
        name="competitor-set",
        task=6,
        title="Build the five rivals through the same pipeline",
        kind="transform",
        command=("python", "src/build_competitor_set.py"),
        requires=("collect", "trends"),
        after=("validate-skills",),
        reads=("data/raw/google/_hf_data_jobs_full.parquet",),
        writes=("data/raw/task-06/",
                "data/processed/",
                # Named explicitly even though `data/processed/` already covers
                # it, because this is the contested path: `features` writes 848
                # Google rows here and this stage overwrites them with 846. A
                # directory-shaped declaration hides that behind a prefix.
                "data/processed/google/google_features.parquet",
                "data/processed/google/google_skills_long.parquet",
                "members/ankit-google/task-06-tables/company-feasibility-screen.csv",
                "members/ankit-google/task-06-tables/employer-matching-audit.csv",
                "members/ankit-google/task-06-tables/competitor-set-manifest.csv",
                "members/ankit-google/task-06-tables/competitor-set-manifest.json"),
        volatility="clock_stamped",
        note="Reads the full cache, not the Google slice, so the `requires` "
             "edge to collect is about a file collect writes as a by-product "
             "— exactly the kind of edge a README loses. The edge to trends "
             "is not a data edge at all: this stage overwrites the Google "
             "features frame with the 846-row selection (C4), and trends must "
             "have already read the 848-row one Task 05 reported on. Run it "
             "earlier and Task 05's committed numbers change.",
    ),
    Stage(
        name="comparison",
        task=6,
        title="Compare the six on a common publisher pool",
        kind="analyse",
        command=("python", "src/build_comparison.py"),
        requires=("competitor-set", "features"),
        reads=("data/processed/",
               "members/ankit-google/task-06-tables/company-feasibility-screen.csv"),
        writes=("members/ankit-google/task-06-tables/",
                "members/ankit-google/task-06-figures/",
                "members/ankit-google/task-06-comparison-report.json"),
        volatility="clock_stamped",
    ),
    Stage(
        name="forecast",
        task=7,
        title="Gate, backtest and publish the forecast",
        kind="analyse",
        command=("python", "src/build_forecast.py"),
        requires=("comparison",),
        reads=("data/processed/",
               "members/ankit-google/task-06-tables/"),
        writes=("members/ankit-google/task-07-tables/",
                "members/ankit-google/task-07-figures/",
                "members/ankit-google/task-07-forecast-report.json"),
        volatility="clock_stamped",
    ),
    Stage(
        name="similarity",
        task=8,
        title="Score company similarity against two nulls",
        kind="analyse",
        command=("python", "src/build_similarity.py"),
        requires=("comparison", "forecast"),
        reads=("data/processed/",
               "members/ankit-google/task-07-tables/panel-share-series.csv",
               "members/ankit-google/task-07-tables/forecastability-gate.csv"),
        writes=("members/ankit-google/task-08-tables/",
                "members/ankit-google/task-08-figures/",
                "members/ankit-google/task-08-similarity-report.json"),
        volatility="clock_stamped",
        note="Bootstraps and nulls are seeded from similarity.SEED, which is "
             "why an interval is reproducible rather than merely stable.",
    ),
    Stage(
        name="insights",
        task=9,
        title="Generate every candidate claim and gate it",
        kind="analyse",
        command=("python", "src/build_insights.py"),
        requires=("trends", "comparison", "forecast", "similarity"),
        reads=("data/processed/",
               "members/ankit-google/task-06-tables/"),
        writes=("members/ankit-google/task-09-tables/",
                "members/ankit-google/task-09-figures/",
                "members/ankit-google/task-09-insight-report.json"),
        volatility="clock_stamped",
        note="Recomputes from the processed frames rather than re-reading the "
             "earlier tasks' tables, so three of its four edges are "
             "sequencing: the ledger cites Task 05, 07 and 08 findings and "
             "must not be built against a stale version of them.",
    ),
    Stage(
        name="presentation",
        task=10,
        title="Compile the deck and audit the workspace",
        kind="publish",
        command=("python", "src/build_presentation.py"),
        requires=("insights",),
        reads=("members/ankit-google/task-09-tables/claim-ledger.csv",
               "members/ankit-google/task-09-tables/prohibited-patterns.csv",
               "members/ankit-google/task-06-tables/",
               "README.md",
               "docs/corrections.md",
               "tests/"),
        writes=("members/ankit-google/task-10-tables/",
                "members/ankit-google/task-10-figures/",
                "members/ankit-google/task-10-slides.md",
                "members/ankit-google/task-10-presentation-report.json"),
        volatility="clock_stamped",
        note="Counts the repository, and the repository contains its own "
             "output. Within a run it iterates to a fixpoint; across tasks it "
             "cannot — see §8.",
    ),
    Stage(
        name="tests",
        task=11,
        title="The suite, as the gate on everything above",
        kind="gate",
        command=("python", "-m", "pytest", "tests/", "-q"),
        requires=("presentation",),
        reads=("tests/", "src/"),
        writes=(),
        note="Writes nothing and blocks nothing upstream. It is last because "
             "several tests read committed tables, so it grades the artefacts "
             "the run just produced rather than the ones it found.",
    ),
    # ---- validators: the scipy cross-checks, deliberately out of the schedule
    Stage(
        name="validate-skills",
        task=4,
        title="Cross-check skill extraction against scikit-learn",
        kind="validate",
        command=("python", "src/validate_skill_extraction.py"),
        requires=("features",),
        reads=("data/processed/google/",),
        writes=("docs/task-04-skill-extraction-validation.md",),
        optional=True,
    ),
    Stage(
        name="validate-forecast",
        task=7,
        title="Cross-check the forecast against scipy and statsmodels",
        kind="validate",
        command=("python", "src/validate_forecast.py"),
        requires=("forecast",),
        reads=("members/ankit-google/task-07-tables/",),
        writes=("docs/task-07-forecast-validation.md",),
        volatility="clock_stamped",
        optional=True,
    ),
    Stage(
        name="validate-similarity",
        task=8,
        title="Cross-check similarity against scipy",
        kind="validate",
        command=("python", "src/validate_similarity.py"),
        requires=("similarity",),
        reads=("members/ankit-google/task-08-tables/",),
        writes=("docs/task-08-similarity-validation.md",),
        optional=True,
    ),
    Stage(
        name="validate-insights",
        task=9,
        title="Cross-check Task 09's hand-written statistics against scipy",
        kind="validate",
        command=("python", "src/validate_insights.py"),
        requires=("insights",),
        reads=("members/ankit-google/task-09-tables/",),
        writes=("docs/task-09-insight-validation.md",),
        optional=True,
    ),
    Stage(
        name="validate-text",
        task=3,
        title="Exercise the Layer B cleaner on live posting HTML",
        kind="validate",
        command=("python", "src/validate_text_pipeline.py"),
        requires=("preprocess",),
        reads=("https://www.arbeitnow.com/api/job-board-api",),
        writes=("docs/task-03-text-pipeline-validation.md",),
        volatility="source_volatile",
        scheduled=False,
        optional=True,
        note="Samples three postings from a rolling public feed, so its output "
             "is different every run by construction — 42 diff lines that "
             "carry no information about this project. Kept out of the "
             "schedule for that reason, not because it is unimportant: it is "
             "the only exercise the Layer B text path gets.",
    ),
)

STAGES_BY_NAME = {stage.name: stage for stage in STAGES}

#: Tasks the pipeline is required to cover. Task 01 is a legal review with no
#: code, and Task 11 is the pipeline itself.
COVERED_TASKS = (2, 3, 4, 5, 6, 7, 8, 9, 10)


def stage_table(stages: tuple[Stage, ...] = STAGES) -> pd.DataFrame:
    """The registry as a committed table — the pipeline's own documentation."""
    order = {name: i for i, name in enumerate(topological_order(stages))}
    rows = [{
        "stage": s.name,
        "task": s.task,
        "run_order": order.get(s.name, -1),
        "kind": s.kind,
        "title": s.title,
        "command": " ".join(s.command),
        "requires": " ".join(s.requires),
        "after": " ".join(s.after),
        "reads": len(s.reads),
        "writes": len(s.writes),
        "volatility": s.volatility,
        "scheduled": s.scheduled,
        "optional": s.optional,
    } for s in stages]
    return pd.DataFrame(rows).sort_values("run_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# B. DAG mechanics — order, edges, plans
# ---------------------------------------------------------------------------


class CycleError(ValueError):
    """Raised when the declared dependencies do not admit a run order."""


def deps(stage: Stage) -> tuple[str, ...]:
    """Everything that must precede a stage, for both reasons.

    Ordering never distinguishes the two: a sequencing edge constrains the run
    order exactly as hard as a data edge. Only selection distinguishes them,
    and only in one direction — see `plan_run`.
    """
    return tuple(stage.requires) + tuple(stage.after)


def topological_order(stages: tuple[Stage, ...] = STAGES) -> list[str]:
    """A deterministic run order, or `CycleError`.

    Kahn's algorithm with the ready set drained in registry order rather than
    alphabetically or by whatever a set iterates today. Two runs of the same
    registry must produce the same order, because the run order appears in a
    committed table and a table that reshuffles itself is a diff nobody can
    read.
    """
    index = {s.name: i for i, s in enumerate(stages)}
    pending = {s.name: set(deps(s)) for s in stages}
    order: list[str] = []
    while pending:
        ready = sorted((n for n, waiting in pending.items() if not waiting),
                       key=index.__getitem__)
        if not ready:
            raise CycleError(
                "dependencies do not admit a run order; unresolved: "
                + ", ".join(sorted(pending)))
        for name in ready:
            order.append(name)
            del pending[name]
        for waiting in pending.values():
            waiting.difference_update(ready)
    return order


def dag_edges(stages: tuple[Stage, ...] = STAGES) -> pd.DataFrame:
    """Every declared edge, with the artefact that justifies it where there is one.

    An edge whose `via` is empty is a *sequencing* edge: the stage does not read
    the upstream stage's output, it just may not run before it. Naming those
    separately stops them being read as data dependencies — and stops the
    reverse mistake, which is deleting one because "nothing flows along it".
    """
    by_name = {s.name: s for s in stages}
    rows = []
    for stage in stages:
        for parent in deps(stage):
            upstream = by_name.get(parent)
            shared: set[str] = set()
            if upstream is not None:
                for written in upstream.writes:
                    for read in stage.reads:
                        if _covers(written, read):
                            # Name the narrower of the two. One side is often a
                            # directory and the other the file inside it, and
                            # "task-09-tables/claim-ledger.csv" tells a reader
                            # what the edge is for where "task-09-tables/" does
                            # not.
                            shared.add(max((written, read), key=len))
            via = "; ".join(sorted(shared))
            rows.append({"parent": parent, "child": stage.name,
                         "via": via,
                         "edge": "data" if via else "sequencing"})
    return pd.DataFrame(rows, columns=["parent", "child", "via", "edge"])


def _covers(written: str, read: str) -> bool:
    """Does an output path account for an input path?

    Directory-shaped declarations cover anything beneath them in either
    direction: `data/processed/` covers a read of
    `data/processed/google/x.parquet`, and a *write* of
    `data/processed/google/google_features.parquet` is covered by a read
    declared as `data/processed/`. Both directions occur in the registry and
    both are genuine.
    """
    written, read = written.rstrip("/"), read.rstrip("/")
    if written == read:
        return True
    return read.startswith(written + "/") or written.startswith(read + "/")


#: One row per path that more than one stage writes. `sees` is the column that
#: matters: it names, per reader, whose content that reader actually gets.
CONTESTED_COLUMNS = ("path", "writers", "winner", "readers", "sees",
                     "writers_ordered", "readers_pinned")


def contested_artefacts(stages: tuple[Stage, ...] = STAGES) -> pd.DataFrame:
    """Paths with more than one writer, and which writer each reader sees.

    A path two stages both write has no content of its own. What a reader gets
    is whatever the last writer left, so the content is a property of the run
    order rather than of any one stage. That is perfectly sound when the DAG
    fixes the order and silently wrong when it does not — and "silently" is the
    operative word, because nothing fails: you get a complete, plausible,
    different number.

    This repository has exactly one such path and it is load-bearing.
    `data/processed/google/google_features.parquet` is written by `features`
    with 848 Google rows and by `competitor-set` with 846, and the difference
    is C4 — the correction that removed a reseller and a role string from the
    denominator. Task 05 reported on 848 and Task 06 onward reports on 846, so
    *both* contents are correct, for different readers. The repository is only
    reproducible if `trends` reads before `competitor-set` writes.
    """
    order = topological_order(stages)
    rank = {name: i for i, name in enumerate(order)}
    writes = [(s.name, w) for s in stages for w in s.writes]

    groups: dict[str, set[str]] = {}
    for i, (a_stage, a_path) in enumerate(writes):
        for b_stage, b_path in writes[i + 1:]:
            if a_stage == b_stage or not _covers(a_path, b_path):
                continue
            # Key on the narrower path: `data/processed/` and
            # `.../google_features.parquet` are the same contest, and naming it
            # by the directory would lose which file is actually at stake.
            groups.setdefault(max((a_path, b_path), key=len), set()).update(
                {a_stage, b_stage})

    rows = []
    for path in sorted(groups):
        writers = sorted(groups[path], key=lambda n: rank[n])
        readers = sorted((s.name for s in stages
                          if any(_covers(r, path) for r in s.reads)),
                         key=lambda n: rank[n])

        # Ordered means: for every pair, one is a transitive ancestor of the
        # other. Anything less and the run order is an accident of list order.
        ordered = all(w2 in ancestors(w1, stages) or w1 in ancestors(w2, stages)
                      for i, w1 in enumerate(writers) for w2 in writers[i + 1:])

        sees, pinned = [], True
        for reader in readers:
            anc = ancestors(reader, stages)
            # A stage that reads a path it also writes reads the version that
            # was there when it started, so its own write is never ambiguous to
            # it. `comparison` does exactly this with the feasibility screen.
            others = [w for w in writers if w != reader]
            upstream = [w for w in others if w in anc]
            # Otherwise a reader is pinned only if every writer is either
            # upstream or downstream of it. A writer that is neither can land
            # on either side, and then `sees` is a coin toss.
            if any(w not in anc and reader not in ancestors(w, stages)
                   for w in others):
                pinned = False
                sees.append(f"{reader}=undetermined")
            else:
                sees.append(f"{reader}={upstream[-1] if upstream else 'itself'}")

        rows.append({
            "path": path,
            "writers": "; ".join(writers),
            "winner": writers[-1] if ordered else "undetermined",
            "readers": "; ".join(readers),
            "sees": "; ".join(sees),
            "writers_ordered": ordered,
            "readers_pinned": pinned,
        })
    return pd.DataFrame(rows, columns=list(CONTESTED_COLUMNS))


def plan_run(stages: tuple[Stage, ...] = STAGES, *,
             only: tuple[str, ...] = (),
             skip: tuple[str, ...] = (),
             include_optional: bool = False,
             scheduled_only: bool = False) -> list[Stage]:
    """Resolve which stages run, in order, under a selection.

    `only` is expanded to include everything it depends on, because "run the
    forecast" on a fresh clone means "run what the forecast needs". A selection
    that silently ran one stage against absent inputs would fail in a way that
    looks like a code bug rather than a plan bug.
    """
    chosen = {s.name for s in stages}
    if only:
        wanted: set[str] = set()
        queue = list(only)
        while queue:
            name = queue.pop()
            if name in wanted:
                continue
            if name not in STAGES_BY_NAME:
                raise KeyError(f"unknown stage {name!r}")
            wanted.add(name)
            # `requires` only. Following `after` here would drag an optional
            # validator into `--only comparison` to satisfy an ordering rule
            # that is vacuous when the validator is not running.
            queue.extend(STAGES_BY_NAME[name].requires)
        chosen &= wanted
    chosen -= set(skip)
    order = topological_order(stages)
    out = []
    for name in order:
        stage = STAGES_BY_NAME[name]
        if name not in chosen:
            continue
        if stage.optional and not include_optional and name not in only:
            continue
        if scheduled_only and not stage.scheduled:
            continue
        out.append(stage)
    return out


def ancestors(name: str, stages: tuple[Stage, ...] = STAGES) -> set[str]:
    """Every stage that must have run before this one, transitively.

    Ordering is guaranteed by any ancestor, not only by a direct parent, so
    this is what `requires_covers_reads` checks against. Demanding a direct
    edge would force the registry to carry the full transitive closure —
    every redundant edge spelled out — and a graph nobody can read is a graph
    nobody maintains. The declared edges stay the transitive reduction.
    """
    by_name = {s.name: s for s in stages}
    seen: set[str] = set()
    queue = list(deps(by_name[name])) if name in by_name else []
    while queue:
        current = queue.pop()
        if current in seen or current not in by_name:
            continue
        seen.add(current)
        queue.extend(deps(by_name[current]))
    return seen


def blocked_by(failed: set[str], stages: tuple[Stage, ...] = STAGES) -> set[str]:
    """Everything downstream of a failure, transitively.

    A stage whose input never got written did not fail; it was never eligible.
    Recording it as `blocked` rather than `failed` is the difference between a
    run ledger that names one defect and one that names seven.
    """
    blocked: set[str] = set()
    changed = True
    while changed:
        changed = False
        for stage in stages:
            if stage.name in failed or stage.name in blocked:
                continue
            if set(deps(stage)) & (failed | blocked):
                blocked.add(stage.name)
                changed = True
    return blocked


# ---------------------------------------------------------------------------
# C. Volatility — what a re-run is allowed to change
# ---------------------------------------------------------------------------
#
# The whole point of a scheduled rebuild is the question "did anything change".
# Answering it with `git diff` fails, because every JSON report carries the
# wall clock and one validator samples a rolling feed. Both are legitimate;
# neither is a finding. So a digest is taken over the artefact with the
# declared volatile parts removed, and *that* is what a run compares.

#: JSON keys whose value is the wall clock. Stripped before hashing, at any
#: depth. Registered rather than pattern-matched, so adding a timestamp to a
#: report is a deliberate act that shows up in this list.
VOLATILE_JSON_KEYS = frozenset({
    "generated", "generated_at", "collected_at", "scraped_date", "built_at",
    "run_at", "timestamp",
})

#: Markdown lines that are a timestamp and nothing else.
VOLATILE_LINE = re.compile(
    r"^\s*[_*]*\s*(generated|built|last updated|last run)\b.*$",
    re.IGNORECASE)


def strip_volatile(value):
    """Remove every registered clock field, recursively, leaving structure."""
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items()
                if k not in VOLATILE_JSON_KEYS}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def raw_digest(path: Path) -> str:
    """SHA-256 of the bytes, which is what `git diff` effectively compares."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: Significant figures a numeric table is compared at. A float64 written in
#: full — `0.500884621879673` — does not reproduce to its last digit across
#: platforms: the Linux runner's BLAS and libm land a ULP away from macOS, so a
#: byte comparison of a CSV reports a change that is not one, exactly as a
#: wall-clock line does in a JSON report. Six figures is finer than anything the
#: analysis reports (three to four decimals) and many orders coarser than that
#: platform noise, so a real move survives and the noise does not.
CSV_SIGNIFICANT_FIGURES = 6


def _canonical_number(value) -> str:
    """A float at the comparison precision, with the sign of zero removed."""
    if pd.isna(value):
        return "nan"
    if value == 0:                      # -0.0 and 0.0 are the same number, and
        return "0"                      # their sign is not reproducible either
    return format(float(value), f".{CSV_SIGNIFICANT_FIGURES}g")


def csv_content_digest(path: Path) -> str:
    """SHA-256 of a table with its floats quantised to the reported precision.

    Integer and text columns pass through untouched, so a changed label, a new
    column or a different row count is still a change; only the trailing,
    platform-dependent digits of a float are normalised away before hashing.
    """
    frame = pd.read_csv(path)
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(_canonical_number)
    canonical = frame.to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


#: A PNG's signature and IHDR chunk sit at a fixed offset and carry width,
#: height, bit depth and colour type. `figsize * dpi` fixes those in the code,
#: so they are identical on every runner; the pixels below them are not.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_HEADER_BYTES = 33


def figure_shape_digest(path: Path) -> str:
    """SHA-256 of a figure's dimensions, not its pixels.

    Figure bytes are not reproducible across matplotlib and freetype builds —
    the same code renders text a pixel differently on another runner — which is
    why the workflow's check job diffs the tables and not the figures: the
    tables carry every number a figure draws. A figure that changes *shape* (a
    stage added to a diagram, a row count moved) still moves its IHDR, and is
    still a change.
    """
    header = path.read_bytes()[:PNG_HEADER_BYTES]
    if len(header) < PNG_HEADER_BYTES or not header.startswith(PNG_SIGNATURE):
        return raw_digest(path)
    return hashlib.sha256(header).hexdigest()


def stable_digest(path: Path) -> str:
    """SHA-256 of the artefact's *content*, with platform noise removed.

    JSON is parsed and re-serialised canonically, so key order and whitespace
    cannot masquerade as a change, and registered clock fields are dropped.
    Markdown drops whole lines that are nothing but a generation stamp. A CSV's
    floats are quantised to the precision the analysis reports, because their
    last digit is not reproducible across platforms. A figure is reduced to its
    dimensions, because its pixels are not. Anything else is hashed as bytes.
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            return raw_digest(path)
        canonical = json.dumps(strip_volatile(parsed), sort_keys=True,
                               separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if suffix == ".md":
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return raw_digest(path)
        kept = [ln for ln in text.splitlines() if not VOLATILE_LINE.match(ln)]
        return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()
    if suffix == ".csv":
        try:
            return csv_content_digest(path)
        except (OSError, ValueError, pd.errors.ParserError,
                pd.errors.EmptyDataError):
            return raw_digest(path)
    if suffix == ".png":
        try:
            return figure_shape_digest(path)
        except OSError:
            return raw_digest(path)
    return raw_digest(path)


#: Everything a run may rewrite that git carries. Row-level output under
#: `data/` is git-ignored by Task 01's data-handling rule and is not snapshot:
#: it is an input to the next stage, not a deliverable.
SNAPSHOT_GLOBS = (
    "members/*/task-*-tables/*.csv",
    "members/*/task-*-tables/*.json",
    "members/*/task-*-figures/*.png",
    "members/*/*report*.json",
    "members/*/data-manifest.json",
    "members/*/task-*-slides.md",
    "docs/task-*-validation.md",
)


def snapshot(repo_root: Path = REPO_ROOT,
             globs: tuple[str, ...] = SNAPSHOT_GLOBS) -> dict:
    """Digest every rebuildable committed artefact, raw and stable."""
    out = {}
    for pattern in globs:
        for path in sorted(repo_root.glob(pattern)):
            key = str(path.relative_to(repo_root))
            out[key] = {"raw": raw_digest(path), "stable": stable_digest(path)}
    return out


REPRO_COLUMNS = ("path", "verdict", "detail")

#: What a comparison of two snapshots can conclude, worst last.
REPRO_VERDICTS = ("identical", "volatile_only", "changed", "added", "removed")


def reproducibility_audit(before: dict, after: dict) -> pd.DataFrame:
    """Compare two snapshots and say, per artefact, what moved.

    `volatile_only` is the interesting verdict and the reason the whole
    apparatus exists: the bytes differ, the content does not, and a schedule
    that treated it as a change would commit noise forever.
    """
    rows = []
    for path in sorted(set(before) | set(after)):
        was, now = before.get(path), after.get(path)
        if was is None:
            rows.append({"path": path, "verdict": "added",
                         "detail": "not present before the run"})
        elif now is None:
            rows.append({"path": path, "verdict": "removed",
                         "detail": "present before the run, gone after"})
        elif was["raw"] == now["raw"]:
            rows.append({"path": path, "verdict": "identical",
                         "detail": "byte for byte"})
        elif was["stable"] == now["stable"]:
            rows.append({"path": path, "verdict": "volatile_only",
                         "detail": "bytes moved; content did not"})
        else:
            rows.append({"path": path, "verdict": "changed",
                         "detail": "content differs"})
    return pd.DataFrame(rows, columns=list(REPRO_COLUMNS))


def measured_volatility(audit: pd.DataFrame, stage: Stage) -> str:
    """The strictest class consistent with what this stage's outputs did."""
    mine = audit[audit.path.apply(
        lambda p: any(_covers(w, p) for w in stage.writes))]
    if mine.empty or (mine.verdict == "identical").all():
        return "deterministic"
    if set(mine.verdict) <= {"identical", "volatile_only"}:
        return "clock_stamped"
    return "source_volatile"


def volatility_rank(name: str) -> int:
    return VOLATILITY.index(name)


# ---------------------------------------------------------------------------
# D. Sources — what a schedule could possibly refresh
# ---------------------------------------------------------------------------
#
# A schedule is a claim that running again produces something new. That claim
# is about the *sources*, not about the code, and it is checkable one source at
# a time. Task 01 approved three; this is what each can deliver today.


@dataclass(frozen=True)
class Source:
    """One approved source, and whether a re-run could return anything new."""

    name: str
    task: int
    licence: str
    window: str
    verdict: str
    blocker: str
    falsifier: str

    def __post_init__(self):
        if self.verdict not in SOURCE_VERDICTS:
            raise ValueError(f"source {self.name!r}: unknown verdict")


#: `frozen` — the source cannot grow. `blocked` — it could, but something
#: outside the code stops it. `live_empty` — it is live and returns nothing for
#: this company. `refreshable` — a re-run can return a posting we do not have.
SOURCE_VERDICTS = ("refreshable", "frozen", "blocked", "live_empty")

SOURCES: tuple[Source, ...] = (
    Source(
        name="hf:lukebarousse/data_jobs",
        task=2,
        licence="Apache-2.0",
        window="2023 only, single snapshot",
        verdict="frozen",
        blocker="A published snapshot of 2023 data-role postings. Re-downloading "
                "returns the same 785k rows; the file is the dataset, not a feed.",
        falsifier="The dataset publishes a later revision covering 2024+.",
    ),
    Source(
        name="adzuna",
        task=2,
        licence="official API, free tier",
        window="live, rolling",
        verdict="blocked",
        blocker="No ADZUNA_APP_ID / ADZUNA_APP_KEY in the environment, and the "
                "search endpoint truncates `description` to 500 characters, so "
                "even with keys the Layer B text path stays idle.",
        falsifier="Credentials are provisioned and a pull returns Google "
                  "postings with untruncated descriptions.",
    ),
    Source(
        name="themuse",
        task=2,
        licence="public API",
        window="live, rolling",
        verdict="live_empty",
        blocker="Live and reachable, but lists no Google postings (checked at "
                "Task 02, 2026-07-20).",
        falsifier="A Muse pull returns one or more Google postings.",
    ),
    Source(
        name="google-careers",
        task=1,
        licence="rejected",
        window="n/a",
        verdict="blocked",
        blocker="Rejected on robots.txt at Task 01. Not a scheduling question.",
        falsifier="None that this project would act on; the rejection is legal, "
                  "not technical.",
    ),
)


def refreshability_table(sources: tuple[Source, ...] = SOURCES) -> pd.DataFrame:
    """Per source: can a scheduled run return a posting we do not already have?"""
    return pd.DataFrame([{
        "source": s.name,
        "task": s.task,
        "licence": s.licence,
        "window": s.window,
        "verdict": s.verdict,
        "refreshable": s.verdict == "refreshable",
        "blocker": s.blocker,
        "falsifier": s.falsifier,
    } for s in sources])


#: Cadences a schedule could take, and what each would actually detect. The
#: column that decides is `detects`, not `cost`: a run that cannot detect
#: anything is not cheap, it is misleading.
CADENCES = (
    ("on_push", "every push and pull request",
     "a code change that breaks a stage or moves a number"),
    ("weekly", "Mondays, 06:00 UTC",
     "dependency and environment drift — a pandas or numpy release that "
     "changes a default under unchanged code"),
    ("daily", "every night",
     "the same as weekly, seven times as often"),
    ("monthly", "the first of the month",
     "new postings, if any source were refreshable"),
    ("on_demand", "workflow_dispatch",
     "whatever a human is investigating"),
)


def schedule_rationale(sources: tuple[Source, ...] = SOURCES) -> pd.DataFrame:
    """Derive the cadence from the sources rather than from habit.

    With no refreshable source, a *refresh* schedule detects nothing and a
    *verification* schedule still earns its place: the code and its
    dependencies keep moving even when the data does not. That is the whole
    justification for the cron in the workflow, and it is a much smaller claim
    than "the pipeline keeps the analysis up to date".
    """
    refreshable = [s for s in sources if s.verdict == "refreshable"]
    rows = []
    for name, when, detects in CADENCES:
        if name == "monthly":
            adopted = bool(refreshable)
            why = ("a source can return new postings" if refreshable else
                   f"no approved source is refreshable ({len(sources)} checked, "
                   "0 refreshable), so a refresh cadence detects nothing")
        elif name == "daily":
            adopted = False
            why = ("weekly already covers dependency drift; daily multiplies "
                   "runs without widening what they can find")
        else:
            adopted = True
            why = "detects something no other trigger does"
        rows.append({"cadence": name, "when": when, "detects": detects,
                     "adopted": adopted, "why": why})
    return pd.DataFrame(rows)


def refresh_verdict(sources: tuple[Source, ...] = SOURCES) -> dict:
    """The one-line answer: is a refresh schedule justified today?"""
    refreshable = [s.name for s in sources if s.verdict == "refreshable"]
    return {
        "sources_checked": len(sources),
        "refreshable": len(refreshable),
        "refreshable_names": refreshable,
        "mode": "refresh" if refreshable else "verify",
        "reason": ("at least one approved source can return new postings"
                   if refreshable else
                   "no approved source can return a posting the repository "
                   "does not already hold, so the schedule verifies rather "
                   "than refreshes"),
    }


# ---------------------------------------------------------------------------
# E. The DAG linter — rules that fail the build
# ---------------------------------------------------------------------------
#
# These are claims about the registry itself: they are true or false regardless
# of what is on disk, they are cheap, and a violation means the pipeline
# definition is wrong. Task 10's precedent is followed exactly — the linter
# fails the build, and the audit in section F does not.

LINT_COLUMNS = ("rule", "subject", "status", "detail")


def _l(rule, subject, status, detail):
    return {"rule": rule, "subject": str(subject), "status": status,
            "detail": detail}


def lint_dag(stages: tuple[Stage, ...] = STAGES,
             repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    """Every structural rule the pipeline definition must satisfy."""
    rows: list[dict] = []
    names = {s.name for s in stages}
    # Resolve against the registry passed in, never the global one. The first
    # draft used STAGES_BY_NAME here, which made every rule below vacuous on
    # any registry but the real one — including the toy graphs the tests use to
    # prove the rules have teeth.
    by_name = {s.name: s for s in stages}

    # -- the graph is a graph ----------------------------------------------
    try:
        order = topological_order(stages)
        rows.append(_l("dag_acyclic", f"{len(stages)} stages", "pass",
                       "dependencies admit a run order"))
    except CycleError as exc:
        order = []
        rows.append(_l("dag_acyclic", f"{len(stages)} stages", "fail", str(exc)))

    rows.append(_l("order_covers_registry", f"{len(order)} ordered",
                   "pass" if set(order) == names else "fail",
                   "every registered stage appears in the run order exactly once"))

    position = {name: i for i, name in enumerate(order)}
    for stage in stages:
        for parent in deps(stage):
            if parent not in by_name:
                rows.append(_l("requires_known", f"{stage.name} -> {parent}",
                               "fail", "depends on a stage that is not registered"))
            elif order and position[parent] >= position[stage.name]:
                rows.append(_l("order_is_topological",
                               f"{stage.name} -> {parent}", "fail",
                               "runs no later than the stage it depends on"))
    rows.append(_l("requires_known",
                   f"{sum(len(deps(s)) for s in stages)} edges",
                   "fail" if any(r["rule"] == "requires_known"
                                 and r["status"] == "fail" for r in rows)
                   else "pass",
                   "every dependency names a registered stage"))
    rows.append(_l("order_is_topological", f"{len(order)} stages",
                   "fail" if any(r["rule"] == "order_is_topological"
                                 and r["status"] == "fail" for r in rows)
                   else "pass",
                   "no stage runs before something it depends on"))

    # -- each stage is well formed -----------------------------------------
    for stage in stages:
        script = _script_of(stage)
        if script is None:
            rows.append(_l("command_resolves", stage.name, "pass",
                           "runs a module, not a repository script"))
        else:
            rows.append(_l("command_resolves", f"{stage.name}: {script}",
                           "pass" if (repo_root / script).exists() else "fail",
                           "the script the stage runs exists"))
        if stage.kind == "gate":
            continue
        rows.append(_l("writes_declared", stage.name,
                       "pass" if stage.writes else "fail",
                       "a stage that is not a gate declares an output"))

    # There was an `outputs_unique` rule here, asserting that no two stages
    # write the same path. It passed for one bad reason: `competitor-set`
    # declared the directory `data/processed/` while `features` declared a file
    # inside it, and the rule compared strings. Declaring the file honestly
    # made the rule fail on nine paths, every one of them deliberate — this
    # repository builds Google twice on purpose, once at 848 rows and once at
    # 846 (C4). A rule that forbids a thing the design requires is not a rule,
    # so it is gone, and `contested_writes_ordered` below carries the real
    # constraint: two writers are fine, two *unordered* writers are not.

    # -- every input is accounted for --------------------------------------
    # A path may have several writers, so this maps to a set. As a dict of
    # path -> one name it silently kept whichever stage came last in the
    # registry, which made `trends` look like it read a frame only
    # `competitor-set` produces when `features` produces it too.
    produced: dict[str, set[str]] = {}
    for stage in stages:
        for path in stage.writes:
            produced.setdefault(path, set()).add(stage.name)
    for stage in stages:
        for read in stage.reads:
            if read.startswith("http"):
                continue
            upstream = {n for w, names in produced.items() if _covers(w, read)
                        for n in names if n != stage.name}
            if upstream:
                reachable = ancestors(stage.name, stages)
                # One ancestor is enough. A writer that runs *after* this
                # stage is not its producer — it is the next version, and
                # `contested_reads_pinned` is the rule that governs that.
                if not (upstream & reachable):
                    rows.append(_l("requires_covers_reads",
                                   f"{stage.name} reads {read}", "fail",
                                   "written by " + ", ".join(sorted(upstream))
                                   + ", none of which is an ancestor"))
            elif not (repo_root / read.rstrip("/")).exists():
                rows.append(_l("inputs_produced", f"{stage.name} reads {read}",
                               "fail",
                               "no stage writes it and it is not in the repository"))
    rows.append(_l("requires_covers_reads",
                   f"{sum(len(s.reads) for s in stages)} declared inputs",
                   "fail" if any(r["rule"] == "requires_covers_reads"
                                 and r["status"] == "fail" for r in rows)
                   else "pass",
                   "a stage that reads another's output has at least one of "
                   "its writers as an ancestor"))
    rows.append(_l("inputs_produced",
                   f"{sum(len(s.reads) for s in stages)} declared inputs",
                   "fail" if any(r["rule"] == "inputs_produced"
                                 and r["status"] == "fail" for r in rows)
                   else "pass",
                   "every input is produced upstream or committed"))

    # -- the brief's five steps are all here -------------------------------
    covered = {s.task for s in stages}
    missing = [t for t in COVERED_TASKS if t not in covered]
    rows.append(_l("task_covered", f"tasks {COVERED_TASKS[0]}-{COVERED_TASKS[-1]}",
                   "fail" if missing else "pass",
                   f"uncovered: {missing}" if missing
                   else "every analysis task has at least one stage"))

    # -- a required stage may not depend on an optional one ----------------
    bad = [(s.name, p) for s in stages if not s.optional
           for p in s.requires
           if p in by_name and by_name[p].optional]
    for child, parent in bad:
        rows.append(_l("optional_not_required", f"{child} -> {parent}", "fail",
                       "a default-run stage depends on one the default run skips"))
    rows.append(_l("optional_not_required",
                   f"{sum(1 for s in stages if s.optional)} optional stages",
                   "fail" if bad else "pass",
                   "nothing in the default run needs a stage the default run skips"))

    # -- the schedule is closed under dependency ---------------------------
    unscheduled = {s.name for s in stages if not s.scheduled}
    leaks = [(s.name, p) for s in stages if s.scheduled
             for p in s.requires if p in unscheduled]
    for child, parent in leaks:
        rows.append(_l("schedule_closed", f"{child} -> {parent}", "fail",
                       "a scheduled stage depends on an unscheduled one"))
    rows.append(_l("schedule_closed", f"{len(unscheduled)} unscheduled",
                   "fail" if leaks else "pass",
                   "the scheduled subset depends on nothing outside itself"))

    # -- a path with two writers must have a declared winner ---------------
    # This pair of rules is the whole reason section B grew `contested_artefacts`
    # and Stage grew `after`. Nothing here fails loudly when it is violated: a
    # stage reads a path that a later stage will overwrite, gets a complete and
    # plausible frame, and reports a different number. The first version of this
    # registry ordered `competitor-set` before `trends` and silently reversed
    # C4 across 116 files. Neither rule existed then.
    try:
        contested = contested_artefacts(stages)
    except CycleError:
        contested = pd.DataFrame(columns=list(CONTESTED_COLUMNS))

    for _, row in contested[~contested.writers_ordered].iterrows():
        rows.append(_l("contested_writes_ordered", row.path, "fail",
                       f"written by {row.writers} with no order between them; "
                       "the surviving content is an accident of list order"))
    rows.append(_l("contested_writes_ordered", f"{len(contested)} contested paths",
                   "fail" if not contested.empty
                   and not contested.writers_ordered.all() else "pass",
                   "every path with two writers has them ordered, so one wins by "
                   "declaration"))

    for _, row in contested[~contested.readers_pinned].iterrows():
        rows.append(_l("contested_reads_pinned", row.path, "fail",
                       f"{row.sees} — a reader is unordered against a writer, so "
                       "which content it reads depends on the run order"))
    rows.append(_l("contested_reads_pinned",
                   f"{contested.readers.str.count(';').sum() + len(contested) if not contested.empty else 0} reads",
                   "fail" if not contested.empty
                   and not contested.readers_pinned.all() else "pass",
                   "every reader of a contested path is ordered against every "
                   "writer of it"))

    return pd.DataFrame(rows, columns=list(LINT_COLUMNS))


def _script_of(stage: Stage) -> str | None:
    """The repository file a stage runs, or None if it runs a module."""
    for token in stage.command:
        if token.endswith(".py"):
            return token
    return None


def lint_violations(lint: pd.DataFrame) -> pd.DataFrame:
    return lint[lint.status == "fail"]


# ---------------------------------------------------------------------------
# F. The pipeline audit — findings that do not fail the build
# ---------------------------------------------------------------------------
#
# These are claims about the repository and the environment. They can be false
# for reasons the pipeline is not allowed to fix on its own — a missing
# credential, a source-volatile document someone chose to commit — so they are
# reported, exactly as Task 10's workspace audit is reported.

AUDIT_COLUMNS = ("area", "check", "subject", "status", "detail")


def _a(area, check, subject, status, detail):
    return {"area": area, "check": check, "subject": str(subject),
            "status": status, "detail": detail}


def git_ignored(paths: list[str], repo_root: Path = REPO_ROOT) -> set[str]:
    """Which of these paths git is ignoring — asked of git, not of a regex.

    A directory-shaped declaration is probed as a file inside it. `.gitignore`
    carries `data/processed/*`, which ignores everything under the directory
    without ignoring the directory itself, so asking about `data/processed/`
    returns nothing and the check would report a violation that does not
    exist. What the rule means is "would a file written here be ignored", so
    that is what it asks.
    """
    if not paths:
        return set()
    probe = {p: (p.rstrip("/") + "/__probe__" if p.endswith("/") else p)
             for p in paths}
    try:
        done = subprocess.run(["git", "check-ignore", "--stdin"],
                              cwd=repo_root, input="\n".join(probe.values()),
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return set()
    hit = {line.strip() for line in done.stdout.splitlines() if line.strip()}
    return {original for original, asked in probe.items() if asked in hit}


def tracked_files(repo_root: Path = REPO_ROOT) -> set[str]:
    try:
        done = subprocess.run(["git", "ls-files"], cwd=repo_root,
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.strip() for line in done.stdout.splitlines() if line.strip()}


def pipeline_audit(stages: tuple[Stage, ...] = STAGES,
                   repo_root: Path = REPO_ROOT,
                   sources: tuple[Source, ...] = SOURCES) -> pd.DataFrame:
    """What the pipeline can see about the repository it is about to rebuild."""
    rows = []
    add = lambda *args: rows.append(_a(*args))
    tracked = tracked_files(repo_root)

    # -- row-level output stays out of git ---------------------------------
    data_writes = sorted({w for s in stages for w in s.writes
                          if w.startswith("data/")})
    ignored = git_ignored(data_writes, repo_root)
    for path in data_writes:
        add("privacy", "row_level_ignored", path,
            "pass" if path in ignored else "fail",
            "output under data/ is git-ignored by the Task 01 rule")

    # -- committed outputs are actually committed --------------------------
    for stage in stages:
        for path in stage.writes:
            if path.startswith("data/") or path.endswith("/"):
                continue
            add("outputs", "committed_output_tracked", path,
                "pass" if path in tracked else "fail",
                f"{stage.name} writes a deliverable git carries")

    # -- a source-volatile stage must not own a committed file -------------
    for stage in stages:
        if stage.volatility != "source_volatile":
            continue
        committed = [w for w in stage.writes if w in tracked]
        add("volatility", "source_volatile_not_committed", stage.name,
            "fail" if committed else "pass",
            ("commits " + ", ".join(committed) +
             " — every run rewrites it from a rolling feed"
             if committed else "writes nothing git carries"))

    # -- the inputs a fresh clone would not have ---------------------------
    for stage in stages:
        for read in stage.reads:
            if read.startswith("http"):
                continue
            bare = read.rstrip("/")
            if any(_covers(w, read) for s in stages for w in s.writes):
                continue
            add("inputs", "input_present", f"{stage.name}: {read}",
                "pass" if (repo_root / bare).exists() else "fail",
                "an input no stage produces is present in the working tree")

    # -- what the schedule can refresh -------------------------------------
    verdict = refresh_verdict(sources)
    add("sources", "refresh_possible", f"{verdict['sources_checked']} approved",
        "fail" if not verdict["refreshable"] else "pass",
        verdict["reason"])
    for source in sources:
        add("sources", "source_verdict", source.name,
            "pass" if source.verdict == "refreshable" else "fail",
            f"{source.verdict}: {source.blocker}")

    # -- credentials are never committed -----------------------------------
    add("secrets", "env_untracked", ".env",
        "fail" if ".env" in tracked else "pass",
        "credentials are never committed; the workflow reads repository secrets")

    # Whether a local `.env` exists is deliberately *not* a row here. This
    # table is committed and a CI rebuild must reproduce it byte for byte, so a
    # check whose answer depends on the machine would make the committed
    # version permanently wrong somewhere. Credential availability goes to the
    # run report instead — see `credential_state`.
    return pd.DataFrame(rows, columns=list(AUDIT_COLUMNS))


def credential_state(repo_root: Path = REPO_ROOT) -> dict:
    """What this machine can authenticate to, for the report but not the table.

    Reports presence, never a value. Task 01's rule is that credentials live in
    a git-ignored `.env` and nothing else; a check that printed a key to a run
    log would break that rule while appearing to enforce it.
    """
    import os
    env_file = repo_root / ".env"
    return {
        "env_file_present": env_file.exists(),
        "adzuna_credentials": bool(os.environ.get("ADZUNA_APP_ID")
                                   and os.environ.get("ADZUNA_APP_KEY")),
        "note": "presence only; no value is ever read into a report or a log",
    }


# ---------------------------------------------------------------------------
# G. Drift — a generated number against the repository it counted
# ---------------------------------------------------------------------------
#
# Task 10 resolves a register of facts about the repository — task count,
# committed tables, committed figures, suite size, corrections — and renders
# them onto slides. Within a build it iterates to a fixpoint. Across tasks it
# cannot: the next task adds tables, figures, tests and a correction, and every
# one of those numbers moves while the submitted deck keeps the old value.
#
# That is checkable exactly, without regexes over prose, because both sides are
# machine-generated: the committed `deck-facts.csv` against the register
# resolved now.


#: The commit that submitted Task 10 — the last one to touch `deck-facts.csv`
#: before this task existed. Its message is "Rebuild the deck against the
#: repository as it now stands", which is the same repair Task 11 automates,
#: applied once by hand and then not repeatable. Pinning the sha is what makes
#: the `submitted` column below permanent evidence rather than a value that
#: moves every time the deck is rebuilt.
TASK10_SUBMISSION = "24cb4dc"

DRIFT_COLUMNS = ("key", "submitted", "committed", "live", "moved",
                 "moved_since_submission", "source")


def facts_at(ref: str = TASK10_SUBMISSION,
             path: str = "members/ankit-google/task-10-tables/deck-facts.csv",
             repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """The deck's fact register as it stood at a git ref.

    Returns an empty mapping when git cannot answer — a shallow CI checkout,
    or a clone that does not carry the commit. The drift table then reports
    the columns it can and says the rest is unavailable, rather than failing a
    run over the absence of history it does not need.
    """
    try:
        done = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=repo_root,
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {}
    if done.returncode != 0 or not done.stdout.strip():
        return {}
    rows = [line.split(",", 1) for line in done.stdout.splitlines()[1:] if line]
    return {key: value.rsplit(",", 2)[0].strip().strip('"')
            for key, value in rows if key}


def fact_drift(repo_root: Path = REPO_ROOT,
               member_root: Path = DEFAULT_MEMBER,
               focus: str = "google") -> pd.DataFrame:
    """Every Task 10 fact, as committed against as it resolves today.

    Three columns, three questions. `submitted` is the value at the commit that
    shipped Task 10; `committed` is the value in git now; `live` is what the
    repository resolves to today. `moved` compares the last two — is the deck
    in git stale? — and `moved_since_submission` compares the first and last,
    which is anchored to a commit and so does not settle when the deck is
    rebuilt.

    Called *after* the current run has written its own tables and figures,
    because several of these facts count exactly those files. Measuring before
    would answer a different question on a first run than on a second, and an
    audit whose answer depends on how many times it has been run is not an
    audit. Task 11's own artefacts moving the deck's counts is not a flaw in
    the measurement; it is the thing being measured.

    Imported lazily: `present` shells out to pytest to count the suite, and a
    module-level import would make every test that touches `pipeline` pay for
    it.
    """
    import present as pr

    committed_path = member_root / "task-10-tables" / "deck-facts.csv"
    if not committed_path.exists():
        return pd.DataFrame(columns=list(DRIFT_COLUMNS))

    # From git, not from the working tree. The column is called `committed` and
    # it has to mean that: reading the file on disk made an *uncommitted*
    # rebuild look committed, which is the one state this table exists to
    # catch. Falling back to the file keeps a shallow clone working, and says
    # so rather than quietly answering a different question.
    committed = facts_at(ref="HEAD", repo_root=repo_root)
    basis = "HEAD"
    if not committed:
        committed = dict(pd.read_csv(committed_path, dtype=str)
                         .set_index("key").value.astype(str))
        basis = "working tree (git could not read HEAD)"

    live = pr.resolve_facts(repo_root, member_root, focus)
    submitted = facts_at(repo_root=repo_root)

    rows = []
    for key, entry in live.items():
        was = str(committed.get(key, ""))
        then = submitted.get(key, "")
        now = entry["text"]
        rows.append({"key": key,
                     "submitted": then,
                     "committed": was,
                     "live": now,
                     "moved": was != now,
                     "moved_since_submission": bool(then) and then != now,
                     "source": entry["source"]})
    out = pd.DataFrame(rows, columns=list(DRIFT_COLUMNS))
    out.attrs["committed_basis"] = basis
    return out


def drift_summary(drift: pd.DataFrame) -> dict:
    """`moved` is the live question; `since_submission` is the historical one.

    Once the pipeline rebuilds the deck, `moved` goes to zero and stays there —
    that is the mechanism working. `moved_since_submission` does not, because
    it is measured against a pinned commit, so what the mentor saw stays
    readable after every future rebuild.
    """
    moved = drift[drift.moved] if len(drift) else drift
    since = drift[drift.moved_since_submission] if len(drift) else drift
    return {
        "facts": int(len(drift)),
        "moved": int(len(moved)),
        "keys": list(moved.key) if len(moved) else [],
        "stale": bool(len(moved)),
        "submission_ref": TASK10_SUBMISSION,
        "moved_since_submission": int(len(since)),
        "keys_since_submission": list(since.key) if len(since) else [],
    }


# ---------------------------------------------------------------------------
# H. The interpreter floor — syntax a CI runner may not have
# ---------------------------------------------------------------------------
#
# The pipeline's first CI run failed before it linted anything: `insights.py`
# would not parse. Four lines written during Task 09 reuse the enclosing
# quote inside an f-string replacement field, which PEP 701 legalised in
# Python 3.12 and which is a SyntaxError on 3.11.
#
# Nothing in the repository was wrong on the machine it was written on, and
# nothing ever will be — that is the whole difficulty. A version floor is
# invisible to every run that happens to satisfy it, so it is not discovered
# by running the code. It is discovered by running the code somewhere else,
# and until this task there was nowhere else.
#
# The scan below never imports, so it answers on the interpreter that is too
# new as well as the one that is too old: a developer on 3.12 is told that
# four lines will not run on 3.11 *before* pushing, instead of after.

#: The oldest interpreter the workflows install, and therefore the oldest one
#: this repository claims to run on. `.github/workflows/pipeline.yml` pins the
#: same string in both jobs; `test_ci_installs_the_floor_this_module_declares`
#: keeps the two from drifting apart.
PYTHON_FLOOR = (3, 11)


def _quote_of(tok: str) -> str | None:
    for ch in tok:
        if ch in "'\"":
            return ch
    return None


def interpreter_floor_scan(repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    """Sources that need a newer interpreter than `PYTHON_FLOOR`.

    Columns: `path`, `line`, `construct`. Never imports, so a file that cannot
    run on this interpreter is still readable.

    **The check has two halves, and which one runs depends on the interpreter
    you are standing on.** That is not an implementation detail to hide; it is
    the shape of the problem. CI found this out the hard way: the first version
    of this function used `token.FSTRING_START`, which 3.12 added along with
    the syntax it detects, so the check written to catch a version-dependent
    defect was itself version-dependent and died on the floor it was policing.

    - **Above the floor** (3.12+, where the syntax is legal): tokenise, and
      report every replacement field that reuses a quote still open around it.
      Nothing here fails, so a structural scan is the only way to see it.
    - **At or below the floor** (< 3.12, where the syntax is a `SyntaxError`):
      `ast.parse` each file and report what will not parse. The interpreter is
      the check; the scan just collects its verdicts instead of dying on the
      first one.

    What it does not attempt: a runtime call to a 3.12-only library function
    sails past both halves. This covers the class that fails at *parse* time,
    because that is the one that takes the whole pipeline down with it rather
    than failing a single stage.
    """
    import ast as _ast
    import token as _token
    import tokenize as _tokenize

    fstring_start = getattr(_token, "FSTRING_START", None)
    fstring_end = getattr(_token, "FSTRING_END", None)
    rows = []

    for path in sorted(repo_root.rglob("*.py")):
        if ".git" in path.parts:
            continue
        source = path.read_bytes()

        if fstring_start is None:
            # Below the floor: parsing is the verdict.
            try:
                _ast.parse(source, str(path))
            except SyntaxError as exc:
                rows.append((path, exc.lineno or 0, exc.msg))
            continue

        try:
            with open(path, "rb") as handle:
                toks = list(_tokenize.tokenize(handle.readline))
        except (SyntaxError, _tokenize.TokenError) as exc:
            rows.append((path, getattr(exc, "lineno", 0) or 0,
                         "does not tokenise"))
            continue

        open_quotes: list[str | None] = []
        for tok in toks:
            if tok.type == fstring_start:
                quote = _quote_of(tok.string)
                if quote in open_quotes:
                    rows.append((path, tok.start[0], "nested f-string"))
                open_quotes.append(quote)
            elif tok.type == fstring_end:
                if open_quotes:
                    open_quotes.pop()
            elif tok.type == _token.STRING and open_quotes:
                if _quote_of(tok.string) in open_quotes:
                    rows.append((path, tok.start[0], "nested string"))

    return pd.DataFrame(
        [(str(p.relative_to(repo_root)), line, kind) for p, line, kind in rows],
        columns=["path", "line", "construct"],
    )


def ci_python_versions(repo_root: Path = REPO_ROOT) -> list[str]:
    """Every `python-version:` the pipeline workflow pins, in file order."""
    workflow = repo_root / ".github" / "workflows" / "pipeline.yml"
    if not workflow.is_file():
        return []
    return re.findall(r'python-version:\s*"([^"]+)"', workflow.read_text())


# ---------------------------------------------------------------------------
# I. The run ledger
# ---------------------------------------------------------------------------

RUN_STATUSES = ("ok", "failed", "blocked", "skipped")

LEDGER_COLUMNS = ("run_order", "stage", "task", "kind", "status", "returncode",
                  "seconds", "artefacts_changed", "artefacts_volatile", "note")


@dataclass
class RunRecord:
    """What one stage did, in the form the ledger publishes."""

    stage: str
    status: str
    returncode: int = 0
    seconds: float = 0.0
    artefacts_changed: int = 0
    artefacts_volatile: int = 0
    note: str = ""
    stdout_tail: str = field(default="", repr=False)

    def __post_init__(self):
        if self.status not in RUN_STATUSES:
            raise ValueError(f"unknown run status {self.status!r}")


def run_ledger(records: list[RunRecord]) -> pd.DataFrame:
    """The ledger: one row per stage, in the order the run took them."""
    rows = []
    for i, record in enumerate(records):
        stage = STAGES_BY_NAME.get(record.stage)
        rows.append({
            "run_order": i,
            "stage": record.stage,
            "task": stage.task if stage else -1,
            "kind": stage.kind if stage else "",
            "status": record.status,
            "returncode": record.returncode,
            "seconds": round(record.seconds, 2),
            "artefacts_changed": record.artefacts_changed,
            "artefacts_volatile": record.artefacts_volatile,
            "note": record.note,
        })
    return pd.DataFrame(rows, columns=list(LEDGER_COLUMNS))


def run_verdict(records: list[RunRecord], audit: pd.DataFrame) -> dict:
    """The single line a scheduled run reports back.

    A run is `clean` when every stage succeeded and no artefact's *content*
    moved. `drifted` means content moved with no new data, which on a frozen
    source means the code or its dependencies changed — the only thing this
    schedule can actually discover.
    """
    failed = [r.stage for r in records if r.status == "failed"]
    blocked = [r.stage for r in records if r.status == "blocked"]
    changed = list(audit[audit.verdict == "changed"].path) if len(audit) else []
    volatile = list(audit[audit.verdict == "volatile_only"].path) if len(audit) else []
    if failed:
        state = "failed"
    elif changed:
        state = "drifted"
    else:
        state = "clean"
    return {
        "state": state,
        "stages_run": len(records),
        "failed": failed,
        "blocked": blocked,
        "artefacts_changed": len(changed),
        "artefacts_volatile_only": len(volatile),
        "changed_paths": changed[:20],
    }


FAILURE_MODES = (
    ("network_unreachable", "collect",
     "The Hugging Face parquet cannot be downloaded.",
     "The stage fails; everything downstream is recorded `blocked`, not "
     "`failed`, so the ledger names one defect rather than ten."),
    ("cache_absent", "competitor-set",
     "A fresh clone has no `data/`, and the competitor set reads the 75 MB "
     "full cache the collector leaves behind.",
     "`requires` carries the edge, so the plan runs `collect` first. This is "
     "the edge that was undeclared before Task 11."),
    ("credentials_absent", "collect",
     "No Adzuna keys in the environment.",
     "Backfill-only, which is the committed dataset anyway. Recorded as a "
     "source verdict, not a failure."),
    ("dependency_drift", "any",
     "A pandas or numpy release changes a default under unchanged code.",
     "The weekly verify run reports `drifted` with the artefact list. This is "
     "the only thing the schedule can find on a frozen source, and it is why "
     "the schedule exists."),
    ("clock_noise", "any stage writing JSON",
     "Every report carries the wall clock, so bytes always move.",
     "Stable digests strip the registered clock keys, so the run reports "
     "`volatile_only` instead of committing ten meaningless lines."),
    ("rolling_feed", "validate-text",
     "A live public feed returns different postings every call.",
     "Declared `source_volatile` and kept out of the schedule; the audit "
     "reports that its output is nonetheless committed."),
    ("fixpoint_stale", "presentation",
     "The deck counts a repository that later tasks keep changing.",
     "`fact_drift` compares the committed register against the live one, so "
     "the stale key can be named rather than merely detected."),
)


def failure_mode_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"mode": m, "stage": s, "what_happens": w, "what_the_pipeline_does": d}
         for m, s, w, d in FAILURE_MODES])
