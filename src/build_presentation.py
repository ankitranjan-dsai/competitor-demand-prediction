"""Task 10 — build the final deck, the question bank, and the workspace audit.

Runs the shared layer in `src/present.py` over the Task 09 claim ledger and
over the repository itself, and writes:

    members/ankit-google/task-10-tables/        (committed — aggregate only)
        deck-sections.csv        the team's slot order, and how this deck fills it
        deck-slides.csv          one row per slide: bullets, figure, caveat
        deck-bullets.csv         one row per bullet, and where its authority comes from
        deck-lint.csv            every rule the deck was checked against, and its result
        mentor-qa.csv            the question bank, one row per answer line
        qa-shape.csv             which questions end in a refusal, and which do not
        qa-lint.csv              the bank, checked by the same rules as the deck
        workspace-audit.csv      is the repository the one the README says it is
        deck-facts.csv           every number on a slide, recomputed from the repo
        suite-growth.csv         the suite size quoted at the end of each task
        refusal-coverage.csv     each prohibition, and the refusal that answers it

    members/ankit-google/task-10-figures/       (committed)
        01-deck-provenance.png
        02-answer-shape.png
        03-workspace-audit.png
        04-ledger-coverage.png
        05-slide-density.png
        06-refusal-coverage.png
        07-suite-growth.png
        08-section-order.png

    members/ankit-google/task-10-slides.md               the deck itself
    members/ankit-google/task-10-presentation-report.json  quality evidence

The build **fails** if the deck or the bank does not lint. That is the whole
point of the task: Task 09's Google report §13 predicted this deliverable
would be the first not checkable by a test, and the correction turns on the deck
being rebuildable rather than hand-maintained. A runner that wrote a deck with
known violations and reported them as a summary line would concede the
prediction while appearing to refute it.

The workspace audit does *not* fail the build. It is a report on the
repository, and the repository is exactly what this task is allowed to change;
failing on it would make the runner unusable at the moment it is most needed.
Its failures travel in the JSON report and in the member report instead.

    python src/build_presentation.py
    python src/build_presentation.py --focus google
"""

from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display in CI or on a headless run
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

import insights as ins            # noqa: E402
import present as pr              # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "members" / "ankit-google"
TABLES = OUT / "task-10-tables"
FIGURES = OUT / "task-10-figures"
TASK09 = OUT / "task-09-tables"

# Task 05's palette, unchanged through Tasks 06 to 09.
INK = "#1f2933"
GREY = "#9aa5b1"
BLUE = "#2f6fb5"
ORANGE = "#e08a1e"
RED = "#c0392b"
GREEN = "#2e7d5b"
PURPLE = "#7d5ba6"

#: Bullet colours. A refusal is grey, matching Task 09's status palette, for
#: the same reason: it is an output of this project, not a failure of it.
KIND_COLOURS = {
    "claim": GREEN,
    "fact": BLUE,
    "plain": "#cbd2d9",
    "refusal": GREY,
}


def _write(df: pd.DataFrame, path: Path) -> None:
    """Write a table, refusing the two column families Task 06 has banned."""
    bad = ins.forbidden_columns(df)
    if bad:
        raise ValueError(f"{path.name} carries forbidden columns {bad}")
    personal = ins.personal_data_columns_present(df)
    if personal:
        raise ValueError(f"{path.name} carries personal-data columns {personal}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _style(ax, title: str, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=10)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.grid(axis="y", color=GREY, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_provenance(bullets: pd.DataFrame, path: Path) -> str:
    """What each slide is made of, slide by slide.

    Stacked counts rather than proportions, because the question this figure
    answers is "does any slide run on narration alone", and a proportion of
    three bullets hides how few bullets there were.
    """
    order = ("claim", "refusal", "fact", "plain")
    pivot = (bullets.pivot_table(index="slide", columns="kind", values="bullet",
                                 aggfunc="count")
             .reindex(columns=order).fillna(0))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bottom = np.zeros(len(pivot))
    for kind in order:
        ax.bar(pivot.index, pivot[kind], bottom=bottom,
               color=KIND_COLOURS[kind], label=kind, width=0.72)
        bottom += pivot[kind].to_numpy()
    ax.set_xticks(list(pivot.index))
    _style(ax, "Every bullet, and the binding that justifies it",
           "bullets", "slide")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.set_ylim(0, bottom.max() + 1.8)
    return _save(fig, path)


def fig_answer_shape(shape: pd.DataFrame, path: Path) -> str:
    """How the question bank answers, question by question.

    The interesting number is the refusal count, so it is the one plotted:
    a mentor review where most questions resolve to a published claim would
    mean the questions were soft, the gates were, or both.
    """
    block = shape.copy()
    colour = {"refused": GREY, "answered": GREEN, "explained": BLUE}
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    y = np.arange(len(block))
    ax.barh(y, [1] * len(block),
            color=[colour[v] for v in block.verdict],
            alpha=0.85, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.fill(q, 44) for q in block.question],
                       fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylim(len(block) + 0.2, -0.8)
    ax.set_xticks([])
    for i, row in enumerate(block.itertuples()):
        ax.text(0.015, i, f"{row.question_id}  ·  slide {row.backs_slide}",
                va="center", fontsize=7.5, color="white", weight="bold")
    counts = block.verdict.value_counts()
    _style(ax, f"The question bank: {counts.get('refused', 0)} of "
               f"{len(block)} likely questions open with a refusal", "", "")
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[k]) for k in colour]
    ax.legend(handles, list(colour), frameon=False, fontsize=8, ncol=3,
              loc="lower right", bbox_to_anchor=(1.0, -0.06))
    return _save(fig, path)


def fig_workspace(audit: pd.DataFrame, path: Path) -> str:
    """The workspace audit, by area, pass against fail."""
    grouped = (audit.groupby(["area", "status"]).size().unstack(fill_value=0)
               .reindex(columns=["pass", "fail"], fill_value=0))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    x = np.arange(len(grouped))
    ax.bar(x - 0.19, grouped["pass"], width=0.38, color=GREEN, label="pass")
    ax.bar(x + 0.19, grouped["fail"], width=0.38, color=RED, label="fail")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    for i, row in enumerate(grouped.itertuples()):
        if row.fail:
            ax.text(i + 0.19, row.fail + 0.4, str(row.fail), ha="center",
                    fontsize=8, color=RED, weight="bold")
    _style(ax, "Workspace audit — what the README promises against what is here",
           "checks", "")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    return _save(fig, path)


def fig_ledger_coverage(ledger: pd.DataFrame, bullets: pd.DataFrame,
                        qa: pd.DataFrame, path: Path) -> str:
    """How much of the evidence base the deck actually spends.

    A deck is a selection, and the honest thing to show is the size of what
    was left out. Most published sentences never reach a slide; that is not a
    gap, it is what a twenty-minute talk means, and the ledger is where the
    rest stays available.
    """
    used = set(bullets.claim_id) | set(qa.claim_id)
    used.discard("")
    counts = []
    for status in (ins.PUBLISHED, ins.QUALIFIED, ins.REFUSED):
        rows = ledger[ledger.status == status]
        counts.append((status, len(rows),
                       len(set(rows.claim_id) & used)))

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    y = np.arange(len(counts))
    totals = [c[1] for c in counts]
    spent = [c[2] for c in counts]
    ax.barh(y, totals, color=GREY, alpha=0.3, height=0.6, label="in the ledger")
    ax.barh(y, spent, color=BLUE, height=0.6, label="on a slide or in the bank")
    ax.set_yticks(y)
    ax.set_yticklabels([c[0].replace("_", " ") for c in counts])
    ax.invert_yaxis()
    for i, (_, total, used_n) in enumerate(counts):
        ax.text(total + 4, i, f"{used_n} of {total}", va="center", fontsize=8,
                color=INK)
    ax.set_xlim(0, max(totals) * 1.22)
    _style(ax, "The deck spends a fraction of the ledger, and says so",
           "", "claims")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    return _save(fig, path)


def fig_density(bullets: pd.DataFrame, path: Path) -> str:
    """Characters per bullet against the spoken limit.

    The bars over the line are not violations. They are claim sentences,
    reproduced verbatim because shortening them would drop a clause, and the
    deck marks them read-from-slide rather than rewriting them.
    """
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(bullets))
    ax.bar(x, bullets.characters,
           color=[KIND_COLOURS[k] for k in bullets.kind], width=0.85)
    ax.axhline(pr.SPOKEN_MAX, color=RED, linewidth=1.1, linestyle="--")
    ax.text(1, pr.SPOKEN_MAX + 8, f"spoken limit — {pr.SPOKEN_MAX} characters",
            fontsize=8, color=RED)
    over = bullets[bullets.characters > pr.SPOKEN_MAX]
    for row in over.itertuples():
        ax.text(row.Index, row.characters + 8, f"slide {row.slide}",
                fontsize=7.5, color=INK, ha="center")
    ax.set_xticks([])
    _style(ax, "Bullet length, in slide order", "characters", "bullets")
    handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLOURS[k])
               for k in ("claim", "refusal", "fact", "plain")]
    ax.legend(handles, ["claim", "refusal", "fact", "narration"],
              frameon=False, fontsize=8, ncol=4, loc="upper right")
    return _save(fig, path)


def refusal_coverage(patterns: pd.DataFrame, ledger: pd.DataFrame,
                     used: set) -> pd.DataFrame:
    """Which prohibitions have a prepared answer, matched on the ledger.

    Matched on the refused row's own `blocked_by`/`reason`, not on a word
    search of the answer prose. A substring match would report a prohibition
    as covered because the presenter happened to use the word, which is the
    failure this table is supposed to detect.
    """
    stopped = ledger[(ledger.claim_id.isin(used))
                     & (ledger.status == ins.REFUSED)
                     & (ledger.blocked_by == "lint")]
    banked = {str(r).split(" (")[0].strip() for r in stopped.reason}
    rows = []
    for row in patterns.itertuples():
        answers = sorted(stopped[stopped.reason.astype(str)
                                 .str.startswith(row.rule)].claim_id)
        rows.append({
            "rule": row.rule,
            "source": row.source,
            "why": row.why,
            "banked": row.rule in banked,
            "answered_by": ", ".join(answers),
        })
    return pd.DataFrame(rows)


def fig_refusal_coverage(coverage: pd.DataFrame, path: Path) -> str:
    """Does the deck have an answer ready for every prohibition?

    Each prohibited construction is a question a mentor can ask in good faith.
    A pattern with no banked answer is the one the presenter improvises, so
    the figure is drawn to make an empty row loud rather than tidy.
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    y = np.arange(len(coverage))
    ax.barh(y, [1] * len(coverage),
            color=[GREEN if b else RED for b in coverage.banked], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([r.replace("_", " ") for r in coverage.rule], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    for i, row in enumerate(coverage.itertuples()):
        label = (f"{row.source}  ·  {row.answered_by}" if row.banked
                 else f"{row.source}  ·  no prepared answer")
        ax.text(0.012, i, label, va="center", fontsize=7.5, color="white",
                weight="bold")
    _style(ax, f"{int(coverage.banked.sum())} of {len(coverage)} prohibitions "
               f"are answered by a quoted refusal, not by improvisation", "", "")
    ax.grid(False)
    return _save(fig, path)


def fig_suite_growth(growth: pd.DataFrame, path: Path) -> str:
    """The suite size quoted at the end of each task, and the size today.

    The last bar is the check: a README quotes a number that was true when it
    was typed, and the only one still checkable is the most recent.
    """
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    block = growth.dropna(subset=["tests"])
    ax.bar(block.task, block.tests, color=BLUE, width=0.62)
    current = pr.count_tests(REPO_ROOT)
    ax.axhline(current, color=ORANGE, linewidth=1.2, linestyle="--")
    ax.text(block.task.iloc[0] - 0.4, current + 10,
            f"the suite today — {current} tests", fontsize=8, color=ORANGE)
    for row in block.itertuples():
        ax.text(row.task, row.tests + 10, str(int(row.tests)), ha="center",
                fontsize=7.5, color=INK)
    ax.set_xticks(list(block.task))
    ax.set_ylim(0, current * 1.18)
    _style(ax, "Tests in the suite at the end of each task", "tests", "task")
    return _save(fig, path)


def fig_sections(sections: pd.DataFrame, path: Path) -> str:
    """The team standard's slot order, and which slots may not be dropped."""
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    y = np.arange(len(sections))
    widths = [len(s.split(", ")) if s else 0 for s in sections.slides]
    ax.barh(y, widths,
            color=[ORANGE if r else BLUE for r in sections.required],
            height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(sections.section, fontsize=8)
    ax.invert_yaxis()
    for i, row in enumerate(sections.itertuples()):
        ax.text(0.05, i, row.purpose, va="center", fontsize=7.5,
                color="white" if widths[i] else INK, weight="bold")
    ax.set_xlim(0, max(widths) + 0.2)
    ax.set_xticks(range(max(widths) + 1))
    _style(ax, "The aligned storyline — orange slots are mandatory in every "
               "member's deck", "", "slides")
    ax.grid(False)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def suite_growth(repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    """The test count each task's README row quotes, as a table.

    Historical rows are snapshots and are recorded as such; only the newest is
    a claim about the repository as it stands, and that one the audit checks.
    """
    rows = []
    task_rows = pr.readme_task_rows(repo_root)
    done = [n for n, cell in task_rows if pr.DONE in cell]
    newest = max(done) if done else None
    for number, cell in task_rows:
        quoted = pr.QUOTED_TESTS.findall(cell)
        rows.append({
            "task": number,
            "tests": int(quoted[0].replace(",", "")) if quoted else None,
            "quoted_in": "README.md",
            "kind": "current" if number == newest else "snapshot",
        })
    return pd.DataFrame(rows)


#: The deck reports on the repository, and the deck's own tables and figures
#: are part of that repository. The first pass therefore writes files that the
#: fact register would have counted had they already existed, so the build runs
#: to a fixpoint: resolve the facts, emit, re-resolve, and emit again only if a
#: number moved. Three passes is one more than convergence needs — write,
#: recount, confirm — because an emission that adds no new file cannot move a
#: count. A build still moving after that is reporting on something unsettled,
#: which is worth raising rather than quietly emitting the last state reached.
MAX_PASSES = 3


def _emit(deck: list[pr.Slide], bank: list[pr.Question], ledger: pd.DataFrame,
          patterns: pd.DataFrame, facts: dict, focus: str) -> dict:
    """Write every artefact once, against one resolution of the fact register."""
    audit = pr.workspace_audit(REPO_ROOT, OUT)
    sections = pr.section_table(deck)
    slides = pr.slide_table(deck, ledger, facts)
    bullets = pr.bullet_table(deck, ledger, facts)
    qa = pr.qa_table(bank, ledger, facts)
    shape = pr.answer_shape(bank)
    growth = suite_growth(REPO_ROOT)
    quoted = set(bullets.claim_id) | set(qa.claim_id)
    quoted.discard("")
    coverage = refusal_coverage(patterns, ledger, quoted)

    # The ordering here matters and is not cosmetic. The tables are written
    # *before* the deck is linted, because one of the linter's rules is that
    # every figure and table a slide points at resolves to a committed file —
    # and the backup slide points at the question bank this build produces.
    # Linting first would fail on the deck's own output and make the rule
    # untestable, which is the opposite of what it is for.
    written = []
    for name, table in (
        ("deck-sections.csv", sections),
        ("deck-slides.csv", slides),
        ("deck-bullets.csv", bullets),
        ("mentor-qa.csv", qa),
        ("qa-shape.csv", shape),
        ("workspace-audit.csv", audit),
        ("deck-facts.csv", pr.fact_table(facts)),
        ("suite-growth.csv", growth),
        ("refusal-coverage.csv", coverage),
    ):
        _write(table, TABLES / name)
        written.append(name)

    deck_lint = pr.lint_deck(deck, ledger, facts, REPO_ROOT, OUT)
    bank_lint = pr.lint_qa_bank(bank, deck, ledger, facts, OUT)
    # The lint tables are written whatever the verdict, because a reader of a
    # failed build needs to see what failed; the raise comes after the write.
    for name, table in (("deck-lint.csv", deck_lint),
                        ("qa-lint.csv", bank_lint)):
        _write(table, TABLES / name)
        written.append(name)

    if len(deck_lint) or len(bank_lint):
        raise SystemExit(
            f"deck does not lint: {len(deck_lint)} slide violation(s), "
            f"{len(bank_lint)} question violation(s) — see "
            f"{(TABLES / 'deck-lint.csv').relative_to(REPO_ROOT)}"
        )

    # -- the deck ----------------------------------------------------------
    slides_path = OUT / "task-10-slides.md"
    slides_path.write_text(pr.deck_markdown(deck, bank, ledger, facts, focus),
                           encoding="utf-8")

    # The audit reads the markdown it finds on disk, and the deck is markdown
    # on disk — so the run above graded the *previous* build's deck. A link
    # broken in `deck_markdown` would ship once, under an audit line reading
    # "all checks pass", and be reported only on the next build. It happened:
    # the deck went out quoting a filename for the standard that has never
    # existed. So the audit is re-run over the deck this build just wrote, and
    # the table is rewritten before the figure draws it. The early run stays,
    # because `asset_exists` needs the table on disk before the deck lints.
    audit = pr.workspace_audit(REPO_ROOT, OUT)
    _write(audit, TABLES / "workspace-audit.csv")

    figs = [
        fig_provenance(bullets, FIGURES / "01-deck-provenance.png"),
        fig_answer_shape(shape, FIGURES / "02-answer-shape.png"),
        fig_workspace(audit, FIGURES / "03-workspace-audit.png"),
        fig_ledger_coverage(ledger, bullets, qa,
                            FIGURES / "04-ledger-coverage.png"),
        fig_density(bullets, FIGURES / "05-slide-density.png"),
        fig_refusal_coverage(coverage, FIGURES / "06-refusal-coverage.png"),
        fig_suite_growth(growth, FIGURES / "07-suite-growth.png"),
        fig_sections(sections, FIGURES / "08-section-order.png"),
    ]

    failures = audit[audit.status == "fail"]
    report = {
        "task": 10,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "focus": focus,
        "deck": {
            "slides": len(deck),
            "sections": len(sections),
            "required_sections": list(pr.REQUIRED_SECTIONS),
            "bullets": len(bullets),
            "by_kind": bullets.kind.value_counts().to_dict(),
            "read_from_slide": sorted(pr.long_bullets(deck, ledger, facts)),
            "carries": sorted({c for slide in deck for c in slide.carries}),
            "violations": len(deck_lint),
        },
        "bank": {
            "questions": len(bank),
            "answer_lines": len(qa),
            "verdicts": shape.verdict.value_counts().to_dict(),
            "violations": len(bank_lint),
            "prohibitions_answered": int(coverage.banked.sum()),
            "prohibitions_open": sorted(coverage[~coverage.banked].rule),
        },
        "ledger": {
            "rows": len(ledger),
            "spent_on_deck": int(bullets.claim_id.astype(bool).sum()),
            "spent_in_bank": int(qa.claim_id.astype(bool).sum()),
        },
        "workspace": {
            "checks": len(audit),
            "failing": len(failures),
            "failures": failures[["check", "subject", "detail"]]
                        .to_dict("records"),
        },
        "facts": {key: value["value"] for key, value in facts.items()},
        "figures": figs,
        "privacy": {
            "personal_data_columns_present": [],
            "row_level_written": False,
        },
    }
    report_path = OUT / "task-10-presentation-report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    return {
        "written": written,
        "figures": figs,
        "slides_path": slides_path,
        "report_path": report_path,
        "audit": audit,
        "shape": shape,
        "bullets": bullets,
    }


def build(focus: str = "google") -> None:
    ledger = pd.read_csv(TASK09 / "claim-ledger.csv")
    patterns = pd.read_csv(TASK09 / "prohibited-patterns.csv")
    deck = pr.deck_spec(focus)
    bank = pr.qa_bank(focus)

    emitted: dict | None = None    # the fact values the last emission used
    passes = 0
    for _ in range(MAX_PASSES):
        facts = pr.resolve_facts(REPO_ROOT, OUT, focus)
        values = {key: fact["value"] for key, fact in facts.items()}
        if values == emitted:
            break
        out = _emit(deck, bank, ledger, patterns, facts, focus)
        emitted, passes = values, passes + 1
    else:
        moved = [key for key, value in values.items() if emitted[key] != value]
        raise SystemExit(
            f"the build did not settle in {MAX_PASSES} passes; still moving: "
            f"{', '.join(moved)}"
        )

    audit, shape = out["audit"], out["shape"]
    failures = audit[audit.status == "fail"]
    print(f"tables  -> {TABLES.relative_to(REPO_ROOT)} "
          f"({len(out['written'])} csv)")
    print(f"figures -> {FIGURES.relative_to(REPO_ROOT)} "
          f"({len(out['figures'])} png)")
    print(f"deck    -> {out['slides_path'].relative_to(REPO_ROOT)} "
          f"({len(deck)} slides, {len(out['bullets'])} bullets)")
    print(f"report  -> {out['report_path'].relative_to(REPO_ROOT)}")
    print(f"\ndeck lints clean; question bank lints clean"
          f" (settled in {passes} pass{'' if passes == 1 else 'es'})")
    print(f"questions banked: {len(bank)}, of which "
          f"{shape.verdict.value_counts().get('refused', 0)} open with a refusal")
    if len(failures):
        print(f"\nworkspace audit: {len(failures)} of {len(audit)} checks fail")
        for row in failures.itertuples():
            print(f"  - {row.check}: {row.subject} — {row.detail}")
    else:
        print(f"\nworkspace audit: all {len(audit)} checks pass")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus", default="google",
                        help="the specialist's own company; named in the deck")
    args = parser.parse_args()
    build(focus=args.focus)


if __name__ == "__main__":
    main()
