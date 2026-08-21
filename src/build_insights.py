"""Task 09 — build the claim ledger, the insight tables, figures and report.

Runs the shared layer in `src/insights.py` over every table Tasks 04 to 08
committed, and writes:

    members/ankit-google/task-09-tables/        (committed — aggregate only)
        salary-disclosure.csv          who discloses pay, and how selected it is
        salary-disclosure-by-publisher.csv   the missingness mechanism
        salary-feasible-cells.csv      which within-publisher pairs are testable
        salary-pairs.csv               the pairs, unstratified
        salary-pairs-stratified.csv    the same pairs inside job function
        publisher-cell-floor.csv       C8 — unanimity recounted under a floor
        unanimity-verdict.csv          C8 — whose `confirmed` survives it
        sign-test-power.csv            what a unanimous count can establish
        claim-ledger.csv               EVERY candidate, gate by gate
        insight-yield.csv              the funnel: 415 candidates in, 96 out
        evidence-bindings.csv          each quoted number against its cell
        lint-audit.csv                 each prohibited construction, per claim
        prohibited-patterns.csv        the rules, and which task earned them
        cross-task-conflicts.csv       quantities two tasks both quote
        refused-claims.csv             what was refused, and what would lift it
        question-coverage.csv          the brief's questions, answered or not
        brief-promises.csv             the brief's six promises, paid or not
        actionability.csv              per audience, what changes
        strategy-position.csv          the "position" question, three readings
        audience-brief-strategy.csv    the sentences one audience may be handed
        falsifiers.csv                 every published sentence, and its undoing
        claim-provenance.csv           which task each surviving sentence rests on

    members/ankit-google/task-09-figures/       (committed)
        01-insight-yield.png
        02-gate-attrition.png
        03-audience-reach.png
        04-publisher-cell-floor.png
        05-salary-missingness.png
        06-brief-promises.png
        07-claim-provenance.png
        08-refusal-ledger.png

    members/ankit-google/task-09-insight-report.json      quality evidence

The ordering inside `build` matters and is not cosmetic. The salary tables are
written *first*, because the salary audit is the only analysis Task 09 performs
itself and its claims cite committed CSVs like every other claim. Compiling
before writing them would silently drop the family — an empty denominator that
reads as a clean sheet.

Nothing row-level is written. The task consumes `data/processed/`, which stays
git-ignored, and emits only aggregates.

    python src/build_insights.py
    python src/build_insights.py --companies google,meta,nvidia
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display in CI or on a headless run
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

import companies as co            # noqa: E402
import compare as cmp             # noqa: E402
import insights as ins            # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "members" / "ankit-google"
TABLES = OUT / "task-09-tables"
FIGURES = OUT / "task-09-figures"
TASK06 = OUT / "task-06-tables"

# Task 05's palette, unchanged through Tasks 06, 07 and 08.
INK = "#1f2933"
GREY = "#9aa5b1"
BLUE = "#2f6fb5"
ORANGE = "#e08a1e"
RED = "#c0392b"
GREEN = "#2e7d5b"
PURPLE = "#7d5ba6"

COMPANY_COLOURS = {
    "google": BLUE, "microsoft": ORANGE, "meta": PURPLE,
    "nvidia": GREEN, "snowflake": RED, "databricks": "#4a5568",
}

#: Status colours. Refused is grey rather than red on purpose — a refusal is
#: an output of this task, not a failure of it.
STATUS_COLOURS = {
    ins.PUBLISHED: GREEN,
    ins.QUALIFIED: ORANGE,
    ins.REFUSED: GREY,
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


def _colour(company: str) -> str:
    return COMPANY_COLOURS.get(company, GREY)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_yield(yields: pd.DataFrame, path: Path) -> str:
    """The funnel, per question, as a proportion.

    Proportions rather than counts, and a linear axis rather than a log one.
    Both choices were forced. `tech_stack` proposes 310 claims and
    `future_demand` proposes 10, so a linear count axis renders four of the
    five questions as slivers — but a log axis makes 6 of 33 look like a
    nearly-full bar, which is the opposite of what this figure is for. The
    denominator survives as the `n =` label on the right.
    """
    block = yields[yields.question != "all"].copy()
    for col in ("published", "published_qualified", "refused"):
        block[col + "_pct"] = 100 * block[col] / block.generated

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    y = np.arange(len(block))
    ax.barh(y, block.published_pct, color=GREEN, label="verdict firm")
    ax.barh(y, block.published_qualified_pct, left=block.published_pct,
            color=ORANGE, label="verdict qualified")
    ax.barh(y, block.refused_pct,
            left=block.published_pct + block.published_qualified_pct,
            color=GREY, alpha=0.35, label="refused")
    ax.set_yticks(y)
    ax.set_yticklabels([q.replace("_", " ") for q in block.question])
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    for i, row in enumerate(block.itertuples()):
        ax.text(102, i, f"n = {row.generated}", va="center", fontsize=8,
                color=INK)
        if row.published + row.published_qualified:
            ax.text(1.5, i, f"{row.published + row.published_qualified}"
                            f"  ({row.yield_pct}%)",
                    va="center", fontsize=8, color="white", weight="bold")
    _style(ax, "Insight yield: of everything this evidence base can be asked "
               "to say, what it answers",
           xlabel="share of the claims proposed for that question (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GREY, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, -0.32))
    return _save(fig, path)


def fig_gate_attrition(ledger: pd.DataFrame, path: Path) -> str:
    """Where candidates die. Four bars, in gate order."""
    total = len(ledger)
    surviving = [
        total,
        int(ledger.gate_evidence.sum()),
        int((ledger.gate_evidence & ledger.gate_lint).sum()),
        int((ledger.gate_evidence & ledger.gate_lint
             & ledger.gate_identification).sum()),
        int((ledger.status != ins.REFUSED).sum()),
    ]
    labels = ["generated", "1 evidence", "2 lint", "3 identification",
              "4 consistency"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bars = ax.bar(labels, surviving, color=[GREY, BLUE, BLUE, RED, GREEN],
                  alpha=0.9)
    for bar, value, previous in zip(bars, surviving, [None] + surviving[:-1]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + total * 0.02,
                f"{value}" + ("" if previous is None
                              else f"\n−{previous - value}"),
                ha="center", fontsize=8, color=INK)
    refused = total - surviving[-1]
    ident = int((ledger.blocked_by == "identification").sum())
    _style(ax, f"Gate attrition: {ident} of the {refused} refusals are "
               f"identification, not wording", ylabel="claims surviving")
    ax.set_ylim(0, total * 1.18)
    return _save(fig, path)


def fig_actionability(actionable: pd.DataFrame, path: Path) -> str:
    """What each audience the brief names can actually be handed.

    This is the figure the brief's "Who benefits" list asks for and the one
    the answer is least comfortable in. Product managers get 81 sentences;
    investors get none, because every sentence an investor would want is a
    sentence about levels, trajectories or a forecast, and all three are
    refused upstream.
    """
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    y = np.arange(len(actionable))
    ax.barh(y, actionable.actionable, color=GREEN,
            label="changes a decision")
    ax.barh(y, actionable.informational_only, left=actionable.actionable,
            color=GREY, alpha=0.55, label="informational only")
    ax.set_yticks(y)
    ax.set_yticklabels([a.replace("_", " ") for a in actionable.audience])
    ax.invert_yaxis()
    ax.set_xlim(0, actionable.claims_available.max() * 1.75)
    for i, row in enumerate(actionable.itertuples()):
        if row.claims_available == 0:
            ax.text(1, i, "no publishable sentence — every question this "
                          "audience asks is about a level, a trajectory "
                          "or a forecast",
                    va="center", fontsize=8, color=RED)
            continue
        questions = (row.questions_covered if isinstance(
            row.questions_covered, str) else "")
        ax.text(row.claims_available + 1.5, i,
                f"{row.claims_available}   {questions}",
                va="center", fontsize=8, color=INK)
    _style(ax, "The audiences the brief names, and what each can be handed",
           xlabel="publishable claims")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GREY, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    return _save(fig, path)


def fig_cell_floor(recount: pd.DataFrame, path: Path) -> str:
    """C8. Two panels: the count that shrinks, and the p-value that follows."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for company, block in recount.groupby("company"):
        block = block.sort_values("cell_floor")
        style = dict(color=_colour(company), marker="o", markersize=4,
                     linewidth=2 if company == "nvidia" else 1.1,
                     alpha=1.0 if company == "nvidia" else 0.55)
        ax1.plot(block.cell_floor, block.publishers_tested,
                 label=company, **style)
        ax2.plot(block.cell_floor, block.sign_test_p, **style)
        confirmed = block[block.verdict == "confirmed"]
        ax1.scatter(confirmed.cell_floor, confirmed.publishers_tested,
                    facecolors="white", edgecolors=_colour(company),
                    s=70, zorder=5, linewidths=1.6)
    ax2.axhline(0.05, color=RED, linewidth=1, linestyle="--")
    ax2.text(0.15, 0.058, "0.05", fontsize=8, color=RED)
    _style(ax1, "C8 — publishers still tested as the cell floor rises",
           ylabel="publishers tested", xlabel="per-company cell floor "
                                              "(postings per half)")
    _style(ax2, "…and what unanimity is then worth",
           ylabel="two-sided sign-test p", xlabel="per-company cell floor")
    ax1.legend(fontsize=7, frameon=False, ncol=2)
    ax1.text(0.02, 0.03, "hollow marker = verdict 'confirmed'",
             transform=ax1.transAxes, fontsize=7, color=GREY)
    return _save(fig, path)


def fig_salary(by_publisher: pd.DataFrame, disclosure: pd.DataFrame,
               path: Path) -> str:
    """The missingness mechanism, next to what conditioning on it does."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    block = by_publisher[by_publisher.postings >= 20].head(12)
    ax1.barh(np.arange(len(block)), block.disclosed_pct, color=BLUE, alpha=0.8)
    ax1.set_yticks(np.arange(len(block)))
    ax1.set_yticklabels([p.replace("via ", "") for p in block.publisher],
                        fontsize=7)
    ax1.invert_yaxis()
    _style(ax1, "Salary disclosure is a publisher behaviour",
           xlabel="% of postings disclosing a salary")
    ax1.grid(axis="y", visible=False)
    ax1.grid(axis="x", color=GREY, alpha=0.25, linewidth=0.6)

    shift = disclosure.dropna(subset=["country_shift_pp"])
    colours = [_colour(c) for c in shift.company]
    ax2.bar(shift.company, shift.country_shift_pp, color=colours, alpha=0.9)
    ax2.axhline(0, color=GREY, linewidth=1)
    for i, row in enumerate(shift.itertuples()):
        ax2.text(i, row.country_shift_pp + (2 if row.country_shift_pp >= 0
                                            else -4),
                 f"{row.country_shift_pp:+.0f}", ha="center", fontsize=8,
                 color=INK)
    _style(ax2, "…so conditioning on it moves the country mix",
           ylabel="modal-country share, disclosed − all (pp)")
    ax2.tick_params(axis="x", labelrotation=30)
    return _save(fig, path)


def fig_promises(promises: pd.DataFrame, path: Path) -> str:
    """The brief's six promises, on the same proportional scale as the yield.

    The promise with no families is drawn full width in grey rather than as a
    zero-length bar: nothing was proposed because nothing in this repository
    measures the construct, which is a different fact from nothing surviving.
    """
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(promises))
    for i, row in enumerate(promises.itertuples()):
        if row.generated == 0:
            ax.barh(i, 100, color=GREY, alpha=0.25, hatch="///",
                    edgecolor=GREY)
            ax.text(50, i, "nothing in this repository measures it",
                    va="center", ha="center", fontsize=8, color=INK)
            continue
        firm = 100 * row.published / row.generated
        qual = 100 * row.published_qualified / row.generated
        ax.barh(i, firm, color=GREEN)
        ax.barh(i, qual, left=firm, color=ORANGE)
        ax.barh(i, 100 - firm - qual, left=firm + qual, color=GREY, alpha=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels([p if len(p) < 46 else p[:43] + "…"
                        for p in promises.promise], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 150)
    for i, row in enumerate(promises.itertuples()):
        ax.text(103, i, f"{row.status}   (n = {row.generated})",
                va="center", fontsize=8,
                color={"paid": GREEN, "partly paid": ORANGE,
                       "not paid": RED, "not payable": INK}[row.status])
    _style(ax, "The brief's six promises, against what the evidence base pays",
           xlabel="share of the claims proposed for that promise (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GREY, alpha=0.25, linewidth=0.6)
    ax.set_xticks([0, 25, 50, 75, 100])
    return _save(fig, path)


def fig_provenance(provenance: pd.DataFrame, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colours = {"05": BLUE, "06": ORANGE, "07": RED, "08": PURPLE, "09": GREEN}
    ax.bar([f"task {t}" for t in provenance.source_task], provenance.claims,
           color=[colours.get(t, GREY) for t in provenance.source_task],
           alpha=0.9)
    for i, row in enumerate(provenance.itertuples()):
        plural = "table" if row.tables_cited == 1 else "tables"
        ax.text(i, row.claims + 1.5,
                f"{row.claims}\n{row.tables_cited} {plural}",
                ha="center", fontsize=8, color=INK)
    _style(ax, "Every surviving sentence rests on a table an earlier task "
               "committed", ylabel="publishable claims")
    ax.set_ylim(0, provenance.claims.max() * 1.3)
    return _save(fig, path)


def fig_refusals(ledger: pd.DataFrame, path: Path) -> str:
    """Refusals by family and blocking gate — the report's real content."""
    refused = ledger[ledger.status == ins.REFUSED]
    pivot = (refused.groupby(["family", "blocked_by"]).size()
             .unstack(fill_value=0).sort_values(
                 by=list(refused.blocked_by.unique()), ascending=True))
    fig, ax = plt.subplots(figsize=(9, 5))
    left = np.zeros(len(pivot))
    gate_colour = {"evidence": BLUE, "lint": RED,
                   "identification": ORANGE, "consistency": PURPLE}
    for gate in pivot.columns:
        ax.barh(pivot.index, pivot[gate], left=left,
                color=gate_colour.get(gate, GREY), label=gate, alpha=0.9)
        left = left + pivot[gate].to_numpy()
    _style(ax, f"Why {len(refused)} proposed sentences were refused",
           xlabel="claims refused")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GREY, alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, frameon=False, title="blocked by",
              title_fontsize=8, loc="lower right")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(keys: list[str], focus: str = "google") -> dict:
    frames = cmp.load_frames(keys)
    report: dict = {
        "task": "09 — insight generation and reporting",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "focus": focus,
        "companies": sorted(keys),
    }

    # -- 1. the salary audit, written before anything cites it --------------
    disclosure = ins.salary_disclosure(frames)
    by_publisher = ins.salary_disclosure_by_publisher(frames)
    feasible = ins.salary_feasible_cells(frames)
    pairs = ins.salary_pair_table(frames, feasible, focus)
    stratified = pd.concat(
        [ins.salary_pair_stratified(frames, row.company_a, row.company_b,
                                    row.publisher)
         for row in pairs.itertuples()] or [pd.DataFrame()],
        ignore_index=True)
    if not stratified.empty:
        stratified["identified"] = stratified.verdict == "difference"
    _write(disclosure, TABLES / "salary-disclosure.csv")
    _write(by_publisher, TABLES / "salary-disclosure-by-publisher.csv")
    _write(feasible, TABLES / "salary-feasible-cells.csv")
    _write(pairs, TABLES / "salary-pairs.csv")
    _write(stratified, TABLES / "salary-pairs-stratified.csv")
    salary = ins.salary_verdict(disclosure, feasible, pairs, stratified, focus)
    report["salary"] = salary

    # -- 2. C8: what a unanimity count is worth ------------------------------
    by_pub = pd.read_csv(TASK06 / "relative-share-by-publisher.csv")
    verdicts = pd.read_csv(TASK06 / "relative-share-verdict.csv")
    recount = ins.publisher_cell_floor(by_pub, verdicts)
    unanimity = ins.unanimity_is_floor_dependent(recount)
    power = ins.sign_test_power(7)
    _write(recount, TABLES / "publisher-cell-floor.csv")
    _write(unanimity, TABLES / "unanimity-verdict.csv")
    _write(power, TABLES / "sign-test-power.csv")
    nvidia = unanimity[unanimity.company == "nvidia"]
    report["c8"] = {
        "companies_whose_verdict_moves_with_the_floor": int(
            unanimity.floor_dependent.sum()),
        "companies_that_gain_confirmation_by_dropping_tests": int(
            unanimity.confirmation_gained_by_dropping_tests.sum()),
        "nvidia_tested_at_floor_0": int(
            recount[(recount.company == "nvidia")
                    & (recount.cell_floor == 0)].publishers_tested.iloc[0]),
        "nvidia_tested_at_floor_10": int(
            recount[(recount.company == "nvidia")
                    & (recount.cell_floor == 10)].publishers_tested.iloc[0]),
        "nvidia_p_at_floor_0": float(nvidia.sign_test_p_at_floor_0.iloc[0]),
        "min_publishers_for_significant_unanimity": int(
            power[power.clears_005].publishers_tested.min()),
    }

    # -- 3. compile every candidate -----------------------------------------
    candidates = ins.all_candidates(OUT, focus)
    ledger = ins.compile_claims(candidates, OUT)
    yields = ins.insight_yield(ledger)
    bindings = ins.evidence_bindings(candidates, OUT)
    countries = ins.country_vocabulary(OUT)
    lint = ins.lint_audit(candidates, countries)
    patterns = ins.prohibited_pattern_table(lint)
    conflicts = ins.consistency_table(OUT)
    refusals = ins.refusal_ledger(ledger)

    _write(ledger, TABLES / "claim-ledger.csv")
    _write(yields, TABLES / "insight-yield.csv")
    _write(bindings, TABLES / "evidence-bindings.csv")
    _write(lint, TABLES / "lint-audit.csv")
    _write(patterns, TABLES / "prohibited-patterns.csv")
    _write(conflicts, TABLES / "cross-task-conflicts.csv")
    _write(refusals, TABLES / "refused-claims.csv")

    # -- 4. the brief's questions and promises ------------------------------
    coverage = ins.question_coverage(ledger)
    promises = ins.promise_audit(ledger)
    actionable = ins.actionability_table(ledger)
    position = ins.strategy_position(ledger, OUT, focus)
    provenance = ins.claim_provenance(ledger)
    falsifiers = ins.falsifier_table(ledger)
    _write(coverage, TABLES / "question-coverage.csv")
    _write(promises, TABLES / "brief-promises.csv")
    _write(actionable, TABLES / "actionability.csv")
    _write(pd.DataFrame([vars(position)]), TABLES / "strategy-position.csv")
    _write(provenance, TABLES / "claim-provenance.csv")
    _write(falsifiers, TABLES / "falsifiers.csv")
    for audience in ("strategy", "hr_talent", "product", "exec"):
        _write(ins.audience_brief(ledger, audience),
               TABLES / f"audience-brief-{audience}.csv")

    report["yield"] = {
        "generated": int(len(ledger)),
        "published": int((ledger.status == ins.PUBLISHED).sum()),
        "published_qualified": int((ledger.status == ins.QUALIFIED).sum()),
        "refused": int((ledger.status == ins.REFUSED).sum()),
        "yield_pct": float(yields[yields.question == "all"].yield_pct.iloc[0]),
        "blocked_by": {k: int(v) for k, v in
                       ledger[ledger.status == ins.REFUSED]
                       .blocked_by.value_counts().items()},
    }
    report["gates"] = {
        "evidence_failures": int((~ledger.gate_evidence).sum()),
        "lint_failures": int((~ledger.gate_lint).sum()),
        "identification_failures": int((~ledger.gate_identification).sum()),
        "consistency_failures": int((~ledger.gate_consistency).sum()),
        "prohibited_patterns": int(len(patterns)),
        "patterns_that_fired": int((patterns.claims_caught > 0).sum())
        if "claims_caught" in patterns.columns else 0,
    }
    report["position"] = vars(position)
    report["promises"] = {
        row.status: int(n) for row, n in
        zip(promises.itertuples(), promises.status.value_counts()
            .reindex(promises.status).fillna(0))
    }
    report["conflicts"] = {
        "restated_quantities": int(len(conflicts)),
        "in_conflict": int(conflicts.conflict.sum()),
        "registers": sorted(conflicts[conflicts.conflict].register.unique()),
    }

    # -- 5. figures ----------------------------------------------------------
    figs = [
        fig_yield(yields, FIGURES / "01-insight-yield.png"),
        fig_gate_attrition(ledger, FIGURES / "02-gate-attrition.png"),
        fig_actionability(actionable, FIGURES / "03-audience-reach.png"),
        fig_cell_floor(recount, FIGURES / "04-publisher-cell-floor.png"),
        fig_salary(by_publisher, disclosure, FIGURES / "05-salary-missingness.png"),
        fig_promises(promises, FIGURES / "06-brief-promises.png"),
        fig_provenance(provenance, FIGURES / "07-claim-provenance.png"),
        fig_refusals(ledger, FIGURES / "08-refusal-ledger.png"),
    ]
    report["figures"] = figs

    # -- 6. standing checks --------------------------------------------------
    written = sorted(TABLES.glob("*.csv"))
    offenders = {}
    for path in written:
        df = pd.read_csv(path)
        bad = (ins.forbidden_columns(df)
               + ins.personal_data_columns_present(df))
        if bad:
            offenders[path.name] = bad
    report["privacy"] = {
        "tables_checked": len(written),
        "forbidden_columns": sorted(ins.FORBIDDEN_COLUMNS),
        "offenders": offenders,
        "passed": not offenders,
    }

    path = OUT / "task-09-insight-report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"tables  -> {TABLES.relative_to(REPO_ROOT)} ({len(written)} csv)")
    print(f"figures -> {FIGURES.relative_to(REPO_ROOT)} ({len(figs)} png)")
    print(f"report  -> {path.relative_to(REPO_ROOT)}")
    print(f"\nclaims generated: {report['yield']['generated']}")
    print(f"published on a firm verdict: {report['yield']['published']}, "
          f"on a qualified verdict: {report['yield']['published_qualified']}, "
          f"refused: {report['yield']['refused']} "
          f"({report['yield']['yield_pct']}% yield)")
    print(f"refused by gate: {report['yield']['blocked_by']}")
    print(f"C8: {report['c8']['companies_whose_verdict_moves_with_the_floor']} "
          f"of 6 verdicts move with the cell floor")
    print(f"salary benchmark available: {salary['benchmark_available']}")
    print(f"position: {position.detail}")
    print(f"privacy check passed: {report['privacy']['passed']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus", default="google",
                        help="the specialist's own company; named in the report")
    parser.add_argument("--companies", default=None,
                        help="comma-separated keys; defaults to everything that "
                             "passed the Task 06 feasibility screen")
    args = parser.parse_args()

    if args.companies:
        keys = [k.strip() for k in args.companies.split(",")]
    else:
        screen = pd.read_csv(TASK06 / "company-feasibility-screen.csv")
        keys = co.included_companies(screen)
    print(f"compiling claims for {len(keys)} companies: "
          f"{', '.join(sorted(keys))}")
    build(sorted(keys), focus=args.focus)


if __name__ == "__main__":
    main()
