"""Task 12 — build the learned skill-extractor tables, figures and report.

Runs the contest in `src/skill_model.py` on the committed Google frames and
writes:

    members/ankit-google/task-12-tables/            (committed — aggregate only)
        contest-headline.csv          taxonomy vs learned, micro and macro P/R/F1
        per-skill-learned.csv         the learned model, one row per skill
        per-skill-taxonomy.csv        the taxonomy title path, same skills
        learned-title-associations.csv  the fitted model as legible weights
        adoption-gate.csv             the gate, cleared, and why not adopted
        dataset-summary.csv           what the contest ran on

    members/ankit-google/task-12-figures/           (committed)
        01-contest-headline.png
        02-per-skill-f1.png
        03-association-gap.png
        04-gate-scorecard.png

    members/ankit-google/task-12-skill-model-report.json

This driver is deliberately **not** a registered pipeline stage. The model is
not adopted (see the gate), so wiring it into the production DAG would overstate
its role — the pipeline keeps extracting skills with the taxonomy in the
`features` stage. It is run by hand, like a one-off experiment, and its evidence
is committed like any other task's.

    python src/build_skill_model.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display on a headless run
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

import skills as sk               # noqa: E402
import skill_model as sm          # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "members" / "ankit-google"
TABLES = OUT / "task-12-tables"
FIGURES = OUT / "task-12-figures"
PROCESSED = REPO_ROOT / "data" / "processed"

# Task 05's palette, unchanged, so these figures read beside the earlier ones.
INK = "#1f2933"
GREY = "#9aa5b1"
BLUE = "#2f6fb5"
ORANGE = "#e08a1e"
RED = "#c0392b"
GREEN = "#2e7d5b"


def _write(df: pd.DataFrame, path: Path) -> None:
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
# Tables
# ---------------------------------------------------------------------------


def headline_table(scores: dict) -> pd.DataFrame:
    """Both extractors on one grid, so the margin is a subtraction, not prose."""
    tax, learned = scores["taxonomy_title_path"], scores["learned_model"]
    rows = []
    for average in ("micro", "macro"):
        for metric in ("precision", "recall", "f1"):
            t, m = tax[average][metric], learned[average][metric]
            rows.append({"averaging": average, "metric": metric,
                         "taxonomy_title_path": t, "learned_model": m,
                         "margin": round(m - t, sm.ROUND)})
    return pd.DataFrame(rows)


def gate_table(verdict: dict, blockers: dict) -> pd.DataFrame:
    """The gate verdict and each adoption blocker, as one readable ledger."""
    rows = [{
        "item": "gate",
        "status": "cleared" if verdict["gate_cleared"] else "not cleared",
        "detail": (f"learned micro-F1 {verdict['learned_micro_f1']} vs taxonomy "
                   f"{verdict['taxonomy_micro_f1']} "
                   f"(margin {verdict['margin_micro_f1']:+})"),
    }]
    for name, blocker in blockers.items():
        rows.append({
            "item": name,
            "status": "blocks adoption" if blocker["blocked"] else "clear",
            "detail": blocker["detail"],
        })
    rows.append({
        "item": "decision",
        "status": "adopted" if verdict["adopted"] else "not adopted",
        "detail": (f"taxonomy retained ({verdict['retained']}); the model is "
                   "delivered with metrics as an experiment"),
    })
    return pd.DataFrame(rows)


def dataset_table(dataset: dict) -> pd.DataFrame:
    # Values as strings so integer counts render as "554", not "554.0" — the
    # column mixes counts and a rate, which otherwise coerces the whole column
    # to float.
    return pd.DataFrame(
        [{"key": k, "value": str(v)} for k, v in dataset.items()])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_headline(scores: dict, path: Path) -> str:
    """The two extractors across all six metrics — the contest at a glance."""
    tax, learned = scores["taxonomy_title_path"], scores["learned_model"]
    labels = ["micro\nprecision", "micro\nrecall", "micro\nF1",
              "macro\nprecision", "macro\nrecall", "macro\nF1"]
    keys = [("micro", "precision"), ("micro", "recall"), ("micro", "f1"),
            ("macro", "precision"), ("macro", "recall"), ("macro", "f1")]
    t = [tax[a][m] for a, m in keys]
    l = [learned[a][m] for a, m in keys]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.bar(x - w / 2, t, w, color=GREY, label="rule taxonomy (title path)")
    ax.bar(x + w / 2, l, w, color=BLUE, label="learned model (title -> skills)")
    for xi, (tv, lv) in enumerate(zip(t, l)):
        ax.text(xi - w / 2, tv + 0.01, f"{tv:.2f}", ha="center", fontsize=7.5,
                color=INK)
        ax.text(xi + w / 2, lv + 0.01, f"{lv:.2f}", ha="center", fontsize=7.5,
                color=INK)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    _style(ax, "Predicting the source-skill set from the job title, out of "
               "fold, on the same 554 postings", "score", "")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _save(fig, path)


def fig_per_skill(per_skill: pd.DataFrame, micro_f1: float, path: Path) -> str:
    """Learned F1 per skill, ordered by support — where the recall comes from."""
    block = per_skill.sort_values("support")
    y = np.arange(len(block))
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.barh(y, block.f1, color=BLUE, alpha=0.85)
    ax.axvline(micro_f1, color=RED, lw=1.1, ls="--")
    ax.text(micro_f1, len(block) - 0.5, f" micro-F1 {micro_f1:.2f}", color=RED,
            fontsize=8, va="top")
    ax.set_yticks(y, [f"{r.skill} ({r.support})" for r in block.itertuples()],
                  fontsize=7.5)
    ax.set_xlim(0, 1.0)
    _style(ax, "Learned F1 by skill (posting support in brackets) — the "
               "well-supported skills carry the score", "", "F1")
    return _save(fig, path)


def fig_association_gap(dataset: dict, scores: dict, path: Path) -> str:
    """The one figure that makes the win honest: recall above the literal ceiling.

    A rule that matches skill names in titles can recover at most the share of
    labels whose name is literally in the title. The learned model's recall sits
    far above that line because it predicts from association, and the taxonomy's
    title path sits far below it because it is deliberately conservative.
    """
    ceiling = dataset["title_literal_coverage"]
    tax_recall = scores["taxonomy_title_path"]["micro"]["recall"]
    learned_recall = scores["learned_model"]["micro"]["recall"]
    bars = [("taxonomy\ntitle path", tax_recall, GREY),
            ("literal title\nceiling", ceiling, ORANGE),
            ("learned\nmodel", learned_recall, BLUE)]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for i, (label, val, colour) in enumerate(bars):
        ax.bar(i, val, 0.6, color=colour)
        ax.text(i, val + 0.015, f"{val:.3f}", ha="center", fontsize=9, color=INK)
    ax.axhline(ceiling, color=ORANGE, lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=8.5)
    ax.set_ylim(0, 1.0)
    _style(ax, "Recall of the source labels: only 14.5% of skills are named in "
               "the title, yet the model recovers far more", "recall (micro)", "")
    return _save(fig, path)


def fig_scorecard(verdict: dict, blockers: dict, path: Path) -> str:
    """The verdict as a card: beaten baseline, cleared gate, retained taxonomy."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axis("off")
    ax.text(0.0, 0.94, "Task 12 gate — learned extractor vs the 176-skill rule "
            "taxonomy", fontsize=12.5, color=INK, weight="bold", va="top")
    ax.text(0.0, 0.82,
            f"micro-F1   taxonomy {verdict['taxonomy_micro_f1']:.4f}   →   "
            f"learned {verdict['learned_micro_f1']:.4f}   "
            f"(margin {verdict['margin_micro_f1']:+.4f})",
            fontsize=11, color=INK, va="top", family="monospace")
    ax.text(0.0, 0.72,
            f"gate: {'CLEARED' if verdict['gate_cleared'] else 'NOT CLEARED'}"
            f"      decision: {'ADOPTED' if verdict['adopted'] else 'NOT ADOPTED'}"
            f" — taxonomy retained",
            fontsize=11, color=GREEN if verdict["gate_cleared"] else RED,
            va="top", weight="bold")
    ax.text(0.0, 0.60, "Adoption requires more than beating the title path. "
            "Three facts about this dataset block it:", fontsize=9.5, color=INK,
            va="top")
    y = 0.50
    for name in verdict["adoption_blockers"]:
        ax.text(0.02, y, f"• {name.replace('_', ' ')}", fontsize=9.5, color=RED,
                va="top", weight="bold")
        ax.text(0.30, y, blockers[name]["detail"], fontsize=8.5, color=INK,
                va="top", wrap=True)
        y -= 0.16
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(company: str = "google") -> dict:
    clean = pd.read_parquet(
        PROCESSED / company / f"{company}_jobs_clean.parquet")
    long = pd.read_parquet(
        PROCESSED / company / f"{company}_skills_long.parquet")

    postings_with_text = int((clean.job_description.fillna("").str.len() > 0).sum())
    postings_total = int(len(clean))
    taxonomy_vocab_size = len(sk.SKILLS)

    res = sm.run_contest(
        clean, long,
        taxonomy_vocab_size=taxonomy_vocab_size,
        postings_with_text=postings_with_text,
        postings_total=postings_total,
    )

    scores, verdict, blockers = res["scores"], res["verdict"], res["blockers"]

    # -- tables -------------------------------------------------------------
    _write(headline_table(scores), TABLES / "contest-headline.csv")
    _write(res["per_skill_learned"], TABLES / "per-skill-learned.csv")
    _write(res["per_skill_taxonomy"], TABLES / "per-skill-taxonomy.csv")
    _write(res["associations"], TABLES / "learned-title-associations.csv")
    _write(gate_table(verdict, blockers), TABLES / "adoption-gate.csv")
    _write(dataset_table(res["dataset"]), TABLES / "dataset-summary.csv")

    # -- figures ------------------------------------------------------------
    figs = [
        fig_headline(scores, FIGURES / "01-contest-headline.png"),
        fig_per_skill(res["per_skill_learned"],
                      scores["learned_model"]["micro"]["f1"],
                      FIGURES / "02-per-skill-f1.png"),
        fig_association_gap(res["dataset"], scores,
                            FIGURES / "03-association-gap.png"),
        fig_scorecard(verdict, blockers, FIGURES / "04-gate-scorecard.png"),
    ]

    # -- report -------------------------------------------------------------
    report = {
        "task": "12-fine-tune-skill-extractor",
        "focus_company": company,
        "seed": sm.SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": res["dataset"],
        "model": res["model"],
        "scores": scores,
        "blockers": blockers,
        "verdict": verdict,
        "tables": sorted(p.name for p in TABLES.glob("*.csv")),
        "figures": figs,
    }
    path = OUT / "task-12-skill-model-report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"tables  -> {TABLES.relative_to(REPO_ROOT)} "
          f"({len(report['tables'])} csv)")
    print(f"figures -> {FIGURES.relative_to(REPO_ROOT)} ({len(figs)} png)")
    print(f"report  -> {path.relative_to(REPO_ROOT)}")
    print(f"\npostings scored: {res['dataset']['postings_scored']} | "
          f"skills learnable: {res['dataset']['skills_learnable']} of "
          f"{taxonomy_vocab_size}")
    print(f"taxonomy title-path micro-F1: {verdict['taxonomy_micro_f1']}")
    print(f"learned model     micro-F1: {verdict['learned_micro_f1']} "
          f"(margin {verdict['margin_micro_f1']:+})")
    print(f"gate cleared: {verdict['gate_cleared']} | adopted: "
          f"{verdict['adopted']} | blockers: "
          f"{', '.join(verdict['adoption_blockers'])}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", default="google",
                        help="the specialist's own company frame to score")
    args = parser.parse_args()
    build(args.company)


if __name__ == "__main__":
    main()
