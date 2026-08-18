"""Task 08 — build the company similarity tables, figures and evidence report.

Runs the shared layer in `src/similarity.py` over the six built company frames
and writes:

    members/ankit-google/task-08-tables/        (committed — aggregate only)
        similarity-pairs.csv               every pair on all five metrics
        similarity-matrix-cosine.csv       the heatmap's own numbers
        similarity-calibration.csv         THE RULER — read this before a score
        similarity-bootstrap.csv           intervals, rank stability, tiers
        metric-concordance.csv             how much the five metrics disagree
        vendor-sensitivity.csv             own products in, out, and per pair
        mix-sensitivity.csv                crude vs role-mix standardised
        support-sensitivity.csv            all skills vs the common core
        numerator-contribution.csv         which skills carry the score
        concept-skill-removal.csv          Task 04 §2.3's prediction, tested
        cluster-support.csv                partitions and bootstrap support
        network-thresholds.csv             the sweep behind the network graph
        trajectory-similarity.csv          co-movement, gated and closure-aware
        pair-verdicts.csv                  what survives all four checks
        skill-profiles.csv                 the vectors everything is built on

    members/ankit-google/task-08-figures/       (committed)
        01-similarity-heatmap.png
        02-calibration.png
        03-metric-disagreement.png
        04-rank-stability.png
        05-vendor-sensitivity.png
        06-dendrogram.png
        07-network-thresholds.png
        08-trajectory-refusal.png

    members/ankit-google/task-08-similarity-report.json   quality evidence

Nothing row-level is written. The task consumes `data/processed/`, which stays
git-ignored, and emits only aggregates.

    python src/build_similarity.py
    python src/build_similarity.py --companies google,meta,nvidia
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
import similarity as sim          # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "members" / "ankit-google"
TABLES = OUT / "task-08-tables"
FIGURES = OUT / "task-08-figures"
TASK06 = OUT / "task-06-tables"
TASK07 = OUT / "task-07-tables"

# Task 05's palette, unchanged, so the eight Task 08 figures sit beside the
# Task 06 and Task 07 ones without a reader re-learning what a colour means.
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

#: Skills carried into the committed profile table and the heatmap. The
#: similarity itself always uses the whole vocabulary — this is a reading
#: limit on one figure, not an analytical one.
TOP_SKILLS = 25


def _write(df: pd.DataFrame, path: Path) -> None:
    """Write a table, refusing the two column families Task 06 has banned."""
    bad = sim.forbidden_columns(df)
    if bad:
        raise ValueError(f"{path.name} carries forbidden columns {bad}")
    personal = sim.personal_data_columns_present(df)
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


def _label(row) -> str:
    return f"{row.company_a} – {row.company_b}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_heatmap(matrix: pd.DataFrame, cal: pd.DataFrame, path: Path) -> str:
    """The similarity matrix twice — raw, then on the calibrated scale.

    The two panels look alike, and that is the honest result: with an
    unrelated null near 0.13 and an identical null near 0.99, calibration is
    close to an affine rescaling, so it cannot and does not reorder the pairs.
    What it changes is the *reading* of a cell. 0.55 in the left panel invites
    "moderately similar"; the right panel says it is halfway between two
    companies with nothing in common and two companies that are the same one.
    """
    keys = list(matrix.index)
    cal_m = pd.DataFrame(np.nan, index=keys, columns=keys, dtype=float)
    for row in cal.itertuples():
        cal_m.loc[row.company_a, row.company_b] = row.calibrated
        cal_m.loc[row.company_b, row.company_a] = row.calibrated
    np.fill_diagonal(cal_m.values, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, m, title in (
            (axes[0], matrix, "Raw cosine — spans 0.50 to 0.92, which reads as "
                              "'all six are similar'"),
            (axes[1], cal_m, "Calibrated — 0 is two unrelated profiles, 1 is the "
                             "same company twice")):
        im = ax.imshow(m.to_numpy(dtype=float), cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(range(len(keys)), keys, rotation=45, ha="right", fontsize=8, color=INK)
        ax.set_yticks(range(len(keys)), keys, fontsize=8, color=INK)
        for i in range(len(keys)):
            for j in range(len(keys)):
                v = m.iat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.6 else INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
        for side in ax.spines.values():
            side.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Skill-demand similarity, share_of_skilled vectors — calibration "
                 "is near-affine here, so it rescales the levels without "
                 "reordering the pairs", fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.88)
    return _save(fig, path)


def fig_calibration(cal: pd.DataFrame, path: Path) -> str:
    """Every pair against both nulls, on one axis.

    The gap between the observed points and the identical-null band is the
    whole argument for the calibrated column: it is wide for every pair, so no
    two of these companies are statistically indistinguishable, and it is
    invisible in the raw matrix.
    """
    block = cal.sort_values("observed")
    y = np.arange(len(block))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hlines(y, block.null_unrelated, block.null_identical, color=GREY, lw=1.2, alpha=0.6)
    ax.scatter(block.null_unrelated, y, s=34, color=GREY, marker="|",
               label="unrelated null (mean)")
    ax.scatter(block.null_identical, y, s=34, color=INK, marker="|",
               label="identical-companies null (mean)")
    ax.scatter(block.observed, y, s=44, color=BLUE, zorder=3, label="observed")
    ax.set_yticks(y, [_label(r) for r in block.itertuples()], fontsize=8)
    _style(ax, "Observed cosine sits far below what two identical companies score",
           "", "cosine similarity")
    ax.set_xlim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=3)
    return _save(fig, path)


def fig_metric_disagreement(pt: pd.DataFrame, conc: pd.DataFrame, path: Path) -> str:
    """Each pair's rank under all five metrics, and the concordance behind it.

    A parallel-coordinates plot rather than five bar charts because the claim
    is about *crossing*: google – meta is rank 1 on the left of the axis and
    near the bottom on the right, and a reader should see the lines cross
    rather than be told they do.
    """
    metrics = [m for m in sim.ALL_METRICS if f"rank_{m}" in pt.columns]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(metrics))
    for row in pt.itertuples():
        ranks = [getattr(row, f"rank_{m}") for m in metrics]
        top = row.rank_cosine <= 2
        ax1.plot(x, ranks, color=BLUE if top else GREY, lw=1.8 if top else 0.9,
                 alpha=1.0 if top else 0.55, zorder=3 if top else 1)
        if top:
            ax1.text(-0.08, ranks[0], f"{row.company_a} – {row.company_b}",
                     fontsize=8, color=BLUE, ha="right", va="center")
    ax1.set_xticks(x, [m.replace("_", " ") for m in metrics], fontsize=8, rotation=20)
    ax1.invert_yaxis()
    _style(ax1, "Rank of each pair under each metric (1 = most similar)", "rank", "")
    ax1.set_xlim(-0.9, len(metrics) - 0.6)

    order = conc.sort_values("rank_correlation")
    y = np.arange(len(order))
    ax2.barh(y, order.rank_correlation,
             color=[BLUE if s else ORANGE for s in order.same_family])
    ax2.set_yticks(y, [f"{r.metric_a} vs {r.metric_b}" for r in order.itertuples()],
                   fontsize=7.5)
    ax2.axvline(0, color=INK, lw=0.9)
    _style(ax2, "Rank correlation: blue within a family, orange across", "",
           "Spearman rho between the two rankings")
    fig.suptitle("The ranking is metric-dependent: prevalence-weighted and "
                 "rank/set metrics do not order the same pairs the same way",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.88)
    return _save(fig, path)


def fig_rank_stability(tiers: pd.DataFrame, path: Path) -> str:
    """Bootstrap intervals and rank stability, tier-coloured.

    The point of the figure is the middle band. Eleven of fifteen pairs have
    overlapping rank intervals, so the ordering inside that band is a list of
    fifteen numbers, not a ranking of fifteen pairs. The bottom two form their
    own tier — separated from the band, but not from each other, which is why
    neither clears the stability floor.
    """
    block = tiers.sort_values("rank", ascending=False)
    y = np.arange(len(block))
    palette = {1: BLUE, 2: GREEN, 3: GREY, 4: ORANGE, 5: PURPLE}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.8),
                                   gridspec_kw={"width_ratios": [1.4, 1]})
    for i, row in enumerate(block.itertuples()):
        c = palette.get(row.tier, GREY)
        ax1.plot([row.rank_low, row.rank_high], [i, i], color=c, lw=3, alpha=0.55)
        ax1.scatter([row.rank], [i], s=40, color=c, zorder=3)
    ax1.set_yticks(y, [_label(r) for r in block.itertuples()], fontsize=8)
    _style(ax1, "Bootstrap rank interval per pair, coloured by tier", "",
           "rank (1 = most similar)")

    ax2.barh(y, block.rank_stability,
             color=[palette.get(t, GREY) for t in block.tier])
    ax2.axvline(sim.RANK_STABILITY_FLOOR, color=RED, lw=1.1, ls="--")
    ax2.text(sim.RANK_STABILITY_FLOOR, len(block) - 1.2,
             f" floor {sim.RANK_STABILITY_FLOOR:.0%}", color=RED, fontsize=8, va="top")
    ax2.set_yticks(y, [""] * len(block))
    _style(ax2, "Share of resamples keeping the observed rank", "",
           "rank stability")
    ax2.set_xlim(0, 1.05)
    fig.suptitle("Two of fifteen ranks are identified: eleven pairs share one "
                 "indistinguishable band and the bottom two are a tie neither "
                 "resample can separate", fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.9)
    return _save(fig, path)


def fig_vendor(vend: pd.DataFrame, path: Path) -> str:
    """Each pair with its own products in and out — the largest single lever."""
    block = vend.sort_values("all_skills")
    y = np.arange(len(block))
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.hlines(y, block.all_skills, block.all_products_dropped, color=GREY, lw=2)
    ax.scatter(block.all_skills, y, s=42, color=INK, label="all skills", zorder=3)
    ax.scatter(block.all_products_dropped, y, s=42, color=ORANGE,
               label="every company's own products dropped", zorder=3)
    ax.set_yticks(y, [_label(r) for r in block.itertuples()], fontsize=8)
    _style(ax, "Own products move a pair by up to "
               f"{block.delta_all_products.max():.2f} cosine points", "", "cosine similarity")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, path)


def _leaf_order(merges, keys) -> list[str]:
    """Leaves in tree order, so the drawn branches never cross.

    Alphabetical leaves make a six-company dendrogram unreadable — the
    horizontals cross and the reader cannot tell which pair merged.
    """
    members = {frozenset([k]): [k] for k in keys}
    for ca, cb, _ in merges:
        members[ca | cb] = members[ca] + members[cb]
    return members[frozenset(keys)] if merges else list(keys)


def _draw_dendrogram(ax, merges, keys, title: str) -> None:
    keys = _leaf_order(merges, keys)
    pos = {frozenset([k]): float(i) for i, k in enumerate(keys)}
    height = {frozenset([k]): 0.0 for k in keys}
    for ca, cb, d in merges:
        xa, xb = pos[ca], pos[cb]
        ha, hb = height[ca], height[cb]
        ax.plot([xa, xa, xb, xb], [ha, d, d, hb], color=INK, lw=1.4)
        new = ca | cb
        pos[new] = (xa + xb) / 2
        height[new] = d
    ax.set_xticks(range(len(keys)), keys, rotation=45, ha="right", fontsize=8)
    for lab in ax.get_xticklabels():
        lab.set_color(_colour(lab.get_text()))
    _style(ax, title, "cosine distance (1 − similarity)", "")
    ax.grid(axis="y", color=GREY, alpha=0.25, linewidth=0.6)


def fig_dendrogram(inc, skills, support: pd.DataFrame, path: Path) -> str:
    """The tree on both vocabularies, with its bootstrap support printed.

    Side by side because the pair of trees is the finding. The left one is
    supported in most resamples and would be publishable on its own; the right
    one is the same data with own products removed and it puts a different
    company on its own branch. Resampling stability is not specification
    stability.
    """
    keys = sorted(inc)
    drop = set().union(*sim.vendor_skills().values())
    keep = np.array([s not in drop for s in skills])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, mask, name in ((axes[0], None, "all_skills"),
                           (axes[1], keep, "all_products_dropped")):
        prof = sim.profiles(inc)
        dist = pd.DataFrame(index=keys, columns=keys, dtype=float)
        for a in keys:
            for b in keys:
                va, vb = prof[a], prof[b]
                if mask is not None:
                    va, vb = va[mask], vb[mask]
                dist.loc[a, b] = 0.0 if a == b else 1.0 - sim.cosine(va, vb)
        merges = sim.average_linkage(dist)
        sup = support[(support.vocabulary == name) & (support.clusters == 3)]
        s = float(sup.bootstrap_support.iloc[0]) if len(sup) else float("nan")
        _draw_dendrogram(ax, merges, keys,
                         f"{name.replace('_', ' ')} — k=3 partition holds in "
                         f"{s:.0%} of resamples")
    fig.suptitle("Average linkage on cosine distance — the same postings, two "
                 "defensible vocabularies, two different outliers",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.88)
    return _save(fig, path)


def fig_network(sweep: pd.DataFrame, matrix: pd.DataFrame, path: Path) -> str:
    """The threshold sweep, and the graph it produces at three cut points.

    A network graph is a threshold in a trench coat. Publishing three of them
    beside the sweep is the cheapest honest alternative to picking one and
    hoping the reader does not ask why.
    """
    fig = plt.figure(figsize=(13, 6.4))
    ax0 = fig.add_subplot(2, 1, 1)
    ax0.plot(sweep.threshold, sweep.edges, color=BLUE, lw=1.8, marker="o", ms=4,
             label="edges")
    ax0.plot(sweep.threshold, sweep.components, color=ORANGE, lw=1.8, marker="s", ms=4,
             label="connected components")
    _style(ax0, "No plateau: the graph goes from complete to empty with no "
                "stable range to justify one picture", "count", "edge threshold (cosine)")
    ax0.legend(frameon=False, fontsize=8)

    keys = list(matrix.index)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(keys), endpoint=False)
    xy = {k: (np.cos(a), np.sin(a)) for k, a in zip(keys, angles)}
    for i, t in enumerate((0.55, 0.65, 0.75)):
        ax = fig.add_subplot(2, 3, 4 + i)
        for a, b in sim.pairs(keys):
            v = float(matrix.loc[a, b])
            if v >= t:
                ax.plot([xy[a][0], xy[b][0]], [xy[a][1], xy[b][1]],
                        color=GREY, lw=1 + 6 * (v - t), alpha=0.8, zorder=1)
        for k in keys:
            ax.scatter(*xy[k], s=120, color=_colour(k), zorder=3)
            ax.text(xy[k][0] * 1.28, xy[k][1] * 1.28, k, fontsize=7.5,
                    ha="center", va="center", color=INK)
        n_edges = int(sweep.loc[np.isclose(sweep.threshold, t), "edges"].iloc[0])
        ax.set_title(f"threshold {t:.2f} — {n_edges} of 15 edges",
                     fontsize=9, color=INK)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")
    fig.suptitle("The network graph is a choice of threshold, so the sweep is "
                 "published with it", fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.9, hspace=0.45)
    return _save(fig, path)


def fig_trajectory(traj: pd.DataFrame, null: dict, path: Path) -> str:
    """Why trajectory similarity is refused, in the three ways it fails.

    Left: the observed correlations against the band six independent series
    produce after closure. Right: the width of each pair's interval. Neither
    panel needs the other to make the point, which is why both are here.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    block = traj.sort_values("r_log_share")
    y = np.arange(len(block))
    ax1.axvspan(null["p2_5"], null["p97_5"], color=ORANGE, alpha=0.22,
                label="closure null, 95% of draws")
    ax1.axvline(null["analytic"], color=ORANGE, lw=1.2, ls="--",
                label=f"−1/(D−1) = {null['analytic']:.2f}")
    ax1.hlines(y, block.ci_low, block.ci_high, color=GREY, lw=2)
    ax1.scatter(block.r_log_share, y, s=40,
                color=[BLUE if e else GREY for e in block.eligible], zorder=3)
    ax1.axvline(0, color=INK, lw=0.9)
    ax1.set_yticks(y, [_label(r) + (" *" if r.eligible else "") for r in block.itertuples()],
                   fontsize=8)
    _style(ax1, "Correlation of log panel shares, 95% Fisher intervals", "",
           "correlation (* = both members pass the Task 07 gate)")
    ax1.legend(frameon=False, fontsize=8, loc="lower right")

    ax2.barh(y, block.ci_width, color=GREY)
    ax2.axvline(2.0, color=RED, lw=1.1, ls="--")
    ax2.text(2.0, len(block) - 0.5, " the whole range", color=RED, fontsize=8, va="top")
    ax2.set_yticks(y, [""] * len(block))
    ax2.set_xlim(0, 2.1)
    _style(ax2, "Width of that interval", "", "interval width, out of 2.0")
    fig.suptitle("Trajectory similarity is not identified here: one eligible "
                 "pair, a mean correlation inside the closure null, and "
                 "intervals half the range wide",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.88)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(keys: list[str], focus: str = "google") -> dict:
    frames = cmp.load_frames(keys)
    longs = cmp.load_long(keys)
    report: dict = {
        "task": "08-company-similarity",
        "focus_company": focus,
        "companies": keys,
        "primary_metric": sim.PRIMARY_METRIC,
        "seed": sim.SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # -- the vectors --------------------------------------------------------
    inc, skills = sim.incidence(longs, frames)
    prof = sim.profile_frame(inc, skills)
    sets = sim.supported_sets(longs, frames)
    top = prof.mean(axis=1).sort_values(ascending=False).head(TOP_SKILLS).index
    _write(prof.loc[top].round(4).reset_index(), TABLES / "skill-profiles.csv")
    report["profiles"] = {
        "skills_in_vocabulary": len(skills),
        "postings_per_company": {k: int(v.shape[0]) for k, v in inc.items()},
        "denominator": "skilled postings, Facilities/Operations excluded (Task 06)",
        "top_skills_published": int(len(top)),
    }

    # -- the ranking, five ways ---------------------------------------------
    pt = sim.pair_table(prof, sets)
    _write(pt, TABLES / "similarity-pairs.csv")
    matrix = sim.similarity_matrix(prof)
    _write(matrix.round(4).reset_index(), TABLES / "similarity-matrix-cosine.csv")
    conc = sim.metric_concordance(pt)
    _write(conc, TABLES / "metric-concordance.csv")
    cross = conc[~conc.same_family]
    within = conc[conc.same_family]
    report["metrics"] = {
        "metrics": sim.ALL_METRICS,
        "families": sim.METRIC_FAMILY,
        "within_family_rank_correlation": [round(float(within.rank_correlation.min()), 4),
                                           round(float(within.rank_correlation.max()), 4)],
        "cross_family_rank_correlation": [round(float(cross.rank_correlation.min()), 4),
                                          round(float(cross.rank_correlation.max()), 4)],
        "max_rank_spread": int(pt.rank_spread.max()),
        "pair_with_max_spread": _label(pt.loc[pt.rank_spread.idxmax()]),
    }

    # -- the ruler ----------------------------------------------------------
    cal = sim.calibration_table(prof, inc)
    _write(cal, TABLES / "similarity-calibration.csv")
    report["calibration"] = {
        "raw_range": [round(float(pt[sim.PRIMARY_METRIC].min()), 4),
                      round(float(pt[sim.PRIMARY_METRIC].max()), 4)],
        "identical_null_range": [round(float(cal.null_identical.min()), 4),
                                 round(float(cal.null_identical.max()), 4)],
        "unrelated_null_range": [round(float(cal.null_unrelated.min()), 4),
                                 round(float(cal.null_unrelated.max()), 4)],
        "calibrated_range": [round(float(cal.calibrated.min()), 4),
                             round(float(cal.calibrated.max()), 4)],
        "pairs_distinct_from_identical": int(cal.distinct.sum()),
        "pairs_above_unrelated": int(cal.above_unrelated.sum()),
        "draws": sim.N_NULL,
    }

    # -- what part of the ranking is real -----------------------------------
    boot = sim.bootstrap_pairs(inc)
    tiers = sim.rank_tiers(boot)
    _write(boot.merge(tiers[["company_a", "company_b", "tier"]],
                      on=["company_a", "company_b"]),
           TABLES / "similarity-bootstrap.csv")
    report["stability"] = {
        "draws": sim.N_BOOTSTRAP,
        "rank_stability_floor": sim.RANK_STABILITY_FLOOR,
        "ranks_identified": int(boot.rank_identified.sum()),
        "pairs": int(len(boot)),
        "tiers": int(tiers.tier.max()),
        "largest_tier_size": int(tiers.tier.value_counts().max()),
        "identified_pairs": [_label(r) for r in boot[boot.rank_identified].itertuples()],
        "mean_ci_width": round(float((boot.ci_high - boot.ci_low).mean()), 4),
    }

    # -- the four levers ----------------------------------------------------
    vend = sim.vendor_sensitivity(prof)
    _write(vend, TABLES / "vendor-sensitivity.csv")
    mix = sim.mix_sensitivity(longs, frames)
    _write(mix, TABLES / "mix-sensitivity.csv")
    supp = sim.support_sensitivity(prof, sets)
    _write(supp, TABLES / "support-sensitivity.csv")

    concepts = sim.concept_skills(longs)
    counts = pd.Series(sum(v.sum(axis=0) for v in inc.values()), index=skills)
    groups = {
        "concept": concepts,
        "postings_le_1": sorted(counts[counts <= 1].index),
        "postings_le_10": sorted(counts[counts <= 10].index),
    }
    contrib = sim.numerator_contribution(prof, groups)
    _write(contrib, TABLES / "numerator-contribution.csv")
    concept_drop = sim.group_removal(prof, concepts)
    _write(concept_drop, TABLES / "concept-skill-removal.csv")

    report["sensitivity"] = {
        "vendor_max_delta": round(float(vend.delta_all_products.max()), 4),
        "vendor_max_delta_pair": _label(vend.loc[vend.delta_all_products.idxmax()]),
        "vendor_rank_correlation": round(float(
            vend.rank_all_skills.corr(vend.rank_all_products_dropped, method="spearman")), 4),
        "vendor_max_rank_move": int(vend.rank_move.abs().max()),
        "mix_rank_correlation": round(float(
            mix.rank_crude.corr(mix.rank_standardised, method="spearman")), 4),
        "mix_max_rank_move": int(mix.rank_move.abs().max()),
        "mix_max_effect": round(float(mix.mix_effect.abs().max()), 4),
        "support_rank_correlation": round(float(
            supp.rank_all.corr(supp.rank_core, method="spearman")), 4),
        "core_skills": int(supp.n_core.iloc[0]),
        "concept_skills": len(concepts),
        "concept_numerator_share_mean": round(float(contrib.share_concept.mean()), 6),
        "concept_numerator_share_max": round(float(contrib.share_concept.max()), 6),
        "concept_removal_rank_correlation": round(float(
            concept_drop.rank_with.corr(concept_drop.rank_without, method="spearman")), 4),
        "concept_removal_max_delta": round(float(concept_drop.delta.abs().max()), 4),
        "share_postings_le_1_mean": round(float(contrib.share_postings_le_1.mean()), 6),
        "share_postings_le_10_mean": round(float(contrib.share_postings_le_10.mean()), 6),
        "share_top5_mean": round(float(contrib.share_top5.mean()), 4),
    }

    # -- structure ----------------------------------------------------------
    drop_all = sorted(set().union(*sim.vendor_skills().values()))
    support = pd.concat([
        sim.cluster_support(inc),
        sim.cluster_support(inc, drop_skills=drop_all, skills=skills),
    ], ignore_index=True)
    _write(support, TABLES / "cluster-support.csv")
    sweep = sim.network_thresholds(matrix)
    _write(sweep, TABLES / "network-thresholds.csv")

    base = support[support.vocabulary == "all_skills"].set_index("clusters")
    alt = support[support.vocabulary == "all_products_dropped"].set_index("clusters")
    report["structure"] = {
        "partitions": {int(k): {"all_skills": base.loc[k, "partition"],
                                "all_products_dropped": alt.loc[k, "partition"],
                                "same": bool(base.loc[k, "partition"] == alt.loc[k, "partition"]),
                                "support_all_skills": float(base.loc[k, "bootstrap_support"]),
                                "support_products_dropped": float(alt.loc[k, "bootstrap_support"])}
                       for k in base.index},
        "partition_changes_with_vocabulary": bool((base.partition != alt.partition).any()),
        "network_thresholds_with_all_edges": int((sweep.edges == sweep.possible_edges).sum()),
        "network_thresholds_with_no_edges": int((sweep.edges == 0).sum()),
        "largest_edge_plateau": int(sweep.groupby("edges").size().max()),
    }

    # -- trajectory, gated and refused --------------------------------------
    series = pd.read_csv(TASK07 / "panel-share-series.csv")
    gate = pd.read_csv(TASK07 / "forecastability-gate.csv")
    traj = sim.trajectory_table(series, gate)
    _write(traj, TABLES / "trajectory-similarity.csv")
    null = sim.closure_null(int(traj.n_periods.iloc[0]), len(keys))
    verdict = sim.trajectory_verdict(traj, null)
    report["trajectory"] = {
        "identified": verdict.identified,
        "reason": verdict.reason,
        **verdict.detail,
        "lowest_variation_pair": _label(traj.iloc[0]),
        "lowest_variation": float(traj.aitchison_variation.iloc[0]),
        "highest_variation_pair": _label(traj.iloc[-1]),
        "highest_variation": float(traj.aitchison_variation.iloc[-1]),
        "february_filled": False,
    }

    # -- what survives ------------------------------------------------------
    verdicts = sim.pair_verdicts(boot, pt, vend, mix)
    _write(verdicts, TABLES / "pair-verdicts.csv")
    robust = verdicts[verdicts.verdict == "robust"]
    report["verdicts"] = {
        "counts": {k: int(v) for k, v in verdicts.verdict.value_counts().items()},
        "robust_pairs": [_label(r) for r in robust.itertuples()],
        "headline": (f"{_label(robust.iloc[0])} is the most similar pair"
                     if len(robust) else "no pair survives all four checks"),
    }

    # -- figures ------------------------------------------------------------
    figs = [
        fig_heatmap(matrix, cal, FIGURES / "01-similarity-heatmap.png"),
        fig_calibration(cal, FIGURES / "02-calibration.png"),
        fig_metric_disagreement(pt, conc, FIGURES / "03-metric-disagreement.png"),
        fig_rank_stability(tiers, FIGURES / "04-rank-stability.png"),
        fig_vendor(vend, FIGURES / "05-vendor-sensitivity.png"),
        fig_dendrogram(inc, skills, support, FIGURES / "06-dendrogram.png"),
        fig_network(sweep, matrix, FIGURES / "07-network-thresholds.png"),
        fig_trajectory(traj, null, FIGURES / "08-trajectory-refusal.png"),
    ]
    report["figures"] = figs

    # -- standing checks ----------------------------------------------------
    written = sorted(TABLES.glob("*.csv"))
    offenders = {}
    for p in written:
        df = pd.read_csv(p)
        bad = sim.forbidden_columns(df) + sim.personal_data_columns_present(df)
        if bad:
            offenders[p.name] = bad
    report["privacy"] = {
        "tables_checked": len(written),
        "forbidden_columns": sorted(sim.FORBIDDEN_COLUMNS),
        "offenders": offenders,
        "passed": not offenders,
    }

    path = OUT / "task-08-similarity-report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"tables  -> {TABLES.relative_to(REPO_ROOT)} ({len(written)} csv)")
    print(f"figures -> {FIGURES.relative_to(REPO_ROOT)} ({len(figs)} png)")
    print(f"report  -> {path.relative_to(REPO_ROOT)}")
    print(f"\nranks identified: {report['stability']['ranks_identified']} of "
          f"{report['stability']['pairs']} pairs, in "
          f"{report['stability']['tiers']} tiers")
    print(f"headline: {report['verdicts']['headline']}")
    print(f"trajectory identified: {verdict.identified} ({verdict.reason})")
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
    print(f"scoring {len(keys)} companies: {', '.join(sorted(keys))}")
    build(sorted(keys), focus=args.focus)


if __name__ == "__main__":
    main()
