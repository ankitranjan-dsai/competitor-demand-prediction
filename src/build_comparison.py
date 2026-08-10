"""Task 06 — build the competitor comparison tables, figures and evidence report.

Reads the six built company frames from `data/processed/<company>/`, runs the
shared comparison layer in `src/compare.py` over them, and writes:

    members/ankit-google/task-06-tables/        (committed — aggregate only)
        company-feasibility-screen.csv     who is in the set and why
        employer-matching-audit.csv        every employer string, kept/dropped
        competitor-set-manifest.csv        what was built, per company
        company-comparability.csv          THE GATE — read this first
        common-publisher-panel.csv         postings per shared publisher
        common-panel-by-month.csv          the panel collapses within months
        volume-panel-sensitivity.csv       4 treatments x 6 companies
        volume-verdict.csv                 whose direction is identified
        half-over-half.csv                 H2 vs H1, raw and balanced
        collection-artefact-check.csv      thin months vs thin publisher roster
        company-monthly-correlation.csv    how much the six move together
        relative-share-by-half.csv         the headline estimator
        relative-share-by-publisher.csv    its robustness check
        relative-share-verdict.csv         sign agreement across channels
        job-function-mix.csv               with Wilson intervals
        job-function-mix-distance.csv      pairwise total-variation distance
        skill-coverage-standardised.csv    crude vs mix-adjusted coverage
        self-reference-audit.csv           employer name leaking into skills
        skill-share-by-company.csv         share_of_skilled, Facilities out
        skill-share-standardised.csv       crude vs mix-adjusted skill shares
        skill-distinctiveness-<company>.csv log-lift vs rest of sector + BH
        google-vs-<rival>-skill-gaps.csv   Newcombe intervals per rival
        skill-stratified-verdicts.csv      does a gap survive stratification
        skill-panel-robustness.csv         does it survive the common panel

    members/ankit-google/task-06-figures/       (committed)
        01-comparability-gate.png
        02-volume-index-by-company.png
        03-relative-share.png
        04-collection-artefact-check.png
        05-job-function-mix.png
        06-coverage-standardised.png
        07-skill-heatmap.png
        08-google-distinctiveness.png
        09-google-vs-rivals.png

    members/ankit-google/task-06-comparison-report.json   quality evidence

Nothing row-level is written. The comparison consumes `data/processed/`, which
stays git-ignored, and emits only aggregates.

    python src/build_comparison.py
    python src/build_comparison.py --focus google --companies google,meta
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

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "members" / "ankit-google"
TABLES = OUT / "task-06-tables"
FIGURES = OUT / "task-06-figures"

# Task 05's palette, reused so Task 05 and Task 06 figures sit together.
INK = "#1f2933"
GREY = "#9aa5b1"
BLUE = "#2f6fb5"
ORANGE = "#e08a1e"
RED = "#c0392b"
GREEN = "#2e7d5b"
PURPLE = "#7d5ba6"

#: One colour per company, fixed here so every figure in the task — and every
#: figure a teammate builds on top of it — identifies a company the same way.
COMPANY_COLOURS = {
    "google": BLUE, "microsoft": ORANGE, "meta": PURPLE,
    "nvidia": GREEN, "snowflake": RED, "databricks": "#4a5568",
}

#: How many skills the heatmap and the standardisation table carry. Enough to
#: show the shape of each company's stack, few enough to read on one page.
TOP_SKILLS = 25


def _write(df: pd.DataFrame, path: Path) -> None:
    """Write a table, refusing the two column families Task 06 has banned."""
    bad = cmp.forbidden_columns(df)
    if bad:
        raise ValueError(f"{path.name} carries forbidden columns {bad}")
    personal = cmp.personal_data_columns_present(df)
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


def fig_gate(gate: pd.DataFrame, path: Path) -> str:
    """The three diagnostics that decide what may be compared."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.2))
    order = gate.sort_values("common_panel_share")
    y = np.arange(len(order))
    colours = [_colour(c) for c in order.company]

    ax1.barh(y, order.common_panel_share * 100, color=colours)
    ax1.axvline(cmp.MIN_COMMON_PANEL_SHARE * 100, color=RED, ls="--", lw=1.2)
    ax1.set_yticks(y, order.company, fontsize=8)
    _style(ax1, "Postings on the shared publisher panel", "", "% of company")
    ax1.text(cmp.MIN_COMMON_PANEL_SHARE * 100 + 1, -0.4,
             f"floor {cmp.MIN_COMMON_PANEL_SHARE:.0%}", color=RED, fontsize=7)

    ax2.barh(y, order.own_channel_share * 100, color=colours)
    ax2.set_yticks(y, order.company, fontsize=8)
    _style(ax2, "Published through their own careers site", "", "% of company")

    ax3.barh(y, order.skill_coverage * 100, color=colours)
    ax3.set_yticks(y, order.company, fontsize=8)
    _style(ax3, "Postings with any extracted skill", "", "% of company")
    fig.suptitle("Task 06 comparability gate — three ways the same six "
                 "companies are not observed alike",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.82)
    return _save(fig, path)


def fig_volume_index(panel: pd.DataFrame, verdicts: pd.DataFrame,
                     path: Path) -> str:
    """Small multiples: four panel treatments per company, on one scale."""
    keys = list(verdicts.company)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True, sharey=True)
    for ax, key in zip(axes.ravel(), keys):
        block = panel[panel.company == key]
        for col, colour, label in (
            ("raw_index", GREY, "raw"),
            ("balanced_index", BLUE, "balanced"),
            ("chained_index", ORANGE, "chained"),
            ("bilateral_index", GREEN, "bilateral"),
        ):
            ax.plot(block.period, block[col], color=colour, lw=1.6, label=label)
        ax.axhline(100, color=INK, lw=0.8, ls=":")
        row = verdicts[verdicts.company == key].iloc[0]
        mark = "agree" if row.treatments_agree else "DISAGREE"
        _style(ax, f"{key} — {row.direction} ({mark}, spread {row.spread:.0f})")
        ax.tick_params(axis="x", rotation=90, labelsize=6)
    axes.ravel()[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("Monthly volume index (January = 100) under four publisher-panel "
                 "treatments — only Meta and Snowflake agree on sign",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.88)
    return _save(fig, path)


def fig_relative_share(rel: pd.DataFrame, by_pub: pd.DataFrame,
                       path: Path) -> str:
    """Pooled relative move, with each shared publisher's own estimate on top."""
    fig, ax = plt.subplots(figsize=(9.5, 5))
    order = rel.sort_values("log_share_change")
    y = np.arange(len(order))
    ax.barh(y, order.log_share_change, color=[_colour(c) for c in order.company],
            alpha=0.85, height=0.55, zorder=2)
    for i, key in enumerate(order.company):
        pts = by_pub.loc[by_pub.company == key, "log_share_change"]
        ax.scatter(pts, np.full(len(pts), i), s=16, color=INK, alpha=0.55,
                   zorder=3, label="per publisher" if i == 0 else None)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(y, order.company, fontsize=9)
    _style(ax, "Change in share of the shared-publisher panel, H1 -> H2 2023",
           "", "log share change (0 = grew with the panel)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    return _save(fig, path)


def fig_collection(art: pd.DataFrame, path: Path) -> str:
    """The identification argument for H2 > H1, in one chart."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
    x = np.arange(len(art))
    ax1.bar(x, art.postings, color=[RED if t else BLUE
                                    for t in art.shared_thin_month], width=0.62)
    ax1b = ax1.twinx()
    ax1b.plot(x, art.distinct_publishers, color=INK, lw=1.6, marker="o", ms=3)
    ax1b.set_ylabel("distinct publishers", fontsize=8, color=INK)
    ax1b.tick_params(labelsize=8, colors=INK)
    ax1b.spines["top"].set_visible(False)
    _style(ax1, "Pooled postings (bars; red = every company thin and the "
                "publisher roster thin with them) and publishers (line)",
           "postings")

    ax2.plot(x, art.postings_per_publisher, color=ORANGE, lw=2, marker="o", ms=4)
    _style(ax2, "Postings per observed publisher — H2's rise is more postings "
                "per board, not more boards", "postings / publisher")
    ax2.set_xticks(x, art.period, rotation=90, fontsize=7)
    return _save(fig, path)


def fig_mix(mix: pd.DataFrame, path: Path) -> str:
    """Stacked role mix — the confounder, shown before it is adjusted away."""
    wide = mix.pivot_table(index="company", columns="job_function",
                           values="share", fill_value=0.0)
    order = wide.sum().sort_values(ascending=False).index
    wide = wide[order]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    left = np.zeros(len(wide))
    palette = [BLUE, ORANGE, PURPLE, GREEN, RED, "#4a5568", GREY,
               "#8c6d3f", "#3f8c85", "#b5651d"]
    for i, fn in enumerate(wide.columns):
        vals = wide[fn].to_numpy() * 100
        ax.barh(wide.index, vals, left=left, color=palette[i % len(palette)],
                label=fn, height=0.62)
        left += vals
    _style(ax, "Job-function mix — Snowflake is 70% Engineering, "
               "Meta 44% Analytics, Google 11% data-centre Facilities",
           "", "% of postings")
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.32))
    return _save(fig, path)


def fig_standardised(std: pd.DataFrame, path: Path) -> str:
    """Crude vs mix-standardised skill coverage, as a dumbbell."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    order = std.sort_values("crude")
    y = np.arange(len(order))
    for i, row in enumerate(order.itertuples()):
        ax.plot([row.crude * 100, row.standardised * 100], [i, i],
                color=GREY, lw=2, zorder=1)
    ax.scatter(order.crude * 100, y, s=48, color=GREY, zorder=2, label="crude")
    ax.scatter(order.standardised * 100, y, s=48,
               color=[_colour(c) for c in order.company], zorder=3,
               label="standardised to pooled role mix")
    ax.set_yticks(y, order.company, fontsize=9)
    _style(ax, "Skill coverage before and after standardising the role mix — "
               "the gap survives, so it is not composition",
           "", "% of postings with any extracted skill")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    return _save(fig, path)


def fig_heatmap(matrix: pd.DataFrame, path: Path) -> str:
    """Skill x company share_of_skilled."""
    fig, ax = plt.subplots(figsize=(8.5, 9))
    data = matrix.to_numpy() * 100
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=45,
                  ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=8, color=INK)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 55 else INK)
    ax.set_title("share_of_skilled by company (%) — Facilities/Operations "
                 "excluded from every denominator",
                 fontsize=10, color=INK, loc="left", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.4, label="% of skilled postings")
    return _save(fig, path)


def fig_distinctiveness(dist: pd.DataFrame, company: str, path: Path,
                        n: int = 12) -> str:
    """What this company asks for that the sector does not, and vice versa."""
    sig = dist[dist.significant & dist.supported]
    top = pd.concat([sig.head(n), sig.tail(n)]).drop_duplicates("skill")
    top = top.sort_values("log_lift")
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    y = np.arange(len(top))
    colours = [ORANGE if r else (_colour(company) if v > 0 else GREY)
               for r, v in zip(top.self_referential, top.log_lift)]
    ax.barh(y, top.log_lift, color=colours, height=0.66)
    ax.axvline(0, color=INK, lw=0.9)
    labels = [f"{s} *" if r else s
              for s, r in zip(top.skill, top.self_referential)]
    ax.set_yticks(y, labels, fontsize=8)
    _style(ax, f"{company} vs the rest of the sector — log2 lift in "
               f"share_of_skilled (BH-FDR q <= 0.05)",
           "", "log2( company share / rest-of-sector share )")
    ax.text(0.99, 0.02, "* the company's own product", transform=ax.transAxes,
            ha="right", fontsize=7, color=ORANGE)
    return _save(fig, path)


def fig_pairwise(gaps: dict[str, pd.DataFrame], focus: str, path: Path,
                 n: int = 6) -> str:
    """Newcombe intervals for the focus company against each rival."""
    rivals = sorted(gaps)
    fig, axes = plt.subplots(1, len(rivals), figsize=(3.1 * len(rivals), 5.4),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, rival in zip(axes, rivals):
        g = gaps[rival]
        g = g[g.significant]
        block = pd.concat([g.head(n), g.tail(n)]).drop_duplicates("skill") \
                  .sort_values("diff")
        y = np.arange(len(block))
        ax.hlines(y, block.ci_low * 100, block.ci_high * 100, color=GREY, lw=2)
        ax.scatter(block["diff"] * 100, y, s=26,
                   color=[_colour(focus) if d > 0 else _colour(rival)
                          for d in block["diff"]], zorder=3)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_yticks(y, [f"{s} *" if r else s
                          for s, r in zip(block.skill, block.self_referential)],
                      fontsize=7)
        _style(ax, f"vs {rival}", "", "pp difference")
    fig.suptitle(f"{focus} minus rival, share_of_skilled, 95% Newcombe "
                 f"intervals — right of zero means {focus} asks more often; "
                 f"* marks one side's own product",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.9)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(keys: list[str], focus: str = "google") -> dict:
    frames = cmp.load_frames(keys)
    longs = cmp.load_long(keys)
    report: dict = {
        "task": "06-competitor-comparison",
        "focus_company": focus,
        "companies": keys,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # -- the gate -----------------------------------------------------------
    gate = cmp.comparability_table(frames)
    _write(gate, TABLES / "company-comparability.csv")
    level = cmp.level_verdict(gate)
    report["comparability"] = {
        "level_identified": level.identified,
        "level_reason": level.reason,
        "common_panel_share": level.detail,
        "min_common_panel_share": cmp.MIN_COMMON_PANEL_SHARE,
    }

    pubs = cmp.common_publishers(frames)
    _write(cmp.common_panel_table(frames, pubs),
           TABLES / "common-publisher-panel.csv")
    by_month = cmp.common_panel_by_period(frames, "month")
    _write(by_month, TABLES / "common-panel-by-month.csv")
    report["common_panel"] = {
        "publishers": pubs,
        "n_publishers_window": len(pubs),
        "max_within_month": int(by_month.n_common_publishers.max()),
        "min_within_month": int(by_month.n_common_publishers.min()),
        "months_with_none": by_month.loc[
            by_month.n_common_publishers == 0, "period"].tolist(),
    }

    # -- volume -------------------------------------------------------------
    panel = cmp.volume_panel_table(frames)
    _write(panel, TABLES / "volume-panel-sensitivity.csv")
    verdicts = cmp.volume_verdict_table(frames)
    _write(verdicts, TABLES / "volume-verdict.csv")
    hoh = cmp.half_over_half(frames)
    _write(hoh, TABLES / "half-over-half.csv")

    art = cmp.collection_artefact_table(frames)
    _write(art, TABLES / "collection-artefact-check.csv")
    corr = cmp.company_correlation(frames)
    _write(corr.reset_index(), TABLES / "company-monthly-correlation.csv")

    rel = cmp.relative_share_table(frames, pubs)
    _write(rel, TABLES / "relative-share-by-half.csv")
    by_pub = cmp.relative_share_by_publisher(frames, pubs)
    _write(by_pub, TABLES / "relative-share-by-publisher.csv")
    rel_verdict = cmp.relative_share_verdict(by_pub, rel)
    _write(rel_verdict, TABLES / "relative-share-verdict.csv")

    report["volume"] = {
        "identified_directions": verdicts.loc[
            verdicts.treatments_agree, "company"].tolist(),
        "not_identified": verdicts.loc[
            ~verdicts.treatments_agree, "company"].tolist(),
        "max_treatment_spread": float(verdicts.spread.max()),
        "half_over_half_all_positive": bool((hoh.raw_pct_change > 0).all()
                                            and (hoh.balanced_pct_change > 0).all()),
        "half_over_half_agrees": bool(hoh.agrees.all()),
        "relative_share": rel.set_index("company").log_share_change.to_dict(),
        "relative_share_verdicts": rel_verdict.set_index("company").verdict.to_dict(),
        "shared_thin_months": art.loc[art.shared_thin_month, "period"].tolist(),
        "publishers_first_period": int(art.distinct_publishers.iloc[0]),
        "publishers_last_period": int(art.distinct_publishers.iloc[-1]),
        "postings_per_publisher_first": float(art.postings_per_publisher.iloc[0]),
        "postings_per_publisher_last": float(art.postings_per_publisher.iloc[-1]),
        "mean_pairwise_correlation": round(float(
            corr.to_numpy()[~np.eye(len(corr), dtype=bool)].mean()), 3),
    }

    # -- role mix and standardisation ---------------------------------------
    mix = cmp.mix_table(frames, "job_function")
    _write(mix, TABLES / "job-function-mix.csv")
    dist_matrix = cmp.mix_distance(mix)
    _write(dist_matrix.reset_index(), TABLES / "job-function-mix-distance.csv")
    std_cov = cmp.standardised_table(frames, "has_any_skill")
    _write(std_cov, TABLES / "skill-coverage-standardised.csv")

    off_diag = dist_matrix.to_numpy()[~np.eye(len(dist_matrix), dtype=bool)]
    report["role_mix"] = {
        "max_pairwise_distance": float(off_diag.max()),
        "min_pairwise_distance": float(off_diag.min()),
        "mean_pairwise_distance": round(float(off_diag.mean()), 4),
        "coverage_crude": std_cov.set_index("company").crude.to_dict(),
        "coverage_standardised": std_cov.set_index("company").standardised.to_dict(),
        "max_mix_effect_pp": round(float(std_cov.mix_effect.abs().max() * 100), 2),
    }

    # -- skills -------------------------------------------------------------
    selfref = cmp.self_reference_table(longs, frames)
    _write(selfref, TABLES / "self-reference-audit.csv")
    report["self_reference"] = {
        "own_product_share": selfref.set_index("company").own_product_share.to_dict(),
        "coverage_inflation_pp": selfref.set_index(
            "company").coverage_inflation_pp.to_dict(),
    }

    share = cmp.skill_share_table(longs, frames)
    _write(share, TABLES / "skill-share-by-company.csv")
    top_skills = (share[share.supported].groupby("skill")
                  .postings_with_skill.sum().sort_values(ascending=False)
                  .head(TOP_SKILLS).index.tolist())
    matrix = cmp.skill_matrix(share).reindex(top_skills).dropna(how="all")

    std_skills = cmp.standardised_skill_table(longs, frames, top_skills)
    _write(std_skills, TABLES / "skill-share-standardised.csv")

    dists = {}
    for key in keys:
        d = cmp.skill_distinctiveness(share, key)
        dists[key] = d
        _write(d, TABLES / f"skill-distinctiveness-{key}.csv")

    rivals = [k for k in keys if k != focus]
    gaps = {}
    for rival in rivals:
        g = cmp.pairwise_skill_gap(share, focus, rival)
        gaps[rival] = g
        _write(g, TABLES / f"{focus}-vs-{rival}-skill-gaps.csv")

    # Stratified check on the gaps the pairwise tables actually called.
    verdict_rows = []
    for rival in rivals:
        called = gaps[rival].loc[gaps[rival].significant, "skill"].tolist()
        for skill in called[:40]:
            within = cmp.within_stratum_shares(longs, frames, skill)
            verdict_rows.append(
                cmp.stratified_company_verdict(within, skill, focus, rival))
    strat = pd.DataFrame(verdict_rows)
    _write(strat, TABLES / "skill-stratified-verdicts.csv")

    robust = cmp.panel_robustness(longs, frames, top_skills, pubs)
    _write(robust, TABLES / "skill-panel-robustness.csv")

    focus_dist = dists[focus]
    sig = focus_dist[focus_dist.significant & focus_dist.supported]
    report["skills"] = {
        "skills_compared": int(share.skill.nunique()),
        "top_skills": top_skills,
        "focus_significant": int(len(sig)),
        "focus_significant_self_referential": int(sig.self_referential.sum()),
        "focus_top_lift": sig.head(5).set_index("skill").log_lift.to_dict(),
        "focus_bottom_lift": sig.tail(5).set_index("skill").log_lift.to_dict(),
        "max_standardisation_effect_pp": round(
            float(std_skills.mix_effect.abs().max() * 100), 2),
        "stratified": (strat.verdict.value_counts().to_dict()
                       if len(strat) else {}),
        "channel_sensitive_shares": int(robust.channel_sensitive.sum()),
    }

    # -- figures ------------------------------------------------------------
    figs = [
        fig_gate(gate, FIGURES / "01-comparability-gate.png"),
        fig_volume_index(panel, verdicts, FIGURES / "02-volume-index-by-company.png"),
        fig_relative_share(rel, by_pub, FIGURES / "03-relative-share.png"),
        fig_collection(art, FIGURES / "04-collection-artefact-check.png"),
        fig_mix(mix, FIGURES / "05-job-function-mix.png"),
        fig_standardised(std_cov, FIGURES / "06-coverage-standardised.png"),
        fig_heatmap(matrix, FIGURES / "07-skill-heatmap.png"),
        fig_distinctiveness(focus_dist, focus,
                            FIGURES / "08-google-distinctiveness.png"),
        fig_pairwise(gaps, focus, FIGURES / "09-google-vs-rivals.png"),
    ]
    report["figures"] = figs

    # -- standing privacy check --------------------------------------------
    offenders = {}
    for path in sorted(TABLES.glob("*.csv")):
        table = pd.read_csv(path)
        found = cmp.personal_data_columns_present(table)
        if found:
            offenders[path.name] = found
    report["privacy"] = {
        "tables_checked": len(list(TABLES.glob("*.csv"))),
        "personal_data_columns_present": offenders,
        "passed": not offenders,
    }

    path = OUT / "task-06-comparison-report.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(f"\ntables  -> {TABLES.relative_to(REPO_ROOT)} "
          f"({len(list(TABLES.glob('*.csv')))} csv)")
    print(f"figures -> {FIGURES.relative_to(REPO_ROOT)} ({len(figs)} png)")
    print(f"report  -> {path.relative_to(REPO_ROOT)}")
    print(f"\nlevel comparison identified: {level.identified} ({level.reason})")
    print(f"volume direction identified for: "
          f"{report['volume']['identified_directions'] or 'no company'}")
    print(f"privacy check passed: {report['privacy']['passed']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus", default="google",
                        help="the specialist's own company; gets the pairwise "
                             "tables and the distinctiveness figure")
    parser.add_argument("--companies", default=None,
                        help="comma-separated keys; defaults to everything that "
                             "passed the feasibility screen")
    args = parser.parse_args()

    if args.companies:
        keys = [k.strip() for k in args.companies.split(",")]
    else:
        screen = pd.read_csv(TABLES / "company-feasibility-screen.csv")
        keys = co.included_companies(screen)
    print(f"comparing {len(keys)} companies: {', '.join(keys)}")
    build(sorted(keys), focus=args.focus)


if __name__ == "__main__":
    main()
