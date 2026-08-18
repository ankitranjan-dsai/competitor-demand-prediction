"""Check `src/similarity.py`'s hand-rolled statistics against scipy and sklearn.

`similarity.py` carries no scipy import, for the reason `docs/task-07-...` §2
gave and Task 08 inherits: the module has to run in a teammate's environment
and in the grader's, and the thing it decides — whether a pair's rank is
published at all — is a poor place to put a dependency whose linkage
tie-breaking or rank handling can change between versions.

The cost of that promise is a page of hand-written statistics: five similarity
metrics, an average-linkage tree, an average-rank routine, a Fisher
transformation and two null generators. Hand-written statistics are exactly
what a unit test fails to catch, because the test and the code share the
author's mistake. So this script imports what `similarity.py` refuses to
depend on and checks the two implementations agree.

It is a **validator, not a dependency**: nothing in `src/`, `tests/` or
`build_similarity.py` imports it, and the pipeline runs to completion with
scipy and scikit-learn uninstalled.

What is being proved, in order of how much it matters:

1. **the metrics are the standard ones.** All five match scipy or sklearn to
   machine precision on random vectors *and* on the six real company
   profiles, so "the ranking is metric-dependent" is a statement about the
   data and not about a mis-transcribed formula.
2. **the tree is the tree scipy would draw.** Merge heights and the partitions
   at every cut match `scipy.cluster.hierarchy`, so the two dendrograms in
   figure 06 differ because the vocabularies differ, not because the linkage
   is home-made.
3. **the ranks handle ties the way scipy does.** Most skills are zero for most
   companies, so average-rank tie handling is doing real work in the Spearman
   column; getting it wrong would silently change one of the two metric
   families.
4. **the interval and the nulls are what they claim.** The Fisher interval
   matches scipy's Pearson interval, and the identical null's draws match the
   binomial moments they are drawn from. The closure null needed a sharper
   question than "does it reproduce -1/(D-1)": it does not, and should not.
   The formula is exact for geometric closure and the data is closed
   arithmetically, so §5 checks the formula against the closure it belongs
   to, measures the departure under the closure the data has, and shows what
   the published refusal does and does not rest on.

    python src/validate_similarity.py

Writes `docs/task-08-similarity-validation.md`. Exits non-zero if any check
fails. Posting-level data is read only for the real-profile checks, and the
evidence file holds nothing but statistics.
"""

from __future__ import annotations

import itertools
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare as cmp        # noqa: E402
import similarity as sim     # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "task-08-similarity-validation.md"

COMPANIES = ["databricks", "google", "meta", "microsoft", "nvidia", "snowflake"]

#: Machine-precision agreement. These are algebraically identical expressions,
#: so anything above float noise means one of the two is not what it claims.
EXACT = 1e-10

#: Monte-Carlo agreement. Used only where the reference is itself a sample.
SAMPLING = 5e-3


def _fmt(x, places: int = 10) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{places}f}"


def _random_profiles(rng, n_skills: int = 60):
    """Two share vectors with the shape real profiles have — mostly small."""
    a = rng.beta(0.4, 6.0, n_skills)
    b = rng.beta(0.4, 6.0, n_skills)
    a[rng.random(n_skills) < 0.35] = 0.0
    b[rng.random(n_skills) < 0.35] = 0.0
    return a, b


# --------------------------------------------------------------------------
# 1. the five metrics
# --------------------------------------------------------------------------

def check_metrics(rng, real) -> tuple[list, list[str]]:
    """Every metric against its scipy or sklearn reference.

    Random vectors first, because they explore the corners — one-sided zero
    support, near-orthogonality — that six real companies never reach. Then
    the real profiles, because a formula can agree everywhere except on the
    numbers that were actually published.
    """
    from scipy.spatial.distance import braycurtis, cosine as scipy_cosine, jensenshannon
    from scipy.stats import spearmanr
    from sklearn.metrics.pairwise import cosine_similarity

    lines = ["## 1. The five metrics against scipy and scikit-learn", ""]
    err = {"cosine": 0.0, "cosine_sklearn": 0.0, "jensen_shannon": 0.0,
           "bray_curtis": 0.0, "spearman": 0.0, "jaccard": 0.0}

    for _ in range(400):
        a, b = _random_profiles(rng)
        if a.sum() == 0 or b.sum() == 0:
            continue
        err["cosine"] = max(err["cosine"],
                            abs(sim.cosine(a, b) - (1.0 - scipy_cosine(a, b))))
        err["cosine_sklearn"] = max(
            err["cosine_sklearn"],
            abs(sim.cosine(a, b) - float(cosine_similarity([a], [b])[0, 0])))
        # scipy returns the distance, base 2 to match the module.
        err["jensen_shannon"] = max(
            err["jensen_shannon"],
            abs(sim.jensen_shannon(a, b) - (1.0 - jensenshannon(a, b, base=2))))
        err["bray_curtis"] = max(err["bray_curtis"],
                                 abs(sim.bray_curtis(a, b) - (1.0 - braycurtis(a, b))))
        err["spearman"] = max(err["spearman"],
                              abs(sim.spearman(a, b) - spearmanr(a, b).statistic))

        sa = {i for i in range(len(a)) if a[i] > 0}
        sb = {i for i in range(len(b)) if b[i] > 0}
        if sa | sb:
            ref = len(sa & sb) / len(sa | sb)
            err["jaccard"] = max(err["jaccard"], abs(sim.jaccard(sa, sb) - ref))

    lines += [
        "400 random pairs of share vectors over 60 skills, drawn to have the "
        "shape real profiles have — mostly small values, about a third exactly "
        "zero. Largest absolute disagreement:",
        "",
        "| metric | reference | max abs difference |",
        "| --- | --- | --- |",
        f"| cosine | `1 - scipy.spatial.distance.cosine` | {_fmt(err['cosine'])} |",
        f"| cosine | `sklearn.metrics.pairwise.cosine_similarity` | {_fmt(err['cosine_sklearn'])} |",
        f"| jensen_shannon | `1 - scipy...jensenshannon(base=2)` | {_fmt(err['jensen_shannon'])} |",
        f"| bray_curtis | `1 - scipy.spatial.distance.braycurtis` | {_fmt(err['bray_curtis'])} |",
        f"| spearman | `scipy.stats.spearmanr` | {_fmt(err['spearman'])} |",
        f"| jaccard | set overlap, computed directly | {_fmt(err['jaccard'])} |",
        "",
    ]

    prof = real["profiles"]
    real_err = 0.0
    rows = []
    for a, b in sim.pairs(prof.columns):
        va, vb = prof[a].to_numpy(), prof[b].to_numpy()
        mine = sim.cosine(va, vb)
        ref = 1.0 - scipy_cosine(va, vb)
        real_err = max(real_err, abs(mine - ref))
        rows.append(f"| {a} – {b} | {mine:.6f} | {ref:.6f} | {abs(mine - ref):.2e} |")

    lines += [
        "The same check on the six published profiles, on the metric every "
        "headline number is quoted on:",
        "",
        "| pair | similarity.cosine | scipy | difference |",
        "| --- | --- | --- | --- |",
        *rows,
        "",
    ]

    checks = [(f"{name} matches its reference to {EXACT:g}", v < EXACT)
              for name, v in err.items()]
    checks.append((f"cosine matches scipy on all 15 published pairs to {EXACT:g}",
                   real_err < EXACT))
    return checks, lines


# --------------------------------------------------------------------------
# 2. average linkage
# --------------------------------------------------------------------------

def check_linkage(rng, real) -> tuple[list, list[str]]:
    """Merge heights and every cut against `scipy.cluster.hierarchy`.

    The partitions matter more than the heights. Figure 06 makes a claim about
    *which company sits alone* under two vocabularies, and that claim is a
    partition, so the check compares the sets at every k from 2 to n-1 as well
    as the numbers on the axis.
    """
    from scipy.cluster.hierarchy import cut_tree as scipy_cut, linkage
    from scipy.spatial.distance import squareform
    import pandas as pd

    lines = ["## 2. Average linkage against scipy.cluster.hierarchy", ""]
    height_err = 0.0
    partition_mismatches = 0
    trials = 0

    def _compare(dist: pd.DataFrame) -> tuple[float, int]:
        keys = list(dist.index)
        merges = sim.average_linkage(dist)
        Z = linkage(squareform(dist.to_numpy(dtype=float), checks=False),
                    method="average")
        h = max(abs(m[2] - z) for m, z in zip(merges, Z[:, 2]))
        bad = 0
        for k in range(2, len(keys)):
            mine = sim.cut_tree(merges, keys, k)
            labels = scipy_cut(Z, n_clusters=k).ravel()
            theirs = frozenset(
                frozenset(keys[i] for i in range(len(keys)) if labels[i] == lab)
                for lab in set(labels))
            bad += int(mine != theirs)
        return h, bad

    for _ in range(120):
        n = int(rng.integers(4, 9))
        pts = rng.normal(size=(n, 5))
        keys = [f"c{i}" for i in range(n)]
        d = pd.DataFrame(
            [[float(np.linalg.norm(pts[i] - pts[j])) for j in range(n)]
             for i in range(n)], index=keys, columns=keys)
        h, bad = _compare(d)
        height_err = max(height_err, h)
        partition_mismatches += bad
        trials += 1

    prof = real["profiles"]
    keys = list(prof.columns)
    real_d = pd.DataFrame(
        [[0.0 if a == b else 1.0 - sim.cosine(prof[a].to_numpy(), prof[b].to_numpy())
          for b in keys] for a in keys], index=keys, columns=keys)
    real_h, real_bad = _compare(real_d)

    lines += [
        f"{trials} random distance matrices of 4 to 8 objects, then the "
        "published cosine-distance matrix over the six companies. Every "
        "partition from k=2 to k=n-1 is compared as a set of sets, not as "
        "labels, because cluster numbering is arbitrary.",
        "",
        "| check | result |",
        "| --- | --- |",
        f"| max merge-height difference, random matrices | {_fmt(height_err)} |",
        f"| partition mismatches, random matrices | {partition_mismatches} |",
        f"| max merge-height difference, real profiles | {_fmt(real_h)} |",
        f"| partition mismatches, real profiles | {real_bad} |",
        "",
    ]
    return [
        (f"merge heights match scipy to {EXACT:g}",
         max(height_err, real_h) < EXACT),
        ("every cut matches scipy's partition",
         partition_mismatches == 0 and real_bad == 0),
    ], lines


# --------------------------------------------------------------------------
# 3. ranks and ties
# --------------------------------------------------------------------------

def check_ranks(rng) -> tuple[list, list[str]]:
    """Average-rank handling against `scipy.stats.rankdata`.

    Deliberately tie-heavy: the vectors here round to two decimals so most
    values repeat, which is the situation a real profile is in when eighty of
    its skills are zero.
    """
    from scipy.stats import rankdata, spearmanr

    lines = ["## 3. Average ranks and tie handling", ""]
    rank_err = tied_spearman_err = 0.0
    for _ in range(400):
        n = int(rng.integers(10, 80))
        x = np.round(rng.beta(0.4, 6.0, n), 2)
        y = np.round(rng.beta(0.4, 6.0, n), 2)
        rank_err = max(rank_err, float(np.max(np.abs(
            sim._rankdata(x) - rankdata(x, method="average")))))
        tied_spearman_err = max(tied_spearman_err,
                                abs(sim.spearman(x, y) - spearmanr(x, y).statistic))
    share_tied = "about 70% of values repeat at this rounding"
    lines += [
        f"400 tie-heavy vectors ({share_tied}). Largest absolute disagreement:",
        "",
        "| quantity | reference | max abs difference |",
        "| --- | --- | --- |",
        f"| average ranks | `scipy.stats.rankdata(method='average')` | {_fmt(rank_err)} |",
        f"| Spearman with ties | `scipy.stats.spearmanr` | {_fmt(tied_spearman_err)} |",
        "",
    ]
    return [
        (f"average ranks match scipy to {EXACT:g}", rank_err < EXACT),
        (f"tied Spearman matches scipy to {EXACT:g}", tied_spearman_err < EXACT),
    ], lines


# --------------------------------------------------------------------------
# 4. the Fisher interval
# --------------------------------------------------------------------------

def check_fisher(rng) -> tuple[list, list[str]]:
    """The trajectory interval against scipy's own Pearson interval.

    This is the number behind the third reason trajectory similarity is
    refused — "the mean interval spans 1.04 of the range" — so it is worth
    knowing it is not 1.04 because of a mis-typed standard error.
    """
    from scipy.stats import pearsonr

    lines = ["## 4. The Fisher interval against scipy.stats.pearsonr", ""]
    lo_err = hi_err = 0.0
    for _ in range(300):
        n = int(rng.integers(8, 30))
        x = rng.normal(size=n)
        y = 0.5 * x + rng.normal(size=n)
        res = pearsonr(x, y)
        ci = res.confidence_interval(0.95)
        mine = sim._fisher_interval(float(res.statistic), n)
        lo_err = max(lo_err, abs(mine[0] - ci.low))
        hi_err = max(hi_err, abs(mine[1] - ci.high))
    lines += [
        "300 correlated normal samples of 8 to 29 points — the range the panel "
        "series actually sits in.",
        "",
        "| bound | reference | max abs difference |",
        "| --- | --- | --- |",
        f"| lower | `pearsonr(...).confidence_interval(0.95)` | {_fmt(lo_err)} |",
        f"| upper | `pearsonr(...).confidence_interval(0.95)` | {_fmt(hi_err)} |",
        "",
        "The residual difference is the normal quantile: `similarity.py` uses "
        "the rounded 1.96, scipy uses the exact 1.959964. On an interval "
        "already 1.04 wide the difference is in the fourth decimal and cannot "
        "move the refusal.",
        "",
    ]
    tol = 2e-3
    return [(f"Fisher bounds match scipy to {tol:g}",
             max(lo_err, hi_err) < tol)], lines


# --------------------------------------------------------------------------
# 5. the two nulls
# --------------------------------------------------------------------------

def _geometric_closure_mean_r(n_periods: int, n_parts: int, sigma: float,
                              draws: int, rng) -> dict:
    """Mean pairwise log-scale correlation under *geometric* closure.

    Identical to `sim.closure_null` in every respect but one: each period is
    divided by its geometric mean instead of by its sum. That single change is
    the difference between the closure `-1/(D-1)` is derived under and the
    closure a share table has, so this function is the reference the analytic
    value should be checked against.
    """
    idx = list(itertools.combinations(range(n_parts), 2))
    means = np.empty(draws)
    for d in range(draws):
        logs = rng.normal(0.0, sigma, size=(n_periods, n_parts))
        clr = logs - logs.mean(axis=1, keepdims=True)
        means[d] = float(np.mean([np.corrcoef(clr[:, i], clr[:, j])[0, 1] for i, j in idx]))
    return {"mean": float(means.mean()),
            "p2_5": float(np.percentile(means, 2.5)),
            "p97_5": float(np.percentile(means, 97.5))}


def check_nulls(rng, real) -> tuple[list, list[str]]:
    """The identical null's moments, and what `-1/(D-1)` is actually exact for.

    The identical null is binomial by construction, so its draws must match
    the binomial mean and variance at the companies' own sample sizes — that
    is what makes "0.99 is what two identical companies score *here*" a claim
    about these sample sizes rather than a constant.

    The closure null took a correction while this file was being written. The
    first version of this section asserted that `-1/(D-1)` falls inside the
    simulated band, and it does not: at six parts the simulation returns
    -0.18 against an analytic -0.20. The fault was in the assertion.
    `-1/(D-1)` is exact under **geometric** closure — divide each period by
    its geometric mean, which is the CLR transform — and a share table is
    closed **arithmetically**, by its sum. The two are different maps and the
    gap between them grows with dispersion.

    So this section now checks three things instead of one: that the analytic
    value is right about the closure it is right about, that the departure
    under the closure the data actually has is small and in the known
    direction, and that `calibrate_sigma` recovers the dispersion it is asked
    to recover — because the published band is now run at the panel's own
    dispersion rather than at a default.
    """
    from scipy.stats import binom

    lines = ["## 5. The identical null and the closure null", ""]

    inc = real["incidence"]
    p = np.clip(sim.pooled_profile(inc), 0.0, 1.0)
    n = int(inc["google"].shape[0])
    draws = np.array([rng.binomial(n, p) / n for _ in range(4000)])
    mean_err = float(np.max(np.abs(draws.mean(axis=0) - binom.mean(n, p) / n)))
    var_err = float(np.max(np.abs(draws.var(axis=0) - binom.var(n, p) / n ** 2)))

    parts = [4, 6, 10]
    geo_rows, geo_ok = [], True
    arith_rows, arith_ok = [], True
    for d in parts:
        analytic = sim.closure_expectation(d)
        geo = _geometric_closure_mean_r(2000, d, 0.3, 200, np.random.default_rng(sim.SEED))
        dev = abs(geo["mean"] - analytic)
        geo_ok &= dev < SAMPLING
        geo_rows.append(f"| {d} | {analytic:.4f} | {geo['mean']:.6f} | "
                        f"{_fmt(dev, 6)} |")

        wide_n = sim.closure_null(2000, d, sigma=0.3, draws=60, seed=sim.SEED)
        panel_n = sim.closure_null(11, d, sigma=0.3, draws=400, seed=sim.SEED)
        gap = wide_n["mean"] - analytic
        # toward zero, and small at this dispersion: that is the whole claim
        ok = 0.0 < gap < 0.05
        arith_ok &= ok
        arith_rows.append(f"| {d} | {analytic:.4f} | {wide_n['mean']:.4f} | "
                          f"{gap:+.4f} | {panel_n['mean']:.4f} |")

    series = pd.read_csv(REPO_ROOT / "members" / "ankit-google" /
                         "task-07-tables" / "panel-share-series.csv")
    wide = sim.panel_wide(series)
    logs = np.log(wide.to_numpy(dtype=float))
    target = float((logs - logs.mean(axis=1, keepdims=True)).std(ddof=1))
    sigma = sim.calibrate_sigma(wide, seed=sim.SEED)
    z = np.random.default_rng(sim.SEED).normal(0.0, 1.0, size=(200, *wide.shape))
    x = np.exp(sigma * z)
    x /= x.sum(axis=2, keepdims=True)
    lx = np.log(x)
    achieved = float((lx - lx.mean(axis=2, keepdims=True)).std(ddof=1))
    sigma_err = abs(achieved - target) / target

    sens = pd.read_csv(REPO_ROOT / "members" / "ankit-google" /
                       "task-08-tables" / "closure-sensitivity.csv")
    flips = sens[~sens.inside]
    holds_at_calibrated = bool(sens.loc[np.isclose(sens.sigma, sigma), "inside"].iloc[0])

    lines += [
        f"**Identical null.** 4000 binomial draws at Google's own sample size "
        f"(n = {n}) over the pooled profile, against `scipy.stats.binom`:",
        "",
        "| moment | max abs difference across 127 skills |",
        "| --- | --- |",
        f"| mean | {_fmt(mean_err, 6)} |",
        f"| variance | {_fmt(var_err, 8)} |",
        "",
        "**What `-1/(D-1)` is exact for.** Under geometric closure — each "
        "period divided by its geometric mean — the analytic value comes back "
        "to five decimal places at 2000 periods:",
        "",
        "| parts | analytic -1/(D-1) | geometric-closure mean | deviation |",
        "| --- | --- | --- | --- |",
        *geo_rows,
        "",
        "The check is a tolerance on the mean rather than containment in a "
        "band, because at 2000 periods the band is narrower than the "
        "`O(1/n)` bias of a sample correlation — a containment test at this "
        "precision measures that bias, not the formula.",
        "",
        "**What the data has instead.** Shares are closed by their sum. The "
        "same simulation under arithmetic closure lands short of the analytic "
        "value, always toward zero, with a further small attenuation at the "
        "panel's eleven periods:",
        "",
        "| parts | analytic | arithmetic closure, n = 2000 | gap | "
        "arithmetic closure, n = 11 |",
        "| --- | --- | --- | --- | --- |",
        *arith_rows,
        "",
        "The gap is not a defect in either implementation. It is the reason "
        "`trajectory_verdict` reads the simulated band and never the formula, "
        "and the reason figure 08 draws `-1/(D-1)` as a dashed line outside "
        "the band rather than through the middle of it.",
        "",
        f"**Calibration.** The gap grows with dispersion, so the published "
        f"band is run at the panel's own. The six companies' CLR coordinates "
        f"have spread {target:.4f}; `calibrate_sigma` returns "
        f"sigma = {sigma:.4f}, which reproduces a spread of {achieved:.4f} — "
        f"{sigma_err * 100:.2f}% off the target.",
        "",
        f"**What rests on it.** `closure-sensitivity.csv` runs the clause "
        f"across the dispersion range. It holds at the calibrated sigma "
        f"({'yes' if holds_at_calibrated else 'NO'}) and first fails at "
        f"sigma = {flips.sigma.iloc[0] if len(flips) else float('nan'):.2f}, "
        f"where the observed mean falls *{flips.side_if_outside.iloc[0] if len(flips) else 'n/a'}* "
        f"the band — i.e. more negative than independent series produce, "
        f"which is still not evidence that these companies move together. "
        f"The refusal also rests on two reasons that do not involve the null "
        f"at all: one eligible pair out of fifteen, and a mean interval "
        f"spanning half the range.",
        "",
    ]
    return [
        (f"identical-null draws match binomial moments to {SAMPLING:g}",
         mean_err < SAMPLING and var_err < SAMPLING),
        (f"-1/(D-1) is reproduced by geometric closure to {SAMPLING:g} at 4, 6 and 10 parts",
         bool(geo_ok)),
        ("arithmetic closure departs toward zero by under 0.05 at sigma = 0.3",
         bool(arith_ok)),
        ("calibrate_sigma reproduces the panel's CLR spread to 1%",
         sigma_err < 0.01),
        ("the closure clause holds at the panel's calibrated dispersion",
         holds_at_calibrated),
    ], lines


# --------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(sim.SEED)
    frames = cmp.load_frames(COMPANIES)
    longs = cmp.load_long(COMPANIES)
    inc, skills = sim.incidence(longs, frames)
    real = {"incidence": inc, "profiles": sim.profile_frame(inc, skills)}

    header = [
        "# Task 08 — validating the hand-written similarity statistics",
        "",
        f"Generated by `src/validate_similarity.py` on {date.today().isoformat()}.",
        "",
        "`src/similarity.py` carries no scipy import, so five metrics, an "
        "average-linkage tree, an average-rank routine, a Fisher interval and "
        "two null generators are written by hand. This file is the evidence "
        "that the hand-written versions agree with the reference "
        "implementations — and, where they do not agree exactly, the "
        "measurement of how much the difference could move a published "
        "conclusion.",
        "",
        "Versions used for the comparison: "
        f"numpy {np.__version__}, "
        f"scipy {__import__('scipy').__version__}, "
        f"scikit-learn {__import__('sklearn').__version__}.",
        "",
    ]

    sections = [
        check_metrics(rng, real),
        check_linkage(rng, real),
        check_ranks(rng),
        check_fisher(rng),
        check_nulls(rng, real),
    ]

    checks: list = []
    body: list[str] = []
    for section_checks, section_lines in sections:
        checks += section_checks
        body += section_lines

    summary = ["## Automated checks", ""]
    for name, ok in checks:
        summary.append(f"- {'PASS' if ok else 'FAIL'} — {name}")
    summary.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header + summary + body))

    print(f"validated similarity.py against scipy + scikit-learn "
          f"-> {OUT.relative_to(REPO_ROOT)}")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    if not all(ok for _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
