"""Task 08 — company similarity scoring.

Shared, company-agnostic layer. Task 06 asked whether two companies can be
compared at all; Task 07 asked whether one company's series can be extended
forward. Task 08 asks the question a reader reaches for first — *which of
these companies want the same things?* — and the honest answer needs three
pieces of scaffolding that a plain cosine does not have.

**A similarity score has no zero.** Two companies drawn from an identical
profile do not score 1.0 at these sample sizes; they score about 0.99, and two
unrelated profiles over the same skill vocabulary score about 0.13, not 0.
"0.62" means nothing until both ends of that ruler are published, which is what
`calibration_table` does.

**The ranking depends on the metric, and the metrics here fall into two
families that do not agree.** Prevalence-weighted measures (cosine, Jensen –
Shannon, Bray – Curtis) ask "do these companies demand the same skills in the
same proportions"; rank and set measures (Spearman, Jaccard) ask "do they have
the same skills at all". `metric_concordance` measures the disagreement rather
than hiding it behind one number.

**Own products are the largest single lever on the answer.** Task 06 §7
established that a vendor's own product name is real information and must be
flagged, never filtered. In a similarity score that flag has to become a
published sensitivity, because dropping own products moves pairs by up to 0.30
and changes which company is the outlier.

Everything here runs on pandas, numpy and `math`, in keeping with Tasks 06 and
07: any reviewer can rebuild every published number without matching a solver
version. `src/validate_similarity.py` checks the hand-written statistics
against scipy and scikit-learn.

    python src/build_similarity.py
    python src/validate_similarity.py
    python -m pytest tests/ -q
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import compare as cmp
import forecast as fc

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = cmp.PROCESSED

PUBLISHER_COL = cmp.PUBLISHER_COL
SKILL_EXCLUDED_FUNCTIONS = cmp.SKILL_EXCLUDED_FUNCTIONS

#: Inherited whole from Task 06. A country column would compare aggregator
#: footprints, and ``share_of_all`` is the wrong skill denominator; a
#: similarity table built on either would launder both mistakes into a
#: heatmap, which is harder to argue with than a table.
FORBIDDEN_COLUMNS = cmp.FORBIDDEN_COLUMNS

#: The metric every headline number in this task is quoted on. Chosen because
#: it matches the question the brief asks — how similar is the *demand* these
#: companies express — which weights a skill by how often it is actually
#: demanded. It is a choice among defensible alternatives and it changes the
#: answer, so `pair_table` publishes all five and §3 of the method document
#: states the choice before the ranking.
PRIMARY_METRIC = "cosine"

#: Bootstrap draws for pair intervals, rank stability and cluster support.
#: 600 is enough to resolve a rank band to about ±0.02 in probability, which
#: is finer than any claim this task makes.
N_BOOTSTRAP = 600

#: Draws per pair for each of the two nulls. The nulls are means, not tails,
#: so they converge much faster than the bootstrap.
N_NULL = 300

#: Fixed so the committed tables rebuild bit-for-bit.
SEED = 20260818

#: A pair's rank is *identified* only if the bootstrap keeps it in place this
#: often. Declared before the ranking was computed; at 0.90 exactly two of
#: this repo's fifteen pairs qualify at the top and two at the bottom, and the
#: middle eleven are one band. Lowering it would manufacture an ordering.
RANK_STABILITY_FLOOR = 0.90

#: Minimum observed periods for a trajectory pair, on top of Task 07's own
#: forecastability gate. Eleven points is already too few; this only stops a
#: shorter series reaching the correlation at all.
MIN_TRAJECTORY_PERIODS = 8


# ---------------------------------------------------------------------------
# Profiles: the object being compared
# ---------------------------------------------------------------------------


def incidence(longs: dict[str, pd.DataFrame],
              frames: dict[str, pd.DataFrame]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Posting × skill 0/1 matrices on a shared skill vocabulary.

    The rows are *postings*, not skill mentions, and the denominator is
    Task 06's `skill_denominator` — skilled postings with Facilities /
    Operations removed. Both are inherited: Google's facilities roles are true
    negatives and its facilities share is ten times Snowflake's, so a
    denominator that kept them would make the similarity matrix partly a map
    of who runs data centres.

    Posting-level rather than aggregate because the bootstrap resamples
    postings — a resample of shares would put the sampling unit at the wrong
    level and give intervals that are far too narrow.
    """
    vocab: set[str] = set()
    per: dict[str, pd.DataFrame] = {}
    for key in sorted(frames):
        denom = cmp.skill_denominator(frames[key])
        ids = pd.Index(denom.job_id.unique())
        sub = longs[key][longs[key].job_id.isin(set(ids))]
        if sub.empty:
            per[key] = pd.DataFrame(index=ids)
            continue
        mat = pd.crosstab(sub.job_id, sub.skill).reindex(index=ids, fill_value=0)
        per[key] = (mat > 0).astype(np.int8)
        vocab |= set(mat.columns)
    skills = sorted(vocab)
    out = {k: v.reindex(columns=skills, fill_value=0).to_numpy(dtype=np.int8)
           for k, v in per.items()}
    return out, skills


def profiles(inc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mean incidence per company — ``share_of_skilled`` as a vector."""
    return {k: (v.mean(axis=0) if len(v) else np.zeros(v.shape[1]))
            for k, v in inc.items()}


def profile_frame(inc: dict[str, np.ndarray], skills: list[str]) -> pd.DataFrame:
    """The profiles as a skill × company frame, for tables and figures."""
    prof = profiles(inc)
    return pd.DataFrame({k: prof[k] for k in sorted(prof)}, index=pd.Index(skills, name="skill"))


def concept_skills(longs: dict[str, pd.DataFrame]) -> list[str]:
    """Skills Task 04 flagged ``is_concept`` — practices, not products."""
    parts = [l[["skill", "is_concept"]] for l in longs.values() if len(l)]
    if not parts:
        return []
    flags = pd.concat(parts).drop_duplicates("skill").set_index("skill").is_concept
    return sorted(flags[flags.astype(bool)].index)


def vendor_skills(relations=("own_product", "origin")) -> dict[str, set[str]]:
    """Per company, the skills that are its own — Task 06's `VENDOR_SKILLS`.

    ``origin`` is separable from ``own_product`` and the caller says which it
    wants: TensorFlow and Go originate at Google without being sold by it, so
    a sweep that drops them is answering a different question from one that
    drops BigQuery.
    """
    return {key: {skill for skill, rel in mapping.items() if rel in relations}
            for key, mapping in cmp.VENDOR_SKILLS.items()}


# ---------------------------------------------------------------------------
# Metrics. Five, because one would be a claim this data does not support.
# ---------------------------------------------------------------------------


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Uncentred correlation of two share vectors.

    Dominated by the skills both companies ask for often, which is the
    intended behaviour and also its main limitation: 68% of a typical
    numerator here is five skills. `support_sensitivity` shows what happens
    when that tail is removed on purpose.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = math.sqrt(float(a @ a)), math.sqrt(float(b @ b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def jensen_shannon(a: np.ndarray, b: np.ndarray) -> float:
    """``1 − √JSD`` on the L1-normalised profiles, base 2.

    The square root of the divergence is a true metric bounded in [0, 1], so
    the similarity is bounded in [0, 1] too and 0.5 means the same thing for
    every pair. Undefined support is not a problem: the mixture is positive
    wherever either profile is.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sa, sb = a.sum(), b.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    p, q = a / sa, b / sb
    m = 0.5 * (p + q)
    jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(1.0 - math.sqrt(max(jsd, 0.0)))


def bray_curtis(a: np.ndarray, b: np.ndarray) -> float:
    """``1 − Σ|a−b| / Σ(a+b)``. Abundance-weighted, no normalisation first.

    Kept alongside Jensen – Shannon because it does *not* normalise the
    profiles: a company whose postings simply list more skills stays
    different, where JS would rescale that away. The two agree here
    (ρ = 0.96), which is worth knowing.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    total = float(a.sum() + b.sum())
    if total == 0:
        return float("nan")
    return float(1.0 - np.abs(a - b).sum() / total)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks. Ties are the common case here — most skills are 0."""
    return pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation over the whole vocabulary.

    Treats a 3-posting skill and a 300-posting skill as one observation each,
    which is why it disagrees with cosine so completely (ρ = −0.03 across this
    repo's fifteen pairs). That is not a defect in either — they answer
    different questions, and §3 of the method document says which is being
    asked.
    """
    ra, rb = _rankdata(a), _rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) @ (rb - rb.mean())) / (len(ra) * sa * sb))


def jaccard(a: set, b: set) -> float:
    """Set overlap. Used on the *supported* skills, never on presence.

    Presence would make Jaccard a statement about vocabulary size; the support
    floor (Task 06's ``MIN_CELL``) makes it a statement about which skills each
    company demands often enough to measure.
    """
    union = a | b
    return float(len(a & b) / len(union)) if union else float("nan")


#: Name → function on two share vectors. `jaccard_supported` is handled
#: separately because it consumes sets, not vectors.
METRICS = {
    "cosine": cosine,
    "jensen_shannon": jensen_shannon,
    "bray_curtis": bray_curtis,
    "spearman": spearman,
}

#: The two families a metric can belong to. Not a claim that the split is
#: clean — `metric_concordance` measures how clean it is on the data in hand,
#: and on this repo's six companies the within-family correlations run 0.58 to
#: 0.96 while the cross-family ones run -0.04 to 0.58, so the families overlap
#: at the edges and separate sharply at the extremes. The label is here so a
#: reader can see *which question* a metric answers before reading its
#: ordering as the ordering.
METRIC_FAMILY = {
    "cosine": "prevalence_weighted",
    "jensen_shannon": "prevalence_weighted",
    "bray_curtis": "prevalence_weighted",
    "spearman": "rank_or_set",
    "jaccard_supported": "rank_or_set",
}

ALL_METRICS = list(METRIC_FAMILY)


def pairs(keys) -> list[tuple[str, str]]:
    """Unordered company pairs, sorted, so every table has the same row order."""
    return list(itertools.combinations(sorted(keys), 2))


def similarity_matrix(prof: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """Square company × company matrix for one metric. The heatmap input."""
    fn = METRICS[metric]
    keys = list(prof.columns)
    out = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for a in keys:
        for b in keys:
            out.loc[a, b] = 1.0 if a == b else fn(prof[a].to_numpy(), prof[b].to_numpy())
    return out.rename_axis(index="company", columns="company")


def supported_sets(longs: dict[str, pd.DataFrame], frames: dict[str, pd.DataFrame],
                   min_postings: int = cmp.MIN_CELL) -> dict[str, set[str]]:
    """Per company, the skills clearing the support floor. Jaccard's input."""
    share = cmp.skill_share_table(longs, frames, min_postings=min_postings)
    return {key: set(share[(share.company == key) & share.supported].skill)
            for key in sorted(frames)}


def pair_table(prof: pd.DataFrame, sets: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """One row per company pair, every metric, plus each metric's rank.

    The ranks are the point. A similarity table with five value columns looks
    like agreement; the rank columns are where the disagreement becomes
    visible without the reader doing arithmetic.
    """
    rows = []
    for a, b in pairs(prof.columns):
        row = {"company_a": a, "company_b": b}
        for name, fn in METRICS.items():
            row[name] = round(fn(prof[a].to_numpy(), prof[b].to_numpy()), 4)
        if sets is not None:
            row["jaccard_supported"] = round(jaccard(sets[a], sets[b]), 4)
        rows.append(row)
    out = pd.DataFrame(rows)
    present = [m for m in ALL_METRICS if m in out.columns]
    for m in present:
        out[f"rank_{m}"] = out[m].rank(ascending=False, method="min").astype(int)
    ranks = out[[f"rank_{m}" for m in present]]
    out["rank_spread"] = (ranks.max(axis=1) - ranks.min(axis=1)).astype(int)
    return out.sort_values(PRIMARY_METRIC, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Calibration: the two ends of the ruler
# ---------------------------------------------------------------------------


def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(SEED if seed is None else seed)


def pooled_profile(inc: dict[str, np.ndarray]) -> np.ndarray:
    """Sector-wide share of each skill — the profile of a company with no identity."""
    total = sum(int(v.shape[0]) for v in inc.values())
    if total == 0:
        return np.zeros(0)
    hits = sum(v.sum(axis=0) for v in inc.values())
    return np.asarray(hits, dtype=float) / total


def null_identical(inc: dict[str, np.ndarray], metric: str = PRIMARY_METRIC,
                   draws: int = N_NULL, seed: int | None = None) -> dict[tuple[str, str], np.ndarray]:
    """Similarity between two companies that are the *same* company.

    Each pair is redrawn as binomial samples from the pooled profile at the
    two companies' own sample sizes, so the null carries their sampling noise
    and nothing else. This is the number a reader is implicitly comparing
    against when they call 0.92 "very similar", and it is 0.99.
    """
    rng = _rng(seed)
    fn = METRICS[metric]
    p = np.clip(pooled_profile(inc), 0.0, 1.0)
    n = {k: int(v.shape[0]) for k, v in inc.items()}
    out = {}
    for a, b in pairs(inc):
        vals = np.empty(draws)
        for i in range(draws):
            ka = rng.binomial(n[a], p) / n[a] if n[a] else p * 0
            kb = rng.binomial(n[b], p) / n[b] if n[b] else p * 0
            vals[i] = fn(ka, kb)
        out[(a, b)] = vals
    return out


def null_unrelated(prof: pd.DataFrame, metric: str = PRIMARY_METRIC,
                   draws: int = N_NULL, seed: int | None = None) -> dict[tuple[str, str], np.ndarray]:
    """Similarity between two companies with no shared structure.

    Each company keeps its own set of skill shares and loses which skill they
    belong to. That holds the marginal shape of both profiles fixed and
    destroys only the alignment, so the null answers "how much of this score
    is the two companies agreeing about *which* skills" rather than "how much
    is the shape of a skill distribution".
    """
    rng = _rng(seed)
    fn = METRICS[metric]
    out = {}
    for a, b in pairs(prof.columns):
        va, vb = prof[a].to_numpy(), prof[b].to_numpy()
        vals = np.empty(draws)
        for i in range(draws):
            vals[i] = fn(rng.permutation(va), rng.permutation(vb))
        out[(a, b)] = vals
    return out


def calibration_table(prof: pd.DataFrame, inc: dict[str, np.ndarray],
                      metric: str = PRIMARY_METRIC, draws: int = N_NULL,
                      seed: int | None = None) -> pd.DataFrame:
    """Observed similarity against both nulls, and the score between them.

    ``calibrated = (observed − unrelated) / (identical − unrelated)`` puts 0 at
    "no shared structure" and 1 at "the same company measured twice". It is
    the only similarity number in this task that can be read without also
    reading its ruler, and `distinct` is the finding: an observed value below
    the identical null's 2.5th percentile means the two companies really do
    differ, which at these sample sizes is a low bar and every pair clears it.
    """
    fn = METRICS[metric]
    ident = null_identical(inc, metric, draws, seed)
    unrel = null_unrelated(prof, metric, draws, seed)
    rows = []
    for a, b in pairs(prof.columns):
        obs = fn(prof[a].to_numpy(), prof[b].to_numpy())
        i_draws, u_draws = ident[(a, b)], unrel[(a, b)]
        i_mean, u_mean = float(np.mean(i_draws)), float(np.mean(u_draws))
        span = i_mean - u_mean
        rows.append({
            "company_a": a, "company_b": b, "metric": metric,
            "observed": round(obs, 4),
            "null_identical": round(i_mean, 4),
            "null_identical_p2_5": round(float(np.percentile(i_draws, 2.5)), 4),
            "null_unrelated": round(u_mean, 4),
            "null_unrelated_p97_5": round(float(np.percentile(u_draws, 97.5)), 4),
            "calibrated": round((obs - u_mean) / span, 4) if span else float("nan"),
            "distinct": bool(obs < float(np.percentile(i_draws, 2.5))),
            "above_unrelated": bool(obs > float(np.percentile(u_draws, 97.5))),
        })
    return pd.DataFrame(rows).sort_values("calibrated", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bootstrap: which part of the ranking is real
# ---------------------------------------------------------------------------


def bootstrap_pairs(inc: dict[str, np.ndarray], metric: str = PRIMARY_METRIC,
                    draws: int = N_BOOTSTRAP, seed: int | None = None) -> pd.DataFrame:
    """Percentile intervals and rank stability for every pair.

    Postings are resampled with replacement *within* each company, which is
    the sampling unit that actually varies — a company's postings are a draw
    from what it might have posted, its skill shares are not an independent
    draw of anything.

    ``rank_stability`` is the fraction of draws in which the pair keeps its
    observed rank. It is the column that decides what this task publishes:
    a ranking whose middle is a coin flip is a list, not a result.
    """
    rng = _rng(seed)
    fn = METRICS[metric]
    keys = sorted(inc)
    pr = profiles(inc)
    obs = {(a, b): fn(pr[a], pr[b]) for a, b in pairs(keys)}
    plist = pairs(keys)
    observed_rank = (pd.Series({p: obs[p] for p in plist})
                     .rank(ascending=False, method="min").astype(int))

    vals = np.empty((draws, len(plist)))
    rank_draws = np.empty((draws, len(plist)))
    for d in range(draws):
        boot = {}
        for k in keys:
            n = inc[k].shape[0]
            if n == 0:
                boot[k] = np.zeros(inc[k].shape[1])
                continue
            boot[k] = inc[k][rng.integers(0, n, n)].mean(axis=0)
        row = np.array([fn(boot[a], boot[b]) for a, b in plist])
        vals[d] = row
        rank_draws[d] = pd.Series(row).rank(ascending=False, method="min").to_numpy()

    rows = []
    for j, (a, b) in enumerate(plist):
        col = vals[:, j]
        rk = rank_draws[:, j]
        r_obs = int(observed_rank[(a, b)])
        rows.append({
            "company_a": a, "company_b": b, "metric": metric,
            "observed": round(obs[(a, b)], 4),
            "ci_low": round(float(np.percentile(col, 2.5)), 4),
            "ci_high": round(float(np.percentile(col, 97.5)), 4),
            "rank": r_obs,
            "rank_low": int(np.percentile(rk, 2.5)),
            "rank_high": int(np.percentile(rk, 97.5)),
            "rank_stability": round(float((rk == r_obs).mean()), 4),
            "draws": draws,
        })
    out = pd.DataFrame(rows)
    out["rank_identified"] = out.rank_stability >= RANK_STABILITY_FLOOR
    return out.sort_values("rank").reset_index(drop=True)


def rank_tiers(boot: pd.DataFrame) -> pd.DataFrame:
    """Collapse the ranking to the tiers the intervals actually separate.

    Two pairs are in the same tier when their bootstrap rank intervals
    overlap. Fifteen pairs with fifteen ranks implies fourteen distinctions;
    this reports how many of them survive, which here is far fewer.
    """
    b = boot.sort_values("rank").reset_index(drop=True)
    tier, tiers, hi = 1, [], None
    for row in b.itertuples():
        if hi is None or row.rank_low > hi:
            if hi is not None:
                tier += 1
            hi = row.rank_high
        else:
            hi = max(hi, row.rank_high)
        tiers.append(tier)
    b["tier"] = tiers
    return b[["company_a", "company_b", "observed", "rank",
              "rank_low", "rank_high", "rank_stability", "tier"]]


# ---------------------------------------------------------------------------
# Sensitivity: the four levers that move the answer
# ---------------------------------------------------------------------------


def metric_concordance(pt: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation between every pair of metrics, plus their families.

    The diagonal block structure is the finding. Within a family the metrics
    are near-substitutes; across the divide they carry almost no common
    information, and a report that quotes one without naming it is quoting a
    coin flip between two orderings.
    """
    present = [m for m in ALL_METRICS if f"rank_{m}" in pt.columns]
    ranks = pt[[f"rank_{m}" for m in present]]
    ranks.columns = present
    corr = ranks.corr(method="spearman")
    rows = []
    for a, b in itertools.combinations(present, 2):
        rows.append({
            "metric_a": a, "metric_b": b,
            "family_a": METRIC_FAMILY[a], "family_b": METRIC_FAMILY[b],
            "same_family": METRIC_FAMILY[a] == METRIC_FAMILY[b],
            "rank_correlation": round(float(corr.loc[a, b]), 4),
        })
    return pd.DataFrame(rows).sort_values("rank_correlation", ascending=False).reset_index(drop=True)


def _drop(prof: pd.DataFrame, skills) -> pd.DataFrame:
    keep = [s for s in prof.index if s not in set(skills)]
    return prof.loc[keep]


def vendor_sensitivity(prof: pd.DataFrame, metric: str = PRIMARY_METRIC,
                       relations=("own_product", "origin")) -> pd.DataFrame:
    """Similarity with and without each company's own products.

    Three variants, and the middle one is deliberately awkward:

    ``all_skills``
        every skill, own products included. What a naive cosine returns.
    ``pair_products_dropped``
        drop only the two companies in the pair. The most defensible *pairwise*
        number and the least usable matrix — the vocabulary differs per pair,
        so it can be ranked but must not be clustered or drawn as a network.
    ``all_products_dropped``
        drop every company's products, so all pairs share one vocabulary. The
        variant the clustering sweep uses.

    Task 06 §7's rule holds: nothing is filtered from the published data. This
    is a sensitivity, and both columns ship.
    """
    own = vendor_skills(relations)
    every = set().union(*own.values()) if own else set()
    fn = METRICS[metric]
    rows = []
    for a, b in pairs(prof.columns):
        base = fn(prof[a].to_numpy(), prof[b].to_numpy())
        pair_drop = _drop(prof, own.get(a, set()) | own.get(b, set()))
        all_drop = _drop(prof, every)
        rows.append({
            "company_a": a, "company_b": b, "metric": metric,
            "all_skills": round(base, 4),
            "pair_products_dropped": round(fn(pair_drop[a].to_numpy(), pair_drop[b].to_numpy()), 4),
            "all_products_dropped": round(fn(all_drop[a].to_numpy(), all_drop[b].to_numpy()), 4),
            "skills_all": int(len(prof)),
            "skills_pair_dropped": int(len(pair_drop)),
            "skills_all_dropped": int(len(all_drop)),
        })
    out = pd.DataFrame(rows)
    out["delta_all_products"] = (out.all_products_dropped - out.all_skills).round(4)
    for col in ("all_skills", "pair_products_dropped", "all_products_dropped"):
        out[f"rank_{col}"] = out[col].rank(ascending=False, method="min").astype(int)
    out["rank_move"] = (out.rank_all_products_dropped - out.rank_all_skills).astype(int)
    return out.sort_values("all_skills", ascending=False).reset_index(drop=True)


def mix_sensitivity(longs: dict[str, pd.DataFrame], frames: dict[str, pd.DataFrame],
                    skills=None, metric: str = PRIMARY_METRIC,
                    by: str = "job_function") -> pd.DataFrame:
    """Similarity on crude shares against similarity on mix-standardised ones.

    Task 06 §4 found role mix worth up to 17.5 points on a *single* skill's
    share, so the expectation going in was that it would dominate a whole-
    profile similarity too. It does not, and the negative result is why this
    function exists: mix moves individual coordinates in offsetting directions
    and the direction of the vector survives.
    """
    share = cmp.skill_share_table(longs, frames)
    if skills is None:
        skills = sorted(share[share.supported].skill.unique())
    std = cmp.standardised_skill_table(longs, frames, skills, by=by)
    crude = std.pivot_table(index="skill", columns="company",
                            values="crude_share_of_skilled").fillna(0.0)
    adj = std.pivot_table(index="skill", columns="company",
                          values="standardised_share").fillna(0.0)
    cover = std.groupby("company").weight_covered.min()
    fn = METRICS[metric]
    rows = []
    for a, b in pairs(crude.columns):
        c = fn(crude[a].to_numpy(), crude[b].to_numpy())
        s = fn(adj[a].to_numpy(), adj[b].to_numpy())
        rows.append({
            "company_a": a, "company_b": b, "metric": metric,
            "crude": round(c, 4), "standardised": round(s, 4),
            "mix_effect": round(c - s, 4),
            "skills_used": int(len(crude)),
            "weight_covered_min": round(float(min(cover.get(a, np.nan),
                                                  cover.get(b, np.nan))), 4),
        })
    out = pd.DataFrame(rows)
    out["rank_crude"] = out.crude.rank(ascending=False, method="min").astype(int)
    out["rank_standardised"] = out.standardised.rank(ascending=False, method="min").astype(int)
    out["rank_move"] = (out.rank_standardised - out.rank_crude).astype(int)
    return out.sort_values("crude", ascending=False).reset_index(drop=True)


def support_sensitivity(prof: pd.DataFrame, sets: dict[str, set[str]],
                        metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """Whole vocabulary against the skills supported in every company.

    The core set is what a reader who distrusts small cells would use. It is
    a different question — "do they agree about the skills everyone measures"
    — and it produces a different ordering, so both ship.
    """
    core = sorted(set.intersection(*sets.values())) if sets else []
    fn = METRICS[metric]
    rows = []
    for a, b in pairs(prof.columns):
        full = fn(prof[a].to_numpy(), prof[b].to_numpy())
        sub = (fn(prof.loc[core, a].to_numpy(), prof.loc[core, b].to_numpy())
               if core else float("nan"))
        rows.append({
            "company_a": a, "company_b": b, "metric": metric,
            "all_skills": round(full, 4), "core_skills": round(sub, 4),
            "n_all": int(len(prof)), "n_core": len(core),
        })
    out = pd.DataFrame(rows)
    out["rank_all"] = out.all_skills.rank(ascending=False, method="min").astype(int)
    out["rank_core"] = out.core_skills.rank(ascending=False, method="min").astype(int)
    out["rank_move"] = (out.rank_core - out.rank_all).astype(int)
    return out.sort_values("all_skills", ascending=False).reset_index(drop=True)


def numerator_contribution(prof: pd.DataFrame, groups: dict[str, list[str]],
                           top_k: int = 5) -> pd.DataFrame:
    """What share of the cosine numerator each named group of skills carries.

    Written to settle Task 04's two predictions about this task rather than to
    argue with them: a skill's contribution to a cosine on *share* vectors is
    the product of two shares, so a skill almost nobody asks for contributes
    almost nothing however many columns it occupies. ``share_top{k}`` is the
    other end of the same arithmetic — the few skills that do carry the score —
    and the correction register cites both columns.
    """
    rows = []
    for a, b in pairs(prof.columns):
        prod = prof[a].to_numpy() * prof[b].to_numpy()
        total = float(prod.sum())
        row = {"company_a": a, "company_b": b, "numerator": round(total, 6)}
        for name, members in groups.items():
            mask = np.array([s in set(members) for s in prof.index])
            row[f"share_{name}"] = round(float(prod[mask].sum() / total), 6) if total else float("nan")
            row[f"n_{name}"] = int(mask.sum())
        top = np.sort(prod)[::-1][:top_k]
        row[f"share_top{top_k}"] = round(float(top.sum() / total), 6) if total else float("nan")
        row[f"top{top_k}_skills"] = ", ".join(
            pd.Series(prod, index=prof.index).sort_values(ascending=False).head(top_k).index)
        rows.append(row)
    return pd.DataFrame(rows)


def group_removal(prof: pd.DataFrame, skills, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """Similarity before and after removing a named group of skills."""
    fn = METRICS[metric]
    sub = _drop(prof, skills)
    rows = []
    for a, b in pairs(prof.columns):
        before = fn(prof[a].to_numpy(), prof[b].to_numpy())
        after = fn(sub[a].to_numpy(), sub[b].to_numpy())
        rows.append({"company_a": a, "company_b": b, "metric": metric,
                     "with_group": round(before, 4), "without_group": round(after, 4),
                     "delta": round(after - before, 4)})
    out = pd.DataFrame(rows)
    out["rank_with"] = out.with_group.rank(ascending=False, method="min").astype(int)
    out["rank_without"] = out.without_group.rank(ascending=False, method="min").astype(int)
    out["rank_move"] = (out.rank_without - out.rank_with).astype(int)
    return out.sort_values("with_group", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Structure: dendrograms and networks, with their stability
# ---------------------------------------------------------------------------


def average_linkage(dist: pd.DataFrame) -> list[tuple[frozenset, frozenset, float]]:
    """UPGMA on a square distance frame, returned as merges.

    Hand-rolled rather than scipy so the whole task rebuilds on pandas and
    numpy; `validate_similarity.py` checks it against
    `scipy.cluster.hierarchy.linkage`. Average linkage because single linkage
    chains through the one high-similarity pair this data has and complete
    linkage is decided entirely by the two most distant members — with six
    objects both failure modes are one pair wide.
    """
    keys = list(dist.index)
    clusters = {frozenset([k]): [k] for k in keys}
    merges = []
    d = {(frozenset([a]), frozenset([b])): float(dist.loc[a, b])
         for a, b in itertools.combinations(keys, 2)}
    while len(clusters) > 1:
        (ca, cb), best = min(d.items(), key=lambda kv: (kv[1], sorted(kv[0][0]), sorted(kv[0][1])))
        new = ca | cb
        merges.append((ca, cb, best))
        for other in list(clusters):
            if other in (ca, cb):
                continue
            na, nb = len(ca), len(cb)
            dac = d.get((ca, other), d.get((other, ca)))
            dbc = d.get((cb, other), d.get((other, cb)))
            d[(new, other)] = (na * dac + nb * dbc) / (na + nb)
        for k in list(d):
            if ca in k or cb in k:
                del d[k]
        del clusters[ca]
        del clusters[cb]
        clusters[new] = sorted(new)
    return merges


def cut_tree(merges, keys, k: int) -> frozenset:
    """The partition into ``k`` clusters, as a set of frozensets."""
    clusters = {frozenset([x]) for x in keys}
    for ca, cb, _ in merges:
        if len(clusters) <= k:
            break
        clusters.discard(ca)
        clusters.discard(cb)
        clusters.add(ca | cb)
    return frozenset(clusters)


def cluster_support(inc: dict[str, np.ndarray], metric: str = PRIMARY_METRIC,
                    cuts=(2, 3, 4), draws: int = N_BOOTSTRAP,
                    seed: int | None = None,
                    drop_skills=None, skills: list[str] | None = None) -> pd.DataFrame:
    """Bootstrap support for the observed partition at each cut.

    Support here answers "would this dendrogram survive another sample of the
    same postings" — and a high number is *not* a licence to publish the tree.
    Sampling stability and specification stability are different things: this
    repo's partition holds in 87% of resamples and changes shape entirely when
    own products are dropped, so `build_similarity` runs this function on both
    vocabularies and the report quotes the pair.
    """
    rng = _rng(seed)
    fn = METRICS[metric]
    keys = sorted(inc)
    drop_idx = None
    if drop_skills and skills is not None:
        drop = set(drop_skills)
        drop_idx = np.array([s not in drop for s in skills])

    def _matrix(prof_map):
        m = pd.DataFrame(index=keys, columns=keys, dtype=float)
        for a in keys:
            for b in keys:
                va, vb = prof_map[a], prof_map[b]
                if drop_idx is not None:
                    va, vb = va[drop_idx], vb[drop_idx]
                m.loc[a, b] = 0.0 if a == b else 1.0 - fn(va, vb)
        return m

    obs_parts = {}
    merges = average_linkage(_matrix(profiles(inc)))
    for k in cuts:
        obs_parts[k] = cut_tree(merges, keys, k)

    hits = {k: 0 for k in cuts}
    for _ in range(draws):
        boot = {}
        for k in keys:
            n = inc[k].shape[0]
            boot[k] = (inc[k][rng.integers(0, n, n)].mean(axis=0) if n
                       else np.zeros(inc[k].shape[1]))
        mb = average_linkage(_matrix(boot))
        for k in cuts:
            if cut_tree(mb, keys, k) == obs_parts[k]:
                hits[k] += 1

    rows = []
    for k in cuts:
        rows.append({
            "metric": metric, "clusters": k,
            "partition": " | ".join(sorted(",".join(sorted(c)) for c in obs_parts[k])),
            "bootstrap_support": round(hits[k] / draws, 4),
            "draws": draws,
            "vocabulary": "all_products_dropped" if drop_idx is not None else "all_skills",
        })
    return pd.DataFrame(rows)


def network_thresholds(sim: pd.DataFrame, grid=None) -> pd.DataFrame:
    """Edges, components and isolates as the edge threshold sweeps.

    A network graph is a threshold in a trench coat. Publishing the sweep
    makes the arbitrariness measurable: if there is a range over which the
    picture does not change, it is in this table, and if the graph goes from
    complete to empty with no plateau then no single picture is defensible.
    """
    keys = list(sim.index)
    if grid is None:
        grid = np.round(np.arange(0.40, 1.001, 0.05), 2)
    rows = []
    for t in grid:
        edges = [(a, b) for a, b in itertools.combinations(keys, 2)
                 if float(sim.loc[a, b]) >= t]
        parent = {k: k for k in keys}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in edges:
            parent[find(a)] = find(b)
        degree = {k: 0 for k in keys}
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
        isolated = sorted(k for k in keys if degree[k] == 0)
        rows.append({
            "threshold": float(t), "edges": len(edges),
            "possible_edges": len(keys) * (len(keys) - 1) // 2,
            "components": len({find(k) for k in keys}),
            "isolated": ",".join(isolated),
            "n_isolated": len(isolated),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trajectory similarity, in Aitchison geometry
# ---------------------------------------------------------------------------


def closure_expectation(n_parts: int) -> float:
    """``−1/(D−1)`` — the mean pairwise correlation closure forces on itself.

    Shares of one pool sum to 1 every period, so one company's rise is
    arithmetically someone else's fall. The mean correlation between parts of
    a closed composition is negative before any company does anything, and
    this is that number. Pearson found it in 1897; it is still the most common
    way to read a co-movement that is not there.
    """
    return -1.0 / (n_parts - 1) if n_parts > 1 else float("nan")


def closure_null(n_periods: int, n_parts: int, sigma: float = 0.3,
                 draws: int = N_NULL, seed: int | None = None) -> dict:
    """Mean pairwise correlation of *independent* series after closure.

    The analytic ``−1/(D−1)`` assumes exchangeable parts; this simulates the
    actual shape of the problem — independent lognormal series, closed to a
    composition, correlated on the log scale at the sample size in hand — and
    returns the interval an observed mean has to escape before it is evidence
    of anything.
    """
    rng = _rng(seed)
    means = np.empty(draws)
    idx = list(itertools.combinations(range(n_parts), 2))
    for d in range(draws):
        x = np.exp(rng.normal(0.0, sigma, size=(n_periods, n_parts)))
        s = x / x.sum(axis=1, keepdims=True)
        logs = np.log(s)
        means[d] = float(np.mean([np.corrcoef(logs[:, i], logs[:, j])[0, 1] for i, j in idx]))
    return {
        "mean": float(means.mean()),
        "p2_5": float(np.percentile(means, 2.5)),
        "p97_5": float(np.percentile(means, 97.5)),
        "draws": draws, "n_periods": n_periods, "n_parts": n_parts,
        "analytic": closure_expectation(n_parts),
    }


def aitchison_variation(wide: pd.DataFrame) -> pd.DataFrame:
    """``var(log(x_a / x_b))`` — the scale-invariant distance for compositions.

    A log ratio of two parts is untouched by the closure: multiply every share
    in a period by any constant and the ratio is unchanged, so the quantity
    describes the two companies rather than the pool. Zero means the two
    shares moved in exact proportion all year.
    """
    keys = list(wide.columns)
    logs = np.log(wide.to_numpy(dtype=float))
    out = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            out.loc[a, b] = 0.0 if i == j else float(np.var(logs[:, i] - logs[:, j], ddof=1))
    return out.rename_axis(index="company", columns="company")


def _fisher_interval(r: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 3 or not -1 < r < 1:
        return float("nan"), float("nan")
    zr = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    return math.tanh(zr - z * se), math.tanh(zr + z * se)


def trajectory_table(series: pd.DataFrame, gate: pd.DataFrame,
                     min_periods: int = MIN_TRAJECTORY_PERIODS) -> pd.DataFrame:
    """Pairwise co-movement of the Task 07 panel-share series, three ways.

    Every pair gets a raw log-share correlation, a centred-log-ratio
    correlation and an Aitchison variation, plus the gate Task 07 §14 asked
    for: ``eligible`` is true only when **both** members were called
    ``forecastable``. February is never filled — the periods are the
    intersection of what both companies actually observed.

    The raw correlation column exists to be argued with. It is the number a
    reader would compute, it is biased negative by closure, and putting it
    beside the log-ratio version is the cheapest way to show the size of that
    bias.
    """
    ok = set(gate.loc[gate.verdict == "forecastable", "key"])
    obs = {k: fc.observed(series, k).set_index("period") for k in sorted(series.key.unique())}
    keys = sorted(obs)
    common = sorted(set.intersection(*[set(o.index) for o in obs.values()])) if obs else []
    wide = pd.DataFrame({k: obs[k].loc[common, "share"].to_numpy() for k in keys}, index=common)
    logs = np.log(wide.to_numpy(dtype=float))
    clr = logs - logs.mean(axis=1, keepdims=True)
    var = aitchison_variation(wide)
    n = len(common)
    rows = []
    for a, b in pairs(keys):
        i, j = keys.index(a), keys.index(b)
        r = float(np.corrcoef(logs[:, i], logs[:, j])[0, 1])
        rc = float(np.corrcoef(clr[:, i], clr[:, j])[0, 1])
        lo, hi = _fisher_interval(r, n)
        rows.append({
            "company_a": a, "company_b": b,
            "n_periods": n,
            "r_log_share": round(r, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "ci_width": round(hi - lo, 4),
            "r_clr": round(rc, 4),
            "aitchison_variation": round(float(var.loc[a, b]), 4),
            "gate_a": a in ok, "gate_b": b in ok,
            "eligible": bool(a in ok and b in ok and n >= min_periods),
            "excludes_zero": bool(lo > 0 or hi < 0),
        })
    return pd.DataFrame(rows).sort_values("aitchison_variation").reset_index(drop=True)


@dataclass(frozen=True)
class TrajectoryVerdict:
    """Whether co-movement between companies is identified at all."""

    identified: bool
    reason: str
    detail: dict


def trajectory_verdict(traj: pd.DataFrame, null: dict) -> TrajectoryVerdict:
    """Three independent reasons the answer can be no, checked in order.

    Coverage (how many pairs survive Task 07's gate), closure (is the mean
    correlation outside what independent series produce) and precision (is any
    interval narrow enough to order pairs). A single one of these is enough to
    refuse; reporting all three stops the refusal being read as a quirk of one
    threshold.
    """
    eligible = int(traj.eligible.sum())
    mean_r = float(traj.r_log_share.mean())
    inside = null["p2_5"] <= mean_r <= null["p97_5"]
    widest = float(traj.ci_width.max())
    mean_width = float(traj.ci_width.mean())
    detail = {
        "eligible_pairs": eligible, "total_pairs": int(len(traj)),
        "mean_r_log_share": round(mean_r, 4),
        "closure_null_mean": round(null["mean"], 4),
        "closure_null_p2_5": round(null["p2_5"], 4),
        "closure_null_p97_5": round(null["p97_5"], 4),
        "mean_r_inside_closure_null": bool(inside),
        "mean_ci_width": round(mean_width, 4),
        "max_ci_width": round(widest, 4),
        "pairs_excluding_zero": int(traj.excludes_zero.sum()),
    }
    reasons = []
    if eligible < len(traj):
        reasons.append(f"only {eligible} of {len(traj)} pairs pass the Task 07 gate on both members")
    if inside:
        reasons.append(
            f"mean correlation {mean_r:.3f} sits inside the closure null "
            f"[{null['p2_5']:.3f}, {null['p97_5']:.3f}]")
    if mean_width > 1.0:
        reasons.append(f"mean 95% interval spans {mean_width:.2f} of the [-1, 1] range")
    if reasons:
        return TrajectoryVerdict(False, "; ".join(reasons), detail)
    return TrajectoryVerdict(True, "coverage, closure and precision all clear", detail)


# ---------------------------------------------------------------------------
# The gate this task publishes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarityVerdict:
    """Whether a pair's place in the ranking is a finding or an artefact."""

    pair: tuple[str, str]
    verdict: str
    reason: str


def pair_verdicts(boot: pd.DataFrame, pt: pd.DataFrame, vend: pd.DataFrame,
                  mix: pd.DataFrame) -> pd.DataFrame:
    """One verdict per pair, from the four checks that can each break a rank.

    ``robust``
        the bootstrap holds the rank, every prevalence-weighted metric agrees
        on it, and neither own products nor role mix moves it more than one
        place.
    ``metric_dependent``
        the rank is stable under resampling but the metric families disagree
        about it.
    ``vendor_dependent``
        dropping own products moves it more than one place.
    ``unresolved``
        the bootstrap does not hold the rank. Most pairs land here, and that
        is the shape of the result rather than a shortfall of it.
    """
    fam = [m for m in ALL_METRICS if METRIC_FAMILY[m] == "prevalence_weighted"
           and f"rank_{m}" in pt.columns]
    pt_idx = pt.set_index(["company_a", "company_b"])
    v_idx = vend.set_index(["company_a", "company_b"])
    m_idx = mix.set_index(["company_a", "company_b"])
    rows = []
    for row in boot.itertuples():
        key = (row.company_a, row.company_b)
        ranks = [int(pt_idx.loc[key, f"rank_{m}"]) for m in fam]
        family_agrees = max(ranks) - min(ranks) <= 1
        vendor_move = abs(int(v_idx.loc[key, "rank_move"])) if key in v_idx.index else 0
        mix_move = abs(int(m_idx.loc[key, "rank_move"])) if key in m_idx.index else 0
        if not row.rank_identified:
            verdict, reason = "unresolved", (
                f"rank {row.rank} holds in only {row.rank_stability:.0%} of resamples")
        elif vendor_move > 1:
            verdict, reason = "vendor_dependent", (
                f"dropping own products moves the rank {vendor_move} places")
        elif not family_agrees:
            verdict, reason = "metric_dependent", (
                f"prevalence-weighted metrics rank it {min(ranks)}-{max(ranks)}")
        elif mix_move > 1:
            verdict, reason = "mix_dependent", (
                f"standardising role mix moves the rank {mix_move} places")
        else:
            verdict, reason = "robust", (
                f"rank {row.rank} in {row.rank_stability:.0%} of resamples, "
                f"stable across metric, own products and role mix")
        rows.append({
            "company_a": row.company_a, "company_b": row.company_b,
            "rank": row.rank, "observed": row.observed,
            "rank_stability": row.rank_stability,
            "family_rank_spread": max(ranks) - min(ranks),
            "vendor_rank_move": vendor_move, "mix_rank_move": mix_move,
            "verdict": verdict, "reason": reason,
        })
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Standing guards, inherited
# ---------------------------------------------------------------------------


def forbidden_columns(table: pd.DataFrame) -> list[str]:
    """Task 06's banned column families, re-run on every Task 08 table."""
    return cmp.forbidden_columns(table)


def personal_data_columns_present(table: pd.DataFrame) -> list[str]:
    """The standing Task 01 privacy check."""
    return cmp.personal_data_columns_present(table)
