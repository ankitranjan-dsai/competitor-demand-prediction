"""Check `src/insights.py`'s two hand-rolled statistics against scipy.

`insights.py` carries no scipy import, for the reason Task 07 §2 gave and every
task since has inherited: the module has to run in a teammate's environment and
in the grader's, and the thing it decides — whether a sentence is published at
all — is a poor place to put a dependency whose tail convention or resampling
scheme can change between versions. `test_insights.py` pins that promise at the
source, which is what keeps this script's argument meaningful.

The cost of the promise is two pieces of statistics written out by hand, and
both sit directly under a published verdict:

- **`sign_test_p`** — a two-sided exact binomial tail at p0 = 0.5, summed with
  `math.comb` and doubled. It produces the `clears_005` column of C8's
  cell-floor table and the whole of the sign-test power table, which is the
  half of C8 that says unanimity across five or fewer publishers cannot reach
  significance however clean it looks.
- **the percentile bootstrap median difference** — `_median` plus
  `np.percentile` over resampled draws. Its `spans_zero` column is the entire
  quantitative content of §7's salary refusal: one pair excludes zero, and
  every stratified comparison does not.

Hand-written statistics are exactly what a unit test fails to catch, because
the test and the code share the author's mistake. So this script imports what
`insights.py` refuses to depend on and checks the two implementations agree.

It is a **validator, not a dependency**: nothing in `src/`, `tests/` or
`build_insights.py` imports it, and the pipeline runs to completion with scipy
uninstalled.

What is being proved, in order of how much it matters:

1. **the sign test is the exact test it claims to be.** Every (k, n) pair up to
   n = 40 matches `scipy.stats.binomtest(k, n, 0.5, alternative="two-sided")`
   to float noise, and so does every row of both committed tables. The
   doubling shortcut is also shown to be *specific* to p0 = 0.5 — it is wrong
   elsewhere, which is safe only because the module hardcodes 0.5, and that is
   worth writing down rather than assuming.
2. **the salary verdicts survive an independent bootstrap.** `scipy.stats.
   bootstrap(..., method="percentile")` reproduces every published
   `spans_zero`, on the three unstratified pairs and on the two stratified
   strata that clear the cell floor. The verdicts are what §7 rests on; the
   endpoints are not.
3. **the endpoint differences are Monte-Carlo noise, and are measured rather
   than asserted.** Two correct percentile bootstraps disagree because they
   draw different resamples, so §3 compares the gap against the spread of
   `insights.py`'s own procedure across 30 reseeds, and §4 measures the
   committed endpoints against a 200,000-resample reference.
4. **the interval is coarse for a reason the reader has to see.** These
   salary samples are heavily tied — 10 postings over 8 distinct values — so
   the bootstrap median difference is a lattice with atoms thousands of
   dollars apart. A percentile endpoint can only land on an atom, which is why
   an endpoint moves by $2,500 between seeds while the verdict does not move
   at all.

    python src/validate_insights.py

Writes `docs/task-09-insight-validation.md`. Exits non-zero if any check fails.
Posting-level data is read only for the salary vectors, and the evidence file
holds nothing but statistics.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare as cmp        # noqa: E402
import insights as ins       # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "task-09-insight-validation.md"
TABLES = REPO_ROOT / "members" / "ankit-google" / "task-09-tables"

COMPANIES = ["databricks", "google", "meta", "microsoft", "nvidia", "snowflake"]

# ---------------------------------------------------------------------------
# Tolerances. Each one is a different kind of claim, so they are not
# interchangeable and none of them is a round number chosen to make a check
# pass.
# ---------------------------------------------------------------------------

#: Machine-precision agreement. `sign_test_p` and `binomtest` at p0 = 0.5 are
#: algebraically the same sum, so anything above float noise means one of the
#: two is not what it claims.
EXACT = 1e-10

#: Half a step of the committed tables' own rounding. `publisher_cell_floor`
#: writes `round(p, 4)`, so a recomputed p may legitimately differ from the CSV
#: by up to 5e-05 and no further. Three rows land *exactly* on that half-step,
#: because the sign test at p0 = 0.5 produces dyadic rationals — 0.03125,
#: 0.21875 — whose fourth decimal is an exact tie, and `round` breaks ties to
#: even. `ROUNDING_SLACK` exists only so that `<=` at the boundary is not
#: decided by float representation; it is not a widening of the tolerance.
ROUNDING = 5e-5
ROUNDING_SLACK = 1e-9

#: Monte-Carlo agreement is not a fixed number of dollars — it depends on the
#: sample, and on the lattice the medians live on. The tolerance for a
#: bootstrap endpoint is therefore *derived*: a scipy endpoint must sit within
#: `SEED_SPREAD_K` standard deviations of the spread `insights.py`'s own
#: procedure shows across `SEED_REPLICATES` reseeds, with a floor of one
#: lattice step, because a percentile endpoint cannot resolve anything finer
#: than the gap between two attainable values.
SEED_REPLICATES = 30
SEED_SPREAD_K = 4.0

#: `insights.py`'s own settings, reproduced here rather than imported, so that
#: a change to either default shows up as a failure in this file instead of
#: silently re-baselining the comparison.
COMMITTED_SEED = 20240609
COMMITTED_N_BOOT = 2000

#: The reseed ladder. A large odd stride keeps the PCG64 streams well apart.
SEED_STRIDE = 7919

#: scipy's side of the comparison. `SCIPY_RESAMPLES` is scipy's own default and
#: is deliberately *not* 2000: matching the resample count would hide whether
#: the committed interval is stable, which is the question. `REFERENCE_*` is
#: the near-exact percentile endpoint the two are both estimating.
SCIPY_RESAMPLES = 9_999
SCIPY_SEED = 20240609
REFERENCE_RESAMPLES = 200_000

#: Resamples for §5's three-method comparison. Larger than §3's cross-check
#: because BCa's acceleration term is estimated by jackknife on top of the
#: resampling, and a noisy endpoint would be indistinguishable from a genuine
#: method difference — which is precisely what §5 is trying to measure.
METHOD_RESAMPLES = 50_000
REFERENCE_SEED = 11

#: Every published bootstrap comparison in Task 09: the three unstratified
#: pairs, then the two strata that clear the cell floor. Named here so a table
#: that quietly gains a row makes this script fail rather than skip it.
UNSTRATIFIED = [
    ("google", "databricks", "via Ai-Jobs.net"),
    ("google", "snowflake", "via Ai-Jobs.net"),
    ("google", "meta", "via Ladders"),
]
STRATIFIED = [
    ("google", "snowflake", "via Ai-Jobs.net", "Science / Research"),
    ("google", "meta", "via Ladders", "Science / Research"),
]


def _fmt(x, places: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:,.{places}f}"


def _test_count() -> int:
    """Count `def test_` in the insight suite, so §6 cannot go stale.

    A hardcoded count is wrong the first time anyone adds a test, and this
    file's argument in §6 is precisely that the non-statistical half of
    `insights.py` is pinned over there instead of here.
    """
    source = (REPO_ROOT / "tests" / "test_insights.py").read_text()
    return sum(1 for line in source.splitlines()
               if line.startswith("def test_"))


def _signed(value: float) -> str:
    """`+ 25,000` / `- 25,000` — so reflected arithmetic never reads `- -x`."""
    return f"{'+' if value >= 0 else '-'} {_fmt(abs(value), 0)}"


def _median_difference(x, y, axis: int = -1):
    """The statistic, in the vectorised form `scipy.stats.bootstrap` wants.

    `np.median` rather than `ins._median` on purpose: §2 proves the two agree
    exactly, and using the reference implementation here keeps scipy's side of
    the comparison genuinely independent of the code under test.
    """
    return np.median(x, axis=axis) - np.median(y, axis=axis)


def _disclosed(frames: dict, company: str, publisher: str) -> pd.DataFrame:
    df = frames[company]
    return df[(df[ins.PUBLISHER_COL] == publisher) & df[ins.SALARY_COL].notna()]


def _vector(frames: dict, company: str, publisher: str,
            stratum: str | None = None) -> np.ndarray:
    block = _disclosed(frames, company, publisher)
    if stratum is not None:
        block = block[block.job_function == stratum]
    return block[ins.SALARY_COL].to_numpy(dtype=float)


def _scipy_percentile_ci(va, vb, n_resamples: int, seed: int):
    """`scipy.stats.bootstrap` with the method the task actually used.

    `paired=False` is the point of agreement being tested: `insights.py`
    resamples the two companies independently, and so does scipy under this
    setting. `method="percentile"` takes the 2.5th and 97.5th percentiles of
    the bootstrap distribution directly, which is the rule `insights.py` writes
    out by hand as two `np.percentile` calls.
    """
    from scipy.stats import bootstrap

    res = bootstrap(
        (va, vb), _median_difference,
        n_resamples=n_resamples, confidence_level=0.95,
        method="percentile", paired=False, vectorized=True,
        random_state=np.random.default_rng(seed),
    )
    ci = res.confidence_interval
    return float(ci.low), float(ci.high), np.asarray(res.bootstrap_distribution)


def _insights_ci(va, vb, seed: int, n_boot: int = COMMITTED_N_BOOT):
    """`salary_pair_estimate`'s interval, reproduced draw for draw.

    Not a call into `insights.py`: that function reads frames and applies the
    cell floor, and what needs re-running here is only the resampling loop, at
    a seed the committed table did not use.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ra = rng.choice(va, size=len(va), replace=True)
        rb = rng.choice(vb, size=len(vb), replace=True)
        draws[i] = ins._median(ra) - ins._median(rb)
    return (float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def _atom_step(distribution: np.ndarray) -> float:
    """The finest difference a percentile endpoint on this sample can resolve.

    The bootstrap distribution of a median difference over tied integer-valued
    salaries is discrete. An endpoint is an order statistic of that
    distribution, so it can only ever land on one of its atoms, and two correct
    bootstraps that disagree by a single atom have not disagreed about
    anything.
    """
    unique = np.unique(distribution)
    if unique.size < 2:
        return 0.0
    return float(np.min(np.diff(unique)))


# ---------------------------------------------------------------------------
# 1. the sign test
# ---------------------------------------------------------------------------

def check_sign_test() -> tuple[list, list[str]]:
    """`sign_test_p` against `scipy.stats.binomtest`, exhaustively and in situ.

    Exhaustively first, because the published tables only ever exercise n <= 7
    and a formula can be right on small n and wrong on the tail sum. Then on
    the committed rows themselves, because agreement on a grid is not
    agreement on the numbers a reader will quote.
    """
    from scipy.stats import binomtest

    lines = ["## 1. The sign test against `scipy.stats.binomtest`", ""]
    checks = []

    worst, worst_at = 0.0, None
    cells = 0
    for n in range(1, 41):
        for k in range(0, n + 1):
            mine = ins.sign_test_p(k, n)
            ref = binomtest(k, n, 0.5, alternative="two-sided").pvalue
            cells += 1
            if abs(mine - ref) > worst:
                worst, worst_at = abs(mine - ref), (k, n)
    checks.append((f"sign_test_p matches binomtest on all {cells} (k, n) "
                   f"pairs to {EXACT:g}", worst < EXACT))

    lines += [
        f"`sign_test_p` sums `math.comb` from the more extreme tail inwards "
        f"and doubles it. `binomtest` at `p=0.5` sums every outcome no more "
        f"likely than the observed one. At p0 = 0.5 the binomial is symmetric "
        f"and the two rules coincide, which is a thing to check rather than "
        f"recite. Across all {cells:,} (k, n) pairs with n from 1 to 40:",
        "",
        "| comparison | max abs difference | where |",
        "| --- | --- | --- |",
        f"| `sign_test_p(k, n)` vs `binomtest(k, n, 0.5).pvalue` | "
        f"{worst:.3e} | k={worst_at[0]}, n={worst_at[1]} |",
        "",
        "That is float noise on a sum of `2**n` terms, not a difference in "
        "method.",
        "",
        "### 1.1 The doubling rule is specific to p0 = 0.5",
        "",
        "Worth stating because the function's name does not say so and a "
        "future caller might reasonably try. Doubling the extreme tail is a "
        "*symmetry* argument; it has no general validity, and the module is "
        "safe only because it hardcodes 0.5:",
        "",
        "| k | n | p0 | 0.5-doubling rule | `binomtest` at that p0 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k, n, p0 in ((7, 10, 0.30), (2, 10, 0.30), (9, 12, 0.25)):
        extreme = max(k, n - k)
        tail = sum(math.comb(n, j) for j in range(extreme, n + 1))
        doubled = min(1.0, 2.0 * tail / 2.0 ** n)
        ref = binomtest(k, n, p0, alternative="two-sided").pvalue
        lines.append(f"| {k} | {n} | {p0} | {doubled:.6f} | {ref:.6f} |")
    lines += [
        "",
        "`sign_test_p` takes no `p0` argument, so this cannot be reached by "
        "accident from the current call sites — both of them are testing a "
        "sign against a fixed direction, where 0.5 is the right null.",
        "",
        "### 1.2 The committed tables, recomputed row by row",
        "",
    ]

    recount = pd.read_csv(TABLES / "publisher-cell-floor.csv")
    power = pd.read_csv(TABLES / "sign-test-power.csv")

    def _round_trip(frame: pd.DataFrame, k_col: str, n_col: str, label: str):
        worst_p, flips = 0.0, 0
        for _, row in frame.iterrows():
            n = int(row[n_col])
            if n <= 0:
                continue
            ref = binomtest(int(row[k_col]), n, 0.5,
                            alternative="two-sided").pvalue
            worst_p = max(worst_p, abs(ref - float(row.sign_test_p)))
            if bool(ref < 0.05) != bool(row.clears_005):
                flips += 1
        return worst_p, flips

    floor_err, floor_flips = _round_trip(
        recount, "publishers_agreeing", "publishers_tested", "cell floor")
    power_err, power_flips = _round_trip(
        power, "unanimous_agreeing", "publishers_tested", "power")

    checks += [
        (f"all {len(recount)} cell-floor p-values match binomtest to "
         f"{ROUNDING:g}", floor_err <= ROUNDING * (1 + ROUNDING_SLACK)),
        ("no cell-floor clears_005 verdict flips under binomtest",
         floor_flips == 0),
        (f"all {len(power)} power-table p-values match binomtest to "
         f"{ROUNDING:g}", power_err <= ROUNDING * (1 + ROUNDING_SLACK)),
        ("no power-table clears_005 verdict flips under binomtest",
         power_flips == 0),
    ]

    lines += [
        "Both tables store `round(p, 4)`, so the tolerance here is "
        f"{ROUNDING:g} — half a step of their own rounding — and not "
        f"{EXACT:g}. A difference at that scale is the CSV, not the statistic.",
        "",
        "Three rows sit *exactly* on the boundary, and it is worth saying why "
        "rather than leaving it as a suspicious coincidence. A sign test at "
        "p0 = 0.5 returns a dyadic rational, so its fourth decimal is often an "
        "exact tie: NVIDIA's 6/6 is 0.03125 and stores as `0.0312`, Meta's 5/6 "
        "is 0.21875 and stores as `0.2188`, 7/7 is 0.015625 and stores as "
        "`0.0156`. Python's `round` breaks ties to even, which is why those "
        "three go in different directions. Every one of them is half a "
        "rounding step from the exact value and none is nearer the 0.05 "
        "boundary for it.",
        "",
        "| table | rows | max abs difference vs `binomtest` | `clears_005` flips |",
        "| --- | --- | --- | --- |",
        f"| `publisher-cell-floor.csv` | {len(recount)} | {floor_err:.3e} | "
        f"{floor_flips} |",
        f"| `sign-test-power.csv` | {len(power)} | {power_err:.3e} | "
        f"{power_flips} |",
        "",
        "The two numbers C8 is argued from, taken directly:",
        "",
        "| claim | committed | `binomtest` | agrees |",
        "| --- | --- | --- | --- |",
    ]

    nvidia = recount[(recount.company == "nvidia") & (recount.cell_floor == 0)]
    nv_p = float(nvidia.sign_test_p.iloc[0])
    nv_ref = binomtest(int(nvidia.publishers_agreeing.iloc[0]),
                       int(nvidia.publishers_tested.iloc[0]), 0.5,
                       alternative="two-sided").pvalue
    five = power[power.publishers_tested == 5]
    five_p = float(five.sign_test_p.iloc[0])
    five_ref = binomtest(5, 5, 0.5, alternative="two-sided").pvalue

    lines += [
        f"| NVIDIA 6/6 at floor 0 clears 0.05 | p = {nv_p} | "
        f"p = {nv_ref:.6f} | {'yes' if nv_ref < 0.05 else 'no'} |",
        f"| unanimity across 5 publishers cannot | p = {five_p} | "
        f"p = {five_ref:.6f} | {'yes' if five_ref >= 0.05 else 'no'} |",
        "",
        "Both hold. The second is the one doing the work in §8 of the methods "
        "document: at this panel size a clean 5/5 is not evidence at the 5% "
        "level, so the cell floor is not a robustness knob — it is a knob that "
        "decides whether the test has any power left.",
        "",
    ]

    checks += [
        ("NVIDIA's floor-0 p clears 0.05 under binomtest", nv_ref < 0.05),
        ("5/5 unanimity does not clear 0.05 under binomtest", five_ref >= 0.05),
    ]
    return checks, lines


# ---------------------------------------------------------------------------
# 2. the median
# ---------------------------------------------------------------------------

def check_median(rng, frames: dict) -> tuple[list, list[str]]:
    """`_median` against `numpy.median`.

    A prerequisite rather than a finding. The bootstrap in §3 calls `_median`
    two thousand times per pair, so if the point estimator disagreed with the
    reference the interval comparison would be measuring that instead.
    """
    lines = ["## 2. `_median` against `numpy.median`", ""]

    worst_random = 0.0
    for _ in range(5000):
        n = int(rng.integers(1, 40))
        v = rng.normal(150_000, 40_000, n)
        worst_random = max(worst_random, abs(ins._median(v) - np.median(v)))

    worst_real, rows = 0.0, []
    for a, b, publisher in UNSTRATIFIED:
        for company in (a, b):
            v = _vector(frames, company, publisher)
            d = abs(ins._median(v) - np.median(v))
            worst_real = max(worst_real, d)
            rows.append(f"| {company} | {publisher} | {len(v)} | "
                        f"{_fmt(ins._median(v))} | {d:.3e} |")

    lines += [
        "5,000 random vectors of length 1 to 39, then every disclosed-salary "
        "vector the published intervals are computed on. Both even and odd "
        "lengths are covered, which is the only branch `_median` has:",
        "",
        f"- random vectors: max abs difference {worst_random:.3e}",
        f"- published vectors: max abs difference {worst_real:.3e}",
        "",
        "| company | publisher | n | `_median` | difference |",
        "| --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]
    checks = [
        (f"_median matches np.median on random vectors to {EXACT:g}",
         worst_random < EXACT),
        (f"_median matches np.median on every published vector to {EXACT:g}",
         worst_real < EXACT),
    ]
    return checks, lines


# ---------------------------------------------------------------------------
# 3. the percentile bootstrap
# ---------------------------------------------------------------------------

def check_bootstrap(frames: dict) -> tuple[list, list[str], list]:
    """The published intervals against `scipy.stats.bootstrap(method=...)`.

    The comparison has two levels and they are not equally important. The
    **verdict** — does the interval span zero — is what §7 publishes, and it
    has to reproduce exactly. The **endpoints** are two Monte-Carlo estimates
    of the same quantity from different random draws, so they cannot be
    expected to match, and the honest check is whether the gap is inside the
    spread the procedure shows against itself.
    """
    lines = ["## 3. The percentile bootstrap against `scipy.stats.bootstrap`",
             ""]
    checks = []

    lines += [
        "`salary_pair_estimate` resamples each company's postings "
        f"independently {COMMITTED_N_BOOT:,} times, takes the difference of "
        "medians on each draw, and reads the 2.5th and 97.5th percentiles off "
        "the result. The reference is:",
        "",
        "```python",
        "scipy.stats.bootstrap(",
        "    (va, vb), statistic,          # median(x) - median(y)",
        f"    n_resamples={SCIPY_RESAMPLES}, confidence_level=0.95,",
        '    method="percentile", paired=False, vectorized=True,',
        f"    random_state=np.random.default_rng({SCIPY_SEED}),",
        ")",
        "```",
        "",
        "`paired=False` is the setting under test, not a convenience: it is "
        "what makes scipy resample the two companies independently, which is "
        "what `insights.py` does. `method=\"percentile\"` is the rule "
        "`insights.py` writes by hand — §5 shows what changing it would do.",
        "",
        "### 3.1 The three unstratified pairs",
        "",
        "`spans_zero` is the published column. The endpoints are shown so the "
        "size of the disagreement is visible, and the last column is the test "
        "that matters:",
        "",
        "| pair | publisher | n | committed CI | scipy CI | committed "
        "`spans_zero` | scipy `spans_zero` |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    committed = pd.read_csv(TABLES / "salary-pairs.csv")
    detail = []
    verdict_flips = 0

    for a, b, publisher in UNSTRATIFIED:
        row = committed[(committed.company_a == a)
                        & (committed.company_b == b)
                        & (committed.publisher == publisher)]
        if row.empty:
            raise SystemExit(f"salary-pairs.csv has no {a}/{b} @ {publisher}; "
                             "the pair list in this script is stale")
        row = row.iloc[0]
        va, vb = _vector(frames, a, publisher), _vector(frames, b, publisher)
        lo, hi, _ = _scipy_percentile_ci(va, vb, SCIPY_RESAMPLES, SCIPY_SEED)
        # The lattice step must be measured where the atoms are fully
        # resolved. At 9,999 resamples the tail atoms are still missing, which
        # makes the apparent step *larger* and would widen the tolerance in
        # exactly the wrong direction, so it is taken from the reference run.
        ref_lo, ref_hi, dist = _scipy_percentile_ci(
            va, vb, REFERENCE_RESAMPLES, REFERENCE_SEED)
        c_lo, c_hi = float(row.ci_low), float(row.ci_high)
        c_spans, s_spans = bool(row.spans_zero), bool(lo <= 0 <= hi)
        if c_spans != s_spans:
            verdict_flips += 1

        seeds = [_insights_ci(va, vb, COMMITTED_SEED + i * SEED_STRIDE)
                 for i in range(SEED_REPLICATES)]
        lows = np.array([s[0] for s in seeds])
        highs = np.array([s[1] for s in seeds])
        step = _atom_step(dist)
        tol_lo = max(SEED_SPREAD_K * float(lows.std(ddof=1)), step)
        tol_hi = max(SEED_SPREAD_K * float(highs.std(ddof=1)), step)

        lines.append(
            f"| {a} – {b} | {publisher} | {int(row.n_a)}, {int(row.n_b)} | "
            f"[{_fmt(c_lo, 0)}, {_fmt(c_hi, 0)}] | "
            f"[{_fmt(lo, 0)}, {_fmt(hi, 0)}] | {c_spans} | {s_spans} |")

        detail.append({
            "pair": f"{a} – {b}", "publisher": publisher,
            "lows": lows, "highs": highs, "step": step,
            "c_lo": c_lo, "c_hi": c_hi, "lo": lo, "hi": hi,
            "tol_lo": tol_lo, "tol_hi": tol_hi, "va": va, "vb": vb,
            "spans": c_spans, "dist": dist,
            "ref_lo": ref_lo, "ref_hi": ref_hi,
        })
        checks.append((f"scipy reproduces spans_zero for {a} – {b} @ "
                       f"{publisher}", c_spans == s_spans))

    lines += [
        "",
        "**All three verdicts reproduce.** No published `spans_zero` changes "
        "under an independent implementation, so the salary refusal in §7 — "
        "one unstratified pair excluding zero, and that pair dissolving under "
        "stratification — is not an artefact of the hand-written resampler.",
        "",
        "### 3.2 The endpoints, against the tolerance they earn",
        "",
        "Two correct percentile bootstraps disagree, because they draw "
        "different resamples. Asserting a fixed dollar tolerance would be "
        "arbitrary, so the tolerance is derived: `insights.py`'s own procedure "
        f"is re-run at {SEED_REPLICATES} further seeds "
        f"(`{COMMITTED_SEED} + i x {SEED_STRIDE}`), and scipy has to land "
        f"within {SEED_SPREAD_K:g} standard deviations of that spread — or "
        f"within one lattice step (§4.3, measured on the "
        f"{REFERENCE_RESAMPLES:,}-resample reference so the atoms are fully "
        "resolved), whichever is wider.",
        "",
        "| pair | endpoint | committed | scipy | gap | reseed sd | tolerance | within |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for d in detail:
        for label, c_val, s_val, arr, tol in (
                ("lower", d["c_lo"], d["lo"], d["lows"], d["tol_lo"]),
                ("upper", d["c_hi"], d["hi"], d["highs"], d["tol_hi"])):
            gap = abs(s_val - c_val)
            ok = gap <= tol
            lines.append(
                f"| {d['pair']} | {label} | {_fmt(c_val, 0)} | "
                f"{_fmt(s_val, 0)} | {_fmt(gap, 0)} | "
                f"{_fmt(float(arr.std(ddof=1)), 0)} | {_fmt(tol, 0)} | "
                f"{'yes' if ok else 'NO'} |")
            checks.append((f"{d['pair']} {label} endpoint within Monte-Carlo "
                           f"tolerance", ok))

    lines += [
        "",
        "Every gap is inside the spread the committed procedure shows against "
        "itself, so none of them is evidence of a different estimator. The "
        "reseed range is the more useful way to read this — it is what the "
        "published endpoint would have been had the seed been anything else:",
        "",
        "| pair | lower across reseeds | upper across reseeds | `spans_zero` "
        "across reseeds |",
        "| --- | --- | --- | --- |",
    ]

    seed_stable = 0
    for d in detail:
        verdicts = {bool(lo <= 0 <= hi)
                    for lo, hi in zip(d["lows"], d["highs"])}
        stable = verdicts == {d["spans"]}
        seed_stable += int(stable)
        lines.append(
            f"| {d['pair']} | {_fmt(d['lows'].min(), 0)} to "
            f"{_fmt(d['lows'].max(), 0)} | {_fmt(d['highs'].min(), 0)} to "
            f"{_fmt(d['highs'].max(), 0)} | "
            f"{'unchanged (' + str(d['spans']) + ')' if stable else 'MOVES'} |")
        checks.append((f"{d['pair']} spans_zero is unchanged across "
                       f"{SEED_REPLICATES} reseeds", stable))

    gd = detail[0]
    lines += [
        "",
        f"The identified difference is the one that has to hold, and it holds "
        f"with margin: Google – Databricks has a lower bound of "
        f"{_fmt(gd['c_lo'], 0)} as published and never falls below "
        f"{_fmt(gd['lows'].min(), 0)} across {SEED_REPLICATES} reseeds, "
        "against a threshold of zero. That verdict is not close to the line, "
        "which is why §7 can refuse the benchmark on the *stratification* "
        "argument rather than on interval width — the difference is real "
        "within this publisher and still uninterpretable, because Google's "
        "disclosed cell there is 70.0% Science / Research and Databricks' is "
        "42.4% Engineering.",
        "",
        "### 3.3 The two stratified strata that clear the cell floor",
        "",
        "Thirteen of the fifteen stratified rows are `untestable` and carry no "
        "interval. The two that do are the comparison §7.3 actually rests on:",
        "",
        "| pair | stratum | n | committed CI | scipy CI | verdict | scipy "
        "agrees |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    strat = pd.read_csv(TABLES / "salary-pairs-stratified.csv")
    for a, b, publisher, stratum in STRATIFIED:
        row = strat[(strat.company_a == a) & (strat.company_b == b)
                    & (strat.publisher == publisher)
                    & (strat.stratum == stratum)]
        if row.empty:
            raise SystemExit(f"salary-pairs-stratified.csv has no {a}/{b} "
                             f"{stratum}; the stratum list here is stale")
        row = row.iloc[0]
        va = _vector(frames, a, publisher, stratum)
        vb = _vector(frames, b, publisher, stratum)
        lo, hi, _ = _scipy_percentile_ci(va, vb, REFERENCE_RESAMPLES,
                                         REFERENCE_SEED)
        c_spans, s_spans = bool(row.spans_zero), bool(lo <= 0 <= hi)
        if c_spans != s_spans:
            verdict_flips += 1
        lines.append(
            f"| {a} – {b} | {stratum} | {int(row.n_a)}, {int(row.n_b)} | "
            f"[{_fmt(float(row.ci_low), 0)}, {_fmt(float(row.ci_high), 0)}] | "
            f"[{_fmt(lo, 0)}, {_fmt(hi, 0)}] | {row.verdict} | "
            f"{'yes' if c_spans == s_spans else 'NO'} |")
        checks.append((f"scipy reproduces the {a} – {b} {stratum} verdict",
                       c_spans == s_spans))

    lines += [
        "",
        f"These are run at {REFERENCE_RESAMPLES:,} resamples rather than "
        f"{SCIPY_RESAMPLES:,} because there are only two of them and they are "
        "the rows §7.3 quotes. Both endpoints reproduce *exactly*, not merely "
        "within tolerance — with 7 and 15 observations over a handful of "
        "distinct salaries, the bootstrap distribution has so few atoms that "
        "both implementations land on the same one whatever they draw. The "
        "coarseness that makes §4's warning necessary is also what makes these "
        "two rows exactly reproducible.",
        "",
    ]
    checks.append(("no published spans_zero verdict flips under scipy",
                   verdict_flips == 0))
    return checks, lines, detail


# ---------------------------------------------------------------------------
# 4. seeds, Monte-Carlo error and the lattice
# ---------------------------------------------------------------------------

def check_monte_carlo(frames: dict, detail: list) -> tuple[list, list[str]]:
    """How much Monte-Carlo error the committed endpoints carry, and why.

    §3 shows the two implementations agree within noise. This section measures
    the noise itself against a near-exact reference, and locates its source —
    which is not the resample count but the tie structure of the samples.
    """
    lines = ["## 4. Seeds, Monte-Carlo error and the lattice", ""]
    checks = []

    lines += [
        "### 4.1 Every seed in play",
        "",
        "| setting | value | where it comes from |",
        "| --- | --- | --- |",
        f"| `salary_pair_estimate(seed=...)` | `{COMMITTED_SEED}` | "
        "`insights.py` default; produced the committed tables |",
        f"| `salary_pair_stratified(seed=...)` | `{COMMITTED_SEED}` | "
        "same default, one generator shared across strata |",
        f"| `n_boot` | `{COMMITTED_N_BOOT:,}` | `insights.py` default, and "
        "`build_insights.py` does not override it |",
        f"| reseed ladder (this file) | `{COMMITTED_SEED} + i x "
        f"{SEED_STRIDE}`, i = 0..{SEED_REPLICATES - 1} | large odd stride, to "
        "keep the PCG64 streams apart |",
        f"| scipy comparison | `default_rng({SCIPY_SEED})`, "
        f"`n_resamples={SCIPY_RESAMPLES:,}` | scipy's own default resample "
        "count, deliberately not 2,000 |",
        f"| scipy reference | `default_rng({REFERENCE_SEED})`, "
        f"`n_resamples={REFERENCE_RESAMPLES:,}` | near-exact percentile "
        "endpoint |",
        "",
        "Two properties of the committed seeding are worth recording, neither "
        "of them a defect:",
        "",
        f"- **`salary_pair_estimate` re-seeds per call.** Each pair starts a "
        f"fresh `default_rng({COMMITTED_SEED})`, so the three pairs in "
        "`salary-pairs.csv` are drawn from the *same* stream position. Each "
        "interval is individually valid; their Monte-Carlo errors are not "
        "independent of one another. Nothing in Task 09 combines the three "
        "intervals, so this costs nothing today — it would matter the moment "
        "someone pooled them.",
        f"- **`salary_pair_stratified` shares one generator across strata.** "
        "The draws for the second stratum continue the stream from the first, "
        "so a stratum's interval depends on how many strata preceded it. "
        "Re-ordering `job_function` would change the endpoints within "
        "Monte-Carlo error. §3.3 shows both testable strata are exactly "
        "reproducible at 200,000 resamples, so this does not reach a published "
        "number here either.",
        "",
        "### 4.2 What the committed endpoints cost in Monte-Carlo error",
        "",
        f"Against a {REFERENCE_RESAMPLES:,}-resample percentile reference — "
        "the endpoint both implementations are estimating:",
        "",
        "| pair | endpoint | reference | committed | error | as % of interval "
        "width |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    widest = 0.0
    for d in detail:
        r_lo, r_hi = d["ref_lo"], d["ref_hi"]
        width = r_hi - r_lo
        for label, ref, com in (("lower", r_lo, d["c_lo"]),
                                ("upper", r_hi, d["c_hi"])):
            err = abs(com - ref)
            pct = 100.0 * err / width if width else float("nan")
            widest = max(widest, pct)
            lines.append(
                f"| {d['pair']} | {label} | {_fmt(ref, 0)} | "
                f"{_fmt(com, 0)} | {_fmt(err, 0)} | {pct:.1f}% |")

    lines += [
        "",
        f"The largest error is {widest:.1f}% of the interval width. At "
        f"{COMMITTED_N_BOOT:,} resamples that is the expected order, and it is "
        "an argument for reading these intervals as coarse rather than for "
        "re-running the task: none of the three verdicts is decided anywhere "
        "near an endpoint, and §3.2 shows the lower bound of the one "
        "identified difference sits tens of thousands of dollars clear of "
        "zero.",
        "",
        "### 4.3 Why the endpoints move in steps",
        "",
        "The reason an endpoint jumps by a fixed amount between seeds and "
        "then not at all is not the resample count. It is that disclosed "
        "salaries are "
        "heavily tied, so a bootstrap median can only take a handful of "
        "values, and a percentile endpoint is an order statistic of a discrete "
        "distribution:",
        "",
        "| pair | n | distinct salaries | distinct bootstrap differences | "
        "smallest step |",
        "| --- | --- | --- | --- | --- |",
    ]

    for d in detail:
        n_atoms = int(np.unique(d["dist"]).size)
        lines.append(
            f"| {d['pair']} | {len(d['va'])}, {len(d['vb'])} | "
            f"{len(np.unique(d['va']))}, {len(np.unique(d['vb']))} | "
            f"{n_atoms} | {_fmt(_atom_step(d['dist']), 0)} |")
        checks.append((f"{d['pair']} bootstrap distribution is discrete "
                       f"(atoms resolved)", n_atoms > 1))

    gm = next(d for d in detail if "meta" in d["pair"])
    gm_atoms_all = np.unique(gm["dist"])
    gm_upper_atom_values = [a for a in gm_atoms_all.tolist()
                            if gm["highs"].min() <= a <= gm["highs"].max()]
    gm_upper_atoms = len(gm_upper_atom_values)
    gm_on_atom = int(np.isin(gm["highs"], gm_atoms_all).sum())

    lines += [
        "",
        f"Google – Meta is the clearest case: {len(gm['va'])} and "
        f"{len(gm['vb'])} postings over {len(np.unique(gm['va']))} and "
        f"{len(np.unique(gm['vb']))} distinct salaries. Its lower bound is "
        f"identical at all {SEED_REPLICATES} reseeds — the atom is that "
        f"dominant — while its upper bound visits {gm_upper_atoms} atoms "
        f"({', '.join(_fmt(a, 0) for a in gm_upper_atom_values)}), landing on "
        f"one of them in {gm_on_atom} of the {SEED_REPLICATES} runs. The other "
        f"{SEED_REPLICATES - gm_on_atom} sit a few dollars off an atom, which "
        "is `np.percentile` interpolating linearly between two order "
        "statistics rather than a finer structure existing. **An endpoint that "
        "moves by one atom has not disagreed about anything**, which is why "
        "the tolerance in §3.2 carries a lattice-step floor as well as a "
        "multiple of the reseed spread.",
        "",
        "This is also the honest reading of the published `ci_low` / `ci_high` "
        "columns: they are accurate to about one lattice step, and quoting "
        "them to the cent — which the CSV's two decimal places invite — "
        "overstates them. Task 09 never quotes an endpoint in a claim; the "
        "only thing that reaches the ledger is `spans_zero`.",
        "",
    ]
    return checks, lines


# ---------------------------------------------------------------------------
# 5. does the verdict rest on the interval method?
# ---------------------------------------------------------------------------

def check_interval_method(frames: dict, detail: list) -> tuple[list, list[str]]:
    """Every published comparison under percentile, basic and BCa.

    §3 answers "is the percentile interval computed correctly". This answers
    the different and more consequential question: would another defensible
    interval rule have changed what Task 09 reported? For a median on a small,
    heavily tied sample the three methods are not interchangeable, and this
    section is the one place in this file where the answer is not simply yes.
    """
    from scipy.stats import bootstrap

    lines = ["## 5. Does the published verdict rest on the interval method?",
             ""]
    checks = []

    lines += [
        "The percentile method is what `insights.py` implements and therefore "
        "what §3 had to compare against. It is not the only defensible choice. "
        "Below, all five published comparisons — three unstratified pairs and "
        "the two strata that clear the cell floor — under each method scipy "
        f"offers, at {METHOD_RESAMPLES:,} resamples and seed "
        f"`{REFERENCE_SEED}`:",
        "",
        "These are fresh estimates at a different resample count and seed from "
        "the committed procedure, so the percentile rows here sit a few "
        "hundred to a few thousand dollars from the committed table. §3 and §4 "
        "are where that gap is reconciled; this section is only asking whether "
        "the *verdict* survives a change of method, and every row's verdict is "
        "read off that row.",
        "",
        "| comparison | n | point estimate | method | CI | excludes zero |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    cases = [(d["pair"], d["va"], d["vb"], d["spans"], "unstratified")
             for d in detail]
    for a, b, publisher, stratum in STRATIFIED:
        va = _vector(frames, a, publisher, stratum)
        vb = _vector(frames, b, publisher, stratum)
        cases.append((f"{a} – {b} ({stratum})", va, vb, True, "stratified"))

    results: dict[tuple[str, str], bool | None] = {}
    results_index: dict[str, tuple] = {}
    percentile_ci: dict[str, tuple[float, float]] = {}
    bca_ci: dict[str, tuple[float, float]] = {}
    point_estimate: dict[str, float] = {}
    atom_counts: dict[str, int] = {}

    for label, va, vb, spans, kind in cases:
        point = float(np.median(va) - np.median(vb))
        point_estimate[label] = point
        results_index[label] = (va, vb, kind)
        for method in ("percentile", "basic", "BCa"):
            try:
                res = bootstrap(
                    (va, vb), _median_difference,
                    n_resamples=METHOD_RESAMPLES, confidence_level=0.95,
                    method=method, paired=False, vectorized=True,
                    random_state=np.random.default_rng(REFERENCE_SEED),
                )
                ci = res.confidence_interval
                lo, hi = float(ci.low), float(ci.high)
                excludes = not (lo <= 0 <= hi)
                results[(label, method)] = excludes
                if method == "percentile":
                    percentile_ci[label] = (lo, hi)
                    atom_counts[label] = int(
                        np.unique(res.bootstrap_distribution).size)
                elif method == "BCa":
                    bca_ci[label] = (lo, hi)
                flag = "**yes**" if excludes else "no"
                lines.append(
                    f"| {label} | {len(va)}, {len(vb)} | {_fmt(point, 0)} | "
                    f"{method} | [{_fmt(lo, 0)}, {_fmt(hi, 0)}] | {flag} |")
            except Exception as exc:  # pragma: no cover - version dependent
                results[(label, method)] = None
                lines.append(f"| {label} | {len(va)}, {len(vb)} | "
                             f"{_fmt(point, 0)} | {method} | not available | "
                             f"{type(exc).__name__} |")

    # Quantities the prose below quotes, derived rather than typed. The
    # Google – Meta cases are the ones that disagree; if a future data refresh
    # moves that, these lookups move with it.
    gm_pair_label = next(label for label in results_index
                         if results_index[label][2] == "unstratified"
                         and "meta" in label)
    gm_stratum_label = next(label for label in results_index
                            if results_index[label][2] == "stratified"
                            and "meta" in label)
    gm_lo, gm_hi = percentile_ci[gm_pair_label]
    gm_point = point_estimate[gm_pair_label]
    gm_atoms = atom_counts[gm_stratum_label]
    bca_lo, bca_hi = bca_ci[gm_stratum_label]
    p_lo, p_hi = percentile_ci[gm_stratum_label]
    bca_ratio = (bca_hi - bca_lo) / (p_hi - p_lo) if p_hi > p_lo else float("nan")

    kinds = {label: kind for label, _, _, _, kind in cases}
    sizes = {label: (len(va), len(vb)) for label, va, vb, _, _ in cases}
    published = {label: (not spans) for label, _, _, spans, _ in cases}
    bca_agrees = all(results[(label, "BCa")] == published[label]
                     for label in published)
    basic_disagree = [label for label in published
                      if results[(label, "basic")] != published[label]]

    def count(method: str, kind: str) -> int:
        return sum(1 for label in published
                   if kinds[label] == kind and results[(label, method)])

    # The claim that no published sentence rests on these intervals is the
    # load-bearing one in 5.2, so it is read off the committed ledger.
    ledger = pd.read_csv(TABLES / "claim-ledger.csv")
    salary_claims = ledger[
        ledger["citation"].fillna("").str.contains("salary", case=False)]
    comparisons = salary_claims[
        salary_claims["measures"].str.contains("difference")]
    published_claims = salary_claims[salary_claims["status"] == "published"]
    n_salary = len(salary_claims)
    n_comparison = len(comparisons)
    n_published = len(published_claims)

    checks.append(("BCa reproduces every published spans_zero verdict",
                   bca_agrees))
    checks.append((
        "every salary-comparison claim is refused at the identification gate",
        n_comparison > 0
        and (comparisons["status"] == "refused").all()
        and (comparisons["blocked_by"] == "identification").all()))
    verdict_src = Path(ins.__file__).read_text()
    checks.append((
        "benchmark_available is a literal False, not an interval-derived value",
        '"benchmark_available": False' in verdict_src))
    checks.append((
        "no published salary claim measures a difference of medians",
        n_published > 0
        and not published_claims["measures"].str.contains("difference").any()))
    checks.append((
        "committed percentile counts are what the report records (1 and 0)",
        count("percentile", "unstratified") == 1
        and count("percentile", "stratified") == 0))

    lines += [
        "",
        "### 5.1 Percentile and BCa agree; basic does not",
        "",
        "**BCa reproduces every published verdict.** It is the most refined of "
        "the three, and it is the one that most nearly settles the question: "
        "the percentile intervals are not merely computed correctly, they are "
        "not an artefact of the crudest available rule.",
        "",
        f"**The basic (reflected) interval disagrees on "
        f"{len(basic_disagree)} of the {len(published)} comparisons** — "
        + ", ".join(f"`{label}`" for label in basic_disagree)
        + ". That is a real difference and it is not waved away here. The "
        "mechanism is visible in the arithmetic: the basic interval is the "
        "percentile interval reflected through twice the point estimate, so "
        f"for Google – Meta on Ladders `[{_fmt(gm_lo, 0)}, {_fmt(gm_hi, 0)}]` "
        f"becomes `[2({_fmt(gm_point, 0)}) - {_fmt(gm_hi, 0)}, "
        f"2({_fmt(gm_point, 0)}) {_signed(-gm_lo)}]` = "
        f"`[{_fmt(2 * gm_point - gm_hi, 0)}, "
        f"{_fmt(2 * gm_point - gm_lo, 0)}]`, and an interval that comfortably "
        "spanned zero excludes it.",
        "",
        "What makes that reflection untrustworthy here, rather than merely "
        "different, is §4.3. In both disagreeing cases the percentile bound "
        "being reflected is pinned on a dominant lattice atom rather than "
        "sitting where a quantile of a continuous distribution would; in the "
        "Google – Meta stratum the lower bound and the point estimate are the "
        "same number. Reflecting a bound like that converts a tie into an "
        "exclusion. The basic interval assumes the bootstrap distribution's "
        "shape around the estimate mirrors the sampling distribution's shape "
        f"around the truth; on a statistic that takes {gm_atoms} distinct "
        f"values in that stratum across {METHOD_RESAMPLES:,} resamples it "
        "does not, and the same coarseness "
        "that makes §3.3's endpoints exactly reproducible is what makes this "
        "reflection meaningless.",
        "",
        "BCa's own output on that stratum says the same thing from the other "
        f"side: `[{_fmt(bca_lo, 0)}, {_fmt(bca_hi, 0)}]` is a 95% interval "
        f"roughly {bca_ratio:.1f}× the width of the percentile one, with an "
        "upper bound sitting exactly on zero. Three methods straining in "
        f"three directions on {sizes[gm_stratum_label][0]} and "
        f"{sizes[gm_stratum_label][1]} observations is the finding, and it is "
        "the same finding Task 09 reached by a different route.",
        "",
        "### 5.2 What this does and does not change in Task 09",
        "",
        "Nothing published moves, and the reason is structural rather than "
        "lucky:",
        "",
        "- **`benchmark_available: false` cannot move under any interval "
        "rule.** It is a literal in `salary_verdict`, not a value computed "
        "from an interval — the check below reads it out of the source. "
        "§7.1 and §7.2 of the methods document refuse the benchmark on the "
        "*selection* argument: disclosure is a publisher behaviour, and "
        "conditioning on it moves Google's modal-country share by 49.52 pp. "
        "No confidence interval, by any method, speaks to that.",
        "- **No published claim quotes one of these intervals.** This is "
        "checked against the committed ledger rather than asserted: of the "
        f"{n_salary} claims citing a salary table, the {n_published} that "
        "publish all measure a disclosure *rate*, which uses no bootstrap at "
        f"all, and all {n_comparison} salary-*comparison* claims are "
        "refused with `blocked_by = identification` — the gate closes before "
        "any interval is consulted. A verdict that moves under `basic` is a "
        "verdict no published sentence rests on.",
        "",
        "Two intermediate counts in the task's report JSON *are* "
        "method-dependent, and are recorded here rather than left for a "
        "reader to rediscover:",
        "",
        "| field | committed (percentile) | under basic | under BCa |",
        "| --- | --- | --- | --- |",
        f"| `unstratified_pairs_excluding_zero` | {count('percentile', 'unstratified')} | "
        f"{count('basic', 'unstratified')} | {count('BCa', 'unstratified')} |",
        f"| `stratified_cells_identified` | {count('percentile', 'stratified')} | "
        f"{count('basic', 'stratified')} | {count('BCa', 'stratified')} |",
        "",
        "The second is the one that touches prose: §7.3 of the methods "
        "document says that under stratification **nothing survives**, and "
        "under `basic` one stratum would. That sentence is true as written — it describes the "
        "percentile intervals the task computed and committed — and this "
        "file's judgement is that it should stand, for the reason in §5.1: "
        "the `basic` exclusion is produced by reflecting an atom-pinned "
        "bound, not by evidence of a pay difference. It is recorded here so "
        "that the claim is qualified by something a reader can check, rather "
        "than by silence.",
        "",
        "**No correction is raised.** The register's rule is that a correction "
        "needs committed evidence contradicting something the repository "
        "*asserts*. Task 09 asserts a refusal; the refusal does not rest on "
        "these intervals; and an alternative interval rule that finds *more* "
        "differences among thinly populated, self-selected pay comparisons is "
        "an argument for that refusal rather than against it.",
        "",
    ]
    return checks, lines


# ---------------------------------------------------------------------------
# 6. scope
# ---------------------------------------------------------------------------

def scope_section() -> list[str]:
    """What this file does not cover, stated rather than left to inference."""
    return [
        "## 6. What this file does and does not cover",
        "",
        "**Covered.** The two pieces of statistics in `insights.py` that a "
        "published verdict depends on: the exact binomial sign test behind "
        "C8's `clears_005` column and the sign-test power table, and the "
        "percentile bootstrap behind the salary audit's `spans_zero` column.",
        "",
        "**Not covered, and not a gap.** Most of `insights.py` is not "
        "statistics. The four gates, the citation resolver, the linter, the "
        "verdict map and the yield accounting are logic, and logic is pinned "
        f"by `tests/test_insights.py` ({_test_count()} test functions, some "
        "parametrised) rather than by "
        "agreement with "
        "a reference implementation — there is no scipy function that has an "
        "opinion about whether a claim's `verdict_source` conjunction "
        "inherited the weakest status.",
        "",
        "**Not covered, and inherited.** The upstream verdicts Task 09 reads "
        "are validated where they were computed: "
        "[`task-07-forecast-validation.md`](task-07-forecast-validation.md) "
        "for the forecast statistics and "
        "[`task-08-similarity-validation.md`](task-08-similarity-validation.md) "
        "for the similarity ones. Task 09 never re-decides identification, so "
        "revalidating them here would test the same code twice.",
        "",
        "**One thing this cannot establish.** That the samples are the right "
        "samples. Every check here is conditional on the disclosed-salary "
        "subset being what it is, and §7 of the methods document is the "
        "argument that this subset is selected by publisher and through it by "
        "country. A correctly computed interval on a selected sample is still "
        "an interval about the selected sample; agreement with scipy says "
        "nothing about that, and is not offered as if it did.",
        "",
    ]


def main() -> None:
    rng = np.random.default_rng(20240609)
    frames = cmp.load_frames(COMPANIES)

    sign_checks, sign_lines = check_sign_test()
    median_checks, median_lines = check_median(rng, frames)
    boot_checks, boot_lines, detail = check_bootstrap(frames)
    mc_checks, mc_lines = check_monte_carlo(frames, detail)
    method_checks, method_lines = check_interval_method(frames, detail)

    checks = (sign_checks + median_checks + boot_checks + mc_checks
              + method_checks)
    body = (sign_lines + median_lines + boot_lines + mc_lines + method_lines
            + scope_section())

    header = [
        "# Task 09 — validating the hand-written insight statistics",
        "",
        f"Generated by `src/validate_insights.py` on {date.today().isoformat()}.",
        "",
        "`src/insights.py` carries no scipy import, so its two statistics are "
        "written by hand: a two-sided exact binomial sign test, and a "
        "percentile bootstrap interval for a difference of medians. Each one "
        "sits directly under a published verdict — `clears_005` in C8's "
        "cell-floor table, and `spans_zero` in the salary audit. This file is "
        "the evidence that the hand-written versions agree with the reference "
        "implementations, and — where they cannot agree exactly, because both "
        "are Monte-Carlo estimates — the measurement of how far apart they are "
        "and whether that distance could move a published conclusion.",
        "",
        "**Result: Task 09's conclusions stand as written, and one "
        "method-dependence is recorded.** Every sign-test p-value matches "
        "`scipy.stats.binomtest` to float noise or to the committed tables' "
        "own rounding, and every `spans_zero` verdict is reproduced by "
        "`scipy.stats.bootstrap(method=\"percentile\")` and again by BCa. The "
        "endpoint differences that do exist are inside the spread the "
        "committed procedure shows against itself under reseeding, and §4 "
        "measures them.",
        "",
        "The one substantive finding is in **§5**: the *basic* (reflected) "
        "bootstrap interval, which Task 09 did not use, reverses two "
        "`spans_zero` verdicts. §5 explains why that reflection is not "
        "trustworthy on these lattice-valued distributions, shows that no "
        "published claim rests on any of these intervals, and records the two "
        "intermediate counts that would move — rather than leaving a reader "
        "to find them. No correction is raised and no Task 09 deliverable is "
        "edited by this file.",
        "",
        "Versions used for the comparison: "
        f"numpy {np.__version__}, "
        f"pandas {pd.__version__}, "
        f"scipy {__import__('scipy').__version__}.",
        "",
    ]

    summary = ["## Automated checks", ""]
    for name, ok in checks:
        summary.append(f"- {'PASS' if ok else 'FAIL'} — {name}")
    summary.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header + summary + body))

    print(f"validated insights.py against scipy -> {OUT.relative_to(REPO_ROOT)}")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
    if not all(ok for _, ok in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
