# Task 08 — Company Similarity Scoring

**Team standard.** Task 06 put six companies on one axis and said which
comparisons that axis supports. Task 07 tried to extend them forward and
refused. Task 08 is asked how similar the companies are to each other. This
document records what a similarity number is computed on, what it has to be
compared against before it means anything, how much of the ordering survives
resampling, and which of the differences between companies are differences in
demand rather than differences in vocabulary. Tasks 09 and 10 read the output,
and a heatmap is the easiest chart in this repo to over-read: it is dense, it
is colourful, and every cell looks equally certain.

- **Code:** [`src/similarity.py`](../src/similarity.py) · [`src/build_similarity.py`](../src/build_similarity.py) · [`src/validate_similarity.py`](../src/validate_similarity.py)
- **Tests:** [`tests/test_similarity.py`](../tests/test_similarity.py) (37; 478 in the suite)
- **Validation:** [`docs/task-08-similarity-validation.md`](task-08-similarity-validation.md) — every hand-written statistic checked against scipy
- **Google findings:** [`members/ankit-google/task-08-similarity-report.md`](../members/ankit-google/task-08-similarity-report.md)
- **What this task overturned:** [`docs/corrections.md`](corrections.md) —
  [C6](corrections.md#c6--neither-concept-skills-nor-rare-skills-dominate-a-similarity-score),
  two Task 04 predictions about what would dominate a similarity score
- **Inherited from:** [`docs/task-06-competitor-comparison-methods.md`](task-06-competitor-comparison-methods.md) §11 (the comparable denominator) and [`docs/task-07-demand-forecasting-methods.md`](task-07-demand-forecasting-methods.md) §14 (the gate, and February)
- **Legal position:** unchanged — no new source, no new collection, no new field

```bash
python src/build_similarity.py         # 16 tables, 8 figures, 1 JSON report
python src/validate_similarity.py      # scipy cross-check, 17 checks
python -m pytest tests/ -q
```

**A note on this task's name.** The brief and the repo README call Task 08
*Company Similarity Scoring*, with "similarity tables + heatmaps/network
graphs" as the deliverable. Two earlier documents forecast it under different
names — §11 of the Task 06 methods calls it "Visualisation", §11 of the Google
Task 06 report calls it "Evaluation". The brief wins. This is registered as
[C7](corrections.md#c7--task-08-is-company-similarity-scoring-not-visualisation-and-not-evaluation),
because it is not only a naming drift: the Task 06 report also handed over two
instructions written for the task it thought this was — make standardised
shares the primary object, and build a skill-level significance baseline —
and an instruction written for a task that does not exist gets followed
anyway. Neither is right for a similarity task (§6.2, §5). The Task 06 methods
document's instructions, filed under the wrong name, *were* followed: every
cross-company figure carries its verdict and no raw count is plotted by
company on a shared axis.

The general form of the lesson, which applies to this document's own §14 as
much as to Task 06's §11: **a handover section is a prediction, not an
instruction.** Read the brief first.

Four findings in one line each, so the rest of the document reads as their
justification:

1. **One pair of fifteen has an identified rank.** Google – Meta is rank 1 in
   100% of 600 resamples. Eleven of the fifteen pairs share a single tier
   (§5). The heatmap has fifteen numbers on it and supports two orderings.
2. **A raw similarity number is uninterpretable.** All fifteen pairs score
   0.50–0.92, but two identical companies at these sample sizes would score
   0.99, and two unrelated ones 0.10–0.19. Against that ruler the pairs
   re-spread to 0.43–0.91 (§4).
3. **The biggest single lever is not the metric — it is whose products count
   as skills.** Dropping vendor self-reference moves one pair 0.30 and moves
   ranks by up to 9 places (§6.1). The metric family moves ranks by up to 11
   (§3.3).
4. **Trajectory similarity is refused** (§8). One of fifteen pairs passes
   Task 07's gate on both members; the mean correlation sits inside a null
   built from independent series under closure; the mean interval spans 1.04
   of a 2-wide range.

---

## 1. What is being compared

### 1.1 The object is a share-of-skilled-postings profile

For each company, a vector over a shared skill vocabulary, where entry *s* is
the fraction of that company's **skilled postings** that mention skill *s*.
Not counts — a count vector would make Meta similar to itself at scale and
nothing else. Not share of all postings: Task 06 §2 fixed the denominator as
postings that mention at least one skill, after excluding the
Facilities/Operations job function, and this task inherits that whole
(`SKILL_EXCLUDED_FUNCTIONS`, `FORBIDDEN_COLUMNS`). The six denominators are
340 / 590 / 838 / 504 / 232 / 459 postings.

The vocabulary is the union of skills that clear Task 04's `min_postings`
floor in at least one company — 127 skills here. A company that never mentions
a skill carries a genuine 0 for it, which is information; a skill no company
mentions is not in the vocabulary at all.

### 1.2 Profiles are not a composition, and are not treated as one

The entries are independent prevalences, not parts of a whole: a posting can
ask for Python *and* SQL, so the vector does not sum to 1 and must not be
normalised to make it. This matters in §8, where the object under comparison
*is* a composition (share of a common panel) and the closure geometry that
follows is the whole difficulty.

### 1.3 The comparison is symmetric, and that is a limitation

Every metric here is symmetric: sim(A, B) = sim(B, A). "Databricks looks like
a small Snowflake" and "Snowflake looks like a large Databricks" are the same
number. A directional statement — who is converging on whom — is a trajectory
question, and §8 refuses trajectory questions on this data.

---

## 2. The order of operations, and why it is fixed

Choose the metric, then look. Every design decision in this task was fixed
before the ranking was read, because a similarity ranking has fifteen numbers
and five metrics and thirteen thresholds, and the space of defensible-looking
choices is large enough to reach almost any headline:

1. `PRIMARY_METRIC = "cosine"` — declared in the module, justified in §3.2.
2. `RANK_STABILITY_FLOOR = 0.90` — declared before the bootstrap ran.
3. `N_BOOTSTRAP = 600`, `N_NULL = 300`, `SEED = 20260818` — fixed, so the
   committed tables rebuild bit-for-bit.
4. Every sensitivity in §6 is **published**, not applied. None of them
   filters the primary table.

The last rule is the one that costs something. Vendor self-reference (§6.1)
looks like contamination, and dropping it would produce a cleaner story. It is
published as a sensitivity because "Snowflake postings ask for Snowflake" is a
fact about Snowflake's demand, not an artefact of collection.

---

## 3. Five metrics, two families, one headline

### 3.1 What is computed

| Metric | Family | Reads the profile as |
| --- | --- | --- |
| `cosine` | prevalence-weighted | a direction in skill space |
| `jensen_shannon` | prevalence-weighted | a distribution (bounded, symmetric KL) |
| `bray_curtis` | prevalence-weighted | shared abundance |
| `spearman` | rank-or-set | an ordering of skills |
| `jaccard_supported` | rank-or-set | a set of supported skills |

All five are hand-rolled in `src/similarity.py` (§11) and all five are
committed for every pair in `similarity-pairs.csv`.

### 3.2 Why cosine leads

The brief asks how similar the *demand* these companies express is. A skill
that half of one company's postings ask for should count for more than one
that 1% ask for, which rules the rank-or-set family out of the headline: they
answer "do they want the same things at all", not "do they want them in the
same proportions". Among the three prevalence-weighted metrics, cosine is
scale-free in a way the other two are not — it compares the shape of demand
without being moved by a company that simply lists more skills per posting.

This is a choice among defensible alternatives, and it changes the answer. It
is stated here, before the ranking, rather than defended after it.

### 3.3 The families disagree, and the disagreement is published

`metric-concordance.csv` gives rank correlations for all ten metric pairs.
Within-family correlations run **0.58 to 0.96**; cross-family run **−0.04 to
0.58**. The two ranges overlap — the lowest within-family pair (0.5821) is
barely above the highest cross-family one (0.5750) — so "family" is a useful
label, not a clean partition.

The sharper reading is that **cosine is the outlier**, not the families.
Cosine correlates 0.72 with the other prevalence-weighted metrics but −0.03
with Spearman and −0.04 with Jaccard: on this data the primary metric is the
one that disagrees most with the rest. One pair (Databricks – Snowflake) moves
**11 of 15 places** across metrics. Figure 03 draws it.

A reader who wants "do they list the same skills" gets a different ordering
from a reader who wants "do they weight them the same way", and neither is
wrong. This is why every verdict in §9 carries its metric spread.

---

## 4. A raw similarity number means nothing on its own

Cosine on non-negative vectors lives in [0, 1] and is crowded near the top:
0.65 sounds middling and is not. The fix is two nulls per pair, at that pair's
own sample sizes.

### 4.1 The two ends of the ruler

- **Identical null** (`null_identical`) — draw both companies' postings from a
  single pooled skill profile, at the two real posting counts, and score them.
  This is what "the same company measured twice" scores given this much data.
  Range here: **0.9893 – 0.9957**. It is below 1 because sampling noise alone
  costs about half a percent at these n.
- **Unrelated null** (`null_unrelated`) — permute one company's skill labels,
  destroying the correspondence but preserving both marginals. This is what
  "no relationship, same overall skill prevalences" scores. Range here:
  **0.0956 – 0.1889**. It is well above 0 because Python and SQL are common
  everywhere.

Both are recomputed per pair, because both depend on the pair's own posting
counts and marginals.

### 4.2 The calibrated score

```
calibrated = (observed − unrelated) / (identical − unrelated)
```

0 means "no more similar than chance given these marginals", 1 means "as
similar as the same company measured twice". The fifteen pairs run **0.4288 to
0.9123** calibrated, against **0.4961 to 0.9174** raw.

Calibration is not cosmetic: **six of the fifteen pairs change rank** (rank
correlation 0.9643, max move 3 places). The pairs that fall are NVIDIA's.
NVIDIA's unrelated null is the highest of the six (0.165–0.189 against
Databricks – Meta's 0.096), because its marginals sit on skills everyone else
also asks for — so chance alone already gets an NVIDIA pair most of the way to
0.65, and less of its raw score is left to be explained by a real
correspondence. Microsoft – NVIDIA is raw rank 3 and calibrated rank 6.

Its second job is to establish that **all fifteen pairs are distinct from
identical and above unrelated**: every pair of these six companies is
detectably related and detectably not the same company. Figure 02 draws each
pair against its own two nulls.

---

## 5. Uncertainty: the bootstrap resamples postings, not shares

### 5.1 Why the resampling unit matters more than the draw count

A share-level bootstrap — resampling the 127 skill shares — would produce
intervals that ignore how many postings the shares were computed from, and
Databricks (232 skilled postings) would come out as certain as Meta (838).
`bootstrap_pairs` resamples **postings within each company**, with
replacement, at that company's own n, and recomputes the whole profile from
the resampled incidence matrix.

The signature of doing this correctly is that interval width follows 1/√n. A
company with 25 postings gets an interval about **4×** wider than one with 400
— `tests/test_similarity.py::test_bootstrap_width_tracks_posting_count` brackets
the observed ratio in (2.5, 6.0), which a share-level bootstrap (ratio ≈ 1)
cannot pass.

### 5.2 Rank stability, and the floor that was declared first

600 draws. For each draw the fifteen pairs are re-scored and re-ranked;
`rank_stability` is the fraction of draws in which a pair holds its published
rank. `RANK_STABILITY_FLOOR = 0.90` was fixed in the module before the numbers
existed.

Two pairs clear it, both at 1.00: Google – Meta at rank 1 and Google – NVIDIA
at rank 2. The rest run 0.21 to 0.77. Mean pair CI width is 0.0599.

### 5.3 Tiers, not a ranking

`rank_tiers` merges pairs whose bootstrap rank intervals overlap. The fifteen
pairs collapse to **4 tiers**, and the largest holds **11 pairs**. Figure 04
draws the intervals rather than the ranks.

The honest summary of the ordering is: Google – Meta is first, Google – NVIDIA
is second, eleven pairs are indistinguishable, and the bottom two are
indistinguishable from each other. A table of fifteen ranks is available in
`pair-verdicts.csv` and eleven of its rows are decoration.

---

## 6. What actually moves a similarity score

Four sensitivities, all published as tables, none applied to the headline.

### 6.1 Vendor self-reference — the largest lever (`vendor-sensitivity.csv`)

`VENDOR_SKILLS` maps a company to the skills that are its own product or
originated with it (`own_product`, `origin` in the Task 04 taxonomy relations)
— Databricks and Spark, Snowflake, Azure, PyTorch, CUDA, BigQuery. A posting
mentioning them is partly describing the company rather than the labour market.

Dropping every company's own products moves Microsoft – Snowflake by **0.2989**,
moves ranks by up to **9 places**, and drops the rank correlation with the
published ordering to **0.6071**. It also changes the cluster structure (§7.2).

It is not applied, per Task 06 §7: a Snowflake job asking for Snowflake is a
real demand signal. It is the first thing a reader should check against, which
is why §9 carries a `vendor_dependent` verdict class and one pair earns it.

### 6.2 Role mix (`mix-sensitivity.csv`)

Companies hire different mixes of job functions, and job functions have
different skills, so part of any similarity is mix rather than demand. Direct
standardisation to a pooled `job_function` mix (Task 06's
`standardised_skill_table`) gives rank correlation **0.90**, max rank move
**4**, max effect on a single pair **0.0794**.

Mix is a real but second-order effect here — a third of the vendor lever.

### 6.3 Support restriction (`support-sensitivity.csv`)

Restricting to the **11** skills every company supports gives rank correlation
0.7214 with the full-vocabulary ranking. This is published as a *different
question* — "do they agree about the skills everyone measures" — and
explicitly **not** as a robustness check, for the reason in §6.4.

### 6.4 The rare-skill sweep is arithmetic, not evidence

This is [C6](corrections.md#c6--neither-concept-skills-nor-rare-skills-dominate-a-similarity-score),
and it is the finding this task contributes back to the team's methods.

A skill's contribution to a cosine numerator is the **product of two shares**.
A skill half of each company asks for contributes ≈0.25; a skill one posting
in six hundred asks for contributes ≈0.0000028 — and it contributes that
whether it occupies one column or a hundred. **Column count is not weight.**
`numerator-contribution.csv` measures it on the real vocabulary:

| Group | Skills | Share of the cosine numerator (mean) |
| --- | --- | --- |
| Concept skills (`is_concept`) | 8 | **0.36%** |
| Skills in ≤ 1 posting | 8 | **0.00%** — exactly |
| Skills in ≤ 10 postings | 39 | **0.005%** |
| Top 5 skills of the pair | 5 | **80.5%** |

Removing all eight concept skills leaves the ranking **identical** (rank
correlation 1.0, no pair moves, largest cosine shifts 0.0020 on a
0.4961–0.9174 range).

The consequence for anyone writing one of these reports: **dropping rare
skills is not a robustness check on a prevalence-weighted similarity**, and a
report that runs the sweep and announces the ranking survived it is claiming
credit for arithmetic. It *is* a robustness check on Jaccard, which counts
columns — and Jaccard's ranking correlates −0.04 with cosine's. Name the
metric before predicting what will dominate it.

---

## 7. Structure: clusters and networks, both with their instability drawn

### 7.1 Hierarchical clustering is average linkage on 1 − similarity

`average_linkage` is a hand-rolled UPGMA; `cut_tree` cuts it at k clusters.
Nothing here needs scipy, and a 6×6 distance matrix does not need speed.

### 7.2 Cluster support is bootstrapped, and the vocabulary changes the answer

`cluster_support` re-runs the whole clustering on each of 600 posting-level
resamples and reports how often the published partition reappears
(`cluster-support.csv`):

| k | Partition (all skills) | Support | Support after dropping own products |
| --- | --- | --- | --- |
| 2 | Databricks \| the other five | 0.868 | 0.543 |
| 3 | Databricks \| Google, Meta, Microsoft, NVIDIA \| Snowflake | 0.853 | 0.967 |
| 4 | Databricks \| Google, Meta, NVIDIA \| Microsoft \| Snowflake | 0.863 | 0.678 |

The two-cluster split is well supported on the full vocabulary and **not
supported** once own products are dropped — and at k = 3 and k = 4 the
partition itself changes, not just its support. NVIDIA moves out of the
central group; Snowflake moves into it. Figure 06 draws both dendrograms side
by side for exactly this reason: a single dendrogram would look like a fact.

### 7.3 The network graph is a threshold sweep, not a picture

An edge exists where similarity ≥ t. `network-thresholds.csv` sweeps t from
0.40 to 1.00 and reports edges, components and isolates. The graph is complete
(15/15 edges) at t ≤ 0.45 and empty at t ≥ 0.95; a **4-step plateau** runs
from 0.75 to 0.90 where the picture does not change at all.

There is no natural threshold in this data. Publishing one network graph would
be publishing one arbitrary choice, so Figure 07 draws the sweep with the
plateau marked, and the graph at a stated t beside it.

---

## 8. Trajectory similarity is refused

### 8.1 The question, and the three gates it has to clear

"Do these companies' demand curves move together?" — computed as the
correlation of log share of the common panel, pair by pair, on Task 07's
`panel-share-series.csv`.

Three things have to hold before an answer is publishable:

1. **Both members pass Task 07's forecastability gate.** Task 07 §14 requires
   the gate on both members of a pair, not one. Result: **1 of 15 pairs**.
2. **The intervals have to mean something.** Fisher-z intervals on 11 points.
   Mean 95% interval width is **1.0449** on a range of 2 — the average pair's
   interval covers more than half the possible values. Max is 1.1997.
3. **The correlation has to exceed what closure produces on its own.**

February is never filled. Task 07 §14 is explicit: a similarity measure that
fills February with zero finds companies similar because they were
simultaneously invisible. `panel_wide` intersects observed periods, and
February is absent from every pair's panel.

### 8.2 The closure null, and the σ that was hiding in it

Shares of a common panel must sum to 1 in every period. If one company rises,
the others must fall — so *independent* series, once closed, correlate
negatively. The analytic expectation for D parts is **−1/(D−1)** = −0.20 at
D = 6, and the observed mean r is −0.1746, which is on the same side and the
same order.

The null therefore has to be simulated, not asserted, and it has one free
parameter: the dispersion σ of the underlying independent series. A σ picked
for convenience produces a band of a convenient width, so `calibrate_sigma`
fixes it from the data — bisection on the panel's own CLR spread (0.8047),
which is monotone in σ and therefore deterministic. It returns **σ = 0.8901**,
reproducing a spread of 0.8046.

At the calibrated σ the null band is **[−0.1785, −0.0297]** and the observed
mean **−0.1746 sits inside it, near the lower edge**.

`closure-sensitivity.csv` sweeps σ from 0.2 to 1.4. The clause holds at every
σ up to 1.0 and first fails at **σ = 1.2** — and it fails **below** the band,
which means the observed co-movement is *more negative* than independent
series produce, not less. There is no σ at which this data shows co-movement.

### 8.3 −1/(D−1) is exact for geometric closure, not arithmetic

Worth stating because it is easy to check the wrong thing. **−1/(D−1) is exact
for closure by the geometric mean** — the CLR transform — and only approximate
for the arithmetic closure a share table actually has. Under arithmetic
closure the mean correlation sits *above* the analytic value, and the gap
widens with σ: **+0.013** at σ = 0.2, **+0.067** at the calibrated σ = 0.8901,
**+0.117** at σ = 1.4.

Nothing this task publishes uses the analytic number; `trajectory_verdict`
reads the simulated band, which is arithmetically closed like the data.
`validate_similarity.py` verifies both halves — that geometric closure
reproduces −1/(D−1) to ~1e-5, and that arithmetic closure departs from it
toward zero with σ. Figure 08 draws the dashed analytic line and the simulated
band separately, labelled, so they are not read as the same object.

### 8.4 The refusal rests on three legs, and two do not touch the null

This matters because the null is the part with a modelling choice in it. Even
if the closure band were wrong: 1 of 15 pairs is eligible, and the average
interval spans half the range. The verdict is not identified, and it would not
become identified by moving σ.

`aitchison_variation` is committed alongside — `var(log(x_a/x_b))`, which is
scale-invariant and free of the closure problem. It is descriptive: Google –
Meta is the lowest-variation pair (0.2156), NVIDIA – Snowflake the highest
(1.6738). No verdict rests on it.

---

## 9. The verdict table is the deliverable

`pair-verdicts.csv` gives every pair a class, and the class carries its reason:

- **`robust`** — rank stability ≥ 0.90, no rank move under the vendor or mix
  sensitivity, and a small family spread. **1 pair**: Google – Meta.
- **`vendor_dependent`** — otherwise robust, but the ranking depends on
  counting a company's own products as skills. **1 pair**: Google – NVIDIA,
  which moves 7 places when own products are dropped.
- **`unresolved`** — the rank does not survive resampling. **13 pairs**.

The publishable headline is **"Google – Meta is the most similar pair"** and
it is the only pair-level claim in this task with all four supports under it.
Everything else needs its verdict clause attached, in the same way Task 06's
cross-company claims need their comparability verdict.

---

## 10. What this task overturned

[C6](corrections.md#c6--neither-concept-skills-nor-rare-skills-dominate-a-similarity-score)
— Task 04's taxonomy predicted twice that this task would be dominated by
things that cannot dominate it: concept skills (§2.3) and single-posting
columns (§6.1). Both predictions were made about a metric that had not been
chosen yet, and both would have been right for a set metric. Neither design
decision changes — the concept/tool split and the `min_postings=5` floor both
earn their place for other reasons — but the taxonomy's two sentences are
marked in place with a pointer, per the register's standard.

The general lesson is now item 5 on the register's checklist: **a statement
about "similarity" that would be true of Jaccard and false of cosine is not a
finding.**

---

## 11. Uncertainty, and why this module imports no scipy

Same rule as Tasks 05–07: `src/similarity.py` imports numpy and pandas and
nothing else statistical. Every metric, both nulls, the bootstrap, the Fisher
intervals, the UPGMA linkage and the CLR arithmetic are written out. The
reason is that a hand-written statistic can be read and argued with, and a
`scipy.spatial.distance.cosine` call cannot be checked by a marker who is
reading the repo rather than running it. Wilson intervals, Newcombe
differences and BH-FDR come across from Task 06 unchanged.

`src/validate_similarity.py` is the exception and exists for this purpose. It
imports scipy and re-derives 17 checks against it — the five metrics, the
Fisher intervals, the rank correlations, the linkage merge order, both nulls,
and both halves of the closure geometry. All 17 pass. If you change anything
in `similarity.py` that computes a statistic, re-run it and regenerate
[`docs/task-08-similarity-validation.md`](task-08-similarity-validation.md).

---

## 12. What gets committed

**Committed** — 16 aggregate tables in `members/<you>/task-08-tables/`, 8
figures in `task-08-figures/`, and one `task-08-similarity-report.json`
carrying every headline number with its seed. The JSON is what Tasks 09 and 10
should read; the markdown report is for humans.

**Not committed** — anything row-level. `data/processed/` stays git-ignored,
as it has since Task 03. `forbidden_columns` and
`personal_data_columns_present` run over all 16 tables before they are
written, and the report records the result (`privacy.passed`). A country
column would compare aggregator footprints, not companies; `share_of_all` is
the wrong denominator. Both would be harder to argue with inside a heatmap
than inside a table, which is why the check runs here at all.

---

## 13. Checklist for each specialist

1. Build profiles on **skilled postings**, Facilities/Operations excluded —
   Task 06's denominator, not a new one.
2. Declare your primary metric in the module before you look at the ranking,
   and publish all five.
3. Publish the metric concordance. If your families agree everywhere, say so;
   if one metric is an outlier, name it.
4. Never quote a raw similarity without its two nulls. 0.65 is meaningless
   until the reader knows that identical scores 0.99 and unrelated scores 0.15.
5. Bootstrap **postings**, not shares. Check the 1/√n width law — if a thin
   company's interval is not wider, your resampling unit is wrong.
6. Set your rank-stability floor before the bootstrap runs, and report tiers
   rather than ranks when the intervals overlap.
7. Publish the vendor sensitivity. Do not apply it.
8. Do not present a rare-skill sweep as a robustness check on a
   prevalence-weighted metric (§6.4).
9. Apply Task 07's gate to **both** members of any trajectory pair, and never
   fill February.
10. If you compute a correlation on shares of a common panel, simulate the
    closure null at the panel's **own** dispersion, and sweep σ.
11. Sweep the network threshold; do not publish one graph as if the threshold
    were given.
12. Re-run `python src/validate_similarity.py` if you touch anything in
    `similarity.py` that computes a statistic.

---

## 14. What Tasks 09 and 10 inherit

- **Task 09 (Insight generation):** exactly one pair-level sentence is
  available — **"Google and Meta express the most similar skill demand of any
  pair among the six, and that ordering is stable under resampling, metric
  choice, own-product removal and role-mix standardisation."** Every other
  pair needs its `unresolved` clause. Two structural sentences are also
  available: that all fifteen pairs are detectably related and detectably
  distinct (§4.2), and that Databricks separates from the other five at k = 2
  on the full vocabulary — with the caveat that the split does not survive
  dropping own products. There is **no convergence sentence**: §8 refuses
  trajectory similarity, so "X is becoming more like Y" is unsupported.
- **Task 10 (Final presentation):** if a heatmap appears, it appears with the
  calibration ruler beside it — Figure 01 is the pattern, raw and calibrated
  on one canvas — and it appears next to Figure 04, so the reader sees that
  eleven of fifteen cells are one tier. A dendrogram appears only in the
  two-panel form of Figure 06. A network graph appears only with the threshold
  sweep beside it. A single-threshold graph, or a heatmap without its tiers,
  would show fifteen equally confident numbers where the data supports two.
