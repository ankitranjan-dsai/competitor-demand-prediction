# Task 08 — Company Similarity Scoring Report (Google)

**Specialist:** Ankit Ranjan · **Company:** Google (Alphabet) · **Date:** 2026-08-18

Input: the six-company skill profiles Task 06 built — **590 Google skilled
postings** against Databricks 340, Meta 838, Microsoft 504, NVIDIA 232 and
Snowflake 459, over a shared vocabulary of **127 skills**. Output: 16 aggregate
tables, 8 figures, a machine-readable evidence report, and a scipy cross-check
of every statistic written by hand.

- **Method rationale (team standard):** [`docs/task-08-company-similarity-methods.md`](../../docs/task-08-company-similarity-methods.md)
- **Code:** [`src/similarity.py`](../../src/similarity.py) · [`src/build_similarity.py`](../../src/build_similarity.py) · [`src/validate_similarity.py`](../../src/validate_similarity.py)
- **Tests:** [`tests/test_similarity.py`](../../tests/test_similarity.py) (37) — **474 passing** in the suite
- **Validation evidence:** [`docs/task-08-similarity-validation.md`](../../docs/task-08-similarity-validation.md) — 17/17 checks pass
- **Machine-readable report:** [`task-08-similarity-report.json`](task-08-similarity-report.json)
- **Tables:** [`task-08-tables/`](task-08-tables/) · **Figures:** [`task-08-figures/`](task-08-figures/)
- **What this task overturned:** [`docs/corrections.md`](../../docs/corrections.md) — [C6](../../docs/corrections.md#c6--neither-concept-skills-nor-rare-skills-dominate-a-similarity-score)

```bash
python src/build_similarity.py
python src/validate_similarity.py
python -m pytest tests/ -q
```

**Every number below is within-2023, from a single collection window and a
single upstream dataset**, on postings that mention at least one skill after
the Facilities/Operations function is excluded. None of it is a claim about
what these companies do; it is a claim about what their job postings ask for,
on the boards we can see, relative to five named rivals.

---

## 1. The headline: one of Google's five pairs is a finding

Google appears in five of the fifteen pairs. Ranked by cosine on skill share:

| Pair | Cosine | Rank | Rank stability | Verdict |
| --- | --- | --- | --- | --- |
| **Google – Meta** | **0.9174** | 1 | **1.00** | **robust** |
| Google – NVIDIA | 0.7485 | 2 | 1.00 | vendor_dependent |
| Google – Microsoft | 0.6707 | 4 | 0.32 | unresolved |
| Google – Snowflake | 0.6160 | 9 | 0.77 | unresolved |
| Databricks – Google | 0.4961 | 15 | 0.60 | unresolved |

**Google – Meta is the most similar pair among the six companies, and it is
the only pair in this task with all four supports under it**: rank 1 in 100%
of 600 posting-level resamples, rank 1 under every one of the five metrics,
rank 1 after every company's own products are removed, and rank 1 after
standardising to a pooled job-function mix. Nothing else in this task is that
well supported.

The rest of Google's table is weaker than it looks. Google – NVIDIA holds
rank 2 in every resample but falls to rank 9 when own products are dropped
(§4.1). Google – Microsoft holds rank 4 in fewer than a third of resamples.
The bottom pair, Databricks – Google, is the least similar pair of the fifteen
— but it is indistinguishable from the fourteenth (0.4972), so "least similar"
is a two-way tie, not a placing.

---

## 2. Google and Meta ask for the same five things

The number is high because the top of both profiles is nearly the same list.
Five skills — **Python, SQL, R, C++, Java** — carry **93.2%** of the
Google – Meta cosine numerator, the highest concentration of any pair:

| Skill | Google | Meta | Microsoft | NVIDIA |
| --- | --- | --- | --- | --- |
| Python | 0.631 | 0.869 | 0.474 | 0.655 |
| SQL | 0.432 | 0.857 | 0.417 | 0.228 |
| R | 0.363 | 0.553 | 0.151 | 0.039 |
| C++ | 0.173 | 0.165 | 0.202 | 0.233 |
| Java | 0.224 | 0.086 | 0.278 | 0.168 |

Meta asks for each of the first three more often than Google does — Meta's
postings are more uniformly data-analytical — but the *shape* is the same, and
cosine reads shape. R is the discriminating one: a third of Google's postings
and over half of Meta's ask for it, against 4% of NVIDIA's.

The similarity is a similarity of **stated demand in job postings**, and the
pair is unusual in that both companies post a large volume of generalist
data-and-ML roles described in generic language. It is not evidence that
Google and Meta run similar engineering organisations.

---

## 3. A raw 0.92 means nothing until you know what 1.0 costs

All fifteen pairs score between 0.50 and 0.92, which invites "everyone is
fairly similar". They are not on the same scale as intuition. At these posting
counts:

| Ruler point | Google – Meta | Databricks – Google |
| --- | --- | --- |
| Two draws from one pooled profile ("identical") | 0.9957 | 0.9930 |
| Label-permuted, marginals preserved ("unrelated") | 0.1026 | 0.1231 |
| **Observed** | **0.9174** | **0.4961** |
| **Calibrated** | **0.9123** | **0.4288** |

Two identical companies would score 0.9957, not 1.0 — half a percent is lost
to sampling noise at 590 and 838 postings. Two unrelated ones score 0.10, not
0.0, because Python and SQL are everywhere. Against that ruler Google – Meta
sits **91% of the way** from unrelated to identical, and Databricks – Google
sits 43% of the way.

**All fifteen pairs are distinct from identical and above unrelated.** Every
pair of these six companies is detectably related and detectably not the same
company — which is the strongest statement this task can make about the set as
a whole.

Calibration is not just a rescaling: six of the fifteen pairs change rank, at
most three places. NVIDIA's pairs fall, because NVIDIA's unrelated null is the
highest of the six (0.165–0.189) — its marginals sit on skills everyone else
also asks for, so chance alone already gets an NVIDIA pair most of the way to
0.65. Microsoft – NVIDIA is raw rank 3 and calibrated rank 6.

---

## 4. What would change Google's ranking

### 4.1 Own products — the largest lever, and it is aimed at Google

Google's vendor vocabulary is the largest of the six: 21 skills that are
Google's own product or originated there (GCP, BigQuery, TensorFlow, Kubernetes,
Go, Angular, Looker, Vertex AI, …), against Snowflake's one and NVIDIA's two.
Microsoft's is 18. Dropping every company's own products removes 35 of the 127
skills and moves Google's pairs more than anyone's:

| Pair | All skills | Own products dropped | Δ | Rank move |
| --- | --- | --- | --- | --- |
| Google – Meta | 0.9174 | 0.9460 | +0.029 | 0 |
| **Google – Microsoft** | 0.6707 | **0.9417** | **+0.271** | 4 → 2 |
| Databricks – Google | 0.4961 | 0.7462 | +0.250 | 15 → 13 |
| Google – Snowflake | 0.6160 | 0.8465 | +0.231 | 9 → 6 |
| Google – NVIDIA | 0.7485 | 0.8093 | +0.061 | **2 → 9** |

**Google and Microsoft look dissimilar mainly because they are two vendors
with two stacks.** Strip both stacks and they are the second most similar pair
in the set — the two largest own-product vocabularies in the panel are being
compared against each other, and neither can score on the other's. The gap
0.67 → 0.94 is the largest own-product effect involving Google and the third
largest in the task, behind Microsoft – Snowflake (0.299) and
Databricks – Snowflake (0.288).

Google – NVIDIA moves the other way: its rank falls seven places, not because
the pair got less similar (it gains 0.061) but because everyone else gained
more. That is what earns it the `vendor_dependent` verdict — the pair's
*position* is an artefact of the vocabulary, even though its score is not.

**None of this is applied.** A Google posting asking for BigQuery is a real
demand signal about Google, and Task 06 §7 fixed the rule that vendor terms
are published as a sensitivity rather than filtered out. But no reader should
take Google – Microsoft at 0.67 as a statement about the two companies' skill
demand without also seeing the 0.94.

### 4.2 Role mix is a third-order effect for Google

Standardising to a pooled job-function mix moves Google's largest pair by
**0.0066** and its largest effect anywhere in Google's row is **0.0365**
(Google – NVIDIA). One of Google's five pairs moves rank, by one place. Google
and Meta are not similar because they post the same *kinds* of role; the
similarity survives holding the role mix fixed.

### 4.3 The metric matters more than the mix

Google – Meta is rank 1 under all five metrics — spread 0. Google – NVIDIA
spans **4 ranks** across metrics and Google – Microsoft spans 2. Across the
whole task, one pair (Databricks – Snowflake) moves **11 of 15 places**
depending on which metric is read, and the primary metric is the one that
disagrees most with the others: cosine correlates 0.72 with the two other
prevalence-weighted metrics but **−0.03 with Spearman and −0.04 with Jaccard**.

Restricting to the 11 skills every company supports — a different question,
not a robustness check (§6) — moves Google – Snowflake up six places and
Google – NVIDIA down four. On the skills everyone measures, Google is similar
to everybody: all five of its pairs score above 0.76 and four above 0.92. That
comparison is nearly uninformative, which is the point of publishing it
separately.

---

## 5. Only two ranks in this task are identified, and Google holds both

600 posting-level bootstrap resamples, with `RANK_STABILITY_FLOOR = 0.90`
declared before the numbers existed:

- **2 of 15 pairs** clear the floor — Google – Meta and Google – NVIDIA, both
  at 1.00.
- The fifteen pairs collapse to **4 tiers** by overlapping bootstrap rank
  intervals, and **11 of them are one tier**.
- Mean pair interval width is 0.0599.

Google – Microsoft at rank 4, Google – Snowflake at rank 9 and
Databricks – Google at rank 15 are all inside that eleven-pair tier or tied at
its edge. **The middle of this ranking is not a ranking**, and Figure 04 draws
the intervals rather than the ranks so it cannot be read as one.

The interval widths behave the way posting-level resampling requires:
**NVIDIA**, the thinnest company at 232 skilled postings, carries the widest
intervals in the set — mean 0.0806, against 0.0499 to 0.0599 for every other
company. Width is not a pure function of n beyond that, because each pair has
two members and a concentrated profile resamples more stably than a diffuse
one; Snowflake at 459 postings has narrower intervals than Meta at 838. What a
share-level bootstrap would have done is make them all equally narrow, and
produced identified ranks where there is no evidence for one.

---

## 6. What this task overturned in Task 04 — correction C6

Task 04's taxonomy predicted twice that this task would be dominated by things
that cannot dominate it: concept skills such as Machine Learning and NLP
(§2.3), and skills appearing in a single posting (§6.1). Both are wrong for a
prevalence-weighted metric, because a skill's contribution to a cosine
numerator is the **product of two shares** — a hundred rare columns contribute
a hundred times almost nothing.

On the real vocabulary, over all 15 pairs:

| Group | Skills | Share of the cosine numerator (mean) |
| --- | --- | --- |
| Concept skills | 8 | **0.36%** |
| Skills in ≤ 1 posting | 8 | **0.00%** exactly |
| Skills in ≤ 10 postings | 39 | **0.005%** |
| Top 5 skills of the pair | 5 | **80.5%** |

For Google's own pairs the concept share runs 0.29% (Google – Snowflake) to
0.98% (Google – Meta). Removing all eight concept skills and rescoring leaves
the ranking **identical** — rank correlation 1.0, not one pair moves, the
largest cosine shifts by 0.0020.

Neither Task 04 design decision changes: the concept/tool split earns its
place for the other reason §2.3 gives, and the `min_postings=5` floor still
keeps the row-level matrix honest. What changes is what this report is allowed
to claim it has controlled for. **Dropping rare skills is not a robustness
check on a similarity score** — there was nothing there to be robust to. The
predictions would have been right for Jaccard, which counts columns, and
Jaccard's ranking correlates −0.04 with the one this report leads with.

Both taxonomy passages keep their original wording, marked in place with a
pointer to [C6](../../docs/corrections.md#c6--neither-concept-skills-nor-rare-skills-dominate-a-similarity-score).

---

## 7. Structure: where Google sits

Average-linkage clustering on 1 − cosine, with each partition's support
measured over the same 600 resamples:

| k | Partition | Support | Support without own products |
| --- | --- | --- | --- |
| 2 | Databricks \| **Google**, Meta, Microsoft, NVIDIA, Snowflake | 0.868 | 0.543 |
| 3 | Databricks \| **Google**, Meta, Microsoft, NVIDIA \| Snowflake | 0.853 | 0.967 |
| 4 | Databricks \| **Google**, Meta, NVIDIA \| Microsoft \| Snowflake | 0.863 | 0.678 |

Google is in the central group at every cut, and its nearest neighbours are
Meta at every cut and NVIDIA at three of them. **Databricks is the outlier of
the set** — it is the only company that separates first, and it does so
because Spark is in 72% of its postings against 6% of Google's.

The structure is not stable to the vocabulary. Without own products the
two-cluster split loses most of its support (0.868 → 0.543) and the three- and
four-cluster partitions change membership: NVIDIA leaves the central group and
Snowflake joins it. Figure 06 draws both dendrograms side by side, because a
single one would read as a fact about the companies rather than a fact about
the vocabulary.

The network view has no natural threshold. The graph is complete at t ≤ 0.45,
empty at t ≥ 0.95, and unchanged across a **four-step plateau** from 0.75 to
0.90 — where the only surviving edge is Google – Meta and the other four
companies are isolated. Figure 07 publishes the sweep, not one graph.

---

## 8. Do Google and its rivals move together? Refused

The question "is Google converging on Meta" is a trajectory question, and this
data cannot answer it. Three reasons, of which any one would be enough:

1. **One of fifteen pairs is eligible.** Task 07's forecastability gate has to
   pass on **both** members. Google passes it; four of the five rivals do not.
   The only eligible pair in the whole task is **Google – Meta**.
2. **The one eligible pair's answer is indistinguishable from zero.**
   r = **−0.0411**, 95% interval **[−0.6255, 0.5729]** — a width of 1.20 on a
   range of 2. The mean interval width across all pairs is 1.0449.
3. **The overall pattern is what closure produces on its own.** Shares of a
   common panel must sum to 1, so independent series correlate negatively once
   closed. Simulating that null at the panel's *own* dispersion (σ = 0.8901,
   calibrated by bisection on the observed CLR spread of 0.8047) gives a band
   of **[−0.1785, −0.0297]**, and the observed mean of −0.1746 **sits inside
   it**.

Sweeping σ from 0.2 to 1.4 shows the clause holds up to σ = 1.0 and first
fails at σ = 1.2 — where it fails **below** the band, meaning the observed
co-movement is *more* negative than independent series produce. There is no
dispersion at which this data shows companies moving together.

Two numbers that look like findings and are not. Google – Microsoft correlates
**−0.6681** with an interval excluding zero, and Databricks – Google
**−0.6365**, also excluding zero. Both are stronger than the closure null's
centre — and both fail the gate, because neither Microsoft nor Databricks has
a forecastable series. Reporting them would be reporting the closure
constraint plus two noisy series.

February is never filled. Task 07 established there is no panel in February;
`panel_wide` intersects observed periods and every pair is computed on 11
months. Filling it with zero would have made every pair look co-movement-rich
because they were simultaneously invisible.

---

## 9. What Google's five pairs support, sentence by sentence

Publishable as written:

- "Among the six companies compared, **Google and Meta express the most
  similar skill demand** — 0.92 on cosine over skill shares, 91% of the way
  from unrelated to identical, and rank 1 in 100% of resamples, under every
  metric, and after removing every company's own products."
- "All fifteen company pairs are **detectably related and detectably
  distinct** from each other."
- "**Databricks is the outlier** of the six on the full vocabulary."

Publishable only with the clause attached:

- "Google and NVIDIA are the second most similar pair — **on a vocabulary that
  counts each company's own products**. Removing them moves the pair to rank
  9."
- "Google and Microsoft are the fourth most similar pair — **a rank that holds
  in under a third of resamples**, and that rises to second when own products
  are dropped."

Not supported at all:

- Any statement that Google is *becoming* more or less like a rival (§8).
- Any ordering among the eleven middle pairs.
- Any statement of the form "Google is X% similar to Y" without the ruler in
  §3 next to it.

---

## 10. Limitations

1. **Similarity of postings, not of companies.** Everything here is a property
   of the text of job adverts on seven boards in 2023.
2. **Cosine is a choice.** It is declared before the ranking and all five
   metrics are published, but a reader who cares about "do they list the same
   skills" gets a genuinely different ordering (§4.3).
3. **The vocabulary is the largest free parameter**, and it is not neutral
   toward Google: Google has the largest own-product vocabulary of the six, so
   Google's pairs move most when it changes (§4.1).
4. **Six companies, fifteen pairs.** Every rank statement is over a set this
   small, and the bootstrap says eleven of the fifteen are one tier.
5. **One year, one collection window.** The publisher panel is thin and
   February is missing; §8 is refused partly because of it.
6. **The concentration cuts both ways.** Five skills carry 80% of a typical
   score. This is arithmetically why rare skills cannot swing the answer (§6),
   and it is also why the answer is largely a statement about Python, SQL,
   Java, R and C++.

---

## 11. What Tasks 09 and 10 inherit

- **Task 09 (Insight generation):** the three sentences in §9 are available as
  written; the two clause-bearing ones carry their clause. There is **no
  convergence sentence** — §8 refuses it. And no sentence should quote a
  similarity number without the calibration ruler, because 0.65 reads as
  "moderate" and means "halfway between nothing in common and the same
  company".
- **Task 10 (Final presentation):** the heatmap appears with its calibrated
  panel (Figure 01) and next to the tier chart (Figure 04), so a reader sees
  that eleven of fifteen cells are one band. The dendrogram appears only in
  the two-panel form (Figure 06). The network graph appears only with its
  threshold sweep (Figure 07). Figure 08 is the refusal, and it is a slide in
  its own right: it shows the closure band, the analytic line and the observed
  mean, and why a plausible-looking negative correlation is not a finding.

---

## 12. Deliverables

| Item | Path |
| --- | --- |
| Method rationale (team) | [`docs/task-08-company-similarity-methods.md`](../../docs/task-08-company-similarity-methods.md) |
| This report | `members/ankit-google/task-08-similarity-report.md` |
| Machine-readable report | [`task-08-similarity-report.json`](task-08-similarity-report.json) |
| Tables (16) | [`task-08-tables/`](task-08-tables/) |
| Figures (8) | [`task-08-figures/`](task-08-figures/) |
| Validation (17 checks vs scipy) | [`docs/task-08-similarity-validation.md`](../../docs/task-08-similarity-validation.md) |
| Correction registered | [C6](../../docs/corrections.md#c6--neither-concept-skills-nor-rare-skills-dominate-a-similarity-score) |
| Tests | [`tests/test_similarity.py`](../../tests/test_similarity.py) (37 of 474) |

Row-level data stays git-ignored. All 16 tables pass the forbidden-column and
personal-data checks before they are written (`privacy.passed` in the JSON).
