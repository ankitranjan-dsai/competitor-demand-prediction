# Task 06 — Competitor Comparison

**Team standard.** Task 05 left each of us with a curve for our own company.
Task 06 puts those curves on one axis, and that is a harder thing to do than it
looks: our postings reach us through job boards, the boards carry each employer
differently, and a raw cross-company count is mostly a statement about
syndication. This document records who is being compared, which comparisons
this data identifies, which it does not, and the code that enforces the
difference. Tasks 07 and 09 read the output, so a ranking published here
without its verdict column becomes a forecast and then a recommendation.

- **Code:** [`src/companies.py`](../src/companies.py) · [`src/compare.py`](../src/compare.py) · [`src/build_competitor_set.py`](../src/build_competitor_set.py) · [`src/build_comparison.py`](../src/build_comparison.py)
- **Tests:** [`tests/test_companies.py`](../tests/test_companies.py) (36) · [`tests/test_compare.py`](../tests/test_compare.py) (94) · [`tests/test_corrections.py`](../tests/test_corrections.py) (15; 359 in the suite)
- **Google findings:** [`members/ankit-google/task-06-comparison-report.md`](../members/ankit-google/task-06-comparison-report.md)
- **What this task overturned:** [`docs/corrections.md`](corrections.md) —
  [C4](corrections.md#c4--googles-posting-count-is-846-not-848), the posting
  count every Google report has quoted since Task 02
- **Legal position:** [`docs/legal/huggingface-data-jobs.md`](legal/huggingface-data-jobs.md)
  §"Scope extension (Task 06)" — no new source, no new collection

```bash
python src/build_competitor_set.py     # screen + build the competitor set
python src/build_comparison.py         # 33 tables, 9 figures, 1 JSON report
python -m pytest tests/ -q
```

The brief asks for comparison tables and visuals. Those are the deliverable and
they are committed. The load-bearing part of the task is narrower and less
comfortable: of the three comparisons a reader will want — **who posts more**,
**who is growing**, **who wants which skills** — the first is not identified by
this data at all. §1.3 says so with a number and a gate in code, rather than
publishing a ranking that would be read as market share.

---

## 1. Before any comparison: establishing who is being compared

### 1.1 An employer is a rule, not a substring

Every company on the shortlist has resellers, staffing agencies, partner firms
and consultancies carrying its name, and the source's `company_name` field is
not always an employer at all. Substring matching is wrong in **both**
directions:

| Employer string | Naive match | Truth |
| --- | --- | --- |
| `Geoambiente - Google Cloud Premier Partner` | Google | a reseller hiring for its own practice |
| `Customer Engineer, Machine Learning, Google Cloud - Doha` | Google | a **role title** in the employer field |
| `Meta Recruitment Ltd` | Meta | a staffing agency |
| `Metasys Technologies` / `OpenAirlines` | Microsoft / OpenAI | unrelated companies |
| `CN05 NVIDIA Shanghai WFOE` | *missed* | NVIDIA's Chinese entity |
| `Facebook App` / `Snowflake Computing` | *missed* | Meta / Snowflake |

`src/companies.py` therefore holds one registry entry per company with an
`include` pattern, an `exclude` pattern, and an ordered brand ladder
(`Google Fiber` must be tested before `Google`). `classify_employer` returns a
decision **and a reason**, one of `third_party_marker`, `named_third_party`,
`role_string` or `empty_employer_field`, and `matching_audit()` writes every
distinct employer string with its decision to a committed table.

That audit is not a debugging aid — it is the evidence for the selection. Ours
lists **119 distinct employer strings**, of which 23 are excluded (19
`named_third_party`, 3 `third_party_marker`, 1 `role_string`) carrying **100
postings**. Two of the excluded rows had been inside Google's headline count
since Task 02, which is why this task opens the register at
[C4](corrections.md#c4--googles-posting-count-is-846-not-848): **Google is 846
postings, not 848.** 0.2% of the count, and 100% of the trust in it.

**Rule:** add your company to `src/companies.py` with both patterns, then read
your own rows in `employer-matching-audit.csv` before you quote a total.

### 1.2 The feasibility screen is a published finding, not a private filter

Two shortlisted companies do not survive contact with the source:

| Company | Postings | Months present | Verdict |
| --- | --- | --- | --- |
| Meta | 1,029 | 12 | included |
| Google | 848 → **846** | 12 | included |
| Microsoft | 654 | 12 | included |
| Snowflake | 463 | 12 | included |
| Databricks | 340 | 12 | included |
| NVIDIA | 271 | 12 | included |
| **OpenAI** | **14** | 7 | `excluded_low_support` |
| **Anthropic** | **9** | 4 | `excluded_low_support` |

The floors (`MIN_POSTINGS = 250`, `MIN_MONTHS = 10`) are named constants,
declared before the screen ran, because a threshold chosen after seeing the
answer is not a threshold. The exclusions are **committed**, in
`company-feasibility-screen.csv`, with their reason strings.

Read the OpenAI and Anthropic rows carefully, because the obvious reading is
wrong. This is not evidence that they were not hiring in 2023. It is a fact
about the **source**: a 2023 data-analytics-role dataset assembled from job
boards, and those two companies barely syndicate into it. A comparison built on
14 postings would have been the most quotable table in this repo and the least
defensible.

### 1.3 The comparability gate: cross-company *levels* are not identified

Task 05 §1.1 established that a company's posting count is a joint statement
about its hiring and its publisher panel. Across companies that problem
compounds, because the panels are different panels.

Of the publishers in the feed, exactly **7 carry all six companies** over the
window — `via BeBee`, `via Indeed`, `via Ladders`, `via LinkedIn`,
`via Recruit.net`, `via SimplyHired`, `via Trabajo.org`. Within any single
month the number never exceeds **3**, and in February it is **0**.

`comparability_table()` measures, for each company, what share of its postings
reach us through that common panel:

| Company | Postings | Publishers | Common-panel share | Own-channel share | Largest publisher |
| --- | --- | --- | --- | --- | --- |
| NVIDIA | 271 | 43 | 67.5% | 0.0% | `via Trabajo.org` 26.9% |
| Microsoft | 654 | 88 | 60.9% | 0.0% | `via LinkedIn` 24.6% |
| Google | 846 | 95 | 54.1% | 6.7% | `via LinkedIn` 25.3% |
| Databricks | 340 | 39 | 53.8% | 0.0% | `via LinkedIn` 30.6% |
| Meta | 1,026 | 102 | 44.0% | 0.0% | `via BeBee` 12.9% |
| **Snowflake** | 460 | 37 | **23.5%** | **45.7%** | `via Snowflake Careers` |

Snowflake is the case that decides the task. Nearly half its postings arrive
through its own careers site, and under a quarter through the panel shared with
everyone else. `level_verdict()` requires every company to clear
`MIN_COMMON_PANEL_SHARE = 0.40`; Snowflake does not, so the verdict is
**`identified = False`, reason "common-panel coverage below the floor for
snowflake (24%)"**, and it is carried in the JSON report and on the first
figure.

Concretely: **this repo does not publish "Meta posts 3.8× more than NVIDIA."**
That ratio is 1,026 ÷ 271 and it is measuring how many boards syndicate each
employer. Nothing in the data separates that from hiring volume.

---

## 2. What *is* identified: relative share, not level

Levels are confounded, but not everything built from them is. Take the postings
that arrive through the **shared** publisher panel and ask a different
question: of that pool, what share is each company's, and does the share move
between the first and second half of the year?

```
share(company, half) = panel postings(company, half) / panel postings(all, half)
relative move        = log share(H2) − log share(H1)
```

Anything that scales the whole pool — a crawler that indexes more in H2, a
board that joins mid-year, a seasonal surge in postings generally — multiplies
numerator and denominator alike and **cancels**. This is a
difference-in-differences with the publisher-period effect differenced out, and
it is the only cross-company volume statement this data supports.

| Company | H1 postings | H2 postings | H1 share | H2 share | Δ pp | log change |
| --- | --- | --- | --- | --- | --- | --- |
| NVIDIA | 50 | 133 | 8.06% | 11.46% | **+3.40** | +0.347 |
| Databricks | 52 | 131 | 8.39% | 11.28% | +2.89 | +0.293 |
| Microsoft | 118 | 280 | 19.03% | 24.12% | **+5.09** | +0.237 |
| Meta | 152 | 299 | 24.52% | 25.75% | +1.23 | +0.050 |
| Google | 179 | 279 | 28.87% | 24.03% | **−4.84** | −0.182 |
| Snowflake | 69 | 39 | 11.13% | 3.36% | **−7.77** | −1.190 |

Panel totals: 620 postings in H1, 1,161 in H2.

> **Corrected by Task 07 —
> [C5](corrections.md#c5--task-06s-h1-aggregate-counts-february-on-a-panel-that-does-not-exist-in-february).**
> 97 of that 620 are February's, on a panel §1.3 of this document says has no
> publishers in February. Excluding February (H1 620 → 523) every sign above
> survives; Meta's +1.23 becomes **+4.15** and Google's −4.84 becomes
> **−6.56**. The wording here stays as submitted — see the register.

**The identifying assumption, stated so it can be attacked.** Each company's
propensity to syndicate to these seven boards is constant between the halves.
If Snowflake moved hiring onto its own careers site during 2023 — and 45.7% of
its postings are there — then its panel share falls without its hiring falling.
That is not a hypothetical for Snowflake; it is the most likely reading of the
largest number in the table.

So the estimate is **checked, not asserted**. `relative_share_verdict()`
recomputes the same H1→H2 move *inside each publisher separately* and reports
how many agree in sign:

| Company | Publishers agreeing | Verdict |
| --- | --- | --- |
| NVIDIA | **6 / 6** | `confirmed` |
| Meta | 5 / 6 | `mixed` |
| Snowflake | 5 / 6 | `mixed` |
| Databricks | 4 / 6 | `mixed` |
| Google | 4 / 6 | `mixed` |
| Microsoft | 4 / 6 | `mixed` |

One company clears it. **NVIDIA gained share of the shared pool in every
publisher that carries it** — the single cross-company volume finding in this
task that is not qualified into uselessness. Everything else is directional at
best.

> **Corrected by Task 09 —
> [C8](corrections.md#c8--a-unanimity-count-is-not-a-robustness-statistic-when-the-number-of-tests-moves-with-the-threshold).**
> `min_half` floors the publisher's total across all six companies, not the
> company's own cell, so a publisher holding one NVIDIA posting per half casts
> a full vote. Recount with a per-company floor and Google, Microsoft and
> Snowflake all *gain* `confirmed` by dropping tests, while NVIDIA's test set
> falls 6 → 1 and its sign-test p runs 0.0312 → 1.0000. Six publishers is the
> smallest panel on which unanimity reaches 0.05 at all. The pooled directions
> in the table above are unaffected; the `confirmed` label is withdrawn, and
> there is no unqualified cross-company volume sentence in this repository.

**Rule:** no cross-company volume claim leaves this repo without
`relative-share-verdict.csv` next to it.

---

## 3. Is the second half just the crawler warming up?

The panel pool nearly doubles between halves (620 → 1,161), and every company's
raw count rises. The obvious worry is that 2023 is a story about the collector,
not the market.

It is not, and the check that settles it is cheap. A crawler expanding its
coverage adds **publishers**. This one lost them:

| | January | December |
| --- | --- | --- |
| Distinct publishers | 66 | **51** |
| Postings per observed publisher | 5.02 | **7.59** |

A thickening feed on a shrinking roster is the opposite signature from coverage
expansion. `collection_artefact_check()` publishes the monthly series and flags
**shared thin months** — periods where most companies sit below their own
median at once, which is a collection gap rather than six simultaneous hiring
freezes. Ours are **2023-02, 2023-04 and 2023-05**, and February is also the
month with zero common publishers.

The half-over-half move also survives holding the panel fixed, for all six:

| Company | Raw H1→H2 | Balanced panel H1→H2 |
| --- | --- | --- |
| NVIDIA | +187.1% | +170.5% |
| Microsoft | +71.4% | +172.3% |
| Meta | +77.3% | +59.8% |
| Databricks | +31.3% | +141.2% |
| Google | +25.0% | +56.5% |
| Snowflake | +10.0% | +6.8% |

Raw and balanced agree in sign for every company. That is a real finding about
the window, and it is also a warning: **the treatments agree on H1→H2 and
disagree wildly on the within-year index** (§1.1 of Task 05), so the two
questions must not be answered with the same sentence.

One more number worth keeping: the mean pairwise correlation between companies'
monthly counts is **0.318**. The six series are not moving as one, which is
what a single dominant collection artefact would look like — but 0.32 is not
zero either, so a shared component exists and is exactly what the relative-share
estimator differences out.

---

## 4. Role mix is a confounder, not a nuisance

"Google asks for Python more than Microsoft" is a claim about two companies
only if they are posting comparable jobs. They are not. Total-variation
distance between `job_function` mixes runs from **0.135** (Databricks vs
Snowflake) to **0.678** (Meta vs Snowflake), mean **0.378**. At 0.678, two
thirds of one company's postings would have to be reassigned to look like the
other's.

`standardise()` answers this by **direct standardisation** to the pooled mix
across all six companies: compute each company's share inside each job
function, then reweight by the pooled function distribution. Every standardised
figure is published with:

- `mix_effect` — standardised minus crude, so the size of the correction is
  visible rather than absorbed;
- `weight_covered` — the share of the pooled weight the company actually has
  strata for. A standardised number covering 60% of the weight is a different
  object from one covering 99%, and the column says which.

Applied to skill coverage, the correction is real but small, and that is itself
the finding:

| Company | Crude | Standardised | Mix effect | Weight covered |
| --- | --- | --- | --- | --- |
| Databricks | 100.0% | 100.0% | 0.0 pp | 93.1% |
| Snowflake | 99.8% | 98.7% | +1.1 pp | 93.1% |
| NVIDIA | 86.7% | 88.6% | −1.9 pp | 93.1% |
| Meta | 84.6% | 86.3% | −1.7 pp | 92.9% |
| Microsoft | 79.7% | 80.0% | −0.3 pp | 91.2% |
| **Google** | **70.8%** | **76.4%** | **−5.6 pp** | 99.2% |

Google's extraction coverage is the lowest of the six and it stays lowest after
standardisation. The gap is **not** a role-mix artefact; it is a property of
the postings (Google's data-centre facilities roles genuinely carry no software
skills, which Task 03 §3 documented). Any cross-company skill *share* must
therefore use `share_of_skilled`, never `share_of_all` — the same rule Task 04
handed down, and here it is load-bearing at 29 percentage points of coverage
spread.

On individual skills the correction can be much larger: Snowflake's Java share
moves **46.2% → 28.7%** (17.5 pp) under standardisation, and Meta's R share
55.3% → 38.8%. Crude cross-company skill shares are not reportable.

---

## 5. Simpson's paradox is the default assumption here too

Task 05 §5 established this within one company; across companies it is worse,
because the mixes differ more than one company's mix differs across halves.
`stratified_pair_verdict()` recomputes every pairwise skill difference inside
each `job_function` and returns `confirmed` only when **every** supported
stratum agrees with the pooled sign.

Of 182 supported company-pair skill comparisons: **33 confirmed, 149 mixed.**
Four out of five pooled skill gaps do not hold inside every role type.

An example of each, both Google vs Databricks:

| Skill | Pooled gap | Strata agreeing | Verdict |
| --- | --- | --- | --- |
| Go | +8.1 pp | 5 / 5 | `confirmed` |
| R | +20.9 pp | 3 / 5 | `mixed` — Analytics and Engineering run the other way |

R is the instructive one. Google leads on R by 21 points pooled; inside
Analytics, Databricks leads 52.8% to 42.3%. Both are true, and only one of them
is a statement about what the two companies want from an analyst.

The `detail` column carries every stratum's pair of shares, so a reader can see
the reversal without rebuilding anything.

**Rule inherited from [C2](corrections.md#c2--five-of-task-04s-ten-headline-skill-movers-do-not-survive-stratification):**
every skill claim carries its stratified verdict.

---

## 6. Channel sensitivity: does the skill share survive holding the board fixed?

Publishers do not carry a random sample of a company's postings. Some boards
are engineering-heavy, some are regional, and a company's skill profile can be
partly a profile of its boards.

`panel_robustness()` recomputes each skill's `share_of_skilled` using **only
postings from the common panel** and reports the delta. Five shares move more
than 10 points:

| Company | Skill | All postings | Common panel | Δ |
| --- | --- | --- | --- | --- |
| Snowflake | C++ | 21.8% (100/459) | 6.5% (7/108) | **−0.153** |
| Snowflake | Java | 46.2% (212/459) | 31.5% (34/108) | **−0.147** |
| Snowflake | Azure | 29.9% (137/459) | 19.4% (21/108) | **−0.104** |
| Databricks | Java | 27.9% (95/340) | 38.3% (70/183) | **+0.103** |
| Snowflake | Excel | 34.4% (158/459) | 44.4% (48/108) | **+0.100** |

Four of the five are Snowflake's, which is the same finding as §1.3 arriving
from a different direction: the company with 45.7% of its postings on its own
careers site has the most channel-dependent skill profile. Its own site posts
different jobs than the boards do.

**One implementation detail worth documenting, because getting it wrong
inverts the answer.** A skill that vanishes entirely from the common panel
produces *no row* in the panel table, and a left join fills it with `NaN`. Read
naively, `NaN > 0.10` is `False` and the skill is reported as *not* channel
sensitive — the exact opposite of the truth, since disappearing is the most
channel-sensitive behaviour available. The panel denominator exists, so the
absent share is a **measured zero**, and `panel_robustness()` fills it as one.
A test pins this; it was a live bug in this repo, found by that test.

---

## 7. Self-reference: a vendor's own product is not a skill signal

`Databricks` is a skill in the taxonomy. It is also an employer. 99.4% of
Databricks postings mention Databricks, and any "most distinctive skill"
ranking will hand back the company's own product name and call it a strategy.

| Company | Postings naming an own product | Coverage inflation |
| --- | --- | --- |
| Databricks | 99.4% | +3.2 pp |
| Snowflake | 99.1% | +1.5 pp |
| Microsoft | 66.5% | **+11.6 pp** |
| Google | 14.1% | +3.3 pp |
| Meta | 5.1% | +0.1 pp |
| NVIDIA | 0.4% | +0.4 pp |

Two separable effects, and conflating them overstates the problem:

1. **Coverage inflation** is small — between 0.1 and 11.6 points. Dropping own
   products would not change who has the best skill coverage.
2. **The distinctiveness ranking is dominated by it.** Google's top skills by
   log-lift are BigQuery (+2.86) and Looker (+2.55); both are Google products.
   Six of Google's 48 significant skills are self-referential.

`compare.VENDOR_SKILLS` maps each company to its own products, and every skill
row carries `vendor_relation` (`own_product` or `origin` — TensorFlow and Go
originate at Google without being sold by it) and a boolean `self_referential`.

**They are flagged and never filtered.** A vendor's own product appearing in
its postings is real information — Microsoft asking for Azure in two thirds of
its postings is a fact about Microsoft's stack — it just is not evidence of
market-wide demand. Filtering would hide it; flagging lets the reader discount
it. A test asserts the flag exists and that no row is dropped.

---

## 8. Uncertainty, and why this module imports no scipy

Every difference in this task is a difference of proportions on samples between
271 and 1,026 postings, sliced by skill and sometimes by function. Unlabelled
point estimates would invite exactly the over-reading the rest of the document
is built to prevent. So `src/compare.py` publishes, for every comparison:

| Quantity | Method |
| --- | --- |
| Single-company share interval | **Wilson** score interval |
| Difference of two shares | **Newcombe** method 10, from the two Wilson intervals |
| p-value | pooled-variance two-proportion z, via `math.erfc` |
| Multiplicity | **Benjamini–Hochberg** FDR across the skills in each comparison |
| Effect size | log lift with a **Haldane–Anscombe** ½ correction |

Three choices worth their sentences:

- **Newcombe rather than a normal-approximation difference interval.** Shares
  here sit near 0 and near 1 (Databricks' skill coverage is 1.00), where the
  Wald interval runs outside [0, 1] and its coverage collapses.
- **BH rather than Bonferroni.** 127 skills × 5 pairings is a screening
  exercise; controlling the false-discovery rate keeps power while still
  refusing the 6 or 7 "significant" skills pure noise would hand us.
- **Haldane–Anscombe on the log lift.** Zero cells are common (Microsoft has 0
  Looker postings). Without the correction the lift is infinite and the skill
  sorts to the top of every ranking; with it, the value is finite, comparable,
  and shrinks toward zero as the cell shrinks.
- **`MIN_CELL = 10`.** Below it a row is marked `supported = False` and is
  never eligible for a verdict, whatever its p-value.

None of this needs scipy, and deliberately so: the whole comparison layer runs
on pandas, numpy and `math`, so any reviewer can rebuild every published number
from this repo without matching a solver version. (scipy is in the environment;
the core modules do not import it.)

---

## 9. What gets committed

Aggregate only, same rule as Tasks 04 and 05.

```
members/<member>-<company>/task-06-tables/     33 CSVs (§1–§7)
members/<member>-<company>/task-06-figures/     9 PNGs
members/<member>-<company>/task-06-comparison-report.json
```

Row-level competitor data stays git-ignored, in `data/raw/task-06/` and
`data/processed/<company>/`. Task 06 writes **only** to those two places: an
earlier version of the builder wrote a company's selected rows to
`data/raw/<company>/<company>_jobs_hf_backfill_2023.parquet`, which for Google
is Task 02's own artefact and Task 03's default input, and running Task 06
silently rewrote it two rows shorter. A task may add data; it may not rewrite
the data an earlier task reported on. Two tests pin the rule.

Two guards run on every table before it is written, in `build_comparison._write`:

- **Forbidden columns.** `FORBIDDEN_COLUMNS = {"country", "location_country",
  "share_of_all"}` — the first two because a cross-company country split
  compares aggregator footprints rather than companies (Task 05 §5: geography
  was the least robust breakdown, 21 of 48 country trends surviving), and the
  wrong comparison must not be one column away from being made; the third
  because it is the wrong skill denominator (§4). Writing one raises.
- **The standing Task 01 privacy check.** `personal_data_columns_present()` is
  re-run over all **33** tables; the JSON report records `passed: true` and an
  empty dictionary of findings.

The legal position for using the same file to select five more employers is
recorded in `docs/legal/huggingface-data-jobs.md` §"Scope extension (Task 06)":
no new source, no new request, no new fields, no personal data, and no
competitor contacted, profiled or scraped.

---

## 10. Checklist for each specialist

1. Add your company to `src/companies.py` with `include`, `exclude` and its
   brand ladder. Run `python src/build_competitor_set.py --screen-only` first.
2. **Read your rows in `employer-matching-audit.csv`.** Every excluded string,
   every matched string. Mine had two wrong in 848 — see
   [C4](corrections.md#c4--googles-posting-count-is-846-not-848).
3. Check your `common_panel_share` in `company-comparability.csv`. If you are
   below 0.40, say so in your report; you are the reason the level comparison
   is gated, and that is a finding about your company's channel mix.
4. Quote **no** cross-company level or ratio. Not once, not "roughly", not in a
   figure caption. Use relative share and its verdict.
5. Take your relative-share row **with its `publishers_agreeing` count.**
   `mixed` means directional, not measured.
6. Use standardised skill shares for every cross-company skill sentence, and
   quote `weight_covered` beside them.
7. Read `skill-stratified-verdicts.csv` before writing any "we ask for X more
   than they do". Four out of five pooled gaps do not survive it.
8. Check your own row in `self-reference-audit.csv` and never present your
   employer's own product as a demand signal.
9. Label every claim **within-2023**, and every volume claim as **share of a
   shared publisher panel**, not as postings.
10. Commit `task-06-tables/`, `task-06-figures/` and the JSON report; leave
    `data/raw/task-06/` and `data/processed/` git-ignored.
11. `python -m pytest tests/ -q` must pass before you push.

---

## 11. What Tasks 07, 08 and 09 inherit

- **Task 07 (Demand Forecasting):** forecast **relative share of the common
  panel**, not counts. Levels are not identified (§1.3), so a per-company
  volume forecast would be forecasting syndication. The panel is thin —
  ≤3 publishers in any month, 0 in February — so monthly is the shortest
  usable frequency and February is missing, not zero. Task 05's rules still
  hold: `log_growth`, no annual seasonal term, batch weeks dummied.
- **Task 08 (Visualisation):** every cross-company chart needs its gate on it.
  Figure 01 in `task-06-figures/` is the pattern — the comparability verdict is
  drawn, not footnoted. Do not plot raw counts by company on a shared axis.

  > **Corrected by Task 08 — [C7](corrections.md#c7--task-08-is-company-similarity-scoring-not-visualisation-and-not-evaluation).**
  > Task 08 is **Company Similarity Scoring**, not Visualisation — the brief
  > and the README task table both name it. Both instructions in this bullet
  > were followed; only the name is wrong.
- **Task 09 (Insight Generation):** the only unqualified cross-company volume
  sentence this data supports is **"NVIDIA gained share of the shared
  publisher pool between H1 and H2 2023, in every publisher that carries
  it."** Everything else needs its verdict clause. And no country comparison:
  publishers are regional, so a country split across companies compares
  aggregator footprints.
