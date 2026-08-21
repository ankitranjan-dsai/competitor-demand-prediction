# Task 09 — Insight Generation & Reporting

**Team standard.** Tasks 05 to 08 each ended by narrowing what could be said.
Task 05 found the volume direction unidentified. Task 06 found cross-company
levels unidentified. Task 07 found the maximum useful horizon to be zero. Task
08 found two of fifteen similarity ranks identified and refused trajectory
similarity outright. Task 09 is asked to turn all of that into "clear,
actionable insights" for strategy teams, HR, product managers, CEOs and
investors.

That is a harder brief than it looks, and the failure mode is specific. A
reporting task writes prose. Prose does not carry its own denominator, its own
citation or its own verdict, so the gap between what a table supports and what
a sentence asserts closes silently — and closes in the direction of confidence,
because a confident sentence is the one that reads well. Seven of this
repository's corrections were caught because a *later analysis* re-derived the
number. There is no later analysis after this one: Task 10 is the presentation.

So this document does not describe how insights were written. It describes a
**claim compiler**: insights are typed records, they are generated
mechanically from the verdict tables Tasks 04 to 08 committed, and each one
must pass four gates before it may be published.

- **Code:** [`src/insights.py`](../src/insights.py) · [`src/build_insights.py`](../src/build_insights.py)
- **Tests:** [`tests/test_insights.py`](../tests/test_insights.py) (72; 550 in the suite)
- **Google findings:** [`members/ankit-google/task-09-insight-report.md`](../members/ankit-google/task-09-insight-report.md)
- **What this task overturned:** [`docs/corrections.md`](corrections.md) —
  [C8](corrections.md#c8--a-unanimity-count-is-not-a-robustness-statistic-when-the-number-of-tests-moves-with-the-threshold),
  Task 06's publisher-agreement count
- **Inherited from:** [`docs/task-07-demand-forecasting-methods.md`](task-07-demand-forecasting-methods.md) §8 (the forecast prohibition) and [`docs/task-08-company-similarity-methods.md`](task-08-company-similarity-methods.md) §8 (the trajectory refusal)
- **Legal position:** unchanged — no new source, no new collection, no new
  field. The salary audit in §7 reads a column that has been in the processed
  dataset since Task 03.

```bash
python src/build_insights.py           # 25 tables, 8 figures, 1 JSON report
python -m pytest tests/ -q
```

Four findings in one line each, so the rest of the document reads as their
justification:

1. **436 claims were generated; 120 are publishable, and 102 of those rest on
   a firm upstream verdict.** A 27.5% yield (§11). The denominator is the
   deliverable as much as the numerator is.
2. **306 of the 316 refusals are identification, not wording** (§2). The
   sentences fail on what this data can carry, not on how they are phrased —
   which means no amount of careful writing would have recovered them.
3. **Salary benchmarking is refused** (§7). Disclosure runs from 100% to 0% by
   publisher, so conditioning on a disclosed salary conditions on publisher
   and through it on country and role. Google's disclosed subset is 76.5% US
   against 27.0% overall — a 49.5 pp shift.
4. **A unanimity count is not a robustness statistic** (§8). Task 06 counted
   publisher sign agreement without a per-company cell floor. Impose one and
   three of six companies *gain* `confirmed` by dropping tests. Registered as
   C8.

---

## 1. An insight is a record, not a sentence

Every claim in this task is an instance of `Claim`, and the fields are the
argument:

| field | what it holds |
| --- | --- |
| `text` | the sentence, generated from the cell it describes |
| `citation` | `table#column@selector`, resolvable against a committed CSV |
| `value` | the number the sentence quotes |
| `verdict_source` | the upstream verdict that decides whether it may be said |
| `clause` | the qualifier that must travel with it |
| `falsifier` | the observation that would overturn it |
| `action` | what a named audience would do differently |
| `audience` | one of strategy, HR/talent, product, exec, investor |

A sentence with no citation is not a weak insight; it is not an insight. A
sentence with no falsifier is not a finding; it is a description. And a
sentence with no named audience has no claim on anyone's attention — §10.

The citation format is deliberately narrow:

```
task-06-tables/relative-share-verdict.csv#verdict@company=nvidia
```

The selector must reduce the table to **exactly one row**. Matching zero rows
and matching two are both errors, and both raise. A citation that resolves to
a group is a citation to a summary somebody has yet to compute.

### 1.1 The clause is required of every publishable claim

Not of the hedged ones — of all of them. A claim that reaches the ledger
without a clause is refused as a drafting fault, and none of the 436 did.

This is a result rather than a formality: **this evidence base yields no
context-free sentence.** Every publishable number is a share of a specific
denominator, on a specific publisher panel, within a specific job function, or
all three. The clause is what the reader must hold the sentence with, and
folding it into the sentence at render time (`sentence()`) is what stops it
being the thing that gets trimmed for the slide.

The two published statuses therefore do **not** mean "clean" and "caveated":

- `published` — the upstream task called the finding confirmed, robust or
  identified.
- `published_qualified` — the upstream task hedged it itself, and the reader
  is owed that fact too.

### 1.2 Columns are read by subscript, never by attribute

`diff`, `rank`, `count`, `size`, `mean` and `max` are all real column names in
the committed tables *and* real `pandas.Series` attributes. `row.diff` returns
the bound method; `row.size` returns the length of the Series. Every generator
goes through `_cell(row, name)`, which subscripts. This is not style: 127
claims once quoted `<bound method Series.diff>`, and `size` is worse, because
it returns a plausible integer and nothing downstream looks wrong.

---

## 2. Four gates, in a fixed order

A claim is reported against the **first** gate it fails, so the ledger's
`blocked_by` column is a diagnosis rather than a list.

**Gate 1 — Evidence.** The citation resolves, and the quoted value matches the
cell within `rtol=0.02, atol=5e-4`. The gate also checks *direction*: if the
sentence says "rising" and the cited verdict column records `down`, the number
can agree while the sentence does not. One claim in 436 fails here, and it was
planted (§4.1).

**Gate 2 — Lint.** The text contains no prohibited construction (§4).

**Gate 3 — Identification.** The claim inherits a verdict from the task that
computed it (§6). Task 09 never re-decides whether something is identified.

**Gate 4 — Consistency.** Where two tasks quote the same quantity, the later
one wins and the earlier value may appear only with its correction pointer
(§9).

The attrition is lopsided and that is the headline:

| gate | claims blocked |
| --- | --- |
| 1 evidence | 1 |
| 2 lint | 9 |
| 3 identification | 306 |
| 4 consistency | 0 |

**306 of 316.** The claims that fail are not badly written. They are sentences
this data cannot support at any level of care, and a reporting task that
produced only its survivors would show a hit rate of 100% by construction.

---

## 3. Candidate claims are generated, not written

One claim per row of the upstream verdict tables. Eleven generators walk
`task-05-tables/` through `task-08-tables/` plus this task's own salary
tables, and mint a claim for every company, every skill, every pair and every
segment the tables hold — 436 in total, before any of them is looked at.

This is the single most important design decision in the task, and it is
about the denominator. An author who writes insights by hand writes the ones
they can support; the refusals never take a form anyone can count. Generating
the candidate set first means the yield is measured against **everything this
evidence base could be asked to say**, not against everything the author
happened to think of.

It also removes the author from the selection step. No generated claim was
dropped because it was inconvenient; the 316 refusals each name a rule.

---

## 4. The prohibited-pattern list, and where each rule comes from

Nine rules. None is a style preference — each is a prohibition some earlier
task earned, expressed as a regex over the claim text.

| rule | source | what it stops |
| --- | --- | --- |
| `forecast` | task-07 §8 | a forward-looking sentence; the max useful horizon is 0 |
| `convergence` | task-08 §8 | "converging" / "moving together"; trajectory similarity is refused |
| `cross_company_level` | task-06 §1.3 | "more roles than", "largest hirer"; levels are not identified |
| `product_launch` | task-09 §5 | reading a hiring pattern as a product announcement |
| `unmeasured_construct` | task-02 scope | headcount, attrition, revenue, budget — none is in the dataset |
| `bare_share_of_all` | task-04 §7 | "30% of all postings"; the denominator is share of *skilled* |
| `seasonal` | task-05 §6 | a seasonality claim on a single year of data |
| `causal_strategy` | task-05 §1 | "Google is pivoting to…"; intent is not observed |
| `country_split` | task-05 §9 | a country figure, which compares aggregator footprints |

`country_split` is not a hardcoded list. It is built at run time from the
country vocabulary in `task-05-tables/panel-check-country.csv`, so a
specialist whose panel spans different countries inherits the rule
automatically.

**The exemption, and its limit.** A claim whose *purpose* is to report a
refusal has to quote the forbidden words — "no forecast is supported at any
horizon" contains "forecast". Such a claim declares `refusal_rule`, and is
exempt from **that one rule only**. Setting the field on a claim that asserts
the forbidden thing exempts it from the gate meant to stop it, which is a hole
that was live in this module while it was being written and is now pinned by
`test_exemption_applies_only_to_the_claims_own_rule`.

### 4.1 Ten sentences are generated on purpose to make the linter fire

Every templated claim in §3 is clean by construction, so on a first run the
linter caught nothing. **A gate that nothing trips is not evidence of a
well-behaved corpus; it is an untested gate.**

So the `tempting` family generates ten sentences a reader would actually want
— one per rule, plus one whose number is right and whose direction is wrong.
All ten are refused, each by a different mechanism, and if a future refactor
weakens a rule the corresponding tempting claim starts publishing and the test
suite fails.

The rule set is also only as good as its vocabulary. `cross_company_level`
originally matched "more jobs than" but not "more roles than"; `bare_share_of_all`
matched "share of all postings" but not "30% of all postings" — the likelier
phrasing of the two. Both are widened, and both widenings left all 120
publishable claims untouched, which is the check that says a rule was tightened
rather than fitted.

---

## 5. Product-launch prediction is not payable

The brief lists "predict competitor product launches" among the things this
project would let a reader do. It is the one promise no table in this
repository can pay, and it is worth stating plainly rather than quietly
omitting.

A job posting is evidence that a company is advertising for a skill. Reaching
a product announcement from there requires three links this data does not
have: that the hiring is for a product rather than for platform, support or
replacement; that the product is external rather than internal; and that the
posting leads the launch by a knowable interval. The dataset holds no launch
dates against which any of the three could be estimated, so there is no
version of this analysis that fails — there is only a version that is never
tested.

The construct is therefore **not payable**, distinct from *not paid*: zero
claims were generated for it, because nothing in the repository measures it. A
promise with a zero denominator and a promise with a zero numerator are
different facts, and the promise audit and its figure keep them apart.

`product_launch` is a lint rule as well as a promise verdict, because the
sentence is easy to write anyway ("Google is hiring for X, suggesting a launch
in…") and reads as the most valuable output in the report.

---

## 6. Identification is inherited, never re-decided

Each claim names a `verdict_source` — a family and a citation. `VERDICT_MAP`
translates the upstream verdict value into a status, and Task 09 looks the
answer up rather than forming a view. A claim that asserts its own confidence
is exactly the failure this gate exists to stop.

### 6.1 A conjunction inherits the weakest verdict

`verdict_source` may name several terms joined by `&`, and the **weakest**
status wins. Two live defects made the rule necessary, and both would have
republished something an earlier task explicitly refused:

- **Trajectory.** Reading `excludes_zero` alone publishes 2 of Task 08's 15
  pairs — the two whose correlation interval excludes zero *while being
  ineligible*. The eligibility flag is the other half of the condition, so the
  source is `eligible & excludes_zero` and all 15 refuse.
- **Forecast horizon.** `interval_sufficient` is True at h=1 and h=2 in Task
  07's own table, which contradicts that task's headline of a maximum useful
  horizon of 0. The flag that settles it is `supported`, which is False on
  every forecast row. The source is the conjunction of both.

The general form: **a flag is not a verdict.** An upstream table often carries
several booleans, and reading the permissive one in isolation reconstructs the
claim its author refused.

---

## 7. Salary benchmarking is refused

The brief's fifth promise is the only one not already settled by a closed
task, so Task 09 tests it directly. It fails, and the way it fails is
instructive.

### 7.1 Disclosure is a publisher behaviour

`salary_year_avg` is populated for a small minority of postings, and the
minority is not random. Company disclosure runs 2.60% (Microsoft) to 13.84%
(Meta). By publisher it runs from **100%** (Ladders, Relocation Jobs,
JobServe) to **0%** (The Muse, Trabajo.org, Recruit.net, company career
pages), with Ai-Jobs.net at 86.8% and Indeed at 34.5%.

Since Task 05 the publisher panel has been the control for everything else in
this repository, and salary is the sharpest case yet: the missingness is
essentially *determined* by publisher.

### 7.2 Conditioning on disclosure moves the country mix

If it moved nothing, the missingness would be ignorable. It moves a lot.
Modal-country share on the disclosed subset against the full set:

| company | shift (pp) |
| --- | --- |
| nvidia | +73.06 |
| google | +49.52 |
| snowflake | +31.27 |
| meta | +23.84 |
| databricks | −17.69 |
| microsoft | −19.27 |

Google's postings are 27.0% US overall and 76.5% US among those disclosing
pay. A company-level median salary computed on this column is a median for a
different population than the one the report is about — and NVIDIA's is a
different population again, in the opposite direction from Databricks'.

### 7.3 Within-publisher comparison, and what survives stratification

The remedy is the same one Task 06 used for shares: compare inside a
publisher. At a 5-salary floor only three publishers carry two or more
companies — Ai-Jobs.net (Databricks, Google, Snowflake), Indeed (Meta,
Microsoft, NVIDIA) and Ladders (Google, Meta). That yields three pairs for
Google, of which one excludes zero:

| pair | publisher | median difference | 95% CI |
| --- | --- | --- | --- |
| Google − Databricks | Ai-Jobs.net | +$61,976 | [35,628, 75,800] |
| Google − Snowflake | Ai-Jobs.net | +$3,577 | [−13,674, 22,775] |
| Google − Meta | Ladders | −$15,000 | [−25,000, 37,500] |

Then stratify inside `job_function`, as Task 05 §7 requires, and **nothing
survives**:

- Google–Snowflake, Science / Research, 7 vs 7: **$0** [−7,847, 15,000].
- Google–Meta, Science / Research, 15 vs 11: −$25,000 [−25,000, 50,000].
- Google–Databricks — the one pair that showed a difference — has **no
  testable stratum at all**. Where Google is thick, Databricks is thin
  (Science / Research 7 vs 2) and where Databricks is thick, Google is absent
  (Engineering 2 vs 14, Technical Sales 0 vs 12). Google's postings on that
  publisher are 70.0% Science / Research; Databricks' are 42.4% Engineering.
  The $62k was one role mix compared against a different role mix.

### 7.4 What may still be said

The disclosure rates themselves. Six claims publish, one per company, and each
carries the clause "a property of the publishers this company appears on, not
of its pay policy".

They are deliberately **excluded** from the "benchmark salaries" promise. They
publish, but they say who discloses pay, not what anyone pays, and counting
them would report the promise as paid on the strength of sentences that do not
pay it. A promise is settled by the family that answers it, never by an
adjacent family that happened to survive.

**Verdict: refused.** Not "estimated with caution" — refused. There is no
pay benchmark in this dataset, and the number of steps required to reach one
that would be wrong is small enough that the refusal has to be explicit.

---

## 8. C8 — a unanimity count is not a robustness statistic

Task 06 reported, for each company, how many of the six common-panel
publishers agreed on the sign of its H1→H2 share change, and called a company
`confirmed` when the agreement was unanimous. The `min_half` floor in that
function applies to the *publisher's total across all companies*, not to the
company's own cell. So a publisher carrying one NVIDIA posting in H1 and one
in H2 casts a full vote.

Recount with a floor on both halves of each company's own cell:

| company | verdict at floor 0 | floors confirmed | publishers tested |
| --- | --- | --- | --- |
| databricks | mixed | none | 6 → 2 |
| google | mixed | **10** | 6 → 3 |
| meta | mixed | none | 6 → 4 |
| microsoft | mixed | **5, 10** | 6 → 3 |
| nvidia | **confirmed** | 0, 3, 5, 10 | 6 → 1 |
| snowflake | mixed | **3, 5, 10** | 6 → 2 |

Three of six companies **gain** `confirmed` by raising the floor — that is, by
discarding tests. A statistic that improves as evidence is removed is not
measuring robustness.

### 8.1 What unanimity is worth at this panel size

Under a two-sided exact sign test, unanimous agreement across *n* publishers
gives p = 2/2ⁿ:

| publishers tested | p | clears 0.05 |
| --- | --- | --- |
| 1 | 1.0000 | no |
| 2 | 0.5000 | no |
| 3 | 0.2500 | no |
| 4 | 0.1250 | no |
| 5 | 0.0625 | no |
| 6 | 0.0312 | **yes** |
| 7 | 0.0156 | **yes** |

**Six is the minimum panel at which unanimity can reach significance at all.**
Every `confirmed` gained at a stricter floor is gained on 2, 3 or 4 tests,
where unanimity is not evidence — it is arithmetic.

NVIDIA is the case that matters, because Task 06 §11 hands this task exactly
one unqualified cross-company sentence and this is it. NVIDIA is `confirmed`
at every floor, so it never flips. But its test set falls 6 → 1, and its
p-value moves 0.0312 → 0.1250 → 0.1250 → 1.0000. Its only claim to
significance rests on admitting cells with a single posting per half.

### 8.2 What C8 does and does not correct

**Directions do not change.** The pooled sign is identical at every floor for
all six companies. C8 is about the *confirmation*, not the finding, and the
distinction is the whole content of the correction: Task 06's directions
stand; its `confirmed` labels do not travel.

The register entry is
[C8](corrections.md#c8--a-unanimity-count-is-not-a-robustness-statistic-when-the-number-of-tests-moves-with-the-threshold),
and its quantitative claims are checked against the committed table by
`tests/test_corrections.py`.

---

## 9. Cross-task consistency

Three quantities in this repository are quoted by two tasks with different
values — all three are the H1→H2 share change, and all three are the C5
February correction:

| quantity | Task 06 | Task 07 | delta |
| --- | --- | --- | --- |
| google | −4.84 pp | −6.5617 pp | −1.72 |
| meta | +1.23 pp | +4.1475 pp | +2.92 |
| nvidia | +3.40 pp | +3.6163 pp | +0.22 |

The later value wins. A claim quoting the earlier one fails gate 4 unless it
carries the C5 pointer. No claim in the current ledger does, and the gate is
kept because the register is a moving target: the next correction adds a row
here, and the sentences that quote the superseded value stop compiling.

---

## 10. Audiences, and one that gets nothing

The brief names five audiences. Every claim declares one, and every publishable
claim declares an `action` — what that audience would do differently. A claim
whose action is "note it" is counted as informational rather than actionable,
and the split is published:

| audience | claims | actionable | questions covered |
| --- | --- | --- | --- |
| product | 81 | 81 | tech stack |
| HR / talent | 23 | 23 | hiring patterns, position, skill demand |
| strategy | 14 | 12 | hiring patterns |
| exec | 2 | 0 | future demand |
| **investor** | **0** | **0** | — |

Investors get nothing, and the reason is structural rather than a gap that more
work would close. The questions an investor asks of a competitor — is it
bigger, is it growing, where is it heading — are questions about a level, a
trajectory and a forecast. Task 06 refused the first, Task 08 the second, Task
07 the third. There is no sentence in this repository for that reader, and
saying so is more useful than assembling one.

---

## 11. Reading the yield

| question | generated | publishable | yield |
| --- | --- | --- | --- |
| hiring patterns | 33 | 21 | 63.6% |
| skill demand | 31 | 8 | 25.8% |
| tech stack | 310 | 81 | 26.1% |
| future demand | 10 | 2 | 20.0% |
| position | 52 | 8 | 15.4% |
| **all** | **436** | **120** | **27.5%** |

Two cautions on reading this table.

**The yields are not comparable across questions**, because the denominators
are generated by different mechanisms. `tech_stack` proposes 310 claims
because the skill vocabulary is large, not because the question is harder; its
26.1% is a rate over skills, while `hiring_patterns`' 63.6% is a rate over six
companies and four panel treatments.

**A high yield is not a good result and a low one is not a bad one.** The yield
measures the fit between the questions asked and the evidence available. The
figures use proportions on a linear axis for the same reason: a log axis makes
6 of 33 look like a nearly-full bar.

---

## 12. Limitations

1. **The compiler cannot catch a sentence nobody generated.** The 436
   candidates are one per row of the committed tables. A question no table
   addresses produces no claim and therefore no refusal — it is invisible to
   the yield. §5's product-launch promise is handled explicitly for this
   reason; other such gaps may exist.
2. **The lint rules are regexes over English.** They catch the phrasings
   listed in §4 and the ones the tempting family exercises. A writer who
   paraphrases around them is not stopped, and the mitigation is that the
   published corpus is generated rather than written.
3. **`VERDICT_MAP` is a translation table, and translations embed judgement.**
   Mapping `vendor_dependent` to `published_qualified` rather than `refused` is
   a decision this task made; it is in one place and auditable, but it is not
   inherited.
4. **The salary bootstrap is a percentile bootstrap on medians at n = 7–38.**
   Its intervals are wide and asymmetric and should be read as a feasibility
   check, not an estimate. The refusal in §7 does not depend on them.
5. **One year, one country mix, six companies.** Everything above inherits the
   scope limits of Tasks 02 and 06.

---

## 13. What Task 10 inherits

Task 10 is the final presentation. What it receives:

- **`claim-ledger.csv` is the source of every sentence.** A slide that quotes
  a number not in this table is quoting something nothing checked. The
  `sentence()` renderer emits text and clause together for exactly this.
- **The refusals are content, not an appendix.** "No forecast is supported at
  any horizon", "there is no pay benchmark in this data", "there is no
  sentence here for an investor" are among the most decision-relevant outputs
  the project has, and they are the ones a slide deck drops first.
- **`published` does not mean uncaveated.** Every publishable claim carries a
  clause; the status grades the upstream verdict. A slide that renders the
  text without the clause is publishing a different claim.
- **C8 travels with any NVIDIA share sentence.** Task 06 §11 offered it as the
  one unqualified cross-company sentence in the repository. It is not, and the
  pointer must survive into the deck.
- **The audience table in §10 is a briefing plan.** Product managers can be
  handed 81 sentences; investors none.

And the standing lesson, in the form it takes here: **a handover section is a
prediction, not an instruction** (C7). This one is no exception. Task 10
should read the brief before it reads §13.
