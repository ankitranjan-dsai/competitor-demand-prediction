# Task 07 — Demand Forecasting

**Team standard.** Task 05 gave each of us a curve, Task 06 put the curves on
one axis and said which comparisons that axis supports. Task 07 is asked to
extend them forward. This document records what this data can be extended
forward, what cannot, how a model earns the right to be published instead of
persistence, and where the forecast is required to stop. Tasks 08, 09 and 10
read the output, and a point forecast that escapes this document without its
interval becomes a slide, then a recommendation.

- **Code:** [`src/forecast.py`](../src/forecast.py) · [`src/build_forecast.py`](../src/build_forecast.py) · [`src/validate_forecast.py`](../src/validate_forecast.py)
- **Tests:** [`tests/test_forecast.py`](../tests/test_forecast.py) (66; 425 in the suite)
- **Validation:** [`docs/task-07-forecast-validation.md`](task-07-forecast-validation.md) — every hand-written statistic checked against scipy and statsmodels
- **Google findings:** [`members/ankit-google/task-07-forecast-report.md`](../members/ankit-google/task-07-forecast-report.md)
- **What this task overturned:** [`docs/corrections.md`](corrections.md) —
  [C5](corrections.md#c5--task-06s-h1-aggregate-counts-february-on-a-panel-that-does-not-exist-in-february),
  the February month inside Task 06's own H1 half
- **Inherited from:** [`docs/task-06-competitor-comparison-methods.md`](task-06-competitor-comparison-methods.md) §11
- **Legal position:** unchanged — no new source, no new collection, no new field

```bash
python src/build_forecast.py           # 15 tables, 8 figures, 1 JSON report
python src/validate_forecast.py        # scipy/statsmodels cross-check
python -m pytest tests/ -q
```

The brief asks for a demand forecast with an accuracy statement. The forecast
is committed, the accuracy statement is committed, and the accuracy statement
says the forecast is not usable at any horizon. **That is the deliverable**,
not a failure to produce one. §6 is where the number comes from, §8 is where it
stops, and §9 is the check that keeps a refused forecast from being quietly
rescued by rescaling.

Three findings in one line each, so the rest of the document can be read as
their justification:

1. **All six series carry real signal** — 61% to 86% of their variance survives
   the binomial noise correction (§5.2). "It is all noise" would have been the
   easy conclusion and it is the wrong one.
2. **No model beats persistence** (§6). The best challenger is 6% lower on
   RMSE at p = 0.22, and the per-company winners disagree with each other,
   which is what noise-chasing looks like.
3. **No horizon is supported** (§8). The one-month interval already spans a
   factor of 3.15, and by three months there is no distribution-free interval
   at all.

---

## 1. What is being forecast

### 1.1 The object is share of the common panel, and Task 06 chose it

Task 06 §1.3 established that cross-company **levels are not identified** in
this data: postings reach us through job boards, the boards carry each employer
differently, and Snowflake's share of the seven-publisher common panel is 23.5%
against a 40% floor. The one object that survives is each company's **share of
the common publisher panel** — the intersection of publishers that carry all
six companies.

That inheritance is not a formality; it is the entire reason a share is
forecastable here. Write the observed count for company *c* in month *t* as

```
observed(c, t) = demand(c, t) × collection(t)
```

where `collection(t)` is whatever the panel was doing that month — how many
boards were live, how deep the crawl went, whether a backfill landed. It is
common to every company on the panel by construction. In a share it cancels:

```
share(c, t) = demand(c, t) / Σ_k demand(k, t)
```

This is the same difference-in-differences logic Task 05 used within Google and
Task 06 used across companies, applied one step further out in time. The panel
factor has to be **multiplicative and common** for it to cancel, which is
exactly what restricting to the common panel buys.

`panel_share_series(frames, publishers=...)` builds it. The default publisher
set is `compare.common_publishers(frames)` — the same seven Task 06 published.

### 1.2 A count forecast is refused, and the reason is measurable

The obvious thing a reader wants is "how many postings will Google have in
January". Refusing it on the grounds that "levels are not identified" is
correct but abstract, so Task 07 measures the claim instead of asserting it.

`levels_vs_shares` backtests persistence on three series over the same months:

| series | what it measures | one-step naive RMSE (log) | typical error |
| --- | --- | --- | --- |
| panel pool | total postings on the shared panel — **nobody's demand** | 0.293 | ×1.34 |
| company share | each company's share of that pool — the identified object | 0.534 | ×1.71 |
| company count | each company's panel postings — not identified | 0.646 | ×1.91 |

**The pure-collection series is the most predictable thing in the file.** The
panel pool is a property of the crawler and the boards; it contains no
company's hiring at all, and it is nearly twice as easy to predict one month
ahead as any company's count. A count forecast would therefore be a forecast
whose accuracy is dominated by the component with no demand in it — it would
look respectable for the wrong reason.

That is why `forecast_table` publishes shares only, and why the refusal is a
line in `levels-vs-shares.csv` rather than a sentence in a limitations section.
Pinned by `test_the_crawler_is_the_most_predictable_series_in_the_file`.

### 1.3 Everything happens on the log share

Models are fitted to `log(share)`, not to the share. Three reasons:

- a share is bounded in (0, 1) and a linear model on it will forecast outside
  the bound within two steps at these variances;
- growth in this data is multiplicative — Task 05 established `log_growth` as
  the team's growth measure and Task 07 does not change it;
- errors on the log scale exponentiate into a **factor**, which is the honest
  unit for "this interval spans 3.15×" (§8).

`_safe_log` applies a Haldane–Anscombe ½ floor: a zero share becomes
`0.5 / denominator` rather than `-inf`, so one empty month cannot take every
downstream mean and every model fit to negative infinity with it. It is the
same ½ correction Task 06 §8 uses on log lifts, and it is a floor rather than a
patch — a series with a cell small enough to reach it fails the gate in §5 long
before a model sees it. Pinned by `test_safe_log_survives_an_empty_cell`.

### 1.4 The cancelling assumption is tested, not asserted

§1.1 rests on `collection(t)` being **common to every company on the panel**.
That is an assumption about the world, and Task 05 already found the case where
it fails: a single publisher dumping a backfill of one company's postings
inflates that company's numerator far more than it inflates the shared
denominator, and the share moves for a reason that has nothing to do with
hiring.

Task 05 §3 flagged three such weeks for Google. Two of them leave on their own,
because monthly frequency is not what removes them — the panel is:
`via Google Careers` (W24) and `via The Muse` (W34) are not on the common panel,
so those postings never enter the series at all. **The third does not leave.**
`via Recruit.net` *is* on the panel, and in July it carries 24 of Google's 73
panel postings — against 6 in the next-highest month and 0–3 in all the rest —
and 24 of the 29 Recruit.net
postings the whole panel saw that month. Google's July share is 33.8%; drop
Recruit.net and it is **26.2%**, a 7.6 pp move, larger than Task 06's entire
published Google delta.

So `batch_sensitivity` drops each panel publisher in turn and re-runs the whole
verdict — gate, contest, horizon. `publisher-batch-sensitivity.csv` publishes
all seven. The result:

| dropped | forecastable | any model beats naive | h=1 span | max horizon |
| --- | --- | --- | --- | --- |
| — (published) | google, meta | no | 3.15× | 0 |
| via BeBee | google, meta | no | 3.63× | 0 |
| via Indeed | google, meta | no | 3.14× | 0 |
| via Ladders | google, meta | no | 3.12× | 0 |
| via LinkedIn | google, meta | no | 4.79× | 0 |
| via Recruit.net | google, meta | no | 3.10× | 0 |
| via SimplyHired | google, meta | no | 3.08× | 0 |
| via Trabajo.org | **google only** | no | — | 0 |

Both conclusions survive all seven panels. The one verdict that does move is
Trabajo.org's: without it Meta drops below the cell floor, and with one company
left there are six residuals — not enough to bound an 80% interval at all, so
the horizon verdict stays 0 by a different route.

The point of the section is not that nothing changed. Individual shares move by
several points, and §7.1 shows a panel on which the *contest* nearly changes.
The point is that the refusal is a property of the data rather than of one
publisher. Pinned by
`test_refusal_survives_dropping_any_single_panel_publisher`.

---

## 2. Time is calendar time, not array position

February 2023 has **no common publisher panel at all**. Not a low month — an
absent one. Task 05 traced it and C1 records it: Google's February holds **22
postings from 6 publishers**, against 23 publishers in January and 21 in March.
The boards that vanish in February come back in March. Nothing in the data
distinguishes "Google paused hiring" from "the feed lost 17 boards for four
weeks", and the publisher counts fit the second exactly.

So the series has eleven observed periods in a twelve-month window, and
**January and March are adjacent in the array and two months apart in the
world**. Every model in this module indexes on `period_ordinal`, which converts
`YYYY-MM` to months since epoch, rather than on position:

```python
period_ordinal(["2023-01", "2023-03"])   # -> [24276, 24278]   two steps
#              positional index          # -> [0, 1]           one step
```

A drift model on a positional index spreads January's move to March over one
step instead of two and forecasts a step that is twice too large; a log-linear
trend fits its slope against the wrong denominator; an AR(1) treats a two-month
mean reversion as one. Pinned by `test_gap_is_measured_in_calendar_time`.

The gap is **derived, not declared**. `unobserved_periods(frames)` asks
`compare.common_panel_by_period` which months have an empty panel, so if a
teammate's company set produces a different common panel with a different hole,
the code finds theirs rather than assuming February. Pinned by
`test_unobserved_periods_is_derived_not_hardcoded`.

**February is missing, not zero.** A zero would say the companies stopped
posting; missing says we could not see them. `is_observed` is False for the
month, it is excluded from fitting, from the backtest, and from every
aggregate — and §10 is the correction that arises because Task 06 did not do
this consistently.

---

## 3. There is no seasonal model, and there cannot be one

The data covers 2023-01 to 2023-12: **one year, zero complete seasonal
cycles**. A month-of-year term is therefore perfectly collinear with whatever
else happened in that month, and there is no second occurrence of any month to
separate the two.

This matters more here than the arithmetic suggests, because the two months a
seasonal model would seize on are both already known to be collection
artefacts. February's "trough" is the missing panel (§2). August's "peak" is
one publisher arriving: 24 of that month's 102 Google postings come from The
Muse in its first month in the feed, and 21 of those land on a single day,
`2023-08-23` (C1). A seasonal term would fit both, name them seasonality, and
project them into 2024 with a confident label.

`seasonal_naive` exists in the module and **raises `NotImplementedError`** with
that explanation. It is a refusal at the call site rather than an absence,
because an absent function invites a teammate to add one.

The same applies to any library fit that estimates seasonality by default. If
you reach for `statsmodels.ExponentialSmoothing`, pass `seasonal=None`
explicitly — and read §11 first about why the shared module does not import it.

---

## 4. Frequency: monthly is the shortest usable step

Weekly would give 52 points instead of 12, which is tempting on a series this
short. It is not available:

- the common panel carries **≤3 publishers in any single month**, so a weekly
  cell on that panel is routinely a handful of postings and often zero;
- `posting_date` is an **aggregator first-seen date**, not a publication date
  (C3), so weekly bucketing measures crawl timing more than posting timing;
- Task 05 already found that all three of its weekly spikes were single-day
  publisher backfills.

Monthly, eleven observed points, six one-step origins per company. That is a
small number and the rest of this document is largely about not pretending
otherwise.

---

## 5. The forecastability gate, run before any model is fitted

`forecastability_table` is Task 07's twin of Task 06's comparability gate, and
it answers a different question: not "can these two be compared" but "is this
series thick enough, and moving enough, to be worth a model". Three verdicts.

### 5.1 `too_thin` — the cell decides, not the shape

A series is refused if any observed period has fewer than `MIN_CELL_MONTH = 5`
postings in the numerator, or if there are fewer than `MIN_OBSERVATIONS = 8`
observed periods.

A thin series can look like it is moving beautifully. Snowflake reaches a month
with **one posting** on the common panel; that month's share carries a relative
standard error near 100%, and a model fitted through it is fitting the
denominator's rounding. Thin beats interesting, always.

### 5.2 `noise_only` — is it moving more than sampling makes it move?

The second branch answers a question that a plot cannot. A share of 0.28
observed on 158 postings carries a binomial standard error near 0.036 — and the
whole range Google's series covers over the year is about 0.28. So a visibly
wiggly line is entirely compatible with one constant proportion observed
eleven times through small samples.

`homogeneity_test` runs Pearson's chi-square for homogeneity of proportions
across the observed periods, and splits the variance:

```
overdispersion = chi2 / df           # 1.0 = every wiggle is sampling noise
signal_share   = max(0, 1 - 1/overdispersion)
```

`signal_share` is the fraction of observed variance left once binomial noise is
removed. If a constant share is not rejected at `ALPHA = 0.05`, the verdict is
`noise_only` and the honest forecast is the mean, not a trend.

### 5.3 What the real data says

| company | smallest observed month | overdispersion | signal share | verdict |
| --- | --- | --- | --- | --- |
| google | 25 | 3.82 | 74% | **forecastable** |
| meta | 11 | 4.41 | 77% | **forecastable** |
| microsoft | 4 | 6.76 | 85% | too_thin |
| databricks | 4 | 2.91 | 66% | too_thin |
| nvidia | 2 | 2.57 | 61% | too_thin |
| snowflake | 1 | 6.96 | 86% | too_thin |

**Every one of the six rejects a constant share**, the weakest at p = 0.0043
and four of them below 1e-4. This is the finding that is easy to miss: the
series are not noise. They carry between 61% and 86% real variance. Four of
them are refused anyway, on cell size — which is a different objection, and
worth keeping distinct in any write-up. "Too thin to model" is not "nothing is
happening".

The two branches are genuinely separate, and the skill screen proves it: of
Google's top 15 skills, **11 are `too_thin`, 2 are `noise_only`** (Java at
p = 0.094, R at p = 0.119 — both are moving no more than sampling explains) and
**2 are forecastable** (Python and SQL).

### 5.4 The threshold is published with its sensitivity

`MIN_CELL_MONTH = 5` is a declared choice, so `gate-threshold-sensitivity.csv`
publishes what other choices would have done:

| min_cell | forecastable |
| --- | --- |
| 3 | databricks, google, meta, microsoft |
| **5 (published)** | **google, meta** |
| 10 (Task 06's own floor) | google, meta |

The verdict set is **identical at 5 and at 10**, which is the number that
matters: the published threshold is the permissive end of the range Task 06
already committed to, and tightening it to Task 06's own `MIN_CELL` changes
nothing. Only loosening to 3 admits more companies, and at 3 the admitted set
includes a company with a four-posting month.

### 5.5 The gate refuses; it does not caveat

A refused series gets no model, no point forecast and no interval — not a
forecast with a warning attached. The reason is behavioural rather than
statistical: a number and a caveat travel at different speeds. By the time a
share reaches Task 09's insight list or Task 10's slide, the number is on the
slide and the caveat is in the appendix. `verdict` is a column in every table
so that a downstream chart cannot render the point without it.

---

## 6. The backtest is the only accuracy statement

### 6.1 Rolling origin, expanding window

`rolling_origin_backtest` fits on the first *k* observed periods, forecasts
1, 2 and 3 steps ahead, scores against what actually happened, then advances
the origin. `MIN_TRAIN = 5`. On eleven observed periods that yields **6
one-step origins per company, 5 two-step, 4 three-step**.

An in-sample fit is not an option here. A three-parameter damped Holt fitted on
eleven points and scored on the same eleven points reports how many parameters
it has, and the answer flatters every model in proportion to its complexity.

The property that makes the backtest worth anything is that a model never sees
its own target. That is pinned rather than assumed:
`test_backtest_never_sees_its_own_target` inserts a recording model that
reports the latest training period it was handed, and asserts it is always
strictly before the target.

### 6.2 The models, and why these

`naive` (persistence, the benchmark), `mean`, `drift`, `ses`, `holt_damped`,
`loglinear`, `ar1`. The registry is ordered with `naive` first, so a tie never
unseats it.

`holt_damped` is three parameters on eleven points and is included **on
purpose**: it is what a specialist reaching for `ExponentialSmoothing` gets by
default, and the backtest is the right place to find out what that costs.
`loglinear` is Task 05's `log_growth` per company, so the trend model the
earlier task published is evaluated on equal terms rather than inherited.

### 6.3 The result

One-step, all six companies pooled, 36 paired forecasts:

| model | RMSE (log share) | vs naive | MASE |
| --- | --- | --- | --- |
| ses | 0.5009 | 0.94× | 0.94 |
| **naive** | **0.5344** | 1.00× | 1.00 |
| holt_damped | 0.5482 | 1.03× | 1.06 |
| ar1 | 0.5694 | 1.07× | 1.13 |
| drift | 0.5798 | 1.08× | 1.10 |
| loglinear | 0.5832 | 1.09× | 1.21 |
| mean | 0.6166 | 1.15× | 1.16 |

Every model except SES is **worse** than doing nothing, and SES is 6% better.
Note what that says about §5's finding: the series carry 61–86% signal and
still nothing forecasts them, which is a statement about how little of that
signal is *persistent* — it moves, but not in a way last month predicts.

---

## 7. Selection is a test, not a ranking

### 7.1 The rule

`select_model` picks a challenger over the benchmark only when all three hold:

1. Diebold–Mariano p < `ALPHA` on the paired loss differential;
2. `dm_stat < 0`, so the challenger has the *smaller* loss;
3. lower RMSE than the incumbent challenger.

Otherwise the selection is `naive`, `selected_by` is `benchmark_not_beaten`,
and the report says the forecast is persistence.

**Condition 2 is the trap.** DM is a two-sided test, so a small p-value means
"these two forecasters differ", not "the challenger is better". On the real
contest `drift` posts the **smallest p-value of any model — 0.177** — precisely
because it is reliably worse than naive, and it ranks fifth of seven on RMSE.

It does not clear α = 0.05, so nothing is selected either way. What the p-value
column *does* do is sort the worse model to the top: read as a ranking, or
under a looser α, drift is the model that gets published. On the Google–Meta
pooling it reaches p = 0.0651.

§1.4's sensitivity sweep produces the case outright. On the panel without
BeBee — one publisher's difference from the panel actually published —
`loglinear` lands at **p = 0.0512 with a positive statistic** and an RMSE of
0.611 against naive's 0.494, 24% worse. `mean` is right behind it at p = 0.0534,
41% worse. Had the feed carried one fewer board, a p-value-only rule at α = 0.10
would have published Task 05's own trend model on the strength of it being
reliably wrong. `select_model` refuses it on the sign, and
`test_a_significantly_worse_model_is_refused_on_the_real_data` pins that,
including at α = 0.10.

### 7.2 Why a ranking is not allowed here

Six origins per company. At that sample size the RMSE ordering is mostly noise,
and the real data shows it plainly:

| company | selected | lowest-RMSE model | its RMSE | naive's RMSE |
| --- | --- | --- | --- | --- |
| google | naive | loglinear | 0.300 | 0.372 |
| meta | naive | ses | 0.227 | 0.229 |
| databricks | naive | mean | 0.350 | 0.514 |
| snowflake | naive | ses | 0.957 | 1.002 |
| microsoft | naive | naive | 0.333 | 0.333 |
| nvidia | naive | naive | 0.379 | 0.379 |

**Four of six companies have a lowest-RMSE model that is not naive, and they
name three different models between them.** Six series drawn from the same
panel, the same window and the same collection process do not have three
different true data-generating processes. They have six small samples. Ranking
would have published four different models and a story about why Google is
trending while Databricks is mean-reverting.

The best challenger overall, SES, sits at **p = 0.2165**. Nothing clears the
bar for any company at any horizon, so every published selection is naive.
Pinned in both directions — refusal by
`test_selection_keeps_naive_when_a_challenger_only_leads_on_rmse`, where SES
comes out 41% below naive on a pure-noise series and is still refused, and
acceptance by `test_selection_does_take_a_challenger_that_genuinely_wins`, so
the rule is conservative rather than vacuous.

### 7.3 The DM p-value is the most permissive test available

`diebold_mariano` uses a normal tail via `math.erfc`, and 36 paired errors is
not asymptotic. That approximation errs toward **accepting** a challenger, so
the refusal was reached under the easiest test in the family.
[`task-07-forecast-validation.md`](task-07-forecast-validation.md) §5.2
recomputes every contest under Student's t and under the
Harvey–Leybourne–Newbold small-sample correction, for both poolings. Every
p-value moves up; nothing reaches 0.05 under any of them.

---

## 8. Where the forecast stops

### 8.1 Intervals are order statistics, not quantiles

`empirical_interval` bounds the interval with two actual backtest residuals.
For *n* residuals at level *L*:

```
j = floor((1 - L) / 2 * (n + 1))
interval  = [ residual_(j) , residual_(n+1-j) ]
coverage  = (n + 1 - 2j) / (n + 1)
```

This is distribution-free — it assumes only that residuals are exchangeable,
not that they are normal — and the coverage is exact rather than nominal. When
`j < 1` the level **cannot be bounded from the sample at all** and the function
refuses, returning `sufficient = False` with the number of residuals it would
need.

The first version of this used `np.quantile`, and it was wrong in a way worth
recording because it is the default thing to reach for. `np.quantile`
interpolates *between* order statistics, so its endpoints sit strictly inside
the observed residual range — it reports an interval narrower than any interval
the sample can support. Measured coverage of an interval labelled 80% was 67%,
and on the real backtest the h=1 span read **1.84× interpolated against 3.15×
measured**. That is the difference between clearing §8.3's publication limit
and failing it. `validate_forecast.py` §6 simulates both against normal, t(3)
and lognormal residuals; the interpolated version undercovers in every cell.

The consequence to state in any report: the achieved level is **not** the
requested one. At n = 12 the requested 80% becomes 11/13 = 84.6%, because only
whole order statistics exist and the module rounds **outward**.
`horizon-limits.csv` publishes `achieved_level` alongside `interval_level` for
exactly this reason, and a report should quote the achieved figure.

### 8.2 Residuals are pooled within the published set, not across everything

`empirical_interval`, `interval_coverage`, `horizon_table` and `forecast_table`
all take a `keys` filter. Pooling residuals across all six companies to widen a
thin sample is tempting and it imports the wrong errors: Snowflake's one-step
RMSE is 1.00 — months of a single posting — and pooling it into Google's
interval widened that interval by roughly half.

The published interval pools **only the gated companies**. The all-companies
version is committed too, as `horizon-limits-all-companies.csv`, because the
verdict has to be shown to be robust to the choice rather than dependent on it.

### 8.3 The horizon verdict

A horizon is usable only if it has an interval **and** that interval excludes
something. `max_factor = 3.0` is a declared threshold: an interval spanning a
threefold range of shares is compatible with nearly any story a reader arrives
with, and publishing a point beside it invites the point to be read instead.

| horizon | residuals | interval span | usable |
| --- | --- | --- | --- |
| h=1 | 12 | **3.15×** | no — past the 3× limit |
| h=2 | 10 | 2.19× | unreachable (see below) |
| h=3 | 8 | no interval at all | no — needs ≥10 residuals for 80% |

**Verdict: max useful horizon = 0.** Identical under both poolings — 4.34× at
h=1 across all six companies.

The h=2 row is the one that needs saying out loud. Its span is under the limit,
and it does not rescue the forecast, for two reasons. A horizon only counts if
every shorter horizon is usable — you cannot forecast two months out without
passing through one — and a two-step interval that is *narrower* than the
one-step interval is itself evidence that ten residuals are too few to be
measuring anything stable. `horizon_verdict` enforces the contiguity rule, and
figure 05 annotates the bar rather than leaving a reader to find it.

### 8.4 What gets published anyway, and how

`forecast.csv` still carries the three horizons for Google and Meta, with
`supported = False` on every row and `out_of_window = True` (the targets are
2024-01…03, past the collected window). The point is persistence — 23.4% for
Google, 16.8% for Meta — and the h=1 interval runs 12.8% to 40.3%.

Publishing the refused forecast rather than omitting it is deliberate. An empty
deliverable invites someone to produce the number elsewhere without the
interval; the number with its interval attached shows exactly why it is not
usable. Figure 07 draws it that way — the band, the point, and the red verdict
in the title.

---

## 9. A partial composition cannot be renormalised

Shares of the same panel sum to 1 across all six companies. Only two companies
pass the gate, so the two published forecasts sum to about 0.40 and **do not**
sum to 1.

The tempting fix is to rescale the survivors. It fabricates numbers.
Renormalising Google and Meta to sum to 1 reports Google at **58.2%** of a
panel it holds **23.4%** of, because the four omitted companies still hold
their share of the pool — they were refused a forecast, not removed from the
denominator.

`compositional_normalise` therefore requires `expected_keys` and raises
`ValueError` on any subset, naming what is missing. The report carries
`shares_sum_to_one: false` and `composition_incomplete_by: [databricks,
microsoft, nvidia, snowflake]` so a downstream chart cannot discover the gap by
accident. Pinned by `test_partial_composition_is_refused`, which asserts both
the refusal and the size of the misstatement it prevents.

---

## 10. What this task overturned in Task 06

[C5](corrections.md#c5--task-06s-h1-aggregate-counts-february-on-a-panel-that-does-not-exist-in-february).
Building §2's calendar-time index surfaced an inconsistency inside Task 06:
its §11 told Task 07 that "February is missing for the panel entirely", while
its own H1 aggregate counted **97 February postings** on the window-level
seven-publisher panel. The panel is defined over the whole window; February has
no publisher carrying all six companies within the month, so those 97 postings
sit on a panel that does not exist in the month they are attributed to.

Recomputed with February excluded, per Task 06's own stated rule (H1 n falls
620 → 523):

| company | Task 06 published | corrected | sign |
| --- | --- | --- | --- |
| google | −4.84 pp | **−6.56 pp** | unchanged |
| meta | +1.23 pp | +4.15 pp | unchanged |
| microsoft | +5.09 pp | +4.61 pp | unchanged |
| snowflake | −7.77 pp | −7.73 pp | unchanged |
| databricks | +2.89 pp | +1.91 pp | unchanged |
| nvidia | +3.40 pp | +3.62 pp | unchanged |

The left column is Task 06's own published rounding.
`february-correction.csv` recomputes the with-February delta at full precision
and agrees with it to within 0.01 pp; the `change_pp` column is that
full-precision difference.

**Every sign survives; every magnitude moves.** Task 06's conclusions all hold
— Google's share fell, NVIDIA's rose — and the largest relative move is Meta's,
which more than triples. Task 06's published wording stays as submitted, marked
with a pointer to C5, per the corrections rule.

---

## 11. Uncertainty, and why this module imports no scipy

`src/forecast.py` imports numpy and pandas and nothing else. No scipy, no
statsmodels, no prophet. The chi-square tail (Wilson–Hilferty cube-root normal
approximation), the normal tail (`math.erfc`) and both smoothing recursions are
written out in the module.

The reason is that this module holds gates that decide whether a number is
published at all, and those gates have to give the same answer in a teammate's
environment and in the grader's. A forecasting stack that changes an optimiser
default between minor versions is a poor foundation for that.

The cost is that hand-written statistics can be wrong in ways a unit test
written by the same author will not catch. So
[`src/validate_forecast.py`](../src/validate_forecast.py) imports exactly the
libraries the module refuses and checks them against each other —
it is a validator, never a dependency, and nothing in `src/`, `tests/` or the
build imports it. Results in
[`task-07-forecast-validation.md`](task-07-forecast-validation.md):

- SES, damped Holt (all three horizons), the log-linear trend and the AR(1)
  step match statsmodels to **1e-15** at fixed parameters;
- `dm_stat` is exactly `scipy.stats.ttest_1samp`'s statistic on the loss
  differential, and the `erfc` tail is exactly `scipy.stats.norm.sf`;
- Wilson–Hilferty's worst p-value error over the df range this gate uses is
  **1.85e-3**, and its critical value at α = 0.05 is within **0.15%** of exact —
  in the anti-conservative direction, which is why the gate never refuses on
  that test alone;
- interval coverage matches its claimed level under normal, t(3) and lognormal
  residuals.

`test_forecast_module_imports_no_heavy_stats_dependency` pins the promise at
the source, because the cross-check argument collapses the moment a convenience
import appears.

---

## 12. What gets committed

Row-level data stays git-ignored, as in every task since 02. Committed:

- **15 tables** in `members/<member>/task-07-tables/` — the gate and its
  sensitivity, the panel share series, backtest errors, model accuracy, the
  contest, selection, horizon limits under both poolings, interval coverage,
  the forecast, levels-vs-shares, the publisher-batch sweep, the February
  correction, and the specialist's skill screen;
- **8 figures** in `task-07-figures/`, each carrying its verdict **in the
  figure** rather than in a caption — the Task 06 pattern;
- **one JSON report**, `task-07-forecast-report.json`, machine-readable, with
  every threshold it used recorded alongside every result;
- **the validation evidence**, `docs/task-07-forecast-validation.md`.

Every committed table passes `forbidden_columns` (Task 06's identifier guard)
and the standing `personal_data_columns_present` check from Task 01 —
`_write` raises rather than writing a table that fails either, so the check
cannot be skipped by forgetting it.

---

## 13. Checklist for each specialist

1. Build the share series on the **common panel**, not on your company's own
   publishers. `panel_share_series(frames, publishers=common_publishers(frames))`.
2. Check `unobserved_periods` for your set. Do not assume February.
3. Run `forecastability_table` **before fitting anything**. Publish the verdict
   column even for the series you go on to model.
4. Publish `gate_sensitivity` alongside it. A declared threshold without its
   sensitivity is a choice hidden in a constant.
5. Run `batch_sensitivity` and check your own company's spikes from Task 05
   against the panel. A batch from a publisher that *is* on the panel does not
   cancel; one from a publisher that is not never entered the series (§1.4).
6. Backtest with `rolling_origin_backtest`. Never quote an in-sample fit.
7. Select with `select_model`, never by RMSE ranking. If a challenger wins,
   report its DM p-value **and its sign**.
8. Take intervals from `horizon_table` with `keys=` set to your published set,
   and quote `achieved_level`, not `interval_level`.
9. Get the stopping point from `horizon_verdict`. If it is 0, say so in the
   first paragraph of your report, not the last.
10. Never renormalise a partial composition. If your gate refused a company, its
    share is still in the denominator.
11. Re-run `python src/validate_forecast.py` if you touch anything in
    `forecast.py` that computes a statistic.

---

## 14. What Tasks 08, 09 and 10 inherit

- **Task 08 (Company similarity scoring):** the series in
  `panel-share-series.csv` are the right input for similarity — they are on a
  common panel and a common scale. Two constraints carry over. Correlating
  eleven-point series will produce large coefficients from noise at this
  n, so any similarity score needs the §5 gate applied to **both** members of
  a pair before the pair is scored. And February must stay a gap: a similarity
  measure that fills it with zero will find companies similar because they were
  simultaneously invisible.
- **Task 09 (Insight generation):** there is **no forecast sentence available**
  from this task. The publishable sentences are about *forecastability*: that
  all six series carry 61–86% real signal; that no model beats persistence;
  that the collection process is easier to predict than any company's demand.
  Any sentence of the form "X is expected to reach Y" is unsupported by this
  data at every horizon — §8.
- **Task 10 (Final presentation):** if a forecast chart appears, it appears
  with its interval, and the interval is the 3.15× band. Figure 07 is the
  pattern. A point-only version of that chart would misrepresent the finding,
  and the finding — a well-instrumented refusal — is more defensible than a
  number would have been.
