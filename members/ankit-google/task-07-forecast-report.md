# Task 07 — Demand Forecasting Report (Google)

**Specialist:** Ankit Ranjan · **Company:** Google (Alphabet) · **Date:** 2026-08-16

Input: the six-company panel Task 06 built, restricted to the **7-publisher
common panel** — **439 Google postings** inside a 1,684-posting pool, across
**11 observed months** of 2023 (February has no panel; §7). Output: 15 aggregate
tables, 8 figures, a machine-readable evidence report, and a scipy/statsmodels
cross-check of every statistic written by hand.

- **Method rationale (team standard):** [`docs/task-07-demand-forecasting-methods.md`](../../docs/task-07-demand-forecasting-methods.md)
- **Code:** [`src/forecast.py`](../../src/forecast.py) · [`src/build_forecast.py`](../../src/build_forecast.py) · [`src/validate_forecast.py`](../../src/validate_forecast.py)
- **Tests:** [`tests/test_forecast.py`](../../tests/test_forecast.py) (66) — **425 passing** in the suite
- **Validation evidence:** [`docs/task-07-forecast-validation.md`](../../docs/task-07-forecast-validation.md) — 15/15 checks pass
- **Machine-readable report:** [`task-07-forecast-report.json`](task-07-forecast-report.json)
- **Tables:** [`task-07-tables/`](task-07-tables/) · **Figures:** [`task-07-figures/`](task-07-figures/)

```bash
python src/build_forecast.py
python src/validate_forecast.py
python -m pytest tests/ -q
```

**Every number below is within-2023, from a single collection window and a
single upstream dataset.** None of it is a claim about Google's hiring in any
other year, and none of it is a claim about hiring — only about postings we can
see, on seven job boards, as a share of five named rivals.

---

## 1. The headline: Google is the most forecastable series in the set, and it is not forecastable

Google clears the forecastability gate. It is the thickest series of the six —
a 25-posting floor where Snowflake's floor is one — and it has the **lowest
coefficient of variation of any company on the panel** at 0.274. If any series
here were going to support a forecast, it is this one.

It does not. Three findings, in the order they were established:

1. **Google's share is really moving.** 74% of its month-to-month variance
   survives the binomial-noise correction; a constant share is rejected at
   p = 4.2e-05. The wiggles are not sampling artefacts (§3).
2. **Nothing predicts them.** Seven models, six rolling origins, and the best
   one — Task 05's own log-linear trend — cannot beat *repeating last month*
   at any conventional significance level (§4).
3. **The one-month interval spans a factor of 3.15.** It runs from 12.8% to
   40.3% around a point of 23.4%. It excludes essentially nothing anyone might
   have believed before reading it, so the maximum useful horizon is **zero**
   (§5).

The deliverable is therefore a forecast that is published **marked
unsupported**, together with the measurement that says why. That is the finding.
Producing a confident-looking 2024 number from eleven monthly points would have
been easy, and every part of this task exists to explain why it would have been
wrong.

---

## 2. What Google's share actually did in 2023

Share of the 7-publisher common panel, February excluded:

| Month | Google postings | Panel pool | Share |
| --- | --- | --- | --- |
| 2023-01 | 41 | 158 | 25.9% |
| 2023-02 | — | — | *no panel* |
| 2023-03 | 29 | 79 | 36.7% |
| 2023-04 | 29 | 70 | **41.4%** (high) |
| 2023-05 | 25 | 83 | 30.1% |
| 2023-06 | 36 | 133 | 27.1% |
| 2023-07 | 73 | 216 | 33.8% |
| 2023-08 | 55 | 230 | 23.9% |
| 2023-09 | 34 | 144 | 23.6% |
| 2023-10 | 45 | 182 | 24.7% |
| 2023-11 | 26 | 192 | **13.5%** (low) |
| 2023-12 | 46 | 197 | 23.4% |

Mean 27.7%, range 13.5%–41.4%. Two features matter for everything downstream.

**The denominator more than triples across the year** — 70 in April, 230 in
August. That is the panel filling up, not the market growing, and it is exactly
the reason §6 refuses to forecast Google's posting *count*.

**November is a 10-point drop followed by a 10-point rebound.** Google's
postings fall from 45 to 26 while the pool holds steady at ~190. Nothing in
this data says whether that is a hiring pause, a feed hiccup on one board, or
a rival's batch diluting the pool. It is a single month, it is the largest
move in the series, and §5 shows it is also **half of the published prediction
interval**.

Figure [`02-panel-share-series.png`](task-07-figures/02-panel-share-series.png)
draws all six series with the February gap left open.

---

## 3. The gate: Google passes, four of five rivals do not

Before any model is fitted, `forecastability_table` asks two questions of each
series — is it thick enough, and does it move more than sampling makes it move.

| Company | smallest month | overdispersion | signal share | verdict |
| --- | --- | --- | --- | --- |
| **Google** | **25** | 3.82 | **74%** | **forecastable** |
| Meta | 11 | 4.41 | 77% | forecastable |
| Microsoft | 4 | 6.76 | 85% | too_thin |
| Databricks | 4 | 2.91 | 66% | too_thin |
| NVIDIA | 2 | 2.57 | 61% | too_thin |
| Snowflake | 1 | 6.96 | 86% | too_thin |

Two things here are easy to get backwards.

**Every one of the six rejects a constant share.** Signal shares run 61% to
86%; the weakest, NVIDIA, sits at p = 0.0043 and Snowflake at 5.2e-11. "The
data is too noisy to say anything" is the comfortable summary and it is
**wrong** — these series are moving for real reasons. The four refusals are
about *cell size*, which is a different objection: at a two-posting month you
cannot tell which real reason.

**Google is refused nothing and gains nothing.** Its 25-posting floor is more
than double the next company's, and it still ends up on persistence in §4.
Passing the gate buys the right to be tested, not the right to be believed.

The 5-posting floor is a declared threshold, so
[`gate-threshold-sensitivity.csv`](task-07-tables/gate-threshold-sensitivity.csv)
publishes the alternatives: at 3, four companies pass; at 5 and at **10 — Task
06's own floor** — the answer is Google and Meta both times. Tightening to the
threshold the previous task already committed to changes nothing.

Figure [`01-forecastability-gate.png`](task-07-figures/01-forecastability-gate.png).

---

## 4. Seven models, six origins, and none of them beats doing nothing

Every model is fitted on an expanding window and scored only on months it has
never seen. Google gets six one-step origins (July through December).

Pooled across all six companies, 36 paired one-step forecasts:

| model | RMSE (log share) | vs naive | MASE | DM p vs naive |
| --- | --- | --- | --- | --- |
| ses | 0.5009 | 0.94× | 0.94 | 0.2165 |
| **naive** | **0.5344** | 1.00× | 1.00 | — |
| holt_damped | 0.5482 | 1.03× | 1.06 | 0.6566 |
| ar1 | 0.5694 | 1.07× | 1.13 | 0.4811 |
| drift | 0.5798 | 1.08× | 1.10 | 0.1772 |
| loglinear | 0.5832 | 1.09× | 1.21 | 0.5104 |
| mean | 0.6166 | 1.15× | 1.16 | 0.3364 |

**Five of the six challengers are worse than persistence outright.** The one
that is better, simple exponential smoothing, is 6% better at p = 0.22.

On Google's own six origins the ordering is different again — loglinear 0.300,
ses 0.332, holt_damped 0.342, ar1 0.358, mean 0.366, **naive 0.372**, drift
0.393. Task 05's own trend model is the best of the seven *on Google*, 19%
below naive, and the paired test on six errors returns **t = −1.15, p = 0.25**
— the right sign for a better model, and nowhere near α. Six numbers cannot
separate them, and the honest reading is that they are not separated.

That per-company ordering is worth dwelling on, because it is what selection by
ranking would have published:

| company | lowest-RMSE model | its RMSE | naive | selected |
| --- | --- | --- | --- | --- |
| Google | loglinear | 0.300 | 0.372 | naive |
| Meta | ses | 0.227 | 0.229 | naive |
| Databricks | mean | 0.350 | 0.514 | naive |
| Snowflake | ses | 0.957 | 1.002 | naive |
| Microsoft | naive | 0.333 | 0.333 | naive |
| NVIDIA | naive | 0.379 | 0.379 | naive |

**Four of six companies would have been given a different model, and those four
name three different models between them.** Six series from one panel, one
window and one collection process do not have three data-generating processes.
They have six small samples. `select_model` therefore requires a Diebold–Mariano
test rather than a ranking, and every published selection is
`benchmark_not_beaten`.

The refusal was also reached under the most permissive test available. The DM
p-value uses a normal tail on 36 pairs; recomputed under Student's t it becomes
0.2247, and under the Harvey–Leybourne–Newbold small-sample correction 0.2311.
Both corrections move **every** p-value up
([validation §5.2](../../docs/task-07-forecast-validation.md)).

Figures [`03-backtest-accuracy.png`](task-07-figures/03-backtest-accuracy.png)
and [`04-selection-vs-ranking.png`](task-07-figures/04-selection-vs-ranking.png).

---

## 5. Where the forecast stops — and the interval is two of Google's own months

Prediction intervals here are built from actual backtest residuals rather than
from a normal assumption: the endpoints are order statistics, so an interval
either exists in the sample or is refused.

| horizon | residuals | achieved level | interval span | usable |
| --- | --- | --- | --- | --- |
| h=1 | 12 | 84.6% | **3.15×** | no — past the 3× limit |
| h=2 | 10 | 81.8% | 2.19× | unreachable — h=1 failed |
| h=3 | 8 | — | none exists | no — needs ≥10 residuals |

**Maximum useful horizon: 0.** Unchanged if the residuals are pooled across all
six companies instead of the two published ones (h=1 spans 4.34× there), so the
verdict is not an artefact of the pooling choice.

Now the uncomfortable part, and the single most useful sentence in this report
for anyone who wants to use the number. The published h=1 interval runs from
−0.602 to +0.545 in log points. Those two values are not summary statistics:

- **−0.602 is Google's November residual** — the month the share fell to 13.5%.
- **+0.545 is Google's December residual** — the month it came back to 23.4%.

Both extremes of the twelve-residual pool are Google's, and they are
**consecutive months**. At n = 12 the order-statistic rule puts the interval at
the full observed range of the residuals — the widest the sample can produce —
and that still only buys a claimed 84.6% coverage. So the width of the interval
this task publishes is, almost exactly, the size of one unexplained swing in
Google's own autumn.

That is why the answer is a refusal rather than a caveat. An interval whose
span is set by a single unexplained pair of months is not a quantification of
uncertainty about hiring; it is a measurement of how little one board-share
series pins down.

Figure [`05-horizon-limits.png`](task-07-figures/05-horizon-limits.png).

---

## 6. Why there is no forecast of Google's posting *count*

The question everyone actually wants — "how many jobs will Google post in
January?" — is refused, and Task 07 measures the refusal instead of asserting
it. Persistence was backtested on three series over the same months:

| series | what it measures | naive one-step RMSE | typical error |
| --- | --- | --- | --- |
| panel pool | total panel postings — **nobody's demand** | 0.293 | ×1.34 |
| company share | share of that pool — the identified object | 0.534 | ×1.71 |
| company count | Google's panel postings — not identified | 0.646 | ×1.91 |

**The series containing no company's hiring at all is the easiest of the three
to predict.** A count forecast would score respectably for a reason that has
nothing to do with Google: it would largely be forecasting the crawler. This is
Task 06 §1.3's "levels are not identified" turned into a number, and it is in
[`levels-vs-shares.csv`](task-07-tables/levels-vs-shares.csv) rather than in a
limitations paragraph.

Figure [`06-levels-vs-shares.png`](task-07-figures/06-levels-vs-shares.png).

---

## 7. What Task 05 handed over, item by item

Task 05 §11 gave Task 07 six instructions. Five were followed as written; one
turned out not to survive contact with the panel.

| Task 05 said | What Task 07 did |
| --- | --- |
| Model `log_growth`, not percentage change | All models fit on `log(share)`; errors read as factors |
| Fit no annual seasonal term | `seasonal_naive` **raises** rather than being absent |
| Treat February as missing, not low | Derived from the panel, not hardcoded; the Jan→Mar step is **two** calendar months |
| Monthly is the shortest safe frequency | Monthly; the panel carries ≤3 publishers in any month, so weekly cells are single digits |
| `posting_date` is a discovery date, capping the horizon | The horizon is capped at **0** by measurement, well short of that argument |
| Exclude or dummy W24, W30, W34 | **Partly.** See below. |

The last row is the one that needed work. Two of Task 05's three flagged
Google spike weeks leave on their own — not because of monthly frequency, but
because `via Google Careers` (W24) and `via The Muse` (W34) are **not on the
common panel**, so those postings never enter the series at all.

**W30 does not leave.** `via Recruit.net` is on the panel, and in July it
carries **24 of Google's 73 panel postings** — against 6 in the next-highest
month and 0–3 in all the rest — and 24 of the 29 Recruit.net postings the
*whole panel* saw that month.
A batch that lands almost entirely on one company inflates that company's
numerator far more than the shared denominator, which is precisely the
assumption the share is supposed to rely on, failing.

The size of it: **Google's July share is 33.8%; without Recruit.net it is
26.2%** — a 7.6 pp move, larger than Google's entire published Task 06 delta.

So the panel is swept one publisher at a time
([`publisher-batch-sensitivity.csv`](task-07-tables/publisher-batch-sensitivity.csv)),
and the conclusions are checked on all seven resulting panels. Both hold every
time: no model beats naive, and the maximum useful horizon stays 0. Dropping
Trabajo.org is the only verdict that moves at all — Meta falls below the cell
floor, leaving one company and six residuals, too few to bound an 80% interval
by any route.

One near-miss is worth reporting rather than burying. On the panel without
BeBee, `loglinear` reaches **p = 0.0512 with a positive statistic** — meaning
it is significantly *worse* than naive, 24% higher RMSE. A rule that read the
p-value column alone, or ran at α = 0.10, would have published Task 05's trend
model on the strength of it being reliably wrong. `select_model` refuses it on
the sign of the statistic, and the case is pinned in the suite.

---

## 8. Google's skills: 2 of 15 can be modelled, and the two failures are the interesting ones

The same gate, applied to Google's top 15 skills as a share of skilled postings
per month:

| verdict | n | which |
| --- | --- | --- |
| forecastable | 2 | **Python** (69% signal, p = 0.0005) · **SQL** (60% signal, p = 0.0059) |
| noise_only | 2 | **Java** (p = 0.094) · **R** (p = 0.119) |
| too_thin | 11 | BigQuery, C++, Go, Hadoop, JavaScript, Linux, Looker, Machine Learning, NoSQL, Scala, TensorFlow |

Java and R are the only `noise_only` verdicts anywhere in Task 07 — the only
two series in the whole task whose movement is fully explained by sampling. Both
are large, visible shares (Java averages 22.8%, R 36.0%) that trace a perfectly
plausible-looking line across the year, and neither is moving more than 50
postings a month makes them move. For those two the honest forecast is the
mean, and any sentence of the form "R is declining at Google" is unsupported.

This is Task 05's warning arriving with a number attached: "Looker is emerging
at Google" was the sentence that task existed to prevent, and Looker is
`too_thin` here on a one-posting month.

[`skill-forecastability-google.csv`](task-07-tables/skill-forecastability-google.csv).

---

## 9. What this task overturned in Task 06 — correction C5

Building the calendar-time index surfaced an inconsistency inside Task 06.
Its §11 told Task 07 that "February is missing for the panel entirely", while
its own H1 aggregate counted **97 February postings** on the window-level
7-publisher panel. The panel is defined over the whole window; no publisher
carries all six companies *within* February, so those 97 postings were
attributed to a panel that does not exist in that month.

Recomputed with February excluded, per Task 06's own stated rule (H1 n falls
620 → 523):

| company | Task 06 published | corrected | change |
| --- | --- | --- | --- |
| **Google** | **−4.84 pp** | **−6.56 pp** | −1.72 |
| Meta | +1.23 pp | +4.15 pp | **+2.91** |
| Microsoft | +5.09 pp | +4.61 pp | −0.47 |
| Snowflake | −7.77 pp | −7.73 pp | +0.04 |
| Databricks | +2.89 pp | +1.91 pp | −0.98 |
| NVIDIA | +3.40 pp | +3.62 pp | +0.23 |

(The left column is Task 06's own rounding; `february-correction.csv` recomputes
the with-February delta at full precision and agrees to within 0.01 pp, which is
the column the `change` figures are differenced from.)

**Every sign survives.** Task 06's conclusions all hold — Google's share of the
shared pool fell over the year, NVIDIA's rose — and the correction makes
Google's decline *larger*, not smaller. The largest relative move is Meta's,
which more than triples. Registered as
[C5](../../docs/corrections.md#c5--task-06s-h1-aggregate-counts-february-on-a-panel-that-does-not-exist-in-february);
Task 06's submitted wording stays as submitted, per the team's corrections rule.

Figure [`08-february-correction.png`](task-07-figures/08-february-correction.png).

---

## 10. The published forecast

[`forecast.csv`](task-07-tables/forecast.csv) carries three horizons for the
two companies that passed the gate, every row marked `supported = False`:

| company | 2024-01 point | 80% interval (achieved 84.6%) | supported |
| --- | --- | --- | --- |
| Google | 23.4% | 12.8% – 40.3% | **False** |
| Meta | 16.8% | 9.2% – 28.9% | **False** |

Both targets are outside the collection window (`out_of_window = True`), and
the model is persistence for both.

Two properties of this table are deliberate and should not be "fixed"
downstream:

- **The shares do not sum to 1.** Google and Meta hold about 40% of the panel
  between them; the other four failed the gate but did not leave the
  denominator. Rescaling the two survivors to sum to 1 would report Google at
  **58.2%** of a panel it holds 23.4% of. `compositional_normalise` raises
  rather than allowing it, and the report carries
  `composition_incomplete_by: [databricks, microsoft, nvidia, snowflake]`.
- **The unusable forecast is published anyway, with its band.** Omitting it
  invites the number to be regenerated elsewhere without the interval. The
  point and the 3.15× band together are the argument.

Figure [`07-forecast.png`](task-07-figures/07-forecast.png) draws it that way —
band, point, and the red verdict in the title.

---

## 11. Limitations

- **Eleven monthly observations.** Six one-step origins per company. Every
  interval, every p-value and every ranking in this report is small-sample, and
  the report's conclusions are refusals precisely because of it.
- **One year, so no seasonality is estimable at all** — not weakly, but
  structurally. Any 2024 number that appears to know about January is wrong.
- **The 7-publisher panel is a small slice.** 439 of Google's 846 postings
  ([C4](../../docs/corrections.md#c4--googles-posting-count-is-846-not-848)). A
  share of that panel is not a share of hiring, and §7 shows a single publisher
  can move a month by 7.6 pp.
- **The 3× publication limit and the 5-posting cell floor are declared
  judgements**, not derived constants. Both are published with sensitivity
  tables; neither changes the verdict across the range tested.
- **`posting_date` is an aggregator discovery date** (C3), so month attribution
  is crawl-timed. Monthly smooths this; weekly would not.
- **The DM p-values use a normal tail.** That is the most permissive of the
  three tails checked, and it is the one the refusal was reached under. The
  corrections only make the refusal firmer.
- **Wilson–Hilferty is an approximation.** Measured worst p-value error over
  the df range the gate actually sees (7–11): **1.85e-03**, and the
  approximation errs anti-conservatively — it rejects a constant share slightly
  more readily than the exact tail. No gate verdict flips under exact scipy.

---

## 12. What Tasks 08, 09 and 10 inherit

**Task 08 (Company similarity):**
[`panel-share-series.csv`](task-07-tables/panel-share-series.csv) is the right
input — common panel, common scale, February already open. Two constraints
carry: apply the §3 gate to **both** members of a pair before scoring it
(eleven-point correlations at this n produce large coefficients from nothing),
and never fill February, or companies will look similar because they were
simultaneously invisible.

**Task 09 (Insight generation):** **there is no forecast sentence available
from this task.** The publishable sentences are about forecastability —
that all six series carry 61–86% real signal; that no model beats persistence;
that the collection process is easier to predict than any company's demand;
that Java and R at Google are moving no more than sampling explains. Any
sentence shaped like "Google is expected to reach X% by Q1" is unsupported at
every horizon this data offers.

**Task 10 (Final presentation):** if a forecast chart appears, it appears with
its band, and the band is 12.8%–40.3%. Figure 07 is the pattern. A
point-only version of that chart would misrepresent the single clearest result
this task produced.

---

## 13. Deliverables

| | |
| --- | --- |
| Shared code | [`src/forecast.py`](../../src/forecast.py) — gate, models, backtest, DM test, order-statistic intervals, horizon verdict, compositional guard |
| Runner | [`src/build_forecast.py`](../../src/build_forecast.py) — 15 tables, 8 figures, JSON report |
| Validator | [`src/validate_forecast.py`](../../src/validate_forecast.py) — 15 checks against scipy and statsmodels |
| Tests | [`tests/test_forecast.py`](../../tests/test_forecast.py) — 66 new, 425 in the suite |
| Team standard | [`docs/task-07-demand-forecasting-methods.md`](../../docs/task-07-demand-forecasting-methods.md) |
| Validation evidence | [`docs/task-07-forecast-validation.md`](../../docs/task-07-forecast-validation.md) |
| Correction | [C5](../../docs/corrections.md#c5--task-06s-h1-aggregate-counts-february-on-a-panel-that-does-not-exist-in-february) |
| Tables | [`task-07-tables/`](task-07-tables/) — 15 CSVs |
| Figures | [`task-07-figures/`](task-07-figures/) — 8 PNGs |
| Evidence report | [`task-07-forecast-report.json`](task-07-forecast-report.json) |

Standing checks: the Task 01 `personal_data_columns_present` guard and Task 06's
`forbidden_columns` guard run over all 15 committed tables — `_write` raises
rather than writing a table that fails either. `privacy.passed: true`. Row-level
data remains git-ignored; only aggregates are committed.
