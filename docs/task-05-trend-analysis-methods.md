# Task 05 — Hiring Trend Analysis

**Team standard.** Every specialist runs the same time-series module
(`src/trends.py`) against their own company's Task 04 output. Task 06 puts the
four companies' curves on one axis and Task 07 forecasts them, so if one of us
aggregates by calendar month and another by ISO week, or one indexes to
January and another to the series mean, those tasks will measure our choices
instead of the market. This document records the shared structure, the
denominators, the traps we hit in real data, and what each member must do.

- **Code:** [`src/trends.py`](../src/trends.py) · [`src/build_trends.py`](../src/build_trends.py)
- **Tests:** [`tests/test_trends.py`](../tests/test_trends.py) (57 tests; 214 in the suite)
- **Google findings:** [`members/ankit-google/task-05-trend-report.md`](../members/ankit-google/task-05-trend-report.md)

```bash
python src/build_trends.py --company google
python -m pytest tests/ -q
```

The brief asks for hiring velocity, growth, decline and seasonal spikes. Three
of those four are reportable from one year of aggregator data. The fourth is
not, and §6 says so plainly rather than producing a seasonal index that cannot
exist.

---

## 1. Three decisions that shape everything else

### 1.1 The publisher panel is declared before the trend is read

This is the decision the whole task turns on. Our postings do not arrive from
the employer; they arrive from job boards and aggregators, and the set of
boards in the feed changes month to month. In the Google data there are **96
publishers**, exactly **one** present in all twelve months, and **56 present
in a single month only**.

That makes a rising posting count ambiguous in a way that no amount of
smoothing fixes. When The Muse first appears in August with a 21-posting
backfill, the raw series shows Google's biggest hiring week of the year.
Nothing about Google's hiring changed; our field of view did.

This is the same-store-sales problem, and it has the same answer: compare like
with like. `src/trends.py` computes **three treatments of the same postings**
and reports them side by side.

| Treatment | What it counts | Reads |
| --- | --- | --- |
| `raw` | every posting from every publisher | total visible supply |
| `balanced` | only publishers present in ≥75% of periods | a stable panel, smaller and biased toward incumbents |
| `matched` | period-over-period change on publishers present in *both* periods | like-for-like change, no level |

`matched` comes in two forms. `chained` multiplies consecutive matched links,
which uses every publisher but accumulates **chain drift** — the errors
compound and the index can wander far from the level the raw data supports.
`bilateral` compares every period directly to the base period, which does not
drift but throws away any publisher absent from the base. Report both; where
they diverge, the divergence *is* the finding.

`panel_verdict()` refuses to name a direction when the treatments disagree.
For Google, December indexed to January = 100:

| raw | balanced | chained | bilateral |
| --- | --- | --- | --- |
| 91.1 | 115.4 | 185.4 | 191.2 |

One year, one dataset, four defensible answers spanning decline to +91%. The
honest output is `direction = growth, treatments_agree = False`, and the
report leads with that rather than picking the flattering number.

**Rule:** no volume trend goes into Task 06 or Task 07 without the panel
sensitivity table beside it.

### 1.2 Velocity is per observed day, not per calendar period

"Hiring velocity" cannot be a raw count. February is 10% shorter than March,
so a flat hiring rate shows a 10% February dip; and if collection starts
mid-month, the first period is short for a reason that has nothing to do with
hiring.

We report **postings per 7 observed days**:

```
postings_per_week = 7 * postings / days_observed
```

where `days_observed` is the overlap between the period and the actual
collection window, not the period's length. `is_partial` marks any period the
window does not fully cover.

There is a trap in the denominator. Google's ISO-week series opens with
`2022-W52`, which contains exactly one observed day (2023-01-01) carrying
seven postings. Normalised, that reads as **49 postings/week — the largest
value in the entire series**, and a naive reading has Google's hiring peaking
in the first week and collapsing. Any period observed for less than half its
length therefore reports **no rate at all**; the count is still published so
the gap stays visible.

### 1.3 Composition is reported next to level, always

A company can post more of something while that something becomes a smaller
part of what it posts. Google's Engineering postings rose 13.9% H1→H2 while
Engineering's *share* fell 8.9%. Both numbers are true and they answer
different questions: the first is about volume, the second about direction of
intent. `segment_trend_table` emits `count_change_pct` and `share_change_pct`
side by side and refuses to collapse them into one "growth" number.

---

## 2. The shared time-series structure

All four members aggregate on these keys, which Task 03 already wrote:

| Period | Column | Format | Notes |
| --- | --- | --- | --- |
| week | `posting_week` | `2023-W07` | **ISO** week |
| month | `posting_month` | `2023-07` | |
| quarter | `posting_quarter` | `2023-Q3` | calendar quarter |

ISO weeks are not calendar weeks. 2023-01-01 is a Sunday and ISO files it
under `2022-W52`, so a single calendar year yields **53 week buckets, two of
them partial**. Do not silently drop them and do not relabel them — the tests
pin this boundary.

`volume_series()` is the entry point and always returns one row per period in
the window, **including periods with zero postings**, so a collection gap
shows as a hole rather than vanishing from the index.

| Column | Meaning |
| --- | --- |
| `period` | the key above |
| `postings` | count |
| `n_publishers` | distinct publishers active in the period |
| `days_in_period` / `days_observed` | exposure |
| `is_partial` | window does not cover the period |
| `postings_per_week` | the velocity, blank if under half-observed |

`add_velocity()` adds `growth_pct`, `log_growth`, `rolling_mean`,
`acceleration` and `index_base_100`.

Two conventions worth stating because they are easy to get wrong:

- **`index_base_100` is rebased on the first *complete* period**, never on a
  truncated one.
- **Task 07 should model `log_growth`, not `growth_pct`.** Percentage changes
  are asymmetric — a −50% month followed by a +100% month is flat, but the
  percentages sum to +50. Google's month-to-month swings exceed 70% in both
  directions, so the asymmetry is not a rounding concern. Growth is also never
  padded across a blanked period; a gap stays a gap.

---

## 3. Spikes: detected robustly, then attributed

### 3.1 Detection

A mean-and-standard-deviation rule is useless here: the spike inflates both,
and one large batch hides the next. `robust_spikes()` uses a rolling **median
and MAD** over a 9-period window, flagging `|x − median| / (1.4826 · MAD)` above
3.5.

That has one failure mode which matters in practice. When a local window is
perfectly flat, MAD is 0 and every z-score is undefined — so the largest spike
in the series is silently dropped. The module falls back to a **Poisson scale**
(`√median`) whenever MAD is 0, on the reasoning that counts of independent
arrivals have variance about equal to their mean. Both z-scores are published.

### 3.2 Attribution — the part that changes the conclusion

Detecting a spike is not the deliverable; deciding whether it is *hiring* is.
`attribute_spikes()` compares the period's **excess over its local median**
against each publisher's **surplus over that publisher's own typical level**:

```
excess           = postings(period) − local_median
publisher_surplus = postings(publisher, period) − median(publisher, all periods)
excess_explained  = publisher_surplus / excess
```

A publisher that is simply large every week explains none of the excess. A
publisher that arrives with a backfill explains all of it. The first version of
this function compared raw shares instead, and called Recruit.net's 13-posting
single-day dump `broad_based` at a 0.43 share — the wrong verdict, from a
denominator that ignored what the publisher normally does.

All three of Google's flagged weeks come back `publisher_batch`:

| Week | Postings | Excess | Publisher | Excess explained | Largest single day |
| --- | --- | --- | --- | --- | --- |
| 2023-W24 | 24 | 9 | via Google Careers | 1.56 | 8 |
| 2023-W30 | 35 | 19 | via Recruit.net | 0.79 | 13 |
| 2023-W34 | 46 | 29 | via The Muse | 0.79 | 21 |

`batch_table()` generalises the check: same publisher, same day, ≥5 postings.
For Google that is 8 batches carrying **8.4% of all postings**.

**Rule:** do not report a spike as a hiring event until `attribute_spikes()`
returns `broad_based` for it.

---

## 4. Skill trends inherit four rules from Task 04

Task 04 §8 handed these down and Task 05 enforces them in code, not in prose.

1. **`share_of_skilled`, never `share_of_all`.** On the wrong denominator
   Python "fell" 23% between March and April 2023 while on the right one it
   rose 10%; the entire difference was that month's extraction coverage.
   `skill_velocity_table()` does not emit a `share_of_all` column at all — the
   wrong denominator should not be one column away from the right one, and a
   test asserts its absence.
2. **Exclude `job_function == "Facilities / Operations"`.** Data-centre
   facilities roles have 9.5% skill coverage because they genuinely have no
   software skills. Left in the denominator they drag every share down and
   manufacture a decline. The excluded set is named once, in
   `trends.SKILL_EXCLUDED_FUNCTIONS`, so all four of us exclude the same rows.
3. **Stratify or weight by publisher.** See §1.1.
4. **Re-check every trend within `job_function`.** See §5.

---

## 5. Simpson's paradox is the default assumption, not an edge case

A skill's pooled share can move in the opposite direction to its share inside
every single group that makes up the pool. This is not a curiosity here; it
happened to the headline finding of Task 04.

Task 04 flagged **Looker** as emerging: 9.2% → 13.8% of skilled postings,
+50%. Stratified by job function, inside every function that supports the
comparison it is flat or falling:

| Job function | H1 | H2 |
| --- | --- | --- |
| Analytics | 13.6% | 11.6% |
| Sales | 83.3% | 77.3% |
| Technical Sales | 25.0% | 24.0% |

The pooled rise is pure mix. Sales grew from 2.3% to 6.5% of skilled postings
and Analytics from 17.2% to 25.6%, while Looker-light Science/Research fell
from 29.7% to 23.2%. Google is not adopting Looker; Google is posting more
Looker-shaped jobs.

`stratified_verdict()` returns one of five labels per skill:

| Verdict | Meaning |
| --- | --- |
| `rising_in_all_segments` | real, not a mix effect |
| `falling_in_all_segments` | real decline |
| `mix_dependent` | segments disagree; the pooled number picked a side |
| `insufficient_support` | fewer than 2 segments clear the support floor |
| `flat_in_all_segments` | no movement to explain |

For Google, of 29 skills with enough support: 3 rise everywhere (SQL, R,
Machine Learning), 5 fall everywhere (Looker, Go, Java, Scala, JavaScript),
7 are mix-dependent and 14 lack support in more than one segment. Eight skills
have a pooled direction that stratification overturns.

The same trap applies to volume, not just skills. `compare_panels()` recomputes
every segment trend on the balanced panel, for every breakdown:

| Dimension | Segments | Directions that survive |
| --- | --- | --- |
| `job_function` | 10 | 7 |
| `job_category` | 13 | 7 |
| `country` | 48 | **21** |

Geography is the least robust breakdown, and structurally so: publishers are
regional, so a country trend on the raw feed is substantially a statement
about which regional aggregator was connected that month. Read the failures
carefully, though — most are the balanced panel losing enough postings to
judge at all, not an outright sign flip. Unverifiable and wrong are different
findings and the table distinguishes them.

**Rule:** a skill or segment trend goes into Task 09's insight report only with
its stratified verdict attached.

---

## 6. Seasonality: say what is identifiable, and what is not

The brief asks for seasonal spikes. With one calendar year of data:

- **Day-of-week is identifiable.** 53 observations per level, and the pattern
  is real and useful: Google's postings peak on Friday (index 1.33 vs uniform)
  and bottom out at the weekend (0.75 / 0.74).
- **Month-of-year is not identifiable.** One year gives **one observation per
  calendar month and zero complete annual cycles**. Any "January effect" is
  perfectly confounded with the trend and with whatever the publisher panel
  did that month. There is no statistical procedure that separates them from
  this data.

`seasonality_table()` therefore carries an `identifiable` column and an
`observations_per_level` count, and reports `False` / `0` for every month row.
The monthly index is still published — it is a description of 2023, and Task
06 can compare it across companies — but it must never be called seasonality,
and Task 07 must not fit an annual seasonal term to it.

The day-of-week result also carries a warning for everyone. **21.3% of Google
postings fall at a weekend, against 28.6% under uniformity** — a real
publication date would be near zero, and 7 postings are dated New Year's Day.
`posting_date` is therefore an *aggregator first-seen* date, not the date the
employer published. Task 07 inherits this: the series is a discovery process
convolved with a hiring process, and its short-horizon dynamics belong partly
to the aggregators.

---

## 7. What gets committed

Aggregate only, same rule as Task 04 — counts, shares and indices carry no
posting prose, so they are the deliverable and they are committed. Row-level
data stays git-ignored. Task 05 writes no row-level output at all.

```
members/<member>-<company>/task-05-tables/     19 CSVs (§2, §3, §4, §5, §6)
members/<member>-<company>/task-05-figures/    8 PNGs
members/<member>-<company>/task-05-trend-report.json
```

The JSON report re-runs the standing Task 01 `personal_data_columns_present`
check over every table produced. It must stay empty.

---

## 8. Checklist for each specialist

1. `python src/build_trends.py --company <you>`.
2. **Read `panel-sensitivity.csv` before you write a single sentence about
   growth.** If `treatments_agree` is false, your headline is the disagreement,
   not the direction.
3. Check `publisher-panel.csv`. Report how many publishers you have, how many
   persist, and what share of your postings the stable panel covers.
4. Check `spike-attribution.csv`. Any spike that is not `broad_based` is your
   pipeline, not the market — say so next to the chart.
5. Check `is_partial` on your first and last periods and quote no growth rate
   that starts or ends on one.
6. Read `skill-stratified-verdicts.csv` against your own Task 04 emerging list.
   Every disagreement is a Task 04 finding that needs correcting in your Task
   05 report — do not leave the old claim standing.
7. Confirm `month_of_year_identifiable` in your JSON. If it is `false`, your
   report says seasonality is not measurable, and Task 07 gets told.
8. Label every trend claim **within-2023** (or within your own window). One
   year, one source.
9. Commit `task-05-tables/`, `task-05-figures/` and
   `task-05-trend-report.json`; leave `data/processed/` git-ignored.
10. `python -m pytest tests/ -q` must pass before you push.

---

## 9. What Tasks 06 and 07 inherit

- **Task 06 (Competitor Comparison):** compare companies on the **balanced
  panel and matched index**, not raw counts. Raw volume across companies is
  mostly a comparison of how many boards syndicate each employer. Use
  `share_of_skilled` for every skill comparison and carry the Facilities
  exclusion.
- **Task 07 (Demand Forecasting):** model `log_growth` on the balanced panel;
  fit no annual seasonal term (§6); exclude or dummy the three attributed
  batch weeks, which are measurement, not demand; and treat `posting_date` as
  a discovery date, which caps the credible forecast horizon well short of a
  year.
- **Task 09 (Insight Generation):** every trend sentence carries its panel
  treatment and its stratified verdict. "Looker is emerging at Google" is the
  kind of sentence this task exists to prevent.
