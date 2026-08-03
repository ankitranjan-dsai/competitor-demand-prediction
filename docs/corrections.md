# Corrections Register

**Team standard.** Each task is submitted to the CadetX portal when it is
finished, so a submitted deliverable is **never silently rewritten**. When a
later task disproves an earlier claim, the original wording stays in place,
marked, with a pointer to the entry here. This file is the single place to
learn what the repo used to assert and no longer does.

**Rule.** A correction is only recorded here once the contradicting evidence is
in a committed table. "I think that looks wrong" is a limitation, not a
correction — limitations belong in the task report's own limitations section.

| # | Corrected claim | Origin | Overturned by | Status |
| --- | --- | --- | --- | --- |
| [C1](#c1--raw-monthly-counts-are-not-a-hiring-velocity-signal) | Monthly posting counts are "already a usable hiring-velocity signal" | Task 02 §3 (Google) | Task 05 §1.1, §3 | ✅ corrected |
| [C2](#c2--five-of-task-04s-ten-headline-skill-movers-do-not-survive-stratification) | 10 headline emerging/declining skills, Looker among them | Task 04 §5 (Google) | Task 05 §5 | ✅ corrected |
| [C3](#c3--posting_date-is-an-aggregator-first-seen-date-not-a-publication-date) | `posting_date` is a posting date at "daily granularity" | Task 01 §5.2 schema, Task 02 §3, Task 03 §1.3 | Task 05 §6 | ✅ corrected |

---

## C1 — Raw monthly counts are not a hiring-velocity signal

**What Task 02 said.** That the 848 postings run "min 22 in Feb, max 102 in Aug
— already a usable hiring-velocity signal."

**What is actually true.** Both of those extremes are properties of the
collection, not of Google's hiring.

- **February's trough is a collection gap.** 22 postings arrive from **6
  publishers**, against 23 publishers in January and 21 in March
  (`task-05-tables/publisher-presence.csv`). The publishers that vanish in
  February come back in March. Nothing in the data distinguishes "Google paused
  hiring" from "the feed lost 17 boards for four weeks", and the second
  explanation fits the publisher counts exactly.
- **August's peak is one publisher's backfill.** 24 of August's 102 postings
  come from The Muse, in its **first month in the feed**, and 21 of those land
  on a single day, `2023-08-23` — the largest day of the year at 25 postings.

More generally, **all three** weekly spikes the robust detector flags are
single-publisher batches, not hiring events
(`task-05-tables/spike-attribution.csv`):

| Week | Postings | Excess over local median | Top publisher | Share of excess explained | Verdict |
| --- | --- | --- | --- | --- | --- |
| 2023-W24 | 24 | 9 | via Google Careers | 1.56 | `publisher_batch` |
| 2023-W30 | 35 | 19 | via Recruit.net | 0.79 | `publisher_batch` |
| 2023-W34 | 46 | 29 | via The Muse | 0.79 | `publisher_batch` |

**Consequence.** Velocity is only readable on a declared publisher panel. On
the raw series December indexes to 91.1 against January; on the balanced panel
115.4; chained 185.4; bilateral 191.2 — four defensible answers spanning −9% to
+91%, so the **direction of Google's 2023 volume is not identified**. Task 06
compares on the balanced panel or matched index only, and Task 07 dummies out
W24/W30/W34 and treats February as missing.

Evidence: `members/ankit-google/task-05-tables/` —
`publisher-presence.csv`, `publisher-panel.csv`, `panel-sensitivity.csv`,
`spikes-weekly.csv`, `spike-attribution.csv`, `publisher-batches.csv`.

---

## C2 — Five of Task 04's ten headline skill movers do not survive stratification

**What Task 04 said.** A table of five emerging and five declining skills on
pooled H1 → H2 `share_of_skilled`, with a partial within-`job_function` check
covering only SQL, Hadoop and Go. Task 04 correctly labelled the rest "a
candidate list, not a result, until Task 05 stratifies it." Task 05 stratified
it, and half the list did not hold.

| Skill | Task 04 called it | Task 05 verdict | Outcome |
| --- | --- | --- | --- |
| SQL | emerging | `rising_in_all_segments` (5/5) | ✅ confirmed |
| Go | declining | `falling_in_all_segments` (2/2) | ✅ confirmed |
| Java | declining | `falling_in_all_segments` (2/2) | ✅ confirmed |
| Scala | declining | `falling_in_all_segments` (2/2) | ✅ confirmed |
| **Looker** | **emerging** | **`falling_in_all_segments` (3/3)** | ❌ **reversed** |
| BigQuery | emerging | `mix_dependent` (2 up, 2 down) | ⚠️ unsupported |
| GCP | emerging | `insufficient_support` (1 segment) | ⚠️ unsupported |
| PyTorch | emerging | `insufficient_support` (1 segment) | ⚠️ unsupported |
| Hadoop | declining | `insufficient_support` (1 segment) | ⚠️ unsupported |
| NoSQL | declining | `insufficient_support` (1 segment) | ⚠️ unsupported |

**Looker is a textbook Simpson's paradox.** Pooled, its share of skilled
postings rises 9.2% → 13.8%. Inside every job function that supports the
comparison, it falls:

| job_function | H1 → H2 | Δ |
| --- | --- | --- |
| Analytics | 13.6% → 11.6% | −2.0 pp |
| Sales | 83.3% → 77.3% | −6.1 pp |
| Technical Sales | 25.0% → 24.0% | −1.0 pp |

The pooled rise is pure composition: Sales grows from 2.3% to 6.5% of skilled
postings and Analytics from 17.2% to 25.6%, while Science / Research — which
barely mentions Looker — falls from 29.7% to 23.2%. **Google was posting more
Looker-shaped jobs, not asking for Looker more often.**

The five ⚠️ rows are *not* claims that the trend is false. They mean no
within-segment evidence supports it: `mix_dependent` is a genuine split, and
`insufficient_support` means only one `job_function` cleared the 10-posting
floor, so the pooled move cannot be separated from the mix. Hadoop and Go were
both checked within Engineering in Task 04 and both moved the way Task 04 said;
Go clears two segments and survives, Hadoop clears one and does not.

**Consequence.** Every skill claim from Task 06 onward carries its stratified
verdict. `share_of_skilled` remains the denominator, `Facilities / Operations`
stays excluded, and no skill is reported as emerging or declining on a pooled
share alone.

Evidence: `members/ankit-google/task-05-tables/skill-stratified-verdicts.csv`
(29 supported skills; 8 pooled directions overturned) and
`skill-trend-within-function.csv`.

---

## C3 — `posting_date` is an aggregator first-seen date, not a publication date

**What the schema said.** Task 01 §5.2 lists `posting_date` as a required
shared-schema field; Task 02 §3 recorded it as "daily granularity, full 2023";
Task 03 §1.3 feeds it into Layer A as a structured input. All three read as
though the field records when the employer published the vacancy. It does not.

**What is actually true.** The field records when the aggregator first saw the
posting, and three signatures in the data show it:

1. **Weekends are under-represented.** 21.3% of postings fall on a Saturday or
   Sunday against 28.6% under a uniform calendar (Saturday index 0.75, Sunday
   0.74, Friday 1.33) — a crawler's working week, not an employer's.
2. **Seven postings are dated 1 January 2023**, the first day of the collection
   window. A backfill boundary, not a New Year hiring push.
3. **Batch days.** 25 postings share `2023-08-23` and 16 share `2023-07-30`
   (see [C1](#c1--raw-monthly-counts-are-not-a-hiring-velocity-signal)).
   Publication dates do not cluster like that; ingestion dates do.

**Consequence.** The day-of-week index in
`task-05-tables/seasonality.csv` is real and identifiable (53 observations per
level), but it describes **the aggregator's crawl schedule**, not Google's
posting behaviour, and must be labelled that way wherever it is shown. No
claim below weekly resolution is about the employer. Monthly is the shortest
safe frequency for Task 07, and lag between a vacancy opening and its first
appearance in the feed is unknown and unmodelled.

This is a property of the source, not of Google — **every specialist's
aggregator-sourced `posting_date` carries it.** Run `weekend_share()` on your
own company's data (`src/trends.py`) before quoting any date-level pattern.

---

## What every specialist should take from this

The three corrections are the same mistake in three costumes: **a number
computed over an unstable collection was read as a fact about a company.**
Before a claim leaves your report, check it against the collection:

1. Is the publisher set behind this number stable over the window? If not, the
   number describes the panel. → C1
2. Would this pooled share still move if the segment mix were held fixed? → C2
3. Does this date field mean what its name implies? → C3

`src/trends.py` has a function for each check — `publisher_panel_table` /
`panel_verdict`, `stratified_verdict`, and `weekend_share` / `seasonality_table`
— and `docs/task-05-trend-analysis-methods.md` documents how to run them.
