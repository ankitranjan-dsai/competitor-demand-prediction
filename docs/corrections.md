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
| [C4](#c4--googles-posting-count-is-846-not-848) | Google has **848** postings | Task 02 §1, Tasks 03–05 (Google) | Task 06 §1.1 | ✅ corrected |
| [C5](#c5--task-06s-h1-aggregate-counts-february-on-a-panel-that-does-not-exist-in-february) | H1→H2 relative shares computed on a 620-posting H1 panel | Task 06 §3 (all six companies) | Task 07 §1.4, §10 | ✅ corrected |
| [C6](#c6--neither-concept-skills-nor-rare-skills-dominate-a-similarity-score) | Concept skills "dominate every similarity score in Task 08"; a 1-posting skill "would dominate cosine similarity" | `docs/task-04-skill-taxonomy.md` §2.3, §6.1 | Task 08 §6 | ✅ corrected |

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

## C4 — Google's posting count is 846, not 848

**What Tasks 02–05 said.** 848 Google-family postings, quoted in every Google
report since collection: "848 postings, Jan–Dec 2023" (Task 02 §1), "848 rows ×
44 columns" (Task 03), "601 of 848 postings skilled" (Task 04), "848 postings
indexed each to January = 100" (Task 05).

**What is actually true.** Two of those 848 rows are not Google. Task 02
selected on a name pattern alone (`collect_google_jobs.GOOGLE_FAMILY_PATTERN`).
Task 03 *did* spot the reseller — it wrote the third-party marker list
(`preprocess.NON_ALPHABET_MARKERS`) and set `is_alphabet = False` on that row,
"so the exclusion is visible and reversible" — but the flag lives downstream of
the count, so every headline number stayed at 848. The second row nothing
caught at all. Task 06 needed one employer rule that works for six companies,
so `src/companies.py` applies include **and** exclude at selection and adds a
check for employer fields that are not employers; both rows drop out:

| Employer string | Why it is not Google | Audit reason |
| --- | --- | --- |
| `Geoambiente - Google Cloud Premier Partner` | a named reseller — a partner hiring for its own Google Cloud practice | `named_third_party` |
| `Customer Engineer, Machine Learning, Google Cloud - Doha` | not an employer at all: the source put a role title in `company_name` | `role_string` |

Both are visible in `task-06-tables/employer-matching-audit.csv`, which lists
all 119 distinct employer strings the shortlist matched and the decision taken
on each.

**Consequence — smaller than it looks, and worth stating precisely.** Both rows
are in the *denominator* of Google's counts, not in any skill or trend result
that survived Task 05, and neither is in the balanced panel. The only published
number that moves is the raw January base, which loses the Geoambiente row
(2023-01-04): 90 → 89 postings, so December's **raw index goes from 91.11 to
92.13**. Every other treatment is unchanged, because they are computed on
publishers that carry Google throughout:

| Treatment | Task 05 (848) | Task 06 (846) |
| --- | --- | --- |
| raw | 91.11 | **92.13** |
| balanced | 115.38 | 115.38 |
| chained | 185.38 | 185.38 |
| bilateral | 191.18 | 191.18 |

C1's finding is untouched: the treatments still span decline to +91%, and the
direction of Google's 2023 volume is still not identified.

**What this changes going forward.** Task 06 onward reports Google at **846**.
Tasks 02–05 keep their wording and their 848 — they were computed on 848 and
rewriting them would make the register a lie — and each carries a pointer here.
`data/raw/google/google_jobs_hf_backfill_2023.parquet` still holds all 848 rows;
Task 06 writes its own selection to `data/raw/task-06/` and does not touch it.

**The general lesson for the other three specialists.** A substring match is not
an employer resolver. "Microsoft" catches `Metasys Technologies`; "Meta" catches
`OpenAirlines`; every company on the shortlist has resellers, staffing agencies
and partner firms carrying its name. Add your company to `src/companies.py` with
both patterns, then **read your rows in the audit table** — mine had two wrong
in 848, which is 0.2% of the count and 100% of the trust in it.

Evidence: `members/ankit-google/task-06-tables/employer-matching-audit.csv`,
`competitor-set-manifest.csv`, `volume-panel-sensitivity.csv`, against
`members/ankit-google/task-05-tables/panel-sensitivity.csv`.

---

## C5 — Task 06's H1 aggregate counts February on a panel that does not exist in February

**What Task 06 said.** Two things, and they contradict each other.

It defined the common panel as the **7 publishers that carry all six companies
over the window**, measured that panel month by month, and reported the result
in its own words: "within any single month the number never exceeds 3, and in
February it is **0**" (methods §2). It then handed Task 07 the rule that
follows from that — "February is missing for the panel entirely — treat as
missing, not zero" (report §11).

And it computed its headline H1→H2 relative shares on a **620-posting H1
panel** ([`relative-share-by-half.csv`](../members/ankit-google/task-06-tables/relative-share-by-half.csv)),
which includes February.

**What is actually true.** **97 of those 620 postings are February's** — 15.6%
of the H1 base. They reach the aggregate because panel membership is decided
over the whole window: a publisher that carries all six companies *at some
point in 2023* keeps its postings in every month, including the month in which
no publisher carries all six. The postings are real; the panel they are
attributed to is not one that exists in February.

The month is also not a neutral 97. It is distributed nothing like the year:

| Company | Feb postings | Feb share of the month | Mean monthly panel share |
| --- | --- | --- | --- |
| **Meta** | **39** | **40.2%** | 24.4% |
| Google | 19 | 19.6% | 27.7% |
| Microsoft | 16 | 16.5% | 21.4% |
| Snowflake | 11 | 11.3% | 7.0% |
| NVIDIA | 9 | 9.3% | 9.2% |
| Databricks | 3 | 3.1% | 10.3% |

A month in which Meta is 40% of a panel that on the year averages 24% is
exactly the kind of month C1 warned about: it describes which boards were
reachable in February, not who was hiring.

**Recomputed with February excluded**, per Task 06's own stated rule. H1 falls
620 → 523 postings; H2 is untouched at 1,161:

| Company | H1 with Feb | H1 without | Task 06 published | Corrected | Sign |
| --- | --- | --- | --- | --- | --- |
| **Google** | 179 | 160 | **−4.84 pp** | **−6.56 pp** | unchanged |
| **Meta** | 152 | 113 | **+1.23 pp** | **+4.15 pp** | unchanged |
| Microsoft | 118 | 102 | +5.09 pp | +4.61 pp | unchanged |
| Snowflake | 69 | 58 | −7.77 pp | −7.73 pp | unchanged |
| Databricks | 52 | 49 | +2.89 pp | +1.91 pp | unchanged |
| NVIDIA | 50 | 41 | +3.40 pp | +3.62 pp | unchanged |

**Every sign survives.** This is a correction to magnitudes, not to
conclusions: Google's share of the shared pool still fell, NVIDIA's and
Microsoft's still rose, Snowflake's still collapsed. The claim Task 06 asked
Task 09 to carry forward — that NVIDIA gained share in every publisher that
carries all six — is untouched, because that verdict is computed publisher by
publisher, not on the pooled halves.

Two magnitudes move enough to matter. **Meta's rise more than triples**, from
+1.23 pp to +4.15 pp, because February was its strongest month on the panel and
excluding it lowers the H1 base it is measured against. **Google's decline
deepens** from −4.84 pp to −6.56 pp, for the mirror-image reason: February was
one of Google's weakest panel months, so removing it raises Google's H1 share
to 30.59%.

**Consequence.** The published Task 06 wording stays as submitted — it was
computed on 620 and rewriting it would make this register a lie — and both
Task 06 documents now carry a pointer here. Task 07 onward derives the February
gap from the panel rather than hardcoding it (`unobserved_periods()` in
`src/forecast.py`), so every series the forecasting layer builds has February
open, and the calendar-time index in `period_ordinal()` makes the Jan→Mar step
**two months wide** rather than one.

**The general lesson.** Task 06 was not careless: it identified the gap,
measured it, and wrote the correct rule for the next task. What it did not do
is apply that rule to its own aggregate — the gap was documented in prose and
absent from the code path. **A rule that lives only in a report is not a rule.**
Where a rule can be enforced by the function that builds the number, enforce it
there: `panel_share_series()` marks February `is_observed = False` and every
downstream consumer respects the flag, so no Task 07 aggregate can repeat this
by inattention.

Evidence: `members/ankit-google/task-07-tables/february-correction.csv` and
`panel-share-series.csv`, against
`members/ankit-google/task-06-tables/relative-share-by-half.csv` and
`common-panel-by-month.csv`.

---

## C6 — Neither concept skills nor rare skills dominate a similarity score

**What the Task 04 taxonomy said.** Two predictions about this task, both used
to justify a design decision at the time.

§2.3, on why concepts are split from tools: mixing them "makes the concept
dominate every similarity score in Task 08 while carrying almost no
discriminating information."

§6.1, on why the skill matrix drops skills below `min_postings=5`: "a column
that is 1 for a single posting is an identifier, not a feature, and it would
dominate cosine similarity in Task 08."

**What is actually true.** Neither group can dominate a cosine on share
vectors, and the reason is arithmetic rather than empirical. A skill's
contribution to the numerator is the **product of two shares**. A skill that
half of each company's postings ask for contributes about 0.25; a skill that
one posting in six hundred asks for contributes about 0.0000028 — a hundred
thousand times less — and it contributes that whether it occupies one column
or a hundred. Column *count* is not weight. The predictions treated a long
tail of columns as if it were a long tail of influence.

Measured on the real vocabulary of 127 skills across the six companies
([`numerator-contribution.csv`](../members/ankit-google/task-08-tables/numerator-contribution.csv),
15 pairs):

| Group | Skills | Share of the cosine numerator (min / mean / max) |
| --- | --- | --- |
| **Concepts** (`is_concept`) | 8 | 0.05% / **0.36%** / 0.98% |
| Skills in ≤ 1 posting | 8 | 0.00% / **0.00%** / 0.00% |
| Skills in ≤ 10 postings | 39 | 0.00% / **0.005%** / 0.019% |
| Top 5 skills of the pair | 5 | 70.2% / **80.5%** / 93.2% |

Five skills carry four fifths of every score. The 39 skills below ten postings
carry five thousandths of one percent between them, and the eight that appear
in a single posting carry **exactly zero** — each of them is present in only
one of the two companies, so its product term is 0 by construction.

The eight concept skills present here are Computer Vision, Data Modelling,
Deep Learning, ETL, Generative AI, LLM, Machine Learning and NLP. Removing all
eight and rescoring
([`concept-skill-removal.csv`](../members/ankit-google/task-08-tables/concept-skill-removal.csv))
leaves the ranking **identical** — rank correlation 1.0, not one pair moves a
single place — and moves the largest cosine by **0.0020**, on a scale where
the pairs span 0.4961 to 0.9174.

**Consequence.** None for Task 04's design. The concept/tool split earns its
place for the reason §2.3 gives second — "Machine Learning" and "PyTorch" are
not comparable evidence — and the `min_postings=5` floor still keeps the
row-level matrix honest. What changes is what Task 08 is allowed to claim it
has controlled for: **dropping rare skills is not a robustness check on a
similarity score**, because there was nothing there to be robust to. A report
that ran the sweep and announced the ranking survived it would be claiming
credit for arithmetic.

Task 08 publishes the sweep anyway
([`support-sensitivity.csv`](../members/ankit-google/task-08-tables/support-sensitivity.csv)),
labelled as a different question rather than as a robustness check: restricting
to the skills every company supports asks "do they agree about the skills
everyone measures", which has its own answer and its own ordering.

**The general lesson.** The two predictions were made about a metric that had
not been chosen yet, and they would have been **right for a set metric**.
Jaccard counts columns, so a hundred rare skills genuinely can swamp it — and
Jaccard is in this task's metric list, where its ranking correlates −0.04 with
cosine's. The error was not the reasoning; it was attaching the reasoning to
"similarity" in the abstract when the answer depends entirely on which metric
gets used. **Name the metric before predicting what will dominate it.**

Evidence: `members/ankit-google/task-08-tables/numerator-contribution.csv`,
`concept-skill-removal.csv`, `support-sensitivity.csv` and
`metric-concordance.csv`, against `docs/task-04-skill-taxonomy.md` §2.3, §6.1.

---

## What every specialist should take from this

The first three corrections are the same mistake in three costumes: **a number
computed over an unstable collection was read as a fact about a company.** The
fourth is its twin — **a number computed over an unchecked set of employers.**
Before a claim leaves your report, check it against the collection:

1. Is the publisher set behind this number stable over the window? If not, the
   number describes the panel. → C1
2. Would this pooled share still move if the segment mix were held fixed? → C2
3. Does this date field mean what its name implies? → C3
4. Is every employer string in your denominator actually your company? → C4
5. Does the claim name the metric it is about? A statement about "similarity"
   that would be true of Jaccard and false of cosine is not a finding. → C6

`src/trends.py` has a function for each of the first three —
`publisher_panel_table` / `panel_verdict`, `stratified_verdict`, and
`weekend_share` / `seasonality_table` — and `src/companies.py` has
`matching_audit` for the fourth. `docs/task-05-trend-analysis-methods.md` and
`docs/task-06-competitor-comparison-methods.md` document how to run them.
