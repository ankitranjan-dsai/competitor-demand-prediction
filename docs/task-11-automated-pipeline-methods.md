# Task 11 — Automated Pipeline (optional)

**Team standard.** The brief asks to "connect data collection, preprocessing,
skill extraction, trend analysis, and forecasting into one scheduled,
end-to-end pipeline", and to submit "documented pipeline design & automation
steps" plus "the final workflow in GitHub".

The obvious reading is a shell script with sixteen lines in it. That reading is
wrong here for a reason worth stating up front: **this repository already ran
end to end, every day, in someone's head.** Ten tasks of scripts were executed
in an order that lived in a developer's memory and in the order the files
happened to be edited. A shell script would have written that order down
without ever checking it, and the order was wrong in a way nobody could see —
see §4.

So this document describes a **pipeline as data**. Sixteen stages are frozen
records naming what each reads, what it writes, what must precede it and how
its output behaves over time. The order is *derived* from the records, thirteen
rules run over the graph before anything executes, and the interesting output
of this task is not that the pipeline runs — it is what the rules found when
they were pointed at ten tasks of accumulated practice.

- **Code:** [`src/pipeline.py`](../src/pipeline.py) · [`src/run_pipeline.py`](../src/run_pipeline.py)
- **Tests:** [`tests/test_pipeline.py`](../tests/test_pipeline.py) (94; 746 in the suite)
- **Workflow:** [`.github/workflows/pipeline.yml`](../.github/workflows/pipeline.yml)
- **Google findings:** [`members/ankit-google/task-11-pipeline-report.md`](../members/ankit-google/task-11-pipeline-report.md)
- **What this task overturned:** [`docs/corrections.md`](corrections.md) —
  [C10](corrections.md#c10--fixing-the-seed-does-not-fix-the-input)
  (Task 08 §2, "the committed tables rebuild bit-for-bit")
- **Inherited from:** every task's `## What gets committed` section — those
  lists are the `reads` and `writes` fields of §2, moved from prose into code
- **Legal position:** unchanged, and §6 is where that is enforced. No new
  source, no new field, no scheduled collection. The schedule **verifies**; it
  does not refresh, and §6 explains why that is a finding rather than a
  limitation.

```bash
python src/run_pipeline.py             # check mode: lints, audits, drift — executes nothing
python src/run_pipeline.py --plan      # the derived run order, and why each stage is in it
python src/run_pipeline.py --run       # execute all sixteen stages (~36 s)
```

Five findings in one line each, so the rest of the document reads as their
justification:

1. **One file, two writers, and the right answer depends on the order** —
   nine committed paths are written by two stages apiece, and nothing before
   this task recorded which write a reader was supposed to see (§4, C10).
2. **A committed number needs three readings, not two** — `submitted`,
   `committed` and `live` answer three different questions, and reading one
   for another is how a deck looks current while the file in the repository is
   not (§8).
3. **No approved source can return a posting this repository does not already
   hold** — four sources, zero refreshable, so the schedule verifies rather
   than refreshes (§6).
4. **A rebuild is not reproducible or not; it is reproducible modulo a clock** —
   9 of 209 artefacts move their bytes on every run and none of their content
   (§5).
5. **The pipeline's own gate cannot pass in the run that changes the deck** —
   and that is correct behaviour, not a bug (§8).

---

## 1. What counts as a stage

A stage is one command this repository already had. Task 11 adds no analysis,
no new table and no new number; every stage in the registry is a script written
for Tasks 02–10, and the only new executable is the orchestrator.

Sixteen stages, in four kinds:

| Kind | Count | What it does |
| --- | --- | --- |
| `collect` | 1 | brings the approved backfill onto disk |
| `transform` | 3 | preprocessing, feature extraction, the competitor set |
| `analyse` | 5 | trends, comparison, forecast, similarity, insights |
| `validate` | 5 | the scipy / scikit-learn / statsmodels cross-checks |
| `publish` | 1 | the deck |
| `gate` | 1 | the test suite |

The five `validate` stages are the ones a shell script would have dropped.
They are marked `optional` and excluded from the default run, because each
imports scipy or scikit-learn and the house rule is that core modules do not —
but they are **in the registry**, so `--include-optional` runs them and the
DAG rules apply to them whether or not they run. A stage left out of the graph
is a stage whose ordering constraints are invisible, which is §4's entire
subject.

---

## 2. The registry: sixteen frozen records

```python
@dataclass(frozen=True)
class Stage:
    name: str
    task: int
    kind: str
    command: tuple[str, ...]
    requires: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    volatility: str = "deterministic"
    ...
```

Nothing here is a comment. `reads` and `writes` are paths the rules resolve
against the working tree, `volatility` is the vocabulary of §5, and `requires`
and `after` are the two kinds of edge §3 distinguishes. The declarations came
out of each task's own `## What gets committed` section, which is why the
transcription was worth doing carefully: those sections are prose, prose is not
checked, and §4 is what the difference cost.

[`pipeline-stages.csv`](../members/ankit-google/task-11-tables/pipeline-stages.csv)
is the registry rendered as a table — the deliverable the brief calls a
"documented pipeline design", generated from the thing that runs rather than
written alongside it.

---

## 3. Ordering: `requires`, `after`, and ancestry

Two fields, because there are two reasons one stage must precede another and
collapsing them loses a case the repository actually has.

- **`requires`** is a data edge: *I need what this produced.* Pulling a stage
  into the run pulls its `requires` in with it.
- **`after`** is a sequencing edge: *if this runs, it runs before me.* It
  constrains the order without forcing the stage into the run.

The distinction exists because an **optional** stage can hold an ordering
constraint. `validate-skills` reads the feature frame that `competitor-set`
overwrites, so it must precede it — but the default run skips `validate-skills`
entirely, and making that a `requires` would mean the default run depends on a
stage the default run does not execute. `after` says the true thing.

Ordering is by **ancestry, not parentage**: an edge is guaranteed by any
transitive ancestor, so the registry stays a transitive reduction and declares
each constraint once. `topological_order` treats both fields identically —
a sequencing edge constrains the order exactly as hard as a data edge — and
only `plan_run`'s `--only` expansion distinguishes them, in one direction.

[`pipeline-dag-edges.csv`](../members/ankit-google/task-11-tables/pipeline-dag-edges.csv)
carries all 22 edges with the artefact that justifies each. **Eight carry no
artefact at all.** Naming those separately stops them being read as data
dependencies, and stops the reverse and more expensive mistake — deleting one
because "nothing flows along it".

---

## 4. Contested artefacts, and the defect this task found

**This is the finding.** Everything above exists because of it.

`data/processed/google/google_features.parquet` is written by two scripts:

| Writer | Rows | Why |
| --- | --- | --- |
| `src/build_features.py` | 848 | every posting whose employer string matched the Google family at Task 02 |
| `src/build_competitor_set.py` | 846 | the same frame after the employer audit that raised [C4](corrections.md#c4--googles-posting-count-is-846-not-848) |

Both are correct. Task 05 reports 848 and is right; Task 06 onward reports 846
and is right. The repository has been consistent about this since C4 was
raised — **on the condition** that `trends` reads the file before
`competitor-set` overwrites it, and `comparison`, `forecast`, `similarity` and
`insights` read it after.

That condition was written down nowhere. Not in a Makefile, not in a README,
not in an import. It held because of the order the scripts happened to be run
in, and the first version of this registry got it backwards: `competitor-set`
was ordered before `trends`, the pipeline ran green, every stage succeeded, and
the rebuild silently reversed C4 through the whole downstream repository. No
frame was missing. No assertion fired. Every number was complete, plausible and
from the wrong side of an audit.

A **contested artefact** is a path with more than one writer. Its content is a
property of the run order, not of any one stage.
[`pipeline-contested-artefacts.csv`](../members/ankit-google/task-11-tables/pipeline-contested-artefacts.csv)
finds nine of them — five processed Google frames, four Task 06 tables — with
41 stage-reads landing on one. For each it names the winner and, per reader,
which writer's version that reader actually sees:

| Path | Writers | `trends` sees | `similarity` sees |
| --- | --- | --- | --- |
| `google_features.parquet` | `features`, `competitor-set` | `features` (848) | `competitor-set` (846) |

Two rules make it a constraint rather than an observation:

- **`contested_writes_ordered`** — every pair of writers of one path is
  ancestrally ordered, so exactly one wins by declaration rather than by luck.
- **`contested_reads_pinned`** — every reader sits on one side or the other of
  every writer, so no stage can land in an order-dependent position.

A stage that reads a path it also writes reads the version that was there when
it started, so its own write is never ambiguous to it; `comparison` does
exactly this with the feasibility screen, and the rule excludes the reader from
its own writer set rather than reporting a phantom.

**What this replaced.** The registry's first rule was `outputs_unique`: no two
stages write the same path. It passed, and it was false. It passed because
`competitor-set` declared the directory `data/processed/` while `features`
declared a file inside it, so a string comparison never noticed. Declaring the
contested paths honestly made it fail on nine deliberate ones — nine paths
where two writers is the **correct** design, because one frame before the
employer audit and one after is what C4 requires. The rule was asserting
something this repository has never satisfied and should not. It is gone.

---

## 5. Volatility: what "changed" means

A rebuild that reports "12 files changed" when nine of them differ only in a
timestamp has trained you to ignore it. Every stage declares how its output
behaves:

| Volatility | Stages | Meaning |
| --- | --- | --- |
| `deterministic` | same bytes every run | a diff is a real change |
| `clock_stamped` | writes a report carrying the wall clock | bytes move, content need not |
| `source_volatile` | reads a live feed | content legitimately differs run to run |

`stable_digest` strips the registered clock keys before hashing, which
separates *did the findings change* from *did the clock move*. On the run
recorded in
[`reproducibility-audit.csv`](../members/ankit-google/task-11-tables/reproducibility-audit.csv):
**209 artefacts, 192 byte-identical, 9 volatile-only, 8 changed** — and all
eight were the Task 10 deck, rebuilt because this task moved five of the
eighteen facts it quotes (§8).

One stage is `source_volatile`: `validate-text` exercises the Layer B cleaner
against live posting HTML. It is kept out of the schedule, and the audit
reports — as a standing finding, not a failure — that its output is committed
anyway. That is a real tension in the repository, and the honest thing is to
name it every run rather than resolve it by deleting the check.

---

## 6. Sources: what a schedule could actually find

A pipeline that collects on a schedule needs a source that returns something
new. [`source-refreshability.csv`](../members/ankit-google/task-11-tables/source-refreshability.csv)
asks that of all four approved sources, and each verdict carries a
**falsifier** — the observation that would change it:

| Source | Verdict | Why | Falsifier |
| --- | --- | --- | --- |
| `hf:lukebarousse/data_jobs` | `frozen` | a published 2023 snapshot; re-downloading returns the same 785k rows | the dataset publishes a revision covering 2024+ |
| `adzuna` | `blocked` | no keys in the environment, and `description` truncates to 500 chars regardless | credentials provisioned **and** a pull returns untruncated descriptions |
| `themuse` | `live_empty` | live and reachable, lists no Google postings | a Muse pull returns one |
| `google-careers` | `blocked` | rejected on robots.txt at Task 01 | none this project would act on — the rejection is legal, not technical |

**Zero of four are refreshable**, so a nightly collection job would spend
compute to re-download a file it already has. The pipeline reports this as its
schedule mode: `verify`, not `refresh`. Naming the falsifiers is what keeps
that a finding rather than an excuse — the day Adzuna keys arrive, the verdict
flips on evidence and the cadence question reopens on its own.

[`schedule-rationale.csv`](../members/ankit-google/task-11-tables/schedule-rationale.csv)
carries the same argument over cadences. Three adopted, two rejected in
writing: `daily` because weekly already covers dependency drift and daily
multiplies runs without widening what they can find; `monthly` because a
refresh cadence detects nothing when nothing is refreshable.

---

## 7. The audit: 41 checks the pipeline runs on itself

`pipeline_audit` checks inputs, outputs, sources, volatility, secrets and
privacy — 41 checks, of which **6 report a finding**, and all six are the
standing tensions named above rather than defects: the `source_volatile`
committed document, the "no source is refreshable" verdict, and the four source
verdicts themselves.

The privacy area re-runs the standing check every task carries: no
`personal_data_columns_present` in any committed table. The secrets area
reports credential **presence only** — never a value, never a partial value —
which is Task 01's rule applied to a report that gets committed.

One check was removed during this task. `env_present_locally` asked whether
`.env` existed on the machine running the audit, which is machine-dependent, so
the committed table could never match a CI rebuild — an audit row that is
guaranteed to disagree with itself teaches you to ignore the table. Credential
state moved to the JSON report, where it is a fact about a run rather than a
claim about the repository.

---

## 8. Drift: what a committed number means

Task 10 bound eighteen numbers on its slides to resolvers so that no numeral
was typed by hand, and §4 of its methods doc concluded that the deck "cannot go
stale quietly". That claim held, and it is worth saying why, because the
obvious objection is that a resolver makes a number *rebuildable* and not
*current*. It does — but Task 10 also shipped
`test_the_shipped_deck_quotes_the_repository_as_it_stands`, which fails the
moment any of the eighteen differs from what the deck says. It fired three
times during this task: once for the tables and figures Task 11 committed, once
for the tests it added, once for the register entry it wrote. The deck was
current at `24cb4dc` and could not go stale without saying so.

What that test cannot say is **which** number moved, how far, or whether the
committed file or the working tree is the stale one — it compares the deck to
the repository now and reports a mismatch. `fact_drift` compares three states
instead, and the three columns are the whole point:

| Column | Read from | Answers |
| --- | --- | --- |
| `submitted` | `deck-facts.csv` at `24cb4dc` | what the mentor saw |
| `committed` | `deck-facts.csv` at `HEAD` | **is the file in this repository wrong today** |
| `live` | the resolvers, now | what a rebuild would produce |

`committed` reads **git**, not the working tree. Reading the file on disk made
an uncommitted rebuild look committed, which is the one state the column exists
to catch. A shallow clone falls back to the working tree and says so, rather
than quietly answering a different question.

Five of the eighteen end this task differing from `submitted`, and all five are
Task 11's own doing: `tests_total`, `tables_committed`, `figures_committed`,
`tasks_total` and `corrections_total`. The other thirteen resolve from a
finished analysis table and cannot move unless someone re-runs the analysis
that wrote them. That split is the useful half of the table — a moved value in
the first group means work happened, and a moved value in the second means a
committed result changed, which is a different and much louder event.

**The fixpoint.** This task's build settles within a run, but the repository
keeps moving after it: writing [C10](corrections.md#c10--fixing-the-seed-does-not-fix-the-input)
moved `corrections_total`, which staled the deck, which the drift table then
reported. That is not a paradox — it is an ordering. Write the register,
rebuild the deck, re-run the check, and the three columns agree.

It has one consequence worth stating plainly, because it looks like a bug:
**the `tests` gate cannot pass in the same `--run` that rebuilds the deck.**
The rebuilt deck is uncommitted at that moment, so the drift test correctly
reports the deck as behind. The steady state — the state the weekly job checks —
is a run that changes nothing and passes. A pipeline whose gate went green
while an uncommitted rebuild sat in the working tree would be a pipeline
checking the wrong thing.

---

## 9. Automation: two jobs, neither of which commits

[`.github/workflows/pipeline.yml`](../.github/workflows/pipeline.yml):

- **`check`** — every push and pull request. Lints the DAG, runs the suite,
  and `git diff --exit-code` over `task-11-tables/` to prove the committed
  tables match a rebuild. Uploads the tables and the JSON report as artifacts.
- **`verify`** — Mondays 06:00 UTC and on `workflow_dispatch`. Runs
  `--run --scheduled` and writes the outcome to `$GITHUB_STEP_SUMMARY`.

Two deliberate limits. The diff is scoped to **tables only**: PNG bytes are not
reproducible across matplotlib and freetype versions, so diffing figures would
fail for a reason that has nothing to do with this repository's numbers.
And **nothing commits.** A workflow with write access that pushes a rebuild is
a workflow that can quietly replace a submitted table — the exact move
[`docs/corrections.md`](corrections.md) exists to prevent. CI reports; a human
commits.

`PYTHONHASHSEED: "0"` and `fetch-depth: 0`, the latter because §8's `committed`
column needs history a shallow clone does not have.

**What the first CI run found, and it was not the DAG.** `check` failed before
it linted anything: `src/insights.py` would not parse. Four lines written in
Task 09 reuse the enclosing quote inside an f-string replacement field —
`f"{_cell(row, "spread")}"` — which [PEP 701] legalised in Python 3.12 and
which is a `SyntaxError` on the 3.11 the workflow installs. Every local run had
passed, every test had passed, and the suite had grown past seven hundred tests
without one of them noticing, because the machine the code was written on
satisfies a floor nobody had written down.

That is the general shape of the thing: **a version floor is invisible to every
run that meets it.** It is not found by running the code more times; it is
found by running it somewhere else, and until this task there was nowhere else.
The four lines are fixed by alternating the quotes, and the floor is now
declared in one place, `pipeline.PYTHON_FLOOR`, with two rules over it —
`interpreter_floor_scan` tokenises every source *without importing it*, so a
developer on 3.12 is told the code will not run on 3.11 before pushing rather
than after, and `ci_python_versions` keeps the workflow's two pins equal to the
declared floor. `python src/run_pipeline.py` prints the verdict next to the
schedule mode.

This is not a register entry. C1–C10 correct claims, and Task 09 never claimed
which interpreters it ran on — the defect is that nothing did.

[PEP 701]: https://peps.python.org/pep-0701/

---

## 10. What gets committed

**Committed** — [`members/<member>-<company>/task-11-tables/`](../members/ankit-google/task-11-tables/)
(11 CSVs: stages, dag-edges, contested-artefacts, lint, audit,
source-refreshability, schedule-rationale, failure-modes, deck-fact-drift,
run-ledger, reproducibility-audit), `task-11-figures/` (5 PNGs),
`task-11-pipeline-report.json`, and the workflow.

**Not committed** — nothing new. This task reads what the repository holds and
writes tables about it.

---

## 11. Limitations

1. **The registry is a transcription, and a transcription can be wrong.** §4 is
   the proof: `reads` and `writes` came from prose, and prose was missing an
   edge. The rules check the registry for internal consistency; nothing checks
   the registry against what a script *actually* opens at runtime. A stage that
   reads an undeclared path is invisible until it breaks.
2. **`--check` proves the graph, not the code.** Thirteen rules over sixteen
   records — 42 checks in all — take two seconds and execute nothing. They cannot tell you a stage
   produces a wrong number, only that it cannot produce one out of order.
3. **The schedule verifies against a frozen source.** Weekly `verify` can find
   dependency drift and nothing else, because §6 says there is nothing else to
   find. That is the correct cadence for this repository today and would be the
   wrong one the day a source becomes refreshable.
4. **One member.** The registry declares Google's stages. The competitor-set
   stage builds the other five companies through the same code, but a second
   specialist's per-member stages are not represented, and `--focus` is the
   seam where they would go.

---

## 12. Handover

Task 12 is the other optional extension — fine-tuning a skill-extraction model
(BERT/RoBERTa/spaCy). Read the brief before this section, per
[C7](corrections.md#c7--task-08-is-company-similarity-scoring-not-visualisation-and-not-evaluation);
what follows is a prediction, not an instruction.

What this task hands over is one constraint and one tool.

**The constraint.** A fine-tuned extractor writes
`data/processed/google/google_skills_long.parquet` — which §4 lists as a
contested path, with two writers already. Adding a third writer without adding
its edges will not fail; it will produce a complete, plausible, differently
skilled repository. Declare the stage in
[`src/pipeline.py`](../src/pipeline.py) **before** running it, and let
`contested_writes_ordered` tell you where it goes.

**The tool.** `python src/run_pipeline.py --plan` prints the derived order and
why each stage is in it, in two seconds, without executing anything. Whatever
Task 12 adds, that is the cheapest way to find out what it perturbs.

One thing not to inherit: Task 04's taxonomy is rules, not a model, and
[C6](corrections.md#c6--neither-concept-skills-nor-rare-skills-dominate-a-similarity-score)
already established that column count is not weight in a prevalence-weighted
metric. A model that extracts more skills is not thereby a better input to
Task 08.
