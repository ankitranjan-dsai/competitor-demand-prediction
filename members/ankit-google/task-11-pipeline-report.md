# Task 11 — Automated Pipeline (Google)

**Specialist:** Ankit Ranjan · **Company:** Google (Alphabet) · **Date:** 2026-09-03

Input: ten tasks of scripts that had only ever been run by hand, in an order
nobody had written down. Output: a **sixteen-stage registry** those scripts are
derived from, **13 rules over 42 checks** that run before anything executes,
**one correction** — a latent defect that had been in this repository since
Task 06 and would have silently reversed C4 across every downstream table — and
the first run this code has ever had on a machine that is not mine, which
failed on a Python version floor nobody had declared (§6).

- **Method rationale (team standard):** [`docs/task-11-automated-pipeline-methods.md`](../../docs/task-11-automated-pipeline-methods.md)
- **Code:** [`src/pipeline.py`](../../src/pipeline.py) · [`src/run_pipeline.py`](../../src/run_pipeline.py)
- **Tests:** [`tests/test_pipeline.py`](../../tests/test_pipeline.py) (94) — **746 passing** in the suite
- **Workflow:** [`.github/workflows/pipeline.yml`](../../.github/workflows/pipeline.yml)
- **Machine-readable report:** [`task-11-pipeline-report.json`](task-11-pipeline-report.json)
- **Tables:** [`task-11-tables/`](task-11-tables/) · **Figures:** [`task-11-figures/`](task-11-figures/)
- **What this task overturned:** [`docs/corrections.md`](../../docs/corrections.md) —
  [C10](../../docs/corrections.md#c10--fixing-the-seed-does-not-fix-the-input)

```bash
python src/run_pipeline.py             # ~2 s, executes nothing
python src/run_pipeline.py --plan
python src/run_pipeline.py --run       # ~36 s
```

**The pipeline is not the deliverable; what it found is.** Automating a
repository that already worked should be bookkeeping. It was not, because
"already worked" turned out to mean "worked in an order that existed only in
the order I happened to type things".

---

## 1. The run order, and the one edge that matters

```
collect → preprocess → features → validate-text → trends → validate-skills →
competitor-set → comparison → forecast → similarity → validate-forecast →
insights → validate-similarity → presentation → validate-insights → tests
```

Sixteen stages, derived from the registry rather than declared. Eleven run by
default; the five `validate-*` stages are optional because each imports scipy
or scikit-learn and the house rule keeps those out of core modules.

The edge worth staring at is `trends → competitor-set`, and it carries no data
at all. `competitor-set` does not read anything `trends` writes. It must run
after it anyway, because it **overwrites the feature frame `trends` reads** —
and if it runs first, Task 05's committed numbers rebuild as Task 06's.

---

## 2. The defect: one file, two writers, 846 or 848 depending on the order

`data/processed/google/google_features.parquet`:

| Writer | Rows | Why |
| --- | --- | --- |
| `src/build_features.py` | **848** | every posting matching the Google employer family at Task 02 |
| `src/build_competitor_set.py` | **846** | the same frame after the employer audit that raised C4 |

Both correct. Both committed, in different tables:
[`volume-by-month.csv`](task-05-tables/volume-by-month.csv) sums to 848 and
[`company-comparability.csv`](task-06-tables/company-comparability.csv) has
Google at 846. The register has said since C4 that both stand.

What nothing said was **which one each downstream task reads** — and that is
not a property of any task. It is a property of the run order, which lived
nowhere.

The first version of this registry ordered `competitor-set` before `trends`.
The pipeline ran green. Every stage returned zero. Every table rebuilt. Every
figure redrew. And Task 05's trend analysis had been recomputed on the
post-audit frame, so C4 was silently reversed through the whole downstream
repository — with no missing file, no malformed frame, no failed assertion, and
no number that looked wrong on its own.

That is the failure mode this task exists to close, and it is the one a shell
script would have shipped.

[`pipeline-contested-artefacts.csv`](task-11-tables/pipeline-contested-artefacts.csv)
finds **nine** such paths — five processed Google frames, four Task 06 tables —
with **41** stage-reads landing on one of them, and names the winner and each
reader's version:

| Path | Writers | Winner | `trends` sees | `similarity` sees |
| --- | --- | --- | --- | --- |
| `google_features.parquet` | `features`, `competitor-set` | `competitor-set` | `features` (848) | `competitor-set` (846) |

**0 unordered writers, 0 unpinned readers**, enforced by
`contested_writes_ordered` and `contested_reads_pinned`. Neither rule existed
when the mistake was made. Raised as
[C10](../../docs/corrections.md#c10--fixing-the-seed-does-not-fix-the-input),
against Task 08's claim that fixed seeds mean "the committed tables rebuild
bit-for-bit" — true of the resampling, silent about the frame.

The rule this replaced is worth recording. `outputs_unique` asserted that no
two stages write the same path. It passed, and it was false: `competitor-set`
declared the directory `data/processed/` while `features` declared a file
inside it, so the string comparison never noticed. Declaring the paths honestly
made it fail on nine cases where two writers is the *correct* design. A rule
that passes only because the declarations are vague is worse than no rule.

---

## 3. What a committed number means: three readings, not two

Task 10 bound eighteen slide numbers to resolvers and concluded the deck
"cannot go stale quietly". I spent a while trying to make that claim false,
because a resolver plainly makes a number *rebuildable* rather than *current*,
and this repository ships no scheduled build. It is not false, and the reason
is a test: `test_the_shipped_deck_quotes_the_repository_as_it_stands` compares
every one of the eighteen against the repository and fails on any gap. At
`24cb4dc` the deck was exactly right — 647 tests, 127 tables, 49 figures, all
three verified against that commit. The suite would not have let it be
otherwise.

It fired three times during this task, which is the correct behaviour and also
the limit of what it can tell you: *something* moved. It does not say which
number, how far, or whether the stale copy is the committed one or the one in
my working tree. `fact_drift` answers that by reading three states instead of
one:

| Column | Read from | Answers |
| --- | --- | --- |
| `submitted` | `deck-facts.csv` at `24cb4dc` | what the mentor saw |
| `committed` | `deck-facts.csv` at `HEAD` | is the file in this repository wrong today |
| `live` | the resolvers, now | what a rebuild would produce |

Five of the eighteen end this task differing from `submitted`, and every one is
this task's own doing — the tests I added, the tables and figures I committed,
the README row, the register entry. The other thirteen resolve from a finished
analysis table and cannot move unless someone re-runs the analysis that wrote
them. That is the split worth knowing: movement in the first group means work
happened; movement in the second means a published result changed.

Two details that took a rebuild to get right:

- **`committed` reads git, not the working tree.** Reading the file on disk
  made an uncommitted rebuild look committed — the single state the column
  exists to catch. During the bad-ordered run it reported `committed` = 848
  while git held 846.
- **The gate cannot pass in the run that rebuilds the deck.** The rebuilt deck
  is uncommitted at that moment, so the drift test correctly calls it behind.
  The steady state is a run that changes nothing.

---

## 4. What a schedule can find here: dependency drift, and nothing else

Four approved sources, **zero refreshable**:

| Source | Verdict | Falsifier |
| --- | --- | --- |
| `hf:lukebarousse/data_jobs` | `frozen` | the dataset publishes a 2024+ revision |
| `adzuna` | `blocked` | keys provisioned **and** a pull returns untruncated descriptions |
| `themuse` | `live_empty` | a Muse pull returns one Google posting |
| `google-careers` | `blocked` | none — the robots.txt rejection is legal, not technical |

So the schedule mode is **`verify`, not `refresh`**: no approved source can
return a posting this repository does not already hold, and a nightly
collection job would re-download a file it has. Every verdict carries the
observation that would overturn it, which is what keeps this a finding instead
of an excuse.

[`schedule-rationale.csv`](task-11-tables/schedule-rationale.csv) records the
two rejected cadences in writing — `daily` (weekly already covers dependency
drift; daily multiplies runs without widening what they find) and `monthly`
(a refresh cadence detects nothing when nothing is refreshable).

The legal position is unchanged and enforced rather than asserted: no new
source, no new field, no scheduled collection, credentials reported as present
or absent and never as a value.

---

## 5. The corrected run

| | |
| --- | --- |
| DAG lint | **42 checks over 16 stages, 0 violations** |
| Contested paths | 9 — **0 unordered, 0 unpinned** |
| Audit | 41 checks, **6 findings**, all standing tensions |
| Reproducibility | 209 artefacts — **192 identical, 9 volatile-only, 8 changed** |
| Suite | **746 passing** |
| Interpreter floor | Python **3.11** — every source parses, CI pins match |

The eight changed artefacts were all the Task 10 deck, which is §3. The nine
volatile-only moved a timestamp and no content — `stable_digest` strips the
registered clock keys, which is what makes "changed" mean something.

The six audit findings are the `source_volatile` document that gets committed
anyway, the "no source is refreshable" verdict, and the four source verdicts.
None is a defect; each is a tension worth reporting every run rather than
resolving by deleting the check.

---

## 6. The floor nobody declared

The workflow's first real run failed, and not on anything this task wrote.
`check` never reached the linter: `src/insights.py` would not parse on the
runner. Four lines from Task 09 put a double-quoted subscript inside a
double-quoted f-string —

```python
f"{_fmt(_cell(row, "spread"))} index points around it"
```

— which is legal from Python 3.12 and a `SyntaxError` on 3.11, the version the
workflow installs. My machine runs 3.12.7, so the four lines had been correct
every single time anyone looked at them.

I want to be precise about what was wrong here, because the obvious reading is
that I picked the wrong version in the workflow. The repository declares no
Python version anywhere — no `requires-python`, no `setup.py`, no note in the
README. The workflow was the first file in ten tasks to name one, and naming
one is what turned an unstated assumption into a testable claim. **The defect
was the silence, not the number.** Any teammate installing 3.11 would have hit
an unreadable `SyntaxError` on import, and nothing in the repository would have
told them why.

The fix is four alternated quote characters, which is not interesting. What is
worth keeping is that the floor now lives in one place — `PYTHON_FLOOR` in
[`src/pipeline.py`](../../src/pipeline.py) — with two rules over it:

- `interpreter_floor_scan` tokenises every `.py` file **without importing it**,
  so it answers on the interpreter that is too new as well as the one that is
  too old. On 3.12 it says the four lines will not run on 3.11 *before* the
  push, which is the only version of this warning that is any use.
- `ci_python_versions` reads the workflow's pins back and requires them to
  equal the declared floor, so the two cannot drift.

Both run in the suite, and `python src/run_pipeline.py` prints the verdict
beside the schedule mode.

This is not a correction, and I considered whether it should be. C1–C10 all
overturn a claim someone made. Task 09 never claimed which interpreters it ran
on — the register has nothing to correct, because nothing was ever asserted.
That is exactly why ten tasks of tests could not catch it: **a test can check a
claim, and this was an absence of one.** The only instrument that finds an
unstated environment assumption is a second environment, and this task is the
first time this repository has ever had one.

---

## 7. What I would tell the next specialist

**Automating a working repository is not bookkeeping.** The order in which
things run is a claim, and it is the one claim ten tasks of tests never made.
Every rule in this repo checks a number, a frame, a share or a threshold. None
of them could see that a correct script reading a correct file at the wrong
moment produces a correct-looking wrong repository.

Three things generalise:

1. **Ask who else writes the file you are about to read.** Nine paths here have
   two writers, and every one of them is deliberate. The defect was never the
   second writer; it was leaving the order implicit.
2. **A rule that has never failed may be checking nothing.** `outputs_unique`
   passed for five tasks because one declaration said `data/processed/` and
   another said a file inside it.
3. **A derived number needs a schedule, not just a resolver.** Otherwise it is
   current exactly once, on the day you build it.

---

## 8. Limitations

1. **The registry is transcribed from prose, and prose was already wrong once.**
   Nothing checks `reads`/`writes` against what a script actually opens at
   runtime. An undeclared read is invisible until it breaks.
2. **`--check` proves the graph, not the code.** It can tell you a stage cannot
   run out of order; it cannot tell you the stage computes the right number.
3. **Weekly `verify` can only find dependency drift**, because §4 says there is
   nothing else to find. Correct today; wrong the day a source refreshes.
4. **One member.** `competitor-set` builds the other five companies through the
   same code, but a second specialist's per-member stages are not in the
   registry. `--focus` is the seam where they go.
5. **`validate-text` reads a live feed and its output is committed anyway.**
   Declared `source_volatile`, kept out of the schedule, and reported by the
   audit every run. It is a real tension and it is not resolved.

---

## 9. Handover

Task 12 is the other optional extension — a fine-tuned skill extractor. Read
the brief before this section
([C7](../../docs/corrections.md#c7--task-08-is-company-similarity-scoring-not-visualisation-and-not-evaluation));
this is a prediction, not an instruction.

The one thing that would cost you a day: **a fine-tuned extractor writes
`google_skills_long.parquet`, which already has two writers.** Adding a third
without declaring its edges will not fail. It will produce a complete,
plausible, differently-skilled repository, exactly as §2 describes. Declare the
stage in [`src/pipeline.py`](../../src/pipeline.py) before you run it, and let
`contested_writes_ordered` tell you where it belongs.
