# Task 10 — Final Presentation & Mentor Review (Google)

**Specialist:** Ankit Ranjan · **Company:** Google (Alphabet) · **Date:** 2026-08-30

Input: the claim ledger Task 09 committed, and the repository itself. Output: a
**19-slide deck compiled from that ledger**, a **21-question mentor bank** whose
answers are ledger rows rather than improvisation, and an **audit of the
workspace** that found four defects and drove the rest of this task. 11 tables,
8 figures, 83 tests, and one correction against Task 09.

- **Method rationale (team standard):** [`docs/task-10-final-presentation-methods.md`](../../docs/task-10-final-presentation-methods.md)
- **Code:** [`src/present.py`](../../src/present.py) · [`src/build_presentation.py`](../../src/build_presentation.py)
- **Tests:** [`tests/test_presentation.py`](../../tests/test_presentation.py) (83) — **647 passing** in the suite
- **The deck:** [`task-10-slides.md`](task-10-slides.md)
- **Machine-readable report:** [`task-10-presentation-report.json`](task-10-presentation-report.json)
- **Tables:** [`task-10-tables/`](task-10-tables/) · **Figures:** [`task-10-figures/`](task-10-figures/)
- **Notebooks:** [`notebooks/01-google-evidence-walk.ipynb`](../../notebooks/01-google-evidence-walk.ipynb) · [`notebooks/02-the-refusal-boundary.ipynb`](../../notebooks/02-the-refusal-boundary.ipynb)
- **What this task overturned:** [`docs/corrections.md`](../../docs/corrections.md) — [C9](../../docs/corrections.md#c9--a-deck-is-checkable-the-delivery-is-what-is-not)

```bash
python src/build_presentation.py
python -m pytest tests/ -q
```

**No sentence on a slide was typed twice.** A bullet either points at a row in
[`claim-ledger.csv`](task-09-tables/claim-ledger.csv) and is rendered from it,
or resolves a key in the fact register, or contains no numeral at all. The
build fails on any bullet that does none of the three.

---

## 1. The deck, in one table

19 slides, 61 bullets, 0 violations.

| # | Section | Title | Bullets | Provenance |
| --- | --- | --- | --- | --- |
| 1 | title | Competitor Demand Prediction Using Job Postings | 3 | 2 facts, 1 narration |
| 2 | problem | Job postings are a forward-looking signal | 3 | 3 narration |
| 3 | **legal** | The legal gate came first, and it removed sources | 4 | 4 narration |
| 4 | data | What was collected, and what failed the screen | 3 | 2 facts, 1 narration |
| 5 | pipeline | Rules, not models, so every extraction is inspectable | 3 | 1 fact, 2 narration |
| 6 | **identification** | A posting count is a count of the boards that syndicate it | 4 | 1 claim, 3 narration |
| 7 | trends | Within-year trends: the mix moves, the volume does not resolve | 3 | 2 claims, 1 narration |
| 8 | comparison | Share of a common publisher pool, not headcount | 3 | 2 claims, 1 narration |
| 9 | forecast | The forecast question has an answer, and it is no | 3 | 1 claim, 2 narration |
| 10 | similarity | One pair of six companies is actually separated | 3 | 1 claim, 2 narration |
| 11 | insights | Sentences were generated, then gated — not written | 3 | 1 fact, 2 narration |
| 12 | position | Position is a profile, not a ranking | 3 | 1 fact, 2 narration |
| 13 | stack | A stack difference is not a capability difference | 3 | 2 claims, 1 narration |
| 14 | **refusals** | What this project will not say | 5 | 4 refusals, 1 narration |
| 15 | **corrections** | Corrections are recorded, never overwritten | 3 | 1 fact, 2 narration |
| 16 | **quality** | Every number here is bound to a committed table | 3 | 1 fact, 2 narration |
| 17 | workspace | The repository the next specialists inherit | 3 | 1 fact, 2 narration |
| 18 | next | What would change the answers | 4 | 4 narration |
| 19 | qa | Backup — the question bank | 2 | 2 narration |

The five bold sections are mandatory: `legal`, `identification`, `refusals`,
`corrections`, `quality`. If the session runs short, the slides that go are
from the other fourteen.

Every slide carries speaker notes, and the notes are checked too
(`notes_present`). Slide 6's read: *"This is the load-bearing slide. If it
lands, everything downstream follows."*

---

## 2. Where the 61 bullets come from

| Kind | Count | Text written by |
| --- | --- | --- |
| `plain` | 38 | this task — and forbidden to contain a numeral |
| `fact` | 10 | a template over the fact register |
| `claim` | 9 | Task 09, verbatim from a published ledger row |
| `refusal` | 4 | Task 09, verbatim from a refused ledger row |

13 of 436 ledger rows reach the deck; 17 more are spent in the question bank.
That is a low draw on purpose. A deck is not a report, and the ledger is there
so the presenter can go and get the other 406 when asked, not so all of them
can be shown.

The 38 `plain` bullets are the ones worth being suspicious of, and they are the
reason `plain_numerals_bound` exists: narration is where a number goes when it
is typed from memory rather than read from a table. Here, narration carries no
numbers at all.

---

## 3. The three slides that carry the argument

**Slide 6 — identification.** A posting count is a count of the boards that
syndicate a posting. Every refusal in this project traces back to this one
sentence, and it is the only slide whose failure invalidates the rest.

**Slide 14 — refusals.** Four verbatim refused claims, each with the gate that
stopped it and the observation that would lift it:

| Refused claim | Ledger row | Stopped at |
| --- | --- | --- |
| the 2024 share forecast | `tempting-forecast` | lint → task-07 §8, no model beats persistence |
| the level comparison against Snowflake | `tempting-level` | lint → task-06 §1.3, a posting count counts boards |
| the pay gap against Meta in research roles | `salary-google-meta-science-research` | identification — 4.02% disclosure |
| the shift towards Singapore | `tempting-country` | lint → task-05 §9, publishers are regional |

The wording-gate refusals name the construct that has no table; the salary
refusal names its falsifier — *the same sign on a publisher-balanced sample of
disclosed salaries*. The fifth bullet on the slide is narration, and it is the
one that stings: there is no sentence here for an investor.

`refusals_intact` holds at least four in place. Without that rule this is the
slide that vanishes when the deck is trimmed to fit.

**Slide 16 — quality.** How any of it is checked: the suite, the corrections
register, and the fact that the deck's own numbers are resolved from committed
tables. Its last bullet is C9 — the correction this task raises against itself.

---

## 4. The mentor bank: eleven of twenty-one answers are "no"

21 questions, 48 answer lines. Each line is the same four kinds as a slide
bullet, so an answer cannot be improvised into something the ledger does not
support.

| Verdict | Questions | Example |
| --- | --- | --- |
| refused | 11 | *So can you forecast their hiring for next quarter?* |
| answered | 6 | *Has any of your own headlines been wrong?* |
| explained | 4 | *Did you scrape any of this?* |

A refused answer is not a shrug. It opens with the refused claim verbatim, names
the gate, and ends with the falsifier — what would have to be observed for the
answer to become yes.

Nine of the eleven refusals correspond to a standing prohibition, and
`qa_covers_required` fails the build if any is left unanswered:

| Prohibition | Source | Answered by |
| --- | --- | --- |
| forecast | task-07 §8 | `tempting-forecast` |
| convergence | task-08 §8 | `tempting-convergence` |
| cross-company level | task-06 §1.3 | `tempting-level` |
| product launch | task-09 §5 | `tempting-product` |
| unmeasured construct | task-02 scope | `tempting-headcount` |
| share of all postings | task-04 §7 | `tempting-share-of-all` |
| seasonality | task-05 §6 | `tempting-seasonal` |
| causal strategy | task-05 §1 | `tempting-strategy` |
| country split | task-05 §9 | `tempting-country` |

`prohibitions_open` is empty.

---

## 5. The audit turned on the repository, and found four things

37 checks over five areas in the shipped build
([`workspace-audit.csv`](task-10-tables/workspace-audit.csv)). The audit's size
follows the repository — one check per task row, one per path that row quotes,
one per working directory — so it stood at 35 when it first ran, before the
Task 10 row had deliverables to name. Four checks failed then, and none of them
was in the analysis:

| Check | Subject | What it found |
| --- | --- | --- |
| `suite_size_current` | task 09 | quoted **557 tests**; the suite collected **646** |
| `working_dir_used` | `notebooks/` | 0 files beyond the placeholder |
| `working_dir_used` | `weekly-reports/` | 0 files beyond the placeholder |
| `working_dir_used` | `meeting-minutes/` | 0 files beyond the placeholder |

The test count had been true when Task 09 shipped and false ever since — the
exact drift pattern the corrections register exists for, except that no analysis
re-derived it, so nothing caught it until a check did.

The three empty directories are worse, and worth being blunt about in a report
that will be read by whoever inherits this repo: they had been in the layout
diagram since Task 01. A reader following the README would have found three
promises and a placeholder file.

**What the fix was:**

- [`notebooks/01-google-evidence-walk.ipynb`](../../notebooks/01-google-evidence-walk.ipynb)
  — the Google evidence read end to end from committed tables, 12 code cells,
  2 figures, 8 assertions at the bottom.
- [`notebooks/02-the-refusal-boundary.ipynb`](../../notebooks/02-the-refusal-boundary.ipynb)
  — the four gates run live on six tempting sentences. Three pass the wording
  gate and are refused at identification instead, which is a better answer to
  *"why can't you say X?"* than any slide.
- [`weekly-reports/`](../../weekly-reports/) — six reports, one per week the
  project ran, compiled from the commit record and labelled as such in each
  header.
- [`meeting-minutes/`](../../meeting-minutes/) — the Task 10 review agenda, and
  a README stating that minutes are written after a session and never
  back-filled. No minutes were invented for meetings with no record.
- The suite size: the check reads the **newest done row**, because earlier rows
  are snapshots that cannot be verified without archaeology. Task 09's row was
  the newest, and it had gone stale. It stays as written — the same rule that
  governs the corrections register governs a README — and Task 10's row now
  quotes 647, which is what `pytest` collects. The check moves with the newest
  row, so the number it grades is always one somebody can check today.

---

## 6. C9 — the correction this task raises against Task 09

Task 09 §13 predicted: *"Task 10 is the final presentation, and it is the first
task in this project whose output is not checkable by a test."*

Half right. Six of the instructions §13 wrote for this task turned out to be
properties of a file rather than of a performance, and each is now a named rule
that fails a build: `claim_exists`, `refusals_intact`, `clause_travels`,
`correction_carried`, `asset_exists`, `correction_exists`. The deck is
compiled from the ledger rather than written beside it, 26 rules run over it and
the question bank, and the instructions themselves all stand.

What is genuinely not checkable is the delivery: whether the storyline survives
contact with the room, whether a refusal is delivered as a finding or an
apology, whether a question outside the 21 is answered carefully. Nothing here
tests that, and C9 says so rather than letting 26 passing rules imply otherwise.

---

## 7. Limitations

1. **Provenance is not relevance.** Every bullet is bound to a table. Nothing
   checks that it is the right bullet for the slide, or that the deck as a whole
   answers the brief.
2. **38 narration bullets are unverified prose.** They carry no numerals and
   pass the wording gate, but claims about method — "rules, not models, so every
   extraction is inspectable" — rest on the author's judgement.
3. **The question bank is a guess by the person who wrote the answers.** Nine
   known prohibitions are guaranteed covered. The twenty-second question is not.
4. **The audit counts files.** `working_dir_used` would pass on three files of
   nonsense; what stops that is the commit record and the notebooks' own
   self-checks, not the check.
5. **One year, one country mix, six companies.** Everything above inherits the
   scope limits of Tasks 02 and 06.

---

## 8. Deliverables

| Artefact | Path |
| --- | --- |
| Deck (19 slides, speaker notes) | [`task-10-slides.md`](task-10-slides.md) |
| Mentor question bank (21 questions) | [`task-10-tables/mentor-qa.csv`](task-10-tables/mentor-qa.csv) |
| Deck provenance tables | [`task-10-tables/`](task-10-tables/) (11 CSVs) |
| Figures | [`task-10-figures/`](task-10-figures/) (8 PNGs) |
| Workspace audit | [`task-10-tables/workspace-audit.csv`](task-10-tables/workspace-audit.csv) |
| Machine-readable report | [`task-10-presentation-report.json`](task-10-presentation-report.json) |
| Team standard | [`docs/task-10-final-presentation-methods.md`](../../docs/task-10-final-presentation-methods.md) |
| Notebooks | [`notebooks/`](../../notebooks/) (2) |
| Weekly reports | [`weekly-reports/`](../../weekly-reports/) (6) |
| Mentor review agenda | [`meeting-minutes/task-10-mentor-review-agenda.md`](../../meeting-minutes/task-10-mentor-review-agenda.md) |
| Correction | [C9](../../docs/corrections.md#c9--a-deck-is-checkable-the-delivery-is-what-is-not) |
