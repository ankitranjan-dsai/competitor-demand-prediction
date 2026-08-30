# Task 10 — Final Presentation & Mentor Review

**Team standard.** The brief asks for two things: "a polished final
presentation (per member + aligned storyline)" and "a complete, well-structured
GitHub workspace". Both are delivery, and delivery is where nine tasks of
careful refusal are most easily undone — not by anyone lying, but because a
slide is a small space and a qualifying clause is the first thing that does not
fit. Task 09 generated 436 claims and published 120, each carrying its clause,
its citation and its status. A deck written by hand beside that ledger would be
a deck whose numbers nothing checks.

So this document does not describe how slides were designed. It describes a
**deck compiler**: a slide bullet is a typed record, a numeral may only reach a
slide through a claim record or a resolved fact, and 26 rules run over the deck
and the mentor question bank before either is written to disk. The same machine
then turns on the repository itself, because "finalise the workspace" is a
claim about files and can be checked like one.

- **Code:** [`src/present.py`](../src/present.py) · [`src/build_presentation.py`](../src/build_presentation.py)
- **Tests:** [`tests/test_presentation.py`](../tests/test_presentation.py) (83; 647 in the suite)
- **Google deck:** [`members/ankit-google/task-10-slides.md`](../members/ankit-google/task-10-slides.md)
- **Google findings:** [`members/ankit-google/task-10-presentation-report.md`](../members/ankit-google/task-10-presentation-report.md)
- **Notebooks:** [`notebooks/01-google-evidence-walk.ipynb`](../notebooks/01-google-evidence-walk.ipynb) · [`notebooks/02-the-refusal-boundary.ipynb`](../notebooks/02-the-refusal-boundary.ipynb)
- **What this task overturned:** [`docs/corrections.md`](corrections.md) —
  [C9](corrections.md#c9--a-deck-is-checkable-the-delivery-is-what-is-not),
  Task 09 §13's own prediction about this task
- **Inherited from:** [`docs/task-09-insight-generation-methods.md`](task-09-insight-generation-methods.md)
  §13 (the claim ledger, the refusals as content, `published` ≠ uncaveated) and
  §10 (the audience table)
- **Legal position:** unchanged — no new source, no new collection, no new
  field. This task reads committed tables and writes markdown.

```bash
python src/build_presentation.py       # 11 tables, 8 figures, 1 deck, 1 JSON report
python -m pytest tests/ -q
```

Four findings in one line each, so the rest of the document reads as their
justification:

1. **The deck is compiled from the ledger, not written beside it.** 61 bullets:
   13 bound to a ledger row, 10 rendered from the fact register, and 38 that
   carry no numeral at all because a rule forbids it (§1, §4).
2. **Refusals occupy slide space, by rule.** A deck must carry at least four,
   and 11 of the 21 answers in the mentor bank are "not available" followed by
   the reason and the falsifier (§7). The section is one of five that cannot be
   dropped for time (§3).
3. **The audit turned on the repository and found four defects** — three
   working directories holding nothing but a placeholder, and a README quoting
   a test count that had moved by 89 (§8). That list, rather than a tidy-up by
   eye, is what "finalise the workspace" concretely meant.
4. **A deck is checkable; the delivery is not** (§11). 26 rules can hold a deck
   to its evidence. Nothing in this repository can tell whether the
   presentation was any good, and C9 records that rather than implying
   otherwise.

---

## 1. A bullet is a record, not a sentence

The failure mode Task 09 named for prose applies twice over to slides. Prose at
least has room for a clause; a bullet does not, so the clause is dropped, and
what is left reads as a stronger claim than the ledger supports. Nothing about
the drop is visible afterwards: the slide looks like a slide.

A bullet in this compiler is therefore a record with a `kind`, and the kind
decides what the bullet is allowed to contain:

```python
Bullet(kind="claim",   ref="share-google")     # text comes from the ledger
Bullet(kind="refusal", ref="tempting-forecast")
Bullet(kind="fact",    text="{postings_focus} postings, {publishers_focus} publishers")
Bullet(kind="plain",   text="The panel control is what makes the series comparable")
```

The kinds are closed. There is no fifth kind and no escape hatch, which is the
only reason the rules in §5 can be exhaustive: every bullet on every slide is
one of four shapes, and each shape has a rule that says where its content came
from.

---

## 2. Four kinds, and what each one may contain

| Kind | Text comes from | May contain a numeral? | Checked by |
| --- | --- | --- | --- |
| `claim` | a `claim-ledger.csv` row, verbatim through `ins.sentence()` | yes — the ledger's | `claim_exists`, `claim_publishable`, `clause_travels`, `correction_carried`, `claim_gate_lint` |
| `refusal` | a refused ledger row, verbatim | yes — the ledger's | `claim_exists`, `refusal_refused`, `refusal_attributed` |
| `fact` | a template over the fact register | only through `{key}` | `fact_resolves`, `fact_numerals_bound` |
| `plain` | the author | **no** | `plain_numerals_bound`, `language_clean` |

`LEDGER_KINDS = ("claim", "refusal")` names the two whose text is not written by
Task 10 at all. That is the load-bearing distinction: for those two the deck is
a *view* of the ledger, and the sentence a mentor hears is the sentence Task 09
gated, including its clause.

The `plain` kind is narration — "the panel control is what makes the series
comparable" — and it may not contain a numeral. Not "should not": a rule
rejects the deck. Narration is where a remembered number goes when it is typed
from memory, and a remembered number is exactly the kind that drifts from the
table it came from.

---

## 3. The deck is a fixed set of slots, five of which cannot be dropped

`SECTIONS` fixes 19 slots in one order, and every specialist's deck fills the
same 19 in the same sequence. That is what "align on a consistent slide format"
means when it is enforced rather than agreed:

`title` · `problem` · **`legal`** · `data` · `pipeline` · **`identification`** ·
`trends` · `comparison` · `forecast` · `similarity` · `insights` · `position` ·
`stack` · **`refusals`** · **`corrections`** · **`quality`** · `workspace` ·
`next` · `qa`

The five in bold are `REQUIRED_SECTIONS`. A deck may not drop them, and
`section_required` fails the build if one is missing. The list is not arbitrary:

- **`legal`** — the collection was gated before it happened, and a deck that
  omits it invites the question at the worst moment.
- **`identification`** — what a posting count actually counts. Every refusal
  downstream traces to this slide.
- **`refusals`** — Task 09 §13 predicted this is the first section a deck drops
  and among the most decision-relevant output the project has. Making it
  mandatory is the cheapest possible protection against being right about that.
- **`corrections`** — nine corrections, two of them about the project rather
  than the data. A deck that shows only the surviving claims is a deck that has
  quietly discarded its own error record.
- **`quality`** — how any of it is checked.

`section_order` and `section_known` enforce the sequence and reject a slot that
is not in the table; `slide_numbering` checks that slide *n* is the *n*th.
Density is bounded at `MIN_BULLETS = 1` and `MAX_BULLETS = 5` — a slide with
nine bullets is a document, and it will be read rather than heard.

---

## 4. The fact register: eighteen numbers, each with a resolver

A `fact` bullet is a template. `{postings_focus}` is not a number typed into a
slide; it is a key in `FACTS`, and each entry carries a resolver that reads a
committed table:

| Key | Resolves to | Read from |
| --- | --- | --- |
| `postings_focus` | 846 | the comparability window, not the screen (C4) |
| `publishers_focus` | 95 | the publisher table |
| `months_observed` | 12 | the monthly panel |
| `companies_compared` | 6 | the competitor set |
| `companies_screened_out` | 2 | the employer registry |
| `postings_pool` | 3,605 | the six-company pool |
| `skills_observed` | 91 | the shared taxonomy |
| `claims_generated` / `claims_published` / `claims_refused` | 436 / 120 / 316 | the claim ledger |
| `claim_yield_pct` | 27.5 | derived from the three above |
| `distinctive_skills` | 48 | the distinctiveness table |
| `nearest_score` | 0.9174 | the similarity table |
| `tests_total` | 647 | `pytest --collect-only` |
| `tasks_total` / `corrections_total` | 10 / 9 | the README task table, the register |
| `tables_committed` / `figures_committed` | 127 / 49 | the repository itself |

`fact_resolves` fails on a key with no entry. `fact_numerals_bound` fails on a
bare numeral in a `fact` template — a template that says "846 postings" rather
than "{postings_focus} postings" has hard-coded exactly the number the register
exists to keep current.

The register is why the deck cannot go stale quietly. When the suite grows, the
`quality` slide changes on the next build. When it does not change, that is
because nothing moved.

---

## 5. Twenty-six rules, in four layers

Thirteen rules run over bullets, five over slides, four over deck structure and
four over the question bank. Every one has a test in
[`tests/test_presentation.py`](../tests/test_presentation.py), and the test
asserts the rule *fires* on a constructed violation — a rule that cannot fail
is a rule that is not checking anything.

**Bullets (13).** `claim_exists` (the `ref` is a real ledger row) ·
`refusal_refused` (a `refusal` bullet points at a row whose status is refused) ·
`refusal_attributed` (it says why, and what would lift it) · `claim_publishable`
(a `claim` bullet points at a published or published-qualified row) ·
`clause_travels` (a qualified claim keeps its clause on the slide) ·
`correction_carried` (a claim whose ledger row names a correction carries the
pointer) · `claim_gate_lint` (the ledger records that Task 09's wording gate
passed it) · `fact_resolves` · `fact_numerals_bound` · `plain_numerals_bound` ·
`language_clean` (the nine prohibited patterns, on text Task 10 wrote) ·
`bullet_speakable` (`SPOKEN_MAX = 320` characters — beyond that it is read
aloud, not spoken) · `long_claims` (a slide may not be all long bullets).

**Slides (5).** `slide_density` (1–5 bullets) · `notes_present` (every slide
carries speaker notes) · `refusals_intact` (`MIN_REFUSALS = 4` across the deck)
· `asset_exists` (every figure a slide names is on disk) · `correction_exists`
(a correction a slide cites is in the register).

**Deck (4).** `section_known` · `section_order` · `section_required` ·
`slide_numbering`.

**Question bank (4).** `qa_slide_exists` · `qa_is_a_question` ·
`qa_covers_required` · `qa_ids_unique`.

---

## 6. The negation problem, and why ledger-bound bullets are exempt

`language_clean` runs `ins.lint_text` — the nine-rule wording gate from Task 09
§4 — over text Task 10 wrote. It does **not** run over `claim` and `refusal`
bullets, and the reason is worth stating precisely, because it looks like an
exemption bought for convenience.

A qualifying clause earns its place by naming the forbidden reading in order to
deny it. The published Google share claim ends: *"share of a fixed publisher
pool, **not headcount** and not absolute volume; unanimity here is
floor-dependent, see C8"*. Re-lint that sentence and `unmeasured_construct`
fires on the word *headcount* — a word that is only there because the clause is
denying it. The gate would reject the sentence for containing its own defence.

The resolution is not an exception list. The wording gate already ran on that
sentence once, in Task 09, and the ledger records the verdict in `gate_lint`. So
`claim_gate_lint` checks the recorded verdict instead of re-running the gate,
and `language_clean` is restricted to the words this task typed. Every sentence
in the deck is gated exactly once, by the gate that had the right context.

[`notebooks/02-the-refusal-boundary.ipynb`](../notebooks/02-the-refusal-boundary.ipynb)
runs this live, along with the more uncomfortable half: of six tempting
sentences, the wording gate catches three. *"Google posts more jobs than
Snowflake"* is caught; *"Google is hiring more engineers than Meta"* is not,
because the rule's noun list has `jobs` and not `engineers`. Each rule is a
regex over a finite vocabulary and a synonym walks past it. What the synonym
does not walk past is the ledger: all three uncaught sentences are refused at
identification, and a deck bullet cannot assert one, because a bullet that
asserts something has to point at a row.

The wording gate is a floor. Identification is the boundary.

---

## 7. The mentor question bank: an answer is a claim or a refusal

The brief asks the presenter to "answer questions on methods and results". An
answer given live is prose written under time pressure in front of the person
most likely to check it — the worst possible conditions for a careful clause.

So the answers are compiled too. 21 questions, 48 answer lines, each line the
same four kinds as a slide bullet:

| Verdict | Lines | What it means |
| --- | --- | --- |
| refused | 11 | the honest answer is "not available", with the ledger row, the reason, and the falsifier |
| answered | 6 | a published claim, rendered with its clause |
| explained | 4 | narration, no numerals |

`qa_covers_required` is the rule that matters here. Nine prohibitions are
listed as questions a mentor is likely to ask — the forecast, the level
comparison, the salary gap, the country split, the seasonality, the headcount,
the product launch, convergence, and what an investor gets — and the bank is
rejected if any is unanswered. `prohibitions_open` is empty in the shipped
report.

The bank's shape is deliberate: **11 of 21 answers are refusals**. A question
bank in which everything can be answered is a bank that has quietly widened
what the project claims.

---

## 8. The workspace audit: the deck grades the repository

"A complete, well-structured GitHub workspace" is a claim about files, so the
build checks it: nine checks over five areas, each run once per subject it
applies to — 37 runs in this build, and a number that grows with the repository
— written to
[`workspace-audit.csv`](../members/ankit-google/task-10-tables/workspace-audit.csv).

| Area | Check | What it asserts |
| --- | --- | --- |
| readme | `task_row_backed` | every task row's status matches deliverables that exist on disk |
| readme | `task_row_path` | every path quoted in the task table resolves |
| readme | `suite_size_current` | a quoted test count matches what `pytest` collects |
| docs | `link_resolves` | every relative link in the repository resolves |
| members | `member_readme` | a member folder carries its own README |
| folders | `working_dir_used` | a directory in the layout holds something beyond its placeholder |
| privacy | `env_untracked` | `.env` is never committed |
| privacy | `row_data_untracked` | row-level data under `data/` stays git-ignored |
| privacy | `personal_data_columns_present` | no committed table names a person |

The first run failed four checks, and all four were real:

- `suite_size_current` — the README and the member README both quoted **557
  tests** against a suite that then collected **646**. The figure was true when Task
  09 shipped and had been false since.
- `working_dir_used` × 3 — `notebooks/`, `weekly-reports/` and
  `meeting-minutes/` had been in the layout diagram since Task 01 and held
  nothing but a placeholder.

Those four are what Task 10 actually had to fix, and the fix is the deliverable
rather than the audit: two executed notebooks, six weekly reports compiled from
the commit record, a minutes folder that says what minutes are, and a Task 10
row in both READMEs that quotes the suite as it now stands. Task 09's rows keep
the 557 they shipped with: `suite_size_current` grades only the newest done row,
because an earlier row is a snapshot and rewriting it would be the move the
corrections register exists to prevent.

The privacy checks are the standing ones and are re-run every task;
`personal_data_columns_present` has passed over every committed table since
Task 02, and no committed column name contains `candidate`.

---

## 9. The build is a fixpoint

The deck counts the repository, and the deck is part of the repository. Its own
tables and figures move `tables_committed` and `figures_committed`, and the
`quality` slide quotes them. Emitting once leaves the numbers one build behind.

So `build_presentation.py` resolves the register, emits, re-resolves, and emits
again only if a value moved — up to `MAX_PASSES = 3`, and it fails loudly if the
numbers are still moving after that:

```
the build did not settle in 3 passes; still moving: ...
```

A deck that cannot state its own size is a deck making a claim it cannot check,
and failing is the correct outcome rather than shipping the number from pass
one.

---

## 10. What a deck may drop, and what it may not

Time is the pressure that turns a careful project into an overconfident one, so
the compiler makes the trade explicit. The five required sections stay. Density
is capped, so a slide cannot absorb a cut section by growing. `refusals_intact`
holds four refusals in place. `clause_travels` means a qualified claim keeps its
clause or the build fails — a bullet cannot be shortened by dropping the half
that makes it true.

Everything else is negotiable, and that is the point: the negotiable part is
named, and the non-negotiable part is enforced by something other than the
presenter's judgement five minutes before the session.

---

## 11. What is not checkable

C9 is the correction this task registered against itself, and it is the
honest end of the document.

Task 09 §13 predicted that "Task 10 is the final presentation, and it is the
first task in this project whose output is not checkable by a test." That is
half right, and the half it gets wrong is the productive half: six of the
instructions §13 wrote for this task turned out to be properties of a *file*,
and each is now a named rule — `claim_exists`, `refusals_intact`,
`clause_travels`, `correction_carried`, `asset_exists`, `correction_exists`.

What remains genuinely unchecked is everything that happens in the room:

- whether the storyline is followed or abandoned at slide 4;
- whether a refusal is delivered as a finding or mumbled as an apology;
- whether a question outside the 21 is answered carefully or improvised;
- whether the deck was any good.

No rule in `src/present.py` touches any of that. The presentation is the one
deliverable this repository cannot grade, and the register says so rather than
letting 26 passing rules imply otherwise.

---

## 12. Limitations

1. **The rules check provenance, not relevance.** A bullet can be perfectly
   bound to a ledger row and still be the wrong bullet for the slide. Nothing
   detects a deck that is accurate and useless.
2. **`plain` bullets are unverified prose.** They may carry no numerals and
   they pass the wording gate, but "the panel control is what makes the series
   comparable" is a claim about methods that no table checks.
3. **The question bank is a guess.** 21 questions were written by the person
   who also wrote the answers. `qa_covers_required` guarantees the nine known
   prohibitions are covered; it guarantees nothing about the twenty-second
   question.
4. **The audit checks structure, not quality.** `link_resolves` proves a link
   is not broken, not that the document it points at is worth reading.
   `working_dir_used` counts files, and a directory can be filled with anything.
   What stops that here is the commit record, not the check.
5. **`tests_total` is a count.** 647 passing tests is a fact about coverage of
   the rules that exist, and says nothing about the rules nobody wrote.

---

## 13. What comes after Task 10

There is no Task 11 in the brief; the two optional extensions are an automated
pipeline and a fine-tuned extraction model. Both inherit the same constraint
this task ends on, so it is worth writing down while it is still true:

- **The ledger is the interface.** Anything downstream — a scheduled pipeline,
  a dashboard, a second year of collection — should read `claim-ledger.csv` and
  its four gates rather than the reports. The reports are a rendering.
- **A refusal is not a TODO.** Each one names a falsifier. A pipeline that
  re-runs monthly should re-evaluate the falsifiers, not quietly retry the
  claims.
- **Two limits no engineering removes.** One incomplete annual cycle, and a
  maximum useful forecast horizon of zero. More collection changes the first.
  Nothing in this repository changes the second.

And the standing lesson, in the form C7 gave it: **a handover section is a
prediction, not an instruction.** This one is no exception, and C9 is the proof
— §13 of the previous standard predicted this task, and was corrected by it.
