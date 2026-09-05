# Task 12 — Fine-Tune Skill Extraction Model

**Team standard.** Task 04 built a 176-skill rule taxonomy and validated it.
Task 12 is optional and asks whether a *fine-tuned* model should replace it. The
answer for Google is no, and this document records why the question has only one
honest form on this data, how the model is built and scored against the taxonomy
on identical postings, and what has to be true — beyond beating the taxonomy —
before a learned extractor may retire a rule that covers more than it does. The
brief's deliverable is "fine-tuned model + metrics"; the metrics are the point,
and one of them is a refusal.

- **Code:** [`src/skill_model.py`](../src/skill_model.py) · [`src/build_skill_model.py`](../src/build_skill_model.py)
- **Tests:** [`tests/test_skill_model.py`](../tests/test_skill_model.py) (23; 775 in the suite)
- **Google findings:** [`members/ankit-google/task-12-skill-model-report.md`](../members/ankit-google/task-12-skill-model-report.md)
- **The baseline it is held to:** [`src/skills.py`](../src/skills.py), built in [`docs/task-04-skill-taxonomy.md`](task-04-skill-taxonomy.md) and validated in [`docs/task-04-skill-extraction-validation.md`](task-04-skill-extraction-validation.md)
- **Inherited from:** Task 02's data reality (no Google description text) and Task 04's provenance-tagged extraction (`google_skills_long.parquet`)
- **Legal position:** unchanged — no new source, no new collection, no new field. The model reads committed titles and committed labels only.

```bash
python src/build_skill_model.py        # 6 tables, 4 figures, 1 JSON report
python -m pytest tests/ -q
```

Three findings in one line each, so the rest of the document is their
justification:

1. **The fine-tune the brief pictures is unmeasurable on Google.** 0 of 848
   postings carry description text (§1). The only contest the data admits is
   title → the collector's skill list, and this document builds that one.
2. **The learned model wins that contest by +0.67 micro-F1 and is not adopted**
   (§4). Beating the taxonomy on a narrow title task is necessary, not
   sufficient.
3. **Adoption is blocked by three facts, not three opinions** (§4.2). Vocabulary
   collapse, label circularity and an empty text path are read off the data by
   `adoption_blockers`, so the gate cannot be argued away by tuning.

---

## 1. What is being compared, and why the shape is forced

### 1.1 The data reality that fixes the contest

A fine-tuned skill extractor, in the brief's sense, reads a job description and
names the skills in it. On Google that path has no input, and the numbers are
not marginal:

- **0 of 848** Google postings carry description text (Task 02). The approved
  sources return titles and structured fields; the Hugging Face backfill is a
  2023 snapshot without bodies, Adzuna truncates to 500 characters and has no
  credentials, The Muse lists no Google postings, and Google Careers was
  rejected on robots.txt at Task 01.
- Skills on these postings live in the collector's **`source`** list, parsed
  upstream from description text this repository never holds. **567** postings
  carry one (mean 4.0 skills; 89 distinct; **32 in ≥ 10 postings**).
- The taxonomy's **`title`** path — the only path with a live input on Google —
  recovers almost nothing, because skill names are rarely in titles.

So the one contest measurable on the same postings is **predict the source skill
set from the job title.** This is not a free choice; it is what remains once the
missing text is accounted for. A report that fine-tuned a text extractor here
would be scoring a model on an input that does not exist.

### 1.2 The label matrix both extractors share

`google_skills_long.parquet` is Task 04's own extraction, one row per
(posting, skill) tagged with the path that found it: `source`, `title`, or
`source+title`. That provenance column *is* the contest:

- **Labels `y`** — the `source`-provenance skills per posting (`source_labels`).
  `source+title` counts as a source label too — the substring match in `_prov`
  is deliberate.
- **Taxonomy baseline** — the `title`-provenance skills
  (`taxonomy_title_labels`), the rule's live prediction from the one text a
  posting carries.
- **Learned prediction** — out-of-fold, from the same titles.

All three are the same postings and the same columns. `assemble` builds the
shared 0/1 matrix; `learnable_vocabulary` restricts it to skills in ≥
`MIN_SUPPORT` (10) postings, because below that a fold can hold zero positives
and the metric is noise. A posting whose only skills are rarer than the floor is
**dropped**, not scored as an all-zero row that would flatter both extractors
equally — 13 of 567 postings, leaving **554**.

### 1.3 What the comparison is not

It is not a measure of skill-extraction *correctness*. The labels are the
collector's parse, which is exactly what Task 04 built a taxonomy because it
could not trust blindly. The contest measures agreement with that parse from
titles alone. This is stated up front because the headline margin is large
enough to be mistaken for a quality verdict, and it is not one (§4.3).

---

## 2. The model, declared before it is scored

`pipeline(seed)` is a scikit-learn `Pipeline`:

| Step | Choice | Why |
| --- | --- | --- |
| Vectoriser | TF-IDF, word n-grams (1, 2), `min_df=2`, lowercase | Titles are short; unigrams and bigrams carry "data scientist", "machine learning". Characters would only add noise. |
| Estimator | `OneVsRestClassifier(LogisticRegression(...))` | Multilabel: a posting has several skills. One linear model per skill, legible as weights. |
| `class_weight` | `"balanced"` | The one non-default choice. Without it a 0.5 threshold on a label present in a tenth of postings predicts almost nothing, and the contest would measure the threshold, not the model. |
| `solver` | `"liblinear"`, `random_state=seed`, `n_jobs=1` | Deterministic on sparse text. |

Evaluation is `MIN_SUPPORT = 10`, `N_SPLITS = 5`, `SEED = 20260905`, `ROUND = 4`
— all declared in the module before the numbers were read, the same rule Tasks
05–08 use so the committed tables rebuild bit-for-bit. `out_of_fold_predictions`
scores every posting with a model that did not train on it, so the number is
generalisation rather than memorisation.

---

## 3. Metrics: identical scoring for both extractors

`scores` computes micro and macro precision/recall/F1 over the shared label
matrix, plus the raw counts (`label_instances`, `predicted_instances`,
`true_positives`) so any headline can be interrogated. Both extractors go
through the same function on the same `y`. The gate metric is **micro-F1** —
the head-to-head over all label instances — because macro-F1 on 32 columns of
uneven support answers a different question (per-skill average) that the thin
skills dominate.

`per_skill_scores` breaks both extractors down to one row per skill;
`title_literal_coverage` reports the share of positive labels whose skill name
is verbatim in the title (**0.145** on Google). That number is the ceiling for
any literal title matcher and the evidence that the learned model earns its
recall from association, not from words the title happens to contain (§4.3).
Every metric is rounded to 4 dp.

---

## 4. The gate — computed from the metrics and the data, not asserted

### 4.1 Clearing the gate

`gate_verdict` reads `beats_micro = learned micro-F1 > taxonomy micro-F1`. On
Google that is **0.6874 > 0.0175**, margin **+0.6699** — cleared. Clearing is
necessary and, on its own, meaningless for adoption, because the contest is a
title task the taxonomy was never built to win.

### 4.2 The three blockers, each a fact about the data

`adoption_blockers` returns three conditions, each read off the dataset so it
flips with the dataset rather than with a decision:

1. **`vocabulary_collapse`** — `blocked` when the learnable vocabulary is smaller
   than the taxonomy. 32 < 176: the model cannot fit the 144 skills with too few
   examples. Replacing a 176-skill extractor with a 32-skill one loses coverage
   the gate's single number does not see.
2. **`label_circularity`** — always `blocked` on this construction. The labels
   are the collector's source list, from text the repository does not hold; the
   model distils those labels from titles rather than extracting skills from a
   posting. The "win" is over a baseline reading a different input.
3. **`no_text_path_input`** — `blocked` when `postings_with_text == 0`. 0 of 848:
   a learned text extractor has no live input on Google, so the gap it would
   fill is empty.

### 4.3 Adoption = cleared **and** every blocker clear

```
adopted = beats_micro AND (no active blocker)
```

This is the honest-refusal invariant, pinned directly in
`test_adoption_requires_clearing_the_gate_and_every_blocker`, which sweeps all
four corners so neither half can be dropped by a later refactor. A cleared gate
with a live blocker must still refuse. On Google the gate is cleared and all
three blockers are active, so the taxonomy in `src/skills.py` is **retained**.

The separation is the whole design: the blockers are computed *apart from*
"does it beat the taxonomy" precisely so that beating the taxonomy cannot, by
itself, retire the rule. A number moving is not a reason to ship.

---

## 5. The model is published as a table, not a pickle

`learned_associations` fits the final model and exports the highest-weighted
title n-grams per skill. A committed pickle would not survive a scikit-learn
bump byte-for-byte and could not be read by a marker who is reading the repo
rather than running it. The weights table is reproducible from the seed and
legible: it shows the model recovering Python and SQL from "data scientist"
titles, and Java and Go from "engineer" titles — association, which is both why
it wins (§4.1) and why it is refused (§4.2, blocker 2).

`model_spec` records the family, vectoriser, estimator, input, label definition,
evaluation and seed — enough to rebuild the model from code. That, plus the
associations table and the seed, is the deliverable "model"; there is no binary
artefact.

---

## 6. Why the driver is not a registered pipeline stage

`src/build_skill_model.py` is run by hand, not wired into `src/pipeline.py`'s
`STAGES`. Three reasons:

1. **The model is not adopted.** Registering it would put it in the production
   DAG and overstate its role; the `features` stage keeps extracting skills with
   the taxonomy.
2. **The pipeline's `lint_dag` checks forward only** — a registered stage must
   have a script — so an unregistered driver passes every check, exactly as the
   bare-noun library modules (`skills.py`, `skill_model.py`) already do.
3. **Registering would add ripple** — pipeline-stages, DAG edges, the run ledger
   — on top of the unavoidable deck-fact ripple, for a stage that must not run
   in the production rebuild.

The one unavoidable consequence is the deck facts, which are counted live:
adding Task 12's tests, tables and figures moves `tests_total`,
`tables_committed` and `figures_committed`, which flows into the Task 11 drift
check. Task 10's `deck-facts.csv` and the Task 11 tables are rebuilt after Task
12 so the committed numbers match the repository — that is not optional, it is
the drift check doing its job.

---

## 7. What gets committed

**Committed** — 6 aggregate tables in `members/<you>/task-12-tables/`, 4 figures
in `task-12-figures/`, and one `task-12-skill-model-report.json` carrying every
headline number with its seed. The JSON is the machine-readable deliverable; the
markdown report is for humans. No weight blob.

**Not committed** — anything row-level. `data/processed/` stays git-ignored, as
it has since Task 03. `forbidden_columns` and `personal_data_columns_present`
run over the 6 tables in the suite before this document's claims can be trusted.

The `dataset-summary.csv` values are written as strings so integer counts render
as `554`, not `554.0` — the column mixes counts and a rate, which otherwise
coerces the whole column to float. `test_dataset_summary_records_the_vocabulary_collapse`
reads it with `dtype=str` and pins the on-disk rendering.

---

## 8. Checklist for each specialist

The other three company seats are unfilled, but the standard holds if they are:

1. **Establish the data reality first.** If your company has description text,
   the text-path fine-tune the brief describes is measurable — build that.
   If it does not (as Google does not), say so and build the title contest
   instead. Do not score a text extractor on absent text.
2. **Score both extractors on the identical label matrix** — same postings, same
   vocabulary, out of fold. The provenance split in `skills_long` is the
   substrate; do not invent a new label source.
3. **Set `MIN_SUPPORT` and the seed before you look.** Drop postings whose only
   skills are below the floor; never score all-zero rows.
4. **The gate is micro-F1, cleared before adoption is even considered.** Report
   macro-F1 and per-skill too, so the thin skills are visible.
5. **Compute the adoption blockers off your data.** Vocabulary collapse and text
   availability are company-specific; label circularity is structural and
   applies to everyone reading upstream labels.
6. **Adoption requires clearing the gate AND every blocker.** A cleared gate is
   not adoption. If you adopt, you are retiring `src/skills.py` for your
   company, and the report must justify the coverage you lose.
7. **Publish the model as weights, not a pickle**, and keep it offline and
   seeded. If a rebuild needs a download, the reproducibility guarantee is gone.
8. **Do not register the driver as a stage unless the model is adopted.** Rebuild
   the deck facts and the Task 11 drift check either way.
