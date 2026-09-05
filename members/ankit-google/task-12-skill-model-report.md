# Task 12 — Fine-Tuned Skill Extraction Model Report (Google)

**Specialist:** Ankit Ranjan · **Company:** Google (Alphabet) · **Date:** 2026-09-05

Task 12 is optional, and its deliverable is a *fine-tuned model + metrics*. The
model is delivered, the metrics are below, and the model is **not adopted**. It
beats the rule taxonomy on the only contest this dataset admits — by a margin of
**+0.67 micro-F1** — and is still declined, for three reasons that are facts
about the Google data rather than choices about the model. This is the same
shape as Tasks 07, 08 and 09: a result is delivered with its numbers and
refused for stated reasons, not adopted because a number moved.

- **Method rationale (team standard):** [`docs/task-12-skill-model-methods.md`](../../docs/task-12-skill-model-methods.md)
- **Code:** [`src/skill_model.py`](../../src/skill_model.py) · [`src/build_skill_model.py`](../../src/build_skill_model.py)
- **Tests:** [`tests/test_skill_model.py`](../../tests/test_skill_model.py) (23) — **775 in the suite**
- **Machine-readable report:** [`task-12-skill-model-report.json`](task-12-skill-model-report.json)
- **Tables:** [`task-12-tables/`](task-12-tables/) (6) · **Figures:** [`task-12-figures/`](task-12-figures/) (4)
- **The bar it was held to:** the 176-skill rule taxonomy in [`src/skills.py`](../../src/skills.py), built and validated in [Task 04](../../docs/task-04-skill-taxonomy.md)

```bash
python src/build_skill_model.py        # 6 tables, 4 figures, 1 JSON report
python -m pytest tests/ -q
```

**Everything below is within-2023, on the 848 Google postings this repository
holds.** The model is a scikit-learn multilabel classifier, trained offline with
a fixed seed, evaluated out of fold; every number rebuilds bit-for-bit. It is
not a claim about what Google demands — it is a claim about what a title-only
model can recover from a label set the taxonomy never had, on a dataset whose
text the taxonomy was built for is missing.

---

## 1. The headline: the model wins the contest and is refused anyway

| Extractor | micro-F1 | micro precision | micro recall | macro-F1 |
| --- | --- | --- | --- | --- |
| Rule taxonomy — title path | 0.0175 | 1.0000 | 0.0088 | 0.0194 |
| **Learned model — title → skills** | **0.6874** | 0.5931 | 0.8173 | 0.5971 |
| **Margin** | **+0.6699** | −0.4069 | +0.8085 | +0.5777 |

The gate — *a learned extractor must beat the 176-skill rule taxonomy on the
same postings before it replaces anything* — is **cleared**. The decision is
**not adopted**; the taxonomy in `src/skills.py` is retained. Three blockers
(§5) stand between clearing the gate and replacing the rule, and clearing the
gate on this narrow task is precisely the thing that must not, by itself, retire
the taxonomy.

The margin looks overwhelming and is real, but it is a margin on a contest the
taxonomy was never built to win (§2). The taxonomy's title path predicts **18
skills across 554 postings** and every one of them is correct — precision 1.0,
recall 0.0088. It is a conservative literal matcher doing exactly what it was
designed to do, on the one input it has, and that input carries almost no
signal.

---

## 2. The contest is fixed by the data, not chosen

The brief imagines fine-tuning the *text* path — a learned extractor reading a
job description and naming the skills in it. On Google that path has no input:

- **0 of 848 Google postings carry description text.** Task 02 established the
  approved sources return titles and structured fields but not bodies for
  Google. The path a fine-tuned extractor would improve is empty.
- **Skills live in the collector's `source` list**, parsed upstream from
  description text this repository never holds. **567 postings** carry such a
  list (mean 4.0 skills, 89 distinct, **32 in ≥ 10 postings**).
- **The taxonomy's `title` path recovers almost nothing** — the job title is the
  only per-posting text present, and skill names rarely appear in titles.

So the only contest measurable on the same postings is: **predict the source
skill set from the job title.** The labels `y` are the source-provenance skills
(restricted to the 32 that clear a support floor of 10); the taxonomy baseline
is its title-path extraction; the learned prediction is out-of-fold. All three
are the same 554 postings and the same 32-column label matrix. This is not the
fine-tune the brief pictured — it is the only honest one the Google data allows,
and §5 is careful to say so.

**554 postings scored** — the 567 with a source list, minus **13** whose only
skills fall below the support floor and would score as all-zero rows that
flatter both extractors equally.

---

## 3. The model, and where its score comes from

A TF-IDF over title word unigrams and bigrams into a one-vs-rest
`LogisticRegression(class_weight="balanced", solver="liblinear")`, evaluated by
5-fold out-of-fold prediction (`cross_val_predict`). The full specification is
in the JSON `model` block and is enough to rebuild the model from code — there
is no committed weight blob (§6).

Per-skill, the score is carried by the well-supported skills:

| Skill | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| R | 214 | 0.8091 | 0.9112 | **0.8571** |
| Python | 372 | 0.8795 | 0.7849 | 0.8295 |
| SQL | 255 | 0.8161 | 0.8353 | 0.8256 |
| Java | 132 | 0.6461 | 0.8712 | 0.7419 |
| Go | 95 | 0.5816 | 0.8632 | 0.6949 |
| C++ | 102 | 0.4911 | 0.8137 | 0.6125 |

The macro-F1 (0.5971) sits below the micro-F1 (0.6874) because the thin skills
drag: the model has 32 columns and the least-supported of them clear 10
postings, which is enough to fit but not enough to fit well.
`per-skill-learned.csv` carries all 32; `per-skill-taxonomy.csv` scores the
title path on the identical skills for the side-by-side.

---

## 4. The win is association, not literal matching

The number that keeps the headline honest is **title literal coverage = 0.145**:
only **14.5%** of the positive labels have their skill name appearing verbatim in
the posting title. That is the ceiling for *any* purely literal title matcher —
the taxonomy's title path cannot exceed it — and the learned model's recall
(0.8173) sits far above it. The model is not reading skill names out of titles;
it is reading the *role* and inferring the skills that role asks for.

`learned-title-associations.csv` publishes the fitted model as a table — the
highest-weighted title terms per skill, not a pickle — so the mechanism is
legible:

- **Python** and **SQL** are recovered from `data scientist`, `data analyst` and
  `business` titles, none of which contain the words "Python" or "SQL".
- **Java**, **Go**, **C++** load on `engineer`, `software engineer`,
  `data engineer` — the generic engineering titles.
- **Kubernetes** loads on `infrastructure`, `cloud platform`, `cloud engineer`.

This is exactly why the win does not translate into adoption. The model has
learned the association `title → the collector's usual skills for that title`.
That is a real regularity, and it is also §5's second blocker: it distils the
label set, it does not read a posting.

Figure `03-association-gap.png` draws the one comparison that makes this
concrete: the taxonomy's title-path recall, the 14.5% literal ceiling, and the
model's recall, in that order — the model clears the ceiling the literal matcher
is trapped under.

---

## 5. Why it is not adopted — three facts about this dataset

`adoption-gate.csv` records the gate as cleared and then three blockers, each
read off the data by `adoption_blockers`, each of which alone is disqualifying:

1. **Vocabulary collapse.** The model learns **32 of the taxonomy's 176 skills**
   — only those in ≥ 10 postings. The taxonomy names 144 skills the model has
   too few examples to fit at all. Replacing a 176-skill extractor with a
   32-skill one loses coverage the gate's single number does not see.
2. **Label circularity.** The labels are the collector's `source` list, parsed
   from description text this repository does not hold. The model distils those
   labels from titles; it does not extract skills from a posting. Its "win" is
   over a baseline reading a different input, on labels only the discarded text
   can produce.
3. **No text-path input.** **0 of 848** postings carry description text, so a
   learned text extractor — the thing Task 12 was meant to build — has no live
   input on Google. The gap it would fill is empty.

The first is about coverage, the second about what the task really measures, the
third about whether the improvement has anywhere to land. None is a modelling
choice, so none can be tuned away. Figure `04-gate-scorecard.png` is the verdict
as a card: beaten baseline, cleared gate, retained taxonomy.

---

## 6. Reproducibility: seeded, offline, no weight blob

Same discipline as the rest of the pipeline:

- **Deterministic.** `SEED = 20260905`, `liblinear`, `n_jobs=1`, a fixed
  `KFold`. Two runs produce bit-identical predictions
  (`test_out_of_fold_predictions_are_deterministic`), and every metric is
  rounded to 4 dp against last-ULP drift across BLAS builds.
- **Offline.** scikit-learn only — no spaCy, no sentence-transformers, no
  downloaded weights. This is deliberate: a downloaded-weight extractor would
  make a rebuild depend on a fetch and break the seed's guarantee. Pinned at the
  source in `test_module_pulls_in_no_downloaded_weight_model`.
- **No pickle.** A committed model blob would not survive a scikit-learn bump
  byte-for-byte. The model is published as what it learned —
  `learned-title-associations.csv` — which is reproducible from the seed and,
  unlike a blob, readable.

The driver is deliberately **not** a registered pipeline stage. The model is not
adopted, so wiring it into the production DAG would overstate its role; the
`features` stage keeps extracting skills with the taxonomy. Task 12 is run by
hand, like a one-off experiment, and its evidence is committed like any other
task's.

---

## 7. Limitations

1. **Not the fine-tune the brief pictured.** It cannot be, on Google — there is
   no description text to read (§2). This is a title→label model, and the report
   never calls it a text extractor.
2. **The labels are upstream, not ground truth.** They are the collector's
   parse, which the taxonomy was built precisely because we could not trust
   blindly. The contest measures agreement with that parse, not correctness.
3. **32 skills, not 176.** Every number here is over the skills common enough to
   learn. The long tail the taxonomy covers is out of scope by construction.
4. **One company, one year, one collection window.** The associations are
   Google's title vocabulary in 2023; they would not transfer unchanged to
   another company's postings.
5. **The margin is not a quality claim about the taxonomy.** The taxonomy scores
   0.0175 here because it is a conservative literal matcher on titles — a task it
   was never asked to do. On the text path it was built for, this contest says
   nothing.

---

## 8. What this task changes, and what it does not

**Changes:** nothing in the production pipeline. The taxonomy is retained. No
correction is registered — Task 12 does not overturn a prior task's prediction;
it is a self-contained experiment whose result is a refusal.

**Adds:** the deck facts move, because they are counted live —
`tests_total` 747 → 770 (the 23 tests here), and the Task 12 tables and figures
join the aggregate counts. Task 10's `deck-facts.csv` and the Task 11 drift
check are rebuilt so the committed numbers match the repository.

| Item | Path |
| --- | --- |
| Method rationale (team) | [`docs/task-12-skill-model-methods.md`](../../docs/task-12-skill-model-methods.md) |
| This report | `members/ankit-google/task-12-skill-model-report.md` |
| Machine-readable report | [`task-12-skill-model-report.json`](task-12-skill-model-report.json) |
| Tables (6) | [`task-12-tables/`](task-12-tables/) |
| Figures (4) | [`task-12-figures/`](task-12-figures/) |
| Tests | [`tests/test_skill_model.py`](../../tests/test_skill_model.py) (23 of 775) |
| Retained baseline | [`src/skills.py`](../../src/skills.py) — the 176-skill taxonomy |

Row-level data stays git-ignored. All 6 tables pass the forbidden-column and
personal-data checks before they are written.
