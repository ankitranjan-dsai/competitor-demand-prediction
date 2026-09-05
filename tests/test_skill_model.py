"""Tests for the Task 12 learned skill extractor and its gate.

Task 12 is optional and its result is a refusal: a scikit-learn model that
beats the rule taxonomy on the one contest the data admits, and is still not
adopted. A refusal is only worth committing if the machinery behind it is
pinned, so this suite locks the three things that make the verdict honest:

* **the contest is the same postings, the same columns, for both extractors.**
  The labels are the collector's `source` skills, the taxonomy's prediction is
  its `title` path, and both come out of the one committed `skills_long` frame
  tagged by provenance. `source+title` counts as both — the provenance split is
  the whole experiment, so ``test_provenance_split_counts_source_plus_title_as_both``
  pins it.
* **the metrics are computed, not asserted.** Micro and macro precision/recall
  /F1 are hand-computed on a tiny matrix in
  ``test_scores_match_a_hand_computed_matrix`` so a scikit-learn change that
  moved them would be caught here rather than flattering the headline.
* **adoption needs more than winning.** ``test_adoption_requires_clearing_the
  _gate_and_every_blocker`` pins the honest-refusal invariant directly: a model
  is adopted only if it beats the taxonomy *and* no blocker is active. Beating
  the taxonomy on the narrow title task cannot, by itself, retire the rule.

The learned model is deterministic and offline — a fixed seed, scikit-learn
only, no downloaded weights — and both of those are pinned at the source
(``test_out_of_fold_predictions_are_deterministic``,
``test_module_pulls_in_no_downloaded_weight_model``) because the committed
numbers mean nothing if a rerun cannot reproduce them.

The committed-evidence tests skip when the Task 12 tables have not been built,
and the end-to-end reproduction skips without the row-level parquets — the same
contract the other suites use, so the CI `check` job (which runs no stage) is
green on the committed artefacts alone.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import similarity as sim     # noqa: E402  (reused only for the standing guards)
import skill_model as sm     # noqa: E402
import skills as sk          # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMBER = REPO_ROOT / "members" / "ankit-google"
TABLES = MEMBER / "task-12-tables"
REPORT = MEMBER / "task-12-skill-model-report.json"
PROCESSED = REPO_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Fixtures. Synthetic unless a test is specifically about the committed run.
# ---------------------------------------------------------------------------


def committed(name: str) -> pd.DataFrame:
    """A committed Task 12 table, or skip if the build has not been run."""
    path = TABLES / f"{name}.csv"
    if not path.exists():                                 # pragma: no cover
        pytest.skip(f"task-12 tables not built: {path.name}")
    return pd.read_csv(path)


def load_report() -> dict:
    """The committed Task 12 report, or skip if the build has not been run."""
    if not REPORT.exists():                               # pragma: no cover
        pytest.skip(f"task-12 report not built: {REPORT.name}")
    return json.loads(REPORT.read_text())


def synthetic_dataset() -> sm.Dataset:
    """A small, separable title -> skill contest with a known literal baseline.

    Three skills over strongly associated titles: "data scientist" implies
    Python and SQL without naming either, so a literal title matcher must miss
    them and a learned model can recover them. Every template repeats enough
    that five folds each hold positives and `min_df=2` keeps the terms. The
    `y_taxonomy` here is the literal-match matrix — exactly what the taxonomy's
    title path is — so the synthetic contest mirrors the real one's shape.
    """
    vocab = ["Java", "Python", "SQL"]
    templates = [
        ("data scientist", {"Python", "SQL"}),
        ("senior data scientist", {"Python", "SQL"}),
        ("python developer", {"Python"}),
        ("python software engineer", {"Python"}),
        ("java engineer", {"Java"}),
        ("senior java engineer", {"Java"}),
        ("java backend developer", {"Java", "SQL"}),
        ("database administrator", {"SQL"}),
        ("sql data analyst", {"SQL"}),
    ]
    titles: list[str] = []
    y_rows: list[list[int]] = []
    tax_rows: list[list[int]] = []
    for _ in range(12):
        for title, present in templates:
            titles.append(title)
            y_rows.append([1 if v in present else 0 for v in vocab])
            low = title.lower()
            tax_rows.append([1 if v.lower() in low else 0 for v in vocab])
    y = np.array(y_rows, dtype=int)
    return sm.Dataset(
        job_ids=[f"j{i}" for i in range(len(titles))],
        titles=titles,
        vocab=vocab,
        y=y,
        y_taxonomy=np.array(tax_rows, dtype=int),
        postings_with_source=len(titles),
        postings_excluded=0,
    )


# ---------------------------------------------------------------------------
# A. Labels and the contest matrix — the provenance split is the experiment
# ---------------------------------------------------------------------------


def test_provenance_split_counts_source_plus_title_as_both():
    """"source+title" is a label *and* a taxonomy prediction, not a third class.

    The contest reads the labels off `source` rows and the baseline off `title`
    rows of the one committed frame. A posting the taxonomy found by both paths
    must appear in both sides, or the head-to-head is scored on mismatched sets.
    """
    long = pd.DataFrame({
        "job_id": [1, 1, 2, 2, 3],
        "skill": ["Python", "SQL", "Java", "Python", "SQL"],
        "provenance": ["source", "title", "source+title", "source", "title"],
    })
    src = sm.source_labels(long)
    tit = sm.taxonomy_title_labels(long)
    # job_id 2's "source+title" row belongs to both sides.
    assert src == {"1": ["Python"], "2": ["Java", "Python"]}
    assert tit == {"1": ["SQL"], "2": ["Java"], "3": ["SQL"]}
    # keys are strings, so they join the string job_ids from the clean frame.
    assert all(isinstance(k, str) for k in src)


def test_learnable_vocabulary_respects_min_support():
    """A skill too rare to appear in every fold is noise, and is dropped."""
    labels = {"a": ["X", "Y"], "b": ["X"], "c": ["X", "Y"], "d": ["Y"]}
    #  X in 3 postings, Y in 3 — both survive at 3; at 4 neither does.
    assert sm.learnable_vocabulary(labels, min_support=3) == ["X", "Y"]
    assert sm.learnable_vocabulary(labels, min_support=4) == []
    labels2 = {"a": ["X", "Y"], "b": ["X"], "c": ["Y"], "d": ["Y"]}
    #  X in 2, Y in 3 — at 3, only Y; the result is alphabetical.
    assert sm.learnable_vocabulary(labels2, min_support=3) == ["Y"]
    assert sm.learnable_vocabulary(labels2, min_support=2) == ["X", "Y"]


def test_binary_matrix_ignores_missing_postings_and_out_of_vocab_skills():
    """The matrix is postings x vocab; a rare skill or absent posting is a zero
    row/column, never an error and never a widened matrix."""
    labels = {"a": ["Python", "SQL"], "b": ["Java"], "c": ["Python", "Go"]}
    job_ids = ["a", "b", "c", "d"]           # d has no labels
    vocab = ["Java", "Python", "SQL"]        # Go is out of vocab
    y = sm._binary_matrix(labels, job_ids, vocab)
    expected = np.array([[0, 1, 1],          # a: Python, SQL
                         [1, 0, 0],          # b: Java
                         [0, 1, 0],          # c: Python (Go dropped)
                         [0, 0, 0]])         # d: nothing
    assert np.array_equal(y, expected)


def test_assemble_drops_postings_whose_only_skills_are_rare():
    """A posting whose skills all fall below `min_support` is excluded, not
    scored as an all-zero row that would flatter both extractors equally."""
    long = pd.DataFrame({
        "job_id": ["p1", "p1", "p2", "p2", "p3", "p4"],
        "skill": ["Python", "SQL", "Python", "SQL", "Python", "Rare"],
        "provenance": ["source"] * 6,
    })
    clean = pd.DataFrame({
        "job_id": ["p1", "p2", "p3", "p4"],
        "job_title": ["RAW1", "RAW2", "RAW3", "RAW4"],
        "job_title_clean": ["clean one", "clean two", "clean three", "clean four"],
    })
    ds = sm.assemble(clean, long, min_support=2)
    assert ds.vocab == ["Python", "SQL"]     # Rare (support 1) excluded
    assert ds.job_ids == ["p1", "p2", "p3"]  # p4's only skill is rare -> dropped
    assert ds.postings_with_source == 4
    assert ds.postings_excluded == 1
    assert ds.titles == ["clean one", "clean two", "clean three"]
    assert np.array_equal(ds.y, np.array([[1, 1], [1, 1], [1, 0]]))
    assert ds.y_taxonomy.shape == ds.y.shape


def test_assemble_prefers_the_cleaned_title_but_falls_back_to_raw():
    """The model reads whichever title text exists; an empty cleaned title must
    not blank out a posting that still has a raw one."""
    long = pd.DataFrame({
        "job_id": ["p1", "p2"], "skill": ["Python", "Python"],
        "provenance": ["source", "source"]})
    clean = pd.DataFrame({
        "job_id": ["p1", "p2"],
        "job_title": ["raw one", "raw two"],
        "job_title_clean": ["cleaned one", ""]})   # p2 has no cleaned title
    ds = sm.assemble(clean, long, min_support=1)
    assert ds.titles == ["cleaned one", "raw two"]


# ---------------------------------------------------------------------------
# B. Metrics — hand computed so a library change cannot move them unseen
# ---------------------------------------------------------------------------


def test_scores_match_a_hand_computed_matrix():
    """Every headline number is one of these six, so they are pinned by hand."""
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]])
    s = sm.scores(y_true, y_pred)
    # micro: TP=3, predicted=4, actual=5
    assert s["micro"] == {"precision": 0.75, "recall": 0.6, "f1": 0.6667}
    # macro over the three columns: (1, 0.6667, 0) averaged
    assert s["macro"] == {"precision": 0.6667, "recall": 0.5, "f1": 0.5556}
    assert s["label_instances"] == 5
    assert s["predicted_instances"] == 4
    assert s["true_positives"] == 3


def test_scores_are_rounded_to_the_pipeline_places():
    """4 dp, the standing rule against last-ULP float drift across BLAS builds."""
    y_true = np.array([[1, 1, 1, 0, 0, 0, 1]])
    y_pred = np.array([[1, 0, 0, 1, 1, 1, 1]])   # precision 2/5 = 0.4
    s = sm.scores(y_true, y_pred)
    for avg in ("micro", "macro"):
        for metric in ("precision", "recall", "f1"):
            value = s[avg][metric]
            assert value == round(value, sm.ROUND)


def test_title_literal_coverage_is_the_share_named_verbatim():
    """The literal ceiling any title matcher is bounded by — hand checked."""
    ds = sm.Dataset(
        job_ids=["a", "b"],
        titles=["python developer", "data scientist"],
        vocab=["Python", "SQL"],
        #  a: Python (named in "python developer")
        #  b: Python (NOT named in "data scientist") + SQL (not named)
        y=np.array([[1, 0], [1, 1]]),
        y_taxonomy=np.zeros((2, 2), dtype=int),
        postings_with_source=2, postings_excluded=0)
    # one of three positive labels is literally in its title -> 1/3
    assert sm.title_literal_coverage(ds) == pytest.approx(0.3333, abs=1e-4)


# ---------------------------------------------------------------------------
# C. The learned model — deterministic, offline, and better than literal
# ---------------------------------------------------------------------------


def test_out_of_fold_predictions_are_deterministic():
    """The committed numbers are reproducible only if this is bit-for-bit."""
    ds = synthetic_dataset()
    first = sm.out_of_fold_predictions(ds, seed=sm.SEED)
    second = sm.out_of_fold_predictions(ds, seed=sm.SEED)
    assert np.array_equal(first, second)
    assert first.shape == ds.y.shape


def test_the_learned_model_beats_a_literal_title_matcher():
    """The point of learning: recover skills the title implies but never names.

    On the synthetic contest the literal baseline can only find Python and SQL
    where the words appear; the learned model recovers them from "data
    scientist" too, so it must win on recall and on micro-F1.
    """
    ds = synthetic_dataset()
    learned = sm.scores(ds.y, sm.out_of_fold_predictions(ds, seed=sm.SEED))
    literal = sm.scores(ds.y, ds.y_taxonomy)
    assert learned["micro"]["recall"] > literal["micro"]["recall"]
    assert learned["micro"]["f1"] > literal["micro"]["f1"]


def test_module_pulls_in_no_downloaded_weight_model():
    """Offline by construction: scikit-learn is allowed, network weights are not.

    Unlike `similarity.py`'s no-scipy promise, this module is *meant* to use
    scikit-learn — that is the fine-tuned model. What it must never reach for is
    a downloaded-weight extractor (spaCy, sentence-transformers, a HF model),
    because that would make a rebuild depend on a fetch and break the seed's
    reproducibility guarantee. Checked on the source text, since pytest imports
    some of these through other suites.
    """
    source = Path(sm.__file__).read_text()
    for banned in ("spacy", "sentence_transformers", "transformers",
                   "torch", "huggingface", "requests", "urllib"):
        assert f"import {banned}" not in source, banned


def test_model_spec_names_the_seed_and_the_family():
    spec = sm.model_spec(seed=sm.SEED)
    assert spec["seed"] == sm.SEED
    assert "one-vs-rest" in spec["family"]
    assert "scikit-learn" in spec["library"]


# ---------------------------------------------------------------------------
# D. The gate — the honest-refusal invariant, pinned directly
# ---------------------------------------------------------------------------


def test_adoption_blockers_read_the_data_not_a_decision():
    """Each blocker is a fact about this dataset, so it flips with the fact."""
    ds = synthetic_dataset()   # vocab of 3
    #  collapse (3 < 176), no text (0 of 848), circularity always
    blocked = sm.adoption_blockers(ds, taxonomy_vocab_size=176,
                                   postings_with_text=0, postings_total=848)
    assert blocked["vocabulary_collapse"]["blocked"] is True
    assert blocked["label_circularity"]["blocked"] is True
    assert blocked["no_text_path_input"]["blocked"] is True
    #  full vocabulary and some text -> only circularity remains
    clear = sm.adoption_blockers(ds, taxonomy_vocab_size=3,
                                 postings_with_text=5, postings_total=10)
    assert clear["vocabulary_collapse"]["blocked"] is False
    assert clear["no_text_path_input"]["blocked"] is False
    assert clear["label_circularity"]["blocked"] is True   # never clears here


def test_adoption_requires_clearing_the_gate_and_every_blocker():
    """The honest-refusal invariant: winning is necessary, not sufficient.

    Adoption is `beats_micro AND no active blocker`. This sweeps the four
    corners so neither half can be dropped by a later refactor — a cleared gate
    with a live blocker must still refuse.
    """
    taxonomy = {"micro": {"f1": 0.10}, "macro": {"f1": 0.05}}
    for cleared in (True, False):
        for blocked in (True, False):
            learned = {"micro": {"f1": 0.70 if cleared else 0.01},
                       "macro": {"f1": 0.30}}
            verdict = sm.gate_verdict(learned, taxonomy,
                                      {"b": {"blocked": blocked}})
            assert verdict["gate_cleared"] is cleared
            assert verdict["adopted"] is (cleared and not blocked)


def test_gate_verdict_reports_the_margin_and_the_active_blockers():
    learned = {"micro": {"f1": 0.70}, "macro": {"f1": 0.30}}
    taxonomy = {"micro": {"f1": 0.10}, "macro": {"f1": 0.05}}
    blockers = {"x": {"blocked": True}, "y": {"blocked": False},
                "z": {"blocked": True}}
    verdict = sm.gate_verdict(learned, taxonomy, blockers)
    assert verdict["margin_micro_f1"] == pytest.approx(0.60)
    assert verdict["adoption_blockers"] == ["x", "z"]   # only the active ones
    assert verdict["adopted"] is False
    assert "skills.py" in verdict["retained"]


# ---------------------------------------------------------------------------
# E. The committed evidence — runs on the artefacts, so green in CI `check`
# ---------------------------------------------------------------------------


def test_committed_headline_shows_the_learned_model_winning():
    """The published contest: a large, positive micro-F1 margin for the model."""
    head = committed("contest-headline")
    micro_f1 = head[(head.averaging == "micro") & (head.metric == "f1")].iloc[0]
    assert micro_f1.learned_model > micro_f1.taxonomy_title_path
    assert micro_f1.margin == pytest.approx(
        micro_f1.learned_model - micro_f1.taxonomy_title_path, abs=1e-4)
    #  the finding, bracketed rather than hard-pinned to a real-data last digit
    assert micro_f1.margin > 0.5
    assert micro_f1.taxonomy_title_path < 0.05
    assert micro_f1.learned_model > 0.6


def test_committed_gate_is_cleared_but_the_taxonomy_is_retained():
    """Cleared gate, three blockers, and a "not adopted" decision — the refusal."""
    gate = committed("adoption-gate").set_index("item")
    assert gate.loc["gate", "status"] == "cleared"
    assert gate.loc["decision", "status"] == "not adopted"
    blockers = gate[gate.status == "blocks adoption"]
    assert len(blockers) == 3
    assert {"vocabulary_collapse", "label_circularity",
            "no_text_path_input"} <= set(blockers.index)


def test_report_verdict_is_internally_consistent():
    """The JSON verdict agrees with itself and with the committed headline."""
    report = load_report()
    verdict = report["verdict"]
    assert verdict["adopted"] is False
    assert verdict["gate_cleared"] is True
    assert verdict["beats_micro"] is True
    assert len(verdict["adoption_blockers"]) == 3
    assert "skills.py" in verdict["retained"]
    assert verdict["learned_micro_f1"] > verdict["taxonomy_micro_f1"]
    #  the report and the table are the same run
    head = committed("contest-headline")
    micro_f1 = head[(head.averaging == "micro") & (head.metric == "f1")].iloc[0]
    assert micro_f1.learned_model == pytest.approx(verdict["learned_micro_f1"])
    assert micro_f1.taxonomy_title_path == pytest.approx(
        verdict["taxonomy_micro_f1"])


def test_per_skill_tables_share_the_contest_vocabulary():
    """Both extractors are scored on the identical skills, so the tables must
    list the same ones — and there must be the collapsed 32, not the 176."""
    learned = committed("per-skill-learned")
    taxonomy = committed("per-skill-taxonomy")
    assert set(learned.skill) == set(taxonomy.skill)
    summary = committed("dataset-summary").set_index("key")
    assert len(learned) == int(summary.loc["skills_learnable", "value"])


def test_dataset_summary_records_the_vocabulary_collapse():
    """The model learns far fewer skills than the taxonomy, and the counts
    render as integers — the str() cast that keeps "554" from becoming "554.0".

    Read with ``dtype=str`` so the assertion sees the on-disk text, not a value
    pandas has re-coerced back to float; the point of the cast is what a human
    reading the committed CSV sees.
    """
    path = TABLES / "dataset-summary.csv"
    if not path.exists():                                 # pragma: no cover
        pytest.skip("task-12 tables not built: dataset-summary.csv")
    summary = pd.read_csv(path, dtype=str).set_index("key")
    learnable = summary.loc["skills_learnable", "value"]
    in_taxonomy = summary.loc["skills_in_taxonomy", "value"]
    scored = summary.loc["postings_scored", "value"]
    #  integer-rendered: the fix against float coercion of a mixed column
    for value in (learnable, in_taxonomy, scored):
        assert "." not in value, value
    assert int(learnable) == 32
    assert int(in_taxonomy) == len(sk.SKILLS) == 176
    assert int(learnable) < int(in_taxonomy)             # the collapse itself
    #  the win is association, not literal matching
    assert float(summary.loc["title_literal_coverage", "value"]) < 0.5


# ---------------------------------------------------------------------------
# F. Standing guards, inherited — no forbidden or personal column ships
# ---------------------------------------------------------------------------


def test_no_forbidden_column_reaches_a_task_12_table():
    for path in sorted(TABLES.glob("*.csv")):
        assert sim.forbidden_columns(pd.read_csv(path, nrows=1)) == [], path.name


def test_no_personal_data_column_reaches_a_task_12_table():
    for path in sorted(TABLES.glob("*.csv")):
        assert sim.personal_data_columns_present(
            pd.read_csv(path, nrows=1)) == [], path.name


# ---------------------------------------------------------------------------
# G. End-to-end — the committed tables are what a fresh run produces
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def contest_result():
    """Run the contest on the real frames, or skip without row-level parquets."""
    clean_path = PROCESSED / "google" / "google_jobs_clean.parquet"
    long_path = PROCESSED / "google" / "google_skills_long.parquet"
    if not (clean_path.exists() and long_path.exists()):  # pragma: no cover
        pytest.skip("row-level google parquets not built")
    clean = pd.read_parquet(clean_path)
    long = pd.read_parquet(long_path)
    postings_with_text = int((clean.job_description.fillna("").str.len() > 0).sum())
    return sm.run_contest(
        clean, long,
        taxonomy_vocab_size=len(sk.SKILLS),
        postings_with_text=postings_with_text,
        postings_total=int(len(clean)))


def test_fresh_run_reproduces_the_committed_headline(contest_result):
    """A rebuild lands on the committed numbers — the reproducibility guarantee.

    This is the test that ties the committed tables to a deterministic rerun;
    it skips in the CI `check` job (no row-level data there) and runs in the
    weekly `verify` job and locally.
    """
    verdict = contest_result["verdict"]
    assert verdict["adopted"] is False
    assert verdict["gate_cleared"] is True
    head = committed("contest-headline")
    micro_f1 = head[(head.averaging == "micro") & (head.metric == "f1")].iloc[0]
    assert micro_f1.learned_model == pytest.approx(verdict["learned_micro_f1"])
    assert micro_f1.taxonomy_title_path == pytest.approx(
        verdict["taxonomy_micro_f1"])
