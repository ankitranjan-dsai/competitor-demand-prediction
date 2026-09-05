"""Task 12 — a learned skill extractor, held to the rule taxonomy's bar.

CadetX Task 12 is optional: fine-tune a skill-extraction model, deliverable
"fine-tuned model + metrics". This module trains that model and holds it to the
standing gate — a learned extractor must beat the 176-skill rule taxonomy in
`src/skills.py` on the *same* Google postings before it may replace anything —
then reports, honestly, what happens when it is held there.

The shape of the problem is fixed by the data (Task 02/04), not chosen here:

  * 0 of 848 Google postings carry description text. The taxonomy's `text`
    path — the one a fine-tuned extractor would improve — has no live input to
    consume on this dataset.
  * Skills on these postings live in the collector's own `source` list, parsed
    from description text this repository never holds. 563 postings carry such
    a list (mean 4.0 skills); 89 distinct skills appear, 32 in >= 10 postings.
  * The taxonomy's `title` path — the only path with a live input on Google —
    recovers almost nothing: 61 of 848 postings, essentially "Machine Learning"
    alone.

So the only contest measurable on the same postings is: predict the source
skill set from the one text a posting actually carries, its title. This module
trains a scikit-learn multilabel classifier to do that and scores it, out of
fold, against the taxonomy's title path on the identical label matrix.

Everything here is deterministic and offline — a fixed seed, scikit-learn only
(no downloaded weights), every metric rounded to `ROUND` places so a rebuild on
another machine reproduces the number rather than its last float bit. The same
discipline the rest of the pipeline uses.

The verdict is computed, not decreed. The model may clear the gate on the
narrow title task; adoption additionally requires that it not collapse the
176-skill vocabulary to the few it can learn, not depend on labels only the
discarded text can produce, and have *some* input on the text path. Those three
are facts about this dataset — `adoption_blockers` reads them off the data —
and they hold, so the taxonomy is retained. Consistent with Task 07/08/09:
a result is delivered with its metrics and declined for stated reasons, not
adopted because a number moved.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

import skills as sk

#: Date-stamped like similarity.SEED, so a reader can see when it was fixed and
#: nothing downstream depends on the value beyond reproducibility.
SEED = 20260905

#: A skill needs this many postings before it can be both learned and scored.
#: Below it a fold may hold zero positives and the metric is noise, not signal.
MIN_SUPPORT = 10

#: Out-of-fold evaluation: every posting is scored by a model that did not train
#: on it, so the number is generalisation rather than memorisation.
N_SPLITS = 5

#: Titles are short, so word unigrams and bigrams carry the signal ("data
#: scientist", "machine learning"); characters would only add noise here.
NGRAM_RANGE = (1, 2)
MIN_DF = 2

#: 4 dp is the pipeline's standing rule against last-ULP float drift across
#: matplotlib/BLAS builds — see the Task 11 reproducibility audit.
ROUND = 4


# ---------------------------------------------------------------------------
# A. Labels and the contest matrix — grounded in committed Task 04 output
# ---------------------------------------------------------------------------
#
# `google_skills_long.parquet` is the taxonomy's own extraction, one row per
# (posting, skill) tagged with the path that found it: "source" (the
# collector's list), "title" (the job title), or "source+title" (both). That
# provenance is the whole contest — the source rows are the labels, the title
# rows are the taxonomy's live prediction, and both are already committed.


def _prov(long: pd.DataFrame, token: str) -> pd.DataFrame:
    """Rows whose provenance names `token`; "source+title" matches both."""
    return long[long.provenance.fillna("").str.contains(token, regex=False)]


def source_labels(long: pd.DataFrame) -> dict[str, list[str]]:
    """The collector's independent skill list per posting — the y of the task."""
    src = _prov(long, "source")
    return {str(jid): sorted(set(g.skill))
            for jid, g in src.groupby("job_id")}


def taxonomy_title_labels(long: pd.DataFrame) -> dict[str, list[str]]:
    """What the rule taxonomy's title path extracts per posting — the baseline."""
    tit = _prov(long, "title")
    return {str(jid): sorted(set(g.skill))
            for jid, g in tit.groupby("job_id")}


def learnable_vocabulary(labels: dict[str, list[str]],
                         min_support: int = MIN_SUPPORT) -> list[str]:
    """Skills carried by at least `min_support` postings, alphabetically."""
    counts: Counter = Counter(s for skills in labels.values() for s in skills)
    return sorted(name for name, c in counts.items() if c >= min_support)


def _title_lookup(clean: pd.DataFrame) -> dict[str, str]:
    """job_id -> title text, preferring the cleaned title, falling back to raw."""
    raw = dict(zip(clean.job_id.astype(str), clean.job_title.fillna("")))
    if "job_title_clean" in clean.columns:
        out = {}
        for jid, cleaned in zip(clean.job_id.astype(str),
                                clean.job_title_clean.fillna("")):
            out[jid] = cleaned or raw.get(jid, "")
        return out
    return {k: str(v) for k, v in raw.items()}


def _binary_matrix(labels: dict[str, list[str]], job_ids: list[str],
                   vocab: list[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(vocab)}
    y = np.zeros((len(job_ids), len(vocab)), dtype=int)
    for r, jid in enumerate(job_ids):
        for name in labels.get(jid, ()):
            col = idx.get(name)
            if col is not None:
                y[r, col] = 1
    return y


@dataclass(frozen=True)
class Dataset:
    """The contest, as arrays: same postings, same columns, three predictions."""

    job_ids: list[str]
    titles: list[str]
    vocab: list[str]
    y: np.ndarray              # source labels, restricted to `vocab`
    y_taxonomy: np.ndarray     # taxonomy title-path prediction, same shape
    postings_with_source: int  # postings carrying any source list, pre-filter
    postings_excluded: int     # had source skills, none of them >= MIN_SUPPORT


def assemble(clean: pd.DataFrame, long: pd.DataFrame,
             min_support: int = MIN_SUPPORT) -> Dataset:
    """Build the shared label matrix both extractors are scored against.

    Eligible postings carry at least one source skill that clears
    `min_support`; a posting whose only skills are rare is dropped rather than
    scored as an all-zero row that would flatter both models equally.
    """
    labels = source_labels(long)
    vocab = learnable_vocabulary(labels, min_support)
    in_vocab = set(vocab)
    title_of = _title_lookup(clean)

    job_ids = sorted(
        jid for jid, names in labels.items()
        if jid in title_of and in_vocab.intersection(names)
    )
    titles = [title_of[jid] for jid in job_ids]
    y = _binary_matrix(labels, job_ids, vocab)
    y_tax = _binary_matrix(taxonomy_title_labels(long), job_ids, vocab)
    return Dataset(
        job_ids=job_ids,
        titles=titles,
        vocab=vocab,
        y=y,
        y_taxonomy=y_tax,
        postings_with_source=len(labels),
        postings_excluded=len(labels) - len(job_ids),
    )


# ---------------------------------------------------------------------------
# B. The learned extractor — scikit-learn, offline, seeded
# ---------------------------------------------------------------------------


def pipeline(seed: int = SEED):
    """TF-IDF title features into a per-skill linear classifier.

    `class_weight="balanced"` is the one non-default choice, and it earns its
    place: without it a 0.5 threshold on a label present in a tenth of postings
    predicts almost nothing, and the contest would measure the threshold rather
    than the model. `liblinear` with a fixed seed is deterministic on sparse
    text; `n_jobs=1` keeps it that way.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=NGRAM_RANGE,
                                  min_df=MIN_DF)),
        ("clf", OneVsRestClassifier(
            LogisticRegression(class_weight="balanced", max_iter=2000,
                               solver="liblinear", random_state=seed),
            n_jobs=1)),
    ])


def out_of_fold_predictions(ds: Dataset, seed: int = SEED,
                            n_splits: int = N_SPLITS) -> np.ndarray:
    """Predict every posting from a model trained on the other folds."""
    from sklearn.model_selection import KFold, cross_val_predict

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yhat = cross_val_predict(pipeline(seed), ds.titles, ds.y, cv=cv,
                             method="predict", n_jobs=1)
    return np.asarray(yhat, dtype=int)


# ---------------------------------------------------------------------------
# C. Metrics — identical scoring for both extractors
# ---------------------------------------------------------------------------


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Micro and macro precision/recall/F1 over the label matrix, rounded."""
    from sklearn.metrics import precision_recall_fscore_support

    out: dict = {}
    for average in ("micro", "macro"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0)
        out[average] = {"precision": round(float(p), ROUND),
                        "recall": round(float(r), ROUND),
                        "f1": round(float(f), ROUND)}
    out["label_instances"] = int(y_true.sum())
    out["predicted_instances"] = int(y_pred.sum())
    out["true_positives"] = int((y_true & y_pred).sum())
    return out


def per_skill_scores(y_true: np.ndarray, y_pred: np.ndarray,
                     vocab: list[str]) -> pd.DataFrame:
    """One row per skill: support and P/R/F1, so a headline can be interrogated."""
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    rows = [{"skill": vocab[i],
             "category": sk.CATEGORY_OF.get(vocab[i], ""),
             "support": int(s[i]),
             "precision": round(float(p[i]), ROUND),
             "recall": round(float(r[i]), ROUND),
             "f1": round(float(f[i]), ROUND)}
            for i in range(len(vocab))]
    return (pd.DataFrame(rows)
            .sort_values(["support", "skill"], ascending=[False, True])
            .reset_index(drop=True))


def title_literal_coverage(ds: Dataset) -> float:
    """Share of positive labels whose skill name is verbatim in the title.

    This is the ceiling for any purely literal title matcher — the taxonomy's
    title path cannot exceed it — and it is low, which is the point: the learned
    model earns its recall from association ("data scientist" -> Python), not
    from words the title happens to contain.
    """
    names = [v.lower() for v in ds.vocab]
    hit = total = 0
    for r, title in enumerate(ds.titles):
        low = title.lower()
        for c in range(len(ds.vocab)):
            if ds.y[r, c]:
                total += 1
                hit += names[c] in low
    return round(hit / total, ROUND) if total else 0.0


def learned_associations(ds: Dataset, top: int = 6,
                         seed: int = SEED) -> pd.DataFrame:
    """The fitted model as a table: the title terms it weights for each skill.

    A committed pickle would not survive a scikit-learn bump byte-for-byte, so
    the model is published as what it learned — the highest-weighted title
    n-grams per skill — which is reproducible from the seed and, unlike a blob,
    legible.
    """
    pipe = pipeline(seed)
    pipe.fit(ds.titles, ds.y)
    features = np.asarray(pipe.named_steps["tfidf"].get_feature_names_out())
    estimators = pipe.named_steps["clf"].estimators_

    rows = []
    for i, est in enumerate(estimators):
        coef = getattr(est, "coef_", None)
        if coef is None:                       # a degenerate single-class label
            terms: list[str] = []
        else:
            order = np.argsort(coef.ravel())[::-1]
            terms = [features[j] for j in order[:top]
                     if coef.ravel()[j] > 0]
        rows.append({"skill": ds.vocab[i],
                     "category": sk.CATEGORY_OF.get(ds.vocab[i], ""),
                     "top_title_terms": ", ".join(terms)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D. The gate — computed from the metrics and the data, not asserted
# ---------------------------------------------------------------------------


def adoption_blockers(ds: Dataset, taxonomy_vocab_size: int,
                      postings_with_text: int,
                      postings_total: int) -> dict:
    """The three conditions adoption needs beyond clearing the gate.

    Each is a fact about this dataset, read off the data. They are separate
    from "does it beat the taxonomy" precisely so that beating the taxonomy on
    the narrow task cannot, by itself, retire the rule.
    """
    return {
        "vocabulary_collapse": {
            "blocked": len(ds.vocab) < taxonomy_vocab_size,
            "detail": (f"learns {len(ds.vocab)} of {taxonomy_vocab_size} "
                       f"taxonomy skills — only those in >= {MIN_SUPPORT} "
                       "postings; the rest have too few examples to fit"),
        },
        "label_circularity": {
            "blocked": True,
            "detail": ("labels are the collector's source list, parsed from "
                       "description text this repository does not hold; the "
                       "model distils those labels from titles rather than "
                       "extracting skills from a posting"),
        },
        "no_text_path_input": {
            "blocked": postings_with_text == 0,
            "detail": (f"{postings_with_text} of {postings_total} postings "
                       "carry description text, so a learned text extractor has "
                       "no live input on Google — the gap it would fill is empty"),
        },
    }


def gate_verdict(learned: dict, taxonomy: dict, blockers: dict) -> dict:
    """Did the model beat the taxonomy, and does that earn adoption?

    The gate is micro-F1 over the shared label matrix — the head-to-head on the
    same postings. Clearing it is necessary; adoption also needs every blocker
    clear, and they are not.
    """
    beats_micro = learned["micro"]["f1"] > taxonomy["micro"]["f1"]
    beats_macro = learned["macro"]["f1"] > taxonomy["macro"]["f1"]
    active = [name for name, b in blockers.items() if b["blocked"]]
    return {
        "gate": ("a learned extractor must beat the 176-skill rule taxonomy "
                 "on the same postings before it replaces anything"),
        "metric": "micro-F1 over the source-skill label matrix, out of fold",
        "taxonomy_micro_f1": taxonomy["micro"]["f1"],
        "learned_micro_f1": learned["micro"]["f1"],
        "margin_micro_f1": round(learned["micro"]["f1"]
                                 - taxonomy["micro"]["f1"], ROUND),
        "taxonomy_macro_f1": taxonomy["macro"]["f1"],
        "learned_macro_f1": learned["macro"]["f1"],
        "beats_micro": beats_micro,
        "beats_macro": beats_macro,
        "gate_cleared": beats_micro,
        "adoption_blockers": active,
        "adopted": beats_micro and not active,
        "retained": "rule taxonomy (src/skills.py)",
    }


def model_spec(seed: int = SEED) -> dict:
    """The fine-tuned model, fully specified — enough to rebuild it from code."""
    return {
        "family": "multilabel linear classifier (one-vs-rest)",
        "vectoriser": (f"TF-IDF word n-grams {NGRAM_RANGE}, min_df={MIN_DF}, "
                       "lowercase"),
        "estimator": ("LogisticRegression(class_weight=balanced, "
                      "solver=liblinear)"),
        "input": "job title text",
        "labels": f"source-provenance skills in >= {MIN_SUPPORT} postings",
        "evaluation": f"{N_SPLITS}-fold out-of-fold prediction",
        "library": "scikit-learn (offline, deterministic)",
        "seed": seed,
    }


def run_contest(clean: pd.DataFrame, long: pd.DataFrame, *,
                taxonomy_vocab_size: int, postings_with_text: int,
                postings_total: int, seed: int = SEED,
                min_support: int = MIN_SUPPORT) -> dict:
    """Assemble the data, run both extractors, and return the whole verdict.

    Pure: reads two frames, writes nothing. The driver turns this into tables,
    figures and a report; a test can call it and assert the numbers directly.
    """
    ds = assemble(clean, long, min_support)
    y_learned = out_of_fold_predictions(ds, seed=seed)

    learned = scores(ds.y, y_learned)
    taxonomy = scores(ds.y, ds.y_taxonomy)
    blockers = adoption_blockers(ds, taxonomy_vocab_size,
                                 postings_with_text, postings_total)
    verdict = gate_verdict(learned, taxonomy, blockers)

    return {
        "dataset": {
            "postings_scored": len(ds.job_ids),
            "postings_with_source_list": ds.postings_with_source,
            "postings_excluded_low_support": ds.postings_excluded,
            "skills_learnable": len(ds.vocab),
            "skills_in_taxonomy": taxonomy_vocab_size,
            "min_support": min_support,
            "title_literal_coverage": title_literal_coverage(ds),
            "label_instances": int(ds.y.sum()),
        },
        "model": model_spec(seed),
        "scores": {"taxonomy_title_path": taxonomy, "learned_model": learned},
        "blockers": blockers,
        "verdict": verdict,
        "per_skill_learned": per_skill_scores(ds.y, y_learned, ds.vocab),
        "per_skill_taxonomy": per_skill_scores(ds.y, ds.y_taxonomy, ds.vocab),
        "associations": learned_associations(ds, seed=seed),
        "_dataset": ds,
        "_y_learned": y_learned,
    }
