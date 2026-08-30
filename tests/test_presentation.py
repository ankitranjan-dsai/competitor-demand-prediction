"""Tests for the Task 10 presentation layer.

Task 09's Google report §13 made a prediction about this task:

    "Task 10 is the final presentation, and it is the first task in this
    project whose output is not checkable by a test."

Part of that is true, and it is not the interesting part. Delivery is not
checkable here — pace, tone, whether the refusal slide gets rushed when the
clock runs down — and nothing in this file pretends otherwise. But a slide is
a claim plus an attachment, and both halves are files in this repository: the
claim is a row of the Task 09 ledger, and the attachment is a committed table
or figure. That is checkable, it is checked below, and the prediction is
recorded as overturned in C9.

The traps below were live while `src/present.py` was being written, and they
are the reason this suite is worth reading rather than skimming:

* **the negation problem.** A qualifying clause earns its place by *naming*
  the forbidden reading in order to deny it — "never of all postings", "not
  headcount". So re-running the language gate over a published claim fires on
  the very words that satisfied it: eight of Google's shipped claims trip
  `unmeasured_construct` or `bare_share_of_all` when re-linted. The first
  build reported four such violations and every one was spurious. The fix is
  structural, not a suppression list: a ledger-bound bullet is exempt from the
  language lint, because that gate already ran on it once in Task 09 and the
  ledger records the verdict in `gate_lint`, which is checked instead.
  ``test_a_published_claim_is_exempt_from_the_language_gate`` and
  ``test_a_claim_whose_ledger_row_failed_the_language_gate_is_refused``.
* **a refusal with no clause rendered the word "nan".** Nine of the ten
  `tempting-*` rows were stopped by a pattern rather than by a verdict, so
  they carry no qualifying clause — there was never going to be a wording
  that saved them. The renderer falls back to the gate's own transcript.
  ``test_a_clauseless_refusal_falls_back_to_the_gate_transcript``.
* **narration about refusing is not a refusal.** The refusal slide is the
  first thing a presenter cuts under time pressure, and the first thing to go
  on it is the binding. A bullet that *says* the project declines to forecast
  is prose; only a bullet bound to a refused row quotes the sentence and names
  the gate. ``test_narration_about_refusing_does_not_count_as_a_refusal``.
* **`task_row_backed` failed the legal review for being legal.** Task 01 lands
  as a team document in `docs/` and produces no member report, and the first
  version of the check demanded both. A rule that marks completed work
  incomplete gets ignored, and an ignored audit is worse than no audit.
  ``test_a_docs_only_task_row_is_backed``.
* **a claim is longer than anyone can say in one breath.** `skilltrend-looker`
  runs to 398 characters and may not be shortened, because it is reproduced
  character for character. So length is not a violation for a bound bullet —
  it is a delivery instruction, capped at one per slide.
  ``test_a_long_claim_is_a_delivery_instruction_not_a_violation``.

And the rules the task turns on:

* **four bullet kinds, each with its own binding.** A claim comes from a
  published ledger row, a refusal from a refused one, a fact from a resolver
  that reads the repository, and narration may carry no numeral at all.
* **the deck and the mentor question bank are linted by one function**,
  because a spoken answer is the same object as a bullet.
* **the audit's centrepiece is staleness**, not spelling: the newest task
  row's quoted suite size must equal the suite that exists now.

    python -m pytest tests/test_presentation.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import insights as ins      # noqa: E402
import present as pr        # noqa: E402

MEMBER = REPO_ROOT / "members" / "ankit-google"
TASK09 = MEMBER / "task-09-tables"
TASK10 = MEMBER / "task-10-tables"

needs_ledger = pytest.mark.skipif(
    not (TASK09 / "claim-ledger.csv").exists(),
    reason="Google Task 09 claim ledger not present in this checkout",
)
needs_deck = pytest.mark.skipif(
    not TASK10.is_dir(),
    reason="Task 10 build outputs not present in this checkout",
)


@pytest.fixture(scope="module")
def ledger() -> pd.DataFrame:
    return pd.read_csv(TASK09 / "claim-ledger.csv")


@pytest.fixture(scope="module")
def facts() -> dict:
    return pr.resolve_facts(REPO_ROOT, MEMBER, "google")


@pytest.fixture(scope="module")
def deck() -> tuple:
    return pr.deck_spec("google")


@pytest.fixture(scope="module")
def bank() -> tuple:
    return pr.qa_bank("google")


# ---------------------------------------------------------------------------
# A synthetic ledger, so a negative case can be exact
# ---------------------------------------------------------------------------
#
# The real ledger is 436 rows of evidence and every one of them is correct,
# which makes it useless for proving a rule fires. These helpers build the one
# broken row a rule is meant to catch, and nothing else.

LEDGER_COLUMNS = ("claim_id", "text", "clause", "status", "blocked_by",
                  "reason", "action", "depends_on", "gate_lint")


def row(claim_id: str = "c1", **over) -> dict:
    base = {
        "claim_id": claim_id,
        "text": "Postings mentioning Kubernetes rose between H1 and H2",
        "clause": "measured on postings, never on people",
        "status": ins.PUBLISHED,
        "blocked_by": "",
        "reason": "",
        "action": "",
        "depends_on": "",
        "gate_lint": True,
    }
    base.update(over)
    return base


def refused_row(claim_id: str = "r1", **over) -> dict:
    base = {"status": ins.REFUSED, "blocked_by": "identification",
            "clause": "", "reason": "volume direction is not identified",
            "action": "ask for the denominator instead"}
    base.update(over)
    return row(claim_id, **base)


def make_ledger(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows) or [row()], columns=list(LEDGER_COLUMNS))


def slide(**over) -> pr.Slide:
    base = {
        "number": 1,
        "section": "trends",
        "title": "What moved",
        "bullets": (pr.plain("Two skills moved far enough to be worth a slide."),),
        "notes": "Thirty seconds. Do not editorialise.",
    }
    base.update(over)
    return pr.Slide(**base)


def rules(violations) -> list:
    """The rule names a lint run fired, deduplicated and sorted."""
    if isinstance(violations, pd.DataFrame):
        return sorted(set(violations["rule"]))
    return sorted({v["rule"] for v in violations})


def lint_one(one_slide, led=None, fact_map=None, **kw) -> list:
    return pr.lint_slide(one_slide, led if led is not None else make_ledger(),
                         fact_map or {}, kw.pop("repo_root", REPO_ROOT),
                         kw.pop("member_root", MEMBER),
                         kw.pop("corrections", {"C8"}))


# ---------------------------------------------------------------------------
# A. The fact register
# ---------------------------------------------------------------------------


def test_every_registered_fact_resolves(facts):
    """A fact with no resolver is a number someone typed."""
    assert len(facts) == len(pr.FACTS)
    for key, resolved in facts.items():
        assert resolved["value"] is not None, key
        assert resolved["source"], key


def test_fact_keys_are_unique():
    keys = [f.key for f in pr.FACTS]
    assert len(keys) == len(set(keys))


def test_every_fact_says_where_it_came_from(facts):
    """A number on a slide is traceable to the table it was read from."""
    for key, resolved in facts.items():
        assert resolved["description"].strip(), key
        assert resolved["source"].strip(), key


def test_a_counted_fact_renders_with_thousands_separators():
    """A five-figure number read aloud from a slide needs its commas."""
    counted = pr.Fact("x", "d", "s", lambda *a: 12345)
    assert counted.render(counted.value(REPO_ROOT, MEMBER, "google")) == "12,345"


def test_a_measured_fact_renders_to_its_declared_precision():
    measured = pr.Fact("y", "d", "s", lambda *a: 27.4567, digits=1)
    assert measured.render(27.4567) == "27.5"
    assert measured.render(None) == "n/a"


def test_collected_suite_is_at_least_the_number_of_test_functions():
    """Parametrised cases collect more than once, never fewer."""
    assert pr.count_tests(REPO_ROOT) >= pr.count_test_functions(REPO_ROOT) > 0


def test_counts_read_the_repository_rather_than_a_constant():
    assert pr.count_tasks(REPO_ROOT) == 10
    assert pr.count_corrections(REPO_ROOT) >= 8
    assert pr.count_committed_tables(REPO_ROOT) > 0
    assert pr.count_committed_figures(REPO_ROOT) > 0


def test_task_count_moves_with_the_readme(tmp_path):
    (tmp_path / "README.md").write_text(
        "| 01 | a | b | ✅ |\n| 02 | a | b | ⬜ |\n", encoding="utf-8")
    assert pr.count_tasks(tmp_path) == 2


# ---------------------------------------------------------------------------
# B. The four bullet kinds, and what each one is bound to
# ---------------------------------------------------------------------------


def test_unknown_bullet_kind_is_rejected():
    with pytest.raises(ValueError):
        pr.Bullet("aside", text="something true")


@pytest.mark.parametrize("kind", pr.LEDGER_KINDS)
def test_a_ledger_bound_bullet_needs_a_claim_id(kind):
    with pytest.raises(ValueError):
        pr.Bullet(kind, text="we do not forecast")


@pytest.mark.parametrize("kind", ("fact", "plain"))
def test_an_authored_bullet_needs_text(kind):
    with pytest.raises(ValueError):
        pr.Bullet(kind, claim_id="share-google")


def test_a_claim_renders_the_ledger_sentence_character_for_character():
    led = make_ledger(row("c1"))
    rendered = pr.render_bullet(pr.claim("c1"), led, {})
    assert rendered == ins.sentence(led.iloc[0])
    assert "measured on postings, never on people" in rendered


def test_a_refusal_quotes_the_sentence_and_names_the_gate():
    led = make_ledger(refused_row("r1"))
    rendered = pr.render_bullet(pr.refusal("r1"), led, {})
    assert "Not available" in rendered
    assert "“Postings mentioning Kubernetes rose" in rendered
    assert "identification gate" in rendered
    assert "Ask for the denominator instead." in rendered


def test_a_clauseless_refusal_falls_back_to_the_gate_transcript():
    """Nine `tempting-*` rows carry no clause; none of them may render "nan"."""
    led = make_ledger(refused_row("r1", clause=float("nan"), blocked_by="lint",
                                  reason="matched forbidden pattern 'forecast'"))
    rendered = pr.render_bullet(pr.refusal("r1"), led, {})
    assert "nan" not in rendered.lower()
    assert "matched forbidden pattern" in rendered


def test_a_claim_id_matching_two_rows_is_a_lookup_error():
    led = make_ledger(row("c1"), row("c1"))
    with pytest.raises(LookupError):
        pr.render_bullet(pr.claim("c1"), led, {})


def test_a_fact_bullet_takes_its_numbers_from_the_register():
    resolved = {"postings": {"value": 10, "text": "10 postings",
                             "source": "x", "description": "y"}}
    assert pr.render_bullet(pr.fact("We kept {postings}."),
                            make_ledger(), resolved) == "We kept 10 postings."


def test_a_fact_bullet_naming_an_unknown_fact_raises():
    with pytest.raises(KeyError):
        pr.render_bullet(pr.fact("We kept {nonesuch}."), make_ledger(), {})


def test_structural_numerals_are_named_when_they_are_stripped():
    remainder, used = pr.strip_structural(
        "Task 06 §11 and C8 cover H2 2023 at h=1")
    assert not any(ch.isdigit() for ch in remainder)
    assert used == ["task_number", "correction_id", "section_mark",
                    "collection_year", "half", "horizon"]


# ---------------------------------------------------------------------------
# C. The slot order
# ---------------------------------------------------------------------------


def test_section_names_are_unique_and_ordered():
    names = [name for name, _ in pr.SECTIONS]
    assert len(names) == len(set(names))
    for required in pr.REQUIRED_SECTIONS:
        assert required in names


def test_the_shipped_deck_fills_the_standard_in_order(deck):
    canonical = [name for name, _ in pr.SECTIONS]
    seen = [s.section for s in deck]
    assert seen == sorted(seen, key=canonical.index)
    assert [s.number for s in deck] == list(range(1, len(deck) + 1))


# ---------------------------------------------------------------------------
# D. The linter — one test per rule it can fire
# ---------------------------------------------------------------------------


def test_a_clean_slide_fires_nothing():
    assert lint_one(slide()) == []


def test_slide_density_catches_an_empty_and_an_overloaded_slide():
    assert "slide_density" in rules(lint_one(slide(bullets=())))
    many = tuple(pr.plain(f"Line {'x' * i}") for i in range(pr.MAX_BULLETS + 1))
    assert "slide_density" in rules(lint_one(slide(bullets=many)))


def test_notes_present_fires_on_a_slide_with_no_speaker_notes():
    assert "notes_present" in rules(lint_one(slide(notes="   ")))


def test_narration_about_refusing_does_not_count_as_a_refusal():
    """The rule the refusal slide exists for: binding, not sentiment."""
    prose = slide(section="refusals", bullets=(
        pr.plain("We do not answer questions the data cannot answer."),))
    assert "refusals_intact" in rules(lint_one(prose))


def test_the_refusal_slide_passes_once_enough_bullets_are_bound():
    led = make_ledger(*(refused_row(f"r{i}") for i in range(pr.MIN_REFUSALS)))
    bound = slide(section="refusals",
                  bullets=tuple(pr.refusal(f"r{i}")
                                for i in range(pr.MIN_REFUSALS)))
    assert "refusals_intact" not in rules(lint_one(bound, led))


def test_asset_exists_fires_on_a_figure_that_was_never_built():
    assert "asset_exists" in rules(
        lint_one(slide(figure="task-10-figures/99-imaginary.png")))
    assert "asset_exists" in rules(
        lint_one(slide(table="docs/no-such-standard.md")))


def test_resolve_asset_reads_docs_from_the_repo_root(tmp_path):
    """Two roots: team documents, and the specialist's own outputs."""
    member = tmp_path / "members" / "someone"
    assert pr.resolve_asset("docs/corrections.md", tmp_path, member) == \
        tmp_path / "docs" / "corrections.md"
    assert pr.resolve_asset("task-10-tables/deck-lint.csv", tmp_path, member) == \
        member / "task-10-tables" / "deck-lint.csv"


def test_correction_exists_fires_on_a_correction_the_register_lacks():
    fired = lint_one(slide(carries=("C99",)), corrections={"C8"})
    assert "correction_exists" in rules(fired)


def test_known_corrections_reads_the_register_headings():
    known = pr.known_corrections(REPO_ROOT)
    assert {"C1", "C7", "C8"} <= known


def test_claim_exists_fires_on_an_id_that_is_not_in_the_ledger():
    fired = lint_one(slide(bullets=(pr.claim("no-such-claim"),)))
    assert rules(fired) == ["claim_exists"]


def test_claim_publishable_fires_when_a_refused_row_is_used_as_a_claim():
    led = make_ledger(refused_row("r1"))
    fired = lint_one(slide(bullets=(pr.claim("r1"),)), led)
    assert "claim_publishable" in rules(fired)


def test_clause_travels_fires_when_a_published_claim_carries_no_clause():
    led = make_ledger(row("c1", clause=""))
    assert "clause_travels" in rules(lint_one(slide(bullets=(pr.claim("c1"),)), led))


def test_correction_carried_fires_when_the_slide_drops_the_dependency():
    """C8 travels with a relative-share sentence, or the sentence does not."""
    led = make_ledger(row("c1", depends_on="C8"))
    bare = slide(bullets=(pr.claim("c1"),))
    assert "correction_carried" in rules(lint_one(bare, led))
    carried = slide(bullets=(pr.claim("c1"),), carries=("C8",))
    assert "correction_carried" not in rules(lint_one(carried, led))


def test_claim_gate_lint_fires_when_the_ledger_records_a_language_failure():
    led = make_ledger(row("c1", gate_lint=False))
    assert "claim_gate_lint" in rules(lint_one(slide(bullets=(pr.claim("c1"),)), led))


def test_refusal_refused_fires_when_a_published_row_is_dressed_as_a_refusal():
    led = make_ledger(row("c1"))
    fired = lint_one(slide(section="refusals",
                           bullets=(pr.refusal("c1"),)), led)
    assert "refusal_refused" in rules(fired)


def test_refusal_attributed_fires_when_no_gate_is_named():
    led = make_ledger(refused_row("r1", blocked_by=""))
    fired = lint_one(slide(bullets=(pr.refusal("r1"),)), led)
    assert "refusal_attributed" in rules(fired)


def test_fact_resolves_fires_on_a_placeholder_with_no_resolver():
    fired = lint_one(slide(bullets=(pr.fact("We kept {nonesuch}."),)))
    assert "fact_resolves" in rules(fired)


def test_fact_numerals_bound_fires_on_a_number_typed_into_a_template():
    resolved = {"postings": {"value": 1, "text": "1 posting",
                             "source": "x", "description": "y"}}
    fired = lint_one(slide(bullets=(pr.fact("We kept {postings} of 1234."),)),
                     fact_map=resolved)
    assert "fact_numerals_bound" in rules(fired)


def test_plain_numerals_bound_fires_on_narration_carrying_a_measurement():
    fired = lint_one(slide(bullets=(pr.plain("Hiring rose 42 percent."),)))
    assert "plain_numerals_bound" in rules(fired)


def test_plain_narration_may_carry_a_structural_numeral():
    ok = slide(bullets=(pr.plain("Task 06 §11 hands this deck one sentence, "
                                 "and C8 travels with it."),))
    assert lint_one(ok) == []


def test_language_clean_fires_on_narration_the_gate_forbids():
    fired = lint_one(slide(bullets=(pr.plain("We forecast next quarter hiring."),)))
    assert "language_clean" in rules(fired)


def test_bullet_speakable_fires_on_narration_nobody_can_say():
    long_line = "A sentence about hiring that keeps going. " * 12
    assert len(long_line) > pr.SPOKEN_MAX
    assert "bullet_speakable" in rules(
        lint_one(slide(bullets=(pr.plain(long_line),))))


def test_a_long_claim_is_a_delivery_instruction_not_a_violation():
    """A claim is reproduced verbatim, so it cannot be cut to fit a breath."""
    long_text = "Postings mentioning Kubernetes rose between H1 and H2, " * 8
    led = make_ledger(row("c1", text=long_text))
    fired = rules(lint_one(slide(bullets=(pr.claim("c1"),)), led))
    assert "bullet_speakable" not in fired
    assert "long_claims" not in fired


def test_long_claims_fires_on_a_second_unreadable_bullet():
    long_text = "Postings mentioning Kubernetes rose between H1 and H2, " * 8
    led = make_ledger(row("c1", text=long_text), row("c2", text=long_text))
    fired = lint_one(slide(bullets=(pr.claim("c1"), pr.claim("c2"))), led)
    assert "long_claims" in rules(fired)


def test_a_published_claim_is_exempt_from_the_language_gate():
    """The negation problem: a clause names the reading it denies.

    `share-google` says the quiet part out loud — "never of all postings" —
    and that phrase is exactly what `unmeasured_construct` looks for. It
    cleared the gate in Task 09; re-running the gate here would refuse the
    project's own published sentence for the words that made it publishable.
    """
    clause = "of the shared publisher pool only, never of all postings"
    led = make_ledger(row("c1", text="Google's share of the pool rose",
                          clause=clause))
    assert ins.lint_text(ins.sentence(led.iloc[0])), \
        "this fixture only means something while the raw text still trips"
    assert lint_one(slide(bullets=(pr.claim("c1"),)), led) == []


def test_the_language_gate_still_catches_a_country_on_a_slide():
    """The prohibition with no regex of its own, and the one easiest to lose.

    `lint_text` takes the country list as an argument, so a caller that omits
    it drops a ninth of the language gate while every visible check still
    passes. The deck linter reads the vocabulary from the same committed panel
    check Task 05 wrote, so the rule travels to a specialist whose postings
    land in different countries.
    """
    named = slide(bullets=(pr.plain("Hiring in Germany looks different."),))
    assert "language_clean" in rules(lint_one(named))
    # ... and only because the vocabulary reached the gate.
    quiet = pr.lint_slide(named, make_ledger(), {}, REPO_ROOT, MEMBER,
                          {"C8"}, countries=())
    assert "language_clean" not in rules(quiet)


def test_the_exemption_does_not_reach_narration_beside_the_claim():
    led = make_ledger(row("c1"))
    fired = lint_one(slide(bullets=(pr.claim("c1"),
                                    pr.plain("The two are converging."))), led)
    assert "language_clean" in rules(fired)


def test_section_known_fires_on_a_slot_the_standard_does_not_define(facts, ledger):
    bad = (slide(section="appendix"),)
    assert "section_known" in rules(pr.lint_deck(bad, ledger, facts,
                                                 REPO_ROOT, MEMBER))


def test_section_order_fires_when_the_storyline_is_shuffled(facts, ledger):
    shuffled = (slide(number=1, section="refusals",
                      bullets=(pr.refusal("tempting-forecast"),
                               pr.refusal("tempting-level"),
                               pr.refusal("tempting-country"),
                               pr.refusal("tempting-seasonal"))),
                slide(number=2, section="legal"))
    assert "section_order" in rules(pr.lint_deck(shuffled, ledger, facts,
                                                 REPO_ROOT, MEMBER))


def test_section_required_fires_for_every_mandatory_slot(facts, ledger):
    fired = pr.lint_deck((slide(),), ledger, facts, REPO_ROOT, MEMBER)
    missing = set(fired[fired.rule == "section_required"].subject)
    assert missing == set(pr.REQUIRED_SECTIONS)


def test_slide_numbering_fires_on_a_gap(facts, ledger):
    gapped = (slide(number=1), slide(number=3))
    assert "slide_numbering" in rules(pr.lint_deck(gapped, ledger, facts,
                                                   REPO_ROOT, MEMBER))


# ---------------------------------------------------------------------------
# E. The mentor question bank
# ---------------------------------------------------------------------------


def test_a_question_needs_an_answer():
    with pytest.raises(ValueError):
        pr.Question("q-empty", 1, "Can you forecast?", ())


def test_qa_slide_exists_fires_when_the_answer_points_nowhere(facts, ledger, deck):
    bad = (pr.Question("q-x", 999, "Where does this live?",
                       (pr.plain("On a slide that does not exist."),)),)
    assert "qa_slide_exists" in rules(pr.lint_qa_bank(bad, deck, ledger, facts))


def test_qa_is_a_question_fires_on_a_statement(facts, ledger, deck):
    bad = (pr.Question("q-x", 1, "Tell me about the denominator.",
                       (pr.plain("It is the shared publisher pool."),)),)
    assert "qa_is_a_question" in rules(pr.lint_qa_bank(bad, deck, ledger, facts))


def test_qa_covers_required_fires_for_an_unbacked_mandatory_section(
        facts, ledger, deck):
    thin = (pr.Question("q-x", 1, "What is this deck?",
                        (pr.plain("A summary of the work."),)),)
    fired = pr.lint_qa_bank(thin, deck, ledger, facts)
    assert set(fired[fired.rule == "qa_covers_required"].subject) <= \
        set(pr.REQUIRED_SECTIONS)
    assert len(fired[fired.rule == "qa_covers_required"]) >= 4


def test_qa_ids_unique_fires_on_a_duplicated_id(facts, ledger, deck):
    twice = pr.Question("q-x", 1, "Same id twice?",
                        (pr.plain("Yes, and that is the bug."),))
    fired = pr.lint_qa_bank((twice, twice), deck, ledger, facts)
    assert "qa_ids_unique" in rules(fired)


def test_the_bank_and_the_deck_are_checked_by_one_function(facts, ledger, deck):
    """The same broken bullet fires the same rule in both places."""
    broken = (pr.plain("We forecast next quarter hiring."),)
    in_deck = rules(pr.lint_deck((slide(bullets=broken),), ledger, facts,
                                 REPO_ROOT, MEMBER))
    in_bank = rules(pr.lint_qa_bank(
        (pr.Question("q-x", 1, "Can you forecast?", broken),),
        deck, ledger, facts))
    assert "language_clean" in in_deck
    assert "language_clean" in in_bank


# ---------------------------------------------------------------------------
# F. The workspace audit
# ---------------------------------------------------------------------------


def fake_repo(tmp_path: Path, table: str, *, member: str = "ankit-google",
              deliverables=("docs/task-01-legal.md",)) -> Path:
    (tmp_path / "README.md").write_text(table, encoding="utf-8")
    for relative in deliverables:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# a deliverable\n", encoding="utf-8")
    (tmp_path / "members" / member).mkdir(parents=True, exist_ok=True)
    (tmp_path / "members" / member / "README.md").write_text("# me\n",
                                                            encoding="utf-8")
    return tmp_path


def audit_for(check: str, table: pd.DataFrame) -> pd.DataFrame:
    return table[table.check == check]


@pytest.fixture(autouse=True)
def _no_nested_collection(monkeypatch):
    """Keep synthetic audits from shelling out to pytest inside a pytest run."""
    real = pr.count_tests

    def stub(repo_root: Path = pr.REPO_ROOT) -> int:
        return real(repo_root) if Path(repo_root) == REPO_ROOT else 12

    monkeypatch.setattr(pr, "count_tests", stub)


def test_a_docs_only_task_row_is_backed(tmp_path):
    """The legal review produces no member report, and is complete anyway."""
    repo = fake_repo(tmp_path,
                     "| 01 | Legal | list | ✅ Drafted → `docs/task-01-legal.md` |\n")
    result = pr.workspace_audit(repo, repo / "members" / "ankit-google")
    backed = audit_for("task_row_backed", result)
    assert list(backed.status) == ["pass"]


def test_a_done_row_naming_nothing_that_exists_fails(tmp_path):
    repo = fake_repo(tmp_path,
                     "| 02 | Collection | data | ✅ done → `docs/ghost.md` |\n",
                     deliverables=())
    result = pr.workspace_audit(repo, repo / "members" / "ankit-google")
    assert list(audit_for("task_row_backed", result).status) == ["fail"]
    assert list(audit_for("task_row_path", result).status) == ["fail"]


def test_an_open_row_with_a_delivered_report_fails(tmp_path):
    repo = fake_repo(tmp_path, "| 03 | Cleaning | tables | ⬜ |\n",
                     deliverables=())
    member = repo / "members" / "ankit-google"
    (member / "task-03-cleaning-report.md").write_text("# done\n",
                                                       encoding="utf-8")
    result = pr.workspace_audit(repo, member)
    assert list(audit_for("task_row_backed", result).status) == ["fail"]


def test_the_newest_done_row_is_the_one_checked_for_staleness(tmp_path):
    """An old row is a snapshot; the newest one describes the repo as it is."""
    repo = fake_repo(tmp_path,
                     "| 01 | Legal | list | ✅ → `docs/task-01-legal.md` (3 tests) |\n"
                     "| 02 | Insight | report | ✅ → `docs/task-01-legal.md` (12 tests) |\n")
    result = pr.workspace_audit(repo, repo / "members" / "ankit-google")
    stale = audit_for("suite_size_current", result)
    assert list(stale.status) == ["pass"]
    assert "task 02" in list(stale.subject)


def test_a_stale_suite_size_fails(tmp_path):
    repo = fake_repo(tmp_path,
                     "| 09 | Insight | report | ✅ → `docs/task-01-legal.md` (557 tests) |\n")
    result = pr.workspace_audit(repo, repo / "members" / "ankit-google")
    stale = audit_for("suite_size_current", result)
    assert list(stale.status) == ["fail"]
    assert "557" in stale.iloc[0].detail and "12" in stale.iloc[0].detail


def test_a_newest_row_quoting_no_suite_size_fails(tmp_path):
    repo = fake_repo(tmp_path,
                     "| 09 | Insight | report | ✅ → `docs/task-01-legal.md` |\n")
    result = pr.workspace_audit(repo, repo / "members" / "ankit-google")
    assert list(audit_for("suite_size_current", result).status) == ["fail"]


def test_a_broken_relative_link_fails(tmp_path, monkeypatch):
    repo = fake_repo(tmp_path,
                     "| 01 | Legal | list | ✅ → `docs/task-01-legal.md` |\n")
    (repo / "docs" / "task-01-legal.md").write_text(
        "See [the plan](./no-such-plan.md).\n", encoding="utf-8")
    monkeypatch.setattr(pr, "tracked_files",
                        lambda root=pr.REPO_ROOT: ["docs/task-01-legal.md"])
    result = pr.workspace_audit(repo, repo / "members" / "ankit-google")
    assert "fail" in list(audit_for("link_resolves", result).status)


def test_a_member_folder_without_a_readme_fails(tmp_path):
    repo = fake_repo(tmp_path,
                     "| 01 | Legal | list | ✅ → `docs/task-01-legal.md` |\n")
    (repo / "members" / "nobody").mkdir()
    result = pr.workspace_audit(repo, repo / "members" / "ankit-google")
    assert sorted(audit_for("member_readme", result).status) == ["fail", "pass"]


def test_a_placeholder_only_directory_is_not_a_used_directory(tmp_path):
    repo = fake_repo(tmp_path,
                     "| 01 | Legal | list | ✅ → `docs/task-01-legal.md` |\n")
    for name in pr.WORKING_DIRS:
        (repo / name).mkdir()
    (repo / pr.WORKING_DIRS[0] / "TEMPLATE.md").write_text("t\n", encoding="utf-8")
    (repo / pr.WORKING_DIRS[1] / "week-01.md").write_text("w\n", encoding="utf-8")
    result = audit_for("working_dir_used",
                       pr.workspace_audit(repo, repo / "members" / "ankit-google"))
    by_name = dict(zip(result.subject, result.status))
    assert by_name[pr.WORKING_DIRS[0]] == "fail"
    assert by_name[pr.WORKING_DIRS[1]] == "pass"
    assert by_name[pr.WORKING_DIRS[2]] == "fail"


def test_a_committed_env_file_fails(tmp_path, monkeypatch):
    repo = fake_repo(tmp_path,
                     "| 01 | Legal | list | ✅ → `docs/task-01-legal.md` |\n")
    monkeypatch.setattr(pr, "tracked_files",
                        lambda root=pr.REPO_ROOT: [".env", "data/postings.csv"])
    result = pr.workspace_audit(repo, repo / "members" / "ankit-google")
    assert list(audit_for("env_untracked", result).status) == ["fail"]
    assert list(audit_for("row_data_untracked", result).status) == ["fail"]


def test_row_level_data_under_data_may_keep_its_placeholders(tmp_path,
                                                             monkeypatch):
    repo = fake_repo(tmp_path,
                     "| 01 | Legal | list | ✅ → `docs/task-01-legal.md` |\n")
    (repo / "data" / "raw").mkdir(parents=True)
    (repo / "data" / "raw" / ".gitkeep").write_text("", encoding="utf-8")
    (repo / "data" / "README.md").write_text("# what lives here\n",
                                             encoding="utf-8")
    monkeypatch.setattr(pr, "tracked_files",
                        lambda root=pr.REPO_ROOT: ["data/raw/.gitkeep",
                                                   "data/README.md"])
    result = pr.workspace_audit(repo, repo / "members" / "ankit-google")
    assert list(audit_for("row_data_untracked", result).status) == ["pass"]


def test_the_audit_runs_the_standing_privacy_check_over_committed_tables(
        tmp_path):
    repo = fake_repo(tmp_path,
                     "| 01 | Legal | list | ✅ → `docs/task-01-legal.md` |\n")
    tables = repo / "members" / "ankit-google" / "task-01-tables"
    tables.mkdir(parents=True)
    (tables / "leak.csv").write_text("candidate_name,count\na,1\n",
                                     encoding="utf-8")
    result = pr.workspace_audit(repo, repo / "members" / "ankit-google")
    check = audit_for("personal_data_columns_present", result)
    assert list(check.status) == ["fail"]
    assert "candidate_name" in check.iloc[0].detail


# ---------------------------------------------------------------------------
# G. The deck as shipped
# ---------------------------------------------------------------------------


@needs_ledger
def test_the_shipped_deck_lints_clean(deck, ledger, facts):
    fired = pr.lint_deck(deck, ledger, facts, REPO_ROOT, MEMBER)
    assert fired.empty, fired.to_string()


@needs_ledger
def test_the_shipped_question_bank_lints_clean(bank, deck, ledger, facts):
    fired = pr.lint_qa_bank(bank, deck, ledger, facts)
    assert fired.empty, fired.to_string()


@needs_ledger
def test_the_deck_spends_only_claims_the_ledger_published(deck, ledger):
    by_id = ledger.set_index("claim_id")
    for one in deck:
        for bullet in one.bullets:
            if bullet.kind == "claim":
                assert by_id.loc[bullet.claim_id, "status"] in pr.PUBLISHABLE
            elif bullet.kind == "refusal":
                assert by_id.loc[bullet.claim_id, "status"] == ins.REFUSED


@needs_ledger
def test_the_refusal_slide_survives_being_cut_down(deck):
    refusals = [s for s in deck if s.section == "refusals"]
    assert len(refusals) == 1
    bound = [b for b in refusals[0].bullets if b.kind == "refusal"]
    assert len(bound) >= pr.MIN_REFUSALS


@needs_ledger
def test_at_most_one_bullet_a_slide_is_read_rather_than_recited(deck, ledger,
                                                               facts):
    for one in deck:
        long = [b for b in one.bullets
                if len(pr._safe_render(b, ledger, facts)) > pr.SPOKEN_MAX]
        assert len(long) <= 1, f"slide {one.number}"


@needs_ledger
def test_every_mentor_answer_that_refuses_is_bound_to_a_refused_row(bank,
                                                                    ledger):
    by_id = ledger.set_index("claim_id")
    refusing = [q for q in bank
                if any(b.kind == "refusal" for b in q.answer)]
    assert len(refusing) >= 8
    for question in refusing:
        for bullet in question.answer:
            if bullet.kind == "refusal":
                assert by_id.loc[bullet.claim_id, "status"] == ins.REFUSED


@needs_deck
def test_every_bound_sentence_reached_the_committed_deck(deck, ledger, facts):
    """The clause travelled all the way to the file a mentor reads."""
    shipped = (MEMBER / "task-10-slides.md").read_text(encoding="utf-8")
    for one in deck:
        for bullet in one.bullets:
            if bullet.kind in pr.LEDGER_KINDS:
                assert pr.render_bullet(bullet, ledger, facts) in shipped


@needs_deck
def test_no_committed_task_10_table_names_a_person():
    """The standing check from Task 01, re-run on this task's own output."""
    for table in sorted(TASK10.glob("*.csv")):
        found = ins.personal_data_columns_present(pd.read_csv(table, nrows=1))
        assert found == [], f"{table.name}: {found}"


@needs_deck
def test_the_committed_lint_tables_are_empty():
    for name in ("deck-lint.csv", "qa-lint.csv"):
        assert len(pd.read_csv(TASK10 / name)) == 0, name


@needs_deck
def test_every_prohibition_has_a_banked_refusal():
    """Nine prohibitions, nine prepared answers, matched on the ledger.

    Nine and not eight: `country_split` is the one rule with no regex of its
    own, because "do not name a country" is a list that comes from the
    committed panel check rather than from a pattern anyone can write down.
    That is also why it is the rule a caller drops by accident — see
    ``test_the_language_gate_still_catches_a_country_on_a_slide``.
    """
    coverage = pd.read_csv(TASK10 / "refusal-coverage.csv")
    assert len(coverage) == len(ins.PROHIBITED_PATTERNS) + 1
    assert "country_split" in set(coverage.rule)
    assert coverage.banked.all(), \
        coverage[~coverage.banked][["rule", "answered_by"]].to_string()


@needs_deck
def test_the_shipped_deck_quotes_the_repository_as_it_stands(facts):
    """The staleness rule, turned on the deck itself.

    The audit reports a stale README; this fails outright, because a slide is
    read aloud to a mentor and "that number was true when I typed it" is not
    an answer. The deck also counts its own tables and figures, so a one-pass
    build is stale the moment it finishes — which is why the runner iterates
    to a fixpoint. Both failure modes land here.
    """
    import json
    report = json.loads(
        (MEMBER / "task-10-presentation-report.json").read_text(encoding="utf-8"))
    drifted = {key: (value, facts[key]["value"])
               for key, value in report["facts"].items()
               if facts[key]["value"] != value}
    assert not drifted, (
        f"the deck quotes {drifted}, written as (on the slide, in the repo) — "
        "re-run `python src/build_presentation.py`")


@needs_ledger
def test_every_link_the_deck_writes_resolves(ledger, facts, deck, bank):
    """The audit reads the deck it *finds*, not the deck about to be written.

    `link_resolves` walks the tracked markdown on disk, and the runner emits
    the deck after it. So a link broken inside ``deck_markdown`` ships once
    and is reported on the next build — which is how the deck went out quoting
    `task-10-final-presentation-standard.md`, a filename that has never
    existed, under an audit line reading "all checks pass". The emitter now
    re-runs the audit after writing the deck; this test is the half that fails
    before anything is committed.
    """
    markdown = pr.deck_markdown(deck, bank, ledger, facts, "google")
    broken = sorted(
        target for target in
        (t.split("#")[0].strip() for t in pr.MD_LINK.findall(markdown))
        if target and not target.startswith(("http", "mailto:"))
        and not (MEMBER / target).exists()
    )
    assert not broken, f"the deck points at {broken}"
