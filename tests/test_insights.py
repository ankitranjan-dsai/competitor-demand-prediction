"""Tests for the Task 09 insight layer.

Same intent as the other suites: every case is either a trap that was actually
hit while compiling Task 09's claims against the real committed tables, or a
rule the other three specialists will depend on, locked in so a later refactor
cannot quietly undo it.

Eight of these pin defects that were live while this module was being
written, and they are the reason the suite is worth reading:

* **`C`, `C++` and `C#` are one skill after a naive slug.** `re.sub` on
  non-alphanumerics collapsed all three to `c`, and the compiler raised on
  four duplicate claim ids. A claim id is the join key between the ledger, the
  refusal register and the report prose, so a collision is not cosmetic.
  ``test_slug_separates_the_c_family``.
* **`row.diff` is a bound method, not a column.** `diff`, `rank`, `count`,
  `size`, `mean` and `max` are all real column names in Task 06's tables *and*
  real `Series` attributes. Attribute access silently returned the method and
  127 distinctiveness claims failed the evidence gate quoting
  ``<bound method Series.diff>``. ``test_cell_reads_columns_that_shadow_series_attributes``.
* **`excludes_zero` alone published a refused construction.** Two of Task 08's
  15 trajectory pairs exclude zero while being *ineligible*; reading the
  interval without the eligibility flag republished exactly what Task 08 §8
  refused. Fixed by conjunction, where the weakest verdict wins.
  ``test_conjunction_inherits_the_weakest_verdict`` and
  ``test_no_trajectory_claim_publishes``.
* **`interval_sufficient` alone contradicted "max useful horizon 0".** It is
  True at h=1 and h=2 in Task 07's own table; the row that settles the
  question is `supported`, which is False on every forecast row.
  ``test_no_forecast_horizon_claim_publishes``.
* **the `refusal_rule` exemption blinded the linter.** A claim that *reports*
  a refusal must quote the forbidden words, so it is exempt from its own rule
  — but setting that field on generated claims that *assert* the forbidden
  thing exempted them from the gate meant to stop them.
  ``test_exemption_applies_only_to_the_claims_own_rule``.
* **a column named `candidates` trips the standing privacy check.** Task 01
  bans any column whose name contains `candidate`, because in a job-postings
  repository that word means a person. Task 09 borrowed it for a proposed
  sentence. The check stayed strict and the vocabulary moved.
  ``test_no_emitted_table_carries_a_candidate_column``.
* **a sentence contradicted the numbers it quoted.** Looker's direction word
  comes from the stratified verdict and its two shares from the pooled
  columns, so the claim asserted a fall while quoting a rise. Both halves are
  right — that is what a Simpson's reversal is — but unnamed it reads as a
  typo, and a reader who trusts the numbers has undone C2.
  ``test_a_reversed_skill_trend_names_the_reversal``.
* **a falsifier written for rank 1 was copied down all fifteen pairs.** It
  asked a rank-2 pair to "leave rank 1", a condition it cannot meet, and the
  matching action called every pair in the table the closest talent
  competitor — including the fifteenth.
  ``test_every_similarity_claim_is_falsified_by_its_own_rank``.

And the rules the task turns on:

* **an insight is a record, not a sentence.** Four gates, in order, and the
  ledger reports against the first one that fails.
* **every publishable claim carries a clause**, and `published` versus
  `published_qualified` grades the upstream verdict, not the caveat.
* **the linter must fire.** Ten sentences a reader would want are generated on
  purpose so each prohibited pattern is exercised against real text.
* **C8: a unanimity count is not a robustness statistic** when the number of
  tests moves with the threshold.

    python -m pytest tests/test_insights.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import insights as ins  # noqa: E402

MEMBER = REPO_ROOT / "members" / "ankit-google"
TABLES = MEMBER / "task-09-tables"
TASK06 = MEMBER / "task-06-tables"

needs_tables = pytest.mark.skipif(
    not TABLES.is_dir(),
    reason="Google Task 09 tables not present in this checkout",
)


@pytest.fixture(scope="module")
def ledger() -> pd.DataFrame:
    return pd.read_csv(TABLES / "claim-ledger.csv")


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_citation_round_trips():
    text = "task-06-tables/relative-share-verdict.csv#verdict@company=nvidia"
    cite = ins.parse_citation(text)
    assert cite.table == "task-06-tables/relative-share-verdict.csv"
    assert cite.column == "verdict"
    assert dict(cite.selector) == {"company": "nvidia"}


def test_citation_accepts_a_compound_selector():
    cite = ins.parse_citation(
        "task-07-tables/forecast.csv#supported@key=google,horizon=1")
    assert dict(cite.selector) == {"key": "google", "horizon": "1"}


def test_citation_rejects_a_string_with_no_column():
    with pytest.raises(ValueError):
        ins.parse_citation("task-06-tables/relative-share-verdict.csv@x=1")


def test_select_row_refuses_an_ambiguous_selector():
    """A citation must name exactly one cell, or it is not a citation."""
    frame = pd.DataFrame({"company": ["a", "a"], "value": [1, 2]})
    cite = ins.parse_citation("t.csv#value@company=a")
    with pytest.raises(LookupError):
        ins.select_row(frame, cite)


def test_select_row_refuses_a_selector_that_matches_nothing():
    frame = pd.DataFrame({"company": ["a"], "value": [1]})
    cite = ins.parse_citation("t.csv#value@company=b")
    with pytest.raises(LookupError):
        ins.select_row(frame, cite)


# ---------------------------------------------------------------------------
# Slugs — the C/C++/C# collision
# ---------------------------------------------------------------------------


def test_slug_separates_the_c_family():
    """The defect: `re.sub(r"[^a-z0-9]+", "-", ...)` maps all three to `c`."""
    slugs = {ins._slug(name) for name in ("C", "C++", "C#")}
    assert len(slugs) == 3, slugs


def test_slug_separates_dotted_names():
    assert ins._slug(".NET") != ins._slug("Node.js")
    assert ins._slug("Ai-Jobs.net") == ins._slug("Ai-Jobs.net")


@needs_tables
def test_every_claim_id_is_unique(ledger):
    assert not ledger.claim_id.duplicated().any()


# ---------------------------------------------------------------------------
# Cell access — columns that shadow Series attributes
# ---------------------------------------------------------------------------


def test_cell_reads_columns_that_shadow_series_attributes():
    """`row.diff` is `Series.diff`; `_cell(row, "diff")` is the number.

    `size` is the sharpest case in the set. It shadows to an `int` rather than
    a method, so attribute access returns 4 — the length of the Series — and
    nothing downstream would look wrong enough to catch.
    """
    row = pd.Series({"diff": 4.2, "rank": 1, "count": 7, "size": 3})
    for name, expected in (("diff", 4.2), ("rank", 1), ("count", 7),
                           ("size", 3)):
        assert ins._cell(row, name) == expected
    assert callable(row.diff) and callable(row.count)
    assert row.size == 4 and ins._cell(row, "size") == 3


def test_cell_raises_on_a_column_that_is_not_there():
    with pytest.raises(KeyError):
        ins._cell(pd.Series({"a": 1}), "b")


# ---------------------------------------------------------------------------
# Gate 1 — evidence
# ---------------------------------------------------------------------------


def test_values_agree_within_the_declared_tolerance():
    assert ins.values_agree(4.02, 4.0201)
    assert not ins.values_agree(4.02, 4.9)


def test_values_agree_handles_a_rounded_zero():
    """A share rounded to 0.0 must not fail against a tiny true value."""
    assert ins.values_agree(0.0, 0.0004)
    assert not ins.values_agree(0.0, 0.4)


def test_asserted_direction_reads_both_ways():
    assert ins.asserted_direction("Kubernetes is rising at Google") == "up"
    assert ins.asserted_direction("Looker is falling at Google") == "down"
    assert ins.asserted_direction("Google mentions Kubernetes") == ""


def test_direction_disagreement_fails_the_evidence_gate():
    """A number can match while the sentence points the other way."""
    claim = ins.Claim(
        claim_id="t", question="skill_demand", family="skill_trend",
        subject="google", measures="share of skilled postings",
        text="Looker is rising at Google",
        citation="task-05-tables/skill-stratified-verdicts.csv"
                 "#verdict@skill=Looker",
        value=None, verdict_source="structural:", clause="c",
        falsifier="f", action="a", audience="hr_talent", source_task=5)
    gate = ins.direction_agrees(claim, MEMBER)
    assert not gate.passed
    assert "records" in gate.detail


@needs_tables
def test_exactly_one_claim_fails_the_evidence_gate(ledger):
    """The tempting Looker sentence, and nothing else."""
    failures = ledger[~ledger.gate_evidence]
    assert len(failures) == 1
    assert failures.iloc[0].claim_id == "tempting-looker"


@needs_tables
def test_every_published_claim_resolves_to_a_committed_cell(ledger):
    published = ledger[ledger.status != ins.REFUSED]
    bindings = pd.read_csv(TABLES / "evidence-bindings.csv")
    bound = set(bindings[bindings.bound].claim_id)
    assert set(published.claim_id) <= bound


# ---------------------------------------------------------------------------
# Gate 2 — the linter
# ---------------------------------------------------------------------------


def test_every_prohibited_pattern_cites_the_task_that_earned_it():
    for rule, _pattern, source, why in ins.PROHIBITED_PATTERNS:
        assert source.startswith("task-"), rule
        assert why, rule


def test_the_forecast_rule_catches_a_forecast_sentence():
    hits = ins.lint_text("Google's postings will rise next quarter", ())
    assert "forecast" in {hit["rule"] for hit in hits}


@pytest.mark.parametrize("text", [
    "Google has more postings than Meta",
    "Google posted more roles than Snowflake",
    "Google is the largest hirer of the six",
    "Google hires more than Databricks",
])
def test_the_level_rule_catches_a_cross_company_count(text):
    """The noun list matters: a rule that catches "more jobs than" and not
    "more roles than" is one a writer clears by accident."""
    hits = ins.lint_text(text, ())
    assert "cross_company_level" in {hit["rule"] for hit in hits}


@pytest.mark.parametrize("text", [
    "Python appears in a 30% share of all postings",
    "Python appears in 30% of all postings",
    "Python appears in 30% of all its postings",
])
def test_the_share_rule_catches_share_of_all_postings(text):
    hits = ins.lint_text(text, ())
    assert "bare_share_of_all" in {hit["rule"] for hit in hits}


def test_the_share_rule_leaves_the_correct_denominator_alone():
    """`share_of_skilled` is the denominator Task 04 §7 requires."""
    hits = ins.lint_text(
        "Kubernetes appears in 12% of Google's skilled postings", ())
    assert "bare_share_of_all" not in {hit["rule"] for hit in hits}


def test_the_country_rule_is_built_from_the_committed_vocabulary():
    """The rule cannot be a hardcoded list; it reads Task 05's panel check."""
    vocabulary = ins.country_vocabulary(MEMBER)
    assert vocabulary, "no country vocabulary was recovered"
    hits = ins.lint_text(f"Google hires most in {vocabulary[0]}", vocabulary)
    assert "country_split" in {hit["rule"] for hit in hits}


def test_exemption_applies_only_to_the_claims_own_rule():
    """A claim reporting a refusal may quote its own forbidden words — only.

    The defect this pins: `refusal_rule` was set on generated claims that
    *assert* the forbidden thing, exempting them from the gate meant to stop
    them.
    """
    def claim(refusal_rule):
        return ins.Claim(
            claim_id="t", question="future_demand", family="tempting",
            subject="google", measures="none",
            text="Google's postings will rise, and the two stacks converge",
            citation="task-07-tables/forecast.csv#supported@key=google,"
                     "horizon=1",
            value=None, verdict_source="structural:", clause="c",
            falsifier="f", action="a", audience="exec", source_task=7,
            refusal_rule=refusal_rule)

    exempt_from_forecast = ins.lint_claim(claim("forecast"), ())
    assert not exempt_from_forecast.passed, (
        "the convergence rule must still fire")
    assert "convergence" in exempt_from_forecast.detail
    assert "forecast" not in exempt_from_forecast.detail


@needs_tables
def test_the_linter_actually_fires(ledger):
    """A gate nothing trips is an untested gate, not a clean corpus."""
    blocked = ledger[ledger.blocked_by == "lint"]
    assert len(blocked) == 9


@needs_tables
def test_every_prohibited_pattern_is_exercised():
    patterns = pd.read_csv(TABLES / "prohibited-patterns.csv")
    silent = patterns[(patterns.claims_caught == 0)
                      & (patterns.refusals_exempted == 0)]
    assert silent.empty, f"never exercised: {sorted(silent.rule)}"


# ---------------------------------------------------------------------------
# Gate 3 — inherited identification
# ---------------------------------------------------------------------------


def test_conjunction_inherits_the_weakest_verdict():
    assert ins.STATUS_STRENGTH[ins.REFUSED] < ins.STATUS_STRENGTH[ins.QUALIFIED]
    assert ins.STATUS_STRENGTH[ins.QUALIFIED] < ins.STATUS_STRENGTH[ins.PUBLISHED]


def test_a_verdict_map_never_promotes_an_unsupported_finding():
    for family, mapping in ins.VERDICT_MAP.items():
        for key, status in mapping.items():
            assert status in ins.STATUSES, (family, key)


@needs_tables
def test_no_trajectory_claim_publishes(ledger):
    """Task 08 §8 refused trajectory similarity; 2 of 15 pairs still exclude
    zero, so `excludes_zero` alone would republish it."""
    block = ledger[ledger.family == "trajectory"]
    assert len(block) == 15
    assert (block.status == ins.REFUSED).all()


@needs_tables
def test_no_forecast_horizon_claim_publishes(ledger):
    """Task 07's max useful horizon is 0, and `interval_sufficient` is True at
    h=1 and h=2 — so the flag that settles it is `supported`."""
    block = ledger[ledger.family == "forecast_horizon"]
    assert not block.empty
    assert (block.status == ins.REFUSED).all()


@needs_tables
def test_no_claim_asserts_a_cross_company_level(ledger):
    """The two volume claims that publish are *within-panel* readings, and
    each says so in its clause. What Task 06 §1.3 refuses is the comparison
    across companies, so the assertion is on the sentences, not the family."""
    published = ledger[ledger.status != ins.REFUSED]
    for text in published.text:
        hits = {hit["rule"] for hit in ins.lint_text(text, ())}
        assert "cross_company_level" not in hits, text
    volume = published[published.family == "volume"]
    assert (volume.status == ins.QUALIFIED).all()
    assert volume.clause.str.contains("panel").all()


@needs_tables
def test_identification_is_the_dominant_refusal(ledger):
    """The finding, not an accident: the sentences fail on what the data can
    carry, not on how they are worded."""
    refused = ledger[ledger.status == ins.REFUSED]
    assert (refused.blocked_by == "identification").mean() > 0.9


# ---------------------------------------------------------------------------
# Gate 4 — cross-task consistency
# ---------------------------------------------------------------------------


@needs_tables
def test_every_restated_quantity_is_registered():
    conflicts = pd.read_csv(TABLES / "cross-task-conflicts.csv")
    assert conflicts.conflict.any()
    assert (conflicts[conflicts.conflict].register == "C5").all()


@needs_tables
def test_no_claim_quotes_a_superseded_value(ledger):
    assert (ledger.gate_consistency).all()


# ---------------------------------------------------------------------------
# The compiler and its ledger
# ---------------------------------------------------------------------------


@needs_tables
def test_the_ledger_reports_against_the_first_failed_gate(ledger):
    refused = ledger[ledger.status == ins.REFUSED]
    assert set(refused.blocked_by) <= set(ins.GATE_ORDER)
    for _, row in refused.iterrows():
        gates = [row.gate_evidence, row.gate_lint,
                 row.gate_identification, row.gate_consistency]
        first = ins.GATE_ORDER[gates.index(False)]
        assert row.blocked_by == first, row.claim_id


@needs_tables
def test_every_publishable_claim_carries_a_clause(ledger):
    """The invariant, not a statistic: this evidence base yields no
    context-free sentence."""
    publishable = ledger[ledger.status != ins.REFUSED]
    assert publishable.clause.notna().all()


@needs_tables
def test_every_publishable_claim_carries_a_falsifier(ledger):
    publishable = ledger[ledger.status != ins.REFUSED]
    assert publishable.falsifier.notna().all()


def test_a_claim_without_a_clause_cannot_publish():
    claim = ins.Claim(
        claim_id="t", question="tech_stack", family="distinctiveness",
        subject="google", measures="share of skilled postings",
        text="Google mentions BigQuery in 12.00% of its skilled postings",
        citation="task-06-tables/company-feasibility-screen.csv"
                 "#postings@company=google",
        value=None, verdict_source="structural:", clause="",
        falsifier="f", action="a", audience="product", source_task=6)
    row = ins.compile_claims([claim], MEMBER).iloc[0]
    assert row.status == ins.REFUSED
    assert "reading clause" in row.reason


def test_an_unknown_question_is_rejected_at_construction():
    with pytest.raises(ValueError):
        ins.Claim(
            claim_id="t", question="revenue", family="f", subject="google",
            measures="m", text="t", citation="a.csv#b@c=d", value=None,
            verdict_source="structural:", clause="c", falsifier="f",
            action="a", audience="exec", source_task=6)


def test_an_unknown_audience_is_rejected_at_construction():
    with pytest.raises(ValueError):
        ins.Claim(
            claim_id="t", question="tech_stack", family="f", subject="google",
            measures="m", text="t", citation="a.csv#b@c=d", value=None,
            verdict_source="structural:", clause="c", falsifier="f",
            action="a", audience="the board", source_task=6)


@needs_tables
def test_the_yield_denominator_is_every_claim_generated(ledger):
    yields = pd.read_csv(TABLES / "insight-yield.csv")
    total = yields[yields.question == "all"].iloc[0]
    assert total.generated == len(ledger)
    assert (total.published + total.published_qualified + total.refused
            == len(ledger))


@needs_tables
def test_the_yield_is_reported_per_question(ledger):
    yields = pd.read_csv(TABLES / "insight-yield.csv")
    per_question = yields[yields.question != "all"]
    assert set(per_question.question) == set(ins.QUESTIONS)
    assert per_question.generated.sum() == len(ledger)


# ---------------------------------------------------------------------------
# C8 — unanimity under a cell floor
# ---------------------------------------------------------------------------


def test_sign_test_p_is_two_sided_and_exact():
    assert ins.sign_test_p(6, 6) == pytest.approx(2 / 64)
    assert ins.sign_test_p(3, 6) == pytest.approx(1.0)
    assert ins.sign_test_p(1, 1) == pytest.approx(1.0)


def test_unanimity_below_six_tests_can_never_be_significant():
    """The claim C8 rests on: 2/2**n > 0.05 for every n < 6."""
    power = ins.sign_test_power(8)
    below = power[power.publishers_tested < 6]
    assert not below.clears_005.any()
    assert power[power.publishers_tested == 6].clears_005.all()


@needs_tables
def test_raising_the_cell_floor_only_drops_tests():
    recount = pd.read_csv(TABLES / "publisher-cell-floor.csv")
    for _, block in recount.groupby("company"):
        block = block.sort_values("cell_floor")
        assert block.publishers_tested.is_monotonic_decreasing


@needs_tables
def test_three_companies_gain_confirmation_by_dropping_tests():
    """C8 in one assertion: a verdict that improves as evidence is removed."""
    unanimity = pd.read_csv(TABLES / "unanimity-verdict.csv")
    gained = unanimity[unanimity.confirmation_gained_by_dropping_tests]
    assert set(gained.company) == {"google", "microsoft", "snowflake"}


@needs_tables
def test_nvidias_significance_depends_on_single_posting_cells():
    """Task 06 §11 calls this the one unqualified cross-company sentence."""
    recount = pd.read_csv(TABLES / "publisher-cell-floor.csv")
    nvidia = recount[recount.company == "nvidia"].set_index("cell_floor")
    assert nvidia.loc[0, "sign_test_p"] < 0.05
    assert nvidia.loc[10, "sign_test_p"] > 0.05
    assert nvidia.loc[0, "publishers_tested"] == 6
    assert nvidia.loc[10, "publishers_tested"] == 1


@needs_tables
def test_the_pooled_direction_never_moves_with_the_floor():
    """C8 corrects the confirmation, not the direction — the distinction the
    register has to make."""
    recount = pd.read_csv(TABLES / "publisher-cell-floor.csv")
    for _, block in recount.groupby("company"):
        assert block.pooled_direction.nunique() == 1


# ---------------------------------------------------------------------------
# The salary audit
# ---------------------------------------------------------------------------


@needs_tables
def test_salary_disclosure_is_missing_not_at_random():
    by_publisher = pd.read_csv(TABLES / "salary-disclosure-by-publisher.csv")
    sizeable = by_publisher[by_publisher.postings >= 20]
    assert sizeable.disclosed_pct.max() > 95
    assert sizeable.disclosed_pct.min() < 5


@needs_tables
def test_conditioning_on_disclosure_moves_the_country_mix():
    disclosure = pd.read_csv(TABLES / "salary-disclosure.csv")
    assert disclosure.country_shift_pp.abs().max() > 50


@needs_tables
def test_no_salary_pair_survives_stratification():
    """Every unstratified gap is a job-function composition difference."""
    stratified = pd.read_csv(TABLES / "salary-pairs-stratified.csv")
    assert not stratified.empty
    assert not stratified.identified.any()


@needs_tables
def test_the_benchmark_promise_is_not_paid():
    promises = pd.read_csv(TABLES / "brief-promises.csv")
    row = promises[promises.promise == "Benchmark salaries"].iloc[0]
    assert row.status == "not paid"
    assert row.published == 0


@needs_tables
def test_disclosure_rate_claims_do_not_pay_the_benchmark_promise(ledger):
    """They publish, but they say who discloses pay, not what anyone pays.

    Counting an adjacent family would report the promise as paid on the
    strength of sentences that do not pay it.
    """
    coverage = ledger[ledger.family == "salary_coverage"]
    assert (coverage.status != ins.REFUSED).all()
    assert "salary_coverage" not in {
        family
        for item in ins.BRIEF_PROMISES
        if item["promise"] == "Benchmark salaries"
        for family in item["families"]
    }


# ---------------------------------------------------------------------------
# The brief's promises, questions and audiences
# ---------------------------------------------------------------------------


@needs_tables
def test_the_product_launch_promise_is_not_payable():
    promises = pd.read_csv(TABLES / "brief-promises.csv")
    row = promises[promises.promise ==
                   "Predict competitor product launches"].iloc[0]
    assert row.status == "not payable"
    assert row.generated == 0


@needs_tables
def test_future_demand_is_answerable_only_on_a_qualified_verdict():
    coverage = pd.read_csv(TABLES / "question-coverage.csv")
    row = coverage[coverage.question == "future_demand"].iloc[0]
    assert row.verdict_firm == 0
    assert row.verdict_qualified > 0


@needs_tables
def test_investors_get_nothing():
    """Every question an investor asks is about a level, a trajectory or a
    forecast, and all three are refused upstream."""
    actionable = pd.read_csv(TABLES / "actionability.csv")
    row = actionable[actionable.audience == "investor"].iloc[0]
    assert row.claims_available == 0


@needs_tables
def test_an_audience_brief_holds_only_publishable_claims():
    for audience in ("strategy", "hr_talent", "product", "exec"):
        brief = pd.read_csv(TABLES / f"audience-brief-{audience}.csv")
        if brief.empty:
            continue
        assert (brief.status != ins.REFUSED).all()
        # the clause is not a separate column a reader could drop; it is
        # folded into the sentence they are handed
        assert brief.sentence.str.contains(" — ").all()


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


@needs_tables
def test_position_is_available_as_a_profile_and_not_as_a_level(ledger):
    verdict = ins.strategy_position(ledger, MEMBER, "google")
    assert verdict.profile_available
    assert not verdict.level_available
    assert not verdict.trajectory_available


@needs_tables
def test_the_nearest_neighbour_is_meta_and_the_rank_is_robust(ledger):
    verdict = ins.strategy_position(ledger, MEMBER, "google")
    assert verdict.nearest_pair == "meta"
    assert verdict.nearest_verdict == "robust"


# ---------------------------------------------------------------------------
# Refusals are an output
# ---------------------------------------------------------------------------


@needs_tables
def test_every_refusal_says_what_would_lift_it():
    refusals = pd.read_csv(TABLES / "refused-claims.csv")
    assert not refusals.empty
    assert refusals.what_would_lift_it.notna().all()


@needs_tables
def test_the_tempting_sentences_are_all_refused(ledger):
    """Ten sentences a reader would want, each blocked by a different rule."""
    tempting = ledger[ledger.family == "tempting"]
    assert len(tempting) == 10
    assert (tempting.status == ins.REFUSED).all()


# ---------------------------------------------------------------------------
# Standing privacy check
# ---------------------------------------------------------------------------


@needs_tables
def test_no_emitted_table_carries_personal_data():
    for path in sorted(TABLES.glob("*.csv")):
        frame = pd.read_csv(path)
        assert not ins.personal_data_columns_present(frame), path.name


@needs_tables
def test_no_emitted_table_carries_a_candidate_column():
    """Task 09's own vocabulary collided with Task 01's blocklist. The check
    stayed strict and this task's column names moved."""
    for path in sorted(TABLES.glob("*.csv")):
        frame = pd.read_csv(path)
        assert not [c for c in frame.columns if "candidate" in c.lower()]


@needs_tables
def test_no_emitted_table_carries_a_forbidden_column():
    for path in sorted(TABLES.glob("*.csv")):
        frame = pd.read_csv(path)
        assert not ins.forbidden_columns(frame), path.name


# ---------------------------------------------------------------------------
# A sentence must not contradict the numbers it quotes
# ---------------------------------------------------------------------------


def test_ordinal_suffixes():
    """`f"{rank}th"` writes "2th", which shipped once."""
    assert [ins._ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd"]


@needs_tables
def test_a_reversed_skill_trend_names_the_reversal(ledger):
    """Looker falls in all three functions while the pooled share rises.

    The direction word comes from the stratified verdict and the two numbers
    come from the pooled columns, so the sentence quotes a rise and asserts a
    fall. Both halves are correct — that is what a Simpson's reversal is —
    but a reader who resolves the clash in favour of the numbers has undone
    C2. The reversal has to be in the sentence, not left to be noticed.
    """
    row = ledger[ledger.claim_id == "skilltrend-looker"].iloc[0]
    assert row.status == ins.PUBLISHED
    assert "reversal" in row.text.lower()
    assert "does not overrule" in row.clause


@needs_tables
def test_every_similarity_claim_is_falsified_by_its_own_rank(ledger):
    """Written for the rank-1 pair and copied down the table, the falsifier
    asks a rank-2 pair to "leave rank 1" — a condition it cannot meet."""
    pairs = ledger[ledger.family == "similarity"]
    assert len(pairs) == 15
    for _, row in pairs.iterrows():
        rank = int(row.text.split("ranking ")[1].split(" of 15")[0])
        assert f"leaving rank {rank} " in row.falsifier, row.claim_id


@needs_tables
def test_only_the_top_pair_is_called_the_closest_competitor(ledger):
    pairs = ledger[ledger.family == "similarity"]
    closest = pairs[pairs.action.str.contains("closest talent competitor")]
    assert len(closest) == 1
    assert closest.iloc[0].claim_id == "pair-google-meta"
