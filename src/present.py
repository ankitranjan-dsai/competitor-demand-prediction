"""Task 10 — final presentation and mentor review.

Shared, company-agnostic layer. Task 09 decided *which sentences this
repository may print*. Task 10 decides *which of them go on a slide*, and that
is a narrower question than it sounds, because a slide does three things a
report does not:

1. **It drops the clause.** A report sentence carries its qualifier on the same
   line. A bullet gets shortened to fit, and the qualifier is the half that
   fits worst. Task 09 §13 is explicit that rendering text without clause
   publishes a different claim from the one that passed the gates.
2. **It adds numbers nothing checked.** A deck needs "846 postings", "six
   companies", "559 tests" — structural facts about the project that no claim
   in the ledger carries. These are exactly the numbers that go stale, because
   they are typed once and never recomputed.
3. **It is spoken.** The mentor asks a question and the answer is improvised.
   An answer that cites nothing is an answer that can drift from the tables
   under it in the one setting where nobody can check it.

So a slide here is not prose. It is a record, in the same sense Task 09's
insight was:

    section      where in the aligned storyline it sits
    title        the line the mentor reads
    bullets      each one *bound* to its evidence, never free text
    figure       a committed PNG, or none
    notes        speaker notes — what to say, and what not to claim

and every bullet is one of exactly three kinds, which is the whole design:

    ``claim``   quotes a Task 09 ledger row by ``claim_id``. Rendered through
                ``insights.sentence`` so text and clause travel together. A
                slide cannot render one without the other, because it never
                sees them apart.
    ``fact``    a structural fact about the repository — posting counts, test
                counts, task counts. Each one is *recomputed* from the repo at
                build time by a named resolver, never typed into the slide.
    ``plain``   narration. Carries no numeral, and the linter enforces that.

`lint_deck` then checks the deck the way Task 09 checked its claims: every
numeral that reaches a slide is traceable to a committed table or a
recomputable repo property, every claim bullet carries its clause, Task 09's
prohibited-pattern list runs over the rendered text, and the refusals that the
project earned are present rather than dropped.

**That is the correction this task raises.** Task 09 §13 called Task 10 "the
first task in this project whose output is not checkable by a test". A deck is
text, and text is checkable when every number in it is bound — which is the
same argument Task 09 made about a report and then declined to make about a
slide. See C9 in `docs/corrections.md`.

Everything runs on pandas, `re` and the standard library, in keeping with
Tasks 05 to 09 — no scipy in a core module.

Rebuild:

    python src/build_presentation.py
    python -m pytest tests/ -q
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import insights as ins

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMBERS = REPO_ROOT / "members"
DEFAULT_MEMBER = MEMBERS / "ankit-google"

#: The two statuses a claim may hold and still reach a slide. `refused` claims
#: do reach the deck — but as refusals, through a `plain` bullet that names
#: them, never as a `claim` bullet that asserts them.
PUBLISHABLE = (ins.PUBLISHED, ins.QUALIFIED)

#: Bullet kinds. The set is closed on purpose: a fourth kind would be a way to
#: put an unbound number on a slide.
BULLET_KINDS = ("claim", "fact", "plain", "refusal")

#: Kinds bound to a row of the Task 09 claim ledger rather than to prose the
#: presenter wrote. These render verbatim and are exempt from the language
#: lint, which already ran on them once and recorded its verdict.
LEDGER_KINDS = ("claim", "refusal")


# ---------------------------------------------------------------------------
# A. Structural facts, recomputed rather than typed
# ---------------------------------------------------------------------------
#
# Every number on a slide is either a Task 09 claim or one of these. The
# distinction matters because they fail differently. A claim goes wrong when
# the evidence under it moves, which Task 09's gates already catch. A
# structural fact goes wrong when *nothing* moves except the calendar — the
# repository grows a test, a table, a correction, and the deck still says what
# it said in August.
#
# This project has already shipped that failure once: both READMEs describe
# Task 09 as "557 tests" against a suite that collects 559. Nobody mis-counted;
# the number was written before the last commit of the task and never
# recomputed. So none of these are stored. Each one is a resolver that reads
# the repository at build time, and the workspace audit re-reads the READMEs to
# check that the prose still agrees with them.


@dataclass(frozen=True)
class Fact:
    """One recomputable structural fact about the repository."""

    key: str
    description: str
    source: str
    resolver: object          # callable(repo_root, member_root, focus)
    digits: int | None = None

    def value(self, repo_root: Path, member_root: Path, focus: str):
        return self.resolver(repo_root, member_root, focus)   # type: ignore[operator]

    def render(self, value) -> str:
        if value is None:
            return "n/a"
        if self.digits is None:
            return f"{value:,}" if isinstance(value, int) else str(value)
        return f"{float(value):.{self.digits}f}"


def _table(member_root: Path, relative: str) -> pd.DataFrame:
    path = member_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"no committed table at {path}")
    return pd.read_csv(path)


def _cell(member_root: Path, relative: str, column: str, **selector):
    """One cell of a committed table, selected the way a citation selects."""
    frame = _table(member_root, relative)
    for key, wanted in selector.items():
        column_values = frame[key].astype(str).str.strip().str.lower()
        frame = frame[column_values == str(wanted).strip().lower()]
    if len(frame) != 1:
        raise LookupError(f"{relative} selector matched {len(frame)} rows, need 1")
    return frame.iloc[0][column]


def _fact_postings(repo_root, member_root, focus):
    return int(_cell(member_root, "task-06-tables/company-comparability.csv",
                     "postings", company=focus))


def _fact_publishers(repo_root, member_root, focus):
    return int(_cell(member_root, "task-06-tables/company-comparability.csv",
                     "publishers", company=focus))


def _fact_months(repo_root, member_root, focus):
    return int(_cell(member_root, "task-06-tables/company-comparability.csv",
                     "months_observed", company=focus))


def _fact_companies(repo_root, member_root, focus):
    screen = _table(member_root, "task-06-tables/company-feasibility-screen.csv")
    return int((screen["verdict"].astype(str) == "included").sum())


def _fact_companies_screened_out(repo_root, member_root, focus):
    screen = _table(member_root, "task-06-tables/company-feasibility-screen.csv")
    return int((screen["verdict"].astype(str) != "included").sum())


def _fact_pool(repo_root, member_root, focus):
    screen = _table(member_root, "task-06-tables/company-feasibility-screen.csv")
    return int(screen.loc[screen["verdict"].astype(str) == "included",
                          "postings"].sum())


def _fact_skills(repo_root, member_root, focus):
    return int(len(_table(member_root, "task-04-tables/skill-frequency.csv")))


def _fact_claims_generated(repo_root, member_root, focus):
    return int(_cell(member_root, "task-09-tables/insight-yield.csv",
                     "generated", question="all"))


def _fact_claims_published(repo_root, member_root, focus):
    row = _table(member_root, "task-09-tables/insight-yield.csv")
    row = row[row["question"].astype(str) == "all"].iloc[0]
    return int(row["published"]) + int(row["published_qualified"])


def _fact_claims_refused(repo_root, member_root, focus):
    return int(_cell(member_root, "task-09-tables/insight-yield.csv",
                     "refused", question="all"))


def _fact_yield_pct(repo_root, member_root, focus):
    return float(_cell(member_root, "task-09-tables/insight-yield.csv",
                       "yield_pct", question="all"))


def _fact_distinctive_skills(repo_root, member_root, focus):
    return int(_cell(member_root, "task-09-tables/strategy-position.csv",
                     "distinctive_skills", focus=focus))


def _fact_nearest_score(repo_root, member_root, focus):
    return float(_cell(member_root, "task-09-tables/strategy-position.csv",
                       "nearest_score", focus=focus))


TEST_DEF = re.compile(r"^\s*def\s+test_\w+\s*\(", re.MULTILINE)
_COLLECTED = re.compile(r"^(\d+) tests? collected", re.MULTILINE)

#: Collection is a subprocess, so it is cached for the life of the process.
#: A build resolves the fact register once and the audit re-reads it; a test
#: run touches it a handful of times. Neither should pay for it twice.
_TEST_COUNT_CACHE: dict = {}


def count_test_functions(repo_root: Path) -> int:
    """Every ``def test_*`` under ``tests/``.

    Not the suite size. Parametrised cases collect once per parameter set, so
    this undercounts — it is the fallback, and the two are kept separate
    rather than blurred, because the number the READMEs quote is the collected
    one.
    """
    total = 0
    for path in sorted((repo_root / "tests").glob("test_*.py")):
        total += len(TEST_DEF.findall(path.read_text(encoding="utf-8")))
    return total


def count_tests(repo_root: Path = REPO_ROOT) -> int:
    """The suite size as pytest counts it.

    Asked of pytest rather than of a regex, because "how many tests" is a
    question only the collector answers correctly once anything is
    parametrised: this suite holds 453 ``def test_`` lines and collects more.
    Collection does not execute anything, so this is safe to call from inside
    a test run. If pytest cannot be reached the static count stands in, and
    the workspace audit reports which of the two it used.
    """
    key = str(repo_root)
    if key in _TEST_COUNT_CACHE:
        return _TEST_COUNT_CACHE[key]
    count = count_test_functions(repo_root)
    try:
        import sys
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             str(repo_root / "tests")],
            capture_output=True, text=True, timeout=180, cwd=str(repo_root),
        )
        match = _COLLECTED.search(done.stdout)
        if match:
            count = int(match.group(1))
    except Exception:      # pragma: no cover - environment without pytest
        pass
    _TEST_COUNT_CACHE[key] = count
    return count


TASK_ROW = re.compile(r"^\|\s*(\d{2})\s*\|", re.MULTILINE)
CORRECTION_HEADING = re.compile(r"^##\s+(C\d+)\s+—", re.MULTILINE)


def count_tasks(repo_root: Path) -> int:
    """Numbered task rows in the README's task table."""
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    return len({m.group(1) for m in TASK_ROW.finditer(readme)})


def count_corrections(repo_root: Path) -> int:
    """Entries in the corrections register, counted from its own headings."""
    text = (repo_root / "docs" / "corrections.md").read_text(encoding="utf-8")
    return len({m.group(1) for m in CORRECTION_HEADING.finditer(text)})


def count_committed_tables(repo_root: Path) -> int:
    return len(list(repo_root.glob("members/*/task-*-tables/*.csv")))


def count_committed_figures(repo_root: Path) -> int:
    return len(list(repo_root.glob("members/*/task-*-figures/*.png")))


#: The fact register. Adding a number to a slide means adding a resolver here,
#: which is the friction that keeps hand-typed figures off the deck.
FACTS: tuple[Fact, ...] = (
    Fact("postings_focus", "postings collected for the focus company",
         "task-06-tables/company-comparability.csv#postings", _fact_postings),
    Fact("publishers_focus", "publishers syndicating the focus company",
         "task-06-tables/company-comparability.csv#publishers", _fact_publishers),
    Fact("months_observed", "months the focus company is present in",
         "task-06-tables/company-comparability.csv#months_observed", _fact_months),
    Fact("companies_compared", "companies that passed the feasibility screen",
         "task-06-tables/company-feasibility-screen.csv#verdict", _fact_companies),
    Fact("companies_screened_out", "companies the screen excluded",
         "task-06-tables/company-feasibility-screen.csv#verdict",
         _fact_companies_screened_out),
    Fact("postings_pool", "postings across every included company",
         "task-06-tables/company-feasibility-screen.csv#postings", _fact_pool),
    Fact("skills_observed", "distinct skills the taxonomy matched",
         "task-04-tables/skill-frequency.csv (rows)", _fact_skills),
    Fact("claims_generated", "candidate claims Task 09 generated",
         "task-09-tables/insight-yield.csv#generated@question=all",
         _fact_claims_generated),
    Fact("claims_published", "claims that cleared all four gates",
         "task-09-tables/insight-yield.csv#published+published_qualified",
         _fact_claims_published),
    Fact("claims_refused", "claims refused at a gate",
         "task-09-tables/insight-yield.csv#refused@question=all",
         _fact_claims_refused),
    Fact("claim_yield_pct", "publishable share of generated claims",
         "task-09-tables/insight-yield.csv#yield_pct@question=all",
         _fact_yield_pct, digits=1),
    Fact("distinctive_skills", "skills separating the focus company after FDR",
         "task-09-tables/strategy-position.csv#distinctive_skills",
         _fact_distinctive_skills),
    Fact("nearest_score", "cosine similarity to the nearest company",
         "task-09-tables/strategy-position.csv#nearest_score",
         _fact_nearest_score, digits=4),
    Fact("tests_total", "tests the suite collects",
         "pytest --collect-only over tests/",
         lambda repo, member, focus: count_tests(repo)),
    Fact("tasks_total", "numbered tasks in the README table",
         "README.md task table",
         lambda repo, member, focus: count_tasks(repo)),
    Fact("corrections_total", "entries in the corrections register",
         "docs/corrections.md headings",
         lambda repo, member, focus: count_corrections(repo)),
    Fact("tables_committed", "aggregate tables committed across all tasks",
         "members/*/task-*-tables/*.csv",
         lambda repo, member, focus: count_committed_tables(repo)),
    Fact("figures_committed", "figures committed across all tasks",
         "members/*/task-*-figures/*.png",
         lambda repo, member, focus: count_committed_figures(repo)),
)

FACTS_BY_KEY = {fact.key: fact for fact in FACTS}


def resolve_facts(repo_root: Path = REPO_ROOT,
                  member_root: Path = DEFAULT_MEMBER,
                  focus: str = "google") -> dict:
    """Resolve every registered fact against the repository as it stands."""
    resolved = {}
    for fact in FACTS:
        value = fact.value(repo_root, member_root, focus)
        resolved[fact.key] = {"value": value, "text": fact.render(value),
                              "source": fact.source,
                              "description": fact.description}
    return resolved


def fact_table(resolved: dict) -> pd.DataFrame:
    """The fact register as a committed table, so the deck's numbers audit."""
    rows = [{"key": key, "value": entry["text"],
             "description": entry["description"], "source": entry["source"]}
            for key, entry in resolved.items()]
    return pd.DataFrame(rows, columns=["key", "value", "description", "source"])


# ---------------------------------------------------------------------------
# B. The slide record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bullet:
    """One line on a slide, and the binding that justifies it.

    ``kind`` decides how it renders and how it is checked:

    * ``claim`` — ``claim_id`` names a Task 09 ledger row. The rendered line is
      ``insights.sentence`` of that row, character for character. Nothing here
      may rewrite it, which is what makes "the clause travelled" checkable
      rather than a habit.
    * ``fact`` — ``text`` is a template over the fact register. It may contain
      ``{placeholders}`` and no bare numerals; the numbers arrive at render
      time from resolvers that read the repository.
    * ``plain`` — narration, and it may not contain a numeral at all beyond the
      structural ones (a task number, a section mark, the collection year).
    * ``refusal`` — ``claim_id`` names a *refused* ledger row. It renders the
      sentence the project declines to make, the gate that stopped it, and the
      action that follows. A refusal is content, not an absence, so it is
      bound to a committed row exactly as a claim is.
    """

    kind: str
    text: str = ""
    claim_id: str = ""

    def __post_init__(self):
        if self.kind not in BULLET_KINDS:
            raise ValueError(f"unknown bullet kind {self.kind!r}")
        if self.kind in LEDGER_KINDS and not self.claim_id:
            raise ValueError(f"a {self.kind} bullet needs a claim_id")
        if self.kind not in LEDGER_KINDS and not self.text:
            raise ValueError(f"a {self.kind} bullet needs text")


def claim(claim_id: str) -> Bullet:
    return Bullet("claim", claim_id=claim_id)


def fact(text: str) -> Bullet:
    return Bullet("fact", text=text)


def plain(text: str) -> Bullet:
    return Bullet("plain", text=text)


def refusal(claim_id: str) -> Bullet:
    return Bullet("refusal", claim_id=claim_id)


@dataclass(frozen=True)
class Slide:
    """One slide, in the team's aligned storyline."""

    number: int
    section: str
    title: str
    bullets: tuple[Bullet, ...] = ()
    figure: str = ""
    table: str = ""
    notes: str = ""
    #: Pointers that must appear on this slide's rendered text — currently the
    #: correction ids a claim on it is not allowed to travel without.
    carries: tuple[str, ...] = ()


PLACEHOLDER = re.compile(r"\{(\w+)\}")

#: Numerals a slide may carry without a binding, because they are not
#: measurements. A task number, a correction id, a section mark, the year the
#: collection covers, a half or a quarter label, a forecast horizon. Each is
#: stripped before the linter looks for unbound figures, and each one is named
#: so the audit table can say which exemption a line used.
STRUCTURAL_NUMERALS: tuple[tuple[str, str], ...] = (
    ("task_number", r"\bTasks?\s+\d{1,2}(\s*(?:to|–|-|,|and)\s*\d{1,2})*"),
    ("correction_id", r"\bC\d+\b"),
    ("section_mark", r"§\s*\d+(?:\.\d+)*"),
    ("collection_year", r"\b2023\b"),
    ("half", r"\bH[12]\b"),
    ("quarter", r"\bQ[1-4]\b"),
    ("iso_week", r"\b20\d{2}-W\d{1,2}\b"),
    ("horizon", r"\bh\s*=\s*\d+"),
    ("layer", r"\bLayer\s+[AB]\b"),
)

NUMERAL = re.compile(r"\d")


def strip_structural(text: str) -> tuple[str, list[str]]:
    """Remove the exempt numerals, returning the remainder and what fired."""
    used = []
    remainder = text
    for name, pattern in STRUCTURAL_NUMERALS:
        remainder, hits = re.subn(pattern, " ", remainder, flags=re.IGNORECASE)
        if hits:
            used.append(name)
    return remainder, used


def refusal_sentence(row) -> str:
    """How a refusal is spoken: the sentence, the gate, the consequence.

    The refused sentence is quoted rather than paraphrased. Paraphrasing it
    would let the presenter keep the shape of the claim while disowning the
    words, which is the failure mode this whole register exists to stop.
    """
    action = str(row.get("action", "")).strip()
    action = "" if action.lower() in ("", "nan") else f" {action[0].upper()}{action[1:]}."
    # The clause is the reason written for a person; the reason column is the
    # gate's own transcript. Prefer the clause where one exists, and fall back
    # to the transcript for the sentences a pattern stopped outright, which
    # carry no clause because they were never going to be qualified into life.
    clause = str(row.get("clause", "")).strip()
    why = clause if clause and clause.lower() != "nan" else str(row["reason"])
    return (f"Not available — \u201c{str(row['text']).strip()}\u201d. "
            f"Stopped at the {row['blocked_by']} gate: {why}."
            f"{action}")


def render_bullet(bullet: Bullet, ledger: pd.DataFrame, facts: dict) -> str:
    """The line a reader sees, with its evidence already attached."""
    if bullet.kind in LEDGER_KINDS:
        rows = ledger[ledger["claim_id"].astype(str) == bullet.claim_id]
        if len(rows) != 1:
            raise LookupError(
                f"claim_id {bullet.claim_id!r} matched {len(rows)} ledger rows"
            )
        row = rows.iloc[0]
        if bullet.kind == "claim":
            return ins.sentence(row)
        return refusal_sentence(row)
    if bullet.kind == "fact":
        missing = [key for key in PLACEHOLDER.findall(bullet.text)
                   if key not in facts]
        if missing:
            raise KeyError(f"no such fact: {', '.join(sorted(missing))}")
        return PLACEHOLDER.sub(lambda m: facts[m.group(1)]["text"], bullet.text)
    return bullet.text


def render_slide(slide: Slide, ledger: pd.DataFrame, facts: dict) -> dict:
    """A slide as text — the form the deck file and the linter both read."""
    return {
        "number": slide.number,
        "section": slide.section,
        "title": slide.title,
        "lines": [render_bullet(b, ledger, facts) for b in slide.bullets],
        "figure": slide.figure,
        "table": slide.table,
        "notes": slide.notes,
        "carries": list(slide.carries),
    }


def render_deck(deck, ledger: pd.DataFrame, facts: dict) -> list:
    return [render_slide(slide, ledger, facts) for slide in deck]


# ---------------------------------------------------------------------------
# C. The aligned storyline
# ---------------------------------------------------------------------------
#
# The brief asks for "a polished final presentation (per member + aligned
# storyline)". Aligning four decks by agreeing a font is not alignment; the
# thing that has to match is the *order of the argument*, because that is what
# lets a mentor hear four talks and compare them.
#
# The order below is the team standard, and it is not the order the work
# happened in. The work went data -> method -> result. The talk goes
# question -> what the data can carry -> results that survive that -> what is
# refused, because a result presented before its identification limit has
# already been over-claimed by the time the limit arrives. Section 6 is load
# bearing: every later section leans on it.
#
# Each section is a slot. A specialist fills it with their own company's
# claims; the slots, their order, and the rule that the refusal section is
# mandatory do not move.

SECTIONS: tuple[tuple[str, str], ...] = (
    ("title", "Who, what, and how much data"),
    ("problem", "The question the brief asks"),
    ("legal", "The legal gate, before any collection"),
    ("data", "What was actually collected"),
    ("pipeline", "From posting text to a skill table"),
    ("identification", "What a posting count counts"),
    ("trends", "Hiring trends, within one year"),
    ("comparison", "Six companies, compared where comparison holds"),
    ("forecast", "Demand forecasting"),
    ("similarity", "Company similarity"),
    ("insights", "From evidence to sentences"),
    ("position", "The focus company's position"),
    ("stack", "What the stack difference says"),
    ("refusals", "What this project will not say"),
    ("corrections", "The corrections register"),
    ("quality", "How any of this is checked"),
    ("workspace", "The repository the team inherits"),
    ("next", "What would change the answers"),
    ("qa", "Backup — the question bank"),
)

#: Sections whose content is mandatory in every specialist's deck. A deck may
#: reorder nothing and may drop nothing here. `refusals` is on the list for the
#: reason Task 09 §13 gives: the refusals are the first thing a deck drops and
#: among the most decision-relevant outputs the project has.
REQUIRED_SECTIONS = ("legal", "identification", "refusals", "corrections",
                     "quality")


def deck_spec(focus: str = "google") -> tuple[Slide, ...]:
    """The Google deck, filled into the team's slot order.

    Claim ids are Task 09's. A specialist swapping in their own company swaps
    the ids, not the slots.
    """
    slides = [
        Slide(
            1, "title",
            "Competitor Demand Prediction Using Job Postings",
            (
                plain("Ankit Ranjan — specialist for Google, one of four "
                      "company seats feeding one shared model."),
                fact("Google: {postings_focus} postings, {publishers_focus} "
                     "publishers, {months_observed} months of 2023."),
                fact("{tasks_total} tasks, {tables_committed} committed "
                     "tables, {figures_committed} figures, {tests_total} "
                     "tests."),
            ),
            notes="Open on the constraint, not the result: this is a single "
                  "year from a single aggregator, and that decides what the "
                  "rest of the talk is allowed to claim.",
        ),
        Slide(
            2, "problem",
            "Job postings are a forward-looking signal",
            (
                plain("The brief: forecast competitor demand, skill trends "
                      "and market direction from public job postings."),
                plain("A posting states intent before a hire reaches any "
                      "financial report — that is the whole appeal."),
                plain("A posting is also a document a job board chose to "
                      "syndicate. Everything in this deck turns on that "
                      "second sentence."),
            ),
            notes="The pitch and the catch on one slide, so nothing later "
                  "looks like an excuse invented after a null result.",
        ),
        Slide(
            3, "legal",
            "The legal gate came first, and it removed sources",
            (
                plain("Task 01 set an API-first source list and a four-pillar "
                      "checklist: terms of service, robots.txt, public versus "
                      "copyrighted, and never personal data."),
                plain("Google Careers scraping was rejected on robots.txt. It "
                      "was never collected, and the rejection is committed "
                      "alongside the approvals."),
                plain("Reed was never approved, so it was never called, even "
                      "though a key sits in the environment file."),
                plain("A standing privacy check runs every task: no committed "
                      "table may carry a column that names a person."),
            ),
            table="docs/legal/rejected-sources.md",
            notes="The point is that the checklist has teeth — it says no to "
                  "the single most obvious source for a Google specialist.",
        ),
        Slide(
            4, "data",
            "What was collected, and what failed the screen",
            (
                fact("Google: {postings_focus} postings across "
                     "{publishers_focus} publishers, present in every one of "
                     "{months_observed} months."),
                fact("Comparison set: {companies_compared} companies and "
                     "{postings_pool} postings; {companies_screened_out} "
                     "candidates failed the support screen."),
                plain("One source, one year. Every trend sentence here is "
                      "labelled within-2023, and none of them is a "
                      "year-on-year comparison."),
            ),
            table="task-06-tables/company-feasibility-screen.csv",
            notes="Name the two that failed the screen and why — a screen "
                  "that never excludes anything is decoration.",
        ),
        Slide(
            5, "pipeline",
            "Rules, not models, so every extraction is inspectable",
            (
                plain("Task 03 split preprocessing into a structured layer "
                      "and a free-text layer. The free-text layer is "
                      "validated and idle: the API truncates description "
                      "text, so full-text extraction was never available."),
                fact("Task 04 matched {skills_observed} distinct skills with "
                     "rules and aliases rather than a model, so a reviewer "
                     "can check any single extraction by reading it."),
                plain("Skill shares use the skilled-posting denominator, "
                      "never all postings — the alternative confounds demand "
                      "with description coverage."),
            ),
            notes="The truncation is the honest reason there is no embedding "
                  "model here. Say it before the mentor asks why no BERT.",
        ),
        Slide(
            6, "identification",
            "A posting count is a count of the boards that syndicate it",
            (
                claim("panel-google"),
                plain("So levels do not compare across companies, and the "
                      "project compares shares of a fixed publisher panel "
                      "instead."),
                plain("February is a collection gap, not a hiring freeze: the "
                      "publishers that vanish in February come back in "
                      "March."),
                plain("The date field is an aggregator first-seen date, so "
                      "the weekly series describes discovery, not hiring."),
            ),
            figure="task-05-figures/03-publisher-panel.png",
            notes="This is the load-bearing slide. If it lands, every "
                  "refusal later reads as a consequence rather than a "
                  "hedge. Do not rush it.",
        ),
        Slide(
            7, "trends",
            "Within-year trends: the mix moves, the volume does not resolve",
            (
                claim("skilltrend-sql"),
                claim("skilltrend-looker"),
                plain("Google's volume direction is not identified: four "
                      "defensible panel treatments disagree on the sign, so "
                      "no direction is reported."),
            ),
            figure="task-05-figures/02-panel-sensitivity.png",
            notes="Looker is the Simpson's reversal — falling inside every "
                  "job function while the pooled share rises. It overturned "
                  "an earlier headline of ours, which is the honest way to "
                  "introduce the corrections register later.",
        ),
        Slide(
            8, "comparison",
            "Share of a common publisher pool, not headcount",
            (
                claim("share-google"),
                claim("share-nvidia"),
                plain("Unanimity across publishers looks like the strongest "
                      "evidence on this slide and is the weakest: the number "
                      "of tests moves with the threshold, see C8."),
            ),
            figure="task-06-figures/03-relative-share.png",
            carries=("C8",),
            notes="Expect the mentor to like the unanimous count. Volunteer "
                  "C8 before they do.",
        ),
        Slide(
            9, "forecast",
            "The forecast question has an answer, and it is no",
            (
                claim("forecastable-google"),
                plain("Every series carries real signal, and no model beat "
                      "persistence at any horizon. The maximum useful "
                      "horizon is zero, and the forecast ships marked "
                      "unsupported."),
                plain("The collection is easier to predict than any "
                      "company's demand — which is a statement about the "
                      "aggregator, not about hiring."),
            ),
            figure="task-07-figures/05-horizon-limits.png",
            notes="Signal and usability are different questions. The gate "
                  "says the series is not noise; the horizon table says the "
                  "interval is too wide to act on.",
        ),
        Slide(
            10, "similarity",
            "One pair of six companies is actually separated",
            (
                claim("pair-google-meta"),
                plain("It is the only robust pair. Most pairs sit in one "
                      "tier and their ranks are not separated by the "
                      "bootstrap."),
                plain("Own-product vocabulary is the largest single lever on "
                      "these scores, so it is published as a sensitivity and "
                      "never applied to the headline."),
            ),
            figure="task-08-figures/04-rank-stability.png",
            notes="A raw similarity means nothing without its two per-pair "
                  "nulls. Have the identical and unrelated numbers ready.",
        ),
        Slide(
            11, "insights",
            "Sentences were generated, then gated — not written, then defended",
            (
                fact("{claims_generated} candidate claims were generated from "
                     "the verdict tables; {claims_published} publish and "
                     "{claims_refused} are refused, a {claim_yield_pct}% "
                     "yield."),
                plain("Generating them is what makes the yield honest: the "
                      "denominator is everything this evidence base could be "
                      "asked to say, not everything the author thought of."),
                plain("Almost every refusal fails on identification, not on "
                      "phrasing. Rewording does not rescue them."),
            ),
            figure="task-09-figures/01-insight-yield.png",
            notes="The yield is the headline of Task 09. A high yield here "
                  "would mean the gates were not doing anything.",
        ),
        Slide(
            12, "position",
            "Position is a profile, not a ranking",
            (
                fact("Available as a profile: {distinctive_skills} skills "
                     "separate Google from the other five after "
                     "false-discovery control, nearest neighbour Meta at "
                     "{nearest_score}."),
                plain("Position as a level is refused, because a posting "
                      "count is a syndication count."),
                plain("Position as a trajectory is refused, because "
                      "trajectory similarity did not clear its own null."),
            ),
            figure="task-06-figures/08-google-distinctiveness.png",
            notes="Distinctive does not mean more. Of those distinctive "
                  "skills only a minority are ones Google asks for *more* "
                  "than the others.",
        ),
        Slide(
            13, "stack",
            "A stack difference is not a capability difference",
            (
                claim("distinct-bigquery"),
                claim("skillgap-spark-google-databricks"),
                plain("Own products inflate their own mentions, so a "
                      "company's stack vocabulary is partly a marketing "
                      "artefact of its job adverts."),
            ),
            figure="task-06-figures/07-skill-heatmap.png",
            notes="Product managers are the audience with the most "
                  "publishable sentences here. This is the slide they want.",
        ),
        Slide(
            14, "refusals",
            "What this project will not say",
            (
                refusal("tempting-forecast"),
                refusal("tempting-level"),
                refusal("salary-google-meta-science-research"),
                refusal("tempting-country"),
                plain("There is no sentence here for an investor either. "
                      "Every investor question is a level, a trajectory, or "
                      "a forecast, and all three are on this slide."),
            ),
            figure="task-09-figures/03-audience-reach.png",
            notes="Deliver this slide at the same pace as the results "
                  "slides. It is the deliverable the brief's promise list "
                  "actually settles.",
        ),
        Slide(
            15, "corrections",
            "Corrections are recorded, never overwritten",
            (
                fact("{corrections_total} corrections are on the record, and "
                     "each one is checked against the committed table it "
                     "cites."),
                plain("A submitted task is never silently rewritten. The "
                      "original wording stays and gains a pointer to the "
                      "register."),
                plain("Almost all of them have one shape: a sentence outran "
                      "the table under it."),
            ),
            table="docs/corrections.md",
            notes="Volunteer the ones that overturned our own headlines. A "
                  "register with no self-inflicted entries is a register "
                  "nobody used.",
        ),
        Slide(
            16, "quality",
            "Every number here is bound to a committed table",
            (
                fact("{tests_total} tests, including one for every registered "
                     "correction, so the prose cannot drift from a rebuild."),
                plain("Core modules import no scipy: the statistics are "
                      "hand-rolled and cross-checked against scipy in "
                      "separate validation scripts, so a reviewer can rebuild "
                      "without matching a solver version."),
                plain("This deck is linted too — every figure on a slide "
                      "resolves to a claim that passed the gates or to a "
                      "repository fact recomputed at build time."),
            ),
            notes="The last bullet is the correction this task raises "
                  "against our own Task 09 handover, which predicted the "
                  "presentation could not be checked by a test.",
        ),
        Slide(
            17, "workspace",
            "The repository the next specialists inherit",
            (
                fact("{tables_committed} committed tables and "
                     "{figures_committed} figures across {tasks_total} "
                     "tasks, with row-level data git-ignored throughout."),
                plain("Shared, company-agnostic code sits in the top-level "
                      "source folder; each specialist's outputs sit in their "
                      "own member folder."),
                plain("Three seats are open. The comparison set they inherit "
                      "was written by one person and is published as an "
                      "auditable table they are invited to overrule."),
            ),
            notes="Frame the open seats as a handover, not an apology. The "
                  "matching audit exists so a new specialist can disagree "
                  "with a row rather than redo the task.",
        ),
        Slide(
            18, "next",
            "What would change the answers",
            (
                plain("More months, so a within-year trend can become a "
                      "seasonal one and a forecast can be tested."),
                plain("More publishers per month, so the panel stops "
                      "deciding the volume direction."),
                plain("A source that returns full description text, which "
                      "would wake the idle free-text layer and lift skill "
                      "coverage."),
                plain("And the standing lesson: a handover section is a "
                      "prediction, not an instruction. Read the brief "
                      "first."),
            ),
            notes="Each line is a falsifier the reports already name, not a "
                  "wish list.",
        ),
        Slide(
            19, "qa",
            "Backup — the question bank",
            (
                plain("Every likely mentor question, with the answer bound "
                      "to the committed table that settles it."),
                plain("If a question has no bound answer, the honest answer "
                      "is that this evidence base does not settle it."),
            ),
            table="task-10-tables/mentor-qa.csv",
            notes="Do not improvise a number. The bank exists so the spoken "
                  "answer and the written one are the same answer.",
        ),
    ]
    return tuple(slides)


# ---------------------------------------------------------------------------
# D. The deck linter
# ---------------------------------------------------------------------------
#
# Task 09 §13 predicted that this task's output would be the first in the
# project not checkable by a test. That prediction is what this section
# overturns, and the argument is short: a slide is a claim plus an attachment,
# and both halves are files in this repository. The prose is bound to the
# ledger, the numbers are bound to committed tables, the figures are bound to
# committed PNGs, and the caveats are bound to the corrections register.
# What cannot be checked is the delivery — the pace, the tone, whether the
# refusal slide is rushed. Those live in the speaker notes and belong to the
# rehearsal, not the linter, and the honest boundary is worth stating.

#: The longest a single rendered bullet may be. Set from the longest sentence
#: the claim ledger actually produces, not from taste: claim bullets are
#: reproduced verbatim, so a cap below that would force the deck to break its
#: own provenance rule to satisfy a style rule.
SPOKEN_MAX = 320

#: A slide with one bullet is a title card; a slide with six is a document.
MIN_BULLETS = 1
MAX_BULLETS = 5

#: The refusal slide is the one a presenter under time pressure cuts first,
#: and the first thing to go on it is the binding: a bullet that *says* the
#: project refuses forecasts is prose, while one bound to a refused ledger row
#: quotes the sentence and names the gate. Only the second kind counts here.
MIN_REFUSALS = 4

DIGIT = re.compile(r"\d")
CORRECTION_ID = re.compile(r"\bC\d+\b")

LINT_COLUMNS = ("slide", "section", "bullet", "rule", "subject", "detail")


def _v(slide_no, section, bullet, rule, subject, detail) -> dict:
    return {"slide": slide_no, "section": section, "bullet": bullet,
            "rule": rule, "subject": str(subject), "detail": detail}


def known_corrections(repo_root: Path = REPO_ROOT) -> set:
    """The correction ids the register actually defines."""
    path = repo_root / "docs" / "corrections.md"
    if not path.exists():
        return set()
    return set(CORRECTION_HEADING.findall(path.read_text(encoding="utf-8")))


def resolve_asset(reference: str, repo_root: Path, member_root: Path) -> Path:
    """Where a slide's figure or table lives.

    Two roots, because a slide leans on two kinds of evidence: the
    specialist's own outputs, and the team documents that constrain them. A
    reference beginning `docs/` is a team document and resolves from the
    repository root; anything else is the specialist's own and resolves from
    their member folder.
    """
    if reference.startswith("docs/"):
        return repo_root / reference
    return member_root / reference


def lint_bullets(bullets, carries, ledger: pd.DataFrame, facts: dict,
                 here, countries: tuple = ()) -> list:
    """Every way a run of bullets can be wrong.

    Shared by slides and by the question bank, because a spoken answer to a
    mentor is the same object as a bullet: a sentence that has to name the
    evidence it rests on.

    ``countries`` is passed through to the language gate because one of the
    nine prohibitions — Task 05's country split — is parameterised by data
    rather than by a regex: the rule is "do not name a country", and which
    strings those are comes from the committed panel check. Calling the gate
    without them silently drops that rule, which is how a deck ends up with a
    line about hiring in Germany that eight of nine checks approve of.
    """
    out = []
    add = out.append
    for i, bullet in enumerate(bullets):
        if bullet.kind in LEDGER_KINDS:
            rows = ledger[ledger["claim_id"].astype(str) == bullet.claim_id]
            if len(rows) != 1:
                add(here(i, "claim_exists", bullet.claim_id,
                         f"matched {len(rows)} ledger rows, expected 1"))
                continue
            row = rows.iloc[0]
            if bullet.kind == "refusal":
                if row["status"] != ins.REFUSED:
                    add(here(i, "refusal_refused", bullet.claim_id,
                             f"status is {row['status']}; a refusal bullet "
                             "quotes a sentence the gates actually stopped"))
                if not str(row.get("blocked_by", "")).strip():
                    add(here(i, "refusal_attributed", bullet.claim_id,
                             "a refusal names the gate that stopped it"))
                continue
            if row["status"] not in PUBLISHABLE:
                add(here(i, "claim_publishable", bullet.claim_id,
                         f"status is {row['status']}"))
            clause = str(row.get("clause", "")).strip()
            if not clause or clause.lower() == "nan":
                add(here(i, "clause_travels", bullet.claim_id,
                         "published claim carries no qualifying clause"))
            for cid in CORRECTION_ID.findall(str(row.get("depends_on", ""))):
                if cid not in carries:
                    add(here(i, "correction_carried", bullet.claim_id,
                             f"claim depends on {cid}; it is not carried here"))
            if not bool(row.get("gate_lint", False)):
                add(here(i, "claim_gate_lint", bullet.claim_id,
                         "ledger records this sentence failing the language gate"))
        elif bullet.kind == "fact":
            for key in PLACEHOLDER.findall(bullet.text):
                if key not in facts:
                    add(here(i, "fact_resolves", key,
                             "no fact of that name is recomputed at build time"))
            skeleton, _ = strip_structural(PLACEHOLDER.sub("", bullet.text))
            if DIGIT.search(skeleton):
                add(here(i, "fact_numerals_bound", bullet.text[:60],
                         "a number typed into a fact template is not recomputed"))
        else:
            skeleton, _ = strip_structural(bullet.text)
            if DIGIT.search(skeleton):
                add(here(i, "plain_numerals_bound", bullet.text[:60],
                         "narration may not carry an unbound number"))

        try:
            line = render_bullet(bullet, ledger, facts)
        except (LookupError, KeyError):
            continue                       # already reported above

        # The language gate runs on what the author wrote, not on what the
        # ledger reproduces. Re-linting a claim sentence looks rigorous and is
        # wrong: a qualifying clause earns its place by *naming* the forbidden
        # reading in order to deny it, so "never of all postings" and "not
        # headcount" trip the very patterns they satisfy. Those sentences
        # cleared the gate once, in Task 09, and the ledger records it — that
        # is the check above. Here we check the words this task added.
        if bullet.kind not in LEDGER_KINDS:
            for hit in ins.lint_text(line, countries):
                add(here(i, "language_clean", hit["rule"],
                         f"matched {hit['matched']!r} — {hit['why']}"))
            if len(line) > SPOKEN_MAX:
                add(here(i, "bullet_speakable", len(line),
                         f"a spoken bullet stays under {SPOKEN_MAX} characters"))

    # A claim sentence can be longer than anyone can say in one breath, and it
    # may not be shortened, because it is reproduced character for character.
    # So length is not a violation for claims — it is a delivery instruction:
    # that bullet is read off the slide, not recited. One such bullet is a
    # dense slide; two is a wall of text.
    long_claims = [b.claim_id for b in bullets if b.kind in LEDGER_KINDS
                   and len(_safe_render(b, ledger, facts)) > SPOKEN_MAX]
    if len(long_claims) > 1:
        add(here(-1, "long_claims", ", ".join(long_claims),
                 f"at most one bullet over {SPOKEN_MAX} characters here"))
    return out


def lint_slide(slide: Slide, ledger: pd.DataFrame, facts: dict,
               repo_root: Path = REPO_ROOT,
               member_root: Path = DEFAULT_MEMBER,
               corrections: set | None = None,
               countries: tuple | None = None) -> list:
    """Every way one slide can be wrong, checked one rule at a time."""
    corrections = known_corrections(repo_root) if corrections is None else corrections
    countries = ins.country_vocabulary(member_root) if countries is None else countries
    out = []
    add = out.append
    here = lambda i, rule, subject, detail: _v(
        slide.number, slide.section, i, rule, subject, detail)

    # -- shape -------------------------------------------------------------
    if not MIN_BULLETS <= len(slide.bullets) <= MAX_BULLETS:
        add(here(-1, "slide_density", len(slide.bullets),
                 f"a slide carries {MIN_BULLETS}-{MAX_BULLETS} bullets"))
    if not slide.notes.strip():
        add(here(-1, "notes_present", slide.title,
                 "every slide carries speaker notes"))
    if slide.section == "refusals":
        bound = sum(1 for b in slide.bullets if b.kind == "refusal")
        if bound < MIN_REFUSALS:
            add(here(-1, "refusals_intact", bound,
                     f"the refusal slide carries at least {MIN_REFUSALS} "
                     "refusals bound to a refused ledger row — narration "
                     "about refusing is not a refusal"))

    # -- attachments -------------------------------------------------------
    for kind in ("figure", "table"):
        reference = getattr(slide, kind)
        if not reference:
            continue
        if not resolve_asset(reference, repo_root, member_root).exists():
            add(here(-1, "asset_exists", reference,
                     f"{kind} does not resolve to a committed file"))

    for cid in slide.carries:
        if cid not in corrections:
            add(here(-1, "correction_exists", cid,
                     "slide cites a correction the register does not define"))

    out.extend(lint_bullets(slide.bullets, slide.carries, ledger, facts, here,
                            countries))
    return out


def _safe_render(bullet: Bullet, ledger: pd.DataFrame, facts: dict) -> str:
    try:
        return render_bullet(bullet, ledger, facts)
    except (LookupError, KeyError):
        return ""


def long_bullets(deck, ledger: pd.DataFrame, facts: dict) -> set:
    """Claim bullets to be read off the slide rather than recited."""
    return {b.claim_id for slide in deck for b in slide.bullets
            if b.kind in LEDGER_KINDS
            and len(_safe_render(b, ledger, facts)) > SPOKEN_MAX}


def lint_deck(deck, ledger: pd.DataFrame, facts: dict,
              repo_root: Path = REPO_ROOT,
              member_root: Path = DEFAULT_MEMBER) -> pd.DataFrame:
    """The whole deck: structure first, then every slide."""
    corrections = known_corrections(repo_root)
    countries = ins.country_vocabulary(member_root)
    out = []
    canonical = [name for name, _ in SECTIONS]

    seen = [slide.section for slide in deck]
    for slide in deck:
        if slide.section not in canonical:
            out.append(_v(slide.number, slide.section, -1, "section_known",
                          slide.section, "not a section of the team standard"))
    ordered = [s for s in seen if s in canonical]
    if ordered != sorted(ordered, key=canonical.index):
        out.append(_v(0, "", -1, "section_order", " > ".join(ordered),
                      "sections run in the team standard's order"))
    for required in REQUIRED_SECTIONS:
        if required not in seen:
            out.append(_v(0, "", -1, "section_required", required,
                          "a deck may not drop this section"))
    numbers = [slide.number for slide in deck]
    if numbers != list(range(1, len(deck) + 1)):
        out.append(_v(0, "", -1, "slide_numbering", numbers,
                      "slides number 1..n without gaps"))

    for slide in deck:
        out.extend(lint_slide(slide, ledger, facts, repo_root, member_root,
                              corrections, countries))
    return pd.DataFrame(out, columns=list(LINT_COLUMNS))


# ---------------------------------------------------------------------------
# E. The question bank
# ---------------------------------------------------------------------------
#
# The brief's Task 10 is a presentation *and* a mentor review: "answer
# questions on methods and results". The answers are the part that goes wrong,
# because a question is asked at speed and the honest answer to most of the
# interesting ones is a refusal. A presenter improvising under that pressure
# reaches for the sentence the gates already rejected — which is precisely the
# `tempting` family Task 09 generated for exactly this moment.
#
# So the bank is built out of the same bullets the slides are built from, and
# it is linted by the same function. An answer here is not a rehearsed line;
# it is a pointer to a committed row, which is why it survives being asked in
# a different order than it was written.


@dataclass(frozen=True)
class Question:
    """One question, its backing slide, and the bound answer."""

    qid: str
    slide: int
    asked: str
    answer: tuple[Bullet, ...]
    carries: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.answer:
            raise ValueError(f"question {self.qid!r} has no answer")


def qa_bank(focus: str = "google") -> tuple[Question, ...]:
    """Every question a mentor is likely to ask, with a bound answer.

    Ordered by how early in a review each one tends to arrive, not by slide.
    """
    return (
        Question(
            "q-forecast", 9,
            "So can you forecast their hiring for next quarter?",
            (refusal("tempting-forecast"),
             refusal("forecast-h1"),
             plain("The gate says the series is not noise. The horizon table "
                   "says no model beats persistence, so the answer is no at "
                   "every horizon tested.")),
        ),
        Question(
            "q-who-is-biggest", 8,
            "Which of the six is hiring the most?",
            (refusal("tempting-level"),
             claim("share-google"),
             plain("Share of a fixed publisher pool is the comparable "
                   "quantity. The level is not, because a posting count "
                   "counts syndication.")),
            carries=("C8",),
        ),
        Question(
            "q-direction", 7,
            "Is the focus company hiring more or less than at the start of "
            "the year?",
            (refusal("vol-google"),
             plain("Four defensible panel treatments disagree on the sign, so "
                   "no direction is reported for this company. Two of the six "
                   "do survive that test, and they are reported."),),
        ),
        Question(
            "q-february", 6,
            "What happened in February?",
            (refusal("tempting-seasonal"),
             plain("The publishers that vanish in February return in March, "
                   "so it is a collection gap. With one year there is no way "
                   "to test a recurring pattern at all."),),
        ),
        Question(
            "q-converging", 10,
            "Are the two most similar companies converging?",
            (refusal("tempting-convergence"),
             claim("pair-google-meta"),
             plain("Similarity of profiles is measured and reported. Movement "
                   "of profiles towards each other is not: it failed its own "
                   "null.")),
        ),
        Question(
            "q-salary", 14,
            "Who pays more?",
            (refusal("salary-google-meta-science-research"),
             plain("Disclosure is a publisher behaviour, not a company one, "
                   "so the disclosed subset is not a sample of the "
                   "population. There is no pay comparison in this data."),),
        ),
        Question(
            "q-country", 14,
            "How does hiring differ between countries?",
            (refusal("tempting-country"),
             plain("Publishers are regional. A country split would compare "
                   "aggregator footprints and read as a hiring shift, which "
                   "is the most convincing wrong answer available here."),),
        ),
        Question(
            "q-headcount", 6,
            "Does this tell you whether the company grew?",
            (refusal("tempting-headcount"),
             plain("The collected schema holds postings, not people. There "
                   "is no column anywhere in it for the number of employees, "
                   "so no amount of care with the postings recovers one."),),
        ),
        Question(
            "q-product", 13,
            "Can you tell what they are about to launch?",
            (refusal("tempting-product"),
             claim("distinct-bigquery"),
             plain("A distinctive skill is a distinctive advert. Reading a "
                   "launch out of it crosses from what was measured to what "
                   "was never measured.")),
        ),
        Question(
            "q-why-shift", 7,
            "Why did the job function mix shift?",
            (refusal("tempting-strategy"),
             plain("The mix change is observed; the intention behind it is "
                   "not in the data. The honest form is what changed, not "
                   "why."),),
        ),
        Question(
            "q-denominator", 5,
            "Why are skill shares out of skilled postings rather than all of "
            "them?",
            (refusal("tempting-share-of-all"),
             plain("Because the description text is truncated, so whether a "
                   "posting mentions a skill partly measures how much text "
                   "the source returned."),),
        ),
        Question(
            "q-simpsons", 15,
            "Has any of your own headlines been wrong?",
            (claim("skilltrend-looker"),
             fact("Yes — {corrections_total} times, and each one is on the "
                  "record with the table that overturned it."),
             plain("This one reversed inside every job function while the "
                   "pooled figure ran the other way. The within-function "
                   "direction is the identified one.")),
        ),
        Question(
            "q-sample", 4,
            "Is that enough postings to say anything?",
            (fact("For the focus company, {postings_focus} postings across "
                  "{publishers_focus} publishers, present in all "
                  "{months_observed} months."),
             plain("Support is not assumed anywhere: cells below the floor "
                   "are dropped rather than reported thinly, and the floor "
                   "and its casualties are both committed."),),
        ),
        Question(
            "q-why-rules", 5,
            "Why rule-based extraction and not a language model?",
            (fact("Because the source truncates description text, so the "
                  "free-text layer is validated and idle; the "
                  "{skills_observed} skills come from structured fields and "
                  "aliases."),
             plain("A model trained on truncated text would learn the "
                   "truncation. The rules at least fail visibly, and any one "
                   "extraction can be checked by reading it."),),
        ),
        Question(
            "q-scraping", 3,
            "Did you scrape any of this?",
            (plain("No. Every source is an API on the approved list, and the "
                   "one obvious scrape for this company was rejected on "
                   "robots.txt before any collection."),
             plain("A key for a further job board sits in the environment "
                   "file and was never called, because that source never "
                   "passed the checklist."),),
        ),
        Question(
            "q-privacy", 3,
            "Is there any personal data in here?",
            (plain("No. The privacy check runs every task and fails the build "
                   "if a committed table gains a column that names a person."),
             plain("Row-level postings stay out of the repository entirely; "
                   "what is committed is aggregate."),),
        ),
        Question(
            "q-investor", 14,
            "What would you tell an investor?",
            (plain("Nothing from this evidence base. Every investor question "
                   "asks for a level, a trajectory, or a forecast, and all "
                   "three are refused."),
             plain("That is a finding about the source, not modesty: an "
                   "aggregator's syndication footprint cannot be read as a "
                   "company's size or direction."),),
        ),
        Question(
            "q-useful", 13,
            "So what is any of it actually good for?",
            (claim("skillgap-spark-google-databricks"),
             fact("{claims_published} sentences publish out of "
                  "{claims_generated} generated, and the skill-profile ones "
                  "are where the yield concentrates."),
             plain("Skill profiles and stack differences survive; volume, "
                   "pay, and prediction do not.")),
        ),
        Question(
            "q-reproduce", 16,
            "Could someone else rebuild this?",
            (fact("Yes — {tests_total} tests, {tables_committed} committed "
                  "tables and {figures_committed} figures, and the core "
                  "modules import no solver library."),
             plain("The statistics are hand-rolled and cross-checked against "
                   "a library in separate validation scripts, so a rebuild "
                   "does not depend on matching a version."),),
        ),
        Question(
            "q-next", 18,
            "What is the single change that would help most?",
            (plain("More months. Almost every refusal here traces to one year "
                   "of one source: no seasonality, no backtest worth the "
                   "name, and a panel thin enough to decide the sign."),
             plain("More publishers per month is second, because it is the "
                   "panel treatment, not the company, that currently "
                   "determines the volume direction."),),
        ),
        Question(
            "q-team", 17,
            "How does this fit with the other three companies?",
            (fact("The comparison set covers {companies_compared} companies "
                  "and {postings_pool} postings, and the shared code is "
                  "company-agnostic."),
             plain("Three seats are open, so the set was chosen by one "
                   "person. The matching audit is committed precisely so a "
                   "new specialist can overturn a row rather than redo the "
                   "task."),),
        ),
    )


def render_question(question: Question, ledger: pd.DataFrame,
                    facts: dict) -> dict:
    return {
        "qid": question.qid,
        "slide": question.slide,
        "asked": question.asked,
        "answer": [render_bullet(b, ledger, facts) for b in question.answer],
        "carries": list(question.carries),
    }


def lint_qa_bank(bank, deck, ledger: pd.DataFrame, facts: dict,
                 member_root: Path = DEFAULT_MEMBER) -> pd.DataFrame:
    """The bank is checked the way the deck is, plus two rules of its own."""
    countries = ins.country_vocabulary(member_root)
    out = []
    numbers = {slide.number for slide in deck}
    seen_sections = set()
    by_number = {slide.number: slide for slide in deck}

    for question in bank:
        here = lambda i, rule, subject, detail, q=question: _v(
            q.slide, q.qid, i, rule, subject, detail)
        if question.slide not in numbers:
            out.append(here(-1, "qa_slide_exists", question.slide,
                            "answer points at a slide the deck does not have"))
        else:
            seen_sections.add(by_number[question.slide].section)
        if not question.asked.strip().endswith("?"):
            out.append(here(-1, "qa_is_a_question", question.asked,
                            "a bank entry is written as the mentor asks it"))
        out.extend(lint_bullets(question.answer, question.carries, ledger,
                                facts, here, countries))

    for required in REQUIRED_SECTIONS:
        if required not in seen_sections:
            out.append(_v(0, "", -1, "qa_covers_required", required,
                          "no banked question backs this mandatory section"))
    ids = [q.qid for q in bank]
    for qid in sorted({i for i in ids if ids.count(i) > 1}):
        out.append(_v(0, "", -1, "qa_ids_unique", qid, "duplicate question id"))
    return pd.DataFrame(out, columns=list(LINT_COLUMNS))


# ---------------------------------------------------------------------------
# F. The workspace audit
# ---------------------------------------------------------------------------
#
# Task 10's second deliverable is "a complete, well-structured GitHub
# workspace — organised folders, clean notebooks, updated READMEs". Every word
# of that is checkable, and the interesting one is *updated*. A README goes
# stale the commit after it is written, in a way no reader notices: the number
# was true when typed. So the audit's centrepiece is not a spell-check, it is
# the rule that the most recent task's quoted suite size must equal the suite
# that exists now.

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)")
BACKTICK_PATH = re.compile(r"`([A-Za-z0-9_./-]+\.(?:md|py|csv|json))`")
DONE = "✅"
OPEN = "⬜"
TASK_STATUS_ROW = re.compile(r"^\|\s*(\d{2})\s*\|.*?\|.*?\|\s*(.*?)\s*\|\s*$",
                             re.MULTILINE)
QUOTED_TESTS = re.compile(r"([\d,]+)\s+tests")

#: Directories that exist to hold work rather than to hold a placeholder. A
#: folder containing only `.gitkeep` or `TEMPLATE.md` is an intention, and the
#: brief asks for a finished workspace, so the audit names them.
WORKING_DIRS = ("notebooks", "weekly-reports", "meeting-minutes")
PLACEHOLDER_NAMES = {".gitkeep", "TEMPLATE.md", "README.md"}

AUDIT_COLUMNS = ("area", "check", "subject", "status", "detail")


def _a(area, check, subject, status, detail) -> dict:
    return {"area": area, "check": check, "subject": str(subject),
            "status": status, "detail": detail}


def tracked_files(repo_root: Path = REPO_ROOT) -> list:
    """What git actually carries, which is the only definition that matters."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=repo_root,
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def readme_task_rows(repo_root: Path = REPO_ROOT) -> list:
    """The root README's task table, as (number, status cell) pairs."""
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    return [(int(n), cell) for n, cell in TASK_STATUS_ROW.findall(text)]


def workspace_audit(repo_root: Path = REPO_ROOT,
                    member_root: Path = DEFAULT_MEMBER) -> pd.DataFrame:
    """Is the repository the one the README says it is?"""
    rows = []
    add = lambda *args: rows.append(_a(*args))
    tracked = set(tracked_files(repo_root))

    # -- the task table describes files that exist -------------------------
    task_rows = readme_task_rows(repo_root)
    for number, cell in task_rows:
        marked_done = DONE in cell
        # A task is backed by whatever it produced. Some tasks land as a team
        # standard in docs/ and some as a member report; requiring both would
        # mark the legal review incomplete for having no member deliverable.
        quoted = BACKTICK_PATH.findall(cell)
        present = [q for q in quoted if (repo_root / q).exists()]
        reports = sorted(member_root.glob(f"task-{number:02d}-*report*.md"))
        if marked_done and not present:
            add("readme", "task_row_backed", f"task {number:02d}", "fail",
                "row is marked done and names no deliverable that exists")
        elif not marked_done and (present or reports):
            add("readme", "task_row_backed", f"task {number:02d}", "fail",
                "a deliverable exists for a row still marked open")
        else:
            add("readme", "task_row_backed", f"task {number:02d}", "pass",
                f"{DONE if marked_done else OPEN} with "
                f"{len(present)} deliverable(s) on disk")
        for reference in quoted:
            status = "pass" if (repo_root / reference).exists() else "fail"
            add("readme", "task_row_path", reference, status,
                "path quoted in the task table")

    # -- the most recent task's numbers are the current numbers ------------
    # Earlier rows are historical snapshots and cannot be checked without
    # archaeology; the newest row describes the repository as it stands, so
    # that one is checkable and is the one that goes stale.
    done = [(n, cell) for n, cell in task_rows if DONE in cell]
    if done:
        number, cell = max(done)
        quoted = [int(v.replace(",", "")) for v in QUOTED_TESTS.findall(cell)]
        actual = count_tests(repo_root)
        if not quoted:
            add("readme", "suite_size_current", f"task {number:02d}", "fail",
                "the newest task row quotes no suite size")
        elif actual in quoted:
            add("readme", "suite_size_current", f"task {number:02d}", "pass",
                f"quotes {actual} tests, and the suite collects {actual}")
        else:
            add("readme", "suite_size_current", f"task {number:02d}", "fail",
                f"quotes {quoted[0]} tests; the suite collects {actual}")

    # -- links resolve -----------------------------------------------------
    broken = 0
    checked = 0
    for relative in sorted(f for f in tracked if f.endswith(".md")):
        path = repo_root / relative
        for target in MD_LINK.findall(path.read_text(encoding="utf-8")):
            target = target.split("#")[0].strip()
            if not target or target.startswith(("http", "mailto:")):
                continue
            checked += 1
            if not (path.parent / target).exists():
                broken += 1
                add("docs", "link_resolves", f"{relative} → {target}", "fail",
                    "relative link does not resolve")
    add("docs", "link_resolves", f"{checked} relative links", 
        "pass" if not broken else "fail",
        f"{checked - broken} of {checked} resolve")

    # -- every member folder introduces itself -----------------------------
    for member in sorted(p for p in (repo_root / "members").iterdir()
                         if p.is_dir()):
        status = "pass" if (member / "README.md").exists() else "fail"
        add("members", "member_readme", member.name, status,
            "a member folder carries its own README")

    # -- folders that are still only an intention --------------------------
    for name in WORKING_DIRS:
        directory = repo_root / name
        if not directory.exists():
            add("folders", "working_dir_used", name, "fail",
                "directory named in the layout does not exist")
            continue
        contents = [p.name for p in directory.iterdir()
                    if p.name not in PLACEHOLDER_NAMES]
        add("folders", "working_dir_used", name,
            "pass" if contents else "fail",
            f"{len(contents)} file(s) beyond the placeholder")

    # -- secrets and row-level data stay out -------------------------------
    add("privacy", "env_untracked", ".env",
        "fail" if ".env" in tracked else "pass",
        "the credentials file is never committed")
    leaked = [f for f in tracked if f.startswith("data/")
              and not f.endswith((".gitkeep", "README.md"))]
    add("privacy", "row_data_untracked", f"{len(leaked)} file(s)",
        "fail" if leaked else "pass",
        "row-level data under data/ stays git-ignored")

    # -- the standing privacy check, run over everything committed ---------
    offenders = []
    tables = sorted(repo_root.glob("members/*/task-*-tables/*.csv"))
    for table in tables:
        try:
            found = ins.personal_data_columns_present(pd.read_csv(table, nrows=1))
        except (OSError, ValueError, pd.errors.ParserError):
            found = []
        if found:
            offenders.append(f"{table.name}: {', '.join(found)}")
    add("privacy", "personal_data_columns_present", f"{len(tables)} tables",
        "fail" if offenders else "pass",
        "; ".join(offenders) if offenders
        else "no committed table names a person")

    return pd.DataFrame(rows, columns=list(AUDIT_COLUMNS))


# ---------------------------------------------------------------------------
# G. Output — the deck, and the tables that prove it
# ---------------------------------------------------------------------------
#
# The deck ships as markdown rather than as a binary slide file, for the same
# reason the tables ship as CSV: a reviewer has to be able to diff it. Slide
# software is the last step and it is a formatting step; the argument is here,
# and so is every check on it.


def bullet_table(deck, ledger: pd.DataFrame, facts: dict) -> pd.DataFrame:
    """One row per bullet, and where its authority comes from."""
    rows = []
    for slide in deck:
        for i, bullet in enumerate(slide.bullets):
            line = _safe_render(bullet, ledger, facts)
            source, status, citation = "", "", ""
            if bullet.kind in LEDGER_KINDS:
                match = ledger[ledger["claim_id"].astype(str) == bullet.claim_id]
                if len(match) == 1:
                    row = match.iloc[0]
                    status = str(row["status"])
                    citation = str(row["citation"])
                    source = str(row["source_task"])
            elif bullet.kind == "fact":
                keys = PLACEHOLDER.findall(bullet.text)
                source = "; ".join(FACTS_BY_KEY[k].source for k in keys
                                   if k in FACTS_BY_KEY)
                citation = ", ".join(keys)
            else:
                _, structural = strip_structural(bullet.text)
                source = "narration"
                citation = ", ".join(sorted(set(structural)))
            rows.append({
                "slide": slide.number,
                "section": slide.section,
                "bullet": i,
                "kind": bullet.kind,
                "claim_id": bullet.claim_id,
                "status": status,
                "source": source,
                "citation": citation,
                "characters": len(line),
                "delivery": "read from slide" if len(line) > SPOKEN_MAX
                            else "spoken",
                "text": line,
            })
    return pd.DataFrame(rows)


def slide_table(deck, ledger: pd.DataFrame, facts: dict) -> pd.DataFrame:
    """One row per slide: what it says, what it shows, what it carries."""
    rows = []
    for slide in deck:
        kinds = [b.kind for b in slide.bullets]
        rows.append({
            "slide": slide.number,
            "section": slide.section,
            "required": slide.section in REQUIRED_SECTIONS,
            "title": slide.title,
            "bullets": len(slide.bullets),
            "claims": kinds.count("claim"),
            "refusals": kinds.count("refusal"),
            "facts": kinds.count("fact"),
            "narration": kinds.count("plain"),
            "figure": slide.figure,
            "table": slide.table,
            "carries": ", ".join(slide.carries),
            "notes": slide.notes,
        })
    return pd.DataFrame(rows)


def section_table(deck) -> pd.DataFrame:
    """The team standard's slot list, and how this deck filled it."""
    filled = {}
    for slide in deck:
        filled.setdefault(slide.section, []).append(slide.number)
    rows = []
    for order, (name, purpose) in enumerate(SECTIONS, start=1):
        slides = filled.get(name, [])
        rows.append({
            "order": order,
            "section": name,
            "purpose": purpose,
            "required": name in REQUIRED_SECTIONS,
            "slides": ", ".join(str(n) for n in slides),
            "present": bool(slides),
        })
    return pd.DataFrame(rows)


def qa_table(bank, ledger: pd.DataFrame, facts: dict) -> pd.DataFrame:
    """The question bank, one row per answer line."""
    rows = []
    for question in bank:
        for i, bullet in enumerate(question.answer):
            line = _safe_render(bullet, ledger, facts)
            status = ""
            if bullet.kind in LEDGER_KINDS:
                match = ledger[ledger["claim_id"].astype(str) == bullet.claim_id]
                if len(match) == 1:
                    status = str(match.iloc[0]["status"])
            rows.append({
                "question_id": question.qid,
                "backs_slide": question.slide,
                "question": question.asked,
                "line": i,
                "kind": bullet.kind,
                "claim_id": bullet.claim_id,
                "status": status,
                "carries": ", ".join(question.carries),
                "answer": line,
            })
    return pd.DataFrame(rows)


def answer_shape(bank) -> pd.DataFrame:
    """How the bank answers: how many questions end in a refusal, and where.

    The point of counting is that a review is won or lost on this ratio. If
    most of the questions a mentor asks are answered with a published claim,
    either the questions are soft or the gates are.
    """
    rows = []
    for question in bank:
        kinds = [b.kind for b in question.answer]
        leads = question.answer[0].kind
        rows.append({
            "question_id": question.qid,
            "question": question.asked,
            "backs_slide": question.slide,
            "opens_with": leads,
            "refusals": kinds.count("refusal"),
            "claims": kinds.count("claim"),
            "facts": kinds.count("fact"),
            "narration": kinds.count("plain"),
            "verdict": "refused" if leads == "refusal"
                       else "answered" if kinds.count("claim") or kinds.count("fact")
                       else "explained",
        })
    return pd.DataFrame(rows)


def _md_escape(text: str) -> str:
    return text.replace("|", r"\|")


def deck_markdown(deck, bank, ledger: pd.DataFrame, facts: dict,
                  focus: str = "google") -> str:
    """The deck as a reviewable document.

    Each slide prints its bullets, its attachment, its speaker notes and,
    where it carries one, the correction it must not be presented without.
    """
    long = long_bullets(deck, ledger, facts)
    out = [
        f"# Task 10 — Final Presentation ({focus.title()})",
        "",
        "One deck, built from the claim ledger rather than written alongside "
        "it. Every bullet below is one of four kinds, and each kind is checked "
        "differently:",
        "",
        "- **claim** — reproduced character for character from a published row "
        "of [`task-09-tables/claim-ledger.csv`](task-09-tables/claim-ledger.csv), "
        "so its qualifying clause cannot be dropped on the way to a slide.",
        "- **refusal** — a sentence this project declines to make, quoted with "
        "the gate that stopped it.",
        "- **fact** — a template over the repository, with the numbers "
        "recomputed at build time. No number is typed here.",
        "- **narration** — prose, and it may carry no numeral beyond a task "
        "number, a section mark or the collection year.",
        "",
        "The slot order is the team standard in "
        "[`docs/task-10-final-presentation-standard.md`]"
        "(../../docs/task-10-final-presentation-standard.md); the checks are "
        "in [`src/present.py`](../../src/present.py) and "
        "[`tests/test_presentation.py`](../../tests/test_presentation.py).",
        "",
        "---",
        "",
    ]

    for slide in deck:
        rendered = render_slide(slide, ledger, facts)
        out.append(f"## Slide {slide.number} — {slide.title}")
        out.append("")
        out.append(f"*Section: `{slide.section}`"
                   + ("  ·  **mandatory in every deck**"
                      if slide.section in REQUIRED_SECTIONS else "")
                   + "*")
        out.append("")
        for bullet, line in zip(slide.bullets, rendered["lines"]):
            marker = {"claim": "", "refusal": "**Refused.** ",
                      "fact": "", "plain": ""}[bullet.kind]
            tail = ""
            if bullet.kind in LEDGER_KINDS:
                tail = f" `[{bullet.claim_id}]`"
                if bullet.claim_id in long:
                    tail += " *(read from the slide, do not recite)*"
            body = line
            if bullet.kind == "refusal":
                body = line.replace("Not available — ", "", 1)
            out.append(f"- {marker}{body}{tail}")
        out.append("")
        if slide.figure:
            out.append(f"![{slide.title}]({slide.figure})")
            out.append("")
        if slide.table:
            prefix = "../../" if slide.table.startswith("docs/") else ""
            out.append(f"Evidence: [`{slide.table}`]({prefix}{slide.table})")
            out.append("")
        if slide.carries:
            for cid in slide.carries:
                out.append(f"> **Carries {cid}.** This slide may not be shown "
                           f"without it — see "
                           f"[`docs/corrections.md`](../../docs/corrections.md).")
            out.append("")
        out.append(f"**Notes.** {slide.notes}")
        out.append("")
        backing = [q for q in bank if q.slide == slide.number]
        if backing:
            out.append("**Backed by:** "
                       + ", ".join(f"`{q.qid}`" for q in backing))
            out.append("")
        out.append("---")
        out.append("")

    out.append("## Question bank")
    out.append("")
    out.append("Answers are bound the same way bullets are. A question with no "
               "bound answer is answered by saying this evidence base does not "
               "settle it — never by improvising a number.")
    out.append("")
    for question in bank:
        out.append(f"**{question.qid}** (slide {question.slide}) — "
                   f"*{_md_escape(question.asked)}*")
        out.append("")
        for bullet in question.answer:
            line = _safe_render(bullet, ledger, facts)
            tail = f" `[{bullet.claim_id}]`" if bullet.claim_id else ""
            out.append(f"- {line}{tail}")
        if question.carries:
            out.append(f"- Carries {', '.join(question.carries)}.")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
