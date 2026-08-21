"""Task 09 — insight generation and reporting.

Shared, company-agnostic layer. Tasks 05 to 08 each asked whether a particular
*number* is identified. Task 09 asks the question that decides whether any of
it reaches a reader: **which sentences may this repository print?**

That sounds like a writing problem. It is not. Eight tasks in, this project has
raised seven corrections, and every one of them has the same shape — a sentence
outran the table under it. C1 read a monthly minimum as a hiring signal. C2
found five of ten skill movers were mix. C6 predicted rare skills would
dominate a score they contribute exactly 0% of. A ninth task that writes prose
by hand would raise the eighth correction the same way, except this time there
is no Task 10 analysis left to catch it — Task 10 is the presentation.

So an insight here is not a sentence. It is a record:

    text            the sentence, generated from the cell it describes
    citation        table#column@selector, resolvable against a committed CSV
    verdict_source  the upstream verdict that decides whether it may be said
    status          published / published_qualified / refused
    clause          the qualifier that must travel with it
    falsifier       the observation that would overturn it
    action          what a named audience would do differently

and it passes four gates before it is publishable:

1. **Evidence** — the citation resolves and the quoted value matches the cell.
   A sentence whose number cannot be found in a committed table is not a weak
   insight, it is not an insight.
2. **Lint** — the text contains no prohibited construction. The prohibitions
   are not style preferences; each one is a rule some earlier task earned.
   §8 of Task 07 forbids the forecast sentence. §8 of Task 08 forbids the
   convergence sentence. Task 05 §9 forbids the unstratified skill trend.
   Task 06 §1.3 forbids the cross-company level. Here they are regexes.
3. **Identification** — the claim inherits a verdict from the task that
   computed it. Task 09 never re-decides whether something is identified; it
   looks the answer up. A claim that asserts its own confidence is the failure
   mode this gate exists to stop.
4. **Consistency** — where two tasks quote the same quantity, the later one
   wins and the earlier value may appear only with its correction pointer.
   The repository currently holds three such pairs.

`published` and `published_qualified` do not divide the survivors into
caveated and clean ones. **Every** publishable claim carries a clause — a
sentence that reaches the ledger without one is refused as a drafting fault,
and none of the 436 proposed claims did. What the two statuses separate is the
strength of the *upstream verdict*: `published` inherits a finding an earlier
task called confirmed or robust, `published_qualified` inherits one that task
itself hedged, and the reader is owed both facts. This evidence base yields no
context-free sentence, which is a result rather than an inconvenience.

The proposed claims are **generated**, not written, one per row of the
upstream verdict tables. That is what makes the yield honest: the denominator
is everything this evidence base could be asked to say, not everything the
author happened to think of.

Everything runs on pandas, numpy, `math` and `re`, in keeping with Tasks 05 to
08 — no scipy in a core module, so any reviewer can rebuild every published
number without matching a solver version.

    python src/build_insights.py
    python -m pytest tests/ -q
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import compare as cmp

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMBERS = REPO_ROOT / "members"

#: Default evidence root. Every citation resolves relative to one specialist's
#: folder, so a second specialist runs this module against their own without
#: touching the code.
DEFAULT_MEMBER = MEMBERS / "ankit-google"

PUBLISHER_COL = cmp.PUBLISHER_COL
FORBIDDEN_COLUMNS = cmp.FORBIDDEN_COLUMNS

#: The four questions the brief asks Task 09 to answer, plus the fifth the
#: deliverable list adds ("a short explanation of the company's hiring strategy
#: & position"). Every candidate claim is assigned to exactly one, so the
#: coverage table can show which questions the evidence base leaves empty.
QUESTIONS = (
    "hiring_patterns",
    "skill_demand",
    "tech_stack",
    "future_demand",
    "position",
)

#: The readers the brief names. An insight with no audience is a fact, not an
#: insight, so `actionability_table` scores a claim partly on having one.
AUDIENCES = ("strategy", "hr_talent", "product", "exec", "investor")

PUBLISHED = "published"
QUALIFIED = "published_qualified"
REFUSED = "refused"
STATUSES = (PUBLISHED, QUALIFIED, REFUSED)

#: Tolerance when a claim's quoted value is checked against its cell. Values in
#: the committed tables are rounded to 4 decimals at write time and quoted at 2
#: to 4 in prose, so the binder compares on a relative scale with an absolute
#: floor rather than demanding an exact match.
BIND_RTOL = 0.02
BIND_ATOL = 5e-4


# ---------------------------------------------------------------------------
# A. Citations
# ---------------------------------------------------------------------------


_CITATION_RE = re.compile(
    r"^(?P<table>[\w./-]+\.csv)"
    r"#(?P<column>[\w.]+)"
    r"(?:@(?P<selector>.+))?$"
)


@dataclass(frozen=True)
class Citation:
    """A pointer to one cell of one committed table.

    Written as ``task-06-tables/relative-share-verdict.csv#verdict@company=nvidia``.
    The selector is a comma-separated list of ``column=value`` pairs that must
    reduce the table to exactly one row — "exactly" being the point. A selector
    that matches three rows is a claim about three things quoted as if it were
    about one, which is how a mix effect gets published as a trend.
    """

    table: str
    column: str
    selector: tuple[tuple[str, str], ...] = ()

    def __str__(self) -> str:
        base = f"{self.table}#{self.column}"
        if self.selector:
            base += "@" + ",".join(f"{k}={v}" for k, v in self.selector)
        return base


def parse_citation(text: str) -> Citation:
    """Parse ``table.csv#column@key=value,key=value``."""
    match = _CITATION_RE.match(text.strip())
    if not match:
        raise ValueError(f"not a citation: {text!r}")
    raw = match.group("selector")
    selector: list[tuple[str, str]] = []
    if raw:
        for part in raw.split(","):
            if "=" not in part:
                raise ValueError(f"selector term needs '=': {part!r} in {text!r}")
            key, value = part.split("=", 1)
            selector.append((key.strip(), value.strip()))
    return Citation(match.group("table"), match.group("column"),
                    tuple(selector))


def read_table(citation: Citation, root: Path) -> pd.DataFrame:
    """Load the CSV a citation points at, with a readable failure."""
    path = root / citation.table
    if not path.is_file():
        raise FileNotFoundError(f"no committed table at {path}")
    return pd.read_csv(path)


def select_row(table: pd.DataFrame, citation: Citation) -> pd.Series:
    """Reduce a table to the single row a citation's selector names."""
    frame = table
    for key, value in citation.selector:
        if key not in frame.columns:
            raise KeyError(f"{citation.table} has no column {key!r}")
        column = frame[key].astype(str).str.strip()
        frame = frame[column.str.lower() == str(value).strip().lower()]
    if len(frame) != 1:
        raise LookupError(
            f"selector on {citation.table} matched {len(frame)} rows, need 1"
        )
    return frame.iloc[0]


def resolve_citation(citation: Citation | str, root: Path = DEFAULT_MEMBER):
    """Return the value a citation points at."""
    if isinstance(citation, str):
        citation = parse_citation(citation)
    table = read_table(citation, root)
    if citation.column not in table.columns:
        raise KeyError(f"{citation.table} has no column {citation.column!r}")
    if not citation.selector:
        if len(table) != 1:
            raise LookupError(
                f"{citation.table} has {len(table)} rows and the citation has "
                "no selector"
            )
        return table.iloc[0][citation.column]
    return select_row(table, citation)[citation.column]


# ---------------------------------------------------------------------------
# B. The claim record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """One candidate sentence, bound to the cell that would justify it.

    ``value`` is what the sentence asserts; ``citation`` is where that number
    lives. They are separate fields on purpose — the binder's whole job is to
    notice when they stop agreeing, which is what a rebuild does silently.
    """

    claim_id: str
    question: str
    family: str
    subject: str
    measures: str
    text: str
    citation: str
    value: float | str | None = None
    verdict_source: str = ""
    clause: str = ""
    falsifier: str = ""
    action: str = ""
    audience: str = ""
    source_task: str = ""
    depends_on: tuple[str, ...] = ()
    refusal_rule: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.question not in QUESTIONS:
            raise ValueError(
                f"{self.claim_id}: question {self.question!r} is not one of "
                f"{QUESTIONS}"
            )
        if self.audience and self.audience not in AUDIENCES:
            raise ValueError(
                f"{self.claim_id}: audience {self.audience!r} is not one of "
                f"{AUDIENCES}"
            )


def claim_frame(claims) -> pd.DataFrame:
    """Flatten claims into the ledger's base columns."""
    rows = []
    for claim in claims:
        row = dict(vars(claim))
        row["depends_on"] = ", ".join(claim.depends_on)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C. Gate 1 — evidence binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """One gate's verdict on one claim."""

    name: str
    passed: bool
    detail: str = ""


def _numeric(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def values_agree(quoted, actual,
                 rtol: float = BIND_RTOL, atol: float = BIND_ATOL) -> bool:
    """Does a quoted value match the cell it cites?

    Numbers compare on a relative tolerance with an absolute floor, because a
    share quoted as 0.0051 and a share stored as 0.00514 are the same claim
    while 4.84 and 6.56 are not. Strings compare case-folded and stripped.
    """
    q_num, a_num = _numeric(quoted), _numeric(actual)
    if q_num is not None and a_num is not None:
        return abs(q_num - a_num) <= max(atol, rtol * abs(a_num))
    if quoted is None:
        return False
    return str(quoted).strip().lower() == str(actual).strip().lower()


#: Columns in the committed tables that record a direction, in the order the
#: binder should trust them. The stratified verdict beats the pooled direction:
#: that ordering *is* C2, written down.
DIRECTION_COLUMNS = ("verdict", "direction", "pooled_direction",
                     "direction_balanced")

#: Vocabulary that commits a sentence to a direction. Deliberately narrow —
#: a word that only sometimes means "up" would make the check unreliable, and
#: an unreliable gate gets switched off.
_UP_WORDS = re.compile(
    r"\b(rising|rises|rose|emerging|emerges|growing|grew|grows|up\s+to"
    r"|increasing|increased|gaining|gained|climbing|climbed)\b", re.I)
_DOWN_WORDS = re.compile(
    r"\b(falling|falls|fell|declining|declined|shrinking|shrank|dropping"
    r"|dropped|losing|lost|down\s+to|contracting|contracted)\b", re.I)


#: Ordinal suffixes, because ``f"{rank}th"`` writes "2th".
_ORDINAL_SUFFIX = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(n: int) -> str:
    """``2`` -> ``2nd``, and the teens, which take ``th`` regardless."""
    suffix = "th" if 11 <= (n % 100) <= 13 else _ORDINAL_SUFFIX.get(n % 10, "th")
    return f"{n}{suffix}"


def asserted_direction(text: str) -> str:
    """The direction a sentence commits to: ``up``, ``down`` or ``""``."""
    up, down = bool(_UP_WORDS.search(text)), bool(_DOWN_WORDS.search(text))
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return ""


def recorded_direction(row) -> tuple[str, str]:
    """The direction the cited row records, and the column it came from."""
    for column in DIRECTION_COLUMNS:
        if column not in row.index:
            continue
        value = str(row[column]).strip().lower()
        if not value or value == "nan":
            continue
        if value.startswith("rising") or value in ("up", "growth", "gaining",
                                                   "gaining share"):
            return "up", column
        if value.startswith("falling") or value in ("down", "decline",
                                                    "losing", "losing share"):
            return "down", column
    return "", ""


def direction_agrees(claim: Claim, root: Path = DEFAULT_MEMBER) -> Gate:
    """Does the sentence point the same way as the table it cites?

    This is C1 and C2 as a check. Both corrections were sentences that agreed
    with a *pooled* number and disagreed with the stratified verdict sitting in
    the same row — "Looker is emerging" against ``falling_in_all_segments``.
    Nothing about the arithmetic catches that; only reading the direction the
    table itself recorded does.
    """
    stated = asserted_direction(claim.text)
    if not stated:
        return Gate("direction", True, "")
    try:
        citation = parse_citation(claim.citation)
        row = select_row(read_table(citation, root), citation)
    except (FileNotFoundError, KeyError, LookupError, ValueError):
        return Gate("direction", True, "no row to compare against")
    recorded, column = recorded_direction(row)
    if not recorded or recorded == stated:
        return Gate("direction", True, column)
    return Gate("direction", False,
                f"text says {stated}, {citation.table} records "
                f"{recorded} in {column}")


def bind_claim(claim: Claim, root: Path = DEFAULT_MEMBER) -> Gate:
    """Gate 1. Resolve the citation and compare it to the quoted value."""
    try:
        actual = resolve_citation(claim.citation, root)
    except (FileNotFoundError, KeyError, LookupError, ValueError) as exc:
        return Gate("evidence", False, f"unresolved: {exc}")
    if claim.value is None:
        return Gate("evidence", True, f"resolved to {actual!r}, nothing quoted")
    if values_agree(claim.value, actual):
        return Gate("evidence", True, f"{claim.value!r} == {actual!r}")
    return Gate("evidence", False,
                f"claim quotes {claim.value!r}, table holds {actual!r}")


def evidence_bindings(claims, root: Path = DEFAULT_MEMBER) -> pd.DataFrame:
    """One row per claim: what it cites, what the table holds, whether they agree."""
    rows = []
    for claim in claims:
        citation = parse_citation(claim.citation)
        try:
            actual = resolve_citation(citation, root)
            error = ""
        except (FileNotFoundError, KeyError, LookupError, ValueError) as exc:
            actual, error = None, str(exc)
        gate = bind_claim(claim, root)
        rows.append({
            "claim_id": claim.claim_id,
            "table": citation.table,
            "column": citation.column,
            "selector": ",".join(f"{k}={v}" for k, v in citation.selector),
            "quoted": claim.value,
            "table_value": actual,
            "bound": gate.passed,
            "detail": error or gate.detail,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D. Gate 2 — the linter
# ---------------------------------------------------------------------------

#: Sentence constructions this repository may not print, each with the task
#: that earned the rule. These are not house style. Every one of them is a
#: sentence some earlier task showed the data cannot carry, and writing the
#: rule as a regex is the difference between a standard the other three
#: specialists can follow and a paragraph they can forget.
PROHIBITED_PATTERNS: tuple[tuple[str, str, str, str], ...] = (
    (
        "forecast",
        r"\b(will\s+(rise|fall|grow|reach|increase|decline|continue)"
        r"|expected\s+to|is\s+projected|projected\s+to|forecast\s+to"
        r"|going\s+forward|over\s+the\s+next\s+(three|six|twelve|\d+)"
        r"|in\s+202[4-9]|by\s+202[4-9]|next\s+(quarter|year|half))\b",
        "task-07 §8",
        "no model beats persistence and the h=1 interval spans 3.15x, so the "
        "maximum useful horizon is 0",
    ),
    (
        "convergence",
        r"\b(becoming\s+(more|less)\s+(like|similar)|converg\w+|diverg\w+"
        r"|increasingly\s+(similar|alike)|drifting\s+(apart|together)"
        r"|moving\s+closer\s+to)\b",
        "task-08 §8",
        "trajectory similarity is refused: 1 of 15 pairs is eligible and the "
        "mean correlation sits inside the closure null",
    ),
    (
        "cross_company_level",
        # The noun list is the point. A rule that catches "more jobs than"
        # and not "more roles than" is a rule a writer clears by accident.
        r"\b((more|fewer|less)\s+(postings?|jobs?|roles?|openings?"
        r"|positions?|vacancies|hires)\s+than|posted\s+(more|fewer)"
        r"|hires?\s+more|hires?\s+fewer|out-?hir\w+"
        r"|(largest|biggest|smallest)\s+hirer"
        r"|hiring\s+volume\s+(exceeds|is\s+higher))\b",
        "task-06 §1.3",
        "levels are not identified: a company's posting count is a count of "
        "the boards that syndicate it",
    ),
    (
        "product_launch",
        r"\b(plans?\s+to\s+launch|is\s+preparing\s+to|signals?\s+a\s+"
        r"(launch|pivot|product)|ahead\s+of\s+a\s+launch|points?\s+to\s+a\s+"
        r"(new\s+)?product|building\s+towards)\b",
        "task-09 §5",
        "no table in this repository links a posting to a product decision; "
        "the construct is never measured",
    ),
    (
        "unmeasured_construct",
        r"\b(headcount|attrition|layoffs?|revenue|market\s+share|profit"
        r"|budget|R&D\s+spend|employees\s+left)\b",
        "task-02 scope",
        "the collected schema holds postings, not the company; the construct "
        "has no column",
    ),
    (
        "bare_share_of_all",
        # "30% of all postings" is the same prohibited denominator as "share
        # of all postings", and the likelier phrasing of the two.
        r"\bshare_of_all\b|\bof\s+all\s+(its\s+|their\s+)?postings\b",
        "task-04 §7",
        "the skill denominator is share_of_skilled; share_of_all confounds "
        "demand with description coverage",
    ),
    (
        "seasonal",
        r"\b(seasonal(ity|ly)?\s+(peak|pattern|effect|dip)|every\s+(summer|"
        r"winter|spring|autumn)|each\s+year|annual\s+cycle"
        r"|year\s+on\s+year)\b",
        "task-05 §6",
        "month-of-year seasonality needs more than one cycle and this "
        "collection has zero complete ones",
    ),
    (
        "causal_strategy",
        r"\b(because\s+(google|meta|microsoft|nvidia|snowflake|databricks)"
        r"|in\s+response\s+to|as\s+a\s+result\s+of\s+(a|the)\s+(strategy|"
        r"decision)|deliberate(ly)?\s+(shift|pivot))\b",
        "task-05 §1",
        "posting_date is an aggregator first-seen date; the series describes "
        "discovery, which cannot carry an intent story",
    ),
)

#: Constructs whose refusal is worth naming even though no regex would catch a
#: careful phrasing of them. Used by the brief-promise audit, not the linter.
UNMEASURED_CONSTRUCTS = ("product launches", "salaries", "headcount",
                         "attrition", "revenue")


def country_vocabulary(root: Path = DEFAULT_MEMBER) -> tuple[str, ...]:
    """Country names this specialist's data actually contains.

    Read from the committed panel check rather than hard-coded, so the rule
    travels to a specialist whose postings land in different countries. Task 05
    kept only 21 of 48 country directions on the balanced panel: publishers are
    regional, so a country split is a map of aggregator footprints.
    """
    path = root / "task-05-tables" / "panel-check-country.csv"
    if not path.is_file():
        return ()
    segments = pd.read_csv(path)["segment"].astype(str)
    return tuple(sorted({s.strip() for s in segments if s.strip()}))


def lint_text(text: str, countries: tuple[str, ...] = ()) -> list[dict]:
    """Return every prohibited construction found in a sentence."""
    hits = []
    lowered = text.lower()
    for rule, pattern, source, why in PROHIBITED_PATTERNS:
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            hits.append({"rule": rule, "matched": match.group(0).strip(),
                         "source": source, "why": why})
    for country in countries:
        if re.search(rf"\b{re.escape(country.lower())}\b", lowered):
            hits.append({
                "rule": "country_split",
                "matched": country,
                "source": "task-05 §9",
                "why": "publishers are regional, so a country figure compares "
                       "aggregator footprints, not hiring",
            })
            break
    return hits


def lint_claim(claim: Claim, countries: tuple[str, ...] = ()) -> Gate:
    """Gate 2. A claim that states a refusal may name the thing it refuses.

    Without this exemption the rule set would forbid the sentence "no forecast
    is available", which is the most useful sentence Task 07 produced.
    """
    hits = lint_text(claim.text, countries)
    if claim.refusal_rule:
        hits = [h for h in hits if h["rule"] != claim.refusal_rule]
    if not hits:
        return Gate("lint", True, "")
    return Gate("lint", False,
                "; ".join(f"{h['rule']} ({h['source']})" for h in hits))


def lint_audit(claims, countries: tuple[str, ...] = ()) -> pd.DataFrame:
    """One row per (claim, rule hit). Empty when nothing is caught."""
    rows = []
    for claim in claims:
        for hit in lint_text(claim.text, countries):
            exempt = hit["rule"] == claim.refusal_rule
            rows.append({"claim_id": claim.claim_id, "family": claim.family,
                         "rule": hit["rule"], "matched": hit["matched"],
                         "source": hit["source"], "exempt": exempt,
                         "why": hit["why"]})
    columns = ["claim_id", "family", "rule", "matched", "source", "exempt",
               "why"]
    return pd.DataFrame(rows, columns=columns)


def prohibited_pattern_table(audit: pd.DataFrame) -> pd.DataFrame:
    """The rule set itself, with how many candidates each rule caught."""
    caught = (audit[~audit.exempt].groupby("rule").size()
              if not audit.empty else pd.Series(dtype=int))
    exempted = (audit[audit.exempt].groupby("rule").size()
                if not audit.empty else pd.Series(dtype=int))
    rows = [{
        "rule": rule,
        "source": source,
        "why": why,
        "claims_caught": int(caught.get(rule, 0)),
        "refusals_exempted": int(exempted.get(rule, 0)),
    } for rule, _pattern, source, why in PROHIBITED_PATTERNS]
    rows.append({
        "rule": "country_split",
        "source": "task-05 §9",
        "why": "publishers are regional, so a country figure compares "
               "aggregator footprints, not hiring",
        "claims_caught": int(caught.get("country_split", 0)),
        "refusals_exempted": int(exempted.get("country_split", 0)),
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E. Gate 3 — identification is inherited, never re-decided
# ---------------------------------------------------------------------------

#: How each upstream verdict maps onto a Task 09 status. The keys are the
#: verdict *values* the earlier tasks wrote; the values are what this task is
#: allowed to do with them.
#:
#: Reading this table is the whole of Gate 3. Task 09 owns no opinion about
#: whether Google's volume direction is identified — Task 05 answered that,
#: wrote `treatments_agree` into a CSV, and this dictionary obeys it. The
#: failure this prevents is the one every reporting layer invites: a writer who
#: re-reads the evidence, finds it "clear enough", and quietly promotes a
#: `mixed` verdict to a headline.
VERDICT_MAP: dict[str, dict[str, str]] = {
    # Task 05 / 06 volume: do the four panel treatments agree?
    "volume_treatments": {"True": QUALIFIED, "False": REFUSED},
    # Task 06 relative share: does every shared publisher point the same way?
    "relative_share": {"confirmed": QUALIFIED, "mixed": QUALIFIED,
                       "not supported": REFUSED},
    # Task 05 segment mix: does the raw direction survive the balanced panel?
    "segment_panel": {"True": QUALIFIED, "False": REFUSED},
    # Task 05 skill trends, stratified inside job function.
    "skill_stratified": {"rising_in_all_segments": PUBLISHED,
                         "falling_in_all_segments": PUBLISHED,
                         "mix_dependent": REFUSED,
                         "insufficient_support": REFUSED},
    # Task 06 cross-company skill gaps, stratified inside job function.
    "skill_gap": {"confirmed": PUBLISHED, "mixed": REFUSED,
                  "reversed": REFUSED, "insufficient_support": REFUSED},
    # Task 06 distinctiveness, after Benjamini-Hochberg.
    "distinctiveness": {"True": PUBLISHED, "False": REFUSED},
    # Task 07 forecastability gate.
    "forecastable": {"forecastable": QUALIFIED, "too_thin": REFUSED,
                     "no_signal": REFUSED},
    # Task 07 horizon: can the residuals bound an interval at all?
    "horizon": {"True": QUALIFIED, "False": REFUSED},
    # Task 08 pair verdicts.
    "similarity_pair": {"robust": PUBLISHED, "vendor_dependent": QUALIFIED,
                        "unresolved": REFUSED},
    # Task 09's own audits, where the verdict is a boolean feasibility flag.
    "feasible": {"True": QUALIFIED, "False": REFUSED},
    # Statements about the collection process itself. These are properties of
    # the panel, not estimates from it, so they carry no sampling verdict.
    "structural": {"": PUBLISHED},
}


#: Statuses ordered weakest first. A claim inheriting from several upstream
#: verdicts takes the weakest of them — a sentence is only as identified as the
#: least identified thing it rests on.
STATUS_STRENGTH = {REFUSED: 0, QUALIFIED: 1, PUBLISHED: 2}


def _inherit_one(term: str, claim_id: str,
                 root: Path) -> tuple[str, str]:
    family, _, citation = term.partition(":")
    if family not in VERDICT_MAP:
        raise KeyError(f"{claim_id}: unknown verdict family {family!r}")
    table = VERDICT_MAP[family]
    if not citation:
        return "", table.get("", REFUSED)
    try:
        value = resolve_citation(citation, root)
    except (FileNotFoundError, KeyError, LookupError, ValueError):
        return "", REFUSED
    key = str(value).strip()
    return key, table.get(key, REFUSED)


def inherit_verdict(claim: Claim, root: Path = DEFAULT_MEMBER) -> tuple[str, str]:
    """Gate 3. Return ``(verdict_value, status)`` for a claim.

    ``verdict_source`` is written as ``family:citation``, and several may be
    joined with ``&`` when a sentence rests on more than one upstream answer.
    Task 08's trajectory claim is the case that forced this: a pair needs to be
    *eligible* and its interval needs to *exclude zero*, and two of the fifteen
    pairs exclude zero while being ineligible — reading either cell alone
    publishes a correlation computed on a series that failed its own gate.
    Where sources are joined the weakest status wins, and the returned verdict
    value names every cell that was read.
    """
    if not claim.verdict_source:
        return "", REFUSED
    terms = [t for t in claim.verdict_source.split("&") if t.strip()]
    results = [_inherit_one(t.strip(), claim.claim_id, root) for t in terms]
    weakest = min(results, key=lambda pair: STATUS_STRENGTH[pair[1]])
    value = " & ".join(v or "structural" for v, _ in results)
    return value, weakest[1]


# ---------------------------------------------------------------------------
# F. Gate 4 — cross-task consistency
# ---------------------------------------------------------------------------

#: Quantities this repository computes twice. Each entry names the earlier
#: citation, the later one, and the register entry that settles them.
#:
#: A reporting layer is where these collide, because it is the first task that
#: reads every earlier table at once. The rule is fixed and one-directional:
#: the later task wins, and the earlier number may appear only alongside its
#: correction pointer.
RESTATED_QUANTITIES: tuple[dict, ...] = (
    {
        "quantity": "google H1 to H2 change in share of the common panel",
        "earlier": "task-06-tables/relative-share-by-half.csv"
                   "#share_change_pp@company=google",
        "later": "task-07-tables/february-correction.csv"
                 "#corrected_delta_pp@company=google",
        "register": "C5",
        "unit": "pp",
    },
    {
        "quantity": "meta H1 to H2 change in share of the common panel",
        "earlier": "task-06-tables/relative-share-by-half.csv"
                   "#share_change_pp@company=meta",
        "later": "task-07-tables/february-correction.csv"
                 "#corrected_delta_pp@company=meta",
        "register": "C5",
        "unit": "pp",
    },
    {
        "quantity": "nvidia H1 to H2 change in share of the common panel",
        "earlier": "task-06-tables/relative-share-by-half.csv"
                   "#share_change_pp@company=nvidia",
        "later": "task-07-tables/february-correction.csv"
                 "#corrected_delta_pp@company=nvidia",
        "register": "C5",
        "unit": "pp",
    },
)


def consistency_table(root: Path = DEFAULT_MEMBER,
                      restated=RESTATED_QUANTITIES) -> pd.DataFrame:
    """Every quantity two tasks both quote, with the value that wins."""
    rows = []
    for item in restated:
        try:
            earlier = _numeric(resolve_citation(item["earlier"], root))
            later = _numeric(resolve_citation(item["later"], root))
        except (FileNotFoundError, KeyError, LookupError, ValueError):
            earlier = later = None
        conflict = (earlier is not None and later is not None
                    and not values_agree(earlier, later))
        rows.append({
            "quantity": item["quantity"],
            "earlier_citation": item["earlier"],
            "earlier_value": earlier,
            "later_citation": item["later"],
            "later_value": later,
            "unit": item["unit"],
            "delta": (None if earlier is None or later is None
                      else round(later - earlier, 4)),
            "conflict": conflict,
            "wins": "later",
            "register": item["register"],
        })
    return pd.DataFrame(rows)


def consistency_gate(claim: Claim, conflicts: pd.DataFrame) -> Gate:
    """Gate 4. A claim citing the superseded side of a conflict must say so."""
    if conflicts.empty:
        return Gate("consistency", True, "")
    superseded = conflicts[conflicts.conflict]
    hit = superseded[superseded.earlier_citation == claim.citation]
    if hit.empty:
        return Gate("consistency", True, "")
    row = hit.iloc[0]
    pointer = str(row.register).lower()
    if pointer in claim.clause.lower() or pointer in claim.text.lower():
        return Gate("consistency", True,
                    f"superseded by {row.register}, pointer carried")
    return Gate("consistency", False,
                f"cites the value {row.register} superseded "
                f"({row.earlier_value} -> {row.later_value} {row.unit})")


# ---------------------------------------------------------------------------
# G. The compiler
# ---------------------------------------------------------------------------

#: Order matters: a claim is reported against the first gate it fails, so a
#: sentence with no evidence is never also blamed on its wording.
GATE_ORDER = ("evidence", "lint", "identification", "consistency")


def compile_claims(claims, root: Path = DEFAULT_MEMBER) -> pd.DataFrame:
    """Run every claim through the four gates and return the ledger.

    The output is the deliverable. Prose in the report quotes this table; it
    never restates a number the table does not hold.
    """
    countries = country_vocabulary(root)
    conflicts = consistency_table(root)
    rows = []
    for claim in claims:
        evidence = bind_claim(claim, root)
        if evidence.passed:
            direction = direction_agrees(claim, root)
            if not direction.passed:
                evidence = Gate("evidence", False, direction.detail)
        lint = lint_claim(claim, countries)
        verdict_value, inherited = inherit_verdict(claim, root)
        identification = Gate(
            "identification", inherited in (PUBLISHED, QUALIFIED),
            f"{verdict_value or 'structural'} -> {inherited}",
        )
        consistency = consistency_gate(claim, conflicts)
        gates = {"evidence": evidence, "lint": lint,
                 "identification": identification, "consistency": consistency}

        failed = [name for name in GATE_ORDER if not gates[name].passed]
        if failed:
            status, blocked_by = REFUSED, failed[0]
            reason = gates[failed[0]].detail
        else:
            status, blocked_by = inherited, ""
            reason = identification.detail
        # No sentence leaves this task bare. The clause is what the reader
        # must hold the sentence with — that a share is a share of skilled
        # postings, that a rank is what is identified and not the score. It
        # is required of every publishable claim, not only of the ones whose
        # verdict came back qualified, and a claim that reaches this line
        # without one is a drafting fault rather than a finding.
        if status in (PUBLISHED, QUALIFIED) and not claim.clause:
            status, blocked_by = REFUSED, "identification"
            reason = "a publishable sentence must carry its reading clause"

        rows.append({
            "claim_id": claim.claim_id,
            "question": claim.question,
            "family": claim.family,
            "subject": claim.subject,
            "measures": claim.measures,
            "text": claim.text,
            "clause": claim.clause,
            "citation": claim.citation,
            "quoted_value": claim.value,
            "source_task": claim.source_task,
            "verdict_source": claim.verdict_source,
            "inherited_verdict": verdict_value,
            "gate_evidence": evidence.passed,
            "gate_lint": lint.passed,
            "gate_identification": identification.passed,
            "gate_consistency": consistency.passed,
            "status": status,
            "blocked_by": blocked_by,
            "reason": reason,
            "audience": claim.audience,
            "action": claim.action,
            "falsifier": claim.falsifier,
            "depends_on": ", ".join(claim.depends_on),
            "notes": claim.notes,
        })
    ledger = pd.DataFrame(rows)
    if not ledger.empty and ledger.claim_id.duplicated().any():
        dupes = sorted(ledger.claim_id[ledger.claim_id.duplicated()].unique())
        raise ValueError(f"duplicate claim ids: {dupes}")
    return ledger


def insight_yield(ledger: pd.DataFrame) -> pd.DataFrame:
    """The funnel, per question: candidates in, publishable sentences out.

    This is the number Task 09 exists to produce. A reporting task that shows
    only its output shows a hit rate of 100% by construction; the denominator
    is what makes it a measurement.

    The denominator column is called `generated`, not `candidates`. Task 01's
    standing privacy check bans any column whose name contains `candidate`,
    because in a job-postings repository that word normally means a person.
    Task 09 borrowed it for a proposed sentence and tripped its own check. The
    check was left strict and this task's vocabulary moved instead — a
    substring ban is only worth having if a name collision loses to it.
    """
    rows = []
    for question in QUESTIONS:
        block = ledger[ledger.question == question]
        if block.empty:
            continue
        rows.append({
            "question": question,
            "generated": int(len(block)),
            "evidence_bound": int(block.gate_evidence.sum()),
            "lint_clean": int((block.gate_evidence & block.gate_lint).sum()),
            "identified": int((block.gate_evidence & block.gate_lint
                               & block.gate_identification).sum()),
            "published": int((block.status == PUBLISHED).sum()),
            "published_qualified": int((block.status == QUALIFIED).sum()),
            "refused": int((block.status == REFUSED).sum()),
            "yield_pct": round(
                100 * float((block.status != REFUSED).mean()), 1),
        })
    total = ledger
    rows.append({
        "question": "all",
        "generated": int(len(total)),
        "evidence_bound": int(total.gate_evidence.sum()),
        "lint_clean": int((total.gate_evidence & total.gate_lint).sum()),
        "identified": int((total.gate_evidence & total.gate_lint
                           & total.gate_identification).sum()),
        "published": int((total.status == PUBLISHED).sum()),
        "published_qualified": int((total.status == QUALIFIED).sum()),
        "refused": int((total.status == REFUSED).sum()),
        "yield_pct": round(100 * float((total.status != REFUSED).mean()), 1),
    })
    return pd.DataFrame(rows)


def refusal_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Every refused claim with the gate that stopped it and what would lift it."""
    block = ledger[ledger.status == REFUSED].copy()
    lift = {
        "evidence": "recompute or commit the table the sentence cites",
        "lint": "the construction is unavailable at any sample size; rewrite "
                "the claim around what is measured",
        "identification": "more months, more publishers, or a stratum with "
                          "enough support to settle the upstream verdict",
        "consistency": "quote the later value, or carry the register pointer",
    }
    block["what_would_lift_it"] = block.blocked_by.map(lift).fillna("")
    columns = ["claim_id", "question", "family", "subject", "text",
               "blocked_by", "reason", "what_would_lift_it"]
    return block[columns].reset_index(drop=True)


def published_claims(ledger: pd.DataFrame) -> pd.DataFrame:
    """Claims that survived, clause-bearing ones included."""
    return ledger[ledger.status != REFUSED].reset_index(drop=True)


def sentence(row) -> str:
    """Render one ledger row as the sentence a reader is allowed to see."""
    text = str(row["text"]).rstrip(".")
    clause = str(row.get("clause") or "").strip()
    return f"{text}." if not clause else f"{text} — {clause}"


# ---------------------------------------------------------------------------
# H. The C8 machinery — what a unanimity count is worth
# ---------------------------------------------------------------------------
#
# Task 06 §11 hands this task exactly one unqualified cross-company sentence:
# "NVIDIA gained share of the shared publisher pool between H1 and H2 2023, in
# every publisher that carries it." It is the only volume sentence in the
# repository with a `confirmed` verdict, so Task 09 would print it.
#
# `cmp.relative_share_verdict` counts how many publishers agree with the pooled
# sign. It applies a floor to the *publisher's* total across all six companies
# (`min_half`), and none at all to the company's own cell. NVIDIA's 6 of 6
# therefore includes publishers holding one posting in a half. Raise the floor
# and the count of agreeing publishers can *rise*, because the disagreeing
# cells are the thin ones and they leave the test set first. A statistic that
# improves when you demand more of the data is measuring the test set, not the
# company.

#: Per-company, per-half cell floors to recount unanimity at. 0 reproduces the
#: committed verdict table exactly.
CELL_FLOORS = (0, 3, 5, 10)


def sign_test_p(n_agree: int, n_tested: int) -> float:
    """Two-sided exact binomial p at p0 = 0.5, hand-rolled.

    No scipy in a core module (house rule since Task 05). `math.comb` is exact
    for the sizes involved here — six publishers at the very most.
    """
    if n_tested <= 0:
        return float("nan")
    if n_agree < 0 or n_agree > n_tested:
        raise ValueError(f"n_agree {n_agree} outside 0..{n_tested}")
    extreme = max(n_agree, n_tested - n_agree)
    tail = sum(math.comb(n_tested, k) for k in range(extreme, n_tested + 1))
    return min(1.0, 2.0 * tail / (2.0 ** n_tested))


def publisher_cell_floor(by_publisher: pd.DataFrame,
                         verdict: pd.DataFrame,
                         floors=CELL_FLOORS) -> pd.DataFrame:
    """Recount the publisher sign agreement under a per-company cell floor.

    `by_publisher` is Task 06's `relative-share-by-publisher.csv`; `verdict` is
    its `relative-share-verdict.csv`. The pooled sign is taken from the
    committed verdict and never recomputed — this is a question about the
    confirmation, not about the direction.
    """
    pooled = verdict.set_index("company").pooled_log_share_change
    rows = []
    for floor in floors:
        for company, block in by_publisher.groupby("company"):
            direction = pooled.get(company)
            if direction is None or pd.isna(direction):
                continue
            eligible = block[(block.h1_postings >= floor)
                             & (block.h2_postings >= floor)]
            tested = int(len(eligible))
            agreeing = int(
                (np.sign(eligible.log_share_change) == np.sign(direction)).sum()
            )
            rows.append({
                "cell_floor": floor,
                "company": company,
                "publishers_tested": tested,
                "publishers_agreeing": agreeing,
                "publishers_dropped": int(len(block) - tested),
                "unanimous": bool(tested > 0 and agreeing == tested),
                "verdict": ("confirmed" if tested > 0 and agreeing == tested
                            else "mixed" if tested > 0 else "untestable"),
                "sign_test_p": round(sign_test_p(agreeing, tested), 4)
                               if tested else float("nan"),
                "clears_005": bool(tested > 0
                                   and sign_test_p(agreeing, tested) < 0.05),
                "pooled_log_share_change": round(float(direction), 4),
                "pooled_direction": ("gaining share" if direction > 0
                                     else "losing share"),
            })
    return pd.DataFrame(rows).sort_values(
        ["company", "cell_floor"]).reset_index(drop=True)


@dataclass(frozen=True)
class FloorVerdict:
    """Whether a company's unanimity survives asking more of the data."""

    company: str
    floors_confirmed: tuple
    floors_tested: tuple
    tested_counts: tuple
    monotone: bool
    floor_dependent: bool
    detail: str


def unanimity_is_floor_dependent(recount: pd.DataFrame) -> pd.DataFrame:
    """One row per company: does `confirmed` move when the cell floor moves?

    `floor_dependent` is the finding. `monotone` records the direction that
    makes it a defect rather than noise — the confirmed set *growing* as the
    floor rises means unanimity is being bought by dropping tests.
    """
    rows = []
    for company, block in recount.groupby("company"):
        block = block.sort_values("cell_floor")
        confirmed = tuple(int(f) for f in
                          block.loc[block.verdict == "confirmed", "cell_floor"])
        floors = tuple(int(f) for f in block.cell_floor)
        counts = tuple(int(n) for n in block.publishers_tested)
        conf_flags = list(block.verdict == "confirmed")
        floor_dependent = len(set(conf_flags)) > 1
        gained = (not conf_flags[0]) and any(conf_flags[1:])
        if floor_dependent and gained:
            detail = (f"not confirmed at floor {floors[0]} and confirmed at a "
                      f"stricter one; the test set falls "
                      f"{counts[0]} -> {counts[-1]}")
        elif floor_dependent:
            detail = (f"confirmed at floor {floors[0]} only; the test set "
                      f"falls {counts[0]} -> {counts[-1]}")
        else:
            detail = (f"same verdict at every floor "
                      f"({block.verdict.iloc[0]}); test set "
                      f"{counts[0]} -> {counts[-1]}")
        rows.append({
            "company": company,
            "verdict_at_floor_0": block.verdict.iloc[0],
            "floors_confirmed": ", ".join(str(f) for f in confirmed) or "none",
            "publishers_tested_range": f"{counts[0]}-{counts[-1]}",
            "min_tested": min(counts),
            "floor_dependent": floor_dependent,
            "confirmation_gained_by_dropping_tests": gained,
            "sign_test_p_at_floor_0": float(block.sign_test_p.iloc[0]),
            "clears_005_at_floor_0": bool(block.clears_005.iloc[0]),
            "detail": detail,
        })
    return pd.DataFrame(rows).sort_values("company").reset_index(drop=True)


def sign_test_power(max_publishers: int = 8) -> pd.DataFrame:
    """What a unanimous sign count can and cannot establish at this panel size.

    Seven publishers is the whole common panel. Unanimity across five or fewer
    cannot reach p < 0.05 however clean it looks, which is the second half of
    the C8 point: the floor question only matters because the panel is small
    enough that dropping two tests changes the answer.
    """
    rows = []
    for n in range(1, max_publishers + 1):
        p = sign_test_p(n, n)
        rows.append({
            "publishers_tested": n,
            "unanimous_agreeing": n,
            "sign_test_p": round(p, 4),
            "clears_005": bool(p < 0.05),
            "note": ("unanimity alone cannot clear 0.05 at this panel size"
                     if p >= 0.05 else "unanimity clears 0.05"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# I. The salary audit — the promise this evidence base cannot keep
# ---------------------------------------------------------------------------
#
# Task 01's source review lists "benchmark salaries" among the things this
# project would deliver. `salary_year_avg` exists, so Task 09 is the point at
# which someone writes "Google pays $X". Before that sentence can be linted it
# has to be measured, and the measurement is a missingness problem, not a
# central-tendency one: disclosure is a *publisher* behaviour. Ai-Jobs.net
# discloses most of its postings, four publishers disclose none. Conditioning
# on disclosure therefore conditions on publisher, and through publisher on
# country and role — the salary column is missing not at random.
#
# Note the ordering this forces: the audit needs row-level data, which is
# git-ignored, so Task 09 computes it, commits the aggregate, and cites the
# committed aggregate. A published claim never cites a frame in memory.

SALARY_COL = "salary_year_avg"

#: A within-publisher pair needs this many disclosed salaries on each side
#: before it is worth estimating. Same floor as Task 07's cell floor.
SALARY_MIN_CELL = 5


def salary_disclosure(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per company: how much salary there is, and how selected it is.

    `disclosed_pct` is the headline. `country_shift_pp` is why the headline is
    not enough — it is the difference between the share of the disclosed subset
    sitting in the modal country and the same share in the full set. A large
    shift means the disclosed rows are a different population.
    """
    rows = []
    for key, df in sorted(frames.items()):
        total = int(len(df))
        if SALARY_COL not in df.columns:
            rows.append({"company": key, "postings": total, "disclosed": 0,
                         "disclosed_pct": 0.0, "usable": False,
                         "reason": "no salary column"})
            continue
        disclosed = df[df[SALARY_COL].notna()]
        n = int(len(disclosed))
        entry = {
            "company": key,
            "postings": total,
            "disclosed": n,
            "disclosed_pct": round(100 * n / total, 2) if total else 0.0,
            "publishers_disclosing": int(
                disclosed[PUBLISHER_COL].nunique()) if n else 0,
            "publishers_total": int(df[PUBLISHER_COL].nunique()),
        }
        col = "country" if "country" in df.columns else None
        if col and n:
            modal = df[col].value_counts(normalize=True)
            top = modal.index[0]
            share_all = 100 * float(modal.iloc[0])
            share_disc = 100 * float((disclosed[col] == top).mean())
            entry.update({
                "modal_country": str(top),
                "modal_share_all_pct": round(share_all, 2),
                "modal_share_disclosed_pct": round(share_disc, 2),
                "country_shift_pp": round(share_disc - share_all, 2),
            })
        entry["usable"] = bool(n >= SALARY_MIN_CELL)
        entry["reason"] = ("" if n >= SALARY_MIN_CELL
                           else f"{n} disclosed salaries")
        rows.append(entry)
    return pd.DataFrame(rows)


def salary_disclosure_by_publisher(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Disclosure rate per publisher, pooled across companies.

    This is the table that makes the missingness mechanism visible: if
    disclosure were a company policy the spread here would be narrow.
    """
    stacked = cmp.stack(frames)
    rows = []
    for publisher, block in stacked.groupby(PUBLISHER_COL):
        n = int(len(block))
        disclosed = int(block[SALARY_COL].notna().sum()) \
            if SALARY_COL in block.columns else 0
        rows.append({
            "publisher": str(publisher),
            "postings": n,
            "disclosed": disclosed,
            "disclosed_pct": round(100 * disclosed / n, 2) if n else 0.0,
            "companies_carried": int(block.company.nunique()),
        })
    out = pd.DataFrame(rows).sort_values(
        ["disclosed_pct", "postings"], ascending=[False, False])
    return out.reset_index(drop=True)


def salary_feasible_cells(frames: dict[str, pd.DataFrame],
                          min_cell: int = SALARY_MIN_CELL) -> pd.DataFrame:
    """Which company-publisher cells carry enough disclosed salary to compare.

    A cross-company salary comparison is only interpretable *within* a
    publisher — otherwise the difference is a disclosure-policy difference. So
    the feasibility question is how many companies each publisher can carry.
    """
    stacked = cmp.stack(frames)
    disclosed = stacked[stacked[SALARY_COL].notna()] \
        if SALARY_COL in stacked.columns else stacked.iloc[0:0]
    counts = (disclosed.groupby([PUBLISHER_COL, "company"]).size()
              .rename("disclosed").reset_index())
    counts["meets_floor"] = counts.disclosed >= min_cell
    rows = []
    for publisher, block in counts.groupby(PUBLISHER_COL):
        eligible = block[block.meets_floor]
        rows.append({
            "publisher": str(publisher),
            "companies_with_any": int(len(block)),
            "companies_at_floor": int(len(eligible)),
            "companies": ", ".join(sorted(eligible.company)),
            "pairs_available": int(
                len(eligible) * (len(eligible) - 1) / 2),
            "min_cell": min_cell,
        })
    out = pd.DataFrame(rows).sort_values(
        "companies_at_floor", ascending=False)
    return out.reset_index(drop=True)


def _median(values) -> float:
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    if not n:
        return float("nan")
    mid = n // 2
    return (ordered[mid] if n % 2
            else 0.5 * (ordered[mid - 1] + ordered[mid]))


def salary_pair_estimate(frames: dict[str, pd.DataFrame], a: str, b: str,
                         publisher: str, n_boot: int = 2000,
                         seed: int = 20240609) -> dict:
    """Median salary difference a - b within one publisher, with a bootstrap CI.

    The interval is a percentile bootstrap over postings, matching Task 08's
    resampling unit. Role mix and seniority mix are reported alongside because
    a difference in pay and a difference in *what is being hired* are not
    distinguishable here, and the reader has to be able to see that.
    """
    rng = np.random.default_rng(seed)
    cut = {}
    for key in (a, b):
        df = frames[key]
        block = df[(df[PUBLISHER_COL] == publisher) & df[SALARY_COL].notna()]
        cut[key] = block
    va = cut[a][SALARY_COL].to_numpy(dtype=float)
    vb = cut[b][SALARY_COL].to_numpy(dtype=float)
    out = {
        "publisher": publisher, "company_a": a, "company_b": b,
        "n_a": int(len(va)), "n_b": int(len(vb)),
        "median_a": round(_median(va), 2) if len(va) else float("nan"),
        "median_b": round(_median(vb), 2) if len(vb) else float("nan"),
    }
    if len(va) < SALARY_MIN_CELL or len(vb) < SALARY_MIN_CELL:
        out.update({"median_diff": float("nan"), "ci_low": float("nan"),
                    "ci_high": float("nan"), "spans_zero": True,
                    "feasible": False,
                    "reason": f"cell floor {SALARY_MIN_CELL} not met"})
        return out
    diff = _median(va) - _median(vb)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        ra = rng.choice(va, size=len(va), replace=True)
        rb = rng.choice(vb, size=len(vb), replace=True)
        draws[i] = _median(ra) - _median(rb)
    low, high = (float(np.percentile(draws, 2.5)),
                 float(np.percentile(draws, 97.5)))
    out.update({
        "median_diff": round(diff, 2),
        "ci_low": round(low, 2), "ci_high": round(high, 2),
        "spans_zero": bool(low <= 0 <= high),
        "feasible": True,
        "reason": "",
    })
    for key, label in ((a, "a"), (b, "b")):
        block = cut[key]
        if "job_function" in block.columns and len(block):
            top = block.job_function.value_counts(normalize=True)
            out[f"modal_function_{label}"] = str(top.index[0])
            out[f"modal_function_share_{label}"] = round(
                100 * float(top.iloc[0]), 1)
        if "seniority_level" in block.columns and len(block):
            out[f"senior_share_{label}"] = round(100 * float(
                block.seniority_level.astype(str)
                .str.contains("director|vp|principal|head",
                              case=False, na=False).mean()), 1)
    return out


def salary_pair_table(frames: dict[str, pd.DataFrame],
                      feasible: pd.DataFrame, focus: str,
                      n_boot: int = 2000) -> pd.DataFrame:
    """Every feasible within-publisher pair involving `focus`."""
    rows = []
    for _, row in feasible[feasible.companies_at_floor >= 2].iterrows():
        members = [c for c in str(row.companies).split(", ") if c]
        if focus not in members:
            continue
        for other in members:
            if other == focus:
                continue
            rows.append(salary_pair_estimate(
                frames, focus, other, str(row.publisher), n_boot=n_boot))
    if not rows:
        return pd.DataFrame(columns=["publisher", "company_a", "company_b",
                                     "n_a", "n_b", "median_diff", "ci_low",
                                     "ci_high", "spans_zero", "feasible"])
    return pd.DataFrame(rows)


def salary_pair_stratified(frames: dict[str, pd.DataFrame], a: str, b: str,
                           publisher: str, by: str = "job_function",
                           min_cell: int = SALARY_MIN_CELL,
                           n_boot: int = 2000,
                           seed: int = 20240609) -> pd.DataFrame:
    """The same difference, inside each stratum that carries both companies.

    Task 05 §6 established that an unstratified movement inside a mixed panel
    is a mix statement. The same applies to a pay difference: if one company's
    disclosed cell is research roles and the other's is sales, the difference
    is a role difference. Only a stratum where both sides clear the floor can
    tell those apart — and where none does, the honest output is `untestable`,
    not a smaller number.
    """
    rows = []
    for key in (a, b):
        if by not in frames[key].columns:
            return pd.DataFrame(columns=["stratum", "verdict"])
    blocks = {}
    for key in (a, b):
        df = frames[key]
        blocks[key] = df[(df[PUBLISHER_COL] == publisher)
                         & df[SALARY_COL].notna()]
    strata = sorted(set(blocks[a][by].dropna()) | set(blocks[b][by].dropna()))
    rng = np.random.default_rng(seed)
    for stratum in strata:
        va = blocks[a].loc[blocks[a][by] == stratum, SALARY_COL] \
            .to_numpy(dtype=float)
        vb = blocks[b].loc[blocks[b][by] == stratum, SALARY_COL] \
            .to_numpy(dtype=float)
        entry = {"publisher": publisher, "company_a": a, "company_b": b,
                 "stratum": str(stratum), "n_a": int(len(va)),
                 "n_b": int(len(vb))}
        if len(va) < min_cell or len(vb) < min_cell:
            entry.update({"median_diff": float("nan"), "ci_low": float("nan"),
                          "ci_high": float("nan"), "spans_zero": True,
                          "verdict": "untestable",
                          "reason": f"cell floor {min_cell} not met"})
            rows.append(entry)
            continue
        diff = _median(va) - _median(vb)
        draws = np.array([_median(rng.choice(va, len(va), replace=True))
                          - _median(rng.choice(vb, len(vb), replace=True))
                          for _ in range(n_boot)])
        low, high = (float(np.percentile(draws, 2.5)),
                     float(np.percentile(draws, 97.5)))
        spans = bool(low <= 0 <= high)
        entry.update({
            "median_a": round(_median(va), 2), "median_b": round(_median(vb), 2),
            "median_diff": round(diff, 2), "ci_low": round(low, 2),
            "ci_high": round(high, 2), "spans_zero": spans,
            "verdict": "no difference identified" if spans else "difference",
            "reason": "",
        })
        rows.append(entry)
    return pd.DataFrame(rows)


def salary_verdict(disclosure: pd.DataFrame, feasible: pd.DataFrame,
                   pairs: pd.DataFrame, stratified: pd.DataFrame,
                   focus: str) -> dict:
    """The single answer to "can this project benchmark salaries?".

    Written as a record so the report cannot soften it in prose.
    """
    row = disclosure[disclosure.company == focus]
    disclosed_pct = float(row.disclosed_pct.iloc[0]) if len(row) else 0.0
    shift = float(row.country_shift_pp.iloc[0]) if "country_shift_pp" in row \
        and len(row) else float("nan")
    tested = stratified[stratified.verdict.isin(
        ("difference", "no difference identified"))] \
        if not stratified.empty else stratified
    identified = tested[tested.verdict == "difference"] \
        if not tested.empty else tested
    return {
        "focus": focus,
        "disclosed_pct": round(disclosed_pct, 2),
        "country_shift_pp": round(shift, 2) if shift == shift else float("nan"),
        "publishers_carrying_two_companies": int(
            (feasible.companies_at_floor >= 2).sum()),
        "unstratified_pairs": int(len(pairs)),
        "unstratified_pairs_excluding_zero": int(
            (~pairs.spans_zero).sum()) if len(pairs) else 0,
        "stratified_cells_tested": int(len(tested)),
        "stratified_cells_identified": int(len(identified)),
        "benchmark_available": False,
        "verdict": (
            "refused: salary disclosure is a publisher behaviour, not a "
            "company one, so a company-level pay figure conditions on "
            "publisher and through it on country and role"),
    }


# ---------------------------------------------------------------------------
# J. Candidate generation
# ---------------------------------------------------------------------------
#
# Every candidate below is minted from a *row* of an upstream table, never
# typed by hand, and its text is assembled from the same cell it cites. Two
# consequences follow, and both of them are the point of the section:
#
#   1. The sentence cannot drift from the number. Rebuild the tables and the
#      text changes with them; a stale claim becomes a binding failure rather
#      than a plausible paragraph.
#   2. The denominator is real. The yield is measured against everything this
#      evidence base could be asked to say, not against the subset an author
#      thought to try. A hand-written report has a 100% hit rate by
#      construction, which is why nobody believes one.


def _slug(text: str) -> str:
    """Kebab-case an identifier without collapsing skills that differ by symbol.

    A naive ``re.sub(r"[^a-z0-9]+", "-", ...)`` sends ``C``, ``C++`` and ``C#``
    to the same slug, and the compiler's duplicate-id check catches it as three
    claims wearing one name. The taxonomy is full of these — ``.NET``, ``C++``,
    ``Node.js`` — so the symbols are transliterated before anything is dropped.
    """
    lowered = str(text).lower()
    for symbol, word in (("++", "-plus-plus"), ("+", "-plus"),
                         ("#", "-sharp"), (".", "-dot-"), ("/", "-")):
        lowered = lowered.replace(symbol, word)
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")


def _cell(row, name: str):
    """Read a column off a row by subscript, never by attribute.

    ``row.diff`` on a pandas Series returns ``Series.diff``, the method — the
    column is shadowed silently and the claim ends up quoting a bound method.
    The binder caught it, which is the system working, but every generator here
    goes through this accessor so it cannot happen again. ``diff``, ``rank``,
    ``count``, ``size``, ``mean`` and ``max`` are all real column names in the
    committed tables and all real Series attributes.
    """
    return row[name]


def _fmt(value, digits: int = 2) -> str:
    number = _numeric(value)
    if number is None:
        return str(value)
    if float(number).is_integer() and abs(number) < 1e6:
        return str(int(number))
    return f"{number:.{digits}f}"


def _pct(value, digits: int = 1) -> str:
    number = _numeric(value)
    return "n/a" if number is None else f"{100 * number:.{digits}f}%"


def _load(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def candidates_volume(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """One claim per company: which way did posting volume move in 2023?

    The obvious headline of any hiring report, and the one Task 05 spent a
    whole task refusing. `treatments_agree` decides it.
    """
    table = _load(root, "task-06-tables/volume-verdict.csv")
    claims = []
    for _, row in table.iterrows():
        company = str(row.company)
        claims.append(Claim(
            claim_id=f"vol-{company}",
            question="hiring_patterns",
            family="volume",
            subject=company,
            measures="posting volume, 2023, four panel treatments",
            text=(f"{company.title()}'s 2023 posting volume reads as "
                  f"{row.direction}, and the four panel treatments spread "
                  f"{_fmt(_cell(row, "spread"))} index points around it"),
            citation=f"task-06-tables/volume-verdict.csv#spread@company={company}",
            value=_cell(row, "spread"),
            verdict_source=("volume_treatments:task-06-tables/volume-verdict."
                            f"csv#treatments_agree@company={company}"),
            clause=("treatments disagree, so the direction is a property of "
                    "the panel treatment, not of the company"
                    if not bool(row.treatments_agree) else
                    "measured on the shared publisher panel only"),
            falsifier=("a twelve-month panel in which all four treatments "
                       "agree on the sign"),
            action="none until the panel is balanced",
            audience="strategy",
            source_task="06",
            notes="direction is inherited from Task 05's panel treatments",
        ))
    return claims


def candidates_share(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """One claim per company: share of the common publisher pool, H1 to H2.

    Task 06 §11 calls this the only cross-company volume sentence available.
    Task 09 generates it for all six so the reader can see which ones the
    verdict actually clears — and, through C8, what `confirmed` is worth.
    """
    table = _load(root, "task-06-tables/relative-share-verdict.csv")
    claims = []
    for _, row in table.iterrows():
        company = str(row.company)
        claims.append(Claim(
            claim_id=f"share-{company}",
            question="hiring_patterns",
            family="relative_share",
            subject=company,
            measures="share of the common publisher pool, H1 vs H2 2023",
            text=(f"{company.title()} was {row.direction} of the shared "
                  f"publisher pool between H1 and H2 2023 "
                  f"(log share change {_fmt(row.pooled_log_share_change, 3)}), "
                  f"agreeing in {int(row.publishers_agreeing)} of "
                  f"{int(row.publishers_tested)} publishers"),
            citation=("task-06-tables/relative-share-verdict.csv"
                      f"#pooled_log_share_change@company={company}"),
            value=row.pooled_log_share_change,
            verdict_source=("relative_share:task-06-tables/"
                            f"relative-share-verdict.csv#verdict@company={company}"),
            clause=("share of a fixed publisher pool, not headcount and not "
                    "absolute volume; unanimity here is floor-dependent, see "
                    "C8"),
            falsifier=("the same sign count under a per-company cell floor of "
                       "5 postings a half"),
            action="watch which competitor is displacing whom on shared boards",
            audience="strategy",
            source_task="06",
            depends_on=("C8",),
        ))
    return claims


def candidates_mix(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """One claim per job function: is the focus company's mix shifting?

    Task 05's balanced panel decides. `directions_agree` is the verdict.
    """
    table = _load(root, "task-05-tables/panel-check-job-function.csv")
    claims = []
    for _, row in table.iterrows():
        segment = str(row.segment)
        slug = _slug(segment)
        claims.append(Claim(
            claim_id=f"mix-{slug}",
            question="hiring_patterns",
            family="segment_mix",
            subject=f"{focus}:{segment}",
            measures="share of postings by job function, H1 vs H2 2023",
            text=(f"{segment} moved from {_pct(row.share_h1)} to "
                  f"{_pct(row.share_h2)} of {focus.title()}'s postings between "
                  f"H1 and H2 2023, a change of "
                  f"{_fmt(row.share_delta, 4)} in share"),
            citation=("task-05-tables/panel-check-job-function.csv"
                      f"#share_delta@segment={segment}"),
            value=row.share_delta,
            verdict_source=("segment_panel:task-05-tables/"
                            "panel-check-job-function.csv"
                            f"#directions_agree@segment={segment}"),
            clause=("share of postings within the balanced publisher panel; "
                    "a share can move because a different function moved"),
            falsifier=("the raw and balanced panels disagreeing on the sign "
                       "in a twelve-month window"),
            action="size the recruiting pipeline by function, not in total",
            audience="hr_talent",
            source_task="05",
        ))
    return claims


def candidates_skill_trend(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """One claim per skill Task 05 tested for a trend inside job function.

    This family is where C1 and C2 were raised, so it is also the family whose
    refusal rate is most worth reading.
    """
    table = _load(root, "task-05-tables/skill-stratified-verdicts.csv")
    claims = []
    for _, row in table.iterrows():
        skill = str(row.skill)
        slug = _slug(skill)
        direction = ("rising" if str(row.verdict).startswith("rising")
                     else "falling" if str(row.verdict).startswith("falling")
                     else "moving")
        # The cited value is a *pooled* share and the direction word comes
        # from the *stratified* verdict. For one skill in this table those
        # two point opposite ways — Looker falls inside all three functions
        # while the pooled share rises, which is what `contradiction` records
        # and what C2 was. Quoting both without saying so produces a sentence
        # that reads as a typo, and a reader who resolves it in favour of the
        # numbers has just undone the correction. The reversal is named.
        reversal = str(_cell(row, "contradiction")).strip().lower() == "reversed"
        if reversal:
            text = (f"Demand for {skill} at {focus.title()} is {direction} "
                    f"inside every one of the {int(row.n_segments)} job "
                    f"functions tested, while the pooled share across them "
                    f"runs the other way, {_fmt(row.pooled_share_h1, 4)} to "
                    f"{_fmt(row.pooled_share_h2, 4)} of skilled postings — a "
                    f"Simpson's reversal, and the within-function direction "
                    f"is the identified one")
            clause = ("share of skilled postings, never of all postings; the "
                      "pooled figure disagrees with the within-function "
                      "direction and does not overrule it")
        else:
            text = (f"Demand for {skill} at {focus.title()} is {direction} "
                    f"inside job function, {_fmt(row.pooled_share_h1, 4)} to "
                    f"{_fmt(row.pooled_share_h2, 4)} of skilled postings "
                    f"across {int(row.n_segments)} segments")
            clause = ("share of skilled postings, never of all postings; the "
                      "direction holds in every segment tested")
        claims.append(Claim(
            claim_id=f"skilltrend-{slug}",
            question="skill_demand",
            family="skill_trend",
            subject=f"{focus}:{skill}",
            measures="share of skilled postings mentioning the skill, "
                     "stratified by job function",
            text=text,
            citation=("task-05-tables/skill-stratified-verdicts.csv"
                      f"#pooled_share_h2@skill={skill}"),
            value=row.pooled_share_h2,
            verdict_source=("skill_stratified:task-05-tables/"
                            f"skill-stratified-verdicts.csv#verdict@skill={skill}"),
            clause=clause,
            falsifier=("one job function moving the other way with the same "
                       "support"),
            action="prioritise the training and sourcing pipeline for this skill",
            audience="hr_talent",
            source_task="05",
        ))
    return claims


def candidates_skill_gap(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """One claim per cross-company skill gap Task 06 stratified."""
    table = _load(root, "task-06-tables/skill-stratified-verdicts.csv")
    claims = []
    for _, row in table.iterrows():
        skill, a, b = str(row.skill), str(row.company_a), str(row.company_b)
        slug = f"{_slug(skill)}-{a}-{b}"
        lead = a if float(row.pooled_diff) > 0 else b
        trail = b if lead == a else a
        claims.append(Claim(
            claim_id=f"skillgap-{slug}",
            question="tech_stack",
            family="skill_gap",
            subject=f"{a}|{b}:{skill}",
            measures="difference in share of skilled postings, stratified by "
                     "job function",
            text=(f"{lead.title()} asks for {skill} in a larger share of its "
                  f"skilled postings than {trail.title()} does, by "
                  f"{_fmt(abs(float(row.pooled_diff)), 4)} pooled, holding in "
                  f"{int(row.n_agree)} of {int(row.n_strata)} job functions"),
            citation=("task-06-tables/skill-stratified-verdicts.csv"
                      f"#pooled_diff@skill={skill},company_a={a},company_b={b}"),
            value=row.pooled_diff,
            verdict_source=("skill_gap:task-06-tables/"
                            "skill-stratified-verdicts.csv"
                            f"#verdict@skill={skill},company_a={a},company_b={b}"),
            clause=("share of skilled postings within matched job functions; "
                    "a stack difference, not a capability difference"),
            falsifier="one job function reversing the sign at equal support",
            action="read as a stack signal when scoping integrations and "
                   "competitive positioning",
            audience="product",
            source_task="06",
        ))
    return claims


def candidates_distinctiveness(root: Path = DEFAULT_MEMBER,
                               focus: str = "google"):
    """One claim per skill the focus company over-indexes on.

    Task 06 ran Benjamini-Hochberg over the whole vocabulary, so `significant`
    is already FDR-controlled and Gate 3 simply reads it.
    """
    table = _load(root, f"task-06-tables/skill-distinctiveness-{focus}.csv")
    claims = []
    for _, row in table.iterrows():
        skill = str(row.skill)
        slug = _slug(skill)
        claims.append(Claim(
            claim_id=f"distinct-{slug}",
            question="tech_stack",
            family="distinctiveness",
            subject=f"{focus}:{skill}",
            measures="share of skilled postings against the other five "
                     "companies pooled",
            text=(f"{focus.title()} asks for {skill} in {_pct(_cell(row, "share"))} of "
                  f"its skilled postings against {_pct(_cell(row, "rest_share"))} across "
                  f"the other five, a difference of {_fmt(_cell(row, "diff"), 4)}"),
            citation=(f"task-06-tables/skill-distinctiveness-{focus}.csv"
                      f"#diff@skill={skill}"),
            value=_cell(row, "diff"),
            verdict_source=("distinctiveness:task-06-tables/"
                            f"skill-distinctiveness-{focus}.csv"
                            f"#significant@skill={skill}"),
            clause=("Benjamini-Hochberg controlled across the whole skill "
                    "vocabulary"
                    + ("; this is the company's own product, so the mention "
                       "is self-referential"
                       if bool(row.get("self_referential")) else "")),
            falsifier=("the interval on the difference covering zero once the "
                       "other five are re-weighted to this company's role mix"),
            action="treat as a stack marker when reading this company's "
                   "job ads competitively",
            audience="product",
            source_task="06",
            notes=("self-referential" if bool(row.get("self_referential"))
                   else ""),
        ))
    return claims


def candidates_forecast(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """The forecast sentence, generated so it can be refused on the record.

    Task 07 §14 says there is no forecast sentence at any horizon. That is a
    finding, and a finding has to be *generated* to be counted — a report that
    simply omits the forecast has not measured anything.
    """
    gate = _load(root, "task-07-tables/forecastability-gate.csv")
    horizons = _load(root, "task-07-tables/horizon-limits.csv")
    claims = []
    for _, row in gate.iterrows():
        company = str(row.key)
        claims.append(Claim(
            claim_id=f"forecastable-{company}",
            question="future_demand",
            family="forecastability",
            subject=company,
            measures="share of monthly variance that is signal rather than "
                     "sampling noise",
            text=(f"{company.title()}'s monthly share series carries "
                  f"{_pct(row.signal_share)} signal, enough to be worth "
                  f"modelling"),
            citation=("task-07-tables/forecastability-gate.csv"
                      f"#signal_share@key={company}"),
            value=row.signal_share,
            verdict_source=("forecastable:task-07-tables/"
                            f"forecastability-gate.csv#verdict@key={company}"),
            clause=("passing the forecastability gate means the series is not "
                    "pure noise; it does not mean a forecast is usable"),
            falsifier="a homogeneity test the series fails at 0.05",
            action="none on its own — the horizon table decides usability",
            audience="exec",
            source_task="07",
        ))
    for _, row in horizons.iterrows():
        horizon = int(row.horizon)
        claims.append(Claim(
            claim_id=f"forecast-h{horizon}",
            question="future_demand",
            family="forecast_horizon",
            subject=f"{focus}:h{horizon}",
            measures="width of the prediction interval, multiplicative",
            text=(f"{focus.title()}'s posting share {horizon} month"
                  f"{'s' if horizon != 1 else ''} ahead can be stated within a "
                  f"band {_fmt(row.interval_factor)} times wide"),
            citation=("task-07-tables/horizon-limits.csv"
                      f"#interval_factor@horizon={horizon}"),
            value=row.interval_factor,
            verdict_source=("horizon:task-07-tables/horizon-limits.csv"
                            f"#interval_sufficient@horizon={horizon}"
                            "&feasible:task-07-tables/forecast.csv"
                            f"#supported@key={focus},horizon={horizon}"),
            clause=("an interval this wide contains both a doubling and a "
                    "halving, so it excludes no decision"),
            falsifier=("a model beating persistence on Diebold-Mariano at any "
                       "horizon"),
            action="none; plan without a demand forecast from this source",
            audience="exec",
            source_task="07",
        ))
    return claims


def candidates_similarity(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """One claim per company pair Task 08 ranked."""
    table = _load(root, "task-08-tables/pair-verdicts.csv")
    claims = []
    for _, row in table.iterrows():
        a, b = str(row.company_a), str(row.company_b)
        # Both the falsifier and the action have to follow this pair's own
        # rank. Written once for the rank-1 pair and copied down the table,
        # they claim a rank-2 pair could be falsified by "leaving rank 1" —
        # a condition it does not meet and never could — and call every pair
        # in the table the closest talent competitor, including the last one.
        rank = int(_cell(row, "rank"))
        falsifier = (f"the pair leaving rank {rank} under any of the five "
                     f"metrics or either sensitivity")
        action = ("read as the closest talent competitor for compensation "
                  "and sourcing" if rank == 1 else
                  f"read as the {_ordinal(rank)} closest of fifteen pairs; "
                  f"the rank is the finding, not the gap to the pair above")
        claims.append(Claim(
            claim_id=f"pair-{a}-{b}",
            question="position",
            family="similarity",
            subject=f"{a}|{b}",
            measures="cosine similarity of skill profiles",
            text=(f"{a.title()} and {b.title()} have a skill-profile "
                  f"similarity of {_fmt(row.observed, 4)}, ranking "
                  f"{rank} of 15 pairs"),
            citation=("task-08-tables/pair-verdicts.csv"
                      f"#observed@company_a={a},company_b={b}"),
            value=row.observed,
            verdict_source=("similarity_pair:task-08-tables/pair-verdicts.csv"
                            f"#verdict@company_a={a},company_b={b}"),
            clause=("similarity of what the two companies advertise for, not "
                    "of what they build; the rank is what is identified, not "
                    "the score"),
            falsifier=falsifier,
            action=action,
            audience="hr_talent",
            source_task="08",
        ))
    return claims


def candidates_trajectory(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """The convergence sentence, generated so its refusal is counted."""
    table = _load(root, "task-08-tables/trajectory-similarity.csv")
    claims = []
    for _, row in table.iterrows():
        a, b = str(row.company_a), str(row.company_b)
        claims.append(Claim(
            claim_id=f"traj-{a}-{b}",
            question="position",
            family="trajectory",
            subject=f"{a}|{b}",
            measures="correlation of monthly log share paths",
            text=(f"{a.title()} and {b.title()} moved together across 2023, "
                  f"with a monthly log-share correlation of "
                  f"{_fmt(row.r_log_share, 4)}"),
            citation=("task-08-tables/trajectory-similarity.csv"
                      f"#r_log_share@company_a={a},company_b={b}"),
            value=row.r_log_share,
            verdict_source=("feasible:task-08-tables/trajectory-similarity.csv"
                            f"#eligible@company_a={a},company_b={b}"
                            "&feasible:task-08-tables/trajectory-similarity.csv"
                            f"#excludes_zero@company_a={a},company_b={b}"),
            clause="eleven monthly points on a closed composition",
            falsifier="an interval excluding zero on a longer panel",
            action="none",
            audience="strategy",
            source_task="08",
        ))
    return claims


def candidates_salary(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """The pay-benchmark sentence, against Task 09's own committed audit.

    These cite `task-09-tables/`, which this task writes before it compiles.
    That ordering is deliberate: the audit needs row-level data, the row-level
    data is git-ignored, so the aggregate is committed first and the claim
    cites the aggregate like every other claim.
    """
    disclosure = _load(root, "task-09-tables/salary-disclosure.csv")
    stratified = _load(root, "task-09-tables/salary-pairs-stratified.csv")
    claims = []
    for _, row in disclosure.iterrows():
        company = str(row.company)
        claims.append(Claim(
            claim_id=f"salary-coverage-{company}",
            question="position",
            family="salary_coverage",
            subject=company,
            measures="share of postings disclosing an annual salary",
            text=(f"{company.title()} discloses a salary in "
                  f"{_fmt(row.disclosed_pct)}% of its 2023 postings"),
            citation=("task-09-tables/salary-disclosure.csv"
                      f"#disclosed_pct@company={company}"),
            value=row.disclosed_pct,
            verdict_source="structural:",
            clause=("a property of the publishers this company appears on, "
                    "not of its pay policy"),
            falsifier="a publisher-balanced panel showing the same rate",
            action="do not read disclosure rate as pay transparency",
            audience="hr_talent",
            source_task="09",
        ))
    for _, row in stratified.iterrows():
        a, b = str(row.company_a), str(row.company_b)
        stratum = str(row.stratum)
        slug = f"{a}-{b}-{_slug(stratum)}"
        claims.append(Claim(
            claim_id=f"salary-{slug}",
            question="position",
            family="salary_gap",
            subject=f"{a}|{b}:{stratum}",
            measures="median disclosed annual salary difference within one "
                     "publisher and one job function",
            text=(f"In {stratum} roles on {row.publisher}, {a.title()}'s "
                  f"median disclosed salary differs from {b.title()}'s by "
                  f"{_fmt(row.median_diff)}"),
            citation=("task-09-tables/salary-pairs-stratified.csv"
                      f"#median_diff@company_a={a},company_b={b},"
                      f"stratum={stratum}"),
            value=row.median_diff,
            verdict_source=("feasible:task-09-tables/"
                            "salary-pairs-stratified.csv"
                            f"#identified@company_a={a},company_b={b},"
                            f"stratum={stratum}"),
            clause=("disclosed salaries only, inside one publisher and one "
                    "job function; disclosure is missing not at random"),
            falsifier=("the same sign on a publisher-balanced sample of "
                       "disclosed salaries"),
            action="do not use for compensation benchmarking",
            audience="hr_talent",
            source_task="09",
        ))
    return claims


def candidates_collection(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """Claims about the panel itself.

    Structural facts — how many publishers carry a company, how much of the
    year the panel covers. They carry no sampling verdict because they are
    not estimates, and they matter more than they look: every refusal above is
    downstream of one of them.
    """
    comparability = _load(root, "task-06-tables/company-comparability.csv")
    claims = []
    for _, row in comparability.iterrows():
        company = str(row.company)
        claims.append(Claim(
            claim_id=f"panel-{company}",
            question="hiring_patterns",
            family="panel_structure",
            subject=company,
            measures="publishers carrying the company, and the share of its "
                     "postings inside the common panel",
            text=(f"{company.title()}'s postings arrive through "
                  f"{int(row.publishers)} publishers, of which the common "
                  f"panel carries {_pct(row.common_panel_share)}"),
            citation=("task-06-tables/company-comparability.csv"
                      f"#common_panel_share@company={company}"),
            value=row.common_panel_share,
            verdict_source="structural:",
            clause=("a property of the aggregator, not of the company's "
                    "recruiting"),
            falsifier="a direct-API panel showing a different publisher spread",
            action="scope every cross-company read to the common panel",
            audience="strategy",
            source_task="06",
        ))
    return claims


#: The sentences a reporting task actually reaches for. Each one is quoted or
#: paraphrased from an earlier task's own warning, and each is bound to a real
#: citation so that when it fails it fails at the *linter*, not for want of a
#: number. That distinction is the whole argument of this task: these are not
#: sentences nobody could support, they are sentences whose supporting number
#: exists and still does not license them.
TEMPTING_SENTENCES: tuple[dict, ...] = (
    {
        "claim_id": "tempting-looker",
        "question": "skill_demand",
        "subject": "google:Looker",
        "text": "Looker is emerging at Google, rising to {value} of its "
                "skilled postings",
        "citation": "task-05-tables/skill-stratified-verdicts.csv"
                    "#pooled_share_h2@skill=Looker",
        "why_tempting": "Task 05 §9 names this exact sentence as the one the "
                        "task exists to prevent; the pooled share really does "
                        "rise",
    },
    {
        "claim_id": "tempting-forecast",
        "question": "future_demand",
        "subject": "google",
        "text": "Google's share of the panel is expected to reach {value} in "
                "2024, based on the selected model",
        "citation": "task-07-tables/forecast.csv"
                    "#point_share@key=google,horizon=1",
        "why_tempting": "the number is committed, carries an interval, and "
                        "sits in a table called forecast.csv",
    },
    {
        "claim_id": "tempting-convergence",
        "question": "position",
        "subject": "google|meta",
        "text": "Google and Meta are converging on a single hiring profile, "
                "at a similarity of {value}",
        "citation": "task-08-tables/pair-verdicts.csv"
                    "#observed@company_a=google,company_b=meta",
        "why_tempting": "the pair verdict is robust, so the similarity is "
                        "real; the *movement* is what is unavailable",
    },
    {
        "claim_id": "tempting-level",
        "question": "hiring_patterns",
        "subject": "google|snowflake",
        "text": "Google posts more jobs than Snowflake, at {value} of the "
                "common panel",
        "citation": "task-06-tables/relative-share-by-half.csv"
                    "#h2_share@company=google",
        "why_tempting": "the counts are right there in the table and the "
                        "comparison looks like arithmetic",
    },
    {
        "claim_id": "tempting-seasonal",
        "question": "hiring_patterns",
        "subject": "google:february",
        "text": "Google's February dip is a seasonal pattern, {value} "
                "postings against a January baseline",
        "citation": "task-05-tables/volume-by-month.csv#postings@period=2023-02",
        "why_tempting": "the dip is the largest movement in the series and "
                        "the shape is familiar from other hiring data",
    },
    {
        "claim_id": "tempting-strategy",
        "question": "hiring_patterns",
        "subject": "google:Sales",
        "text": "Sales rose to {value} of postings because Google made a "
                "deliberate shift towards commercial roles",
        "citation": "task-05-tables/panel-check-job-function.csv"
                    "#share_h2@segment=Sales",
        "why_tempting": "the movement survives the balanced panel, so the "
                        "number is publishable and only the *why* is not",
    },
    {
        "claim_id": "tempting-product",
        "question": "tech_stack",
        "subject": "google:Vertex AI",
        "text": "Google's hiring points to a new product, with BigQuery in "
                "{value} of its skilled postings",
        "citation": "task-06-tables/skill-distinctiveness-google.csv"
                    "#share@skill=BigQuery",
        "why_tempting": "this is the reading a competitive-intelligence "
                        "reader most wants, and the skill really is "
                        "distinctive",
    },
    {
        "claim_id": "tempting-headcount",
        "question": "hiring_patterns",
        "subject": "google",
        "text": "Google's headcount grew over 2023, with {value} of the "
                "common panel in H2",
        "citation": "task-06-tables/relative-share-by-half.csv"
                    "#h2_share@company=google",
        "why_tempting": "postings are routinely read as headcount, and no "
                        "column in this schema says otherwise",
    },
    {
        "claim_id": "tempting-country",
        "question": "hiring_patterns",
        "subject": "google:Singapore",
        "text": "Google is shifting hiring towards Singapore, now {value} of "
                "its postings",
        "citation": "task-05-tables/panel-check-country.csv"
                    "#share_h2@segment=Singapore",
        "why_tempting": "Task 06 §11 forbids country comparison because "
                        "publisher coverage is national, and the share still "
                        "moves",
    },
    {
        "claim_id": "tempting-share-of-all",
        "question": "skill_demand",
        "subject": "google:Python",
        "text": "Python appears in {value} share of all postings at Google",
        "citation": "task-05-tables/skill-stratified-verdicts.csv"
                    "#pooled_share_h2@skill=Python",
        "why_tempting": "the denominator swap is invisible in prose and "
                        "makes every skill number look larger",
    },
)


def candidates_tempting(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """The sentences an author would write, bound to numbers that do exist.

    Without this family the linter never fires, because every other candidate
    here is generated from a template that was written to be clean. A gate that
    nothing trips is not evidence of a well-behaved corpus; it is an untested
    gate. These ten are the corpus it was built for.
    """
    claims = []
    for item in TEMPTING_SENTENCES:
        try:
            value = resolve_citation(item["citation"], root)
        except (FileNotFoundError, KeyError, LookupError, ValueError):
            value = None
        text = item["text"].format(value=_fmt(value, 4) if value is not None
                                   else "an unavailable number")
        claims.append(Claim(
            claim_id=item["claim_id"],
            question=item["question"],
            family="tempting",
            subject=item["subject"],
            measures="the sentence a reader expects, against the number that "
                     "exists",
            text=text,
            citation=item["citation"],
            value=value,
            verdict_source="structural:",
            clause="",
            falsifier="a table in this repository that measures the construct "
                      "the sentence names",
            action="none; this sentence is not available",
            audience="strategy",
            source_task="09",
            notes=item["why_tempting"],
        ))
    return claims


#: Every generator, in report order.
GENERATORS = (
    candidates_volume,
    candidates_share,
    candidates_mix,
    candidates_skill_trend,
    candidates_skill_gap,
    candidates_distinctiveness,
    candidates_forecast,
    candidates_similarity,
    candidates_trajectory,
    candidates_salary,
    candidates_collection,
    candidates_tempting,
)


def all_candidates(root: Path = DEFAULT_MEMBER, focus: str = "google"):
    """Mint every candidate claim this evidence base can be asked to make."""
    claims = []
    for generator in GENERATORS:
        claims.extend(generator(root, focus))
    return claims


# ---------------------------------------------------------------------------
# K. What the brief promised, and what the evidence base can pay
# ---------------------------------------------------------------------------
#
# The brief's "Why it matters" section lists six things this project would let
# a reader do. They are the promises the deliverable is judged against, so the
# honest close of a reporting task is to walk them one by one and say which the
# evidence base can pay. Five of the six are decided by tasks already closed;
# only "benchmark salaries" was still open when Task 09 began, which is why the
# salary audit above exists.

#: (promise, the question it maps to, the families that could pay it).
BRIEF_PROMISES: tuple[dict, ...] = (
    {
        "promise": "Forecast future demand for skills, technologies and roles",
        "question": "future_demand",
        "families": ("forecastability", "forecast_horizon"),
        "settled_by": "task-07 §8.3",
    },
    {
        "promise": "Predict competitor product launches",
        "question": "tech_stack",
        "families": (),
        "settled_by": "task-02 scope",
    },
    {
        "promise": "Identify emerging technologies and market skill gaps",
        "question": "skill_demand",
        "families": ("skill_trend", "skill_gap", "distinctiveness"),
        "settled_by": "task-05 §9, task-06 §7",
    },
    {
        "promise": "Plan hiring and training",
        "question": "skill_demand",
        "families": ("skill_trend", "segment_mix"),
        "settled_by": "task-05 §6",
    },
    {
        # `salary_coverage` is deliberately not listed. Those six claims
        # publish, but they say who discloses pay, not what anyone pays — and
        # counting them here would report the promise as paid on the strength
        # of sentences that do not pay it. A promise is settled by the family
        # that answers it, never by an adjacent family that survived.
        "promise": "Benchmark salaries",
        "question": "position",
        "families": ("salary_gap",),
        "settled_by": "task-09 §7",
    },
    {
        "promise": "Detect early signs of expansion or slowdown",
        "question": "hiring_patterns",
        "families": ("volume", "relative_share"),
        "settled_by": "task-05 §4, task-06 §3",
    },
)


def promise_audit(ledger: pd.DataFrame,
                  promises=BRIEF_PROMISES) -> pd.DataFrame:
    """One row per promise in the brief: paid, partly paid, or not paid.

    "Partly paid" is the honest middle and it needs its own label, because a
    promise met only with a clause attached is not the promise a reader heard.
    """
    rows = []
    for item in promises:
        block = ledger[ledger.family.isin(item["families"])] \
            if item["families"] else ledger.iloc[0:0]
        published = int((block.status == PUBLISHED).sum())
        qualified = int((block.status == QUALIFIED).sum())
        candidates = int(len(block))
        if not item["families"]:
            status = "not payable"
            detail = "no table in this repository measures the construct"
        elif published:
            status = "paid"
            detail = (f"{published} of {candidates} proposed claims publish "
                      f"on a firm upstream verdict")
        elif qualified:
            status = "partly paid"
            detail = (f"0 of {candidates} rest on a firm verdict; {qualified} "
                      f"publish on a verdict that is itself qualified")
        else:
            status = "not paid"
            detail = f"all {candidates} proposed claims are refused"
        rows.append({
            "promise": item["promise"],
            "question": item["question"],
            "generated": candidates,
            "published": published,
            "published_qualified": qualified,
            "refused": int((block.status == REFUSED).sum()),
            "status": status,
            "detail": detail,
            "settled_by": item["settled_by"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# L. Actionability, and the company's position
# ---------------------------------------------------------------------------
#
# The brief asks each member for "a short explanation of the company's hiring
# strategy & position". Both nouns are traps. *Strategy* is an intent, and
# `posting_date` is an aggregator's first-seen date — the series describes
# discovery, so it cannot carry an intent story (Task 05 §1). *Position* as a
# level is not identified (Task 06 §1.3). What survives is position as a
# *profile*: what this company asks for relative to the other five, which is
# exactly what Tasks 06 and 08 measured.

#: A published sentence still has to change something. These are the audiences
#: the brief names, and the test is whether a claim would move one of them.
def actionability_table(ledger: pd.DataFrame) -> pd.DataFrame:
    """Per audience: how many surviving claims would change what they do.

    A claim whose action is "none" is still worth publishing — several of the
    most useful lines in this report are refusals that stop a decision being
    taken on a number that cannot carry it — but it should be counted
    separately from one that moves a decision.
    """
    survivors = ledger[ledger.status != REFUSED]
    rows = []
    for audience in AUDIENCES:
        block = survivors[survivors.audience == audience]
        actionable = block[~block.action.fillna("").str.strip()
                           .str.lower().str.startswith("none")]
        rows.append({
            "audience": audience,
            "claims_available": int(len(block)),
            "actionable": int(len(actionable)),
            "informational_only": int(len(block) - len(actionable)),
            "questions_covered": ", ".join(sorted(block.question.unique())),
        })
    return pd.DataFrame(rows)


def question_coverage(ledger: pd.DataFrame) -> pd.DataFrame:
    """Per question the brief asks, what the reader actually gets."""
    rows = []
    for question in QUESTIONS:
        block = ledger[ledger.question == question]
        survivors = block[block.status != REFUSED]
        rows.append({
            "question": question,
            "generated": int(len(block)),
            "answerable": int(len(survivors)),
            "verdict_firm": int((block.status == PUBLISHED).sum()),
            "verdict_qualified": int((block.status == QUALIFIED).sum()),
            "answer": ("no sentence available" if survivors.empty
                       else "answerable only on a qualified verdict"
                       if not (block.status == PUBLISHED).any()
                       else "answerable"),
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class PositionVerdict:
    """What can be said about where a company sits among the six."""

    focus: str
    profile_available: bool
    level_available: bool
    trajectory_available: bool
    nearest_pair: str
    nearest_score: float
    nearest_verdict: str
    distinctive_skills: int
    detail: str


def strategy_position(ledger: pd.DataFrame, root: Path = DEFAULT_MEMBER,
                      focus: str = "google") -> PositionVerdict:
    """The brief's "hiring strategy & position" question, answered in kind.

    Three candidate readings of "position", and only one of them survives:
    a *level* (how big), a *trajectory* (where it is heading), and a *profile*
    (what it asks for). This returns all three flags so the report cannot quote
    the surviving one without the two that did not.
    """
    pairs = _load(root, "task-08-tables/pair-verdicts.csv")
    mine = pairs[(pairs.company_a == focus) | (pairs.company_b == focus)] \
        if not pairs.empty else pairs
    if not mine.empty:
        best = mine.sort_values("observed", ascending=False).iloc[0]
        other = (best.company_b if best.company_a == focus else best.company_a)
        nearest, score, verdict = (str(other), float(best.observed),
                                   str(best.verdict))
    else:
        nearest, score, verdict = "", float("nan"), "unavailable"

    level = bool(((ledger.family == "volume")
                  & (ledger.status == PUBLISHED)).any())
    trajectory = bool(((ledger.family == "trajectory")
                       & (ledger.status != REFUSED)).any())
    distinctive = int(((ledger.family == "distinctiveness")
                       & (ledger.subject.str.startswith(f"{focus}:"))
                       & (ledger.status != REFUSED)).sum())
    profile = distinctive > 0 and verdict in ("robust", "vendor_dependent")

    detail = (
        f"position as a profile is available: {distinctive} skills separate "
        f"{focus.title()} from the other five after FDR control, and its "
        f"nearest neighbour is {nearest.title()} at {score:.4f} "
        f"({verdict}). Position as a level is not — a posting count is a "
        f"count of the boards that syndicate it. Position as a trajectory is "
        f"not — trajectory similarity is refused."
    )
    return PositionVerdict(
        focus=focus, profile_available=profile, level_available=level,
        trajectory_available=trajectory, nearest_pair=nearest,
        nearest_score=round(score, 4), nearest_verdict=verdict,
        distinctive_skills=distinctive, detail=detail,
    )


def audience_brief(ledger: pd.DataFrame, audience: str) -> pd.DataFrame:
    """The sentences one audience may be handed, with their clauses attached."""
    if audience not in AUDIENCES:
        raise ValueError(f"{audience!r} is not one of {AUDIENCES}")
    block = ledger[(ledger.audience == audience)
                   & (ledger.status != REFUSED)].copy()
    block["sentence"] = block.apply(sentence, axis=1)
    columns = ["claim_id", "question", "status", "sentence", "action",
               "falsifier", "citation"]
    return block[columns].reset_index(drop=True)


def claim_provenance(ledger: pd.DataFrame) -> pd.DataFrame:
    """Which task each surviving sentence rests on.

    A reporting layer that quietly re-derives its own numbers is unauditable;
    this table exists so a reviewer can check that Task 09 computed nothing it
    did not have to.
    """
    survivors = ledger[ledger.status != REFUSED]
    rows = []
    for task, block in survivors.groupby("source_task"):
        rows.append({
            "source_task": str(task),
            "claims": int(len(block)),
            "tables_cited": int(block.citation.str.split("#").str[0].nunique()),
            "families": ", ".join(sorted(block.family.unique())),
        })
    return pd.DataFrame(rows).sort_values("source_task").reset_index(drop=True)


def falsifier_table(ledger: pd.DataFrame) -> pd.DataFrame:
    """Every published sentence with the observation that would overturn it.

    A claim with no falsifier is not a finding, it is a position. Task 10 gets
    this table so the presentation can be argued with.
    """
    survivors = ledger[ledger.status != REFUSED].copy()
    survivors["sentence"] = survivors.apply(sentence, axis=1)
    return survivors[["claim_id", "question", "status", "sentence",
                      "falsifier"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# M. Privacy — the standing check, re-run on this task's own tables
# ---------------------------------------------------------------------------


def forbidden_columns(table: pd.DataFrame) -> list[str]:
    """Delegate to the shared rule so there is one definition, not two."""
    return cmp.forbidden_columns(table)


def personal_data_columns_present(table: pd.DataFrame) -> list[str]:
    """Task 01 §5's standing check, run against every table this task commits."""
    return cmp.personal_data_columns_present(table)
