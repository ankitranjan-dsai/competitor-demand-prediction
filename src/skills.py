"""Task 04 — skill extraction & feature engineering.

Shared skill taxonomy + extraction engine for all four company specialists.
Rationale and the team standard live in docs/task-04-skill-taxonomy.md.

Why this module exists
----------------------
Task 06 compares the four companies on their skills, and Task 08 scores how
similar they are. Both are meaningless if one specialist's ``python`` is
another's ``Python 3`` and a third's ``py``. So the taxonomy is **one shared
in-repo table**, not four per-member keyword lists, and it is the single place
a skill's canonical name and category are decided.

Three extraction paths, one output
----------------------------------
============  =============================================  ================
path          input                                          Google coverage
============  =============================================  ================
``source``    the collector's own ``extracted_skills`` list  67% of rows
``title``     ``job_title_clean`` (Task 03)                   100% of rows
``text``      ``cleaned_description`` (Task 03 Layer B)       0% of rows today
============  =============================================  ================

The paths are unioned into ``skills_final`` and every posting records which
path(s) produced its skills. Provenance is kept because the paths are *not*
equally trustworthy: ``source`` skills come from full description text we
never saw, ``title`` skills are precise but sparse, and ``text`` skills are the
ones this module actually controls. The Google backfill carries no description
text, so the ``text`` path is validated separately against real posting HTML by
``src/validate_skill_extraction.py`` — same approach Task 03 used for Layer B.

Ambiguity is the hard part
--------------------------
The high-value skills have short, common-English names. ``excel in a
fast-paced environment``, ``spark innovation``, ``react to customer needs``,
``go-to-market`` and ``swift decision-making`` all appear in real postings and
all become false skills under a naive keyword match. Skills flagged
``context_required`` are only accepted when a qualifier ("proficiency in
Excel", "Spark cluster") or a delimited list containing an unambiguous skill
("Python, R, SQL") confirms them. Rejections are counted, not discarded
silently, so the guard's behaviour is measurable.

Usage
-----
    from skills import extract_skills, build_posting_features
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

# Ordered: earlier categories win when a posting's "primary" category is a tie.
CATEGORIES = (
    "ML / AI",
    "Data Engineering",
    "Programming Language",
    "Data Science Libraries",
    "Cloud Platform",
    "Database",
    "DevOps / Infrastructure",
    "Analytics / BI",
    "Web / Frontend",
    "OS / Shell",
    "Governance / Compliance",
    "Facilities / Data Centre",
    "Office / Productivity",
)

# Excluded from `tech_stack_tags`: knowing a posting mentions Outlook or GDPR
# says nothing about the technology being built. They stay in the long table
# (dropping data is worse than labelling it) but never reach a stack feature.
NON_STACK_CATEGORIES = frozenset(
    {"Office / Productivity", "Governance / Compliance", "Facilities / Data Centre"}
)


def category_slug(category: str) -> str:
    """'ML / AI' -> 'ml_ai', for feature column names."""
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Skill:
    """One canonical skill.

    ``name``        display name used in every output table.
    ``category``    one of CATEGORIES.
    ``aliases``     exact lowercase strings a collector may emit for it.
    ``patterns``    extra regex fragments for free text (name + aliases are
                    matched automatically).
    ``context_required``  name collides with ordinary English; only accept a
                    free-text mention that a qualifier or a skill list confirms.
    ``qualifiers``  regex fragments that on their own confirm an ambiguous
                    mention ("golang", "r studio", "power automate").
    ``is_concept``  a method or practice rather than a named tool, so it feeds
                    ``method_tags`` instead of ``tech_stack_tags``.
    """

    name: str
    category: str
    aliases: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    context_required: bool = False
    qualifiers: tuple[str, ...] = ()
    is_concept: bool = False
    key: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", re.sub(r"\s+", "_", self.name.strip().lower()))


def _s(name, category, *, aliases=(), patterns=(), context=False, qualifiers=(),
       concept=False) -> Skill:
    return Skill(name, category, tuple(aliases), tuple(patterns), context,
                 tuple(qualifiers), concept)


_PROG = "Programming Language"
_MLAI = "ML / AI"
_DSLIB = "Data Science Libraries"
_DENG = "Data Engineering"
_DB = "Database"
_CLOUD = "Cloud Platform"
_DEVOPS = "DevOps / Infrastructure"
_BI = "Analytics / BI"
_WEB = "Web / Frontend"
_OS = "OS / Shell"
_GOV = "Governance / Compliance"
_FAC = "Facilities / Data Centre"
_OFFICE = "Office / Productivity"

# Generic qualifiers used by the context guard, shared across ambiguous skills.
_QUALIFIER_BEFORE = re.compile(
    r"(?:experience|experienced|proficien\w*|expert\w*|knowledge|skilled|skills?|"
    r"familiar\w*|fluent|competen\w*|background|hands[\s-]?on|using|use of|"
    r"utilis\w*|utiliz\w*|written in|write in|coding in|programming in|"
    r"stack|technologies|languages?|tools?|frameworks?)\b[^.;!?\n]{0,25}$",
    re.IGNORECASE,
)
_QUALIFIER_AFTER = re.compile(
    r"^\s*(?:developer|engineer|programming|programmer|language|framework|library|"
    r"libraries|scripting|script|skills?|experience|expertise|proficiency|"
    r"certification|certified|stack|codebase|components?|cluster|clusters|"
    r"environment|pipelines?|jobs?|queries|query|dashboards?|models?|"
    r"spreadsheets?|macros|vba|workbooks?)\b",
    re.IGNORECASE,
)
# A mention sitting in a delimited run ("Python, R, SQL" / "Java/Go/C++").
_LIST_DELIMS = ",;|/&+"
_LIST_WORDS = re.compile(r"\b(?:and|or|plus)\b\s*$", re.IGNORECASE)
_LIST_WINDOW = 60


SKILLS: tuple[Skill, ...] = (
    # ---------------- Programming languages ----------------
    _s("Python", _PROG, aliases=("python3", "python 3")),
    # SQL is a language, not a "database" — the database is what it queries.
    _s("SQL", _PROG),
    # R and C are single letters; without the context guard they match the
    # article "a"-equivalent noise in every posting on earth.
    _s("R", _PROG, context=True,
       qualifiers=(r"\br\s*studio\b", r"\brstudio\b", r"\br\s*/\s*python",
                   r"python\s*/\s*r\b", r"\br\s+shiny\b", r"\bin r\b")),
    _s("C", _PROG, context=True, qualifiers=(r"\bc\s*/\s*c\+\+", r"\bansi c\b",
                                             r"\bembedded c\b")),
    _s("C++", _PROG, aliases=("cpp", "c plus plus")),
    _s("C#", _PROG, aliases=("csharp", "c sharp")),
    _s("Java", _PROG),
    # "go" is the commonest English verb in a job posting; "go-to-market" must
    # never become the Go language.
    _s("Go", _PROG, aliases=("golang",),
       patterns=(r"\bgo\b(?!\s*[-\s]?to[-\s]market)",), context=True,
       qualifiers=(r"\bgolang\b", r"\bgo\s+lang\b")),
    _s("JavaScript", _PROG, aliases=("js",)),
    _s("TypeScript", _PROG, aliases=("ts",)),
    _s("Scala", _PROG),
    _s("MATLAB", _PROG),
    _s("Perl", _PROG),
    _s("Rust", _PROG, context=True),
    _s("Kotlin", _PROG),
    _s("Swift", _PROG, context=True),
    _s("Objective-C", _PROG, aliases=("objective c", "objc")),
    _s("PHP", _PROG),
    _s("Ruby", _PROG, context=True),
    _s("Dart", _PROG, context=True),
    _s("Julia", _PROG, context=True),
    _s("Assembly", _PROG, context=True),
    _s("VBA", _PROG),
    _s("Groovy", _PROG, context=True),

    # ---------------- ML / AI ----------------
    _s("TensorFlow", _MLAI, aliases=("tensor flow",)),
    _s("PyTorch", _MLAI, aliases=("torch",)),
    _s("Keras", _MLAI),
    _s("scikit-learn", _MLAI, aliases=("sklearn", "scikit learn")),
    _s("MXNet", _MLAI),
    _s("JAX", _MLAI, context=True),
    _s("XGBoost", _MLAI),
    _s("LightGBM", _MLAI),
    _s("OpenCV", _MLAI),
    _s("spaCy", _MLAI),
    _s("NLTK", _MLAI),
    _s("Hugging Face", _MLAI, aliases=("huggingface",)),
    _s("Transformers", _MLAI, context=True,
       qualifiers=(r"\btransformer\s+(?:models?|architectures?)\b",)),
    _s("LangChain", _MLAI),
    _s("CUDA", _MLAI),
    _s("ONNX", _MLAI),
    _s("TensorRT", _MLAI),
    _s("Ray", _MLAI, context=True, qualifiers=(r"\bray\s+(?:tune|serve|cluster)\b",)),
    _s("MLflow", _MLAI, aliases=("ml flow",)),
    _s("Kubeflow", _MLAI),
    _s("Vertex AI", _MLAI, aliases=("vertexai",)),
    _s("SageMaker", _MLAI, aliases=("sage maker",)),
    _s("Machine Learning", _MLAI, aliases=("ml",), concept=True),
    _s("Deep Learning", _MLAI, concept=True),
    _s("NLP", _MLAI, aliases=("natural language processing",), concept=True),
    _s("Computer Vision", _MLAI, concept=True),
    _s("Reinforcement Learning", _MLAI, aliases=("rl",), concept=True),
    _s("Generative AI", _MLAI, aliases=("genai", "gen ai"), concept=True),
    _s("LLM", _MLAI, aliases=("llms", "large language model",
                              "large language models"), concept=True),
    _s("RAG", _MLAI, aliases=("retrieval augmented generation",), context=True,
       concept=True),
    _s("Recommender Systems", _MLAI, aliases=("recommendation systems",),
       concept=True),

    # ---------------- Data science libraries ----------------
    _s("pandas", _DSLIB, context=True,
       qualifiers=(r"\bpandas\s*[,/]", r"python.{0,30}\bpandas\b")),
    _s("NumPy", _DSLIB, aliases=("numpy",)),
    _s("SciPy", _DSLIB),
    _s("Matplotlib", _DSLIB),
    _s("Seaborn", _DSLIB),
    _s("Plotly", _DSLIB),
    _s("Jupyter", _DSLIB, aliases=("jupyter notebook", "jupyterlab")),
    _s("ggplot2", _DSLIB, aliases=("ggplot",)),
    _s("dplyr", _DSLIB),

    # ---------------- Data engineering ----------------
    _s("Spark", _DENG, aliases=("apache spark", "pyspark"), context=True,
       qualifiers=(r"\bapache\s+spark\b", r"\bpy\s?spark\b", r"\bspark\s+sql\b")),
    _s("Hadoop", _DENG, aliases=("apache hadoop",)),
    _s("Kafka", _DENG, aliases=("apache kafka",)),
    _s("Airflow", _DENG, aliases=("apache airflow",)),
    _s("dbt", _DENG),
    _s("Apache Beam", _DENG, aliases=("beam",), context=True,
       qualifiers=(r"\bapache\s+beam\b", r"\bdataflow\b")),
    _s("Flink", _DENG, aliases=("apache flink",)),
    _s("Hive", _DENG, aliases=("apache hive",), context=True,
       qualifiers=(r"\bapache\s+hive\b", r"\bhive\s*ql\b")),
    _s("Presto", _DENG, context=True),
    _s("Trino", _DENG),
    _s("NiFi", _DENG),
    _s("Delta Lake", _DENG),
    _s("ETL", _DENG, aliases=("elt", "etl/elt"), concept=True),
    _s("Data Modelling", _DENG, aliases=("data modeling", "data modelling"),
       concept=True),

    # ---------------- Databases ----------------
    _s("MongoDB", _DB, aliases=("mongo",)),
    _s("MySQL", _DB),
    _s("PostgreSQL", _DB, aliases=("postgres", "postgresql")),
    _s("SQL Server", _DB, aliases=("mssql", "microsoft sql server")),
    _s("Oracle", _DB),
    _s("Cassandra", _DB),
    _s("Elasticsearch", _DB, aliases=("elastic search",)),
    _s("Redis", _DB),
    _s("NoSQL", _DB),
    _s("Bigtable", _DB, aliases=("big table",)),
    _s("Spanner", _DB, context=True, qualifiers=(r"\bcloud\s+spanner\b",)),
    _s("DynamoDB", _DB),
    _s("Neo4j", _DB),
    _s("SQLite", _DB),
    _s("MariaDB", _DB),
    _s("Microsoft Access", _DB, aliases=("ms access",)),

    # ---------------- Cloud platforms ----------------
    _s("GCP", _CLOUD, aliases=("google cloud", "google cloud platform")),
    _s("AWS", _CLOUD, aliases=("amazon web services",)),
    _s("Azure", _CLOUD, aliases=("microsoft azure",)),
    _s("BigQuery", _CLOUD, aliases=("big query",)),
    _s("Snowflake", _CLOUD),
    _s("Redshift", _CLOUD, aliases=("amazon redshift",)),
    _s("Databricks", _CLOUD),
    _s("Firebase", _CLOUD),
    _s("OpenStack", _CLOUD, aliases=("open stack",)),
    _s("VMware", _CLOUD, aliases=("vm ware",)),
    _s("S3", _CLOUD, aliases=("amazon s3",)),

    # ---------------- DevOps / infrastructure ----------------
    _s("Kubernetes", _DEVOPS, aliases=("k8s",)),
    _s("Docker", _DEVOPS),
    _s("Terraform", _DEVOPS),
    _s("Jenkins", _DEVOPS),
    _s("Ansible", _DEVOPS),
    _s("Chef", _DEVOPS, context=True,
       qualifiers=(r"\bchef\s+(?:cookbook|recipe|infra)", r"puppet.{0,20}\bchef\b",
                   r"\bchef\b.{0,20}puppet")),
    _s("Puppet", _DEVOPS, context=True),
    _s("Git", _DEVOPS, context=True, qualifiers=(r"\bgit\s+(?:repo|branch|flow)",)),
    _s("GitHub", _DEVOPS, aliases=("github actions",)),
    _s("GitLab", _DEVOPS),
    _s("Prometheus", _DEVOPS),
    _s("Grafana", _DEVOPS),
    _s("Splunk", _DEVOPS),
    _s("Bazel", _DEVOPS),
    _s("Helm", _DEVOPS, context=True),
    _s("CI/CD", _DEVOPS, aliases=("ci/cd", "cicd", "ci cd"), concept=True),

    # ---------------- Analytics / BI ----------------
    _s("Looker", _BI, aliases=("looker studio",)),
    _s("Tableau", _BI),
    _s("Power BI", _BI, aliases=("powerbi",)),
    # SAS is an analytics platform with its own language. The source files it
    # under BOTH "programming" and "analyst_tools"; one home only, here.
    _s("SAS", _BI),
    _s("SPSS", _BI),
    _s("Qlik", _BI, aliases=("qlikview", "qlik sense")),
    _s("Cognos", _BI),
    _s("MicroStrategy", _BI, aliases=("micro strategy",)),
    _s("Alteryx", _BI),
    _s("Stata", _BI),
    _s("Google Analytics", _BI),
    # SAP is an ERP suite, but in a *data* posting it means the reporting and
    # warehouse side (BW / HANA / Analytics Cloud), so it sits with the BI tools.
    _s("SAP", _BI),
    _s("SAP HANA", _DB, aliases=("hana",)),
    # "excel in a fast-paced environment" is the single most common false
    # positive in job-posting skill extraction.
    _s("Excel", _BI, aliases=("microsoft excel", "ms excel"), context=True,
       qualifiers=(r"\b(?:microsoft|ms)\s+excel\b", r"\bexcel\s+(?:spreadsheet|"
                   r"model|macro|pivot|vba)", r"\badvanced excel\b")),
    _s("Google Sheets", _BI, aliases=("sheets", "spreadsheets"), context=True,
       qualifiers=(r"\bgoogle\s+sheets\b",)),
    _s("Data Studio", _BI),

    # ---------------- Web / frontend ----------------
    _s("React", _WEB, aliases=("react.js", "reactjs"), context=True,
       qualifiers=(r"\breact\.?js\b", r"\breact\s+native\b", r"\breact\s+hooks?\b")),
    _s("Angular", _WEB, aliases=("angularjs",), context=True),
    _s("Vue.js", _WEB, aliases=("vue", "vuejs")),
    _s("Node.js", _WEB, aliases=("node", "nodejs")),
    _s("Express", _WEB, aliases=("express.js", "expressjs"), context=True,
       qualifiers=(r"\bexpress\.?js\b", r"\bnode.{0,15}express\b")),
    _s("Django", _WEB),
    _s("Flask", _WEB, context=True),
    _s("Spring", _WEB, context=True,
       qualifiers=(r"\bspring\s+(?:boot|framework|mvc)\b",)),
    _s(".NET", _WEB, aliases=(".net", "dotnet", "asp.net")),
    _s("Ruby on Rails", _WEB, aliases=("rails", "ruby on rails")),
    _s("HTML", _WEB, aliases=("html5",)),
    _s("CSS", _WEB, aliases=("css3",)),
    _s("GraphQL", _WEB, aliases=("graph ql",)),
    _s("Flutter", _WEB),

    # ---------------- OS / shell ----------------
    _s("Linux", _OS),
    _s("Unix", _OS),
    _s("Windows", _OS, context=True,
       qualifiers=(r"\b(?:microsoft\s+)?windows\s+(?:server|10|11|admin)",
                   r"\bwindows\s*/\s*linux", r"\blinux\s*/\s*windows")),
    _s("macOS", _OS, aliases=("mac os", "osx")),
    _s("Ubuntu", _OS),
    _s("Bash", _OS),
    _s("Shell", _OS, aliases=("shell scripting",), context=True,
       qualifiers=(r"\bshell\s+script", r"\b(?:bash|unix|linux)\s+shell\b")),
    _s("PowerShell", _OS, aliases=("power shell",)),

    # ---------------- Governance / compliance ----------------
    _s("GDPR", _GOV),
    _s("HIPAA", _GOV),
    _s("SOC 2", _GOV, aliases=("soc2",)),
    _s("ISO 27001", _GOV),
    _s("CCPA", _GOV),

    # ---------------- Facilities / data centre ----------------
    # Not software skills. Kept because 11% of Google's postings are data-centre
    # facilities roles (Task 03 §2.1) and erasing their skills would hide that.
    _s("Colocation", _FAC, aliases=("colo",)),
    _s("HVAC", _FAC),
    _s("SCADA", _FAC),
    _s("AutoCAD", _FAC, aliases=("auto cad",)),
    _s("BMS", _FAC),
    _s("PLC", _FAC),

    # ---------------- Office / productivity ----------------
    _s("Microsoft Word", _OFFICE, aliases=("word", "ms word"), context=True,
       qualifiers=(r"\b(?:microsoft|ms)\s+word\b",)),
    _s("Microsoft Outlook", _OFFICE, aliases=("outlook", "ms outlook")),
    _s("PowerPoint", _OFFICE, aliases=("power point", "ms powerpoint")),
    _s("SharePoint", _OFFICE, aliases=("share point",)),
    _s("Jira", _OFFICE),
    _s("Confluence", _OFFICE),
)


# Terms a collector may emit that are NOT skills for this project. Recorded
# with a reason and counted in the quality report rather than dropped in
# silence — an unexplained disappearance is indistinguishable from a bug.
EXCLUDED_TERMS: dict[str, str] = {
    "flow": (
        "irreducibly ambiguous — Microsoft Power Automate (ex-Flow), Facebook's "
        "Flow type checker, Apache NiFi flows and the plain English noun all "
        "collapse to this token; 8 Google postings carry it with no way to tell "
        "which was meant"
    ),
    "terminal": (
        "a terminal is an interface, not a skill; the mention carries no "
        "information about what the candidate must be able to do"
    ),
}


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_BY_KEY: dict[str, Skill] = {s.key: s for s in SKILLS}
_BY_NAME: dict[str, Skill] = {s.name.lower(): s for s in SKILLS}

_ALIAS_TO_SKILL: dict[str, Skill] = {}
for _skill in SKILLS:
    for _token in (_skill.name.lower(), *_skill.aliases):
        _existing = _ALIAS_TO_SKILL.get(_token)
        if _existing is not None and _existing is not _skill:
            raise ValueError(
                f"alias {_token!r} claimed by both {_existing.name} and {_skill.name}"
            )
        _ALIAS_TO_SKILL[_token] = _skill

CATEGORY_OF: dict[str, str] = {s.name: s.category for s in SKILLS}

_UNKNOWN_CATEGORIES = {s.category for s in SKILLS} - set(CATEGORIES)
if _UNKNOWN_CATEGORIES:
    raise ValueError(f"skills use undeclared categories: {sorted(_UNKNOWN_CATEGORIES)}")


def _compile(skill: Skill) -> re.Pattern[str]:
    """Word-boundary regex for a skill's name, aliases and extra patterns.

    ``\\b`` does not work on tokens ending in punctuation (``c++``, ``c#``,
    ``.net``), so those get an explicit non-word-adjacent guard instead.
    """
    if skill.patterns:
        frags = list(skill.patterns)
    else:
        frags = []
        for token in (skill.name, *skill.aliases):
            esc = re.escape(token.lower())
            left = r"\b" if token[0].isalnum() else r"(?<![\w.])"
            right = r"\b" if token[-1].isalnum() else r"(?![\w.])"
            frags.append(f"{left}{esc}{right}")
    return re.compile("|".join(frags), re.IGNORECASE)


_PATTERNS: dict[str, re.Pattern[str]] = {s.name: _compile(s) for s in SKILLS}
_QUALIFIER_PATTERNS: dict[str, re.Pattern[str]] = {
    s.name: re.compile("|".join(s.qualifiers), re.IGNORECASE)
    for s in SKILLS
    if s.qualifiers
}

UNAMBIGUOUS_SKILLS: tuple[Skill, ...] = tuple(
    s for s in SKILLS if not s.context_required
)


def get_skill(name: str) -> Skill | None:
    """Look up a skill by canonical name, key or alias."""
    token = str(name).strip().lower()
    return _BY_NAME.get(token) or _BY_KEY.get(token) or _ALIAS_TO_SKILL.get(token)


# ---------------------------------------------------------------------------
# Path A — normalise a collector's own skill list
# ---------------------------------------------------------------------------


def normalize_skill_list(
    raw: Iterable[str] | None, unmapped: Counter | None = None
) -> list[str]:
    """Map a list of raw skill strings onto canonical taxonomy names.

    De-duplicates, preserves first-seen order, records anything the taxonomy
    does not know in ``unmapped`` so coverage is measurable rather than assumed.
    Excluded terms are dropped by design and are *not* counted as unmapped.
    """
    if raw is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = str(item).strip().lower()
        if not token or token in EXCLUDED_TERMS:
            continue
        skill = get_skill(token)
        if skill is None:
            if unmapped is not None:
                unmapped[token] += 1
            continue
        if skill.name not in seen:
            seen.add(skill.name)
            out.append(skill.name)
    return out


# ---------------------------------------------------------------------------
# Paths B & C — extract from free text (description) or a job title
# ---------------------------------------------------------------------------


def _confirmed_by_context(text: str, start: int, end: int, skill: Skill,
                          confident_spans: list[tuple[int, int]]) -> bool:
    """Decide whether an ambiguous mention is really this skill.

    Accepted when any of three things hold:
      1. a skill-specific qualifier appears in the surrounding window
         ("golang", "R Studio", "Microsoft Excel");
      2. a generic qualifier frames it ("proficiency in <skill>",
         "<skill> developer", "<skill> cluster");
      3. it sits in a delimited list that also contains an unambiguous skill
         ("Python, R, SQL").
    """
    window_start = max(0, start - _LIST_WINDOW)
    window = text[window_start : end + _LIST_WINDOW]

    qualifier = _QUALIFIER_PATTERNS.get(skill.name)
    if qualifier is not None and qualifier.search(window):
        return True

    before = text[max(0, start - 40) : start]
    if _QUALIFIER_BEFORE.search(before):
        return True
    if _QUALIFIER_AFTER.match(text[end : end + 30]):
        return True

    # List context: a delimiter must sit next to the mention *and* a skill we
    # are sure about must be nearby. Either alone is not enough — "go to the
    # next level, Python" would pass on proximity, and "we go, we ship" on
    # punctuation.
    left = before.rstrip()
    right = text[end : end + 12].lstrip()
    adjacent = (
        (left[-1:] in _LIST_DELIMS)
        or (right[:1] in _LIST_DELIMS)
        or bool(_LIST_WORDS.search(before))
        or right.lower().startswith(("and ", "or "))
    )
    if not adjacent:
        return False
    return any(
        w_start < end + _LIST_WINDOW and w_end > window_start
        for w_start, w_end in confident_spans
    )


def extract_skills_from_text(text: str | float | None) -> list[str]:
    """Canonical skills mentioned in free text. See ``extract_with_audit``."""
    return extract_with_audit(text)[0]


def extract_with_audit(
    text: str | float | None,
) -> tuple[list[str], list[tuple[str, str]]]:
    """Extract skills and also return the ambiguous mentions that were rejected.

    The rejection list is what makes the context guard testable: a guard whose
    misses nobody can see is a guard nobody can trust.
    """
    if not isinstance(text, str) or not text.strip():
        return [], []
    lowered = text.lower()

    # Unambiguous skills first — they are the evidence the guard leans on.
    spans: list[tuple[int, int, Skill]] = []
    confident_spans: list[tuple[int, int]] = []
    for skill in UNAMBIGUOUS_SKILLS:
        for m in _PATTERNS[skill.name].finditer(lowered):
            spans.append((m.start(), m.end(), skill))
            confident_spans.append((m.start(), m.end()))

    rejected: list[tuple[str, str]] = []
    for skill in SKILLS:
        if not skill.context_required:
            continue
        for m in _PATTERNS[skill.name].finditer(lowered):
            if _confirmed_by_context(lowered, m.start(), m.end(), skill,
                                     confident_spans):
                spans.append((m.start(), m.end(), skill))
            else:
                snippet = text[max(0, m.start() - 25) : m.end() + 25]
                rejected.append((skill.name, " ".join(snippet.split())))

    # Longest match wins: "SQL Server" must not also emit "SQL", and "C++"
    # must not also emit "C".
    spans.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    kept: list[tuple[int, int, Skill]] = []
    for span in spans:
        if any(o_s <= span[0] and span[1] <= o_e and (o_s, o_e) != (span[0], span[1])
               for o_s, o_e, _ in spans):
            continue
        kept.append(span)

    out: list[str] = []
    seen: set[str] = set()
    for _, _, skill in sorted(kept, key=lambda t: t[0]):
        if skill.name not in seen:
            seen.add(skill.name)
            out.append(skill.name)
    return out, rejected


# ---------------------------------------------------------------------------
# The employer-brand trap
# ---------------------------------------------------------------------------
#
# Every company names its own business unit in its job titles, and for a tech
# company that unit is usually also a product in our taxonomy. "Customer
# Engineer, Data Management Practice, Google Cloud" is a Google Cloud *org*
# label, not a statement that the candidate needs GCP — but a keyword matcher
# reads it as a GCP skill on 196 of Google's 848 postings, eight times what the
# description text actually supports.
#
# This is not a Google quirk; it hits every specialist, and hardest where the
# company IS the product: Snowflake's postings say "Snowflake", Databricks'
# say "Databricks". Left alone it manufactures exactly the differences Task 06
# is trying to measure — each company would appear to lead in its own product.
#
# So a title segment that names the employer's own org unit is dropped before
# extraction. Only the *title* path is filtered: in description prose the same
# word is a genuine requirement, and there the source and text paths keep it.

TITLE_ORG_SEGMENTS: dict[str, tuple[str, ...]] = {
    "Google": (
        "google cloud", "google cloud platform", "cloud", "google ads",
        "google workspace", "google play", "google maps", "google health",
        "google marketing platform", "youtube", "fitbit", "google fiber",
        "google customer solutions", "google technical services",
    ),
    # Placeholders for the other three specialists — each owner fills in their
    # own company's units before running the pipeline, and adds a test.
    "Microsoft": ("azure", "microsoft azure", "microsoft 365", "linkedin",
                  "github", "xbox"),
    "Amazon": ("aws", "amazon web services", "alexa", "prime video"),
    "Meta": ("reality labs", "instagram", "whatsapp", "facebook"),
    "NVIDIA": ("cuda", "omniverse", "geforce"),
    "Snowflake": ("snowflake",),
    "Databricks": ("databricks",),
    "OpenAI": ("chatgpt", "api platform"),
}

_SEGMENT_SPLIT = re.compile(r"[,/|]|\s[-–]\s")


def strip_org_segments(title: str | float | None,
                       company: str | float | None = None) -> str:
    """Remove title segments that name the employer's own business unit.

    Splits on the delimiters real titles use (``,`` ``/`` ``|`` `` - ``) and
    drops any segment that *starts with* one of the company's org names, so
    "Google Cloud" and "Google Cloud Data Management" both go. A segment merely
    mentioning a product ("Looker") is kept — that is a real role signal.
    """
    if not isinstance(title, str) or not title.strip():
        return ""
    units = TITLE_ORG_SEGMENTS.get(str(company).strip()) if company else None
    if not units:
        return title
    kept = [
        seg for seg in _SEGMENT_SPLIT.split(title)
        if seg.strip()
        and not any(seg.strip().lower().startswith(u) for u in units)
    ]
    return ", ".join(s.strip() for s in kept)


def extract_skills_from_title(title: str | float | None,
                              company: str | float | None = None) -> list[str]:
    """Skills named in a job title, minus the employer's own org labels.

    Titles are short and deliberate — "Software Engineer, ML" names the
    technology on purpose. This path works on 100% of rows, including the ones
    whose description text we never received.
    """
    return extract_skills_from_text(strip_org_segments(title, company))


# ---------------------------------------------------------------------------
# Combined extraction
# ---------------------------------------------------------------------------

PROVENANCE_ORDER = ("source", "text", "title")


def extract_skills(
    *,
    source_skills: Iterable[str] | None = None,
    title: str | float | None = None,
    description: str | float | None = None,
    company: str | float | None = None,
    unmapped: Counter | None = None,
) -> dict:
    """Run all three paths and merge them.

    Returns the per-path lists, the union, and a per-skill provenance map so a
    later task can weight a skill by how it was found.
    """
    from_source = normalize_skill_list(source_skills, unmapped=unmapped)
    from_text = extract_skills_from_text(description)
    from_title = extract_skills_from_title(title, company)

    provenance: dict[str, list[str]] = {}
    for path, names in (("source", from_source), ("text", from_text),
                        ("title", from_title)):
        for name in names:
            provenance.setdefault(name, []).append(path)

    final = sorted(provenance, key=lambda n: (CATEGORIES.index(CATEGORY_OF[n]), n))
    paths = sorted({p for ps in provenance.values() for p in ps},
                   key=PROVENANCE_ORDER.index)
    return {
        "skills_source": from_source,
        "skills_text": from_text,
        "skills_title": from_title,
        "skills_final": final,
        "skill_provenance": "+".join(paths) if paths else "none",
        "skill_provenance_map": {k: "+".join(v) for k, v in provenance.items()},
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

CATEGORY_COLUMNS = tuple(f"n_skills_{category_slug(c)}" for c in CATEGORIES)


def skill_features(skills: Iterable[str]) -> dict:
    """Per-posting numeric + tag features from a list of canonical skills."""
    names = list(skills)
    objs = [_BY_NAME[n.lower()] for n in names if n.lower() in _BY_NAME]

    per_category = Counter(o.category for o in objs)
    stack = [o.name for o in objs
             if o.category not in NON_STACK_CATEGORIES and not o.is_concept]
    methods = [o.name for o in objs if o.is_concept]

    feats: dict = {
        # Named `_final` because Task 03 already ships a `skill_count` holding
        # the collector's own count; keeping both makes the repair visible.
        "skill_count_final": len(objs),
        "tech_stack_tags": stack,
        "method_tags": methods,
        "n_tech_stack_skills": len(stack),
        "n_method_skills": len(methods),
        "skill_category_count": len(per_category),
        "skill_categories_final": sorted(
            per_category, key=lambda c: CATEGORIES.index(c)
        ),
    }
    for category in CATEGORIES:
        feats[f"n_skills_{category_slug(category)}"] = per_category.get(category, 0)

    # Ties break on CATEGORIES order, which puts the modelling categories first.
    feats["primary_skill_category"] = (
        min(per_category, key=lambda c: (-per_category[c], CATEGORIES.index(c)))
        if per_category
        else ""
    )
    feats["has_ml_skill"] = per_category.get("ML / AI", 0) > 0
    feats["has_cloud_skill"] = per_category.get("Cloud Platform", 0) > 0
    feats["has_programming_skill"] = per_category.get("Programming Language", 0) > 0
    feats["has_facilities_skill"] = per_category.get("Facilities / Data Centre", 0) > 0
    return feats


def as_skill_sequence(value) -> list | None:
    """Coerce a stored skill-list cell to a list, or None if there isn't one.

    Parquet round-trips a list column back as a ``numpy.ndarray``, so an
    ``isinstance(value, list)`` test silently drops every source skill and the
    pipeline still "succeeds" — it just reports 9% coverage instead of 79%.
    Missing cells arrive as ``None`` or as float NaN depending on the writer.
    """
    if value is None or isinstance(value, (str, bytes, float)):
        return None
    try:
        return list(value)
    except TypeError:
        return None


def build_posting_features(df: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    """Add Task 04 skill columns to a Task 03 cleaned table.

    Expects the Task 03 output columns ``skills_parsed`` (may be absent),
    ``job_title_clean`` and ``cleaned_description``.
    """
    unmapped: Counter = Counter()
    out = df.copy()

    source_col = (
        out["skills_parsed"]
        if "skills_parsed" in out.columns
        else pd.Series([[] for _ in range(len(out))], index=out.index)
    )
    title_col = out.get("job_title_clean", out.get("job_title"))
    desc_col = out.get("cleaned_description", pd.Series("", index=out.index))
    company_col = out.get("company_canonical", pd.Series("", index=out.index))

    extracted = [
        extract_skills(
            source_skills=as_skill_sequence(src),
            title=title,
            description=desc,
            company=company,
            unmapped=unmapped,
        )
        for src, title, desc, company in zip(source_col, title_col, desc_col,
                                             company_col)
    ]
    out = pd.concat([out, pd.DataFrame(extracted, index=out.index)], axis=1)

    feats = pd.DataFrame(
        [skill_features(row) for row in out.skills_final], index=out.index
    )
    out = pd.concat([out, feats], axis=1)

    out["has_any_skill"] = out.skill_count_final > 0
    # The measurable repair: rows the collector left empty that the title saved.
    out["skill_recovered_by_title"] = (
        out.skills_source.apply(len).eq(0) & out.skills_title.apply(len).gt(0)
    )
    return out, unmapped


# ---------------------------------------------------------------------------
# Aggregate tables (the "feature tables" deliverable)
# ---------------------------------------------------------------------------


def skills_long_table(df: pd.DataFrame, id_col: str = "job_id") -> pd.DataFrame:
    """One row per (posting, skill) — the extracted-skills dataset.

    Long form because every downstream task groups differently: Task 05 by
    month, Task 06 by company, Task 08 by skill pair. A wide matrix forces one
    of those choices on everyone.
    """
    carry = [c for c in ("company_canonical", "job_category", "job_function",
                         "seniority_level", "posting_month", "posting_quarter",
                         "location_country") if c in df.columns]
    rows = []
    for _, row in df.iterrows():
        prov = row.get("skill_provenance_map") or {}
        for name in row.skills_final:
            skill = _BY_NAME[name.lower()]
            rows.append(
                {
                    id_col: row[id_col],
                    "skill": name,
                    "skill_key": skill.key,
                    "skill_category": skill.category,
                    "is_concept": skill.is_concept,
                    "in_tech_stack": skill.category not in NON_STACK_CATEGORIES
                    and not skill.is_concept,
                    "provenance": prov.get(name, ""),
                    **{c: row[c] for c in carry},
                }
            )
    columns = [id_col, "skill", "skill_key", "skill_category", "is_concept",
               "in_tech_stack", "provenance", *carry]
    return pd.DataFrame(rows, columns=columns)


def skill_frequency_table(long: pd.DataFrame, n_skilled: int,
                          n_total: int) -> pd.DataFrame:
    """Skill counts with **both** denominators, because they disagree.

    ``share_of_skilled`` is the honest one: a posting with no extracted skills
    is a posting we know nothing about, not a posting without Python.
    """
    if long.empty:
        return pd.DataFrame(
            columns=["skill", "skill_category", "n_postings", "share_of_skilled",
                     "share_of_all", "rank"]
        )
    agg = (
        long.groupby(["skill", "skill_category"], as_index=False)
        .agg(n_postings=("job_id", "nunique"))
        .sort_values("n_postings", ascending=False)
        .reset_index(drop=True)
    )
    agg["share_of_skilled"] = (agg.n_postings / max(n_skilled, 1)).round(4)
    agg["share_of_all"] = (agg.n_postings / max(n_total, 1)).round(4)
    agg["rank"] = agg.n_postings.rank(method="min", ascending=False).astype(int)
    return agg


def skill_by_month_table(long: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Monthly skill frequency, normalised by that month's skill coverage.

    Coverage swings by month, so an un-normalised monthly share measures how
    many postings the collector managed to extract skills from as much as it
    measures demand. Task 05 must use ``share_of_skilled``.
    """
    if long.empty:
        return pd.DataFrame(
            columns=["posting_month", "skill", "skill_category", "n_postings",
                     "postings_with_skills", "share_of_skilled"]
        )
    agg = (
        long.groupby(["posting_month", "skill", "skill_category"], as_index=False)
        .agg(n_postings=("job_id", "nunique"))
    )
    agg = agg.merge(
        coverage[["posting_month", "postings_with_skills", "postings"]],
        on="posting_month", how="left",
    )
    agg["share_of_skilled"] = (
        agg.n_postings / agg.postings_with_skills.replace(0, float("nan"))
    ).astype(float).round(4)
    agg["share_of_all"] = (
        agg.n_postings / agg.postings.replace(0, float("nan"))
    ).astype(float).round(4)
    return agg.sort_values(["posting_month", "n_postings"],
                           ascending=[True, False]).reset_index(drop=True)


def coverage_table(df: pd.DataFrame, by: str = "posting_month") -> pd.DataFrame:
    """How many postings we actually know the skills of, per group.

    This is a first-class output, not diagnostics: every share in Task 05-08
    needs it as a denominator.
    """
    agg = (
        df.groupby(by, as_index=False)
        .agg(postings=("job_id", "size"),
             postings_with_skills=("has_any_skill", "sum"),
             mean_skills_when_present=("skill_count_final",
                                       lambda s: s[s > 0].mean()))
    )
    agg["coverage"] = (agg.postings_with_skills / agg.postings).round(4)
    agg["mean_skills_when_present"] = agg.mean_skills_when_present.round(2)
    return agg.sort_values(by).reset_index(drop=True)


def skill_cooccurrence_table(long: pd.DataFrame, min_pairs: int = 5,
                             id_col: str = "job_id") -> pd.DataFrame:
    """Skill pairs with Jaccard overlap — the input Task 08 needs.

    Jaccard rather than raw counts because raw counts just re-rank the popular
    skills: Python co-occurs with everything.
    """
    if long.empty:
        return pd.DataFrame(columns=["skill_a", "skill_b", "n_both", "n_a", "n_b",
                                     "jaccard"])
    per_posting = long.groupby(id_col).skill.apply(lambda s: sorted(set(s)))
    totals = Counter()
    pairs: Counter = Counter()
    for names in per_posting:
        totals.update(names)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                pairs[(a, b)] += 1
    rows = [
        {
            "skill_a": a,
            "skill_b": b,
            "n_both": n,
            "n_a": totals[a],
            "n_b": totals[b],
            "jaccard": round(n / (totals[a] + totals[b] - n), 4),
        }
        for (a, b), n in pairs.items()
        if n >= min_pairs
    ]
    return (
        pd.DataFrame(rows, columns=["skill_a", "skill_b", "n_both", "n_a", "n_b",
                                    "jaccard"])
        .sort_values(["jaccard", "n_both"], ascending=False)
        .reset_index(drop=True)
    )


def skill_trend_table(long: pd.DataFrame, df: pd.DataFrame, min_postings: int = 10,
                      threshold: float = 0.25) -> pd.DataFrame:
    """First-half vs second-half share, and the ``emerging_skill_flag``.

    Deliberately coarse. The Google dataset is a single calendar year from a
    single source, so a month-on-month slope would read sampling noise as a
    trend. Halves with a minimum support of ``min_postings`` is the strongest
    claim this data can carry, and every label is within-year by construction.
    """
    empty = pd.DataFrame(columns=["skill", "skill_category", "n_h1", "n_h2",
                                  "share_h1", "share_h2", "delta",
                                  "relative_change", "emerging_skill_flag"])
    if long.empty or "posting_month" not in df.columns:
        return empty

    def half(month: str) -> str:
        try:
            return "h1" if int(str(month)[5:7]) <= 6 else "h2"
        except (ValueError, TypeError):
            return ""

    long = long.assign(half=long.posting_month.map(half))
    df = df.assign(half=df.posting_month.map(half))
    denom = {
        h: int(g.has_any_skill.sum())
        for h, g in df[df.half.isin({"h1", "h2"})].groupby("half")
    }
    if not denom.get("h1") or not denom.get("h2"):
        return empty

    counts = (
        long[long.half.isin({"h1", "h2"})]
        .groupby(["skill", "skill_category", "half"], as_index=False)
        .agg(n=("job_id", "nunique"))
        .pivot_table(index=["skill", "skill_category"], columns="half", values="n",
                     fill_value=0)
        .reset_index()
    )
    counts.columns.name = None
    for h in ("h1", "h2"):
        if h not in counts.columns:
            counts[h] = 0
    counts = counts.rename(columns={"h1": "n_h1", "h2": "n_h2"})
    counts[["n_h1", "n_h2"]] = counts[["n_h1", "n_h2"]].fillna(0).astype(int)
    counts["share_h1"] = (counts.n_h1 / denom["h1"]).round(4)
    counts["share_h2"] = (counts.n_h2 / denom["h2"]).round(4)
    counts["delta"] = (counts.share_h2 - counts.share_h1).round(4)
    counts["relative_change"] = (
        counts.delta / counts.share_h1.replace(0, float("nan"))
    ).astype(float).round(3)

    total = counts.n_h1 + counts.n_h2
    supported = total >= min_postings
    counts["emerging_skill_flag"] = "insufficient_support"
    counts.loc[supported, "emerging_skill_flag"] = "stable"
    counts.loc[supported & (counts.relative_change >= threshold),
               "emerging_skill_flag"] = "emerging"
    counts.loc[supported & (counts.relative_change <= -threshold),
               "emerging_skill_flag"] = "declining"
    return counts.sort_values("delta", ascending=False).reset_index(drop=True)


def skill_matrix(long: pd.DataFrame, min_postings: int = 5,
                 id_col: str = "job_id") -> pd.DataFrame:
    """Binary posting x skill matrix — the ``skill_frequency_vector`` field.

    Skills below ``min_postings`` are dropped: a column that is 1 for a single
    posting is an identifier, not a feature, and it would dominate any cosine
    similarity in Task 08.
    """
    if long.empty:
        return pd.DataFrame({id_col: []})
    counts = long.groupby("skill")[id_col].nunique()
    keep = sorted(counts[counts >= min_postings].index)
    if not keep:
        return pd.DataFrame({id_col: sorted(long[id_col].unique())})
    wide = (
        long[long.skill.isin(keep)]
        .assign(v=1)
        .pivot_table(index=id_col, columns="skill", values="v", fill_value=0)
        .reindex(columns=keep, fill_value=0)
        .astype(int)
        .reset_index()
    )
    wide.columns = [id_col] + [f"skill_{_BY_NAME[c.lower()].key}" for c in keep]
    return wide
