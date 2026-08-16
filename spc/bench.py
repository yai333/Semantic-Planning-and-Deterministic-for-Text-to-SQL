from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TTL = ROOT / "benchmark" / "acme" / "acme-benchmark.ttl"
DB = ROOT / "database" / "acme" / "acme_N.sqlite"
DDL = ROOT / "database" / "acme" / "ACME_small.ddl"

FLOOR_DDL = ROOT / "database" / "acme" / "ACME_floor.ddl"
FLOOR_DDL_VALUES = ROOT / "database" / "acme" / "ACME_floor_values.ddl"

ONTOLOGY = ROOT / "ontology" / "acme.semantic.yaml"
MAPPING = ROOT / "ontology" / "acme.mapping.yaml"
SEMANTICS = ONTOLOGY
RESULTS = ROOT / "results"

from evaluation.accuracy import PRICE_IN, PRICE_OUT, canon, correct, recast_gold  # noqa: E402

from spc import trace  # noqa: E402

QA = "http://models.data.world/benchmarks/QandA#"
DWT = "https://templates.data.world/"
DCT = "http://purl.org/dc/terms/"

GOLD_DIALECT_FAILURES = {
    "query-578baedd": "DATE_DIFF() is not a SQLite function (gold is BigQuery dialect)",
}

@dataclass(frozen=True)
class Question:
    """One benchmark item: a prompt and the gold SQL that defines its answer."""

    qid: str
    inquiry: str
    prompt: str
    gold_sql: str
    gold_executable: bool
    gold_error: str | None = None
    gold_rows: int | None = None
    alternates: tuple[str, ...] = ()
    normalizations: tuple[str, ...] = ()

    feasible: bool = True
    feasible_strict: bool = True
    feasible_moderate: bool = True
    mf_subject: str | None = None
    mf_reasons: tuple[str, ...] = ()
    mf_strict_reasons: tuple[str, ...] = ()
    requirements: tuple[tuple[str, int], ...] = ()

    mf_join_only: tuple[str, ...] = ()

    mf_notes: tuple[str, ...] = ()

    @property
    def scope(self) -> str:
        """dbt's split, computed rather than declared. Follows the HEADLINE."""
        return "in_scope" if self.feasible else "too_many_hops"

MF_HOP_LIMIT = 2

@dataclass(frozen=True)
class Regime:
    """What a dbt model is allowed to contain. THE symmetry knob.

    Every flag below applies to a junction table and to a subtype/marker table
    IDENTICALLY, which is objection 1's fix: `absorb_auxiliary` governs both, and
    there is no way to answer the question differently for the two.
    """

    name: str
    description: str

    absorb_auxiliary: bool

    collapse_kind_instances: bool

    absorb_many_to_one: bool

    promote_functional: bool
    hop_limit: int = MF_HOP_LIMIT

    multi_fact: bool = True
    max_subjects: int = 3

STRICT = Regime(
    name="strict",
    description=(
        "one semantic model per physical table. A junction costs 2 joins and a "
        "subtype/marker semijoin costs 2 joins -- symmetric, and the harshest "
        "symmetric reading. Reported as an UPPER BOUND on infeasibility, never "
        "as the headline."
    ),
    absorb_auxiliary=False,
    collapse_kind_instances=False,
    absorb_many_to_one=False,
    promote_functional=False,
)

MODERATE = Regime(
    name="moderate",
    description=(
        "reusable dbt models, declared cardinality only. Junction and marker "
        "tables are absorbed; amount kinds become one normalised model read by "
        "filtered measures; many-to-one parents may be denormalised into a "
        "model. Nothing is inferred from the data."
    ),
    absorb_auxiliary=True,
    collapse_kind_instances=True,
    absorb_many_to_one=True,
    promote_functional=False,
)

PERMISSIVE = Regime(
    name="permissive",
    description=(
        "MODERATE, plus entity uniqueness declared from the data the engineer "
        "has: where the fixture shows a declared one-to-many to be functional, "
        "the engineer declares a unique entity and MetricFlow permits the join. "
        "THE HEADLINE -- this is what a competent dbt engineer would build, and "
        "GOALS G5 permits exactly it (bridge, normalised and pre-aggregated "
        "models; question-specific answer marts still forbidden)."
    ),
    absorb_auxiliary=True,
    collapse_kind_instances=True,
    absorb_many_to_one=True,
    promote_functional=True,
)

REGIMES: tuple[Regime, ...] = (STRICT, MODERATE, PERMISSIVE)
HEADLINE_REGIME = PERMISSIVE

def _key_columns(concept: Any) -> list[str]:
    key = concept.key
    return [key] if isinstance(key, str) else list(key)

def _cardinality_sql(onto: Any, edge: Any, forward: bool) -> str:
    """`SELECT <source key>, COUNT(DISTINCT <target key>) ... GROUP BY <source key>`.

    Built from the edge's own physical backing, so it profiles exactly the join
    the compiler would emit -- junction, role predicate, subtype semijoin and
    role-object `backed_where` included.
    """
    src = onto.concept(edge.origin(forward=forward))
    tgt = onto.concept(edge.endpoint(forward=forward))
    frm = [f'"{src.table}" S']
    where = [f"S.\"{c}\" = '{v}'" for c, v in src.backed_where]
    if edge.via_table:
        if forward:
            first = [(s, j) for s, j in edge.via_from_join]
            second = [(j, t) for j, t in edge.via_to_join]
        else:
            first = [(t, j) for j, t in edge.via_to_join]
            second = [(j, s) for s, j in edge.via_from_join]
        frm.append(f'JOIN "{edge.via_table}" J ON '
                   + " AND ".join(f'S."{a}" = J."{b}"' for a, b in first))
        frm.append(f'JOIN "{tgt.table}" T ON '
                   + " AND ".join(f'J."{a}" = T."{b}"' for a, b in second))
        where += [f"J.\"{c}\" = '{v}'" for c, v in edge.role_predicate]
    else:
        pairs = ([(s, t) for s, t in edge.join] if forward
                 else [(t, s) for s, t in edge.join])
        frm.append(f'JOIN "{tgt.table}" T ON '
                   + " AND ".join(f'S."{a}" = T."{b}"' for a, b in pairs))
    if edge.restrict_table:
        side = "T" if forward else "S"
        frm.append(f'JOIN "{edge.restrict_table}" R ON '
                   + " AND ".join(f'{side}."{d}" = R."{r}"'
                                  for d, r in edge.restrict_columns))
    where += [f"T.\"{c}\" = '{v}'" for c, v in tgt.backed_where]
    src_key = ", ".join(f'S."{c}"' for c in _key_columns(src))
    tgt_cols = _key_columns(tgt)
    tgt_key = (" || '#' || ".join(f'T."{c}"' for c in tgt_cols) if len(tgt_cols) > 1
               else f'T."{tgt_cols[0]}"')
    return (f"SELECT {src_key}, COUNT(DISTINCT {tgt_key}) AS n FROM " + " ".join(frm)
            + (" WHERE " + " AND ".join(where) if where else "")
            + f" GROUP BY {src_key}")

_FUNCTIONAL_CACHE: dict[tuple[str, str], frozenset[tuple[str, bool]]] = {}

def functional_edges(onto: Any, db: str | Path = DB) -> frozenset[tuple[str, bool]]:
    """`(edge name, forward)` pairs the FIXTURE shows to be functional.

    A declared one-to-many that never actually produces two rows per source row
    is a relationship a dbt engineer would declare as a unique entity, turning
    the join into the Foreign->Primary shape MetricFlow permits. Only the
    PERMISSIVE regime uses this, and every promotion is printed by name.

    ZERO EVIDENCE IS NOT EVIDENCE. An edge with no rows in the fixture
    (UNDERWROTE, POLICY_LOCATED_AT) is never promoted.

    THE WEAKEST LINK IN THIS FILE. The fixture is 2 claims, 2 policies and 12
    policy amounts; "at most one per source row" over two rows is thin. It is
    used anyway because the alternative -- refusing a join the data says is
    functional -- is the reading that flatters us, and GOALS G5's blinded
    MetricFlow build is what settles it for real.
    """
    key = (str(db), str(ONTOLOGY))
    hit = _FUNCTIONAL_CACHE.get(key)
    if hit is not None:
        return hit
    found: set[tuple[str, bool]] = set()
    try:
        con = sqlite3.connect(str(db))
        for edge in onto.edges:
            for forward in (True, False):
                if edge.fan_out_in(forward=forward) == "none":
                    continue
                try:
                    rows = con.execute(_cardinality_sql(onto, edge, forward)).fetchall()
                except Exception:              # noqa: BLE001
                    continue
                if rows and max(r[-1] for r in rows) <= 1:
                    found.add((edge.name, forward))
        con.close()
    except Exception:                          # noqa: BLE001
        found = set()
    result = frozenset(found)
    _FUNCTIONAL_CACHE[key] = result
    return result

@dataclass(frozen=True)
class Requirements:
    """What a query needs, as concepts rather than as table aliases."""

    counts: Counter = field(default_factory=Counter)

    measures: frozenset[str] = frozenset()

    join_only: frozenset[str] = frozenset()
    notes: tuple[str, ...] = ()

    def size(self) -> tuple[int, int]:
        """Sort key for "the minimum representation": instances, then concepts."""
        return (sum(self.counts.values()), len(self.counts))

def _sqlite_schema(db: str | Path = DB) -> dict[str, dict[str, str]]:
    """Column types per table, for sqlglot's qualifier. Cached per path."""
    cached = _SCHEMA_CACHE.get(str(db))
    if cached is None:
        cached = {}
        try:
            con = sqlite3.connect(str(db))
            for (table,) in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"):
                cached[table] = {r[1]: (r[2] or "TEXT")
                                 for r in con.execute(f'PRAGMA table_info("{table}")')}
            con.close()
        except Exception:                      # noqa: BLE001
            cached = {}
        _SCHEMA_CACHE[str(db)] = cached
    return cached

_SCHEMA_CACHE: dict[str, dict[str, dict[str, str]]] = {}

def concept_requirements(sql: str, onto: Any, *, db: str | Path = DB,
                         join_only_required: bool = False) -> Requirements:
    """Which concepts, at how many INSTANCES, the QUESTION needs.

    Read off the gold SQL rather than the prompt, because gold defines the answer
    and a column list is mechanical where a reading of English is not. Three
    things are dropped, each for a stated reason:

    * JUNCTION AND SUBTYPE-MARKER TABLES are edge machinery, not concepts. The
      role code on an `Agreement_Party_Role` alias still selects WHICH role
      concept (PolicyHolder / Agent / Underwriter) that alias is.

    * ALIASES REFERENCED ONLY IN `JOIN ... ON` are path intermediaries. Objection
      2's fix: `Policy_Coverage_Detail` in "how many claims by policy number" is
      how that gold gets from a claim to a policy, not something the question
      asks for; MetricFlow would route through it (or a model would absorb it)
      without it ever being a requirement. Pass `join_only_required=True` for the
      old, table-list behaviour.

    * DEAD JOINS fall out of the same rule for free. `query-a810b049`'s gold
      joins Catastrophe, Policy_Amount and three extra Claim_Amount aliases whose
      columns are COMMENTED OUT of its SELECT; none of them survives.

    The instance COUNT still matters: four aliases of Claim_Amount whose columns
    are all projected is four traversals of a one-to-many edge, and whether that
    is expressible is exactly what the regime decides.

    NOTES record what was dropped: `marker:<table>` per junction/subtype table,
    `join_only:<concept>` per intermediary, and `ambiguous_role:<alias>-><concept>`
    wherever a role object had to be picked with no literal role predicate to
    read -- a silently wrong role is this project's core error, so it is loud.
    """
    import sqlglot
    from sqlglot import exp

    by_table: dict[str, list[str]] = {}
    for name, concept in onto.concepts.items():
        by_table.setdefault(concept.table.lower(), []).append(name)

    tree = sqlglot.parse_one(sql, read="sqlite")
    schema = _sqlite_schema(db)
    if schema:

        try:
            from sqlglot.optimizer.qualify import qualify

            tree = qualify(tree, schema=schema, dialect="sqlite",
                           qualify_columns=True, validate_qualify_columns=False,
                           identify=False, infer_schema=True)
        except Exception:                      # noqa: BLE001
            tree = sqlglot.parse_one(sql, read="sqlite")

    alias_table = {(t.alias or t.name).lower(): t.name.lower()
                   for t in tree.find_all(exp.Table)}

    def _in_join_condition(node: Any) -> bool:
        current, child = node.parent, node
        while current is not None:
            if isinstance(current, exp.Join) and child is current.args.get("on"):
                return True
            current, child = current.parent, current
        return False

    def _in_aggregate(node: Any) -> bool:
        current = node.parent
        while current is not None:
            if isinstance(current, exp.AggFunc):
                return True
            current = current.parent
        return False

    named: set[str] = set()
    measured: set[str] = set()
    referenced: set[str] = set()
    for column in tree.find_all(exp.Column):
        alias = (column.table or "").lower()
        if not alias:
            continue
        referenced.add(alias)
        if _in_join_condition(column) and not join_only_required:
            continue
        named.add(alias)
        if _in_aggregate(column):
            measured.add(alias)

    named |= set(alias_table) - referenced

    roles: dict[str, str] = {}
    for eq in tree.find_all(exp.EQ):
        left, right = eq.this, eq.expression
        if (isinstance(left, exp.Column) and left.name.lower() == "party_role_code"
                and isinstance(right, exp.Literal)):
            roles[(left.table or "").lower()] = str(right.this)

    counts: Counter = Counter()
    measures: set[str] = set()
    join_only: set[str] = set()
    notes: set[str] = set()
    for alias, table in sorted(alias_table.items()):
        candidates = by_table.get(table)
        if not candidates:
            notes.add(f"marker:{table}")
            continue
        if len(candidates) == 1:
            concept = candidates[0]
        else:

            code = roles.get(alias)
            if code is None and len(set(roles.values())) == 1:
                code = next(iter(roles.values()))
            concept = next((c for c in sorted(candidates)
                            if onto.concept(c).role_code == code), None)
            if concept is None:

                concept = sorted(candidates)[0]
                notes.add(f"ambiguous_role:{alias}->{concept}")
        if alias not in named:
            join_only.add(concept)
            continue
        counts[concept] += 1
        if alias in measured:
            measures.add(concept)
    join_only -= set(counts)
    notes |= {f"join_only:{c}" for c in join_only}
    return Requirements(counts=counts, measures=frozenset(measures),
                        join_only=frozenset(join_only), notes=tuple(sorted(notes)))

@dataclass(frozen=True)
class MFVerdict:
    feasible: bool
    subject: str | None
    reasons: tuple[str, ...]
    regime: str = STRICT.name

def _effective_fan_out(edge: Any, forward: bool, regime: Regime,
                       functional: frozenset[tuple[str, bool]]) -> str:
    declared = edge.fan_out_in(forward=forward)
    if declared == "none":
        return "none"
    if regime.promote_functional and (edge.name, forward) in functional:
        return "none"
    return declared

def _step_cost(edge: Any, forward: bool, regime: Regime,
               functional: frozenset[tuple[str, bool]]) -> int:
    """Joins BETWEEN SEMANTIC MODELS that one governed step costs.

    THE SYMMETRY, stated once. A junction table (`via:`) and a subtype/marker
    table (`restrict:`) are the same kind of object: an auxiliary physical table
    that qualifies the edge. `absorb_auxiliary` answers for both at once, so the
    old asymmetry -- junction 2, marker 0 -- is not expressible here.
    """
    if regime.absorb_many_to_one and _effective_fan_out(edge, forward, regime,
                                                        functional) == "none":
        return 0
    cost = 1
    if not regime.absorb_auxiliary:
        cost += 1 if edge.via_table else 0
        cost += 1 if edge.restrict_table else 0
    return cost

def _cost_terms(edge: Any, forward: bool, regime: Regime,
                functional: frozenset[tuple[str, bool]]) -> str:
    """The arithmetic, spelled out, so a reason can be checked rather than read."""
    cost = _step_cost(edge, forward, regime, functional)
    if cost == 0:
        return f"{edge.name}[absorbed: many-to-one]=0"
    parts = ["base 1"]
    if not regime.absorb_auxiliary and edge.via_table:
        parts.append(f"junction {edge.via_table} 1")
    if not regime.absorb_auxiliary and edge.restrict_table:
        parts.append(f"marker {edge.restrict_table} 1")
    return f"{edge.name}[{' + '.join(parts)}]={cost}"

def metricflow_verdict(requirements: Requirements | Counter | dict, onto: Any,
                       graph: Any, *, regime: Regime = STRICT,
                       functional: frozenset[tuple[str, bool]] | None = None,
                       hop_limit: int | None = None) -> MFVerdict:
    """Feasible under MetricFlow's join rules IN THIS REGIME? With the reason if not.

    RETIRED AS A MEASUREMENT (see the module docstring and `GOALS.md`). This is a
    SIMULATION of a system we do not run, on a substrate we authored, and it was
    argued down once already -- treat its output as an annotation describing why
    a question looked hard, never as a result and never as a comparison. It is
    still exercised by `spc/tests/test_bench.py` so it cannot rot silently.

    The search: choose an admissible subject set (one model, or -- since
    MetricFlow supports multi-fact metrics -- the set of models the measures live
    on), then require every other concept the question names to be reachable from
    every subject within the hop ceiling without crossing a fan-out edge.

    `regime` decides what a model may contain and therefore what a hop costs;
    `functional` is `functional_edges()` and is consulted only when the regime
    says so. Both are explicit so a verdict can be reproduced from its inputs.
    """
    if not isinstance(requirements, Requirements):
        requirements = Requirements(counts=Counter(requirements))
    functional = functional if functional is not None else frozenset()
    limit = regime.hop_limit if hop_limit is None else hop_limit

    counts = Counter(requirements.counts)
    collapsed: list[str] = []
    if regime.collapse_kind_instances:
        for concept, n in list(counts.items()):
            if n <= 1:
                continue
            kinds = sorted({e.name for e in onto.edges
                            if e.restrict_table and concept in (e.source, e.target)})
            if n <= len(kinds):

                counts[concept] = 1
                collapsed.append(f"{concept}x{n} -> one normalised model "
                                 f"(kinds: {', '.join(kinds)})")
    concepts = sorted(counts)
    if not concepts:
        return MFVerdict(True, None, ("no concept resolved",), regime.name)

    def _paths(source: str, target: str):
        return graph.paths(source, target, max_hops=4)

    def _path_cost(path: Any) -> int:
        return sum(_step_cost(onto.edge(s.edge), s.forward, regime, functional)
                   for s in path.steps)

    def _fans(path: Any) -> list[str]:
        return [f"{s.edge}({_effective_fan_out(onto.edge(s.edge), s.forward, regime, functional)})"
                for s in path.steps
                if _effective_fan_out(onto.edge(s.edge), s.forward, regime,
                                      functional) != "none"]

    def allowed(source: str, target: str) -> bool:
        if source == target:
            return True
        for path in _paths(source, target):
            if _path_cost(path) > limit:
                continue
            if _fans(path):
                continue
            return True
        return False

    def _arithmetic(path: Any) -> str:
        terms = [_cost_terms(onto.edge(s.edge), s.forward, regime, functional)
                 for s in path.steps]
        return f"{' + '.join(terms)} = {_path_cost(path)}"

    def blocked(source: str, target: str) -> str:
        """WHICH rule refuses, with the arithmetic or the edge that does it.

        A fan-out-free route that is merely too long is a HOPS refusal and gets
        the join count spelled out. A pair with no fan-out-free route at all is a
        FANOUT refusal and gets the offending edge named, whatever the hop count
        says -- raising the ceiling would not rescue it.
        """
        paths = _paths(source, target)
        if not paths:
            return f"NO_PATH: no governed path {source} -> {target}"
        clean = [p for p in paths if not _fans(p)]
        if clean:
            cheapest = min(clean, key=_path_cost)
            return (f"HOPS: {source} -> {target}; the cheapest fan-out-free route is "
                    f"{_path_cost(cheapest)} semantic-model joins "
                    f"({_arithmetic(cheapest)}); MetricFlow's ceiling is {limit}")
        cheapest = min(paths, key=_path_cost)
        fan = _fans(cheapest)
        return (f"FANOUT: {source} -> {target}; EVERY governed route fans out. The "
                f"cheapest crosses {', '.join(fan)} -- one-to-many in the direction "
                f"travelled, which MetricFlow refuses ({_arithmetic(cheapest)}, "
                f"ceiling {limit}). This regime does not let a model absorb it.")

    def aliasing(concept: str, n: int) -> str:
        kinds = sorted({e.name for e in onto.edges
                        if e.restrict_table and concept in (e.source, e.target)})
        incoming = [e for e in onto.edges if concept in (e.source, e.target)]
        fan = sorted({e.name for e in incoming
                      if (e.target == concept and e.fan_out_in(forward=True) != "none")
                      or (e.source == concept and e.fan_out_in(forward=False) != "none")})
        if len(fan) == len(incoming) and fan:
            evidence = f"every edge into it is one-to-many ({', '.join(fan)})"
        elif fan:
            evidence = f"the edges into it that fan out are {', '.join(fan)}"
        else:
            evidence = "no edge into it fans out, so this rests on aliasing alone"
        tail = ("" if not kinds else
                f"; {len(kinds)} kind-restricted edges exist ({', '.join(kinds)}), so a "
                f"normalised model would collapse this -- which this regime does not allow")
        return (f"FANOUT_ALIAS: {concept} is needed at {n} instances in one query; "
                f"{evidence} and one semantic model cannot be aliased per instance{tail}")

    def evaluate(subjects: tuple[str, ...]) -> list[str]:
        """Empty list = this subject set answers the question."""
        problems: list[str] = []
        for concept in concepts:
            n = counts[concept]
            if n > 1:
                problems.append(aliasing(concept, n))
                continue
            if concept in subjects:
                continue
            for subject in subjects:
                if not allowed(subject, concept):
                    problems.append(blocked(subject, concept))
                    break
        return problems

    scored: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
    for subject in concepts:
        problems = evaluate((subject,))
        if not problems:
            note = ((f"subject {subject}",) if len(concepts) > 1
                    else ("single semantic model",))
            return MFVerdict(True, subject,
                             note + tuple(f"COLLAPSED: {c}" for c in collapsed),
                             regime.name)
        scored.append((len(problems), (subject,), tuple(problems)))

    if regime.multi_fact:
        pool = sorted(requirements.measures & set(concepts))
        for size in range(2, min(regime.max_subjects, len(pool)) + 1):
            for subjects in itertools.combinations(pool, size):
                if len(concepts) > len(subjects) and not evaluate(subjects):
                    return MFVerdict(
                        True, subjects[0],
                        (f"multi-fact: subjects {' + '.join(subjects)} joined on "
                         f"{', '.join(c for c in concepts if c not in subjects)}",)
                        + tuple(f"COLLAPSED: {c}" for c in collapsed),
                        regime.name)

    scored.sort()
    _, subjects, problems = scored[0]
    return MFVerdict(False, subjects[0], problems[:3], regime.name)

def _classifier() -> tuple[Any, Any] | None:
    """Ontology + path graph, or None if either fails to load. Never fatal: the
    harness must still parse questions when the ontology is mid-edit."""
    try:
        from spc.graph import PathGraph
        from spc.ontology import load_ontology

        onto = load_ontology(ONTOLOGY, MAPPING)
        return onto, PathGraph(onto)
    except Exception:                          # noqa: BLE001
        return None

def _normalize_gold(sql: str) -> tuple[str, tuple[str, ...]]:
    """Dialect normalisation ONLY, each one recorded.

    The single case present: one gold uses MySQL `#` line comments, which SQLite
    rejects as a token. Converting `#` to `--` is a lexical dialect fix that
    cannot change which rows the query returns. Nothing here may change
    semantics -- gold is never edited to suit an engine (CLAUDE.md hard rule 3).
    """
    applied: list[str] = []
    if re.search(r"(?m)^\s*#", sql):
        sql = re.sub(r"(?m)^(\s*)#", r"\1--", sql)
        applied.append("hash_comment_to_sql_comment")
    return sql, tuple(applied)

def _classify(q: Question, onto: Any, graph: Any,
              functional: frozenset[tuple[str, bool]], golds: Sequence[str],
              db: str | Path = DB) -> Question:
    """One question's verdict in every regime, over every equivalent gold.

    Feasible in a regime iff SOME equivalent representation is feasible there --
    the fix for "feasibility was a property of which gold we parsed". The
    requirements and the blame reported are the minimum representation's.
    """
    reqs: list[Requirements] = []
    for sql in golds:
        try:
            reqs.append(concept_requirements(sql, onto, db=db))
        except Exception as exc:               # noqa: BLE001
            reqs.append(Requirements(notes=(f"unparsed: {str(exc)[:80]}",)))
    reqs.sort(key=lambda r: r.size())
    canonical = reqs[0]

    verdicts: dict[str, MFVerdict] = {}
    for regime in REGIMES:
        best: MFVerdict | None = None
        for req in reqs:
            try:
                verdict = metricflow_verdict(req, onto, graph, regime=regime,
                                             functional=functional)
            except Exception as exc:           # noqa: BLE001
                verdict = MFVerdict(True, None, (f"unclassified: {exc}",), regime.name)
            if verdict.feasible:
                best = verdict
                break
            if best is None or len(verdict.reasons) < len(best.reasons):
                best = verdict
        verdicts[regime.name] = best or MFVerdict(True, None, ("no gold",), regime.name)

    head = verdicts[HEADLINE_REGIME.name]
    return replace(
        q,
        feasible=head.feasible,
        feasible_strict=verdicts[STRICT.name].feasible,
        feasible_moderate=verdicts[MODERATE.name].feasible,
        mf_subject=head.subject,
        mf_reasons=head.reasons,
        mf_strict_reasons=verdicts[STRICT.name].reasons,
        requirements=tuple(sorted(canonical.counts.items())),
        mf_join_only=tuple(sorted(canonical.join_only)),
        mf_notes=tuple(n for n in canonical.notes
                       if not n.startswith(("marker:", "join_only:"))),
    )

def load_questions(
    ttl: str | Path = TTL,
    db: str | Path = DB,
    *,
    normalize: bool = True,
    executable_only: bool = False,
    classify: bool = True,
) -> list[Question]:
    """Every `QandA:Inquiry` in the Turtle, with its SQL gold, execution-checked.

    An inquiry may carry several `QandA:expects` queries -- SPARQL as well as
    SQL, and occasionally two SQL variants. We take SQL only, and among SQL we
    take the first that EXECUTES against the fixture, recording the others as
    `alternates`. Choosing by executability rather than by document order is a
    stated rule, not a per-question judgement.

    CLASSIFICATION IS INVARIANT UNDER EQUIVALENT GOLD (objection 2). The chosen
    gold decides what is SCORED, but it does not decide feasibility: requirements
    are computed for EVERY gold on the inquiry that executes and returns the same
    rows, and the classifier is run over all of them. A question is feasible in a
    regime if ANY equivalent representation is, and the requirements reported are
    the MINIMUM one -- fewest instances, then fewest concepts, then qid. Both
    tie-breaks favour MetricFlow, which is the standing rule for this file.
    """
    from rdflib import Graph, Namespace, RDF

    g = Graph()
    g.parse(str(ttl), format="turtle")
    qa, dwt, dct = Namespace(QA), Namespace(DWT), Namespace(DCT)
    con = sqlite3.connect(str(db))

    out: list[Question] = []
    equivalents: dict[str, list[str]] = {}
    for subject in g.subjects(RDF.type, qa.Inquiry):
        prompt = str(g.value(subject, qa.prompt) or "").strip()
        golds: list[tuple[str, str, tuple[str, ...]]] = []
        for obj in g.objects(subject, qa.expects):
            if (obj, RDF.type, dwt.SqlQuery) not in g:
                continue
            text = str(g.value(obj, qa.queryText) or "")
            norms: tuple[str, ...] = ()
            if normalize:
                text, norms = _normalize_gold(text)
            golds.append((str(obj).rsplit("/", 1)[-1], text, norms))
        golds.sort()

        chosen: tuple[str, str, tuple[str, ...]] | None = None
        rows: int | None = None
        error: str | None = None
        results: dict[str, Any] = {}
        for qid, text, norms in golds:
            try:
                fetched = con.execute(text).fetchall()
                results[qid] = canon(fetched)
                if chosen is None or error is not None:
                    rows, chosen, error = len(fetched), (qid, text, norms), None
            except Exception as exc:          # noqa: BLE001
                if chosen is None:
                    chosen, error = (qid, text, norms), str(exc)[:120]
        if chosen is None:
            continue
        qid, text, norms = chosen
        short = qid[:14]
        if error and short in GOLD_DIALECT_FAILURES:
            error = GOLD_DIALECT_FAILURES[short]

        mine = results.get(qid)
        equivalents[qid] = [t for i, t, _ in golds
                            if i == qid or (mine is not None and results.get(i) == mine)]
        q = Question(
            qid=qid,
            inquiry=str(subject).rsplit("/", 1)[-1],
            prompt=prompt,
            gold_sql=text,
            gold_executable=error is None,
            gold_error=error,
            gold_rows=rows if error is None else None,
            alternates=tuple(x for x, _, _ in golds if x != qid),
            normalizations=norms,
        )
        out.append(q)
    con.close()

    loaded = _classifier() if classify else None
    if loaded is not None:
        onto, graph = loaded
        functional = functional_edges(onto, db)
        classified = []
        for q in out:
            classified.append(_classify(q, onto, graph, functional,
                                        equivalents.get(q.qid, [q.gold_sql]), db))
        out = classified

    out.sort(key=lambda q: (q.prompt, q.qid))
    if executable_only:
        out = [q for q in out if q.gold_executable]
    return out

def questions_hash(questions: Sequence[Question]) -> str:
    """Identity of the question SET. Two runs are comparable only if it matches."""
    payload = json.dumps(
        [[q.qid, q.prompt, q.gold_sql] for q in sorted(questions, key=lambda x: x.qid)],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]

class LLMUnavailable(RuntimeError):
    """No credentials, or the provider refused. Never raised in stub mode."""

@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: tuple[dict, ...] = ()

    cached_tokens: int = 0
    raw: Any = None

    temperature_dropped: bool = False

    structured_output: bool = False

def api_credentials() -> tuple[str, str | None]:
    """Key and base URL, scavenging sibling projects as this codebase does."""
    key, base = os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_API_BASE")
    if key:
        return key, base

    local = Path(__file__).resolve().parent.parent / ".env"
    if local.exists():
        env = {}
        for line in local.read_text().splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*", line)
            if m:
                env[m.group(1)] = m.group(2).strip("\"'")
        if env.get("OPENAI_API_KEY"):
            return env["OPENAI_API_KEY"], env.get("OPENAI_API_BASE") or None
    for candidate in (
    ):
        if not candidate.exists():
            continue
        env = {}
        for line in candidate.read_text().splitlines():
            m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*", line)
            if m:
                env[m.group(1)] = m.group(2).strip("\"'")
        if env.get("OPENAI_API_KEY"):
            return env["OPENAI_API_KEY"], env.get("OPENAI_API_BASE") or None
    raise LLMUnavailable("no OPENAI_API_KEY in env or sibling .env files")

_VENDOR_HOSTS = frozenset({
    "api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
})

_DEFAULT_BASE = "https://api.openai.com/v1"

def api_endpoint() -> dict[str, Any]:
    """Provenance for the LLM endpoint. NEVER the key -- only a digest of it.

    Three fields, and the third is the one that matters:

      `api_base`        the base URL, verbatim. A URL is not a secret; the key
                        is, and the key never appears here.
      `api_key_sha256`  first 12 hex of the SHA-256 of the key. Enough to say
                        "the same credential as that other run" or "a different
                        one", and not enough to be one. Preimage recovery from a
                        12-hex prefix of a high-entropy secret is not a thing.
      `api_direct`      False when the host is not a first-party vendor endpoint,
                        i.e. when a RESELLER served the run. `meta["model"]` is
                        then a request, not an observation, and every cost figure
                        is reseller pricing rather than list pricing.

    Never raises: a run that cannot resolve credentials still gets a provenance
    line saying so, because "unknown" is a fact and a missing field is not.
    """
    try:
        key, base = api_credentials()
    except LLMUnavailable:
        return {"api_base": None, "api_key_sha256": None, "api_direct": None}
    url = base or _DEFAULT_BASE
    host = urlparse(url).hostname or ""
    return {
        "api_base": url,
        "api_key_sha256": hashlib.sha256(key.encode()).hexdigest()[:12],
        "api_direct": host in _VENDOR_HOSTS,
    }

_CLIENT: Any = None

def _client() -> Any:
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        key, base = api_credentials()
        _CLIENT = OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)
    return _CLIENT

_TEMPERATURE_REFUSED = re.compile(
    r"temperature.*(unsupported|not supported|does not support|unknown|invalid)"
    r"|(unsupported|unknown|invalid).*temperature",
    re.I | re.S,
)

def complete(
    messages: Sequence[dict],
    *,
    model: str,
    temperature: float = 0.0,
    tools: Sequence[dict] | None = None,
    response_format: dict | None = None,
    tool_choice: dict | str | None = None,
) -> Completion:
    """THE seam. Every strategy's every round trip goes through here.

    `tool_choice` is the OTHER structured-output channel, and on the proxy this
    study runs against it is the only one that WORKS for every model family.
    Measured 2026-08-12, `claude-haiku-4-5` + `response_format:
    {"type":"json_schema", strict}`: the request is accepted, 200, and the reply
    is markdown prose. The schema is not enforced and is not refused -- the worst
    of the three possibilities, because nothing in the response says so.
    `json_object` is ignored the same way. The same schema sent as a FORCED tool
    call (`tools=[submit_*]`, `tool_choice={"type":"function", ...}`) comes back
    as a schema-shaped tool call from `claude-haiku-4-5` AND from `gpt-4o-mini`.
    Anything that needs the model's output constrained should therefore force a
    tool call, and anything that sends `response_format` to an Anthropic model
    through this proxy is hoping, not constraining.

    Swapping providers is this function; swapping models is its argument. Tests
    pass their own callable with the same signature and make no network call.

    NO SILENT RETRY. This used to be `except Exception: <call again without
    temperature>`, which is the single most dangerous line a benchmark harness
    can contain: a timeout, a rate limit or a 500 was answered by re-issuing the
    request at the PROVIDER DEFAULT temperature, and the run was then recorded as
    though it had been sampled at the temperature we asked for. On a study whose
    primary outcome is determinism at T=0 versus T=1 that silently manufactures
    agreement. The retry now fires only when the provider says, in as many words,
    that it will not take a temperature -- the reasoning-model case it was
    written for -- and what actually reached the API comes back on the
    `Completion` so the run record can state it rather than assume it.
    """
    kwargs: dict[str, Any] = {"model": model, "messages": list(messages)}

    kwargs["prompt_cache_key"] = f"spc-{model}"
    if tools:
        kwargs["tools"] = list(tools)

        if str(model).startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["reasoning_effort"] = "none"
    if response_format:
        kwargs["response_format"] = dict(response_format)
    if tool_choice:
        kwargs["tool_choice"] = (dict(tool_choice) if isinstance(tool_choice, dict)
                                 else tool_choice)
    dropped = False
    try:
        resp = _client().chat.completions.create(temperature=temperature, **kwargs)
    except Exception as exc:                   # noqa: BLE001
        if not _TEMPERATURE_REFUSED.search(str(exc)):
            raise
        dropped = True
        resp = _client().chat.completions.create(**kwargs)
    choice = resp.choices[0].message
    calls = tuple(
        {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
        for c in (choice.tool_calls or ())
    )
    usage = getattr(resp, "usage", None)
    return Completion(
        text=choice.content or "",
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        cached_tokens=getattr(getattr(usage, "prompt_tokens_details", None),
                              "cached_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        tool_calls=calls,
        raw=choice,
        temperature_dropped=dropped,
        structured_output=bool(response_format or tool_choice),
    )

def sampling_note(asked: float, *completions: Completion) -> dict:
    """What to record when what was SENT is not what was asked for.

    Empty in the ordinary case, so it costs nothing in the record; non-empty
    exactly when a run's sampling provenance would otherwise be a fiction.
    """
    if any(c.temperature_dropped for c in completions):
        return {"temperature_asked": asked, "temperature_sent": "provider default"}
    return {}

def strip_fences(text: str) -> str:
    """Unwrap a reply that IS a fenced block. It cannot find one INSIDE prose.

    Both substitutions are anchored -- `^` and `$` -- so `prose\n```sql\nSELECT
    1\n``` ` loses its trailing fence and keeps every word of the prose, and what
    goes to SQLite is prose plus SQL. That is not a bug in this function, whose
    job is to unwrap; it is a bug wherever this function is the ONLY reader of a
    reply that may contain prose. See `parse_sql_reply`.
    """
    text = re.sub(r"^\s*```(?:sql)?\s*", "", (text or "").strip(), flags=re.I)
    return re.sub(r"\s*```\s*$", "", text).strip()

_FENCED = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.S)

def parse_sql_reply(text: str) -> tuple[str, bool]:
    """`(sql, came_from_a_fence)`, or `ValueError`. BELT AND BRACES ONLY.

    The SQL twin of `parse_json_object`, and it exists for the same reason and
    with the same standing: the `skills` arm's terminal turn is CONSTRAINED --
    one mandatory `submit_sql` whose `sql` parameter is a typed string -- and a
    tool call's arguments do not contain markdown, so on that path the fenced
    branch should never fire and the flag it returns is there to prove it did
    not. It is not a substitute for the constraint and must never be used as one.

    What it closes when the provider does not honour the constraint: `strip_fences`
    is anchored at both ends, so `Here is the query:\n```sql\nSELECT 1\n``` `
    reaches SQLite as `Here is the query:\n```sql\nSELECT 1`, which cannot parse
    and scores WRONG on a question the model answered right. Exactly the defect
    that cost `menu_tools` 4 of its 5 first-probe questions, one type over.

    STRICT, deliberately, and identically to the pick reader: exactly one fenced
    block is extracted; two is a `ValueError` rather than a guess, because a
    reply showing a draft and then a correction has no reading a harness may pick
    on the reader's behalf. With no fence the reply is the query, unwrapped by
    `strip_fences` as before -- an unterminated ```sql opener is the one shape
    only that path catches. An EMPTY reply is `("", False)`, not an error: a
    model that said nothing is `NO_OUTPUT`, which is its own datum and not a
    parsing failure.
    """
    raw = (text or "").strip()
    if not raw:
        return "", False
    blocks = _FENCED.findall(raw)
    if len(blocks) > 1:
        raise ValueError(f"{len(blocks)} fenced blocks; refusing to guess which is the query")
    if blocks:
        return blocks[0].strip(), True
    return strip_fences(raw), False

def parse_json_object(text: str) -> tuple[dict, bool]:
    """`(pick, came_from_a_fence)`, or `ValueError`. BELT AND BRACES ONLY.

    The retrieved-menu arm's terminal turn is CONSTRAINED -- a mandatory tool
    call whose parameters are the pick schema -- and a tool call's arguments
    cannot contain a fence, so on that path this function's second branch should
    never fire, and the flag it returns is there to prove it did not. It exists
    because a proxy may accept a constraint and not honour it (this one accepts
    `response_format` from an Anthropic model and ignores it), and losing a
    correct pick to that is the exact failure this replaced. It is not a
    substitute for the constraint and must never be used as one.

    STRICT, deliberately: exactly one fenced block is extracted. Two blocks is an
    error rather than a guess -- a reply that shows a wrong pick and then a right
    one has no reading a harness may pick for the reader -- and a reply that is
    JSON but not an object is an error too, since a pick is an object.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty reply")
    try:
        direct = json.loads(raw)
    except Exception:                          # noqa: BLE001
        direct = _MISSING
    if direct is not _MISSING:
        if not isinstance(direct, dict):
            raise ValueError(f"reply was {type(direct).__name__}, not a pick object")
        return direct, False
    blocks = _FENCED.findall(raw)
    if not blocks:
        raise ValueError(f"not JSON and no fenced block: {raw[:120]!r}")
    if len(blocks) > 1:
        raise ValueError(f"{len(blocks)} fenced blocks; refusing to guess which is the pick")
    try:
        parsed = json.loads(blocks[0])
    except Exception as exc:                   # noqa: BLE001
        raise ValueError(f"the fenced block is not JSON ({exc}): {blocks[0][:120]!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"the fenced block was {type(parsed).__name__}, not a pick object")
    return parsed, True

_MISSING = object()

def _enum_values(node: Any) -> int:
    """Total enum values in a JSON schema -- the number providers cap."""
    if isinstance(node, dict):
        return len(node.get("enum", ())) + sum(_enum_values(v) for v in node.values())
    if isinstance(node, list):
        return sum(_enum_values(v) for v in node)
    return 0

@dataclass
class Attempt:
    """One strategy's answer to one asking of one question."""

    sql: str
    input_tokens: int = 0
    output_tokens: int = 0
    round_trips: int = 0
    tool_calls: tuple[str, ...] = ()
    violations: tuple[dict, ...] = ()
    violations_after: tuple[dict, ...] | None = None
    repaired: bool = False
    sql_before_repair: str | None = None
    error: str | None = None
    notes: dict = field(default_factory=dict)

SYSTEM_SQL = (
    "You are an expert SQL analyst. Given the information below and a question, "
    "write a single SQLite SELECT query that answers it. "
    "Respond with SQL only - no prose, no markdown fences, no explanation."
)

class StubStrategy:
    """Deterministic, zero-API strategy -- the harness's own test instrument.

    Modes exist to make each measurement provably visible:
      gold          the right answer            -> accuracy 1.0, TARr 1.0
      broken        valid SQL, wrong rows       -> accuracy 0.0, TARr 1.0
      drift         right rows, different text  -> accuracy 1.0, TARr 0.0, TARa 1.0
      role_blind    role predicate deleted      -> a REAL checker violation to repair
      flaky_broken  a DIFFERENT invalid column every run -> accuracy 0, TARr 0,
                    TARa 0. This is the adversarial case for the result metric: a
                    model hallucinating column names raises the same exception
                    class every time, and a result key built from the class alone
                    would call that "fully result-stable". Kept as a stub so the
                    regression cannot come back silently.
    """

    def __init__(self, mode: str = "gold", name: str | None = None) -> None:
        if mode not in ("gold", "broken", "drift", "role_blind", "flaky_broken"):
            raise ValueError(f"unknown stub mode {mode!r}")
        self.mode = mode
        self.name = name or f"stub_{mode}"

    def __call__(self, q: Question, run: int = 0) -> Attempt:
        if self.mode == "gold":
            return Attempt(sql=q.gold_sql)
        if self.mode == "drift":

            return Attempt(sql=f"{q.gold_sql}\n-- variant {run}")
        if self.mode == "broken":
            return Attempt(sql=f"SELECT * FROM (\n{q.gold_sql}\n) LIMIT 0")
        if self.mode == "flaky_broken":
            return Attempt(sql=f"SELECT hallucinated_column_{run} FROM Claim")
        sql = re.sub(
            r"(\w+\.)?party_role_code\s*=\s*'\w+'", "1=1", q.gold_sql, flags=re.I
        )
        return Attempt(sql=sql, notes={"role_predicates_removed": sql != q.gold_sql})

class DdlStrategy:
    """dbt's `sql` arm and our floor: the DDL file, and write SQL."""

    name = "ddl"

    def __init__(self, model: str, *, llm: Callable[..., Completion] = complete,
                 temperature: float = 0.0, ddl: str | Path = DDL) -> None:
        self.model, self.llm, self.temperature = model, llm, temperature

        self.ddl_path = Path(ddl)
        self._schema = self.ddl_path.read_text()

    def _messages(self, q: Question) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_SQL},
            {"role": "user",
             "content": f"Database schema:\n\n{self._schema}\n\n"
                        f"Question: {q.prompt}\n\nSQL:"},
        ]

    def __call__(self, q: Question, run: int = 0) -> Attempt:
        c = self.llm(self._messages(q), model=self.model, temperature=self.temperature)
        return Attempt(sql=strip_fences(c.text), input_tokens=c.input_tokens,
                       output_tokens=c.output_tokens, round_trips=1,
                       notes=sampling_note(self.temperature, c))

def render_ontology(
    ontology: str | Path = ONTOLOGY, mapping: str | Path = MAPPING
) -> str:
    """The ontology as prompt context: the two layers JOINED, re-serialised.

    The parsed document is re-serialised rather than the file text being pasted,
    for one reason: the ontology's comments argue for this project's design
    ("THE CONTRIBUTION", "a semantic-layer YAML would emit ONE permissive
    relationship"). Feeding that to the arm under test is prompt engineering
    wearing an ontology's clothes. Descriptions, which are domain knowledge, are
    data and survive; commentary, which is advocacy, does not.

    The A1 arm needs the JOINED document, not the semantic layer alone: it
    writes SQL, so it must see tables and columns. `spc/ontology.join_layers`
    produces exactly the mapping the single-file ontology parsed to (verified
    deep-equal at the split), so this arm's payload is unchanged by the split
    apart from per-node key order.
    """
    import yaml

    from spc.ontology import join_layers, read_ddl_schema

    sem = yaml.safe_load(Path(ontology).read_text())
    phys = yaml.safe_load(Path(mapping).read_text())
    doc = join_layers(sem, phys, schema=read_ddl_schema(DDL))
    doc.pop("_unverified_tables", None)
    return yaml.safe_dump(doc, sort_keys=False, width=100)

class ContextStrategy:
    """A1: ontology knowledge WITHOUT enforcement. The ablation that says whether
    the governance or merely the knowledge is doing the work."""

    name = "context"

    def __init__(self, model: str, *, llm: Callable[..., Completion] = complete,
                 temperature: float = 0.0, include_ddl: bool = False) -> None:
        self.model, self.llm, self.temperature = model, llm, temperature
        self.context = render_ontology()
        if include_ddl:
            self.context += "\n\n-- physical schema --\n" + Path(DDL).read_text()
        self.include_ddl = include_ddl

    def _messages(self, q: Question) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_SQL},
            {"role": "user",
             "content": "Business ontology (concepts map to tables, edges are the "
                        "sanctioned joins, metrics are the governed definitions):\n\n"
                        f"{self.context}\n\nQuestion: {q.prompt}\n\nSQL:"},
        ]

    def __call__(self, q: Question, run: int = 0) -> Attempt:
        c = self.llm(self._messages(q), model=self.model, temperature=self.temperature)
        return Attempt(sql=strip_fences(c.text), input_tokens=c.input_tokens,
                       output_tokens=c.output_tokens, round_trips=1,
                       notes=sampling_note(self.temperature, c))

SKILL_FUNCTIONS = {
    "find_paths": "Every governed path between two concepts. Use before writing any join.",
    "describe_concept": "A concept's table, columns, grain and edges.",
    "find_metric": "The governed definition of a business metric by name or phrase.",
    "search_values": "Canonical stored values for a column, so a literal is grounded.",
    "check_sql": "Check a draft query against the ontology; returns violations.",
}

class SkillsUnavailable(RuntimeError):
    pass

def load_skills(module: Any = None) -> tuple[list[dict], Callable[[str, dict], Any]]:
    """Tool schemas and a dispatch function, from spc.skills. THE SEAM.

    The shipped module publishes `SKILL_SPECS` (the same records `spc/mcp_server`
    serves, so the in-process arm and an MCP client see identical tools) and one
    `call(name, arguments)` entry point that turns every failure into an `ok=False`
    result. That is the preferred shape; the others below are fallbacks kept so a
    reshaped skills module does not silently disable the arm.
    """
    if module is None:
        try:
            from spc import skills as module  # type: ignore[no-redef]
        except Exception as exc:              # noqa: BLE001
            raise SkillsUnavailable(f"spc/skills.py not importable: {exc}") from exc

    if hasattr(module, "SKILL_SPECS") and hasattr(module, "call"):
        schemas = [{"type": "function", "function": {
            "name": spec.name, "description": spec.description,
            "parameters": dict(spec.input_schema)}} for spec in module.SKILL_SPECS]

        def dispatch(name: str, args: dict) -> Any:
            result = module.call(name, args)

            text = getattr(result, "text", None)
            if text:
                return {"ok": getattr(result, "ok", True), "text": text}
            return getattr(result, "data", result)

        return schemas, dispatch
    if hasattr(module, "build_tools"):
        schemas, dispatch = module.build_tools(ontology=ONTOLOGY, mapping=MAPPING, db=DB)
        return list(schemas), dispatch
    if hasattr(module, "TOOLS") and hasattr(module, "dispatch"):
        return list(module.TOOLS), module.dispatch

    import inspect

    schemas, funcs = [], {}
    for name, description in SKILL_FUNCTIONS.items():
        fn = getattr(module, name, None)
        if fn is None:
            continue
        funcs[name] = fn
        props, required = {}, []
        for pname, param in inspect.signature(fn).parameters.items():
            if pname in ("ontology", "graph", "db", "onto"):
                continue
            props[pname] = {"type": "string", "description": pname}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        schemas.append({"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required}}})
    if not funcs:
        raise SkillsUnavailable(
            "spc.skills exposes none of build_tools/TOOLS/"
            + "/".join(SKILL_FUNCTIONS)
        )

    def dispatch(name: str, args: dict) -> Any:
        fn = funcs[name]
        import inspect as _i

        accepted = _i.signature(fn).parameters
        bound = dict(args)
        for key, value in (("ontology", ONTOLOGY), ("db", DB), ("graph", None)):
            if key in accepted and key not in bound and value is not None:
                bound[key] = value
        return fn(**bound)

    return schemas, dispatch

class SkillsStrategy:
    """The LLM calls the governed tools, then writes SQL.

    Round trips are counted and reported: a tool-calling strategy is several
    calls where the DDL arm is one, and that cost must be visible rather than
    hidden inside a single "accuracy" number.

    THE TERMINAL TURN, added 2026-08-12, and it is the SAME RULE `menu_tools`
    runs -- read that class's docstring for the argument; only the payload
    differs. This arm emits SQL rather than a pick, so the constraint is not a
    menu schema: it is a typed `sql` string on one mandatory tool. Nothing about
    what the arm may DO changed. It still writes any SQLite it likes and is still
    judged by executing it (and by whatever it chose to run through `check_sql`);
    the change is that its answer is no longer discarded.

    The two holes it closes, both measured on the same 5-question probe
    (`results/probe_haiku_t0_skills.jsonl`, claude-haiku-4-5, T=0):

      * TERMINATION. "How many claims do we have?" spent all 8 rounds on
        `check_sql`/`find_metric` and never emitted SQL --
        `TOOL_LOOP_EXHAUSTED`, one of five questions, no answer recorded. The
        retrieval budget is now `max_rounds - 1` and the turn after it is
        terminal, so a loop that runs out of rounds ends in a query over
        whatever it did retrieve instead of ending in nothing.
      * PARSING. `strip_fences` is anchored at both ends, so a reply that reasons
        in prose and THEN fences its SQL was handed to SQLite whole and scored
        wrong. It did not fire in the 5-question probe and it is live: the same
        model produced exactly that shape for `menu_tools` on 4 of 5 questions.
        `parse_sql_reply` is the reader now, and it runs on the text channel
        where the mangling was possible.

    Two deterministic triggers end retrieval, both in `notes["terminal_reason"]`:
    `model_stopped` (a reply with no tool calls) and `budget_spent`. The terminal
    turn is issued ALWAYS -- including when the model's last message already
    looked like SQL -- because an arm that is only sometimes constrained has a
    claim that only sometimes holds, and the price is one round trip per question
    and is in `round_trips`.

    Sent as a FORCED tool call, not as `response_format`: this proxy accepts a
    strict `json_schema` from `claude-haiku-4-5`, returns 200 and replies in
    markdown anyway (measured; see `complete`). A provider that REFUSES the
    schema fails loudly for that run -- `schema refused: ...`, `SCHEMA_REFUSED`
    -- and never falls back to an unconstrained request. A provider that ACCEPTS
    it and ignores it cannot be made to fail, so it is recorded:
    `notes["sql_channel"]` is `tool_call` when the query came back through the
    constrained channel and `text` when it did not, and `notes["fence_extracted"]`
    / `notes["fence_rescued"]` mark a run that needed the fence reader and a run
    the old reader would have MANGLED. Any run not marked `tool_call` is a run
    whose output was not constrained, whatever the request asked for.
    """

    name = "skills"

    def __init__(self, model: str, *, llm: Callable[..., Completion] = complete,
                 temperature: float = 0.0, max_rounds: int = 8,
                 skills_module: Any = None) -> None:
        self.model, self.llm, self.temperature = model, llm, temperature
        self.max_rounds = max_rounds
        self.tools, self.dispatch = load_skills(skills_module)

    TERMINAL = (
        "Retrieval is over. Submit the single SQLite SELECT that answers the "
        "question, through submit_sql. It is executed exactly as you write it."
    )

    SUBMIT = "submit_sql"
    SQL_PARAMETERS: dict = {
        "type": "object",
        "properties": {"sql": {
            "type": "string",
            "description": "One SQLite SELECT statement, and nothing else: "
                           "no prose, no markdown fences, no trailing semicolon "
                           "commentary.",
        }},
        "required": ["sql"],
        "additionalProperties": False,
    }

    def __call__(self, q: Question, run: int = 0) -> Attempt:
        messages: list[dict] = [
            {"role": "system", "content":
                SYSTEM_SQL + " Tools are available that answer questions about the "
                "governed ontology: use them to establish the join path, the metric "
                "definition and any literal value before you write SQL."},
            {"role": "user", "content": f"Question: {q.prompt}"},
        ]
        tin = tout = trips = 0
        used: list[str] = []
        reason = "budget_spent"

        for _ in range(max(self.max_rounds - 1, 0)):
            c = self.llm(messages, model=self.model, temperature=self.temperature,
                         tools=self.tools)
            tin, tout, trips = tin + c.input_tokens, tout + c.output_tokens, trips + 1
            if not c.tool_calls:

                if c.text:
                    messages.append({"role": "assistant", "content": c.text})
                reason = "model_stopped"
                break
            messages.append({"role": "assistant", "content": c.text or None,
                             "tool_calls": [
                                 {"id": tc["id"], "type": "function",
                                  "function": {"name": tc["name"],
                                               "arguments": tc["arguments"]}}
                                 for tc in c.tool_calls]})
            for tc in c.tool_calls:
                used.append(tc["name"])
                try:
                    args = json.loads(tc["arguments"] or "{}")
                    result = self.dispatch(tc["name"], args)
                except Exception as exc:      # noqa: BLE001
                    result = {"error": str(exc)[:200]}
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(result, default=str)[:6000]})
        return self._terminal(messages, tin, tout, trips, used, reason)

    def _terminal(self, messages: list[dict], tin: int, tout: int, trips: int,
                  used: list[str], reason: str) -> Attempt:
        """ONE constrained request, and its answer is the arm's answer.

        `submit_sql` is deliberately NOT appended to `Attempt.tool_calls`: that
        list is read as "which GOVERNED tools did the model choose", and one
        mandatory call the harness makes on every question is not a choice. The
        terminal turn is visible in `round_trips` and in `notes` instead.
        """
        base = {"input_tokens": tin, "output_tokens": tout, "round_trips": trips,
                "tool_calls": tuple(used)}
        note: dict[str, Any] = {"terminal_reason": reason}
        submit = [{"type": "function", "function": {
            "name": self.SUBMIT, "strict": True,
            "description": "Submit the query that answers the question. It is run "
                           "against the database exactly as written.",
            "parameters": dict(self.SQL_PARAMETERS)}}]
        messages = [*messages, {"role": "user", "content": self.TERMINAL}]
        try:

            c = self.llm(messages, model=self.model, temperature=self.temperature,
                         tools=submit,
                         tool_choice={"type": "function",
                                      "function": {"name": self.SUBMIT}})
        except Exception as exc:                # noqa: BLE001
            return Attempt(sql="", error=f"schema refused: {type(exc).__name__}: {exc}"[:300],
                           notes=note, **base)
        base = {**base, "input_tokens": tin + c.input_tokens,
                "output_tokens": tout + c.output_tokens, "round_trips": trips + 1}
        note.update(structured_output=c.structured_output,
                    **sampling_note(self.temperature, c))
        submitted = next((tc for tc in c.tool_calls if tc["name"] == self.SUBMIT), None)
        if submitted is not None:
            note["sql_channel"] = "tool_call"
            raw_args = submitted["arguments"] or ""
            try:
                args = json.loads(raw_args or "{}")
            except Exception as exc:            # noqa: BLE001
                return Attempt(sql="", notes={**note, "sql_raw": raw_args[:600]},
                               error=f"sql was not readable: submit_sql arguments were "
                                     f"not JSON ({exc})"[:300], **base)
            if not isinstance(args, dict) or not isinstance(args.get("sql"), str):
                return Attempt(sql="", notes={**note, "sql_raw": raw_args[:600]},
                               error="sql was not readable: submit_sql carried no `sql` "
                                     f"string ({raw_args[:80]!r})"[:300], **base)
            raw = args["sql"]
        else:

            note["sql_channel"] = "text"
            if c.tool_calls:

                note["terminal_tool_calls"] = [tc["name"] for tc in c.tool_calls]
            raw = c.text or ""
        try:
            sql, fenced = parse_sql_reply(raw)
        except ValueError as exc:
            return Attempt(sql="", error=f"sql was not readable: {exc}"[:300],
                           notes={**note, "sql_raw": raw[:600]}, **base)
        if fenced:

            note["fence_extracted"] = True
            if sql != strip_fences(raw):
                note["fence_rescued"] = True
        return Attempt(sql=sql, notes=note, **base)

_CHECKERS: dict[str, Callable[[str], list[Any]]] = {}

def load_checker(db: str | Path = DB) -> Callable[[str], list[Any]]:
    """`sql -> [Violation]`, bound to the ontology and the database."""
    cached = _CHECKERS.get(str(db))
    if cached is not None:
        return cached
    import yaml

    from spc import check as check_mod

    from spc.ontology import join_layers, read_ddl_schema

    doc = join_layers(yaml.safe_load(Path(ONTOLOGY).read_text()),
                      yaml.safe_load(Path(MAPPING).read_text()),
                      schema=read_ddl_schema(DDL))
    doc.pop("_unverified_tables", None)
    try:
        from spc.graph import PathGraph

        graph = PathGraph.load(ONTOLOGY, MAPPING)
    except Exception:                          # noqa: BLE001
        graph = None

    def checker(sql: str) -> list[Any]:
        return check_mod.check(sql, doc, graph, db=str(db))

    _CHECKERS[str(db)] = checker
    return checker

def violation_dict(v: Any) -> dict:
    return {"code": getattr(v, "code", "?"), "severity": getattr(v, "severity", "violation"),
            "message": getattr(v, "message", str(v))[:300],
            "fragment": getattr(v, "fragment", "")[:200]}

def certify(sql: str, checker: Callable[[str], list[Any]], *, base: dict,
            notes: dict) -> Attempt:
    """The last stage of a COMPILED arm: `... -> COMPILE -> CHECK -> SQL`.

    DESIGN 3 draws the pipeline with CHECK in it, and `spc/check.py` is the
    module every "the checker catches X" claim is about. Until 2026-08-12 the
    menu arms went `parse_pick -> compile -> return Attempt(sql=...)` and never
    called it, so those claims were true of the module and false of the system
    the benchmark measured. Both arms come through here now.

    THREE OUTCOMES, not two, because `check()` reports two severities:

      certified     no violations                -> the SQL is the answer
      violation     severity "violation"         -> proven unsafe. REFUSED: an
                    uncertifiable number is worse than no number (the compiler's
                    own rule 4), and returning it "with a warning attached" is
                    exactly the silent-wrong-number the study exists to prevent.
      undecidable   only severity "undecidable"  -> the checker cannot decide
                    this shape. NOT a pass -- nothing was certified -- and not a
                    violation either, since nothing was proven wrong. Refused
                    with its OWN reason so the two never sum together in a table.

    A refusal here is `sql=""` like every other refusal, so `Scorer` scores it as
    one; the codes and a digest of the rejected SQL go into `error`, which is what
    `refusal_key`/`result_key` hash, so two runs rejected for different codes --
    or for the same code on different SQL -- do not agree on TARa@N. The rejected
    SQL is kept in `notes` because a refusal nobody can read is not a datum.
    """
    found = list(checker(sql))
    violations = tuple(violation_dict(v) for v in found)
    if not found:
        return Attempt(sql=sql, violations=violations, notes=notes, **base)
    from spc.check import blocking as _blocking
    blocking = _blocking(found)
    reported = blocking or found
    codes = ", ".join(sorted({getattr(v, "code", "?") for v in reported}))
    prefix = "checker rejected" if blocking else "checker undecidable"
    digest = hashlib.sha256(sql.encode()).hexdigest()[:12]
    return Attempt(
        sql="", violations=violations,
        error=f"{prefix}: {codes} [sql {digest}]",
        notes={**notes, "rejected_sql": sql[:2000]}, **base)

class CheckedStrategy:
    """Any strategy, plus the checker and AT MOST ONE targeted repair.

    One repair, not a loop: the question is whether a violation report is enough
    to fix a query, and an unbounded loop answers a different question while
    spending an unbounded amount of money. The pre-repair SQL is kept so the
    runner can score both and say whether the repair actually helped.
    """

    def __init__(self, inner: Any, *, model: str | None = None,
                 llm: Callable[..., Completion] = complete, temperature: float = 0.0,
                 db: str | Path = DB, name: str | None = None,
                 checker: Callable[[str], list[Any]] | None = None) -> None:
        self.inner = inner
        self.model = model or getattr(inner, "model", None)
        self.llm = llm
        self.temperature = temperature
        self.checker = checker or load_checker(db)
        self.name = name or f"{getattr(inner, 'name', 'inner')}_checked"

    def __call__(self, q: Question, run: int = 0) -> Attempt:
        attempt = self.inner(q, run)
        if not attempt.sql.strip():
            return attempt
        found = self.checker(attempt.sql)
        from spc.check import blocking as _blocking
        blocking = _blocking(found)
        attempt.violations = tuple(violation_dict(v) for v in found)
        if not blocking:
            return attempt

        report = "\n".join(
            f"- {v.code}: {v.message}" + (f"  [{v.fragment}]" if v.fragment else "")
            for v in blocking
        )
        if self.model is None or self.llm is None:
            attempt.notes["repair_skipped"] = "no model bound"
            return attempt
        c = self.llm(
            [{"role": "system", "content": SYSTEM_SQL},
             {"role": "user", "content":
                 f"Question: {q.prompt}\n\nThis query violates the governed "
                 f"ontology:\n\n{attempt.sql}\n\nViolations:\n{report}\n\n"
                 "Rewrite it so every violation is resolved. SQL only."}],
            model=self.model, temperature=self.temperature)
        repaired_sql = strip_fences(c.text)
        after = self.checker(repaired_sql) if repaired_sql.strip() else []
        attempt.sql_before_repair = attempt.sql
        attempt.sql = repaired_sql or attempt.sql
        attempt.violations_after = tuple(violation_dict(v) for v in after)
        attempt.repaired = True
        attempt.input_tokens += c.input_tokens
        attempt.output_tokens += c.output_tokens
        attempt.round_trips += 1
        return attempt

class MenuUnavailable(RuntimeError):
    pass

class MenuStrategy:
    """ONE structured pick from a governed menu, then a deterministic compile.

    MetricFlow's shape -- the model chooses measures and dimensions, the compiler
    writes the SQL -- plus the thing MetricFlow's rules forbid: the PATH is also
    on the menu, so a three-hop dimension across a fan-out edge is selectable
    rather than refused. Determinism is structural here: every run that picks the same
    menu index emits byte-identical SQL, so TARr@N should be 1.0 whenever the
    pick is stable, and the residual variance is entirely the LLM's choice.

    THE SEAM, as shipped: `menu.build_menu(concepts, onto, graph)` builds the
    governed choice set, `menu.render()` is what the model reads,
    `menu.parse_pick(pick, menu) -> Plan` rejects anything off the menu with
    `PickError`, and `compile.compile(plan, onto, graph) -> SQL` lowers it. Every
    refusal along that chain is RECORDED AS A DATUM -- an ungoverned pick that
    cannot be compiled is the arm's designed behaviour, not a harness failure.

    COST NOTE, measured not guessed -- and re-measure it, do not cite this:
    the menu is a pure function of the ontology, so every edge anyone declares
    multiplies the routes and this number moves under you. As of 2026-08-12 the
    menu over all 13 concepts is 1448 routes and 58,990 tokens (tiktoken
    cl100k) -- 1.6x the 904/36k this docstring used to claim, entirely from two
    edges added to acme.yaml. Print the current figure with
    `python spc/menu.py`, or `Menu.token_size()`. `concepts=` narrows the menu;
    with no grounding stage in the system there is no automatic narrowing, and
    inventing one here would be an unmeasured selection step inside the arm
    under test.

    STRUCTURED OUTPUT: the pick schema is SENT, as a strict `json_schema`
    response format, not merely available for validation afterwards. Note the
    enum budget -- 1448 route ids appear in three separate enums -- may exceed
    what a provider will accept for constrained decoding. That failure is a
    per-run error datum, not a silent fallback to free text.
    """

    name = "menu"

    def __init__(self, model: str, *, llm: Callable[..., Completion] = complete,
                 temperature: float = 0.0, menu_module: Any = None,
                 compile_module: Any = None, concepts: Sequence[str] | None = None,
                 max_hops: int = 4, db: str | Path = DB,
                 checker: Callable[[str], list[Any]] | None = None) -> None:
        self.model, self.llm, self.temperature = model, llm, temperature
        self.db = db
        self._checker = checker
        try:
            if menu_module is None:
                from spc import menu as menu_module  # type: ignore[no-redef]
            if compile_module is None:
                from spc import compile as compile_module  # type: ignore[no-redef]
            from spc.graph import PathGraph
            from spc.ontology import load_ontology
        except Exception as exc:               # noqa: BLE001
            raise MenuUnavailable(
                f"spc/menu.py + spc/compile.py not importable: {exc}") from exc
        self.menu_module, self.compile_module = menu_module, compile_module
        self.onto = load_ontology(ONTOLOGY, MAPPING)
        self.graph = PathGraph(self.onto)
        self.concepts = tuple(concepts) if concepts else tuple(sorted(self.onto.concepts))
        self.max_hops = max_hops

        self.menu = menu_module.build_menu(self.concepts, self.onto, self.graph,
                                           max_hops=max_hops)
        self.rendered = self.menu.render()

        self.response_format = self.menu.response_format()

    @property
    def checker(self) -> Callable[[str], list[Any]]:
        """Built on first use, not in `__init__`.

        `spc/tests/test_prompts.py` constructs this arm dozens of times to read
        one prompt payload off it, and none of those constructions checks any
        SQL. `load_checker` is memoised per database, so the first real use pays
        once for the whole process.
        """
        if self._checker is None:
            self._checker = load_checker(self.db)
        return self._checker

    SYSTEM = (
        "Choose from the governed menu below. Reply with JSON only -- the pick "
        "object the response schema describes. Do not write SQL: the SQL is "
        "compiled from your pick, and anything not on the menu is rejected."
    )

    def __call__(self, q: Question, run: int = 0) -> Attempt:
        c = self.llm(
            [{"role": "system", "content": self.SYSTEM},
             {"role": "user", "content": f"{self.rendered}\n\nQuestion: {q.prompt}\n\nPick:"}],
            model=self.model, temperature=self.temperature,
            response_format=self.response_format)
        raw = strip_fences(c.text)
        base = {"input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                "round_trips": 1}
        note = {"structured_output": c.structured_output,
                **sampling_note(self.temperature, c)}
        try:
            pick = json.loads(raw)
        except Exception:                      # noqa: BLE001
            return Attempt(sql="", error=f"pick was not JSON: {raw[:120]!r}",
                           notes={**note, "pick_raw": raw[:600]}, **base)
        try:
            plan = self.menu_module.parse_pick(pick, self.menu)
        except Exception as exc:               # noqa: BLE001
            return Attempt(sql="", error=f"pick rejected: {type(exc).__name__}: {exc}"[:300],
                           notes={**note, "pick": pick}, **base)
        try:
            sql = self.compile_module.compile(plan, self.onto, self.graph)
        except Exception as exc:               # noqa: BLE001
            return Attempt(sql="", error=f"compile refused: {type(exc).__name__}: {exc}"[:300],
                           notes={**note, "pick": pick}, **base)
        return certify(sql, self.checker, base=base, notes={**note, "pick": pick})

def subject_route_id() -> str:
    """`spc.menu.SUBJECT_ID` -- the id meaning "no traversal", never a retrieval.

    Resolved lazily and with a fallback because `spc/bench.py` must still import
    when the compiled arms' modules are absent (`MenuUnavailable`). The fallback
    is pinned equal to the real constant by `test_bench.test_terminal_turn`, so
    the two cannot drift apart unnoticed.
    """
    try:
        from spc.menu import SUBJECT_ID  # noqa: PLC0415

        return SUBJECT_ID
    except Exception:                     # noqa: BLE001
        return "SELF"

def load_pick_skills(module: Any = None) -> tuple[list[dict], Callable[[str, dict], Any], Any]:
    """The PICK arm's seam: tool schemas, a dispatch, and the menu bridge.

    Three things rather than two, because this arm needs one more than the SQL
    arm does: something that turns the model's pick back into a `Plan` over the
    routes it RETRIEVED. That is `spc.skills.Skills`, and it is returned rather
    than reconstructed here so the ids the model was shown and the ids that are
    resolved come from one object.
    """
    if module is None:
        try:
            from spc import skills as module  # type: ignore[no-redef]
        except Exception as exc:              # noqa: BLE001
            raise SkillsUnavailable(f"spc/skills.py not importable: {exc}") from exc
    for attribute in ("PICK_SKILL_SPECS", "call_pick", "Skills"):
        if not hasattr(module, attribute):
            raise SkillsUnavailable(
                f"spc/skills.py has no {attribute}: the retrieved-menu arm needs the "
                f"pick registry, its dispatch and the menu bridge"
            )
    skills = module.Skills()
    schemas = [{"type": "function", "function": {
        "name": spec.name, "description": spec.description,
        "parameters": dict(spec.input_schema)}} for spec in module.PICK_SKILL_SPECS]

    def dispatch(name: str, args: dict) -> Any:
        """The `SkillResult` itself, not a rendering of it.

        The caller needs BOTH halves and they used to be thrown away here: the
        `text` is what goes back to the model, and `data` is where `find_paths`
        puts the route ids -- which are what the terminal turn's response schema
        enumerates. `RetrievedMenuStrategy.tool_payload` builds the message, so
        the bytes the model reads are still decided in one place.
        """
        return module.call_pick(name, args, skills=skills)

    return schemas, dispatch, skills

class RetrievedMenuStrategy:
    """The menu arm, with the menu RETRIEVED instead of handed over.

    `MenuStrategy` renders EVERY governed route into the prompt -- 1448 of them
    and 58,990 cl100k tokens as of 2026-08-12, against 6.4k characters of DDL for
    the floor arm. That ratio is the whole reason this class exists, and it grows
    with the ontology rather than staying where a docstring pinned it. Here the
    model asks for the routes between the concepts it cares about, sees 5-15, and
    picks over those.

    Everything after the pick is unchanged and shared with `MenuStrategy`:
    `spc.menu.parse_pick` refuses an id that names no governed route, and
    `spc.compile.compile` writes the SQL. The model still cannot express an
    ungoverned join -- it simply learns the vocabulary a few routes at a time.

    Cost moves from input tokens to ROUND TRIPS, and both are recorded, so the
    trade is reported rather than assumed.

    STRUCTURED OUTPUT, on the TERMINAL TURN. Until 2026-08-12 this arm's pick was
    free text: the route ids do not exist until retrieval, so the FIRST request
    cannot enumerate them, and "cannot constrain the first request" was allowed to
    stand as "does not constrain any request". It cost the arm every question in
    the first real probe -- 4 of 5 replies were a correct pick wrapped in prose and
    a ```json fence, discarded as NON_JSON_PICK, plus one loop that never
    terminated. The model was right and the harness threw the answer away.

    The rule now, and it is a rule of the LOOP, not a sentence in the prompt:

      * every request while the model is calling tools is unconstrained, exactly
        as before -- retrieval is what mints the ids;
      * the turn AFTER retrieval ends is TERMINAL. It carries
        `Skills.pick_schema(retrieved_route_ids)` -- whose route enum is `SELF`
        plus exactly the ids `find_paths` returned in THIS question -- as a
        FORCED tool call, `submit_pick`. The retrieval tools are not offered on
        that turn and the one tool that is offered is mandatory, so the turn is
        terminal by construction rather than by instruction. Sent as a tool call
        rather than as `response_format` because that is what the provider
        HONOURS: this proxy accepts a strict `json_schema` from an Anthropic
        model, returns 200, and ignores it (measured; see `complete`);
      * retrieval ends when the model stops calling tools (`model_stopped`) or
        when the retrieval budget `max_rounds - 1` is spent (`budget_spent`).
        The second trigger is what replaces TOOL_LOOP_EXHAUSTED: a loop that ran
        out of rounds now ends in a constrained pick over whatever it did
        retrieve, instead of ending in nothing;
      * the terminal turn is issued ALWAYS -- including when the model's own last
        message already looked like a pick. A pick that is sometimes constrained
        is an arm whose expressibility claim holds sometimes, and the whole point
        of the arm is that the claim is structural. The price is exactly one
        extra round trip per question, and it is in `round_trips`.

    If the provider REFUSES the schema (the enum budget is the known risk; a
    retrieved menu is ~90-150 enum values against ~4400 for the full menu, but
    that is measured per run, not assumed) the arm FAILS LOUDLY for that run --
    `schema refused: ...`, its own refusal class -- and never falls back to
    unconstrained text. A silent fallback would put an unconstrained run in a
    column labelled constrained, which is the defect this docstring exists about.
    A provider that ACCEPTS the constraint and ignores it is the nastier case and
    cannot be made to fail, so it is recorded instead: `notes["pick_channel"]` is
    `tool_call` when the pick came back through the constrained channel and
    `text` when it did not, and `notes["fence_extracted"]` marks a run that only
    parsed because of the belt-and-braces reader. Any run not marked `tool_call`
    is a run whose pick was NOT constrained, whatever the request asked for.
    """

    name = "menu_tools"

    def __init__(self, model: str, *, llm: Callable[..., Completion] = complete,
                 temperature: float = 0.0, max_rounds: int = 8,
                 skills_module: Any = None, compile_module: Any = None,
                 db: str | Path = DB,
                 checker: Callable[[str], list[Any]] | None = None) -> None:
        self.model, self.llm, self.temperature = model, llm, temperature
        self.db = db
        self._checker = checker
        self.max_rounds = max_rounds
        self.tools, self.dispatch, self.skills = load_pick_skills(skills_module)
        try:
            if compile_module is None:
                from spc import compile as compile_module  # type: ignore[no-redef]
        except Exception as exc:                # noqa: BLE001
            raise MenuUnavailable(f"spc/compile.py not importable: {exc}") from exc
        self.compile_module = compile_module

    @property
    def checker(self) -> Callable[[str], list[Any]]:
        """Built on first use -- see `MenuStrategy.checker`."""
        if self._checker is None:
            self._checker = load_checker(self.db)
        return self._checker

    SYSTEM = (
        "You answer questions over a governed ontology. You do NOT write SQL.\n\n"
        "Work in two phases.\n"
        "1. RETRIEVE. Call the tools to find the subject concept, the attribute names, "
        "the governed metric definitions, and -- with find_paths -- the route ids that "
        "reach everything the question needs. Call as many as you need and no more.\n"
        "2. PICK. Reply with ONE JSON object and nothing else:\n"
        '{"subject": <concept>, "measures": [{"metric": <name>, "route": <route id>} | '
        '{"aggregation": <sum|count|count_distinct|avg|min|max>, "route": <route id>, '
        '"attribute": <name>}], "dimensions": [{"route": <route id>, "attribute": <name>}], '
        '"filters": [{"route": <route id>, "attribute": <name>, "operator": <op>, '
        '"value": <literal>}], "top": {"by": "measure"|"dimension", "index": 0, '
        '"descending": true, "n": 10}}\n\n'
        "Every `route` is either a route id find_paths returned, or \"SELF\" for an "
        "attribute of the subject concept itself. Omit any list that is empty. The SQL "
        "is compiled from your pick, so a route you did not retrieve is a route you "
        "cannot use."
    )

    TERMINAL = (
        "Retrieval is over. Submit your pick. Its schema enumerates every route "
        "id you retrieved; `SELF` means an attribute of the subject concept itself."
    )

    SUBMIT = "submit_pick"

    @staticmethod
    def tool_payload(result: Any) -> str:
        """What one tool answer looks like to the model. Unchanged bytes."""
        if isinstance(result, dict):
            body: Any = result
        else:
            body = {"ok": result.ok, "text": result.text}
        return json.dumps(body, default=str)[:6000]

    @staticmethod
    def route_ids_of(result: Any) -> tuple[str, ...]:
        """The route ids one tool answer MINTED, in the order it listed them.

        Read off the structured payload rather than scraped out of the rendered
        text, so the enum the terminal turn sends is the enum the model was shown
        -- a rendering change cannot silently empty it.

        WALKED, not read off `data["paths"]`, and that is the repair of a real
        defect. `find_paths` mints ids under `paths[*].route_id`; `find_metric`
        mints them too, under `routes_from_subject[*].compatible_route_ids`, and
        that is the ONE pairing the model cannot guess -- which route may carry a
        governed metric. Reading only `paths` meant the arm SHOWED the model an
        id ("from Policy use route: Policy>ClaimAmount#4"), left it out of the
        terminal turn's route enum, and then rejected the pick that used it as
        `UnretrievedRouteError: a governed route this question never retrieved`.
        Every such refusal in `results/camp_haiku_t0_n5.jsonl` (10 of them, on the
        two questions whose only retrieval was `find_metric`) is this bug, not a
        model failure: the transcript contains the id, minted by our own tool.

        The provenance check itself is unchanged and stays -- an id the model
        invented is still rejected. What changes is that "retrieved" now means
        "any tool put this id in front of the model", which is what the check was
        always trying to say.
        """
        data = getattr(result, "data", None)
        if not isinstance(data, dict):
            return ()
        found: list[str] = []
        self_id = subject_route_id()

        def note(value: Any) -> None:
            text = str(value or "")
            if not text or text == self_id or text in found:
                return
            found.append(text)

        def walk(node: Any) -> None:
            if isinstance(node, Mapping):
                note(node.get("route_id"))
                for identifier in node.get("compatible_route_ids") or ():
                    note(identifier)
                for value in node.values():
                    if isinstance(value, (Mapping, list, tuple)):
                        walk(value)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    walk(item)

        walk(data)
        return tuple(found)

    def __call__(self, q: Question, run: int = 0) -> Attempt:
        messages: list[dict] = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": f"Question: {q.prompt}"},
        ]
        tin = tout = trips = 0
        used: list[str] = []

        retrieved: dict[str, None] = {}
        reason = "budget_spent"

        for _ in range(max(self.max_rounds - 1, 0)):
            c = self.llm(messages, model=self.model, temperature=self.temperature,
                         tools=self.tools)
            tin, tout, trips = tin + c.input_tokens, tout + c.output_tokens, trips + 1
            if not c.tool_calls:

                if c.text:
                    messages.append({"role": "assistant", "content": c.text})
                reason = "model_stopped"
                break
            messages.append({"role": "assistant", "content": c.text or None,
                             "tool_calls": [
                                 {"id": tc["id"], "type": "function",
                                  "function": {"name": tc["name"],
                                               "arguments": tc["arguments"]}}
                                 for tc in c.tool_calls]})
            for tc in c.tool_calls:
                used.append(tc["name"])
                try:
                    args = json.loads(tc["arguments"] or "{}")
                    result = self.dispatch(tc["name"], args)
                except Exception as exc:        # noqa: BLE001
                    result = {"error": str(exc)[:200]}
                for rid in self.route_ids_of(result):
                    retrieved.setdefault(rid, None)
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": self.tool_payload(result)})
        return self._terminal(messages, list(retrieved), tin, tout, trips, used, reason)

    def _terminal(self, messages: list[dict], retrieved: list[str], tin: int, tout: int,
                  trips: int, used: list[str], reason: str) -> Attempt:
        """ONE constrained request, then pick -> Plan -> SQL.

        The schema is built here and nowhere else, from the ids this question
        retrieved. `concepts=` names the ontology's concepts so the subject and
        attribute enums do not depend on retrieval: a question answered on the
        subject's own columns retrieves NO route -- 2 of the 5 probe questions
        were exactly that -- and the route enum `["SELF"]` is the correct
        contract for it. What retrieval governs is the ROUTES, and those are
        enumerated from the transcript alone.

        `submit_pick` is deliberately NOT appended to `Attempt.tool_calls`: that
        list is read as "which RETRIEVAL tools did the model choose", and one
        mandatory call the harness makes on every question is not a choice. The
        terminal turn is visible in `round_trips` and in `notes` instead.
        """
        base = {"input_tokens": tin, "output_tokens": tout, "round_trips": trips,
                "tool_calls": tuple(used)}
        note: dict[str, Any] = {"terminal_reason": reason,
                                "retrieved_routes": list(retrieved)}
        try:
            schema = self.skills.pick_schema(
                retrieved, concepts=tuple(sorted(self.skills.ontology.concepts)))
        except Exception as exc:                # noqa: BLE001
            return Attempt(sql="", error=f"schema refused: {type(exc).__name__}: {exc}"[:300],
                           notes=note, **base)
        note["schema_enum_values"] = _enum_values(schema)
        submit = [{"type": "function", "function": {
            "name": self.SUBMIT, "strict": True,
            "description": "Submit the governed pick. The SQL is compiled from it.",
            "parameters": schema}}]
        messages = [*messages, {"role": "user", "content": self.TERMINAL}]
        try:

            c = self.llm(messages, model=self.model, temperature=self.temperature,
                         tools=submit,
                         tool_choice={"type": "function",
                                      "function": {"name": self.SUBMIT}})
        except Exception as exc:                # noqa: BLE001
            return Attempt(sql="", error=f"schema refused: {type(exc).__name__}: {exc}"[:300],
                           notes=note, **base)
        base = {**base, "input_tokens": tin + c.input_tokens,
                "output_tokens": tout + c.output_tokens, "round_trips": trips + 1}
        note.update(structured_output=c.structured_output,
                    **sampling_note(self.temperature, c))
        submitted = next((tc for tc in c.tool_calls if tc["name"] == self.SUBMIT), None)
        if submitted is not None:
            note["pick_channel"] = "tool_call"
            return self._finish(submitted["arguments"] or "", base, note, retrieved)

        note["pick_channel"] = "text"
        return self._finish(c.text or "", base, note, retrieved)

    def _finish(self, raw: str, base: dict, note: dict,
                retrieved: Sequence[str] = ()) -> Attempt:
        """Pick -> Plan -> SQL. Every refusal is a DATUM, not a harness failure."""
        try:
            pick, fenced = parse_json_object(raw)
        except ValueError as exc:
            return Attempt(sql="", error=f"pick was not JSON: {exc}"[:300],
                           notes={**note, "pick_raw": raw[:600]}, **base)

        if fenced:
            note["fence_extracted"] = True

        stray = [rid for rid in self.skills.route_ids_in(pick) if rid not in set(retrieved)]
        if stray:
            return Attempt(
                sql="",
                error=(f"pick rejected: UnretrievedRouteError: {stray[0]!r} is a governed "
                       f"route this question never retrieved "
                       f"({len(retrieved)} were)")[:300],
                notes={**note, "pick": pick, "unretrieved_routes": stray}, **base)
        try:
            plan = self.skills.parse_pick(pick)
        except Exception as exc:                # noqa: BLE001
            return Attempt(sql="", error=f"pick rejected: {type(exc).__name__}: {exc}"[:300],
                           notes={**note, "pick": pick}, **base)
        try:
            sql = self.compile_module.compile(plan, self.skills.ontology, self.skills.graph)
        except Exception as exc:                # noqa: BLE001
            return Attempt(sql="", error=f"compile refused: {type(exc).__name__}: {exc}"[:300],
                           notes={**note, "pick": pick}, **base)
        return certify(sql, self.checker, base=base, notes={**note, "pick": pick})

class StagedPlannerStrategy:
    """`skills/planner/v1`, executed. Four stages, TWO model calls.

    WHAT IT REPLACES. `RetrievedMenuStrategy` was built to retrieve and, measured
    over the 2026-08-12 campaign, did not: `list_concepts` was its first call on
    215 of 215 runs and `search_concepts` was used on 5. The whole ontology went
    into context and the model linked entities there, at a cost linear in
    ontology size and paid per question. Its 51.2% was obtained WITH that oracle.

    THE PROCEDURE IS NOT IN THIS CLASS. It is `skills/planner/v1/SKILL.md`, whose
    markdown body is passed verbatim as the system prompt and whose
    `decompose.schema.json` is stage 1's contract. This class sequences the four
    stages and owns no instruction of its own -- so a change to how the planner
    thinks is a new directory, and `--skill-version 2` runs it beside this one.

      1 decompose   ONE call, forced against the skill's schema: the subjects the
                    question names, plus verbatim spans for six other slots.
      2 resolve     NO call. Every span goes through the deterministic resolvers
                    -- `find_metric`, `find_attribute`, `search_values`, and
                    `find_paths` over each ordered pair of subjects.
      3 pick        ONE call, over `pick_schema` built from what stage 2 actually
                    resolved. The route enum IS the retrieved set.
      4 certify     NO call. parse_pick -> compile -> check, and stop.

    WHY STAGE 4 DOES NOT EXECUTE. The skill originally ended in compile -> check
    -> EXECUTE -> one revision. Execution feedback is measured at +23.0/+24.9 for
    a frozen model (Chen et al., ACL 2024, arXiv:2402.10890 Table 2) -- larger
    than the entire effect this study is trying to detect -- so giving it to this
    arm and not to the DDL floor would confound staged retrieval with a mechanism
    worth more than the thing being measured. Stage 2's `search_values` is NOT
    the same mechanism and is kept: it reads the database for a literal's
    canonical spelling and tells the model nothing about whether its answer is
    right. `ACME_floor_values.ddl` gives the floor the matching capability.

    WHY IT MAY SCORE LOWER THAN THE MENU ARM, and that being the point: 51.2% was
    a model entity-linking over a dumped ontology; this is retrieval. Different
    instruments. A drop is the price of the claim being true.
    """

    name = "staged"

    DECOMPOSE = "submit_decomposition"
    SUBMIT = "submit_pick"

    ENGINE_RESOLVERS = frozenset({"list_subjects"})

    def __init__(self, model: str, *, llm: Callable[..., Completion] = complete,
                 temperature: float = 0.0, skills_module: Any = None,
                 compile_module: Any = None, db: str | Path = DB,
                 skill_version: int = 1,
                 checker: Callable[[str], list[Any]] | None = None) -> None:
        self.model, self.llm, self.temperature = model, llm, temperature
        self.db = db
        self._checker = checker
        self.tools, self.dispatch, self.skills = load_pick_skills(skills_module)
        try:
            if compile_module is None:
                from spc import compile as compile_module  # type: ignore[no-redef]
        except Exception as exc:                # noqa: BLE001
            raise MenuUnavailable(f"spc/compile.py not importable: {exc}") from exc
        self.compile_module = compile_module

        from spc import skillfile
        self.skill = skillfile.load("planner", skill_version)

        declared = set(self.skill.meta.get("resolvers") or ())
        if not declared:
            raise skillfile.SkillFileError(
                f"{self.skill.path.name}: frontmatter has no `resolvers:` -- a skill "
                f"that does not name what runs on its behalf cannot be cited as the "
                f"procedure that ran"
            )

        invocable = {
            name for name in dir(self.skills)
            if not name.startswith("_") and callable(getattr(self.skills, name, None))
        }
        if declared - invocable:
            raise skillfile.SkillFileError(
                f"{self.skill.path.name}: declares resolvers the engine cannot invoke: "
                f"{sorted(declared - invocable)}"
            )

        for key, entry in (("resolve", "resolve"), ("narrow", "narrow")):
            if not callable(getattr(self.skill.resource(key), entry, None)):
                raise skillfile.SkillFileError(
                    f"{self.skill.path.name}: `{key}:` names a script with no "
                    f"`{entry}(...)` entry point"
                )
        step = self.skill.resource("resolve")
        invoked = set(getattr(step, "RESOLVERS", ())) | self.ENGINE_RESOLVERS
        if invoked - declared:
            raise skillfile.SkillFileError(
                f"{self.skill.path.name}: its steps invoke {sorted(invoked - declared)}, "
                f"which the skill does not declare in `resolvers:`"
            )

        if not str(self.skill.meta.get("pick_turn") or "").strip():
            raise skillfile.SkillFileError(
                f"{self.skill.path.name}: frontmatter missing 'pick_turn' -- stage 3's "
                f"user turn is instruction text and belongs in the versioned artifact, "
                f"not in the engine"
            )

    @property
    def checker(self) -> Callable[[str], list[Any]]:
        if self._checker is None:
            self._checker = load_checker(self.db)
        return self._checker

    def __call__(self, q: Question, run: int = 0) -> Attempt:
        base = {"input_tokens": 0, "output_tokens": 0, "round_trips": 0}
        note: dict = {"skill": f"{self.skill.name} v{self.skill.version}",
                      "stages": []}

        with trace.span("decompose", question=q.prompt, qid=q.qid):
            decomp, base, note, failed = self._decompose(q, base, note)
        if failed is not None:
            return failed

        with trace.span("resolve", subjects=decomp.get("subjects")):
            resolved = self._resolve(decomp, q)
        note["resolved"] = resolved["report"]
        note["stages"].append("resolve")

        if not resolved["subjects"]:
            return Attempt(
                sql="",
                error=("abstained: no subject -- "
                       f"{resolved['abstain_reason']}")[:300],
                notes={**note, "decomposition": decomp}, **base)

        with trace.span("pick", routes=len(resolved["route_ids"])):
            return self._pick(q, decomp, resolved, base, note)

    def _decompose(self, q: Question, base: dict, note: dict):
        """ONE call, forced against `decompose.schema.json`.

        `list_subjects` is called FIRST and deterministically -- the skill says to
        call it, but leaving that to the model is how the menu arm ended up
        dumping the ontology on every run. The one decision that stays the
        model's is which subjects the question names.
        """
        subjects = self.skills.list_subjects(q.prompt)
        menu = subjects.data if hasattr(subjects, "data") else subjects
        note["stages"].append("decompose")
        note["subject_menu"] = [s["subject"] for s in menu.get("subjects", ())]
        note["undocumented_subjects"] = menu.get("undocumented_subjects")

        note["resolver_calls"] = {"list_subjects": 1}

        tool = [{"type": "function", "function": {
            "name": self.DECOMPOSE,
            "description": "The stage-1 decomposition.",
            "parameters": self.skill.resource("schema")}}]
        messages = [
            {"role": "system", "content": self.skill.prompt},
            {"role": "user", "content":
                f"{json.dumps(menu, ensure_ascii=False)}\n\nQuestion: {q.prompt}"},
        ]
        try:
            c = self.llm(messages, model=self.model, temperature=self.temperature,
                         tools=tool,
                         tool_choice={"type": "function",
                                      "function": {"name": self.DECOMPOSE}})
        except Exception as exc:                # noqa: BLE001
            return {}, base, note, Attempt(
                sql="", error=f"schema refused: {type(exc).__name__}: {exc}"[:300],
                notes=note, **base)
        base = {"input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                "round_trips": 1}
        note.update(**sampling_note(self.temperature, c))

        call = next((tc for tc in c.tool_calls if tc["name"] == self.DECOMPOSE), None)
        raw = (call["arguments"] or "") if call else (c.text or "")
        note["decompose_channel"] = "tool_call" if call else "text"
        try:
            decomp, _ = parse_json_object(raw)
        except ValueError as exc:
            return {}, base, note, Attempt(
                sql="", error=f"pick was not JSON: stage 1: {exc}"[:300],
                notes={**note, "decompose_raw": raw[:600]}, **base)
        note["decomposition"] = decomp
        return decomp, base, note, None

    def _resolve(self, decomp: Mapping[str, Any], q: Question) -> dict:
        """Stage 2, which lives in the SKILL and not here.

        The engine's whole knowledge of this stage is the entry point it calls:
        `resolve(decomposition, question, skills) -> dict`. Everything the step
        does -- which resolvers, in what order, under what condition, and what it
        reports -- is `skills/planner/v1/resolve.py`, versioned with the artifact
        that declares it.

        It was 200 lines in this file until 2026-08-13. That was the bug behind
        every mismatch found that day: the skill DESCRIBED a procedure the engine
        HELD, so the two could drift, and did, in four separate places. A step in
        the artifact cannot drift from itself.
        """
        return self.skill.resource("resolve").resolve(decomp, q.prompt, self.skills)

    def _narrow(self, schema: dict, resolved: Mapping[str, Any],
                decomp: Mapping[str, Any]) -> dict:
        """Stage 3's schema rule, which lives in the SKILL and not here.

        The engine's whole knowledge of it is the entry point:
        `narrow(schema, resolved, decomposition) -> dict`. It was 85 lines in
        this file, three lines below a docstring reading "THE PROCEDURE IS NOT IN
        THIS CLASS" -- a contradiction a different-model review called on
        2026-08-13. Deciding which slots stage 3 may fill is a rule OF stage 3.
        """
        return self.skill.resource("narrow").narrow(schema, resolved, decomp)

    def _pick(self, q: Question, decomp: Mapping[str, Any], resolved: dict,
              base: dict, note: dict) -> Attempt:
        """ONE call over what stage 2 resolved, then compile and check. No rows."""
        note["stages"].append("pick")
        subject = resolved["subjects"][0]
        retrieved = list(dict.fromkeys(resolved["route_ids"]))
        try:
            schema = self._narrow(
                self.skills.pick_schema(
                    retrieved, subject=subject, metrics=resolved["metrics"] or None,
                    concepts=resolved["concepts"]),
                resolved, decomp)
        except Exception as exc:                # noqa: BLE001
            return Attempt(sql="", error=f"schema refused: {type(exc).__name__}: {exc}"[:300],
                           notes=note, **base)
        note["schema_offers"] = sorted(schema.get("properties", {}))

        tool = [{"type": "function", "function": {
            "name": self.SUBMIT, "description": "The governed pick.",
            "parameters": schema}}]
        messages = [
            {"role": "system", "content": self.skill.prompt},
            {"role": "user", "content": f"Question: {q.prompt}"},
            {"role": "assistant", "content":
                f"Decomposition: {json.dumps(decomp, ensure_ascii=False)}"},
            {"role": "user", "content":
                f"Stage 2 resolved:\n{json.dumps(resolved['report'], ensure_ascii=False)}"
                f"\n\n{self.skill.meta['pick_turn']}"},
        ]
        try:
            c = self.llm(messages, model=self.model, temperature=self.temperature,
                         tools=tool,
                         tool_choice={"type": "function",
                                      "function": {"name": self.SUBMIT}})
        except Exception as exc:                # noqa: BLE001
            return Attempt(sql="", error=f"schema refused: {type(exc).__name__}: {exc}"[:300],
                           notes=note, **base)
        base = {"input_tokens": base["input_tokens"] + c.input_tokens,
                "output_tokens": base["output_tokens"] + c.output_tokens,
                "round_trips": base["round_trips"] + 1}
        note.update(structured_output=c.structured_output,
                    **sampling_note(self.temperature, c))

        call = next((tc for tc in c.tool_calls if tc["name"] == self.SUBMIT), None)
        raw = (call["arguments"] or "") if call else (c.text or "")
        note["pick_channel"] = "tool_call" if call else "text"
        try:
            pick, fenced = parse_json_object(raw)
        except ValueError as exc:
            return Attempt(sql="", error=f"pick was not JSON: {exc}"[:300],
                           notes={**note, "pick_raw": raw[:600]}, **base)
        if fenced:
            note["fence_extracted"] = True

        stray = [r for r in self.skills.route_ids_in(pick) if r not in set(retrieved)]
        if stray:
            return Attempt(
                sql="",
                error=(f"pick rejected: UnretrievedRouteError: {stray[0]!r} is a "
                       f"governed route this question never retrieved "
                       f"({len(retrieved)} were)")[:300],
                notes={**note, "pick": pick, "unretrieved_routes": stray}, **base)
        try:
            plan = self.skills.parse_pick(pick)
        except Exception as exc:                # noqa: BLE001
            return Attempt(sql="", error=f"pick rejected: {type(exc).__name__}: {exc}"[:300],
                           notes={**note, "pick": pick}, **base)
        try:
            sql = self.compile_module.compile(plan, self.skills.ontology,
                                              self.skills.graph)
        except Exception as exc:                # noqa: BLE001
            return Attempt(sql="", error=f"compile refused: {type(exc).__name__}: {exc}"[:300],
                           notes={**note, "pick": pick}, **base)
        note["stages"].append("certify")
        return certify(sql, self.checker, base=base, notes={**note, "pick": pick})

class AgentSkillStrategy:
    """One agent running a workflow declared in `skills/sql_agent/v1`.

    THE DIFFERENCE FROM EVERY OTHER ARM, and it is the only variable: the agent
    ORCHESTRATES. `staged` has no tools and a sequence hardcoded in this file;
    `menu_tools` has six retrieval tools and no structure at all, which measured
    as `list_concepts` first on 215 of 215 runs. Here the agent has tools, each
    deterministic, each ending a declared step, and the skill says which step
    runs when, what is compulsory, what is conditional, and what each returns.

    WHAT THE AGENT CANNOT DO, unchanged from the other governed arms: write SQL,
    name a table, choose a join, or reach a route that was not retrieved. Every
    tool below is deterministic and the last one compiles and checks. Agency is
    added ABOVE the guardrail, not through it.

    THE WORKFLOW IS THE ARTIFACT'S. This class reads `skill.meta["workflow"]` to
    build its tool surface and holds no step name of its own beyond the entry
    points it must call. A v2 that adds a step is a directory.

    COST: unbounded in principle, capped in practice by `max_rounds` and by the
    skill's own `critique.cap`. Round trips are recorded per run, because the
    2-call bound is exactly what this arm gives up and the number is the price.
    """

    name = "agent_skill"

    ENGINE_RESOLVERS = frozenset({"list_subjects"})

    def __init__(self, model: str, *, llm: Callable[..., Completion] = complete,
                 temperature: float = 0.0, skills_module: Any = None,
                 compile_module: Any = None, db: str | Path = DB,
                 skill_version: int = 1, agent_version: int = 1,
                 critic_model: str = "", max_rounds: int = 8,
                 checker: Callable[[str], list[Any]] | None = None) -> None:
        self.model, self.llm, self.temperature = model, llm, temperature
        self.db, self._checker, self.max_rounds = db, checker, max_rounds
        self.tools, self.dispatch, self.skills = load_pick_skills(skills_module)
        try:
            if compile_module is None:
                from spc import compile as compile_module  # type: ignore[no-redef]
        except Exception as exc:                # noqa: BLE001
            raise MenuUnavailable(f"spc/compile.py not importable: {exc}") from exc
        self.compile_module = compile_module

        from spc import skillfile

        self.agent = skillfile.load_agent("sql_analyst", agent_version)
        self.skill = skillfile.load("sql_agent", skill_version)
        declared_skills = dict(self.agent.meta.get("skills") or {})
        if "sql_workflow" not in declared_skills:
            raise skillfile.SkillFileError(
                f"{self.agent.path.name}: does not declare the skill this arm runs; "
                f"declares {sorted(declared_skills)}")

        self.critic_model = critic_model or "gemini-3.5-lite"
        self.spec = dict(self.skill.meta.get("metadata") or {})
        self.workflow = list(self.spec.get("workflow") or ())
        if not self.workflow:
            raise skillfile.SkillFileError(
                f"{self.skill.path.name}: no `workflow:` -- this arm's whole "
                f"premise is that the sequence lives in the artifact"
            )
        for key, entry in (("resolve", "resolve"), ("narrow", "narrow")):
            if not callable(getattr(self.skill.resource(key), entry, None)):
                raise skillfile.SkillFileError(
                    f"{self.skill.path.name}: `{key}:` names a script with no "
                    f"`{entry}(...)` entry point")

        self._validate_registries()

        invocable = {n for n in dir(self.skills)
                     if not n.startswith("_") and callable(getattr(self.skills, n, None))}
        invoked = set(self.skill.resource("resolve").RESOLVERS) | self.ENGINE_RESOLVERS
        if invoked - invocable:
            raise skillfile.SkillFileError(
                f"{self.skill.path.name}: steps invoke {sorted(invoked - invocable)}, "
                f"which the engine cannot call")

    @property
    def checker(self) -> Callable[[str], list[Any]]:
        if self._checker is None:
            self._checker = load_checker(self.db)
        return self._checker

    def _step(self, name: str) -> dict:
        """A declared step, by name. Raises if the artifact does not declare it,
        so a tool this class offers that the workflow never mentions is a load
        error rather than an undeclared capability."""
        for step in self.workflow:
            if step.get("step") == name:
                return step
        raise KeyError(f"{self.skill.path.name} declares no step {name!r}")

    def _validate_registries(self) -> None:
        """Every `runs:` points at a declaration, and every declaration is used."""
        from spc import skillfile  # noqa: PLC0415

        tools = dict(self.spec.get("tools") or {})
        scripts = dict(self.spec.get("scripts") or {})
        referenced: set[str] = set()
        for st in self.workflow:
            runs = str(st.get("runs") or "")
            if runs.startswith("tool:"):
                target = runs.split(":", 1)[1]
                if target not in tools:
                    raise skillfile.SkillFileError(
                        f"{self.skill.path.name}: step {st['step']!r} runs "
                        f"tool:{target}, which `tools:` does not declare")
                referenced.add(target)
            elif runs.endswith(".py"):
                stem = runs.rsplit("/", 1)[-1][:-3]
                if stem not in scripts:
                    raise skillfile.SkillFileError(
                        f"{self.skill.path.name}: step {st['step']!r} runs {runs}, "
                        f"which `scripts:` does not declare")
        unused = sorted(set(tools) - referenced)
        if unused:
            raise skillfile.SkillFileError(
                f"{self.skill.path.name}: declares tools no step runs: {unused} -- "
                f"a tool the agent is offered but the workflow never accounts for "
                f"is a capability outside the procedure")
        for name, spec in tools.items():
            source = str(spec.get("schema") or "")
            if not (source.startswith(("inline:", "script:")) or source in self.skill.resources):
                raise skillfile.SkillFileError(
                    f"{self.skill.path.name}: tool {name!r} declares schema "
                    f"{source!r}, which is neither a loaded resource nor "
                    f"`inline:`/`script:`")

    def _tool_schema(self, name: str, spec: Mapping[str, Any],
                     resolved: dict | None) -> dict | None:
        """A declared tool's parameter schema, resolved from `schema:`.

        Three sources, and the artifact says which: a loaded resource key
        (`schema` -> decompose.schema.json), a script that builds one per
        question (`script:narrow`), or a tiny inline shape the engine knows
        (`inline:reason`). Anything else is a load error rather than a tool that
        silently never appears.
        """
        source = str(spec.get("schema") or "")
        if source == "inline:reason":
            return {"type": "object", "additionalProperties": False,
                    "required": ["reason"],
                    "properties": {"reason": {"type": "string"}}}
        if source.startswith("script:"):
            if not (resolved and resolved.get("subjects")):
                return None
            script = self.skill.resource(source.split(":", 1)[1])
            return script.narrow(
                self.skills.pick_schema(
                    list(dict.fromkeys(resolved["route_ids"])),
                    subject=resolved["subjects"][0],
                    metrics=resolved["metrics"] or None,
                    concepts=resolved["concepts"]),
                resolved, resolved.get("decomposition") or {})
        return self.skill.resource(source)

    def _agent_tools(self, resolved: dict | None) -> list[dict]:
        """The agent's tool surface, built FROM the artifact's `tools:` registry.

        It was three hardcoded literals here, which meant the workflow's
        `runs: tool:pick` pointed at a name only the engine could define -- the
        artifact declared a step it could not describe, and a v2 could not add a
        tool without an engine edit. Now the name, the description the agent
        sees, and the schema source all come from the declaration.

        `available:` decides existence, not advice. A tool whose schema is built
        from what resolved cannot exist before resolving, so ordering is enforced
        by the surface rather than asked for in prose.
        """
        tools = []
        for name, spec in (self.spec.get("tools") or {}).items():
            schema = self._tool_schema(name, spec, resolved)
            if schema is None:
                continue
            tools.append({"type": "function", "function": {
                "name": name, "description": str(spec.get("for") or ""),
                "parameters": schema}})
        return tools

    def __call__(self, q: Question, run: int = 0) -> Attempt:
        base = {"input_tokens": 0, "output_tokens": 0, "round_trips": 0}
        note: dict = {"skill": f"{self.skill.name} v{self.skill.version}",
                      "workflow": [s["step"] for s in self.workflow],
                      "called": [], "deviations": []}

        menu = self.skills.list_subjects(q.prompt)
        menu = menu.data if hasattr(menu, "data") else menu
        note["resolver_calls"] = {"list_subjects": 1}
        note["undocumented_subjects"] = menu.get("undocumented_subjects")

        system = (f"{self.agent.prompt}\n\n"
                  f"---\n\n"
                  f"# Loaded skill: sql_workflow "
                  f"({self.skill.name} v{self.skill.version})\n\n"
                  f"{self.skill.prompt}")
        note["agent"] = f"{self.agent.name} v{self.agent.version}"
        note["loaded_skill"] = "sql_workflow"
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content":
                f"{json.dumps(menu, ensure_ascii=False)}\n\nQuestion: {q.prompt}"},
        ]
        resolved: dict | None = None
        cap = int(self._step("review").get("cap") or 0)
        repairs = 0

        for _ in range(self.max_rounds):
            try:
                c = self.llm(messages, model=self.model, temperature=self.temperature,
                             tools=self._agent_tools(resolved))
            except Exception as exc:            # noqa: BLE001
                return Attempt(sql="", error=f"schema refused: {type(exc).__name__}: {exc}"[:300],
                               notes=note, **base)
            base = {"input_tokens": base["input_tokens"] + c.input_tokens,
                    "output_tokens": base["output_tokens"] + c.output_tokens,
                    "round_trips": base["round_trips"] + 1}
            note.update(**sampling_note(self.temperature, c))

            if not c.tool_calls:

                note["deviations"].append({"round": base["round_trips"],
                                           "why": "no tool call", "text": (c.text or "")[:160]})
                messages.append({"role": "assistant", "content": c.text or ""})
                continue

            call = c.tool_calls[0]
            name = call["name"]
            note["called"].append(name)
            try:
                args, _ = parse_json_object(call["arguments"] or "{}")
            except ValueError as exc:
                return Attempt(sql="", error=f"pick was not JSON: {name}: {exc}"[:300],
                               notes=note, **base)

            if name == "abstain":
                return Attempt(sql="", notes=note, **base,
                               error=f"abstained: {args.get('reason', 'out of scope')}"[:300])

            if name == "decompose":
                resolved = self.skill.resource("resolve").resolve(args, q.prompt, self.skills)
                resolved["decomposition"] = args
                note["decomposition"], note["resolved"] = args, resolved["report"]
                if not resolved["subjects"]:

                    return Attempt(sql="", notes=note, **base,
                                   error=f"abstained: no subject -- {resolved['abstain_reason']}"[:300])
                messages += [
                    {"role": "assistant", "content": f"decompose: {json.dumps(args)}"},
                    {"role": "user", "content":
                        f"resolve returned:\n{json.dumps(resolved['report'], ensure_ascii=False)}"},
                ]
                continue

            if name == "pick":
                if resolved is None:
                    note["deviations"].append({"round": base["round_trips"],
                                               "why": "pick before resolve"})
                    continue
                attempt = self._certify(args, resolved, base, note)

                if repairs < cap:
                    verdict, spent = self._review(q, args, attempt, resolved)
                    base = {"input_tokens": base["input_tokens"] + spent[0],
                            "output_tokens": base["output_tokens"] + spent[1],
                            "round_trips": base["round_trips"] + 1}
                    note.setdefault("reviews", []).append(
                        {"verdict": verdict["verdict"], "why": verdict["why"][:160],
                         "refusal": (attempt.error or "")[:100]})
                    if verdict["verdict"] == "pass":

                        note["review"] = "pass"
                        return replace(attempt, notes={**attempt.notes,
                                                       "review": "pass",
                                                       "reviews": note["reviews"]}, **base)

                    repairs += 1
                    messages += [
                        {"role": "assistant", "content": f"pick: {json.dumps(args)}"},
                        {"role": "user", "content":
                            (f"certify refused it: {attempt.error}\n\n" if attempt.error else "")
                            + f"A reviewer failed this plan: {verdict['why']}\n\n"
                            f"Submit a corrected pick."},
                    ]
                    continue

                note["review"] = f"cap reached ({cap})"
                return replace(attempt, notes={**attempt.notes,
                                               "review": note["review"],
                                               "reviews": note.get("reviews", [])}, **base)

            note["deviations"].append({"round": base["round_trips"],
                                       "why": f"undeclared tool {name!r}"})

        return Attempt(sql="", error="tool loop did not terminate"[:300],
                       notes=note, **base)

    CRITIC_SYSTEM = (
        "You review a query PLAN against the question it is meant to answer. You "
        "never write SQL, never name a table, and never see the data or any rows.\n\n"
        "Fail the plan only for a reason you can point at:\n"
        "  - the grain is wrong -- it counts or groups the wrong thing\n"
        "  - the quantity is not the one the question named\n"
        "  - a filter or grouping the question asked for is missing, or one it "
        "did not ask for is present\n"
        "  - the route reaches the right concept by a relationship the question "
        "did not mean\n\n"
        "Pass it otherwise. A plan you merely would have written differently is a "
        "PASS. If a checker refusal is shown, say what about the plan caused it.\n\n"
        "Answer as JSON and nothing else: "
        '{"verdict": "pass" | "fail", "why": "<one or two sentences>"}'
    )

    def _review(self, q: Question, pick: Mapping[str, Any], attempt: Attempt,
                resolved: dict) -> tuple[dict, tuple[int, int]]:
        """One turn of AGENT 2. Returns its verdict and what it cost.

        IT SEES THE QUESTION, THE PLAN AND ANY REFUSAL -- NEVER THE ROWS. Showing
        a model its own answer's rows and letting it revise is measured at
        +23.0/+24.9 for a frozen model (Chen et al., ACL 2024, arXiv:2402.10890
        Table 2), larger than the effect this study is trying to detect and held
        by this arm alone. Reviewing a PLAN against a QUESTION is a different
        mechanism: it is self-consistency, and it carries no result data.

        A FAILURE HERE PASSES. The reviewer is an aid, not a gate: if the second
        agent errors or answers unreadably, the run keeps the plan it had rather
        than discarding a certified answer because a critic was unavailable.
        """
        shown = {"question": q.prompt, "plan": pick,
                 "resolved": {k: resolved["report"].get(k)
                              for k in ("subjects", "metrics", "attributes", "routes")}}
        if attempt.error:
            shown["checker_refused"] = attempt.error
        try:
            c = self.llm([{"role": "system", "content": self.CRITIC_SYSTEM},
                          {"role": "user",
                           "content": json.dumps(shown, ensure_ascii=False)[:4000]}],
                         model=self.critic_model, temperature=self.temperature)
        except Exception as exc:                # noqa: BLE001
            return {"verdict": "pass",
                    "why": f"(reviewer unavailable: {type(exc).__name__})"}, (0, 0)
        spent = (c.input_tokens, c.output_tokens)
        try:
            got, _ = parse_json_object(c.text or "")
        except ValueError:
            return {"verdict": "pass",
                    "why": f"(reviewer unreadable: {(c.text or '')[:80]})"}, spent
        verdict = str(got.get("verdict", "pass")).lower()
        return ({"verdict": "fail" if verdict == "fail" else "pass",
                 "why": str(got.get("why", ""))[:400]}, spent)

    def _certify(self, pick: Mapping[str, Any], resolved: dict,
                 base: dict, note: dict) -> Attempt:
        """pick -> Plan -> SQL -> check. Deterministic, and the agent's last word."""
        retrieved = set(resolved["route_ids"])
        stray = [r for r in self.skills.route_ids_in(pick) if r not in retrieved]
        if stray:
            return Attempt(sql="", notes={**note, "pick": pick}, **base,
                           error=(f"pick rejected: UnretrievedRouteError: {stray[0]!r} is a "
                                  f"governed route this question never retrieved")[:300])
        try:
            plan = self.skills.parse_pick(pick)
        except Exception as exc:                # noqa: BLE001
            return Attempt(sql="", error=f"pick rejected: {type(exc).__name__}: {exc}"[:300],
                           notes={**note, "pick": pick}, **base)
        try:
            sql = self.compile_module.compile(plan, self.skills.ontology, self.skills.graph)
        except Exception as exc:                # noqa: BLE001
            return Attempt(sql="", error=f"compile refused: {type(exc).__name__}: {exc}"[:300],
                           notes={**note, "pick": pick}, **base)
        return certify(sql, self.checker, base=base, notes={**note, "pick": pick})

def build_strategy(spec: str, *, model: str = "", llm: Callable[..., Completion] = complete,
                   db: str | Path = DB, temperature: float = 0.0,
                   ddl: str | Path = DDL) -> Any:
    """Name -> strategy. The CLI's whole vocabulary, in one place.

    `temperature` is threaded rather than defaulted per class: the study measures
    at T=0 AND T=1, and before this it was reachable only by constructing a
    strategy by hand -- the CLI could not ask for it at all, so "we measure at
    T=1" was not a thing the harness could do.
    """
    kw = {"llm": llm, "temperature": temperature}
    if spec.endswith("_checked") and spec not in ("skills_checked",):
        inner = build_strategy(spec[: -len("_checked")], model=model, llm=llm, db=db,
                               temperature=temperature, ddl=ddl)
        return CheckedStrategy(inner, model=model or None, llm=llm, db=db, name=spec,
                               temperature=temperature)
    if spec.startswith("stub_"):
        return StubStrategy(spec[len("stub_"):], name=spec)
    if spec == "ddl":
        return DdlStrategy(model, ddl=ddl, **kw)
    if spec == "context":
        return ContextStrategy(model, **kw)
    if spec == "skills":
        return SkillsStrategy(model, **kw)

    if spec == "menu":
        return MenuStrategy(model, db=db, **kw)
    if spec == "menu_tools":
        return RetrievedMenuStrategy(model, db=db, **kw)

    if spec == "staged" or (spec.startswith("staged_v") and spec[8:].isdigit()):
        return StagedPlannerStrategy(model, db=db,
                                     skill_version=int(spec[8:]) if spec[8:] else 1,
                                     **kw)
    if spec == "agent_skill" or (spec.startswith("agent_skill_v") and spec[13:].isdigit()):
        return AgentSkillStrategy(model, db=db,
                                  skill_version=int(spec[13:]) if spec[13:] else 1, **kw)
    if spec == "skills_checked":
        return CheckedStrategy(SkillsStrategy(model, **kw), model=model, llm=llm,
                               db=db, name="skills_checked", temperature=temperature)
    raise ValueError(f"unknown strategy {spec!r}")

REFUSAL_CLASSES: tuple[tuple[str, str], ...] = (

    ("abstained:", "ABSTAINED"),
    ("pick was not JSON", "NON_JSON_PICK"),
    ("pick rejected", "OFF_MENU_PICK"),
    ("compile refused", "COMPILE_REFUSED"),

    ("sql was not readable", "UNREADABLE_SQL"),

    ("tool loop did not terminate", "TOOL_LOOP_EXHAUSTED"),

    ("schema refused", "SCHEMA_REFUSED"),

    ("checker rejected", "CHECKER_VIOLATION"),
    ("checker undecidable", "CHECKER_UNDECIDABLE"),
)

_CHECK_CODES = re.compile(r"^checker (?:rejected|undecidable):\s*([A-Z_, ]+?)\s*(?:\[|$)")

_EXC_IN_ERROR = re.compile(r"^[^:]+:\s*([A-Z]\w+(?:Error|Exception|Refused))\b")

def refusal_class(error: str | None) -> str:
    """The COARSE reason an attempt produced no SQL. Reported, not just keyed.

    `None` is not "no failure": a strategy can return an empty completion with
    nothing recorded against it, and that is its own failure mode (the model
    replied with whitespace), distinct from a refusal it explained.
    """
    text = (error or "").strip()
    if not text:
        return "NO_OUTPUT"
    for prefix, label in REFUSAL_CLASSES:
        if text.startswith(prefix):
            codes = _CHECK_CODES.match(text)
            if codes:
                return f"{label}:{'+'.join(codes.group(1).split(', '))}"
            match = _EXC_IN_ERROR.match(text)
            return f"{label}:{match.group(1)}" if match else label

    head = text.split(":", 1)[0].strip()
    return f"PROVIDER_ERROR:{head}" if head and " " not in head else "PROVIDER_ERROR"

def refusal_key(error: str | None) -> str:
    """The FULL identity of a refusal -- class plus a digest of its detail.

    Why the detail and not just the class. `refusal_class` groups for reporting;
    it must not be what TARa@N counts, because two runs whose replies were not
    JSON emitted DIFFERENT bytes and reproduced nothing. That is the same
    argument that put the SQL into the `ERROR:` key, applied one branch over.
    A refusal that really is stable -- the same compile refusal on the same plan
    every run -- still digests identically and still scores 1.0.
    """
    text = (error or "").strip()
    return (f"{refusal_class(text)}:"
            + hashlib.sha256(text.encode()).hexdigest()[:12])

def text_key(sql: str, error: str | None = None) -> str:
    """Canonical identity of what a run PRODUCED -- for TARr@N.

    `pred_sql` was used directly, which is right whenever there is SQL and wrong
    whenever there is not: every refusal is `""`, so N runs that failed for N
    different reasons read as one byte-identical answer. Same defect as the
    result key, same fix, and it has to be the same fix or the two TAR figures
    disagree about what a refusal is.
    """
    if sql.strip():
        return "SQL:" + sql
    return "REFUSED:" + refusal_key(error)

class Scorer:
    """Execution-based correctness, with gold results cached per question."""

    def __init__(self, db: str | Path = DB) -> None:
        self.con = sqlite3.connect(str(db))
        self._gold: dict[str, list] = {}
        self._recast: dict[str, list | None] = {}

    def gold_rows(self, q: Question) -> list:
        if q.qid not in self._gold:
            self._gold[q.qid] = self.con.execute(q.gold_sql).fetchall()
        return self._gold[q.qid]

    def result_key(self, sql: str, error: str | None = None) -> str:
        """Canonical identity of a query's RESULT -- for TARa@N.

        A query that RAISES has no result set, so it must not be able to "agree"
        with another failing query. Keying the failure on the exception class
        alone did exactly that: two runs hallucinating two different column names
        both raise `OperationalError`, and the question was counted as fully
        result-stable while nothing was ever computed -- a falsely good number on
        the study's primary metric. The key therefore includes the SQL, so only
        byte-identical failures agree (and those already agree on TARr).

        The EMPTY branch had the identical defect and kept it two fixes longer.
        Every refusal returns `Attempt(sql="")`, so `pred_sql` and the result key
        were both `""` no matter WHY the run failed, and a question whose N runs
        all refused scored TARr@N = TARa@N = 1.0 -- flawless reproducibility for
        having computed nothing. Worse, it was not evenly distributed: the arms
        that refuse BY DESIGN are the two menu arms, which are the arms this
        study is about, so the inflation landed on the primary outcome. An empty
        result is therefore keyed on the refusal REASON, and `summarize` reports
        the refusal rate beside every TAR figure because a TAR number without one
        is not interpretable.
        """
        if not sql.strip():
            return "EMPTY:" + refusal_key(error)
        try:
            rows = self.con.execute(sql).fetchall()
        except Exception as exc:              # noqa: BLE001
            return "ERROR:" + hashlib.sha256(
                f"{type(exc).__name__}:{exc}:{sql}".encode()).hexdigest()[:12]
        return "ROWS:" + hashlib.sha256(
            json.dumps(canon(rows), default=str).encode()).hexdigest()[:12]

    def score(self, q: Question, sql: str) -> str:
        """"correct" | "defect_corrected" | "wrong". Mirrors evaluation/accuracy."""
        if not sql.strip():
            return "wrong"
        if correct(q.gold_sql, sql, self.con):
            return "correct"

        if q.qid not in self._recast:
            try:
                rg = recast_gold(q.gold_sql)
                self._recast[q.qid] = None if rg == q.gold_sql else [rg]
            except Exception:                 # noqa: BLE001
                self._recast[q.qid] = None
        rg = self._recast[q.qid]
        if rg and correct(rg[0], sql, self.con):
            return "defect_corrected"
        return "wrong"

def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10)
        return out.stdout.strip()[:12] + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:                          # noqa: BLE001
        return "unknown"

def run_bench(
    questions: Sequence[Question],
    strategies: Sequence[Any],
    *,
    n: int = 20,
    db: str | Path = DB,
    out: str | Path | None = None,
    model: str = "",
    scorer: Scorer | None = None,
    progress: bool = False,
) -> dict:
    """Ask every question N times of every strategy. Returns the summary.

    Provenance is written as the FIRST line of the output file. A result without
    it cannot be compared to anything later, which makes it worthless -- so it is
    not optional and not a footnote.
    """
    scorer = scorer or Scorer(db)
    meta = {
        "kind": "run_meta",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": model,
        "n": n,

        "temperature": {getattr(s, "name", type(s).__name__):
                        getattr(s, "temperature", None) for s in strategies},
        "strategies": [getattr(s, "name", type(s).__name__) for s in strategies],
        "questions": len(questions),
        "questions_hash": questions_hash(questions),
        "question_source": str(Path(TTL).relative_to(ROOT)),
        "question_source_hash": file_hash(TTL),
        "database": str(Path(db).relative_to(ROOT)) if Path(db).is_relative_to(ROOT) else str(db),
        "database_hash": file_hash(db),
        "ontology_hash": file_hash(ONTOLOGY),
        "mapping_hash": file_hash(MAPPING),
        "python": sys.version.split()[0],
    }

    floors = {getattr(s, "name", type(s).__name__): s.ddl_path
              for s in strategies if getattr(s, "ddl_path", None) is not None}
    meta["ddl"] = {name: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                   for name, p in floors.items()} or None
    meta["ddl_hash"] = {name: file_hash(p) for name, p in floors.items()} or None

    meta.update(api_endpoint())
    records: list[dict] = []
    handle = Path(out).open("w") if out else None
    if handle:
        handle.write(json.dumps(meta) + "\n")
    try:
        for strategy in strategies:
            sname = getattr(strategy, "name", type(strategy).__name__)
            for qi, q in enumerate(questions):
                for run in range(n):
                    t0 = time.time()
                    try:
                        attempt = strategy(q, run)
                    except Exception as exc:  # noqa: BLE001
                        attempt = Attempt(sql="", error=f"{type(exc).__name__}: {exc}"[:300])
                    seconds = time.time() - t0
                    verdict = scorer.score(q, attempt.sql) if attempt.sql.strip() else "wrong"
                    before = (scorer.score(q, attempt.sql_before_repair)
                              if attempt.sql_before_repair else None)
                    rec = {
                        "kind": "run",
                        "strategy": sname,
                        "qid": q.qid,
                        "question": q.prompt,
                        "scope": q.scope,
                        "mf_reasons": list(q.mf_reasons),
                        "mf_subject": q.mf_subject,
                        "mf_notes": list(q.mf_notes),

                        "feasible_strict": q.feasible_strict,
                        "feasible_moderate": q.feasible_moderate,
                        "feasible_permissive": q.feasible,
                        "run": run,
                        "temperature": getattr(strategy, "temperature", None),
                        "pred_sql": attempt.sql,
                        "gold_sql": q.gold_sql,
                        "verdict": verdict,
                        "verdict_before_repair": before,
                        "result_key": scorer.result_key(attempt.sql, attempt.error),

                        "text_key": text_key(attempt.sql, attempt.error),
                        "refusal_class": (refusal_class(attempt.error)
                                          if not attempt.sql.strip() else None),
                        "seconds": round(seconds, 4),
                        "input_tokens": attempt.input_tokens,
                        "output_tokens": attempt.output_tokens,
                        "round_trips": attempt.round_trips,
                        "tool_calls": list(attempt.tool_calls),
                        "violations": list(attempt.violations),
                        "violations_after": (list(attempt.violations_after)
                                             if attempt.violations_after is not None else None),
                        "repaired": attempt.repaired,
                        "error": attempt.error,
                        "notes": attempt.notes,
                    }
                    records.append(rec)
                    if handle:
                        handle.write(json.dumps(rec) + "\n")
                if progress:
                    print(f"  {sname} [{qi + 1}/{len(questions)}] {q.prompt[:58]}",
                          file=sys.stderr)
    finally:
        if handle:
            handle.close()
    return {"meta": meta, "records": records,
            "summary": summarize(records, questions, n=n)}

def _text_key_of(record: dict) -> str:
    """TARr identity of one record, recomputed from what the record carries."""
    return text_key(record.get("pred_sql") or "", record.get("error"))

def _result_key_of(record: dict) -> str:
    """TARa identity of one record. The stored key is authoritative for
    anything that produced SQL -- it holds the executed ROWS, which cannot be
    recovered here without a database. An empty answer holds no rows by
    definition, so its key is rebuilt from the refusal reason."""
    if (record.get("pred_sql") or "").strip():
        return record.get("result_key", "")
    return "EMPTY:" + refusal_key(record.get("error"))

def _refusal_class_of(record: dict) -> str:
    return record.get("refusal_class") or refusal_class(record.get("error"))

def summarize(records: Sequence[dict], questions: Sequence[Question],
              *, n: int | None = None) -> dict:
    """Every number the comparison table prints, per strategy.

    Kept separate from `run_bench` so a results file can be re-scored later
    without re-running anything -- and so the harness's own tests can check the
    arithmetic against records they construct by hand.
    """
    by_q = {q.qid: q for q in questions}
    runs = [r for r in records if r.get("kind", "run") == "run"]
    out: dict[str, dict] = {}
    for sname in dict.fromkeys(r["strategy"] for r in runs):
        rows = [r for r in runs if r["strategy"] == sname]
        per_q: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            per_q[r["qid"]].append(r)
        n_eff = n or max((len(v) for v in per_q.values()), default=0)

        def acc(subset: Iterable[dict]) -> tuple[int, int]:
            subset = list(subset)
            return sum(1 for r in subset if r["verdict"] == "correct"), len(subset)

        ok, total = acc(rows)
        in_ok, in_total = acc(r for r in rows if r["scope"] == "in_scope")
        hop_ok, hop_total = acc(r for r in rows if r["scope"] == "too_many_hops")
        defect = sum(1 for r in rows if r["verdict"] == "defect_corrected")

        tarr = tara = 0
        agree_r: list[float] = []
        agree_a: list[float] = []
        for qid, rs in per_q.items():
            texts = Counter(_text_key_of(r) for r in rs)
            keys = Counter(_result_key_of(r) for r in rs)
            tarr += len(texts) == 1
            tara += len(keys) == 1
            agree_r.append(max(texts.values()) / len(rs))
            agree_a.append(max(keys.values()) / len(rs))

        refusals = [r for r in rows if not (r["pred_sql"] or "").strip()]
        all_refused = [qid for qid, rs in per_q.items()
                       if all(not (r["pred_sql"] or "").strip() for r in rs)]
        refusal_classes = Counter(_refusal_class_of(r) for r in refusals)

        tin = sum(r["input_tokens"] for r in rows)
        tout = sum(r["output_tokens"] for r in rows)
        cost = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
        fired = [r for r in rows if r["repaired"]]
        fixed = [r for r in fired
                 if r.get("verdict_before_repair") == "wrong" and r["verdict"] == "correct"]
        broke = [r for r in fired
                 if r.get("verdict_before_repair") == "correct" and r["verdict"] != "correct"]
        codes = Counter(v["code"] for r in rows for v in r["violations"])
        sev = Counter(v["severity"] for r in rows for v in r["violations"])

        out[sname] = {
            "runs": total,
            "questions": len(per_q),
            "n": n_eff,
            "accuracy": ok / total if total else 0.0,
            "accuracy_in_scope": in_ok / in_total if in_total else None,
            "accuracy_too_many_hops": hop_ok / hop_total if hop_total else None,

            "in_scope_questions": len({r["qid"] for r in rows if r["scope"] == "in_scope"}),
            "too_many_hops_questions": len({r["qid"] for r in rows
                                            if r["scope"] == "too_many_hops"}),
            "defect_corrected_runs": defect,
            "tarr_at_n": tarr / len(per_q) if per_q else 0.0,
            "tara_at_n": tara / len(per_q) if per_q else 0.0,
            "mean_text_agreement": sum(agree_r) / len(agree_r) if agree_r else 0.0,
            "mean_result_agreement": sum(agree_a) / len(agree_a) if agree_a else 0.0,

            "refusals": len(refusals),
            "refusal_rate": len(refusals) / total if total else 0.0,
            "questions_all_refused": len(all_refused),
            "questions_all_refused_ids": sorted(all_refused),
            "refusals_by_class": dict(refusal_classes.most_common()),
            "input_tokens": tin,
            "output_tokens": tout,
            "cost_usd": cost,
            "cost_per_question": cost / len(per_q) if per_q else 0.0,
            "cost_per_correct": cost / ok if ok else None,
            "seconds_total": sum(r["seconds"] for r in rows),
            "seconds_per_run": sum(r["seconds"] for r in rows) / total if total else 0.0,
            "round_trips_per_run": sum(r["round_trips"] for r in rows) / total if total else 0.0,
            "repair_fired": len(fired),
            "repair_fixed": len(fixed),
            "repair_broke": len(broke),
            "violations_by_code": dict(codes.most_common()),
            "violations_by_severity": dict(sev),
            "errors": sum(1 for r in rows if r["error"]),
        }
    return out

def _pct(x: float | None) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}"

def print_table(summary: dict, *, n: int) -> None:
    """One table. dbt's three accuracy columns first, ours next, cost last.

    `ref` sits IMMEDIATELY after the TAR pair and is not optional. The two
    numbers are read together or not at all: an arm that refuses every run
    scores TARr@N = TARa@N = 100% and has computed nothing, and the arms that
    refuse by design are the ones this study is about. `ref%` is the share of
    runs that produced no SQL; `allref` is the number of QUESTIONS whose every
    run refused, i.e. how much of the TAR columns is vacuous.
    """
    width = max([16] + [len(k) + 2 for k in summary])
    head = (f"{'strategy':<{width}}{'acc':>7}{'in-sc':>7}{'hops':>7}"
            f"{f'TARr@{n}':>9}{f'TARa@{n}':>9}{'ref':>7}{'allref':>8}"
            f"{'$':>9}{'$/corr':>9}"
            f"{'s/run':>8}{'trips':>7}{'rep':>10}")
    print(head)
    print("-" * len(head))
    for name, s in summary.items():
        rep = f"{s['repair_fired']}/{s['repair_fixed']}"
        per_correct = s["cost_per_correct"]
        per_correct = "-" if per_correct is None else f"{per_correct:.4f}"
        all_ref = f"{s.get('questions_all_refused', 0)}/{s['questions']}"
        print(f"{name:<{width}}{_pct(s['accuracy']):>7}{_pct(s['accuracy_in_scope']):>7}"
              f"{_pct(s['accuracy_too_many_hops']):>7}"
              f"{_pct(s['tarr_at_n']):>9}{_pct(s['tara_at_n']):>9}"
              f"{_pct(s.get('refusal_rate', 0.0)):>7}{all_ref:>8}"
              f"{s['cost_usd']:>9.3f}{per_correct:>9}"
              f"{s['seconds_per_run']:>8.2f}{s['round_trips_per_run']:>7.1f}{rep:>10}")
    first = next(iter(summary.values()), None)
    if first:
        print(f"\nn={n} · {first['questions']} questions "
              f"({first['in_scope_questions']} in scope, "
              f"{first['too_many_hops_questions']} too-many-hops) · "
              f"acc/in-sc/hops, TAR and ref are percentages · rep = fired/fixed")
        print("ref = share of RUNS that produced no SQL · allref = questions whose "
              "EVERY run refused;\nthose questions score 1.0 on both TAR columns "
              "for having computed nothing, so read TAR and ref together.")
        print("in-sc/hops are an ANNOTATION from the RETIRED MetricFlow "
              "infeasibility classifier\n(GOALS.md; DESIGN.md §6) -- not a result, "
              "and nothing comparative may be read off them.")
    for name, s in summary.items():
        if s["violations_by_code"]:
            print(f"  {name} violations: " +
                  ", ".join(f"{k}×{v}" for k, v in s["violations_by_code"].items()))
        if s.get("refusals_by_class"):
            print(f"  {name} refusals: " +
                  ", ".join(f"{k}×{v}" for k, v in s["refusals_by_class"].items()))
        if s["errors"]:
            print(f"  {name} strategy errors: {s['errors']}")

def _alternates_agree(q: Question, ttl: str | Path = TTL, db: str | Path = DB) -> str:
    """Do an inquiry's several SQL golds return the same rows?

    Recorded rather than assumed. If two golds for one prompt disagree, the
    choice of gold decides the score, and that has to be visible before anyone
    quotes a number -- not discovered while defending one.
    """
    from rdflib import Graph, Namespace, RDF

    try:
        g = Graph()
        g.parse(str(ttl), format="turtle")
        qa, dwt = Namespace(QA), Namespace(DWT)
        texts = {str(s).rsplit("/", 1)[-1]: str(g.value(s, qa.queryText))
                 for s in g.subjects(RDF.type, dwt.SqlQuery)}
        con = sqlite3.connect(str(db))
        mine = canon(con.execute(q.gold_sql).fetchall())
        for other in q.alternates:
            sql, _ = _normalize_gold(texts[other])
            if canon(con.execute(sql).fetchall()) != mine:
                return f"DISAGREES with {other[:14]} -- the choice of gold changes the score"
        return "alternates return the same rows"
    except Exception as exc:                   # noqa: BLE001
        return f"could not compare: {str(exc)[:60]}"

def question_report(questions: Sequence[Question], *, verbose: bool = True) -> str:
    n = len(questions)
    lines = [f"{n} inquiries parsed from {Path(TTL).name}"]
    execu = [q for q in questions if q.gold_executable]
    lines.append(f"{len(execu)} have gold SQL that executes against {Path(DB).name}")
    for q in questions:
        if not q.gold_executable:
            lines.append(f"  NOT EXECUTABLE  {q.qid[:14]}  {q.gold_error}")
            lines.append(f"                  {q.prompt[:100]}")
    for q in (x for x in questions if x.normalizations):
        lines.append(f"  normalised      {q.qid[:14]}  {', '.join(q.normalizations)}")
    for q in (x for x in questions if x.alternates):
        lines.append(f"  multiple golds  {q.qid[:14]}  chose the executable one; "
                     f"alternates={', '.join(a[:14] for a in q.alternates)} "
                     f"({_alternates_agree(q)})")

    flagged = [q for q in questions if q.mf_notes]
    for q in flagged:
        lines.append(f"  CLASSIFIER NOTE {q.qid[:14]}  {', '.join(q.mf_notes)}")

    lines.append(
        f"\nMETRICFLOW FEASIBILITY, per regime (derived from the governed path "
        f"graph; hop ceiling {MF_HOP_LIMIT} joins BETWEEN SEMANTIC MODELS, "
        f"fan-out joins refused, multi-fact metrics permitted)")
    counted = {
        STRICT.name: sum(1 for q in questions if not q.feasible_strict),
        MODERATE.name: sum(1 for q in questions if not q.feasible_moderate),
        PERMISSIVE.name: sum(1 for q in questions if not q.feasible),
    }
    for regime in REGIMES:
        bad = counted[regime.name]
        mark = "  <-- HEADLINE" if regime is HEADLINE_REGIME else ""
        lines.append(f"  {regime.name:<11} infeasible {bad:>3}/{n} "
                     f"({100 * bad / max(n, 1):.0f}%){mark}")
        for chunk in re.findall(r".{1,86}(?:\s|$)", regime.description):
            if chunk.strip():
                lines.append(f"              {chunk.strip()}")
    head = counted[HEADLINE_REGIME.name]
    lines.append(f"\n  THE NUMBER THAT MAY BE QUOTED: {head}/{n} infeasible under "
                 f"{HEADLINE_REGIME.name}.")
    if head == 0:
        lines.append("  IT IS ZERO. Under normal reusable dbt modelling every question on "
                     "this\n  set is expressible, so the comparative claim rests on "
                     "MODELLING EFFORT,\n  not on expressiveness. Do not quote an "
                     "out-of-scope advantage from this set.")
    lines.append("  The retracted figure was 27/44, which used an asymmetric hop cost, "
                 "one\n  arbitrarily chosen gold SQL per question, and no wide models at "
                 "all.")

    hops = [q for q in questions if q.scope == "too_many_hops"]
    kinds = Counter(r.split(":")[0] for q in hops for r in q.mf_reasons[:1])
    for kind, count in kinds.most_common():
        lines.append(f"      {kind:<14} {count}")
    if verbose:
        for q in sorted(questions, key=lambda x: x.qid):
            flags = "".join("SMP"[i] if f else "."
                            for i, f in enumerate((q.feasible_strict,
                                                   q.feasible_moderate, q.feasible)))
            lines.append(f"  {flags}  {q.qid[:14]} {q.prompt[:86]}")
            lines.append(f"        needs {dict(q.requirements)}"
                         + (f"  (join-only, dropped: {', '.join(q.mf_join_only)})"
                            if q.mf_join_only else ""))
            for reason in (q.mf_reasons if not q.feasible else q.mf_strict_reasons):
                lines.append(f"        {'permissive' if not q.feasible else 'strict'}: "
                             f"{reason}")
        lines.append("  flags: S=feasible strict, M=feasible moderate, "
                     "P=feasible permissive; '.' = infeasible")
    lines.append(f"\nquestion set hash: {questions_hash(questions)}")
    return "\n".join(lines)

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--strategies", default="stub_gold",
                   help="comma-separated: stub_gold, stub_broken, stub_drift, "
                        "stub_role_blind, ddl, context, skills, skills_checked, "
                        "menu, menu_tools, and <any>_checked")
    p.add_argument("--n", type=int, default=20, help="iterations per question (dbt: 20)")
    p.add_argument("--model", default="", help="model id; the LLM arms need it")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="sampling temperature for every LLM arm (the study "
                        "measures at 0.0 and 1.0); recorded per run")
    p.add_argument("--db", default=str(DB))
    p.add_argument("--ddl", default=str(DDL),
                   help="schema text shown to the `ddl` floor arm. Shorthands: "
                        "`floor` = the repaired schema (ACME_floor.ddl: every FK "
                        "resolves, all ontology-reachable tables present), "
                        "`floor_values` = that plus low-cardinality value hints, "
                        "`small` = the original ACME_small.ddl, which declares 10 "
                        "foreign keys to tables it never defines. Recorded and "
                        "hashed in run_meta, so a result file states its own floor.")
    p.add_argument("--out", default=None, help="JSONL results path")
    p.add_argument("--limit", type=int, default=0, help="first K questions (smoke runs)")
    p.add_argument("--qids", default="", help="comma-separated qid prefixes")
    p.add_argument("--scope", default="", choices=["", "in_scope", "too_many_hops"])
    p.add_argument("--include-unexecutable", action="store_true",
                   help="keep questions whose gold does not run (they score 0)")
    p.add_argument("--questions", action="store_true", help="parse report, then exit")
    p.add_argument("--json", action="store_true",
                   help="with --questions: emit the classification as JSON")
    p.add_argument("--progress", action="store_true")
    args = p.parse_args(argv)

    questions = load_questions(db=args.db)
    if args.questions:
        if args.json:
            print(json.dumps({
                "questions_hash": questions_hash(questions),
                "question_source_hash": file_hash(TTL),
                "ontology_hash": file_hash(ONTOLOGY),
                "mapping_hash": file_hash(MAPPING),
                "hop_limit": MF_HOP_LIMIT,
                "headline_regime": HEADLINE_REGIME.name,
                "regimes": {r.name: {"description": r.description,
                                     "infeasible": sum(
                                         1 for q in questions
                                         if not {"strict": q.feasible_strict,
                                                 "moderate": q.feasible_moderate,
                                                 "permissive": q.feasible}[r.name])}
                            for r in REGIMES},
                "items": [{"qid": q.qid, "prompt": q.prompt, "scope": q.scope,
                           "gold_executable": q.gold_executable,
                           "mf_subject": q.mf_subject, "mf_reasons": list(q.mf_reasons),
                           "mf_strict_reasons": list(q.mf_strict_reasons),
                           "feasible_strict": q.feasible_strict,
                           "feasible_moderate": q.feasible_moderate,
                           "feasible_permissive": q.feasible,
                           "join_only_dropped": list(q.mf_join_only),
                           "requirements": dict(q.requirements)}
                          for q in questions]}, indent=1))
            return 0
        print(question_report(questions))
        return 0
    if not args.include_unexecutable:
        questions = [q for q in questions if q.gold_executable]
    if args.scope:
        questions = [q for q in questions if q.scope == args.scope]
    if args.qids:
        wanted = tuple(x.strip() for x in args.qids.split(",") if x.strip())
        questions = [q for q in questions if q.qid.startswith(wanted)]
    if args.limit:
        questions = questions[: args.limit]

    specs = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if any(not s.startswith("stub") for s in specs) and not args.model:
        p.error("non-stub strategies need --model")
    if any(s.startswith("stub") and s.endswith("_checked") for s in specs) and not args.model:
        print("note: stub_*_checked without --model runs the CHECKER but skips the "
              "repair call; repair_fired stays 0 by design.", file=sys.stderr)
    ddl_path = {"floor": FLOOR_DDL, "floor_values": FLOOR_DDL_VALUES,
                "small": DDL}.get(args.ddl, Path(args.ddl))
    if not Path(ddl_path).exists():
        sys.exit(f"--ddl {args.ddl!r} resolves to {ddl_path}, which does not exist")

    sink = trace.configure()
    if sink != "none":
        print(f"trace: {sink}", file=sys.stderr)

    strategies = [build_strategy(s, model=args.model, db=args.db,
                                 temperature=args.temperature, ddl=ddl_path)
                  for s in specs]

    out = args.out
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = (args.model or "stub").replace(".", "").replace("-", "_")
        temp = f"t{args.temperature:g}".replace(".", "")
        out = RESULTS / f"bench_{slug}_{temp}_n{args.n}_{stamp}.jsonl"
    result = run_bench(questions, strategies, n=args.n, db=args.db, out=out,
                       model=args.model, progress=args.progress)
    print_table(result["summary"], n=args.n)
    print(f"\nwrote {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
