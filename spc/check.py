from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import sqlglot
from sqlglot import exp

__all__ = ["Violation", "check", "blocking", "OntologyView", "CODES"]

@dataclass(frozen=True)
class Violation:
    """One governance failure, precise enough to repair from.

    code      machine-readable, stable, one of CODES
    message   one human sentence, naming the ontology object that was violated
    fragment  the offending SQL text -- so a repair prompt can point, not wave
    severity  "violation" (proven unsafe) | "undecidable" (cannot be certified)
    detail    structured extras (edge names, alternatives, measured fan-out)
    """

    code: str
    message: str
    fragment: str = ""
    severity: str = "violation"
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover
        tag = "UNDECIDABLE" if self.severity == "undecidable" else "VIOLATION"
        frag = f"  [{self.fragment}]" if self.fragment else ""
        return f"{tag} {self.code}: {self.message}{frag}"

def blocking(violations: Iterable[Violation]) -> list[Violation]:
    """The violations that must refuse the SQL. Proven-unsafe only.

    THIS EXISTS BECAUSE THE POLICY WAS WRITTEN TWICE AND DRIFTED. The batch arm
    filtered `severity == "violation"`; the agent filtered `severity == "error"`,
    a value `Violation` cannot hold, so the agent's checker refused NOTHING from
    the day it was wired in while its traces recorded "certified". Two copies of
    one rule is how a safety layer goes quietly inert -- there is now one copy,
    and callers that want the severity split can still read it off the objects.

    "undecidable" is deliberately NOT blocking here: it is refused by the caller
    with its own reason and its own count, so proven-unsafe and cannot-decide
    never sum into a single number in a results table.
    """
    return [v for v in violations
            if getattr(v, "severity", "violation") == "violation"]

CODES = {
    "PARSE_ERROR": "the SQL could not be parsed",
    "UNSUPPORTED_STATEMENT": "not a SELECT",
    "UNKNOWN_TABLE": "a table no ontology concept, junction or restrict maps to",
    "UNDECLARED_JOIN": "a join the ontology does not sanction",
    "CARTESIAN": "two sources with no join condition between them",
    "UNCOMMITTED_ROLE": "a role-bearing traversal with no committing predicate",
    "GRAIN_FANOUT": "an aggregate computed over rows a join has duplicated",
    "GRAIN_MISMATCH": "an aggregate whose operand grain is not the answer's grain",
    "ROW_FANOUT": "a row-level expression combining columns across a fan-out",
    "GRAIN_UNDECIDABLE": "grain safety cannot be decided for this query shape",
    "SHAPE_UNDECIDABLE": "a construct this checker does not analyse",
    "METRIC_FIDELITY": "a governed metric computed as an improvised variant",
    "UNGROUNDED_LITERAL": "a literal that matches no row in that column",
}

_ONE = "one"
_MANY = "many"
_FANOUT_MAP = {"none": _ONE, "one": _ONE, "bounded": _MANY, "multiplicative": _MANY}

_SENSITIVE_AGGS = (exp.Sum, exp.Avg, exp.Count, exp.GroupConcat, exp.Stddev, exp.Variance)
_INSENSITIVE_AGGS = (exp.Min, exp.Max)

@dataclass(frozen=True)
class Edge:
    name: str
    src: str
    dst: str
    join: dict[str, str] | None
    via_table: str | None
    via_from: dict[str, str] | None
    via_to: dict[str, str] | None
    restrict_table: str | None
    restrict_cols: dict[str, str] | None
    role_predicate: dict[str, str] | None
    fan_out: str
    fan_out_reverse: str

def _as_dict(onto: Any) -> dict[str, Any]:
    """Accept the merged ontology document, or any loader that can produce one.

    THE SEAM, and the only coupling this module has to the ontology layer: a
    mapping, or an object exposing `.raw` / `.doc` / `.to_dict()`.
    `spc.ontology.Ontology` satisfies it, and so does a bare `yaml.safe_load` of
    acme.yaml merged with acme_semantics.yaml. The checker derives every index it
    needs from that document, so it never has to agree with another module's
    record types.
    """
    if isinstance(onto, Mapping):
        return dict(onto)
    for attr in ("raw", "data", "doc", "document", "yaml"):
        val = getattr(onto, attr, None)
        if isinstance(val, Mapping):
            return dict(val)
    to_dict = getattr(onto, "to_dict", None)
    if callable(to_dict):
        val = to_dict()
        if isinstance(val, Mapping):
            return dict(val)
    raise TypeError(
        "check() needs an ontology mapping; got %r. Pass the merged YAML "
        "document, or expose .raw/.doc/.to_dict() on the loader." % type(onto)
    )

class OntologyView:
    """Normalised, indexed read-only view over the ontology mapping."""

    def __init__(self, onto: Any) -> None:
        doc = _as_dict(onto)
        self.doc = doc
        self.concepts: dict[str, dict] = doc.get("concepts", {}) or {}
        self.metrics: dict[str, dict] = doc.get("metrics", {}) or {}
        self.glossary: list[dict] = doc.get("glossary", []) or []

        self.edges: list[Edge] = []
        for raw in doc.get("edges", []) or []:
            via = raw.get("via") or {}
            res = raw.get("restrict") or {}
            self.edges.append(Edge(
                name=raw["name"],
                src=raw["from"],
                dst=raw["to"],
                join=_lower_map(raw.get("join")),
                via_table=_lower(via.get("table")) if via else None,
                via_from=_lower_map(via.get("from_join")) if via else None,
                via_to=_lower_map(via.get("to_join")) if via else None,
                restrict_table=_lower(res.get("table")) if res else None,
                restrict_cols=_lower_map(res.get("columns")) if res else None,
                role_predicate=raw.get("role_predicate"),
                fan_out=_FANOUT_MAP.get(raw.get("fan_out", ""), _MANY),
                fan_out_reverse=_FANOUT_MAP.get(raw.get("fan_out_reverse", ""), _MANY),
            ))
        self.edge_by_name = {e.name: e for e in self.edges}

        self.table_concepts: dict[str, list[str]] = defaultdict(list)
        for name, c in self.concepts.items():
            self.table_concepts[_lower(c["table"])].append(name)

        self.via_tables = {e.via_table for e in self.edges if e.via_table}
        self.restrict_tables = {e.restrict_table for e in self.edges if e.restrict_table}

        self.metric_by_via: dict[str, str] = {}
        for mname, m in self.metrics.items():
            op = (m.get("operand") or {})
            if op.get("via"):
                self.metric_by_via[op["via"]] = mname

        self.edges_by_kind: dict[tuple[str, str], set[str]] = defaultdict(set)
        for e in self.edges:
            if e.restrict_table:
                self.edges_by_kind[(e.dst, e.restrict_table)].add(e.name)

        for via, mname in list(self.metric_by_via.items()):
            for equivalent in self.same_kind(via):
                self.metric_by_via.setdefault(equivalent, mname)

        self.metric_by_phrase: dict[str, str] = {}
        for mname in self.metrics:
            self.metric_by_phrase[_squash(mname)] = mname
        for entry in self.glossary:
            means = entry.get("means") or {}
            if "metric" in means:
                for phrase in entry.get("phrases", []):
                    self.metric_by_phrase.setdefault(_squash(phrase), means["metric"])

    def same_kind(self, via: str) -> set[str]:
        """Every edge asserting the same subtype as `via`, `via` included."""
        e = self.edge_by_name.get(via)
        if e is None or not e.restrict_table:
            return {via}
        return set(self.edges_by_kind.get((e.dst, e.restrict_table), {via}))

    def table_of(self, concept: str) -> str:
        return _lower(self.concepts[concept]["table"])

    def backed_where(self, concept: str) -> dict[str, str] | None:
        return self.concepts.get(concept, {}).get("backed_where")

    def role_variants(self, table: str) -> list[str]:
        """Concepts on `table` whose identity includes a role predicate."""
        return [c for c in self.table_concepts.get(table, []) if self.backed_where(c)]

    def attr_column(self, concept: str, attr: str) -> str | None:
        a = (self.concepts.get(concept, {}).get("attributes") or {}).get(attr)
        if isinstance(a, Mapping) and a.get("column"):
            return _lower(a["column"])
        return None

    def schema_from_ontology(self) -> dict[str, set[str]]:
        """A partial column map, used when no database is supplied.

        Contains every column the ontology can name: attributes, keys, join
        columns, junction and restrict columns. Enough to resolve unqualified
        columns in governed SQL; a column outside it is, by construction, a
        column the ontology does not govern.
        """
        sch: dict[str, set[str]] = defaultdict(set)
        for name, c in self.concepts.items():
            t = _lower(c["table"])
            key = c.get("key")
            for k in (key if isinstance(key, list) else [key] if key else []):
                sch[t].add(_lower(k))
            for a in (c.get("attributes") or {}).values():
                if isinstance(a, Mapping) and a.get("column"):
                    sch[t].add(_lower(a["column"]))
            for col in (c.get("backed_where") or {}):
                sch[t].add(_lower(col))
        for e in self.edges:
            st, dt = self.table_of(e.src), self.table_of(e.dst)
            if e.join:
                for a, b in e.join.items():
                    sch[st].add(a)
                    sch[dt].add(b)
            if e.via_table:
                for a, b in (e.via_from or {}).items():
                    sch[st].add(a)
                    sch[e.via_table].add(b)
                for a, b in (e.via_to or {}).items():
                    sch[e.via_table].add(a)
                    sch[dt].add(b)
                for col in (e.role_predicate or {}):
                    sch[e.via_table].add(_lower(col))
            if e.restrict_table:
                for a, b in (e.restrict_cols or {}).items():
                    sch[dt].add(a)
                    sch[e.restrict_table].add(b)
        return dict(sch)

def _arg(node, *names):
    """sqlglot renamed `from`/`with` to `from_`/`with_` in v26+. Accept both."""
    for n in names:
        v = node.args.get(n)
        if v is not None:
            return v
    return None

def _lower(s: Any) -> Any:
    return s.lower() if isinstance(s, str) else s

def _lower_map(m: Any) -> dict[str, str] | None:
    if not m:
        return None
    return {str(k).lower(): str(v).lower() for k, v in m.items()}

def _squash(s: str) -> str:
    """Alias/phrase normal form: 'Total_Loss' == 'total loss' == 'TotalLoss'."""
    return "".join(ch for ch in s.lower() if ch.isalnum())

@dataclass
class _JoinPair:
    """The equalities the query states between two aliases."""

    a: str
    b: str
    pairs: set[tuple[str, str]]
    nodes: list[exp.Expression]
    outer: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.a, self.b)))  # type: ignore[return-value]

    def oriented(self, first: str) -> set[tuple[str, str]]:
        if first == self.a:
            return set(self.pairs)
        return {(y, x) for x, y in self.pairs}

    def sql(self) -> str:
        return " AND ".join(n.sql() for n in self.nodes) or f"{self.a} ~ {self.b}"

@dataclass(frozen=True)
class _Prov:
    """Where one output column of a derived table came from.

    `concept` / `column` are ONTOLOGY facts -- which concept the value belongs to
    and which physical column holds it -- so two relations can be compared
    without either of them knowing the other's aliases. `source` is the INNER
    alias the value was read from, which is what makes two output columns of the
    SAME relation comparable (two columns with one `source` and one `column` are
    one column, however they were named on the way out). `vias` is the set of
    governed edges that reached that inner alias, which is where amount KIND
    lives in this ontology, and `agg` records an aggregation applied inside.
    """

    concept: str | None
    column: str | None
    source: str | None
    vias: tuple[str, ...] = ()
    agg: str | None = None

    shared: tuple[str, ...] = ()

@dataclass(frozen=True)
class _Derived:
    """What a CTE or subquery reduces to: a RELATION, not a table.

    `grain` is the set of `(source, column)` provenance keys that determine one
    row -- the GROUP BY of an aggregation, or the whole projection of a
    SELECT DISTINCT, reduced by functional dependency. A relation is only
    reduced when its grain is ESTABLISHED; a derived table whose row
    multiplicity is unknown stays opaque and is reported SHAPE_UNDECIDABLE, as
    before.
    """

    columns: Mapping[str, _Prov]
    grain: frozenset[tuple[str, str]]
    concepts: frozenset[str]

    grain_concepts: frozenset[tuple[str, str]] = frozenset()

    refinable_concepts: frozenset[str] = frozenset()

    def refs(self, columns: Iterable[str]) -> set[tuple[str, str]]:
        """The provenance keys a set of this relation's output columns carries."""
        out: set[tuple[str, str]] = set()
        for name in columns:
            p = self.columns.get(name)
            if p is not None and p.source is not None and p.column is not None:
                out.add((p.source, p.column))
        return out

@dataclass
class _Scope:
    select: exp.Select
    alias_table: dict[str, str]
    opaque: set[str]
    joins: dict[tuple[str, str], _JoinPair]
    literals: list[tuple[str, str, str, Any, exp.Expression]]
    unconnected: list[str]
    schema: dict[str, set[str]]
    derived: dict[str, _Derived] = field(default_factory=dict)

@dataclass
class _Resolved:
    """One edge instantiation the cover accepted."""

    edge: Edge
    src_alias: str
    dst_alias: str
    aux: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    role_ok: bool
    role_concepts: tuple[str, ...] = ()

@dataclass(frozen=True)
class _Identity:
    """A join that is `C.key = C.key` -- the same concept on both sides.

    Not a traversal, so no ontology edge sanctions it and none is needed. What
    it IS depends only on the two relations' declared content: both sides
    project the same concept's key columns, and every stated equality pairs one
    of those columns with itself.

    `concepts` is a TUPLE because one join may be the identity on SEVERAL
    concepts at once -- a query grouped by policy AND by claim joins its
    pre-aggregated measure back on both keys, and that conjunction traverses no
    more than each conjunct does.
    """

    key: tuple[str, str]
    a: str
    b: str
    concepts: tuple[str, ...]
    cols_a: frozenset[str]
    cols_b: frozenset[str]

    pairs: frozenset[tuple[str, str]] = frozenset()

def check(
    sql: str,
    onto: Any,
    graph: Any = None,
    *,
    db: str | Path | None = None,
    expect_metrics: Sequence[str] | None = None,
    dialect: str = "sqlite",
    strict_metric_op: bool = False,
) -> list[Violation]:
    """Check `sql` against the ontology. Empty list == certified governed.

    onto            the ontology (see `_as_dict` -- the swap seam)
    graph           optional governed-path enumerator. Not required: a path
                    composed of declared edges is governed by construction
                    (DESIGN: enumeration produces "every legal reading, and only
                    legal ones"), which is what the edge cover below verifies.
                    If supplied and it exposes `allows_edge(name)`, edges it
                    rejects are treated as undeclared.
    db              sqlite file. Supplies the authoritative column map AND the
                    literal probe. Without it, literal grounding cannot run and
                    says so (severity "undecidable") rather than passing.
    expect_metrics  governed metrics the QUESTION resolved to. Without it,
                    metric fidelity fires only when the SQL names a metric in an
                    alias -- see SOUNDNESS.
    strict_metric_op  also require the aggregation operator to match the registry
                    (count vs count_distinct). Off by default: 'average loss' is
                    a legitimate AVG over a metric defined with SUM.
    """
    o = onto if isinstance(onto, OntologyView) else OntologyView(onto)
    out: list[Violation] = []

    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception as exc:
        return [Violation("PARSE_ERROR", f"SQL could not be parsed: {exc}", sql[:200])]
    if tree is None:
        return [Violation("PARSE_ERROR", "SQL could not be parsed", sql[:200])]

    schema = _schema(o, db)

    if isinstance(tree, (exp.Union, exp.Except, exp.Intersect)):
        out.append(Violation(
            "SHAPE_UNDECIDABLE",
            "set operations are not analysed as a whole; each branch is checked "
            "independently, so grain safety ACROSS the set operation is not certified.",
            tree.sql()[:200], severity="undecidable"))
    if not isinstance(tree, (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)):
        return [Violation("UNSUPPORTED_STATEMENT",
                          f"only SELECT statements are governed; got {type(tree).__name__}",
                          sql[:200])]

    cte_tables = _cte_base_tables(tree, schema)
    by_name, by_node = _reduce_derived(tree, o, graph, schema, cte_tables)

    outputs = {id(select) for select in _output_selects(tree)}
    elsewhere = {
        name for name in expect_metrics or ()
        if name in o.metrics
        and any(_best_expression_for(select, o, name) is not None
                for select in tree.find_all(exp.Select) if id(select) not in outputs)
    }
    for select in tree.find_all(exp.Select):
        scoped = expect_metrics if id(select) in outputs else None
        out.extend(_check_scope(select, o, graph, schema, db, cte_tables,
                                by_name, by_node, scoped, strict_metric_op,
                                computed_elsewhere=elsewhere))
    return out

def _output_selects(tree) -> list:
    """The selects whose rows ARE the query's answer -- set-operation branches
    included, CTEs and scalar subqueries excluded."""
    if isinstance(tree, exp.Subquery):
        return _output_selects(tree.this)
    if isinstance(tree, (exp.Union, exp.Except, exp.Intersect)):
        return _output_selects(tree.this) + _output_selects(tree.expression)
    return [tree] if isinstance(tree, exp.Select) else []

def _check_scope(select, o, graph, schema, db, cte_tables, by_name, by_node,
                 expect_metrics, strict_metric_op,
                 computed_elsewhere: frozenset[str] | set[str] = frozenset()
                 ) -> list[Violation]:
    out: list[Violation] = []
    scope = _read_scope(select, o, schema, cte_tables, by_name, by_node)

    for alias in sorted(scope.opaque):
        out.append(Violation(
            "SHAPE_UNDECIDABLE",
            f"source `{alias}` is a subquery or CTE this checker cannot reduce to a "
            f"base table, so joins and grain involving it are NOT certified.",
            alias, severity="undecidable"))

    known = set(o.table_concepts) | o.via_tables | o.restrict_tables
    for alias, table in sorted(scope.alias_table.items()):
        if table.startswith("<derived"):
            continue
        if table not in known:
            out.append(Violation(
                "UNKNOWN_TABLE",
                f"table `{table}` (alias `{alias}`) is mapped by no ontology concept, "
                f"junction or restrict table -- it is outside the governed surface.",
                f"{table} AS {alias}" if alias != table else table))

    resolved, uncovered, role_missing, identities = _cover_joins(scope, o, graph)

    uncovered, agg_pairs = _aggregation_joins(scope, select, uncovered)

    for key in uncovered:
        jp = scope.joins[key]
        cands = role_missing.get(key)
        if cands:
            names = sorted({c for inst in cands for c in inst.role_concepts})
            edges = sorted({inst.edge.name for inst in cands})
            out.append(Violation(
                "UNCOMMITTED_ROLE",
                f"the join between `{jp.a}` and `{jp.b}` matches {len(edges)} ontology "
                f"relation(s) that differ only by role ({', '.join(edges)}); the query "
                f"commits to none. Add a role predicate selecting exactly one of: "
                f"{', '.join(names)}.",
                jp.sql(),
                detail={"aliases": [jp.a, jp.b], "edges": edges, "alternatives": names}))
        else:
            out.append(Violation(
                "UNDECLARED_JOIN",
                f"the join `{jp.sql()}` corresponds to no declared ontology edge. "
                f"The ontology sanctions joins only along its edges; the data would "
                f"happily join anything.",
                jp.sql(),
                detail={"aliases": [jp.a, jp.b],
                        "columns": sorted(jp.pairs)}))

    for alias in sorted(scope.unconnected):
        out.append(Violation(
            "CARTESIAN",
            f"source `{alias}` has no join condition to any other source -- a cross "
            f"product, which no ontology edge sanctions.",
            alias))

    out.extend(_check_grain(select, scope, o, resolved, identities, bool(uncovered),
                            agg_pairs))

    out.extend(_check_metrics(select, scope, o, resolved, expect_metrics, strict_metric_op,
                              computed_elsewhere))

    out.extend(_check_literals(scope, o, db))

    return out

def _read_scope(select, o, schema, cte_tables, by_name=None, by_node=None) -> _Scope:
    by_name = by_name or {}
    by_node = by_node or {}
    alias_table: dict[str, str] = {}
    opaque: set[str] = set()
    derived: dict[str, _Derived] = {}

    sources: list[exp.Expression] = []
    frm = _arg(select, "from_", "from")
    if frm is not None:
        sources.append(frm.this if isinstance(frm, exp.From) else frm)
    joins = select.args.get("joins") or []
    outer_aliases: set[str] = set()
    for j in joins:
        sources.append(j.this)

    for src in sources:
        if isinstance(src, exp.Table):
            name = _lower(src.name)
            alias = _lower(src.alias_or_name)
            base = cte_tables.get(name, name)
            if base is not None:
                alias_table[alias] = base
            elif name in by_name:
                alias_table[alias] = _derived_table_name(name)
                derived[alias] = by_name[name]
            else:
                opaque.add(alias)
                alias_table[alias] = f"<derived:{name}>"
        elif isinstance(src, (exp.Subquery, exp.Lateral)):
            alias = _lower(src.alias_or_name) or "<anon>"
            base = _base_table_of_subquery(src)
            if base:
                alias_table[alias] = base
            elif id(src) in by_node:
                alias_table[alias] = _derived_node_name(src)
                derived[alias] = by_node[id(src)]
            else:
                alias_table[alias] = f"<derived:{alias}>"
                opaque.add(alias)
        else:  # pragma: no cover
            alias = _lower(getattr(src, "alias_or_name", "") or "<anon>")
            alias_table[alias] = f"<derived:{alias}>"
            opaque.add(alias)

    for j in joins:
        side = (j.args.get("side") or "").upper()
        if side in ("LEFT", "RIGHT", "FULL"):
            outer_aliases.add(_lower(j.this.alias_or_name))

    conds: list[exp.Expression] = []
    for j in joins:
        if j.args.get("on") is not None:
            conds.append(j.args["on"])
    where = select.args.get("where")
    if where is not None:
        conds.append(where.this)

    joins_map: dict[tuple[str, str], _JoinPair] = {}
    literals: list[tuple[str, str, str, Any, exp.Expression]] = []

    for cond in conds:
        for node in _conjuncts(cond):
            if node.find_ancestor(exp.Select) is not select:
                continue
            _classify(node, select, alias_table, schema, joins_map, literals, outer_aliases)

    constant = {a for a, relation in derived.items() if not relation.grain}
    connected = {a for key in joins_map for a in key} | constant
    unconnected = ([a for a in alias_table if a not in connected]
                   if len(alias_table) > 1 else [])

    return _Scope(select, alias_table, opaque, joins_map, literals,
                  sorted(unconnected), schema, derived)

def _conjuncts(node: exp.Expression) -> Iterable[exp.Expression]:
    if isinstance(node, exp.And):
        yield from _conjuncts(node.left)
        yield from _conjuncts(node.right)
    elif isinstance(node, exp.Paren):
        yield from _conjuncts(node.this)
    else:
        yield node

def _classify(node, select, alias_table, schema, joins_map, literals, outer_aliases) -> None:
    """Sort one conjunct into: alias-to-alias join, or column-to-literal filter."""
    if isinstance(node, exp.EQ):
        left, right = node.left, node.right
        lref = _colref(left, select, alias_table, schema)
        rref = _colref(right, select, alias_table, schema)
        if lref and rref:
            if lref[0] == rref[0]:
                return
            key = tuple(sorted((lref[0], rref[0])))
            jp = joins_map.get(key)
            if jp is None:
                jp = _JoinPair(a=key[0], b=key[1], pairs=set(), nodes=[],
                               outer=bool({key[0], key[1]} & outer_aliases))
                joins_map[key] = jp
            pair = (lref[1], rref[1]) if lref[0] == jp.a else (rref[1], lref[1])
            jp.pairs.add(pair)
            jp.nodes.append(node)
            return
        if lref and _is_literal(right):
            literals.append((lref[0], lref[1], "=", _literal_value(right), node))
        elif rref and _is_literal(left):
            literals.append((rref[0], rref[1], "=", _literal_value(left), node))
        return

    if isinstance(node, (exp.NEQ, exp.Like, exp.ILike, exp.GT, exp.GTE, exp.LT, exp.LTE)):
        lref = _colref(node.left, select, alias_table, schema)
        rref = _colref(node.right, select, alias_table, schema)
        op = {exp.NEQ: "!=", exp.Like: "LIKE", exp.ILike: "LIKE",
              exp.GT: ">", exp.GTE: ">=", exp.LT: "<", exp.LTE: "<="}[type(node)]
        if lref and rref and lref[0] != rref[0]:
            key = tuple(sorted((lref[0], rref[0])))
            jp = joins_map.setdefault(key, _JoinPair(a=key[0], b=key[1], pairs=set(), nodes=[]))
            jp.nodes.append(node)
            return
        if lref and _is_literal(node.right):
            literals.append((lref[0], lref[1], op, _literal_value(node.right), node))
        return

    if isinstance(node, exp.In):
        lref = _colref(node.this, select, alias_table, schema)
        vals = [e for e in (node.args.get("expressions") or []) if _is_literal(e)]
        if lref and vals and len(vals) == len(node.args.get("expressions") or []):
            for v in vals:
                literals.append((lref[0], lref[1], "IN", _literal_value(v), node))
        return

def _colref(node, select, alias_table, schema) -> tuple[str, str] | None:
    """(alias, column) for a column reference, resolving unqualified names."""
    if isinstance(node, exp.Paren):
        return _colref(node.this, select, alias_table, schema)
    if not isinstance(node, exp.Column):
        return None
    col = _lower(node.name)
    tbl = _lower(node.table)
    if tbl:
        return (tbl, col) if tbl in alias_table else None
    cands = [a for a, t in alias_table.items() if col in schema.get(t, set())]
    if len(cands) == 1:
        return cands[0], col
    if len(alias_table) == 1:
        return next(iter(alias_table)), col
    return None

def _is_literal(node) -> bool:
    return isinstance(node, (exp.Literal, exp.Boolean)) or (
        isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal))

def _literal_value(node) -> Any:
    if isinstance(node, exp.Neg):
        v = _literal_value(node.this)
        return -v if isinstance(v, (int, float)) else v
    if isinstance(node, exp.Boolean):
        return 1 if node.this else 0
    if node.is_string:
        return node.this
    txt = node.this
    try:
        return int(txt)
    except ValueError:
        try:
            return float(txt)
        except ValueError:
            return txt

def _renames_columns(inner: exp.Select) -> bool:
    """Does this derived table hand its columns out under NEW names?

    A derived table that selects from one base table and renames nothing IS that
    base table -- its columns are the base columns, so the ontology's declared
    join maps still apply to it directly. As soon as it renames a column, the
    outer query's `d.k = x.k` no longer names any column the ontology knows, and
    treating the alias as the base table would silently compare the wrong
    identifiers. Such a relation is reduced by `_reduce_derived` instead.
    """
    for item in inner.expressions:
        if isinstance(item, (exp.Star, exp.Column)):
            continue
        if (isinstance(item, exp.Alias) and isinstance(item.this, exp.Column)
                and _lower(item.alias_or_name) == _lower(item.this.name)):
            continue
        return True
    return False

def _base_table_of_subquery(sub) -> str | None:
    """A subquery reduces to a base table only when it selects from exactly one."""
    inner = sub.this
    if not isinstance(inner, exp.Select):
        return None
    if inner.args.get("joins"):
        return None
    if _renames_columns(inner):
        return None
    frm = _arg(inner, "from_", "from")
    if frm is None:
        return None
    t = frm.this if isinstance(frm, exp.From) else frm
    if isinstance(t, exp.Table):
        return _lower(t.name)
    return None

def _cte_base_tables(tree, schema) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    with_ = _arg(tree, "with_", "with")
    if not with_:
        return out
    for cte in with_.expressions:
        name = _lower(cte.alias_or_name)
        inner = cte.this
        base = None
        if (isinstance(inner, exp.Select) and not inner.args.get("joins")
                and not _renames_columns(inner)):
            frm = _arg(inner, "from_", "from")
            t = frm.this if isinstance(frm, exp.From) else frm
            if isinstance(t, exp.Table):
                base = _lower(t.name)
        out[name] = base
    return out

def _derived_table_name(cte: str) -> str:
    return f"<derived:{cte}>"

def _derived_node_name(node) -> str:
    return f"<derived#{id(node)}>"

def _depth(node) -> int:
    depth, cursor = 0, node.parent
    while cursor is not None:
        depth += 1
        cursor = cursor.parent
    return depth

def _reduce_derived(tree, o, graph, schema, cte_tables):
    """Reduce every CTE and every derived-table source that can be reduced.

    Returns `(by_cte_name, by_node_id)`. Both feed `_read_scope`, and the schema
    is extended with each reduced relation's OUTPUT columns so an unqualified
    reference to one still resolves.
    """
    by_name: dict[str, _Derived] = {}
    by_node: dict[int, _Derived] = {}

    with_ = _arg(tree, "with_", "with")
    if with_:
        for cte in with_.expressions:
            name = _lower(cte.alias_or_name)
            if cte_tables.get(name) is not None:
                continue
            relation = _reduce(cte.this, o, graph, schema, cte_tables, by_name, by_node)
            if relation is not None:
                by_name[name] = relation
                schema[_derived_table_name(name)] = set(relation.columns)

    candidates: list[exp.Expression] = []
    for select in tree.find_all(exp.Select):
        frm = _arg(select, "from_", "from")
        sources = [frm.this if isinstance(frm, exp.From) else frm] if frm is not None else []
        sources += [j.this for j in (select.args.get("joins") or [])]
        candidates += [s for s in sources if isinstance(s, exp.Subquery)]
    for sub in sorted(candidates, key=_depth, reverse=True):
        if _base_table_of_subquery(sub) is not None:
            continue
        relation = _reduce(sub.this, o, graph, schema, cte_tables, by_name, by_node)
        if relation is not None:
            by_node[id(sub)] = relation
            schema[_derived_node_name(sub)] = set(relation.columns)
    return by_name, by_node

def _reduce(select, o, graph, schema, cte_tables, by_name, by_node) -> _Derived | None:
    """A governed, grain-established SELECT read as a relation. Else None."""
    if not isinstance(select, exp.Select):
        return None
    if select.args.get("having") or select.args.get("limit"):
        return None
    scope = _read_scope(select, o, schema, cte_tables, by_name, by_node)
    if scope.opaque or scope.unconnected:
        return None
    resolved, uncovered, _role_missing, identities = _cover_joins(scope, o, graph)
    if uncovered:
        return None
    _adj0 = _adjacency(scope, resolved, identities)
    if len(scope.alias_table) > 1 and _has_cycle(_adj0[0]):
        return None
    alias_concepts = _alias_concepts(scope, resolved)
    incoming = _incoming_edges(resolved)

    columns: dict[str, _Prov] = {}
    for item in select.expressions:
        inner = item.this if isinstance(item, exp.Alias) else item
        if isinstance(item, exp.Star) or isinstance(inner, exp.Star):
            return None
        name = _lower(item.alias_or_name)
        if not name or name in columns:
            return None
        columns[name] = _projection_prov(inner, select, scope, o, alias_concepts, incoming)

    grain, concept_of = _grain_of(select, scope, o, columns, alias_concepts)
    if grain is None:
        return None
    grain_concepts = frozenset((concept_of[ref], ref[1]) for ref in grain if ref in concept_of)

    adj, idpairs = _adjacency(scope, resolved, identities)
    grain_aliases = {source for source, _column in grain}
    reachable: dict[str, set[str]] = {}
    resolved_columns: dict[str, _Prov] = {}
    for name, prov in columns.items():
        shared = set(prov.shared)
        if prov.source is not None and grain_aliases:
            if prov.source not in reachable:
                reachable[prov.source] = _multiplied_from(adj, prov.source, scope, idpairs)

            _p, _b, fixed = _pinned_walk(adj, idpairs, scope, prov.source)
            for alias in reachable[prov.source] & grain_aliases:
                alias_rel = scope.derived.get(alias)
                for ref in grain:
                    if ref[0] != alias or ref not in concept_of:
                        continue

                    if prov.source in scope.alias_table and alias_rel is not None:
                        base_cols = {c for col in alias_rel.columns
                                     for c in alias_rel.refs([col])}
                        if {c for c in base_cols if c[1] == ref[1]} <= fixed \
                                and any(c[1] == ref[1] for c in fixed):
                            continue
                    shared.add(concept_of[ref])
        resolved_columns[name] = replace(prov, shared=tuple(sorted(shared)))

    own = {concept for concept, _column in grain_concepts}
    refinable: set[str] = set()
    for alias in grain_aliases:
        if alias not in reachable:
            reachable[alias] = _multiplied_from(adj, alias, scope, idpairs)
        for reached in reachable[alias]:
            concept = _alias_concept(scope, o, reached, alias_concepts)
            if concept:
                refinable.add(concept)
    for relation in scope.derived.values():
        refinable |= {c for c, _column in relation.grain_concepts}
        refinable |= set(relation.refinable_concepts)

    return _Derived(
        columns=resolved_columns,
        grain=grain,
        concepts=frozenset(p.concept for p in resolved_columns.values() if p.concept),
        grain_concepts=grain_concepts,
        refinable_concepts=frozenset(refinable - own),
    )

def _projection_prov(expr, select, scope, o, alias_concepts, incoming) -> _Prov:
    """What one projected expression is, in ontology terms."""
    agg = None
    inner = expr
    while isinstance(inner, exp.Paren):
        inner = inner.this
    if isinstance(inner, exp.AggFunc):
        agg = _agg_name(inner)
        inner = inner.this
        if isinstance(inner, exp.Distinct):
            expressions = inner.expressions or []
            inner = expressions[0] if len(expressions) == 1 else None
    if inner is None:
        return _Prov(None, None, None, (), agg)
    cols = [c for c in inner.find_all(exp.Column)
            if c.find_ancestor(exp.Select) is select]
    if len(cols) != 1:
        return _Prov(None, None, None, (), agg)
    ref = _colref(cols[0], select, scope.alias_table, scope.schema)
    if ref is None:
        return _Prov(None, None, None, (), agg)
    alias, column = ref
    upstream = scope.derived.get(alias)
    if upstream is not None:
        p = upstream.columns.get(column)
        if p is None:
            return _Prov(None, None, None, (), agg)
        return _Prov(p.concept, p.column, alias, p.vias, agg or p.agg, p.shared)
    concept = _alias_concept(scope, o, alias, alias_concepts)
    vias = tuple(sorted(incoming.get(alias, set())))
    return _Prov(concept, column, alias, vias, agg)

def _resolve_prov(scope, o, alias, column, alias_concepts) -> _Prov | None:
    """The ontology reading of one column reference in this scope.

    A column of a reduced relation is rebased onto the alias THIS scope sees, as
    `_projection_prov` already does. Handing back the inner relation's own source
    made the two disagree, and `_grain_of` compares them: a relation that GROUPS
    BY a column of another reduced relation had its grouping keys read as
    `('party', ...)` and its projections as `('__g0', ...)`, so the two never
    matched and the relation was never reduced. That is why a two-stage measure
    -- deduplicate, then aggregate -- came back SHAPE_UNDECIDABLE.
    """
    relation = scope.derived.get(alias)
    if relation is not None:
        prov = relation.columns.get(column)
        if prov is None:
            return None
        return _Prov(prov.concept, prov.column, alias, prov.vias, prov.agg, prov.shared)
    concept = _alias_concept(scope, o, alias, alias_concepts)
    if concept is None:
        return None
    return _Prov(concept, column, alias, (), None)

def _alias_concept(scope, o: OntologyView, alias, alias_concepts) -> str | None:
    """The one concept a base-table alias denotes, or None if it is ambiguous."""
    if alias in scope.derived:
        return None
    candidates = {c for c in alias_concepts.get(alias, set())}
    if len(candidates) == 1:
        return next(iter(candidates))
    on_table = o.table_concepts.get(scope.alias_table.get(alias, ""), [])
    if len(on_table) == 1:
        return on_table[0]
    committed = [c for c in on_table
                 if o.backed_where(c) and all(
                     _has_predicate(scope, alias, _lower(col), str(val))
                     for col, val in (o.backed_where(c) or {}).items())]
    if len(committed) == 1:
        return committed[0]
    return None

def _concept_key(o: OntologyView, concept: str | None) -> tuple[str, ...]:
    if not concept:
        return ()
    key = (o.concepts.get(concept) or {}).get("key")
    if key is None:
        return ()
    return tuple(_lower(k) for k in (key if isinstance(key, list) else [key]))

def _grain_of(select, scope, o, columns, alias_concepts):
    """The provenance keys that determine one row of this relation, or None.

    Returns `(grain, grain_concepts)` -- the same fact twice, once in the
    relation's own alias terms and once in the ontology's, because a downstream
    scope can only compare the ontology reading against its own GROUP BY.

    Three shapes establish it, all read off the query's own structure:
      GROUP BY   one row per group -- provided every grouping expression is also
                 PROJECTED, because a grouping key the relation does not hand
                 out cannot be matched by anything joining to it.
      aggregates with no GROUP BY   exactly one row.
      DISTINCT   one row per projected tuple.
    Anything else leaves multiplicity unknown, and unknown is not reduced.
    """
    group = select.args.get("group")
    aggregates = [n for n in select.find_all(exp.AggFunc)
                  if n.find_ancestor(exp.Select) is select]
    refs: set[tuple[str, str]] = set()
    concept_of: dict[tuple[str, str], str] = {}

    def remember(p) -> None:
        if p.concept is not None:
            concept_of[(p.source, p.column)] = p.concept

    if group is not None:
        for expression in group.expressions:
            ref = _colref(expression, select, scope.alias_table, scope.schema)
            if ref is None:
                return None, {}
            p = _resolve_prov(scope, o, ref[0], ref[1], alias_concepts)
            if p is None or p.source is None or p.column is None:
                return None, {}
            refs.add((p.source, p.column))
            remember(p)
        visible = {(p.source, p.column) for p in columns.values()
                   if p.agg is None and p.source is not None and p.column is not None}
        if not refs <= visible:
            return None, {}
    elif aggregates:
        refs = set()
    elif select.args.get("distinct"):
        for p in columns.values():
            if p.agg is not None or p.source is None or p.column is None:
                return None, {}
            refs.add((p.source, p.column))
            remember(p)
    else:
        return None, {}
    for p in columns.values():
        if p.agg is None and p.source is not None and p.column is not None:
            remember(p)
    return frozenset(_minimal_grain(refs, scope, o, alias_concepts)), concept_of

def _minimal_grain(refs, scope, o, alias_concepts) -> set[tuple[str, str]]:
    """Drop grain columns a functional dependency already determines.

    If the grain pins a full KEY of some source, every other column of that
    source is a function of it and carries no multiplicity. Without this, a
    relation that hands out one concept's key twice under two names (a subject
    key AND a subject-grain measure read off the same row) would read as finer
    than it is, and a correctly grain-aligned join to it would be reported as a
    fan-out.
    """
    by_source: dict[str, set[str]] = defaultdict(set)
    for source, column in refs:
        by_source[source].add(column)
    out: set[tuple[str, str]] = set()
    for source, cols in by_source.items():
        key = set(_concept_key(o, _alias_concept(scope, o, source, alias_concepts)))
        keep = key if key and key <= cols else cols
        out |= {(source, c) for c in keep}
    return out

def _schema(o: OntologyView, db) -> dict[str, set[str]]:
    sch = o.schema_from_ontology()
    if db is None:
        return sch
    con = sqlite3.connect(str(db))
    try:
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            cols = {r[1].lower() for r in con.execute(f'PRAGMA table_info("{t}")')}
            sch.setdefault(t.lower(), set()).update(cols)
    finally:
        con.close()
    return sch

def _aggregation_joins(scope: _Scope, select, uncovered):
    """-> (still_uncovered, accepted_pairs). See the rule below."""
    """Drop joins that re-associate pre-aggregated relations at a certified grain.

    THE SHAPE is the compiler's multi-measure lowering: a `__spine` of distinct
    group-key tuples, one pre-aggregated satellite per measure, and an outer
    query that joins them and re-aggregates:

        SELECT __spine.__d0, AVG(__m0.__v + __m1.__v + ...)
        FROM __spine JOIN __m0 ON __spine.__k0 = __m0.__k0
                    JOIN __m1 ON __spine.__k0 = __m1.__k0
                             AND __m0.__p  = __m1.__p
        GROUP BY __spine.__d0

    No ontology EDGE describes `__spine.__k0 = __m0.__k0` -- both sides are
    relations this query built -- so `_cover_joins` reports UNDECLARED_JOIN on
    every such join, and did from the first day the compiler emitted the shape.
    Nobody noticed for the checker's whole life in the agent path, because the
    agent's filter was inert (see `blocking`); switching it on refused every
    `over:`-lowered multi-measure query the compiler itself considers safest.

    THE CERTIFICATION IS READ OFF THE SQL, not the ontology, and it is the
    identity-join argument adapted to relations whose identity is their grain:

      * BOTH sides must be REDUCED relations with an ESTABLISHED grain (the
        `SELECT DISTINCT`/GROUP BY of a known-multiplicity CTE). Two base
        tables joined on a shared identifier stay a traversal the ontology
        declares or does not; an opaque relation stays SHAPE_UNDECIDABLE.
      * The matched pairs must FULLY COVER one side's grain. A side whose grain
        is fully matched holds at most one row per matched tuple, so the OTHER
        side's rows are never duplicated through this join -- at worst the
        fully-covered side's rows repeat, which is what the outer GROUP BY
        exists to consume.
      * Every reference, in the outer SELECT's projection, to a side of this
        join must sit under an AGGREGATE. A satellite column read bare, or in
        the GROUP BY, changes the answer's grain rather than feeding an
        aggregate, and such a join is NOT certified here.

    The grain check still runs afterwards with the join accepted, so a value
    computed one row per COARSER grain than the grouping is still refused
    (GRAIN_MISMATCH, the 47603eb6 fan-trap class); only the finer-under-
    aggregate case -- which is what re-aggregation IS -- passes.

    Sound because each condition is load-bearing; remove any one and a wrong
    answer becomes expressible:
      drop "both reduced"    -> an unaggregated detail table multiplies rows
      drop "full cover"      -> the fan trap returns (partial key, many rows)
      drop "under aggregate" -> a bare satellite column leaks a bogus row grain
    """

    matched_cols: dict[str, set[str]] = {}
    for key in uncovered:
        jp = scope.joins[key]
        matched_cols.setdefault(jp.a, set()).update(c for c, _ in jp.pairs)
        matched_cols.setdefault(jp.b, set()).update(c for _, c in jp.pairs)

    def _side_covered(alias: str) -> bool:
        rel = scope.derived.get(alias)
        cols = matched_cols.get(alias) or set()
        return (rel is not None and bool(rel.grain) and bool(cols)
                and rel.refs(cols) >= rel.grain)

    def _only_under_aggregate(alias: str) -> bool:
        """No PROJECTION reference to `alias` outside an aggregate function.

        Scoped to `select.expressions` on purpose: `find_all` on the whole
        select would walk into the FROM/JOIN clauses, where join keys are read
        bare by design -- `__m0.__p = __m1.__p` is a condition, not a value,
        and must not fail this test.
        """
        for proj in select.expressions:
            for col in proj.find_all(exp.Column):
                if col.table != alias:
                    continue
                node, agg = col.parent, False
                while node is not None and node is not select:
                    if isinstance(node, exp.AggFunc):
                        agg = True
                        break
                    node = node.parent
                if not agg:
                    return False
        return True

    still: list[tuple[str, str]] = []
    accepted: list[tuple[str, str, str, str]] = []
    for key in uncovered:
        jp = scope.joins[key]
        left, right = scope.derived.get(jp.a), scope.derived.get(jp.b)
        if left is None or right is None or not jp.pairs:
            still.append(key)
            continue
        a_cov, b_cov = _side_covered(jp.a), _side_covered(jp.b)

        ok = (a_cov and b_cov
              or (a_cov and _only_under_aggregate(jp.b))
              or (b_cov and _only_under_aggregate(jp.a)))
        if not ok:
            still.append(key)
        else:

            accepted.extend((jp.a, ca, jp.b, cb) for ca, cb in sorted(jp.pairs))
    return still, accepted

def _cover_joins(scope: _Scope, o: OntologyView, graph):
    """Explain every stated join as a declared edge. What is left is a violation.

    Composition is free: each hop is matched independently, so a governed path
    of any length is accepted exactly when every hop of it is declared.
    """
    insts = _instantiations(scope, o, graph)

    ok = [i for i in insts if i.role_ok]

    ok.sort(key=lambda i: (-len(i.pairs), -len(i.aux), i.edge.name))

    covered: set[tuple[str, str]] = set()
    resolved: list[_Resolved] = []
    for inst in ok:
        if any(p in covered for p in inst.pairs):
            continue
        covered.update(inst.pairs)
        resolved.append(inst)

    uncovered = []
    for k in scope.joins:
        if k in covered:
            continue
        half = _half_edge(k, scope, o)
        if half is not None:
            resolved.append(half)
            covered.add(k)
            continue
        uncovered.append(k)

    identities, uncovered = _identity_joins(scope, o, uncovered, resolved)

    role_missing: dict[tuple[str, str], list[_Resolved]] = defaultdict(list)
    for inst in insts:
        if inst.role_ok:
            continue
        for p in inst.pairs:
            if p in uncovered:
                role_missing[p].append(inst)

    return resolved, sorted(uncovered), role_missing, identities

def _identity_joins(scope: _Scope, o: OntologyView, uncovered, resolved):
    """Separate `C.key = C.key` from the joins that still need an edge.

    An ontology edge sanctions a TRAVERSAL: it says which concept you may reach
    from which, and at what cost in rows. A join whose two sides denote the SAME
    concept, matched on that concept's own key, traverses nothing -- it is the
    identity relation on C, and demanding an edge for it would demand an edge
    from C to itself. So no edge is required, and the property is established
    from the two relations' declared CONTENT (each side's column resolves,
    through the ontology, to the same concept and the same physical column, and
    together they cover a full key of that concept), never from an alias.

    SEVERAL CONCEPTS AT ONCE. The pairs are partitioned by the concept they
    resolve to and each partition must cover THAT concept's key. One join is
    then the identity on C1 AND the identity on C2, which traverses nothing
    either -- it is the conjunction of two relations that each traverse nothing.
    Requiring a single concept made a query grouped by two dimensions
    unexplainable: a measure pre-aggregated at the output grain joins back on
    every group key, and those keys belong to as many concepts as the question
    named.

    Declaredness is all this decides. What the join costs in ROWS is decided
    separately, in `_adjacency`, from each side's grain -- so a satellite that
    is not pre-aggregated is joined on the identity of its subject and STILL
    multiplies, and still fails the grain check.
    """
    alias_concepts = _alias_concepts(scope, resolved)
    found: list[_Identity] = []
    still: list[tuple[str, str]] = []
    for key in uncovered:
        jp = scope.joins[key]

        if not ({jp.a, jp.b} & set(scope.derived)) or not jp.pairs:
            still.append(key)
            continue
        matched: dict[str, set[str]] = defaultdict(set)
        cols_a: set[str] = set()
        cols_b: set[str] = set()
        ok = True
        for col_a, col_b in sorted(jp.pairs):
            pa = _resolve_prov(scope, o, jp.a, col_a, alias_concepts)
            pb = _resolve_prov(scope, o, jp.b, col_b, alias_concepts)
            if pa is None or pb is None:
                ok = False
                break
            if (pa.concept is None or pa.concept != pb.concept
                    or pa.column is None or pa.column != pb.column):
                ok = False
                break
            matched[pa.concept].add(pa.column)
            cols_a.add(col_a)
            cols_b.add(col_b)
        if not ok or not matched:
            still.append(key)
            continue
        concepts = sorted(matched)
        if any(not _concept_key(o, c) or not set(_concept_key(o, c)) <= matched[c]
               for c in concepts) and not _grain_aligned(scope, jp, cols_a, cols_b):
            still.append(key)
            continue
        found.append(_Identity(key=key, a=jp.a, b=jp.b, concepts=tuple(concepts),
                               cols_a=frozenset(cols_a), cols_b=frozenset(cols_b),
                               pairs=frozenset(jp.pairs)))
    return found, still

def _grain_aligned(scope: _Scope, jp, cols_a, cols_b) -> bool:
    """Both sides are relations whose WHOLE grain is the join columns.

    The second way a join can traverse nothing. `C.key = C.key` is the first:
    the key names the entity, so the join is the identity on C. This is the
    other one -- two relations each of which holds exactly one row per value of
    the joined columns, matched on those columns. Neither side can multiply the
    other (their declared grain IS the join), and neither reaches a concept the
    other did not already carry, because every pair resolves to the SAME
    concept and the SAME physical column.

    Why it is needed: a measure pre-aggregated at the answer's grain is grouped
    on the columns the answer groups by, and those are DISPLAY columns -- a
    catastrophe's name, a premium's amount -- not keys. Requiring a key would
    force the pre-aggregation onto the entity instead, and a group that merges
    two entities then adds two subtotals over overlapping rows. Both sides must
    be reduced relations, so this cannot sanction a join between base tables.
    """
    relation_a = scope.derived.get(jp.a)
    relation_b = scope.derived.get(jp.b)
    if relation_a is None or relation_b is None:
        return False
    return (relation_a.grain <= relation_a.refs(cols_a)
            and relation_b.grain <= relation_b.refs(cols_b))

def _half_edge(key, scope: _Scope, o: OntologyView) -> _Resolved | None:
    """A junction joined to ONE of its two endpoints, with the other absent.

    `WHERE claim_identifier IN (SELECT ... FROM claim_coverage JOIN detail ...)`
    is a governed traversal written as a semi-join: the junction is reached from
    the detail side and the claim side lives in another scope. A junction has no
    identity of its own -- it exists only as an edge's machinery -- so a join to
    one of its declared endpoints is a PREFIX of a governed path, and is accepted
    only while the other endpoint is absent. As soon as both endpoints are in the
    scope the full edge must match, so a half-edge can never be used to smuggle a
    wrong join between two concepts.
    """
    jp = scope.joins[key]
    tables = {a: scope.alias_table.get(a, "") for a in key}
    present = set(scope.alias_table.values())
    for e in o.edges:
        if not e.via_table or e.src not in o.concepts or e.dst not in o.concepts:
            continue
        st, dt = o.table_of(e.src), o.table_of(e.dst)
        for ja, oa in ((key[0], key[1]), (key[1], key[0])):
            if tables[ja] != e.via_table:
                continue
            if tables[oa] == st and dt not in present:
                if _match(scope, oa, ja, e.via_from):
                    if _role_ok_on(e, ja, scope):
                        return _Resolved(e, oa, ja, (), (key,), True)
            if tables[oa] == dt and st not in present:
                if _match(scope, ja, oa, e.via_to):
                    if _role_ok_on(e, ja, scope):
                        return _Resolved(e, ja, oa, (), (key,), True)
    return None

def _role_ok_on(e: Edge, ja: str, scope: _Scope) -> bool:
    if not e.role_predicate:
        return True
    return all(_has_predicate(scope, ja, _lower(c), str(v))
               for c, v in e.role_predicate.items())

def _instantiations(scope: _Scope, o: OntologyView, graph) -> list[_Resolved]:
    allows = getattr(graph, "allows_edge", None) if graph is not None else None
    out: list[_Resolved] = []
    by_table: dict[str, list[str]] = defaultdict(list)
    for a, t in scope.alias_table.items():
        by_table[t].append(a)

    for e in o.edges:
        if allows is not None and not allows(e.name):
            continue
        if e.src not in o.concepts or e.dst not in o.concepts:
            continue
        st, dt = o.table_of(e.src), o.table_of(e.dst)
        for sa in by_table.get(st, []):
            for da in by_table.get(dt, []):
                if sa == da:
                    continue
                out.extend(_instantiate(e, sa, da, scope, o, by_table))
    return out

def _instantiate(e: Edge, sa: str, da: str, scope, o: OntologyView, by_table) -> list[_Resolved]:
    pairs: list[tuple[str, str]] = []
    aux: list[str] = []

    if e.via_table:
        cands = by_table.get(e.via_table, [])
        results = []
        for ja in cands:
            if ja in (sa, da):
                continue
            p1 = _match(scope, sa, ja, e.via_from)
            p2 = _match(scope, ja, da, e.via_to)
            if p1 and p2:
                role_ok, concepts = _role_state(e, sa, da, ja, scope, o)
                results.append(_Resolved(e, sa, da, (ja,), (p1, p2), role_ok, concepts))
        return results

    p = _match(scope, sa, da, e.join)
    if not p:
        return []
    pairs.append(p)

    if e.restrict_table:
        found = None
        for ra in by_table.get(e.restrict_table, []):
            if ra in (sa, da):
                continue
            pr = _match(scope, da, ra, e.restrict_cols)
            if pr:
                found = (ra, pr)
                break
        if found is None:
            return []
        aux.append(found[0])
        pairs.append(found[1])

    role_ok, concepts = _role_state(e, sa, da, None, scope, o)
    return [_Resolved(e, sa, da, tuple(aux), tuple(pairs), role_ok, concepts)]

def _match(scope: _Scope, first: str, second: str, declared) -> tuple[str, str] | None:
    """Does the query's join between two aliases lie inside a declared edge?

    Subset, not equality: gold SQL routinely joins on part of a composite key
    (`claim_coverage` on the detail id but not its effective date). Extra
    equalities are NOT accepted -- they would make it a different relation.
    """
    if not declared:
        return None
    key = tuple(sorted((first, second)))
    jp = scope.joins.get(key)
    if jp is None or not jp.pairs:
        return None
    stated = jp.oriented(first)
    allowed = set(declared.items())
    if stated and stated <= allowed:
        return key  # type: ignore[return-value]
    return None

def _role_state(e: Edge, sa: str, da: str, ja: str | None, scope, o: OntologyView):
    """Is every role commitment this traversal needs actually stated?

    Two ways a role enters: the EDGE carries `role_predicate` (on its junction),
    or a role-object CONCEPT carries `backed_where` (on its own alias). Both are
    identity, not decoration -- so both must appear in the SQL.
    """
    needed: list[tuple[str, dict, str]] = []
    if e.role_predicate and ja:
        needed.append((ja, e.role_predicate, e.name))
    for concept, alias in ((e.src, sa), (e.dst, da)):
        bw = o.backed_where(concept)
        if bw:
            needed.append((alias, bw, concept))
    if not needed:
        return True, ()

    names: list[str] = []
    for alias, pred, label in needed:
        names.append(label)
        for col, val in pred.items():
            if not _has_predicate(scope, alias, _lower(col), str(val)):
                return False, tuple(names)
    return True, tuple(names)

def _has_predicate(scope: _Scope, alias: str, col: str, value: str) -> bool:
    for a, c, op, v, _node in scope.literals:
        if a == alias and c == col and op in ("=", "IN") and str(v) == value:
            return True
    return False

def _reached_multiplicity(scope: _Scope, target: str, columns) -> str:
    """How many rows of `target` one row on the other side can match.

    RULE 3, and the whole of it: a join is duplication-free exactly when its key
    COVERS the other side's grain. For a reduced relation the grain is declared
    by its own GROUP BY / DISTINCT, so a satellite aggregated to one row per
    group-key tuple is reached ONE-to-one on those keys, while the same
    satellite without the aggregation -- or aggregated at a finer grain -- is
    reached MANY. For a base table the join columns cover the concept's declared
    key, or `_identity_joins` would not have accepted the join at all.
    """
    relation = scope.derived.get(target)
    if relation is None:
        return _ONE
    return _ONE if relation.grain <= relation.refs(columns) else _MANY

def _adjacency(scope: _Scope, resolved: list[_Resolved], identities=()):
    """alias -> alias -> (multiplicity, label, columns used on the target).

    ALSO returns `idpairs`: every identity join's stated equalities, keyed by
    its alias pair, as (side, column, side, column) tuples. The static
    per-edge multiplicity above cannot see that a relation is pinned by the
    CONJUNCTION of several joins -- `__m1` is unique per (k0, p) where k0 comes
    from its join to the spine and p from its join to another satellite -- so
    the walkers recompute multiplicity from these pairs plus a fixpoint; see
    `_pinned_walk`.
    """
    adj: dict[str, dict[str, tuple[str, str, frozenset]]] = defaultdict(dict)
    idpairs: dict[frozenset, list[tuple[str, str, str, str]]] = defaultdict(list)
    for a in scope.alias_table:
        adj.setdefault(a, {})
    for i in identities:
        label = f"identity({'+'.join(i.concepts)})"
        adj[i.a][i.b] = (_reached_multiplicity(scope, i.b, i.cols_b), label, i.cols_b)
        adj[i.b][i.a] = (_reached_multiplicity(scope, i.a, i.cols_a), label, i.cols_a)
        for ca, cb in sorted((i.pairs or [])):
            idpairs[frozenset((i.a, i.b))].append((i.a, ca, i.b, cb))
    for r in resolved:
        adj[r.src_alias][r.dst_alias] = (r.edge.fan_out, r.edge.name, frozenset())
        adj[r.dst_alias][r.src_alias] = (r.edge.fan_out_reverse, r.edge.name, frozenset())

        principal = r.src_alias if r.edge.via_table else r.dst_alias
        for aux in r.aux:
            adj[principal][aux] = (_ONE, f"{r.edge.name}:aux", frozenset())
            adj[aux][principal] = (_ONE, f"{r.edge.name}:aux", frozenset())
    return adj, dict(idpairs)

def _pinned_walk(adj, idpairs, scope, root: str):
    """BFS over the join tree computing, per alias, whether it is PINNED.

    A relation is pinned when its whole grain is determined by values already
    fixed for one row of `root` -- directly by the root, or transitively
    through identity-join equalities to aliases that are themselves pinned.
    This is the CONJUNCTION of the tree's join conditions, which no per-edge
    multiplicity can express: `__m1` is one row per (k0, p), pinned by k0 from
    its spine join and p from its satellite join, yet each edge alone reads
    MANY and the static adjacency called the whole shape a fan-out.

    Returns (pinned, fanned_at, fixed) -- `fixed` is the set of provenance
    keys determined per root row, which the shared-value analysis uses to name
    the axis a value is replicated across.
    row meets at most ONE of its rows, and `fanned_at` carries the (u, v,
    label, columns) of edges where pinning FAILED -- the duplicating edges.

    Soundness direction matters and is the reason this is a fixpoint FROM THE
    ROOT: an alias pinned only through a neighbour that is itself multiplied
    is not pinned (the u1-v-u2 shape: v's b comes from u2, which one u1 row
    can meet many of, so v multiplies u1 even though v's grain is 'covered'
    by the conjunction). Pinning only propagates through edges whose source
    side is already pinned, which is exactly that restriction.
    """
    root_rel = scope.derived.get(root)
    fixed: set[tuple[str, str]] = set()
    if root_rel is not None:
        fixed = set(root_rel.refs(root_rel.columns.keys()))
    pinned: dict[str, bool] = {root: True}
    fanned_at: list[tuple[str, str, str, frozenset]] = []
    seen = {root}

    frontier_changed = True
    while frontier_changed:
        frontier_changed = False
        for pairkey, pairs in idpairs.items():
            for (sa, ca, sb, cb) in pairs:
                ra, rb = scope.derived.get(sa), scope.derived.get(sb)
                if ra is None or rb is None:
                    continue
                if pinned.get(sa) and ra.refs([ca]) and ra.refs([ca]) <= fixed:
                    new = rb.refs([cb]) - fixed
                    if new:
                        fixed |= new
                        frontier_changed = True
                if pinned.get(sb) and rb.refs([cb]) and rb.refs([cb]) <= fixed:
                    new = ra.refs([ca]) - fixed
                    if new:
                        fixed |= new
                        frontier_changed = True

        for alias, rel in scope.derived.items():
            if alias in pinned:
                continue
            if rel.grain and rel.grain <= fixed:
                pinned[alias] = True

                fixed |= set(rel.refs(rel.columns.keys()))
                frontier_changed = True

    bad: list[tuple[str, str, str, frozenset]] = []
    seen = {root}
    frontier = [root]
    while frontier:
        u = frontier.pop()
        for v, (mult, label, columns) in sorted(adj.get(u, {}).items()):
            if v in seen:
                continue
            seen.add(v)
            frontier.append(v)
            if mult != _ONE and not pinned.get(v):
                bad.append((u, v, label, columns))
    return pinned, bad, fixed

def _duplicating_edges(adj, root: str, scope=None, idpairs=None) -> list[tuple[str, str, str, frozenset]]:
    """Edges that duplicate `root`'s rows, walking the join tree away from it."""
    if scope is not None and idpairs is not None:
        _pinned, bad, _fixed = _pinned_walk(adj, idpairs, scope, root)
        return bad
    seen = {root}
    frontier = [root]
    bad: list[tuple[str, str, str, frozenset]] = []
    while frontier:
        u = frontier.pop()
        for v, (mult, label, columns) in sorted(adj.get(u, {}).items()):
            if v in seen:
                continue
            seen.add(v)
            frontier.append(v)
            if mult != _ONE:
                bad.append((u, v, label, columns))
    return bad

def _multiplied_from(adj, root: str, scope=None, idpairs=None) -> set[str]:
    """Aliases whose rows one row of `root` can meet MORE THAN ONE of.

    The mirror image of `_duplicating_edges`: same walk, but it returns the
    nodes rather than the edges, because the question here is not "is my
    measure duplicated" but "does my value belong to several of the rows I am
    about to be grouped by".
    """
    multiplied: set[str] = set()
    if scope is not None and idpairs is not None:
        pinned, _bad, _fixed = _pinned_walk(adj, idpairs, scope, root)
        seen = {root}
        frontier = [(root, False)]
        while frontier:
            u, fanned = frontier.pop()
            for v, (mult, _label, _columns) in sorted(adj.get(u, {}).items()):
                if v in seen:
                    continue
                seen.add(v)
                beyond = fanned or (mult != _ONE and not pinned.get(v))
                if beyond:
                    multiplied.add(v)
                frontier.append((v, beyond))
        return multiplied
    seen = {root}
    frontier: list[tuple[str, bool]] = [(root, False)]
    while frontier:
        u, fanned = frontier.pop()
        for v, (mult, _label, _columns) in sorted(adj.get(u, {}).items()):
            if v in seen:
                continue
            seen.add(v)
            beyond = fanned or mult != _ONE
            if beyond:
                multiplied.add(v)
            frontier.append((v, beyond))
    return multiplied

def _group_keys(select, scope) -> set[tuple[str, str]]:
    grp = select.args.get("group")
    out: set[tuple[str, str]] = set()
    if grp is None:
        return out
    for e in grp.expressions:
        ref = _colref(e, select, scope.alias_table, scope.schema)
        if ref:
            out.add(ref)
    return out

def _identifying_columns(o: OntologyView, concept: str) -> set[str]:
    c = o.concepts.get(concept) or {}
    cols: set[str] = set()
    key = c.get("key")
    for k in (key if isinstance(key, list) else [key] if key else []):
        cols.add(_lower(k))
    for aname, a in (c.get("attributes") or {}).items():
        if isinstance(a, Mapping) and a.get("column") and (
                aname == "id" or a.get("value_type") == "Identifier"):
            cols.add(_lower(a["column"]))
    return cols

def _alias_concepts(scope, resolved) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in resolved:
        out[r.src_alias].add(r.edge.src)
        out[r.dst_alias].add(r.edge.dst)
    for alias, relation in scope.derived.items():
        out[alias] |= set(relation.concepts)
    return out

def _absorbed_edge(v: str, columns, scope: _Scope, group_keys,
                   alias_concepts, o: OntologyView) -> bool:
    """Is the duplication on the way into `v` already accounted for?

    For a reduced relation this composes RULE 3 with the GROUP BY: after
    grouping, the rows of `v` a single row on the other side can meet are those
    agreeing on the join columns AND on the grouped columns, so the duplication
    disappears exactly when those together cover `v`'s grain. That is what makes
    a spine of group keys safe to join a satellite pre-aggregated on those same
    keys to: the join pins the grain, and each satellite row contributes its
    pre-aggregated value once per group. It says nothing about whether that
    value BELONGS to the group -- `_check_operand_grain` is that question.
    """
    relation = scope.derived.get(v)
    if relation is None:
        return _absorbed(v, group_keys, alias_concepts, o)
    have = set(columns) | {c for (a, c) in group_keys if a == v}
    return relation.grain <= relation.refs(have)

def _absorbed(v: str, group_keys, alias_concepts, o: OntologyView) -> bool:
    """Is this fan-out already accounted for by the GROUP BY?

    If the grouping fixes a KEY of the entity the traversal fans into, then
    within a group that entity contributes one row, so the measured rows are not
    repeated inside the group -- they are spread ACROSS groups, which is what
    grouping is for. Without this, `count(policies) group by holder_id` would be
    flagged for the perfectly correct reason that a policy has several holders.
    """
    for concept in alias_concepts.get(v, ()):  # noqa: SIM110
        ident = _identifying_columns(o, concept)
        if any(a == v and c in ident for (a, c) in group_keys):
            return True
    return False

def _has_cycle(adj) -> bool:
    seen: set[str] = set()
    for start in sorted(adj):
        if start in seen:
            continue
        stack = [(start, None)]
        local = {start}
        seen.add(start)
        while stack:
            u, parent = stack.pop()
            for v in sorted(adj.get(u, {})):
                if v == parent:
                    continue
                if v in local:
                    return True
                local.add(v)
                seen.add(v)
                stack.append((v, u))
    return False

def _check_grain(select, scope, o, resolved, identities, has_unexplained,
                 agg_pairs=()) -> list[Violation]:
    out: list[Violation] = []
    adj, idpairs = _adjacency(scope, resolved, identities)

    for sa, ca, sb, cb in agg_pairs:
        idpairs.setdefault(frozenset((sa, sb)), []).append((sa, ca, sb, cb))
        label = "aggregation-join"
        adj[sa].setdefault(sb, (_MANY, label, frozenset((cb,))))
        adj[sb].setdefault(sa, (_MANY, label, frozenset((ca,))))

    aggs = [n for n in select.find_all(exp.AggFunc) if n.find_ancestor(exp.Select) is select]
    windows = [n for n in select.find_all(exp.Window) if n.find_ancestor(exp.Select) is select]
    if windows:
        out.append(Violation(
            "GRAIN_UNDECIDABLE",
            "window functions are not analysed: the frame can re-grain the result in "
            "ways the join tree does not express.",
            windows[0].sql()[:160], severity="undecidable"))

    for sub in select.find_all(exp.Select):
        if sub is select:
            continue
        if _is_correlated(sub, scope.alias_table):
            out.append(Violation(
                "GRAIN_UNDECIDABLE",
                "a correlated subquery references this scope's aliases; its contribution "
                "to grain is not analysed.",
                sub.sql()[:160], severity="undecidable"))
            break

    projected = _projected_aliases(select, scope)
    if not aggs and len(projected) < 2:
        return out

    if has_unexplained:
        if aggs:
            out.append(Violation(
                "GRAIN_UNDECIDABLE",
                "the join tree contains a join with no declared edge, so its fan-out is "
                "unknown and no aggregate over this tree can be certified.",
                "", severity="undecidable"))
        return out

    if len(scope.alias_table) > 1 and _has_cycle(adj):

        def _equivalence_edge(label: str) -> bool:
            return label.startswith("identity(") or label == "aggregation-join"

        mult_adj = {u: {v: m for v, m in nbrs.items()
                        if not _equivalence_edge(m[1])}
                    for u, nbrs in adj.items()}
        if _has_cycle(mult_adj):
            out.append(Violation(
                "GRAIN_UNDECIDABLE",
                "the join graph is cyclic; row multiplicity is not determined by any "
                "single join tree, so grain cannot be certified.",
                "", severity="undecidable"))
            return out

    group_keys = _group_keys(select, scope)
    alias_concepts = _alias_concepts(scope, resolved)
    for agg in aggs:
        out.extend(_check_one_agg(agg, select, scope, adj, group_keys, alias_concepts, o, idpairs))
        out.extend(_check_operand_grain(agg, select, scope, group_keys, alias_concepts, o))

    for peak, group, labels in _cross_product_groups(
            adj, projected, alias_concepts, o, scope):
        out.append(Violation(
            "ROW_FANOUT",
            f"{', '.join('`' + g + '`' for g in group)} are projected on the same row "
            f"but none of them determines the others: `{peak}` fans out towards each, "
            f"along {', '.join(labels)}. Every output row is one arbitrary pairing -- an "
            f"accidental cross product, so each row's values do not belong together. "
            f"Aggregate each branch at its own grain before combining them.",
            ", ".join(group),
            detail={"aliases": group, "peak": peak, "edges": labels}))
    return out

def _projected_aliases(select, scope) -> dict[str, set[str]]:
    """alias -> the columns it contributes OUTSIDE any aggregate in SELECT."""
    out: dict[str, set[str]] = defaultdict(set)
    for item in select.expressions:
        e = item.this if isinstance(item, exp.Alias) else item
        if isinstance(e, exp.Star):
            continue
        for col in e.find_all(exp.Column):
            if col.find_ancestor(exp.AggFunc) is not None:
                continue
            if col.find_ancestor(exp.Select) is not select:
                continue
            ref = _colref(col, select, scope.alias_table, scope.schema)
            if ref:
                out[ref[0]].add(ref[1])
    return dict(out)

def _tree_path(adj, a: str, b: str) -> list[str] | None:
    parent: dict[str, str | None] = {a: None}
    queue = [a]
    while queue:
        u = queue.pop(0)
        if u == b:
            path = [u]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])  # type: ignore[arg-type]
            return list(reversed(path))
        for v in sorted(adj.get(u, {})):
            if v not in parent:
                parent[v] = u
                queue.append(v)
    return None

def _is_measure(alias, columns, alias_concepts, o: OntologyView, scope=None) -> bool:
    """Does this source contribute a QUANTITY to the row (not an identity)?

    NOT WEAKENED BY "it also projects its own key", and the attempt is recorded
    because it looked obviously right and was measured wrong. A deduplication
    stage that carries both an operand's key and its amount reads as a quantity
    here, so pairing it with a dimension amount across a fan-out was flagged --
    and the fix tried first was to call an IDENTIFIED source (one projecting a
    full key of its concept) not-a-quantity. That certifies this, which is a
    genuine 20-row arbitrary pairing of every loss amount against every premium
    amount of the same policy:

        select p.policy_number, ca.claim_amount_identifier, ca.claim_amount,
               pa.policy_amount from policy p ... join claim_amount ca ...
               join policy_amount pa ...

    One extra projected column flipped a demonstrably multiplied result from
    rejected to certified. The compiler was changed instead: its bridge stage
    now projects identities only (see `compile._satellite`), so nothing it emits
    needs this rule relaxed.
    """
    relation = scope.derived.get(alias) if scope is not None else None
    if relation is not None:
        for name in columns:
            p = relation.columns.get(name)
            if p is None:
                continue
            if p.agg is not None:
                return True
            if _numeric_attribute(o, p.concept, p.column):
                return True
        return False
    for concept in alias_concepts.get(alias, ()):
        attrs = (o.concepts.get(concept) or {}).get("attributes") or {}
        for a in attrs.values():
            if not isinstance(a, Mapping) or not a.get("column"):
                continue
            if _lower(a["column"]) not in columns:
                continue
            if a.get("type") in ("numeric", "number", "decimal", "integer") or \
                    a.get("value_type") == "Money":
                return True
    return False

def _numeric_attribute(o: OntologyView, concept, column) -> bool:
    attrs = (o.concepts.get(concept or "") or {}).get("attributes") or {}
    for a in attrs.values():
        if not isinstance(a, Mapping) or not a.get("column"):
            continue
        if _lower(a["column"]) != column:
            continue
        if a.get("type") in ("numeric", "number", "decimal", "integer") or \
                a.get("value_type") == "Money":
            return True
    return False

def _cross_product_groups(adj, projected, alias_concepts, o: OntologyView, scope=None):
    """Sets of projected sources joined across a fan-out PEAK.

    Sound on a tree for the pairs it considers: a peak means neither source's
    row determines the other's, so the pairing is arbitrary. Grouped by peak so
    a repair prompt gets one instruction, not n-choose-2 of them.
    """
    ordered = sorted(projected)
    groups: dict[str, tuple[set[str], set[str]]] = {}
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            path = _tree_path(adj, a, b)
            if not path or len(path) < 3:
                continue
            peak = None
            labels: set[str] = set()
            for k in range(1, len(path) - 1):
                c = path[k]
                left = adj[c].get(path[k - 1])
                right = adj[c].get(path[k + 1])
                if left and right and left[0] == _MANY and right[0] == _MANY:
                    peak, labels = c, {left[1], right[1]}
                    break
            if peak is None:
                continue
            same_concept = bool(alias_concepts.get(a, set()) & alias_concepts.get(b, set()))
            both_measures = (_is_measure(a, projected[a], alias_concepts, o, scope)
                             and _is_measure(b, projected[b], alias_concepts, o, scope))
            if not (same_concept or both_measures):
                continue
            g, lab = groups.setdefault(peak, (set(), set()))
            g.update((a, b))
            lab.update(labels)
    return [(peak, sorted(g), sorted(lab)) for peak, (g, lab) in sorted(groups.items())]

def _is_distinct_agg(agg) -> bool:
    """COUNT(DISTINCT x) / SUM(DISTINCT x). sqlglot models the DISTINCT as the
    aggregate's argument, not as a flag, so both spellings are checked."""
    if agg.args.get("distinct"):
        return True
    return isinstance(agg.this, exp.Distinct)

def _check_one_agg(agg, select, scope, adj, group_keys, alias_concepts, o, idpairs=None) -> list[Violation]:
    if isinstance(agg, _INSENSITIVE_AGGS):
        return []
    if _is_distinct_agg(agg):

        return []
    if not isinstance(agg, _SENSITIVE_AGGS):
        return [Violation(
            "GRAIN_UNDECIDABLE",
            f"aggregate `{agg.sql()[:60]}` is not in the checker's duplication-sensitivity "
            f"table, so its grain is not certified.",
            agg.sql()[:120], severity="undecidable")]

    star = isinstance(agg, exp.Count) and isinstance(agg.this, exp.Star)
    measures = set(scope.alias_table) if star else _measure_aliases(agg, select, scope)

    if not measures:
        if len(scope.alias_table) <= 1:
            return []
        return [Violation(
            "GRAIN_UNDECIDABLE",
            f"the columns of `{agg.sql()[:60]}` could not be resolved to a source, so its "
            f"grain is not certified.",
            agg.sql()[:120], severity="undecidable")]

    offenders: list[tuple[str, str, str, frozenset]] = []
    for m in sorted(measures):
        offenders += [b for b in _duplicating_edges(adj, m, scope, idpairs)
                      if not _absorbed_edge(b[1], b[3], scope, group_keys,
                                            alias_concepts, o)]
    if not offenders:
        return []

    names = sorted({b[2] for b in offenders})
    subject = "the joined row" if star else "`" + "`, `".join(sorted(measures)) + "`"
    return [Violation(
        "GRAIN_FANOUT",
        f"`{agg.sql()[:70]}` aggregates {subject} at a grain the join tree does not "
        f"preserve: edge(s) {', '.join(names)} declare fan-out, so each measured row is "
        f"repeated and the result is inflated. Aggregate at the measure's own grain "
        f"(pre-aggregate, or COUNT/SUM DISTINCT on a key) before joining the fan-out.",
        agg.sql()[:200],
        detail={"measure_aliases": sorted(measures), "fanout_edges": names,
                "offending_traversals": [f"{u}->{v} ({lab})"
                                         for u, v, lab, _cols in offenders]})]

def _check_operand_grain(agg, select, scope, group_keys, alias_concepts, o) -> list[Violation]:
    """RULE 3b -- the DUAL of GRAIN_FANOUT, and the one that was missing.

    GRAIN_FANOUT asks whether the join tree DUPLICATED the rows an aggregate
    ran over. That question is silent about the shape this project's own
    compiler emits, because a pre-aggregated relation joined on its own grain
    duplicates nothing. The number can still be wrong, and in two ways -- both
    executed, both certified `[]` by this checker until 2026-08-12:

        OUTPUT FINER THAN THE OPERAND'S GRAIN.  A measure pre-aggregated per
        POLICY, joined to a spine and grouped by (policy, claim), reports the
        policy's 13600 beside every claim of the policy. Nothing is duplicated:
        one satellite row meets one spine row per group. The value simply is
        not a quantity of the group.  Flagged when a group key names a concept
        the relation's own join tree REACHES -- so it could have been computed
        one grain finer -- but its grain does not pin.  A group key the
        relation never touches is NOT flagged: "loss by policy and holder" is a
        roll-up over a many-to-many, where the policy's loss beside each holder
        is the only answer there is, and refusing it would refuse most
        dimensional questions.

        OUTPUT COARSER THAN THE OPERAND'S GRAIN.  A premium that belongs to one
        coverage detail, pre-aggregated per CLAIM and then summed over the
        claims of a policy, is added once per claim: a 40000 denominator for a
        20000 premium.  Flagged from `_Prov.shared`, recorded in the scope that
        could still see the fan-out, whenever the group keys do not pin the
        relation's grain -- i.e. whenever more than one of its rows can land in
        one group.

    Both are decided from DECLARED fan-out and declared keys, never from the
    data, and both are restricted to aggregates whose operand comes from a
    reduced relation. Where the operand is a base-table column the multiplicity
    question is the one `_check_one_agg` already answers.
    """
    if isinstance(agg, _INSENSITIVE_AGGS) or _is_distinct_agg(agg):
        return []
    if not isinstance(agg, _SENSITIVE_AGGS):
        return []

    key_columns: dict[str, set[str]] = defaultdict(set)
    for alias, column in group_keys:
        prov = _resolve_prov(scope, o, alias, column, alias_concepts)
        if prov is not None and prov.concept and prov.column:
            key_columns[prov.concept].add(prov.column)

    operands: list[tuple[str, _Derived, _Prov]] = []
    for col in agg.find_all(exp.Column):
        if col.find_ancestor(exp.Select) is not select:
            continue
        ref = _colref(col, select, scope.alias_table, scope.schema)
        if ref is None:
            continue
        relation = scope.derived.get(ref[0])
        prov = relation.columns.get(ref[1]) if relation is not None else None
        if relation is not None and prov is not None:
            operands.append((ref[0], relation, prov))

    out: list[Violation] = []
    for alias, relation, prov in operands:
        grain_concepts = {concept for concept, _column in relation.grain_concepts}

        finer = sorted(concept for concept in key_columns
                       if prov.agg is not None
                       and concept in relation.refinable_concepts
                       and concept not in grain_concepts)
        if finer:
            out.append(Violation(
                "GRAIN_MISMATCH",
                f"`{agg.sql()[:70]}` reads a value `{alias}` computed one row per "
                f"{'+'.join(sorted(grain_concepts)) or '(the whole relation)'}, but the "
                f"answer is grouped one row per {'+'.join(sorted(key_columns))}. "
                f"{', '.join(finer)} is on `{alias}`'s own join tree, so the measure "
                f"could have been computed at that finer grain and was not -- the "
                f"coarser total is repeated beside every one of them instead. "
                f"Re-group the pre-aggregation on the answer's own keys.",
                agg.sql()[:200],
                detail={"relation": alias, "operand_grain": sorted(grain_concepts),
                        "group_grain": sorted(key_columns), "reachable_finer": finer}))
            continue
        if not prov.shared:
            continue

        unpinned = sorted(
            concept for concept, column in relation.grain_concepts
            if concept in prov.shared
            and column not in key_columns.get(concept, set())
            and not (_identifying_columns(o, concept) & key_columns.get(concept, set())))
        if unpinned:
            out.append(Violation(
                "GRAIN_MISMATCH",
                f"`{agg.sql()[:70]}` adds up a value that belongs to a COARSER grain "
                f"than the rows it is stored on: `{alias}` holds it one row per "
                f"{'+'.join(unpinned)}, and the same underlying row is reached from "
                f"several of them, so grouping to "
                f"{'+'.join(sorted(key_columns)) or 'the whole result'} adds it once per "
                f"{'+'.join(unpinned)} instead of once. Pre-aggregate it at the grain it "
                f"is a quantity OF, deduplicated, before combining.",
                agg.sql()[:200],
                detail={"relation": alias, "operand_grain": sorted(unpinned),
                        "group_grain": sorted(key_columns), "shared": True}))
    return out

def _measure_aliases(node, select, scope) -> set[str]:
    out: set[str] = set()
    for col in node.find_all(exp.Column):
        if col.find_ancestor(exp.Select) is not select:
            continue
        ref = _colref(col, select, scope.alias_table, _scope_schema(scope))
        if ref:
            out.add(ref[0])
    return out

def _scope_schema(scope: _Scope) -> dict[str, set[str]]:
    return scope.schema

def _is_correlated(sub, outer_aliases) -> bool:
    inner = set()
    frm = _arg(sub, "from_", "from")
    if frm is not None:
        t = frm.this if isinstance(frm, exp.From) else frm
        if isinstance(t, exp.Table):
            inner.add(_lower(t.alias_or_name))
    for j in sub.args.get("joins") or []:
        inner.add(_lower(j.this.alias_or_name))
    for col in sub.find_all(exp.Column):
        tbl = _lower(col.table)
        if tbl and tbl in outer_aliases and tbl not in inner:
            return True
    return False

def _check_metrics(select, scope, o, resolved, expect_metrics, strict_op,
                   computed_elsewhere=frozenset()) -> list[Violation]:
    out: list[Violation] = []
    incoming = _incoming_edges(resolved)

    claims: list[tuple[str, exp.Expression, str]] = []
    for item in select.expressions:
        if not isinstance(item, exp.Alias):
            continue
        metric = o.metric_by_phrase.get(_squash(item.alias_or_name))
        if metric:
            claims.append((metric, item.this, item.alias_or_name))

    for name in expect_metrics or ():
        if name not in o.metrics:
            continue
        if any(c[0] == name for c in claims):
            continue
        cand = _best_expression_for(select, o, name)
        if cand is not None:
            claims.append((name, cand, f"<expected {name}>"))
        elif name not in computed_elsewhere:

            out.append(Violation(
                "METRIC_FIDELITY",
                f"the question resolves to governed metric `{name}` but no expression in "
                f"the query computes it.",
                select.sql()[:160], detail={"metric": name}))

    for metric, expr, label in claims:
        problems = _metric_problems(expr, metric, o, scope, incoming, select, strict_op)
        for problem in problems:
            out.append(Violation(
                "METRIC_FIDELITY",
                f"`{label}` claims governed metric `{metric}` but {problem} "
                f"The registry defines it once; a query may use it, not redefine it.",
                expr.sql()[:200], detail={"metric": metric}))
    return out

def _metric_vias(o: OntologyView, metric: str, seen=None) -> set[str]:
    """Every governed edge the metric's definition bottoms out on."""
    seen = seen or set()
    if metric in seen or metric not in o.metrics:
        return set()
    seen.add(metric)
    m = o.metrics[metric]
    out: set[str] = set()
    op = m.get("operand") or {}
    if op.get("via"):
        out |= o.same_kind(op["via"])
    for comp in m.get("components") or []:
        out |= _metric_vias(o, comp, seen)
    return out

def _best_expression_for(select, o: OntologyView, metric: str):
    """Which SELECT item is this query's attempt at `metric`?

    Used only in `expect_metrics` mode, where the caller already knows the
    question resolved to the metric and the only question is whether the SQL
    computes it faithfully. The candidate is the projection touching the most
    aliases the metric's own edges reach -- so a query that reached the amounts
    at all is judged on HOW, and a query that never reached them is reported as
    not computing the metric.
    """
    vias = _metric_vias(o, metric)
    if not vias:
        return None
    best, best_score = None, 0
    for item in select.expressions:
        e = item.this if isinstance(item, exp.Alias) else item
        if isinstance(e, (exp.Column, exp.Star)):
            continue
        score = 0
        for col in e.find_all(exp.Column):
            tbl = _lower(col.table)
            if tbl:
                score += 1
        if score > best_score:
            best, best_score = e, score
    return best

def _incoming_edges(resolved) -> dict[str, set[str]]:
    inc: dict[str, set[str]] = defaultdict(set)
    for r in resolved:
        inc[r.dst_alias].add(r.edge.name)
        inc[r.src_alias].add(r.edge.name + "!reverse")
    return inc

def _metric_problems(expr, metric, o, scope, incoming, select, strict_op) -> list[str]:
    m = o.metrics.get(metric) or {}
    inner = _strip_aggs(expr)

    if m.get("combine") == "add":
        components = list(m.get("components") or [])
        terms = _flatten_add(inner)
        got = [_leaf_metric_of(t, o, scope, incoming, select) for t in terms]
        if sorted(x or "?" for x in got) != sorted(components):
            missing = sorted(set(components) - {g for g in got if g})
            extra = sorted([g or "an ungoverned term" for g in got
                            if g not in components or got.count(g) > components.count(g)])
            return [f"its parts are {sorted(g or '?' for g in got)}, not the governed "
                    f"{sorted(components)} (missing: {missing or 'none'}; "
                    f"unexpected: {extra or 'none'})."]
        return []

    if m.get("combine") == "divide":
        components = list(m.get("components") or [])
        if not isinstance(inner, exp.Div):
            return [f"it is not a division; `{metric}` is "
                    f"{components[0]} / {components[1]}."]
        bad = []
        for side, comp in ((inner.left, components[0]), (inner.right, components[1])):
            bad += [f"its {'numerator' if comp == components[0] else 'denominator'} "
                    f"is not {comp}: {p}"
                    for p in _metric_problems(side, comp, o, scope, incoming, select, strict_op)]
        return bad

    operand = m.get("operand") or {}
    via = operand.get("via")
    attr_col = o.attr_column(operand.get("concept", ""), operand.get("attribute", ""))
    cols = [c for c in inner.find_all(exp.Column)
            if c.find_ancestor(exp.Select) is select]
    if not cols:
        return [f"it references no column of {operand.get('concept')}."]
    problems = []
    for c in cols:
        ref = _colref(c, select, scope.alias_table, _scope_schema(scope))
        if ref is None:
            problems.append(f"column `{c.sql()}` could not be resolved to a source.")
            continue
        alias, col = ref

        base_col, vias = _column_origin(scope, alias, col, incoming)
        if attr_col and base_col != attr_col:
            problems.append(f"it reads `{base_col}`, not {operand.get('concept')}."
                            f"{operand.get('attribute')} (`{attr_col}`).")
            continue
        if via and not (o.same_kind(via) & vias):
            reached = sorted(e for e in vias if not e.endswith("!reverse"))
            problems.append(f"`{alias}` is reached by {reached or ['no governed edge']}, "
                            f"not by the governed `{via}`.")
    if strict_op and m.get("op"):
        want = m["op"]
        agg = expr if isinstance(expr, exp.AggFunc) else next(
            iter(expr.find_all(exp.AggFunc)), None)
        got = _agg_name(agg)
        if got and got != want:
            problems.append(f"it aggregates with {got}, not the governed {want}.")
    return problems

def _column_origin(scope: _Scope, alias: str, column: str, incoming):
    """(physical column, governed edges that reached it) for a column reference.

    Identical for a base-table column and for a column a reduced relation hands
    out, which is what lets metric fidelity survive pre-aggregation.
    """
    relation = scope.derived.get(alias)
    if relation is not None:
        p = relation.columns.get(column)
        if p is not None:
            return (p.column or column), set(p.vias)
        return column, set()
    return column, set(incoming.get(alias, set()))

def _agg_name(agg) -> str | None:
    if agg is None:
        return None
    if isinstance(agg, exp.Count):
        return "count_distinct" if agg.args.get("distinct") else "count"
    return type(agg).__name__.lower()

def _null_guard(expr):
    """The guarded expression inside a null-guard, or None.

    Three wrappers mean "the same quantity, defined where it is defined":

        CASE WHEN <d> = 0 THEN NULL ELSE <x> END     divide-by-zero guard
        CAST(<x> AS REAL)                            integer-division guard
        NULLIF(<x>, 0)                               the same guard, inline

    They are not stylistic. `compile._safe_divide` emits the first two because
    ACME's LossRatio returned 0 without them: SQLite divides integers as
    integers, and a zero denominator is a data condition rather than a plan
    error. Anything strictly wider than this -- a CASE with a non-null branch,
    a CAST that changes what is counted -- is NOT peeled, because that would be
    a query saying something different about the metric.
    """
    if isinstance(expr, exp.Cast):
        return expr.this
    if isinstance(expr, exp.Nullif):
        return expr.this
    if isinstance(expr, exp.Case) and not expr.args.get("default") is None:
        ifs = expr.args.get("ifs") or []
        if len(ifs) == 1 and isinstance(ifs[0].args.get("true"), exp.Null):
            return expr.args["default"]
    return None

def _strip_aggs(expr):
    """Peel down to the expression a metric claim is really about.

    Aggregates, parentheses and NULL-GUARDS (`_null_guard`). The guards were
    missing, and their absence was the compile->check contract failing on the
    compiler's OWN output: `compile._safe_divide` wraps every governed ratio in
    `CASE WHEN <denominator> = 0 THEN NULL ELSE CAST(<numerator> AS REAL) /
    <denominator> END`, and `_metric_problems` then asked `isinstance(inner,
    exp.Div)` of a `exp.Case` and answered "it is not a division". Every
    `CHECKER_VIOLATION:METRIC_FIDELITY` in `results/camp_haiku_t0_n5.jsonl` is
    that sentence, on SQL that computes LossRatio exactly as authored.

    It survived because no GOLD plan names `LossRatio`: gold writes the ratio as
    a `combine` measure, whose derived output label is `TotalLoss_per_...`, and
    the alias-driven claim never fired. The one construction the checker could
    not certify was the one only the compiler emits.
    """
    e = expr
    while True:
        if isinstance(e, exp.Paren):
            e = e.this
        elif isinstance(e, exp.AggFunc) and e.this is not None and not isinstance(e.this, exp.Star):
            e = e.this
        else:
            guarded = _null_guard(e)
            if guarded is None:
                return e
            e = guarded

def _flatten_add(node) -> list[exp.Expression]:
    if isinstance(node, exp.Paren):
        return _flatten_add(node.this)
    if isinstance(node, exp.Add):
        return _flatten_add(node.left) + _flatten_add(node.right)
    return [node]

def _leaf_metric_of(term, o, scope, incoming, select) -> str | None:
    """Which governed leaf metric does this term compute -- by the EDGE that
    reached its alias, which is where amount kind lives in this ontology."""
    inner = _strip_aggs(term)
    cols = [c for c in inner.find_all(exp.Column) if c.find_ancestor(exp.Select) is select]
    if len(cols) != 1:
        return None
    ref = _colref(cols[0], select, scope.alias_table, _scope_schema(scope))
    if ref is None:
        return None
    _base_col, vias = _column_origin(scope, ref[0], ref[1], incoming)
    for e in sorted(vias):
        if e in o.metric_by_via:
            return o.metric_by_via[e]
    return None

def _check_literals(scope: _Scope, o: OntologyView, db) -> list[Violation]:
    equalities = [(a, c, op, v, n) for (a, c, op, v, n) in scope.literals
                  if op in ("=", "IN", "LIKE")]
    if not equalities:
        return []
    if db is None:
        return [Violation(
            "UNGROUNDED_LITERAL",
            "no database was supplied, so literals could not be grounded. DESIGN Rule 2: "
            "canonical values are PROBED, never asserted -- an unprobed literal is how a "
            "query came to filter 'deputy' against data holding 'Deputy'.",
            "; ".join(n.sql() for *_x, n in equalities)[:200], severity="undecidable")]

    out: list[Violation] = []
    con = sqlite3.connect(str(db))
    try:
        for alias, col, op, val, node in equalities:
            table = scope.alias_table.get(alias)
            if not table or table.startswith("<derived"):
                continue
            try:
                cols = {r[1].lower() for r in con.execute(f'PRAGMA table_info("{table}")')}
            except sqlite3.Error:
                continue
            if col not in cols:
                continue
            pred = f'"{col}" LIKE ?' if op == "LIKE" else f'"{col}" = ?'
            try:
                hit = con.execute(
                    f'SELECT 1 FROM "{table}" WHERE {pred} LIMIT 1', (val,)).fetchone()
            except sqlite3.Error:
                continue
            if hit:
                continue
            near = None
            if isinstance(val, str):
                row = con.execute(
                    f'SELECT "{col}" FROM "{table}" WHERE lower("{col}") = lower(?) LIMIT 1',
                    (val,)).fetchone()
                if row:
                    near = row[0]
                else:
                    row = con.execute(
                        f'SELECT DISTINCT "{col}" FROM "{table}" '
                        f'WHERE "{col}" IS NOT NULL AND "{col}" <> "" LIMIT 4').fetchall()
                    near = ", ".join(str(r[0]) for r in row) or None
            hint = f" Did you mean `{near}`?" if near else ""
            out.append(Violation(
                "UNGROUNDED_LITERAL",
                f"`{table}.{col}` holds no row matching {val!r}, so this predicate "
                f"silently returns nothing.{hint}",
                node.sql()[:200],
                detail={"table": table, "column": col, "value": val, "candidates": near}))
    finally:
        con.close()
    return out

SOUNDNESS = """
Per-check honesty statement. "Sound" means: no false negatives WITHIN the stated
shape; anything outside the shape raises severity="undecidable" instead of
passing. Measured on the 24 ACME gold queries and on mutation classes built from
them -- see spc/tests/test_check.py, which prints the numbers.

1 UNDECLARED_JOIN -- SOUND for single-block SELECTs over base tables: aliases,
  self-joins, junction (`via`) tables, kind (`restrict`) tables, WHERE-clause
  joins, CROSS JOINs, and CTEs/subqueries that reduce to one base table or to a
  RELATION (see the reduction below). Every stated equality between two sources
  must lie inside exactly one declared edge's column map, or be an IDENTITY
  JOIN. A subquery that reduces to neither raises SHAPE_UNDECIDABLE; it is never
  silently accepted. A junction joined to one endpoint while the other is out of
  scope (a semi-join) is accepted as a governed PREFIX -- but only while the far
  endpoint is absent, so it cannot be used to smuggle a wrong join between two
  concepts.
  IDENTITY JOIN: accepted when at least one side is a reduced relation, every
  stated equality pairs a column with ITSELF in ontology terms (same concept,
  same physical column, resolved through each side's provenance), and the
  matched columns cover a full declared key of that concept. Partial-key and
  same-concept-different-column joins are NOT identity joins and are still
  reported. Declaredness is all this decides: the join's cost in ROWS is decided
  separately from grain, so an identity join to a relation that was not
  pre-aggregated is accepted as a join and still fails the grain check.

  DERIVED-TABLE REDUCTION (what makes composition analysable at all). A CTE or
  subquery reduces to a relation -- per-column provenance (concept, physical
  column, the governed edges that reached it, any aggregation) plus a GRAIN --
  when BOTH hold: every join inside it is a declared edge, and its row
  multiplicity is ESTABLISHED by its own structure (a GROUP BY whose every key
  is also projected, a SELECT DISTINCT, or aggregates with no GROUP BY, which is
  one row). `*`, HAVING, LIMIT, an unresolvable projection, a cyclic or
  disconnected inner tree, or any ungoverned join inside it all refuse the
  reduction, and the relation stays opaque -- SHAPE_UNDECIDABLE, as before.
  Nothing about a CTE's NAME or its position enters this: a relation is reduced
  for what it declares about itself.
  BEST-EFFORT: non-equality join predicates are reported as undeclared rather
  than analysed. Set operations are checked branch by branch, not as a whole.

2 UNCOMMITTED_ROLE -- SOUND for the ACME role pattern: several edges/concepts
  over one physical join distinguished by a role code. Fires when a join matches
  only role-bearing candidates and no committing equality/IN is present on the
  role-bearing alias. Covers both spellings of role in this ontology: an edge
  `role_predicate` on its junction, and a role-object concept's `backed_where`.
  BEST-EFFORT: a role committed by anything other than a literal equality on the
  role column -- a subquery, a CASE, or a join to Party_Role by NAME -- is not
  recognised as a commitment (the last one is reported, because Party_Role is
  outside the governed surface). A role predicate placed in the ON clause of a
  LEFT JOIN does not actually filter, and that distinction is not modelled.

3 GRAIN_FANOUT / ROW_FANOUT -- SOUND for acyclic, fully-explained join trees of
  base tables, using DECLARED fan-out.
  * aggregates: SUM / AVG / COUNT / COUNT(*) / GROUP_CONCAT are flagged when any
    edge leading AWAY from the measured source declares fan-out. MIN, MAX and
    any DISTINCT aggregate are exempt because duplication cannot change their
    value -- an operator fact, not a heuristic. A fan-out whose target key is in
    the GROUP BY is absorbed, because grouping spreads those rows across groups
    instead of repeating them inside one.
  * row level: two projected sources whose join path has a PEAK (a node fanning
    out towards both) are an accidental cross product. Restricted to pairs that
    are two readings of the SAME concept or two MEASURES, because pairing
    identities across a fan-out is a legitimate listing ("policy holders and the
    claims they made") while pairing quantities makes the ROW false.
  * COMPOSITION: a join into a reduced relation multiplies rows exactly when its
    key does NOT cover that relation's grain, and that duplication is absorbed
    exactly when the join columns TOGETHER WITH the GROUP BY do cover it. This
    is the same coverage fact stated twice, and it is what certifies a spine of
    group keys joined to satellites pre-aggregated on those same keys: the join
    pins the grain and each satellite contributes its value once per group. A
    satellite that is not pre-aggregated, or aggregated finer than the key it is
    joined on, fails the same test. Grain is deduplicated by functional
    dependency first: a relation that hands one concept's key out twice under
    two names is not finer for it.
  * GRAIN_MISMATCH: the DUAL of the above, and the check that was missing until
    2026-08-12. Duplication is not the only way a pre-aggregated number goes
    wrong: it can be computed at a grain that is not the answer's and be
    repeated across groups (coarser than the GROUP BY, while a finer grain was
    available in its own join tree), or be a value REPLICATED across the rows it
    is stored on and then added up (finer than what it is a quantity of). Both
    are decided from declared fan-out and declared keys, both are restricted to
    aggregates whose operand comes from a reduced relation -- where the operand
    is a base column, duplication is the whole question and the rules above
    answer it.
  UNDECIDABLE, never passed: window functions, correlated subqueries, cyclic
  join graphs, unresolvable columns, any tree containing an undeclared join, any
  derived table whose grain is not established, and any aggregate outside the
  sensitivity table.
  CONSERVATIVE ON PURPOSE: fan-out is read from the ontology's declaration, not
  measured on a snapshot, so an edge declared `bounded` counts as unsafe even
  where today's data holds one row. A checker that trusted the tiny fixture
  would certify exactly the queries `acme_cf_fanout.sqlite` breaks.
  KNOWN GAP: aggregates over an IRREDUCIBLE derived table are still checked
  within their own scope only; the composition across those scopes is raised as
  SHAPE_UNDECIDABLE / GRAIN_UNDECIDABLE, never passed.

4 METRIC_FIDELITY -- BEST-EFFORT by construction, in two modes.
  With `expect_metrics` (what the pipeline has, having resolved the question
  through the glossary) it is sound for the registry shapes ACME uses:
  `combine: add` over leaf metrics, `combine: divide`, and leaf
  `op + operand{concept, attribute, via}`. Composition, amount KIND (which edge
  reached the alias) and operand column are all checked. Both facts survive
  pre-aggregation: a column a reduced relation hands out carries the physical
  column it aggregated and the governed edge that reached it, so
  `SUM(m.v) + SUM(m2.v)` over four per-kind relations is judged as the four
  kinds, not as four anonymous numbers.
  Without it, the check fires only when a SELECT alias NAMES a metric or a
  glossary phrase for one, normalised (`Total_Loss` == `total loss`). An
  improvised variant under a neutral alias (`as LossAmount`, `as NoOfClaims`) is
  NOT caught -- the price of not guessing intent from a query that never states
  it.
  Operator fidelity (count vs count_distinct) is off unless
  `strict_metric_op=True`, because 'average loss' is a legitimate AVG over a
  metric the registry defines with SUM.

5 UNGROUNDED_LITERAL -- SOUND for `=`, `IN` and `LIKE` against a resolvable
  base-table column, probed against the database. Without `db` it returns
  UNDECIDABLE rather than passing.
  KNOWN GAPS: range predicates (`>=`, `<`, BETWEEN) are not probed, so an empty
  date window is not caught; literals compared to an EXPRESSION
  (`upper(col) = 'X'`) are not checked; and a column renamed by a subquery
  cannot be traced back to its base column, so it is skipped.

NAMED HOLES -- unsafe query classes this checker CANNOT catch:
  * WRONG-BUT-COMMITTED ROLE. `party_role_code = 'AG'` where the question meant
    the policy holder is fully governed SQL. Only the question can decide it;
    the checker sees a legal traversal. Measured catch rate: 0%.
  * A GOVERNED QUERY THAT ANSWERS A DIFFERENT QUESTION. The checker validates
    SQL against the ontology, never against intent.
  * ONTOLOGY GAPS READ AS VIOLATIONS. If the ontology omits an edge the data
    supports, correct SQL using it is reported UNDECLARED_JOIN, and nothing here
    can tell that from bad SQL: the checker is right about the ontology and
    wrong about the query. The class is structural and cannot be closed from
    inside this file; a specific INSTANCE can only be closed in the knowledge
    layer. Two of the 24 ACME gold queries were exactly this -- the DDL declares
    Policy_Amount.Policy_Identifier -> Policy and acme.yaml did not -- and both
    cleared when the edge was declared, with no change here. They were the only
    two false positives on gold; there are now none.
  * ROW-LEVEL IDENTITY CROSS PRODUCTS. Pairing two identity columns across a
    fan-out peak is deliberately allowed; if the question wanted one row per
    entity, the extra rows are wrong and nothing here says so.
"""
