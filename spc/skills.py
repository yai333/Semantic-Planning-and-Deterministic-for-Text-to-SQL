from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, is_dataclass
from pathlib import Path as FsPath
from typing import Any, Callable, Mapping, Sequence

from spc.graph import DEFAULT_MAX_HOPS, PathGraph, path_signature
from spc.menu import SUBJECT_ID, Menu, MenuRoute
from spc.menu import parse_pick as _menu_parse_pick
from spc.ontology import (
    Attribute,
    Concept,
    Edge,
    Metric,
    Ontology,
    load_ontology,
)
from spc.plan import Path, Plan

__all__ = [
    "SkillResult",
    "SkillSpec",
    "Skills",
    "Snapshot",
    "VALUE_LADDER",
    "SKILL_SPECS",
    "SKILL_NAMES",
    "PICK_SKILL_SPECS",
    "PICK_SKILL_NAMES",
    "ROUTE_ID_MAX_HOPS",
    "RouteIdError",
    "call",
    "call_pick",
    "default_skills",
    "route_id",
    "parse_route_id",
    "find_paths",
    "describe_concept",
    "list_concepts",
    "list_subjects",
    "search_concepts",
    "find_metric",
    "find_attribute",
    "search_values",
    "check_sql",
]

_REPO_ROOT = FsPath(__file__).resolve().parent.parent

DEFAULT_DB = _REPO_ROOT / "database" / "acme" / "acme_N.sqlite"

_FAN_OUT_RANK = {"none": 0, "bounded": 1, "multiplicative": 2}

ROUTE_ID_MAX_HOPS = 8

_ROUTE_ARROW = ">"
_ROUTE_HASH = "#"

_COMBINE_SYMBOL = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}

_EXPECTED_STORAGE = {
    "string": ("text",),
    "numeric": ("real", "integer"),
    "integer": ("integer",),
    "date": ("text",),
    "datetime": ("text",),
    "boolean": ("integer",),
}

@dataclass(frozen=True)
class SkillResult:
    """What every tool returns: structured data AND a short rendering.

    An LLM consumes both — the text goes in the transcript, the data is what a
    program downstream can rely on. They are produced from the same call, so
    they cannot disagree.

    `ok` answers "did the tool answer the question asked", NOT "was the news
    good": `check_sql` on a query with violations is `ok=True` with
    `data["governed"] is False`. `ok=False` means the tool could not answer at
    all — an unknown concept, or a seam module that is not installed — and
    `data["code"]` then carries a machine-readable reason.
    """

    name: str
    ok: bool
    data: Mapping[str, Any]
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {"skill": self.name, "ok": self.ok, "data": dict(self.data), "text": self.text}

    def to_json(self, indent: int | None = None) -> str:
        """Byte-identical for identical arguments. Key order is INSERTION order
        (the order this module builds it in), never `sort_keys` — the field
        order is part of how a model reads the answer."""
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)

@dataclass(frozen=True)
class SkillSpec:
    """One tool as an MCP client sees it. `spc/mcp_server.py` publishes these
    verbatim and adds nothing."""

    name: str
    description: str
    input_schema: Mapping[str, Any]

_WORD_SPLIT = re.compile(r"[^0-9A-Za-z]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

def _stem(word: str) -> str:
    """Crude, deterministic, purely morphological. Applied to BOTH sides of
    every comparison, so it only has to be consistent."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w

def _tokens(text: str) -> tuple[str, ...]:
    """`PolicyCoverageDetail`, `policy_coverage_detail` and `HAS_LOSS_PAYMENT`
    all split the same way."""
    parts: list[str] = []
    for chunk in _WORD_SPLIT.split(str(text or "")):
        if not chunk:
            continue
        parts.extend(p for p in _CAMEL.split(chunk) if p)
    return tuple(_stem(p) for p in parts)

def _phrase(text: str) -> str:
    return " ".join(_tokens(text))

def _score(query_tokens: Sequence[str], field_text: str) -> int:
    """0..100, integer so the output has no float formatting to vary.

    100 equal after normalisation, 70 one contains the other as whole words,
    otherwise Jaccard token overlap scaled to 0..60.
    """
    if not field_text:
        return 0
    field_tokens = _tokens(field_text)
    if not field_tokens or not query_tokens:
        return 0
    q, f = " ".join(query_tokens), " ".join(field_tokens)
    if q == f:
        return 100
    if f" {q} " in f" {f} " or f" {f} " in f" {q} ":
        return 70
    qs, fs = set(query_tokens), set(field_tokens)
    overlap = len(qs & fs)
    if not overlap:
        return 0
    return (60 * overlap) // len(qs | fs)

_LABEL_MATCH = 70

def _jsonable(value: Any) -> Any:
    """Coerce a seam module's payload into ordered, serialisable data.

    Sets are sorted (this is the one place a foreign module's set iteration
    could otherwise leak into our output and break determinism), dataclasses
    become dicts, everything unknown becomes its `str`.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {f: _jsonable(getattr(value, f)) for f in value.__dataclass_fields__}
    return str(value)

VALUE_LADDER = ("exact", "nocase", "prefix", "contains")

_BLANK = ""

_LADDER_SQL = {
    "exact": "{col} = ?",
    "nocase": "LOWER({col}) = LOWER(?)",
    "prefix": "LOWER({col}) LIKE LOWER(?) || '%' ESCAPE '\\'",
    "contains": "LOWER({col}) LIKE '%' || LOWER(?) || '%' ESCAPE '\\'",
}

def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

def _quote(identifier: str) -> str:
    """SQLite identifier quoting. Column and table names come from the ontology,
    never from a caller, but quoting them costs nothing and keeps a name that
    collides with a keyword from becoming a parse error."""
    return '"' + str(identifier).replace('"', '""') + '"'

class Snapshot:
    """One read-only database file, with every answer memoised by content hash.

    Deliberately NOT a probe framework: four fixed measurements
    (`values`, `rows`, and what the two build on) and no way for a caller to
    ask an arbitrary question. Every query this class runs is constructed from
    the ontology by `Skills`, never from model input — the only caller-supplied
    text is a bound parameter.

    `digest` is the content hash of the file, and it is part of every cache key,
    so an answer can never be served for data it did not come from.
    """

    def __init__(self, path: str | FsPath) -> None:
        self.path = str(FsPath(path).resolve())
        self.digest = _digest(self.path)
        self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self._conn.text_factory = str
        self._memo: dict[tuple, Any] = {}
        self.tables: frozenset[str] = frozenset(
            name.lower()
            for (name,) in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        )

    def close(self) -> None:
        self._conn.close()

    def rows(self, sql: str, params: tuple = ()) -> tuple[tuple, ...]:
        """Run a fixed query, memoised on (snapshot, sql, params)."""
        key = (self.digest, sql, params)
        hit = self._memo.get(key)
        if hit is None:
            hit = tuple(self._conn.execute(sql, params).fetchall())
            self._memo[key] = hit
        return hit

    def has(self, table: str) -> bool:
        return str(table).lower() in self.tables

    def values(
        self, table: str, column: str, phrase: str,
        *, where: Sequence[str] = (), limit: int = 25,
    ) -> tuple[str, tuple[str, ...]]:
        """Walk the ladder; return `(rung, canonical values)`.

        Results are `ORDER BY`-ed inside SQL, so the tuple does not inherit
        storage order. `('absent', ())` when no rung matched.
        """
        clauses = list(where)
        col = _quote(column)
        for rung in VALUE_LADDER:
            predicate = _LADDER_SQL[rung].format(col=f"t.{col}")
            argument = phrase if rung in ("exact", "nocase") else _like_escape(phrase)
            sql = (
                f"SELECT DISTINCT t.{col} FROM {_quote(table)} t "
                f"WHERE {predicate} AND t.{col} IS NOT NULL AND t.{col} <> ''"
                + "".join(f" AND {c}" for c in clauses)
                + f" ORDER BY t.{col} LIMIT {int(limit)}"
            )
            found = self.rows(sql, (argument,))
            if found:
                return rung, tuple(str(r[0]) for r in found)
        return "absent", ()

def _digest(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()[:16]

def _rel(path: Any) -> str:
    """A repo-relative source path, so provenance strings do not carry an
    absolute path that differs per machine."""
    try:
        return str(FsPath(str(path)).resolve().relative_to(_REPO_ROOT))
    except (ValueError, OSError):
        return str(path)

class RouteIdError(ValueError):
    """A route id that is malformed, or that names no governed route."""

def route_id(from_concept: str, to_concept: str, index: int) -> str:
    """The stable name of the `index`-th governed route between two concepts.

    `index` is 1-based, matching what `find_paths` prints, so an id copied out
    of a transcript is the id the tool emitted.
    """
    return f"{from_concept}{_ROUTE_ARROW}{to_concept}{_ROUTE_HASH}{int(index)}"

def parse_route_id(value: str) -> tuple[str, str, int]:
    """`'Policy>ClaimAmount#1'` -> `('Policy', 'ClaimAmount', 1)`.

    Purely syntactic: it does not know whether the concepts exist. Resolution
    against the ontology is `Skills.route`, so a caller can tell a TYPO
    (`RouteIdError` here) from a route that is simply not governed (an empty
    `find_paths`).
    """
    text = str(value or "")
    head, _, tail = text.partition(_ROUTE_HASH)
    source, _, target = head.partition(_ROUTE_ARROW)
    if not source or not target or not tail:
        raise RouteIdError(
            f"{value!r} is not a route id. The form is "
            f"'<from>{_ROUTE_ARROW}<to>{_ROUTE_HASH}<n>', exactly as find_paths "
            f"printed it."
        )
    try:
        index = int(tail)
    except ValueError:
        raise RouteIdError(f"{value!r} has a non-numeric index {tail!r}") from None
    if index < 1:
        raise RouteIdError(f"{value!r} has index {index}; route ids are 1-based")
    return source, target, index

def _check_route_id_alphabet(ontology: Ontology) -> None:
    """Refuse an ontology whose concept names would make ids ambiguous.

    Costs one pass over 13 names and removes the possibility that a renamed
    concept silently turns `A>B#1` into two readings.
    """
    for name in ontology.concept_names():
        if _ROUTE_ARROW in name or _ROUTE_HASH in name:
            raise RouteIdError(
                f"concept {name!r} contains {_ROUTE_ARROW!r} or {_ROUTE_HASH!r}, "
                f"which are route-id separators; route ids would be ambiguous"
            )

class Skills:
    """The closed tool set over one ontology and one database snapshot.

    Construction loads the ontology and builds the path graph once. Every
    method is pure with respect to that pair: same arguments, same bytes.
    """

    def __init__(
        self,
        ontology: Ontology | None = None,
        graph: PathGraph | None = None,
        *,
        db_path: str | FsPath | None = DEFAULT_DB,
    ) -> None:
        if graph is not None:
            self.graph = graph
            self.ontology = graph.ontology if ontology is None else ontology
        else:
            self.ontology = ontology if ontology is not None else load_ontology()
            self.graph = PathGraph(self.ontology)
        _check_route_id_alphabet(self.ontology)
        self.db_path = None if db_path is None else str(db_path)
        self._snapshot: Snapshot | None = None
        self._snapshot_tried = False
        self._paths_memo: dict[tuple[str, str, int, str], SkillResult] = {}
        self._route_memo: dict[str, MenuRoute] = {}
        sources = self.ontology.sources
        self._graph_source = _rel(sources[0]) if sources else "ontology"
        self._semantic_source = _rel(sources[-1]) if sources else "ontology"

    def route(self, identifier: str) -> MenuRoute:
        """Resolve a route id to the governed route it names.

        The hop bound the id was minted under is NOT recorded in the id and does
        not need to be: `path_sort_key` orders by hop count first, so the routes
        within a smaller bound are a prefix of those within a larger one and an
        index means the same thing under both. This walks the bound upward until
        the index is reachable, which is deterministic and terminates at
        `ROUTE_ID_MAX_HOPS`.
        """
        cached = self._route_memo.get(identifier)
        if cached is not None:
            return cached
        source, target, index = parse_route_id(identifier)
        for name in (source, target):
            if name not in self.ontology.concepts:
                raise RouteIdError(
                    f"route id {identifier!r} names {name!r}, which is not a concept "
                    f"in this ontology"
                )
        found: tuple[Path, ...] = ()
        for bound in range(DEFAULT_MAX_HOPS, ROUTE_ID_MAX_HOPS + 1):
            found = self.graph.paths(source, target, max_hops=bound)
            if index <= len(found):
                break
        if index > len(found):
            raise RouteIdError(
                f"route id {identifier!r} asks for route {index} of "
                f"{source} -> {target}, but only {len(found)} are governed within "
                f"{ROUTE_ID_MAX_HOPS} hops"
            )
        path = found[index - 1]
        route = MenuRoute(
            id=identifier, subject=source, target=target, path=path,
            roles=self.graph.role_signature(source, path),
            fans_out=self._path_fans_out(path),
        )
        self._route_memo[identifier] = route
        return route

    def _path_fans_out(self, path: Path) -> bool:
        """Whether ANY step on the path duplicates rows. Recomputed here rather
        than imported from `spc.menu` so the retrieved route carries exactly the
        flag a built route carries, and the test that they agree is a real
        test rather than a tautology."""
        return any(
            self.ontology.edge(step.edge).fan_out_in(forward=step.forward) != "none"
            for step in path.steps
        )

    def menu(
        self,
        route_ids: Sequence[str],
        *,
        subject: str | None = None,
        metrics: Sequence[str] | None = None,
        concepts: Sequence[str] = (),
    ) -> Menu:
        """A `spc.menu.Menu` over exactly the retrieved routes.

        The concept list is derived, never supplied: it is the subject plus
        every concept a retrieved route lands on. That is what makes the pick's
        attribute vocabulary correct — `parse_pick` validates an attribute
        against the concept its route reaches, and a concept nothing reaches
        cannot be referenced anyway.

        Route ids keep the names `find_paths` gave them, so the ids in the
        model's pick are the ids in its transcript. This is the ONE place the
        two menu constructions differ: `build_menu` renumbers, this does not.

        `concepts=` WIDENS that derived vocabulary; it never widens the ROUTES,
        which stay exactly the ids passed in. It exists for one caller: the
        structured-output contract on the retrieved arm's terminal turn, which
        must be constructable before the subject is known — a question answered
        entirely on the subject's own columns retrieves no route at all, and a
        menu with no routes and no subject has nothing to name itself with.
        Naming the ontology's concepts there constrains the ROUTE enum (the
        governed thing) without forcing a subject the model has not chosen, and
        without narrowing the attribute enum to a vocabulary the model would
        then have to violate. The concept-to-attribute binding the schema cannot
        express is enforced by `parse_pick`, as it always was.
        """
        routes = tuple(self.route(identifier) for identifier in dict.fromkeys(route_ids))
        widen = tuple(dict.fromkeys(concepts))
        head = subject or (routes[0].subject if routes else None)
        if head is None and not widen:
            raise RouteIdError(
                "a menu needs a subject: pass one, retrieve at least one route, "
                "or name the concepts it may range over"
            )
        if head is not None and head not in self.ontology.concepts:
            raise RouteIdError(f"subject {head!r} is not a concept in this ontology")
        for name in widen:
            if name not in self.ontology.concepts:
                raise RouteIdError(f"{name!r} is not a concept in this ontology")
        vocabulary = tuple(sorted(
            ({head} if head else set()) | set(widen)
            | {r.target for r in routes} | {r.subject for r in routes}
        ))
        chosen = (tuple(sorted(metrics)) if metrics is not None
                  else tuple(sorted(self.ontology.metrics)))
        for name in chosen:
            self.ontology.metric(name)
        return Menu(
            concepts=vocabulary,
            routes=routes,
            metrics=chosen,
            ontology=self.ontology,
            graph=self.graph,
            _by_id={r.id: r for r in routes},
        )

    @staticmethod
    def route_ids_in(pick: Mapping[str, Any] | str) -> tuple[str, ...]:
        """Every route id the pick mentions, in first-use order, `SELF` dropped.

        Walked structurally rather than by regex so a nested composite measure
        cannot hide a route the assembled menu then lacks.
        """
        data = json.loads(pick) if isinstance(pick, str) else dict(pick)
        found: list[str] = []

        def note(value: Any) -> None:
            text = str(value)
            if value is None or text == SUBJECT_ID or text in found:
                return
            found.append(text)

        def walk(node: Any) -> None:
            if isinstance(node, Mapping):
                if "route" in node:
                    note(node.get("route"))
                for key in ("parts", "measures", "dimensions", "filters"):
                    walk(node.get(key))
            elif isinstance(node, (list, tuple)):
                for item in node:
                    walk(item)

        walk(data)
        return tuple(found)

    def parse_pick(self, pick: Mapping[str, Any] | str) -> Plan:
        """The model's structured choice -> a `Plan`, through `spc.menu`.

        The menu is assembled from the ids the pick itself names, so the arm
        never has to remember which routes it showed the model. A pick naming a
        route that was never retrieved still resolves — the id is a name for a
        governed route, not a claim about the transcript — but a pick naming a
        route that is not GOVERNED raises, which is the property that matters.
        """
        data = json.loads(pick) if isinstance(pick, str) else dict(pick)
        assembled = self.menu(self.route_ids_in(data), subject=data.get("subject"))
        return _menu_parse_pick(data, assembled)

    def pick_schema(
        self,
        route_ids: Sequence[str],
        *,
        subject: str | None = None,
        metrics: Sequence[str] | None = None,
        concepts: Sequence[str] = (),
    ) -> dict[str, Any]:
        """The structured-output contract over RETRIEVED ids.

        Same schema `spc.menu.Menu.pick_schema` emits, over a menu of 5-15
        routes instead of 904 — which is the whole point of the arm. A
        constrained decoder given this cannot name a route it did not retrieve.

        `concepts=` widens only the subject/attribute vocabulary — see `menu`.
        With no `route_ids` at all the route enum is `["SELF"]`, which is the
        right contract for a question answered on the subject's own columns.
        """
        return self.menu(
            route_ids, subject=subject, metrics=metrics, concepts=concepts
        ).pick_schema(subject=subject)

    def response_format(
        self,
        route_ids: Sequence[str],
        *,
        subject: str | None = None,
        metrics: Sequence[str] | None = None,
        concepts: Sequence[str] = (),
        name: str = "governed_pick",
    ) -> dict[str, Any]:
        """`pick_schema` wrapped as a chat-completions `response_format`.

        Here rather than in the harness for the reason `spc.menu.Menu` keeps its
        own: the schema a caller SENDS and the schema `parse_pick` enforces are
        then one object, built once, and cannot drift.
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": True,
                "schema": self.pick_schema(
                    route_ids, subject=subject, metrics=metrics, concepts=concepts
                ),
            },
        }

    def compile_pick(self, pick: Mapping[str, Any] | str) -> str:
        """Pick -> SQL, end to end. THE COMPILER SEAM.

        Imported lazily for the same reason `check_sql` is: `spc.compile` pulls
        sqlglot, and a caller that only retrieves structure should not pay for
        it.
        """
        from spc.compile import compile as _compile  # noqa: PLC0415

        return _compile(self.parse_pick(pick), self.ontology, self.graph)

    def snapshot(self) -> tuple[Snapshot | None, str | None]:
        """The snapshot, or `(None, reason)`. Opened once, lazily: a caller that
        only asks structural questions never touches a file."""
        if self._snapshot is not None:
            return self._snapshot, None
        if self._snapshot_tried:
            return None, self._snapshot_reason
        self._snapshot_tried = True
        if not self.db_path or not FsPath(self.db_path).exists():
            self._snapshot_reason = f"no database snapshot at {self.db_path!r}"
            return None, self._snapshot_reason
        try:
            self._snapshot = Snapshot(self.db_path)
        except sqlite3.Error as exc:  # pragma: no cover
            self._snapshot_reason = f"{self.db_path!r} could not be opened ({exc})"
            return None, self._snapshot_reason
        return self._snapshot, None

    _snapshot_reason: str | None = None

    @staticmethod
    def _checker() -> tuple[Any, str | None]:
        """`spc.check.check`, or `(None, reason)`. THE CHECKER SEAM."""
        try:
            from spc.check import check as _check  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            return None, f"spc.check is not available ({exc})"
        return _check, None

    def _fail(self, name: str, code: str, message: str, **extra: Any) -> SkillResult:
        data: dict[str, Any] = {"code": code, "error": message}
        data.update(extra)
        return SkillResult(name=name, ok=False, data=data, text=f"{code}: {message}")

    def _did_you_mean(self, phrase: str, limit: int = 5) -> list[str]:
        query = _tokens(phrase)
        scored = [
            (-_score(query, name), name)
            for name in self.ontology.concept_names()
            if _score(query, name) > 0
        ]
        return [name for _, name in sorted(scored)[:limit]]

    def _attribute_row(self, concept: Concept, attr: Attribute) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": attr.name,
            "type": attr.type,
            "value_type": attr.value_type,
        }
        if attr.is_resolved:
            edge = self.ontology.edge(attr.via or "")
            far = edge.target if edge.source == concept.name else edge.source
            row["column"] = None
            row["resolved_via"] = {
                "edge": attr.via,
                "concept": far,
                "attribute": attr.via_attribute,
                "column": self.ontology.concept(far)
                .attributes[attr.via_attribute or ""]
                .column,
            }
        else:
            row["column"] = attr.column
            row["qualified_column"] = f"{concept.table}.{attr.column}"
            row["resolved_via"] = None
        row["searchable"] = attr.searchable
        row["is_key"] = bool(attr.column and attr.column in concept.key)
        row["is_title"] = concept.display == attr.name
        row["description"] = attr.description
        row["label"] = attr.label
        row["aliases"] = list(attr.aliases)
        return row

    def _concept_predicates(self, concept: Concept) -> list[dict[str, Any]]:
        """The `backed_where` of a role object, as SQL-ready predicates.

        A role object's predicate is part of the concept's IDENTITY, so any
        query touching the concept must carry it. Reported per concept wherever
        one is reachable, never left for the caller to remember.
        """
        return [
            {
                "table": concept.table,
                "column": column,
                "value": value,
                "sql": f"{concept.table}.{column} = '{value}'",
                "reason": f"concept {concept.name} is a role object (backed_where)",
                "authorised_by": {"kind": "concept", "name": concept.name,
                                  "declared_in": self._graph_source},
            }
            for column, value in concept.backed_where
        ]

    def _where(self, concept: Concept, alias: str) -> list[str]:
        """A role object's identity predicate, as SQL on one alias."""
        return [
            f"{alias}.{_quote(column)} = '{value}'"
            for column, value in concept.backed_where
        ]

    def _source(self, concept: Concept, restrict: tuple[str, tuple[tuple[str, str], ...]] | None,
                alias: str) -> str:
        """A concept as a FROM item. A subset-restricted endpoint becomes a
        derived table so the restriction cannot be dropped by a LEFT JOIN."""
        table = _quote(concept.table)
        if restrict is None:
            return f"{table} {alias}"
        rtable, columns = restrict
        conditions = " AND ".join(
            f"x.{_quote(a)} = r.{_quote(b)}" for a, b in columns
        )
        return (f"(SELECT DISTINCT x.* FROM {table} x "
                f"JOIN {_quote(rtable)} r ON {conditions}) {alias}")

    def _measure_fan_out(self, edge: Edge, forward: bool) -> dict[str, Any]:
        """How many rows one origin row ACTUALLY multiplies into, right now.

        Reported beside the declaration rather than instead of it: the pair is
        the audit. `contradicts_declared` is true when the ontology says a step
        adds no rows and the data says it does — which is exactly the shape of
        a grain bug that produces a plausible wrong number.
        """
        snapshot, reason = self.snapshot()
        declared = edge.fan_out_in(forward=forward)
        if snapshot is None:
            return {"available": False, "reason": reason, "declared": declared}

        onto = self.ontology
        origin = onto.concept(edge.origin(forward=forward))
        landing = onto.concept(edge.endpoint(forward=forward))
        src_alias, dst_alias = ("o", "l") if forward else ("l", "o")
        restrict = (
            (str(edge.restrict_table), edge.restrict_columns) if edge.is_restricted else None
        )

        origin_restrict = restrict if (restrict and dst_alias == "o") else None
        landing_restrict = restrict if (restrict and dst_alias == "l") else None

        if not origin.key or not landing.key:
            return {"available": False, "declared": declared,
                    "reason": "fan-out is measured per origin key, and one endpoint "
                              "declares no key columns"}
        for table in filter(None, [origin.table, landing.table, edge.via_table,
                                   edge.restrict_table]):
            if not snapshot.has(table):
                return {"available": False, "declared": declared,
                        "reason": f"table {table!r} is not in this snapshot"}

        joins: list[str] = []
        if edge.is_junction:
            from_conditions = [f"{src_alias}.{_quote(a)} = v.{_quote(b)}"
                               for a, b in edge.via_from_join]
            to_conditions = [f"v.{_quote(a)} = {dst_alias}.{_quote(b)}"
                             for a, b in edge.via_to_join]
            role = [f"v.{_quote(c)} = '{v}'" for c, v in edge.role_predicate]
            near, far = (from_conditions, to_conditions) if forward else \
                        (to_conditions, from_conditions)
            joins.append(f"LEFT JOIN {_quote(str(edge.via_table))} v ON "
                         + " AND ".join(near + role))
            joins.append(f"LEFT JOIN {self._source(landing, landing_restrict, 'l')} ON "
                         + " AND ".join(far + self._where(landing, "l")))
        else:
            conditions = [f"{src_alias}.{_quote(a)} = {dst_alias}.{_quote(b)}"
                          for a, b in edge.join]
            joins.append(f"LEFT JOIN {self._source(landing, landing_restrict, 'l')} ON "
                         + " AND ".join(conditions + self._where(landing, "l")))

        group = ", ".join(f"o.{_quote(k)}" for k in origin.key)
        counted = f"l.{_quote(landing.key[0])}"
        where = self._where(origin, "o")
        sql = (
            "SELECT COUNT(*), MAX(n), SUM(n), SUM(CASE WHEN n = 0 THEN 1 ELSE 0 END) FROM ("
            f"SELECT COUNT({counted}) AS n FROM "
            f"{self._source(origin, origin_restrict, 'o')} "
            + " ".join(joins)
            + ("" if not where else " WHERE " + " AND ".join(where))
            + f" GROUP BY {group})"
        )
        try:
            (origins, largest, total, orphans), = snapshot.rows(sql)
        except sqlite3.Error as exc:  # pragma: no cover
            return {"available": False, "declared": declared,
                    "reason": f"measurement failed ({exc})", "sql": sql}

        origins = int(origins or 0)
        largest = int(largest or 0)
        orphans = int(orphans or 0)
        return {
            "available": True,
            "declared": declared,
            "origin_rows": origins,
            "max_rows_per_origin": largest,
            "mean_rows_per_origin": round((total or 0) / origins, 3) if origins else 0.0,
            "origins_with_no_match": orphans,
            "join_must_be_left_to_keep_them": orphans > 0,
            "multiplies_rows": largest > 1,

            "contradicts_declared": declared == "none" and largest > 1,
        }

    def _column_facts(self, concept: Concept, attr: Attribute) -> dict[str, Any]:
        """Storage type, blanks and distinctness for one physical column."""
        snapshot, reason = self.snapshot()
        if snapshot is None:
            return {"available": False, "reason": reason}
        if attr.column is None:
            return {"available": False,
                    "reason": f"{attr.qualified} is resolved through edge {attr.via!r} "
                              f"and has no column on {concept.table}"}
        if not snapshot.has(concept.table):
            return {"available": False, "reason": f"table {concept.table!r} is not in "
                                                  f"this snapshot"}
        column, table = _quote(attr.column), _quote(concept.table)
        where = self._where(concept, "t")
        tail = ("" if not where else " WHERE " + " AND ".join(where))
        stats_sql = (
            f"SELECT COUNT(*), SUM(CASE WHEN t.{column} IS NULL THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN t.{column} = '' THEN 1 ELSE 0 END), "
            f"COUNT(DISTINCT t.{column}) FROM {table} t{tail}"
        )
        types_sql = (
            f"SELECT typeof(t.{column}), COUNT(*) FROM {table} t{tail} "
            f"GROUP BY 1 ORDER BY 2 DESC, 1 ASC"
        )
        try:
            (rows, nulls, blanks, distinct), = snapshot.rows(stats_sql)
            types = snapshot.rows(types_sql)
        except sqlite3.Error as exc:  # pragma: no cover
            return {"available": False, "reason": f"measurement failed ({exc})"}
        observed = [{"storage_type": str(t), "rows": int(n)} for t, n in types]
        dominant = observed[0]["storage_type"] if observed else None
        expected = _EXPECTED_STORAGE.get(str(attr.type), ())
        facts: dict[str, Any] = {
            "available": True,
            "rows": int(rows or 0),
            "nulls": int(nulls or 0),
            "blanks": int(blanks or 0),
            "distinct_values": int(distinct or 0),
            "declared_type": attr.type,
            "storage_types": observed,
            "storage_type": dominant,
            "storage_matches_declared": (
                None if not expected or dominant in (None, "null")
                else dominant in expected
            ),
        }

        if dominant == "integer" and attr.type in ("numeric", "integer"):
            facts["integer_division_hazard"] = (
                f"{concept.table}.{attr.column} is stored as INTEGER: a ratio computed "
                f"from it truncates (13600/20000 = 0). Cast to REAL before dividing."
            )
        if facts["rows"] and facts["nulls"] + facts["blanks"] >= facts["rows"]:
            facts["empty_column"] = (
                f"{concept.table}.{attr.column} is NULL or blank in every row of this "
                f"snapshot. Projecting it returns nothing readable and filtering on it "
                f"matches no rows — whatever the ontology says the attribute means."
            )
        if dominant == "text" and attr.type in ("numeric", "integer"):
            facts["text_arithmetic_warning"] = (
                f"{concept.table}.{attr.column} is declared {attr.type} but STORED AS TEXT. "
                f"SQLite coerces it inside SUM/AVG, but comparisons and ORDER BY are "
                f"lexicographic ('9' > '10'). CAST it before comparing or sorting."
            )
        if attr.type in ("date", "datetime") and dominant not in (None, "null", "text"):
            facts["date_storage_warning"] = (
                f"declared {attr.type} but stored as {dominant}; date comparisons "
                f"against ISO strings will not behave as written."
            )
        return facts

    def _row_count(self, concept: Concept) -> dict[str, Any]:
        snapshot, reason = self.snapshot()
        if snapshot is None:
            return {"available": False, "reason": reason}
        if not snapshot.has(concept.table):
            return {"available": False,
                    "reason": f"table {concept.table!r} is not in this snapshot"}
        where = self._where(concept, "t")
        sql = (f"SELECT COUNT(*) FROM {_quote(concept.table)} t"
               + ("" if not where else " WHERE " + " AND ".join(where)))
        (count,), = snapshot.rows(sql)
        return {
            "available": True,
            "rows": int(count or 0),
            "populated": bool(count),
            "note": None if count else
                    "This concept has no rows in this snapshot: every query over it "
                    "returns nothing, however well formed.",
        }

    _DETAIL = ("routes", "joins")

    def find_paths(
        self, from_concept: str, to_concept: str, max_hops: int = DEFAULT_MAX_HOPS,
        detail: str = "joins",
    ) -> SkillResult:
        name = "find_paths"
        if detail not in self._DETAIL:
            return self._fail(name, "BAD_ARGUMENT",
                              f"detail {detail!r} is not one of {list(self._DETAIL)}")
        for label, value in (("from_concept", from_concept), ("to_concept", to_concept)):
            if value not in self.ontology.concepts:
                return self._fail(
                    name, "UNKNOWN_CONCEPT",
                    f"{label} {value!r} is not a concept in this ontology",
                    did_you_mean=self._did_you_mean(str(value)),
                    known_concepts=list(self.ontology.concept_names()),
                )
        try:
            hops = int(max_hops)
        except (TypeError, ValueError):
            return self._fail(name, "BAD_ARGUMENT", f"max_hops {max_hops!r} is not an integer")
        if hops < 0:
            return self._fail(name, "BAD_ARGUMENT", "max_hops must be >= 0")

        memo_key = (from_concept, to_concept, hops, detail)
        cached = self._paths_memo.get(memo_key)
        if cached is not None:
            return cached

        source = self.ontology.concept(from_concept)
        found = self.graph.paths(from_concept, to_concept, max_hops=hops)
        row = self._route_row if detail == "routes" else self._path_row
        paths = [row(from_concept, p, index) for index, p in enumerate(found, 1)]

        data: dict[str, Any] = {
            "from_concept": from_concept,
            "to_concept": to_concept,
            "max_hops": hops,
            "from_table": source.table,
            "to_table": self.ontology.concept(to_concept).table,
            "subject_predicates": self._concept_predicates(source),
            "path_count": len(paths),
            "paths": paths,
        }
        if not paths:
            data["note"] = (
                f"No governed route from {from_concept} to {to_concept} within {hops} hops. "
                "Either raise max_hops, or these concepts are not related in the ontology — "
                "in which case NO join between them is authorised and none may be written. "
                "`describe_concept` lists each concept's direct neighbours."
            )
        render = self._render_routes if detail == "routes" else self._render_paths
        result = SkillResult(name, True, data, render(data))
        self._paths_memo[memo_key] = result
        return result

    def _route_row(self, subject: str, path: Path, index: int) -> dict[str, Any]:
        """One governed route as the pick arm needs it.

        Deliberately WITHOUT join keys, SQL sketches or measured fan-out: this
        arm's model does not write the join, `spc/compile.py` does, and every
        token spent describing a join it will not write is the cost this arm
        exists to remove. What survives is what the CHOICE turns on — how long
        the route is, which roles it commits to, whether it duplicates rows, and
        which declared edges authorise it.
        """
        worst = "none"
        steps: list[dict[str, Any]] = []
        node = subject
        for position, step in enumerate(path.steps):
            edge = self.ontology.edge(step.edge)
            landing = edge.endpoint(forward=step.forward)
            fan_out = edge.fan_out_in(forward=step.forward)
            if _FAN_OUT_RANK[fan_out] > _FAN_OUT_RANK[worst]:
                worst = fan_out
            steps.append({
                "position": position,
                "edge": edge.name,
                "direction": "forward" if step.forward else "reverse",
                "from_concept": node,
                "to_concept": landing,
                "role": step.role,
                "fan_out": fan_out,

                "authorised_by": {
                    "kind": "edge", "name": edge.name,
                    "declared_as": f"{edge.source} -> {edge.target}",
                    "declared_in": self._graph_source,
                },
            })
            node = landing
        return {
            "route_id": route_id(subject, path.target, index),
            "index": index,
            "path_id": path_signature(path),
            "hops": len(path.steps),
            "from_concept": subject,
            "to_concept": path.target,
            "edges": [s.edge for s in path.steps],
            "role_signature": list(self.graph.role_signature(subject, path)),
            "max_fan_out": worst,
            "duplicates_rows": worst != "none",
            "steps": steps,
        }

    def _render_routes(self, data: Mapping[str, Any]) -> str:
        """One line per route. This is the arm's per-call token cost, and it is
        the number the whole design is judged on, so nothing decorative is in
        it."""
        lines = [
            f"find_paths {data['from_concept']} -> {data['to_concept']} "
            f"(max_hops={data['max_hops']}): {data['path_count']} governed route(s). "
            f"Use `route_id` as `route` in your pick; `{SUBJECT_ID}` = the subject "
            f"itself, no traversal."
        ]
        for predicate in data["subject_predicates"]:
            lines.append(f"  subject predicate (always applied): {predicate['sql']}")
        if not data["path_count"]:
            lines.append(f"  {data['note']}")
            return "\n".join(lines)
        for path in data["paths"]:
            roles = "+".join(path["role_signature"]) or "-"
            arrow = " ".join(
                f"{s['edge']}{'>' if s['direction'] == 'forward' else '<'}"
                for s in path["steps"]
            )
            lines.append(
                f"  {path['route_id']}  {path['hops']}h  roles:{roles}  "
                f"{'fans-out' if path['duplicates_rows'] else 'one-to-one'}  {arrow}"
            )
        lines.append(
            f"  Every edge above is a declared edge in {self._graph_source}; a pair of "
            f"concepts with no route listed has NO authorised join."
        )
        return "\n".join(lines)

    def _path_row(self, subject: str, path: Path, index: int) -> dict[str, Any]:
        steps: list[dict[str, Any]] = []
        clauses: list[dict[str, Any]] = []
        node = subject
        worst = "none"
        for position, step in enumerate(path.steps):
            edge = self.ontology.edge(step.edge)
            landing = edge.endpoint(forward=step.forward)
            fan_out = edge.fan_out_in(forward=step.forward)
            if _FAN_OUT_RANK[fan_out] > _FAN_OUT_RANK[worst]:
                worst = fan_out
            join = self._join_spec(edge, forward=step.forward, arriving_at=landing)
            clauses.extend(join["clauses"])
            steps.append({
                "position": position,
                "edge": edge.name,
                "direction": "forward" if step.forward else "reverse",
                "from_concept": node,
                "to_concept": landing,
                "role": step.role,
                "cardinality": edge.cardinality,
                "fan_out": fan_out,
                "duplicates_rows": fan_out != "none",
                "description": edge.description,

                "authorised_by": {
                    "kind": "edge",
                    "name": edge.name,
                    "declared_as": f"{edge.source} -> {edge.target}",
                    "declared_in": self._graph_source,
                    "evidence": edge.evidence,
                },
                "join": join,
                "landing_predicates": self._concept_predicates(self.ontology.concept(landing)),

                "measured_fan_out": self._measure_fan_out(edge, step.forward),
            })
            node = landing

        measured = [s["measured_fan_out"] for s in steps]
        contradictions = [
            f"{s['edge']} ({s['direction']}) is declared fan_out={s['fan_out']} but one "
            f"origin row reaches {s['measured_fan_out']['max_rows_per_origin']} rows"
            for s in steps
            if s["measured_fan_out"].get("contradicts_declared")
        ]
        orphans = [
            f"{s['measured_fan_out']['origins_with_no_match']} {s['from_concept']} row(s) "
            f"have no {s['to_concept']} across {s['edge']} — an inner join drops them"
            for s in steps
            if s["measured_fan_out"].get("origins_with_no_match")
        ]
        return {
            "route_id": route_id(subject, path.target, index),
            "index": index,
            "path_id": path_signature(path),
            "hops": len(path.steps),
            "from_concept": subject,
            "to_concept": path.target,
            "edges": [s.edge for s in path.steps],

            "role_signature": list(self.graph.role_signature(subject, path)),
            "max_fan_out": worst,
            "duplicates_rows": worst != "none",
            "distinct_required": worst != "none",
            "measured_max_rows_per_origin": max(
                [m.get("max_rows_per_origin", 0) for m in measured if m.get("available")] or [None]
            ),
            "declaration_contradicted_by_data": contradictions,
            "rows_lost_to_inner_join": orphans,
            "steps": steps,
            "join_clauses": clauses,
            "sql_sketch": self._sql_sketch(subject, clauses),
        }

    def _join_spec(self, edge: Edge, *, forward: bool, arriving_at: str) -> dict[str, Any]:
        """The PHYSICAL join keys needed to write the SQL for one traversal.

        Equalities are reported in the edge's DECLARED orientation (`a.x = b.y`
        is symmetric, so direction cannot change them), while `clauses` are
        emitted in TRAVERSAL order — the order a writer adds JOINs walking this
        path. Each clause names the edge that authorised it.
        """
        onto = self.ontology
        src_table = onto.concept(edge.source).table
        dst_table = onto.concept(edge.target).table
        landing_table = onto.concept(arriving_at).table
        origin_table = onto.concept(edge.origin(forward=forward)).table

        def pair(left_table: str, left_col: str, right_table: str, right_col: str) -> dict[str, Any]:
            return {
                "left": f"{left_table}.{left_col}",
                "right": f"{right_table}.{right_col}",
                "sql": f"{left_table}.{left_col} = {right_table}.{right_col}",
                "authorised_by": edge.name,
            }

        on: list[dict[str, Any]] = []
        predicates: list[dict[str, Any]] = []
        clauses: list[dict[str, Any]] = []

        def clause(table: str, conditions: Sequence[dict[str, Any]], note: str) -> None:
            body = " AND ".join(c["sql"] for c in conditions)
            clauses.append({
                "sql": f"JOIN {table} ON {body}",
                "table": table,
                "authorised_by": edge.name,
                "declared_in": self._graph_source,
                "role": note,
            })

        for column, value in edge.role_predicate:
            table = edge.via_table or landing_table
            predicates.append({
                "table": table,
                "column": column,
                "value": value,
                "sql": f"{table}.{column} = '{value}'",
                "reason": f"role_predicate of edge {edge.name} — the role is part of "
                          f"the edge's identity and may not be omitted",
                "authorised_by": edge.name,
            })

        if edge.is_junction:
            via = str(edge.via_table)
            from_pairs = [pair(src_table, a, via, b) for a, b in edge.via_from_join]
            to_pairs = [pair(via, a, dst_table, b) for a, b in edge.via_to_join]
            on = from_pairs + to_pairs
            near, far = (from_pairs, to_pairs) if forward else (to_pairs, from_pairs)
            clause(via, list(near) + list(predicates), "junction table")
            clause(landing_table, far, "landing concept")
            kind = "junction"
            tables = [origin_table, via, landing_table]
        else:
            on = [pair(src_table, a, dst_table, b) for a, b in edge.join]
            clause(landing_table, on + list(predicates), "landing concept")
            kind = "direct"
            tables = [origin_table, landing_table]

        restrict: dict[str, Any] | None = None
        if edge.is_restricted:
            rtable = str(edge.restrict_table)
            rpairs = [pair(dst_table, a, rtable, b) for a, b in edge.restrict_columns]
            restrict = {
                "table": rtable,
                "on": rpairs,
                "reason": f"edge {edge.name} is a SUBSET of the rows its join reaches; "
                          f"the traversal is only governed WITH this semijoin",
            }
            clause(rtable, rpairs, "subset restriction (required)")
            tables.append(rtable)

        return {
            "kind": kind,
            "tables": tables,
            "on": on,
            "predicates": predicates,
            "restrict": restrict,
            "clauses": clauses,
        }

    def _sql_sketch(self, subject: str, clauses: Sequence[Mapping[str, Any]]) -> str:
        table = self.ontology.concept(subject).table
        lines = [f"FROM {table}"]
        lines.extend(str(c["sql"]) for c in clauses)
        for predicate in self._concept_predicates(self.ontology.concept(subject)):
            lines.append(f"WHERE {predicate['sql']}")
        return "\n".join(lines)

    def _render_paths(self, data: Mapping[str, Any]) -> str:
        head = (
            f"find_paths {data['from_concept']} -> {data['to_concept']} "
            f"(max_hops={data['max_hops']}): {data['path_count']} governed path(s)."
        )
        lines = [head]
        for predicate in data["subject_predicates"]:
            lines.append(f"  subject predicate (required): {predicate['sql']}")
        if not data["path_count"]:
            lines.append(f"  {data['note']}")
            return "\n".join(lines)
        for path in data["paths"]:
            roles = "+".join(path["role_signature"]) or "none"
            lines.append(
                f"  [{path['index']}] {path['hops']} hop(s)  roles={roles}  "
                f"fan-out={path['max_fan_out']}"
                f"{'  (rows duplicate — aggregate with care)' if path['duplicates_rows'] else ''}"
            )
            arrow = path["from_concept"]
            for step in path["steps"]:
                mark = "" if step["direction"] == "forward" else "~"
                arrow += f" -{mark}{step['edge']}-> {step['to_concept']}"
            lines.append(f"      {arrow}")
            for clause in path["join_clauses"]:
                lines.append(f"      {clause['sql']}   [authorised by {clause['authorised_by']}]")
            for step in path["steps"]:
                for predicate in step["join"]["predicates"] + step["landing_predicates"]:
                    lines.append(f"      AND {predicate['sql']}   [{predicate['reason']}]")
            for note in path["declaration_contradicted_by_data"]:
                lines.append(f"      !! measured: {note}")
            for note in path["rows_lost_to_inner_join"]:
                lines.append(f"      note: {note}")
        return "\n".join(lines)

    def describe_concept(self, name: str, detail: str = "storage") -> SkillResult:
        """One concept, as the pick needs it.

        `detail="declared"` is the PICK arm's form: the ontology only, no
        database. The storage facts below exist to stop a model writing
        `a / b` over two INTEGER columns; in the pick arm the model does not
        write the division — `spc/compile.py` does, and it casts — so the
        measurement is a query per attribute bought for nothing.
        """
        skill = "describe_concept"
        if detail not in ("declared", "storage"):
            return self._fail(skill, "BAD_ARGUMENT",
                              f"detail {detail!r} is not one of ['declared', 'storage']")
        if name not in self.ontology.concepts:
            return self._fail(
                skill, "UNKNOWN_CONCEPT", f"{name!r} is not a concept in this ontology",
                did_you_mean=self._did_you_mean(str(name)),
                known_concepts=list(self.ontology.concept_names()),
            )
        concept = self.ontology.concept(name)
        attributes = []
        for attr in concept.attributes.values():
            row = self._attribute_row(concept, attr)

            row["data"] = ({"available": False, "reason": "detail='declared': not measured"}
                           if detail == "declared" else self._column_facts(concept, attr))
            attributes.append(row)
        neighbours = [
            {
                "edge": t.edge.name,
                "direction": "forward" if t.step.forward else "reverse",
                "to_concept": t.landing,
                "role": t.step.role,
                "fan_out": t.fan_out,
                "cardinality": t.edge.cardinality,
                "description": t.edge.description,
            }
            for t in self.graph.traversals_from(name)
        ]
        data: dict[str, Any] = {
            "concept": name,
            "table": concept.table,
            "description": concept.description,
            "grain": concept.grain,
            "key_columns": list(concept.key),
            "title_attribute": concept.display,
            "title_column": concept.title_attribute.column if concept.title_attribute else None,
            "is_role_object": concept.is_role_object,
            "role_code": concept.role_code,
            "role_name": (
                self.ontology.party_roles[concept.role_code].name
                if concept.role_code in self.ontology.party_roles else None
            ),
            "role_predicates": self._concept_predicates(concept),
            "implements": concept.implements,
            "attributes": attributes,
            "searchable_attributes": [a.name for a in concept.searchable_attributes],
            "neighbours": neighbours,
            "data": ({"available": False, "reason": "detail='declared': not measured"}
                     if detail == "declared" else self._row_count(concept)),
            "database": (_rel(self.db_path)
                         if self.db_path and detail != "declared" else None),
            "declared_in": self._graph_source,
        }
        data["hazards"] = [
            attr["data"][key]
            for attr in attributes
            for key in ("empty_column", "integer_division_hazard",
                        "text_arithmetic_warning", "date_storage_warning")
            if attr["data"].get(key)
        ]

        data["ungroundable_attributes"] = [
            attr["name"] for attr in attributes
            if attr["type"] in ("string", None)
            and not attr["searchable"]
            and not attr["is_key"]
            and not (attr["resolved_via"] and self._inherits_searchable(concept, attr["name"]))
        ]
        return SkillResult(skill, True, data, self._render_concept(data))

    def _inherits_searchable(self, concept: Concept, attribute: str) -> tuple[str, str] | None:
        """`(far concept, far attribute)` when a `via` attribute resolves to a
        searchable one, else None.

        The ontology FORMAT cannot express this — there is no `searchable:
        inherit` — so the inheritance is inferred here and always reported, never
        applied silently.
        """
        attr = concept.attributes.get(attribute)
        if attr is None or attr.via is None:
            return None
        edge = self.ontology.edge(attr.via)
        far = edge.target if edge.source == concept.name else edge.source
        target = self.ontology.concept(far).attributes.get(attr.via_attribute or "")
        return (far, target.name) if target and target.searchable else None

    @staticmethod
    def _render_concept(data: Mapping[str, Any]) -> str:
        rows = data["data"].get("rows")
        head = f"{data['concept']}  (table {data['table']}"
        head += f", {rows} rows in this snapshot)" if rows is not None else ")"
        lines = [head]
        if data["description"]:
            lines.append(f"  {' '.join(str(data['description']).split())}")
        lines.append(f"  key: {', '.join(data['key_columns']) or '(none declared)'}")
        if data["is_role_object"]:
            role = data["role_name"] or data["role_code"]
            lines.append(
                f"  ROLE OBJECT — commits to role {data['role_code']} ({role}). "
                f"Every query over this concept MUST carry: "
                + "; ".join(p["sql"] for p in data["role_predicates"])
            )
        if data["implements"]:
            lines.append(f"  implements: {data['implements']}")
        lines.append(f"  title attribute: {data['title_attribute']} ({data['title_column']})")
        lines.append("  attributes:")
        for attr in data["attributes"]:
            if attr["resolved_via"]:
                where = (f"via {attr['resolved_via']['edge']} -> "
                         f"{attr['resolved_via']['concept']}.{attr['resolved_via']['attribute']}")
            else:
                where = str(attr["qualified_column"])
            stored = attr["data"].get("storage_type")
            flags = "".join([
                " [key]" if attr["is_key"] else "",
                " [title]" if attr["is_title"] else "",
                " [searchable]" if attr["searchable"] else "",
                f" [stored as {stored}]" if attr["data"].get("storage_matches_declared") is False
                else "",
            ])
            lines.append(f"    {attr['name']}: {where}  type={attr['type']}{flags}")
        for hazard in data["hazards"]:
            lines.append(f"  !! {hazard}")
        if data["ungroundable_attributes"]:
            lines.append(
                "  not searchable (no literal can be grounded against these; the ontology "
                f"declares no `searchable` for them): {', '.join(data['ungroundable_attributes'])}"
            )
        if data["data"].get("note"):
            lines.append(f"  !! {data['data']['note']}")
        lines.append("  neighbours (one hop):")
        for neighbour in data["neighbours"]:
            mark = "" if neighbour["direction"] == "forward" else "~"
            role = f" role={neighbour['role']}" if neighbour["role"] else ""
            lines.append(
                f"    -{mark}{neighbour['edge']}-> {neighbour['to_concept']}"
                f"  fan-out={neighbour['fan_out']}{role}"
            )
        return "\n".join(lines)

    def list_concepts(self) -> SkillResult:
        skill = "list_concepts"
        concepts = []
        for name in self.ontology.concept_names():
            concept = self.ontology.concept(name)
            concepts.append({
                "concept": name,
                "table": concept.table,
                "description": (
                    " ".join(str(concept.description).split()) if concept.description else None
                ),
                "is_role_object": concept.is_role_object,
                "role_code": concept.role_code,
                "attribute_count": len(concept.attributes),
                "attributes": sorted(concept.attributes),
                "neighbour_count": self.graph.degree(name),
            })
        data: dict[str, Any] = {
            "domain": self.ontology.domain,
            "version": self.ontology.version,
            "concept_count": len(concepts),
            "concepts": concepts,
            "edges": sorted(e.name for e in self.ontology.edges),
            "metrics": sorted(self.ontology.metrics),
            "links": sorted(self.ontology.links),
            "party_roles": [
                {"code": code, "name": self.ontology.party_roles[code].name,
                 "description": self.ontology.party_roles[code].description}
                for code in sorted(self.ontology.party_roles)
            ],
            "declared_in": [self._graph_source, self._semantic_source],
        }
        lines = [
            f"{data['domain']} v{data['version']}: {data['concept_count']} concepts, "
            f"{len(data['edges'])} governed edges, {len(data['metrics'])} governed metrics."
        ]
        for concept in concepts:
            tag = f" [role object: {concept['role_code']}]" if concept["is_role_object"] else ""
            lines.append(
                f"  {concept['concept']} (table {concept['table']}, "
                f"{concept['attribute_count']} attrs, {concept['neighbour_count']} neighbours){tag}"
            )
        lines.append(f"  metrics: {', '.join(data['metrics'])}")
        return SkillResult(skill, True, data, "\n".join(lines))

    SUBJECT_LIST_CAP = 40

    def list_subjects(self, question: str | None = None) -> SkillResult:
        """The subjects a question may be anchored on, and the measure names.

        REPLACES `list_concepts`, which returned the WHOLE ontology -- every
        concept with its table, attributes and neighbour count, plus every edge,
        metric, link and party role. Measured on the 2026-08-12 campaign, that
        payload was ~1,192 tokens and was requested on 215 of 215 runs and was
        the FIRST call on 215 of 215, while `search_concepts` -- the tool built
        to avoid it -- was used on 5. There was no retrieval step: the ontology
        was dumped and the model did entity-linking in context.

        Two things are wrong with that and only one is cost. The cost is linear
        in ontology size and paid per question (~46k tokens at 500 concepts, at
        which point "low cost" is not a claim we can make). The deeper problem is
        that a system which shows the model the entire graph cannot be described
        as retrieving over a graph, and its accuracy is not the accuracy of one
        that does.

        What step 1 needs is exactly: what can I anchor on, and what quantities
        are named? So that is what this returns -- subject name, ONE short
        sentence, and the measure names. No tables, no attributes, no edges, no
        neighbour counts, no party roles. Those are step 2, and they are fetched
        for the few subjects the question actually reaches.

        Above `SUBJECT_LIST_CAP` the list is RANKED against `question` and the top
        `SUBJECT_LIST_CAP` are returned -- a page size, not a cliff. Three bugs in
        the first version of this method are the reason it works that way, all
        found in review on 2026-08-13:

          * `measures` was returned IN FULL regardless of the cap, and a metric
            registry grows with ontology size exactly as the concept list does.
            The cap bounded one of two linear terms and the docstring claimed
            "bounded by construction" anyway. Both are capped now.
          * `undocumented_subjects` was computed from the withheld list, so it
            vanished precisely in the regime where the planner is most blind.
            It is now computed from what is actually returned, always.
          * withholding the list entirely left stage 1 with no enum to validate
            against, so every subject the model named became `unknown`. Ranking
            keeps the enum non-empty at any ontology size.

        `question` is optional only so the tool stays callable for orientation.
        Without it, and over the cap, ranking is impossible and the list is
        withheld -- which is the honest answer, not a silent truncation to an
        arbitrary alphabetical prefix.
        """
        skill = "list_subjects"
        names = list(self.ontology.concept_names())
        metrics = sorted(self.ontology.metrics)
        cap = self.SUBJECT_LIST_CAP

        def short(text: Any) -> str | None:
            """One sentence, whitespace-collapsed. A paragraph is step 2's job."""
            if not text:
                return None
            flat = " ".join(str(text).split())
            head = flat.split(". ")[0].strip().rstrip(".")
            return head or None

        def entry(name: str) -> dict[str, Any]:
            return {"subject": name,
                    "description": short(self.ontology.concept(name).description)}

        over = len(names) > cap
        ranked = bool(question) and over
        if not over:
            subjects = [entry(n) for n in names]
        elif ranked:

            query = _tokens(question or "")
            scored = sorted(
                ((max(_score(query, n),
                      _score(query, self.ontology.concept(n).description or "")), n)
                 for n in names),
                key=lambda pair: (-pair[0], pair[1]),
            )
            subjects = [entry(n) for score, n in scored[:cap] if score > 0]
        else:
            subjects = []

        measures_shown = metrics if len(metrics) <= cap else metrics[:cap]

        undocumented = [s["subject"] for s in subjects if not s["description"]]

        data: dict[str, Any] = {
            "domain": self.ontology.domain,
            "version": self.ontology.version,
            "subject_count": len(names),
            "subjects": subjects,
            "measures": measures_shown,
            "measure_count": len(metrics),
            "truncated": over,
            "ranked_against_question": ranked,
            "declared_in": [self._graph_source, self._semantic_source],
        }
        if undocumented:
            data["undocumented_subjects"] = undocumented
        if over and not ranked:
            data["next_call"] = "list_subjects(question=<the question>)"

        head = f"{data['domain']} v{data['version']}: {len(names)} subjects"
        if ranked:
            lines = [f"{head}; the {len(subjects)} most relevant to this question:"]
        elif over:
            lines = [f"{head} -- too many to list (cap {cap}). Call again with "
                     f"`question` to get the ones this question names."]
        else:
            lines = [f"{head}."]
        for s in subjects:
            lines.append(f"  {s['subject']}"
                         + (f" -- {s['description']}" if s["description"] else ""))
        lines.append(f"  measures: {', '.join(measures_shown)}"
                     + (f" (+{len(metrics) - len(measures_shown)} more)"
                        if len(measures_shown) < len(metrics) else ""))

        return SkillResult(skill, True, data, "\n".join(lines))

    _KIND_RANK = {"concept": 0, "attribute": 1, "metric": 2, "link": 3, "edge": 4}

    def search_concepts(self, phrase: str, limit: int = 10) -> SkillResult:
        skill = "search_concepts"
        query = _tokens(phrase)
        if not query:
            return self._fail(skill, "BAD_ARGUMENT", "phrase is empty after normalisation")
        try:
            top = int(limit)
        except (TypeError, ValueError):
            return self._fail(skill, "BAD_ARGUMENT", f"limit {limit!r} is not an integer")

        hits: list[dict[str, Any]] = []

        def add(kind: str, name: str, fields: Sequence[tuple[str, Any]], **extra: Any) -> None:
            best, where = 0, ""
            for field_name, text in fields:
                score = _score(query, str(text or ""))
                if score > best:
                    best, where = score, field_name
            if best <= 0:
                return
            row = {"kind": kind, "name": name, "score": best, "matched_on": where}
            row.update(extra)
            hits.append(row)

        for cname in self.ontology.concept_names():
            concept = self.ontology.concept(cname)
            add("concept", cname,
                [("name", cname), ("table", concept.table), ("description", concept.description)],
                table=concept.table, next_call=f"describe_concept({cname!r})")
            for aname in sorted(concept.attributes):
                attr = concept.attributes[aname]

                labels = [((f"label {text!r}" if i == 0 else f"alias {text!r}"), text)
                          for i, text in enumerate(attr.labels)]
                add("attribute", attr.qualified,
                    [("name", aname), ("column", attr.column or "")]
                    + labels + [("description", attr.description)],
                    concept=cname, column=attr.column, searchable=attr.searchable,
                    next_call=f"describe_concept({cname!r})")
        for edge in sorted(self.ontology.edges, key=lambda e: e.name):
            add("edge", edge.name,
                [("name", edge.name), ("description", edge.description)],
                from_concept=edge.source, to_concept=edge.target,
                next_call=f"find_paths({edge.source!r}, {edge.target!r})")
        for mname in sorted(self.ontology.metrics):
            metric = self.ontology.metrics[mname]
            add("metric", mname,
                [("name", mname), ("description", metric.description)],
                next_call=f"find_metric({mname!r})")
        for lname in sorted(self.ontology.links):
            link = self.ontology.links[lname]
            add("link", lname,
                [("name", lname), ("predicate", link.predicate),
                 ("description", link.description)],
                from_concept=link.source, to_concept=link.target,
                backed_by=list(link.backed_by),
                next_call=f"find_paths({link.source!r}, {link.target!r})")
        for entry in self.ontology.glossary:
            for text in entry.phrases:
                score = _score(query, text)
                if score > 0:
                    hits.append({
                        "kind": entry.kind, "name": entry.target, "score": score,
                        "matched_on": f"glossary phrase {text!r}",
                        "next_call": (f"find_metric({text!r})" if entry.kind == "metric"
                                      else f"find_metric({text!r})"),
                    })

        best_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for hit in hits:
            key = (str(hit["kind"]), str(hit["name"]))
            current = best_by_key.get(key)
            if current is None or int(hit["score"]) > int(current["score"]):
                best_by_key[key] = hit
        ordered = sorted(
            best_by_key.values(),
            key=lambda h: (-int(h["score"]), self._KIND_RANK.get(str(h["kind"]), 9), str(h["name"])),
        )
        results = ordered[: max(top, 0)]
        data: dict[str, Any] = {
            "phrase": phrase,
            "normalised": _phrase(phrase),
            "match_count": len(ordered),
            "returned": len(results),
            "results": results,
        }
        lines = [f"search_concepts({phrase!r}): {len(ordered)} match(es)"]
        for hit in results:
            lines.append(
                f"  {hit['score']:3d}  {hit['kind']:<9} {hit['name']}"
                f"   (matched {hit['matched_on']}) -> {hit.get('next_call', '')}"
            )
        if not results:
            lines.append("  nothing matched; `list_concepts()` shows every entry point.")
        return SkillResult(skill, True, data, "\n".join(lines))

    def find_metric(self, phrase: str, subject: str | None = None) -> SkillResult:
        """A business term -> the one authored definition, and how to reach it.

        `subject` is the pick arm's addition. A simple governed metric names the
        EDGE its operand must be reached by, and the compiler enforces that: a
        measure carrying a route id whose last step is a different edge to the
        same concept is refused, because a different edge to the same column is
        a DIFFERENT quantity. That pairing was previously recoverable only by
        cross-referencing this tool's `via_edge` against `find_paths`' edge
        chains by eye. Passing the pick's subject makes it explicit — the answer
        names the route ids that are compatible, and says so when none is.
        """
        skill = "find_metric"
        query = _tokens(phrase)
        if not query:
            return self._fail(skill, "BAD_ARGUMENT", "phrase is empty after normalisation")
        if subject is not None and subject not in self.ontology.concepts:
            return self._fail(
                skill, "UNKNOWN_CONCEPT", f"subject {subject!r} is not a concept in this ontology",
                did_you_mean=self._did_you_mean(str(subject)),
            )

        scores: dict[str, tuple[int, str, str]] = {}
        link_hits: dict[str, tuple[int, str, str]] = {}

        def bump(table: dict[str, tuple[int, str, str]], key: str, score: int,
                 why: str, text: str) -> None:
            if score <= 0:
                return
            current = table.get(key)
            if current is None or score > current[0]:
                table[key] = (score, why, text)

        for entry in self.ontology.glossary:
            table = scores if entry.kind == "metric" else link_hits
            for text in entry.phrases:
                bump(table, entry.target, _score(query, text),
                     f"glossary phrase {text!r}", text)
        for mname in sorted(self.ontology.metrics):
            metric = self.ontology.metrics[mname]
            bump(scores, mname, _score(query, mname), "metric name", mname)
            bump(scores, mname, min(_score(query, metric.description or ""), 60),
                 "description", metric.description or "")

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1][0], kv[0]))
        links = sorted(link_hits.items(), key=lambda kv: (-kv[1][0], kv[0]))

        unmatched: list[str] = []
        if ranked:
            accounted = set(_tokens(ranked[0][1][2])) | set(_tokens(ranked[0][0]))
            unmatched = [w for w in dict.fromkeys(query) if w not in accounted]

        if not ranked:
            status = "absent"
            definition = None
        elif len(ranked) > 1 and ranked[0][1][0] == ranked[1][1][0]:
            status = "ambiguous"
            definition = None
        elif unmatched:
            status = "partial"
            definition = self._expand_metric(ranked[0][0])
        else:
            status = "unique"
            definition = self._expand_metric(ranked[0][0])

        data: dict[str, Any] = {
            "phrase": phrase,
            "status": status,
            "unmatched_words": unmatched,
            "matches": [
                {"metric": name, "score": score, "matched_on": why}
                for name, (score, why, _text) in ranked
            ],
            "definition": definition,
            "subject": subject,
            "routes_from_subject": (
                self._metric_routes(subject, definition) if subject and definition else []
            ),

            "links": [
                {
                    "link": name, "score": score, "matched_on": why,
                    "predicate": self.ontology.links[name].predicate,
                    "from_concept": self.ontology.links[name].source,
                    "to_concept": self.ontology.links[name].target,
                    "backed_by": list(self.ontology.links[name].backed_by),
                    "description": self.ontology.links[name].description,
                }
                for name, (score, why, _text) in links
            ],

            "attributes": [
                {"attribute": qualified, "score": score, "matched_on": why}
                for qualified, score, why, _text in self._attribute_hits(query)
                if score >= _LABEL_MATCH
            ][:3],

            "registry": sorted(self.ontology.metrics),
            "declared_in": self._semantic_source,
        }
        if status == "absent":
            grounded = ", ".join(hit["attribute"] for hit in data["attributes"])
            data["note"] = (
                f"No governed metric matches {phrase!r}. The registry is closed: "
                f"{', '.join(data['registry'])}. If the question needs a quantity that is "
                "not in it, aggregate an attribute directly (`describe_concept`) — do not "
                "invent a variant of a governed metric."
                + (f" This phrase does name a labelled ATTRIBUTE ({grounded}); it is a "
                   "property to project or filter on, not a quantity to compute."
                   if grounded else "")
            )
        elif status == "ambiguous":
            data["note"] = (
                f"{phrase!r} matches more than one governed metric equally well "
                f"({', '.join(n for n, _ in ranked[:2])}). Ask, or name the metric exactly."
            )
        elif status == "partial":
            data["note"] = (
                f"The registry's closest entry is {ranked[0][0]}, but it accounts for only "
                f"part of {phrase!r}: {', '.join(repr(w) for w in unmatched)} "
                f"{'is' if len(unmatched) == 1 else 'are'} not in the governed definition. "
                f"The definition below is EXACT for what it covers; whatever the remaining "
                f"words ask for is NOT governed and you must express it yourself — as a plain "
                f"aggregation, or by combining governed metrics with `combine`."
            )

        if data["attributes"] and ranked and data["attributes"][0]["score"] > ranked[0][1][0]:
            stronger = data["attributes"][0]
            data["note"] = (
                (data.get("note", "") + " ") if data.get("note") else ""
            ) + (
                f"NOTE: {phrase!r} matches the attribute {stronger['attribute']} "
                f"({stronger['score']}) better than any metric ({ranked[0][0]}, "
                f"{ranked[0][1][0]}). If the question asks for that property rather than "
                "a quantity, project the attribute and compute nothing."
            )
        return SkillResult(skill, True, data, self._render_metric(data))

    def _metric_routes(self, subject: str, definition: Mapping[str, Any]) -> list[dict[str, Any]]:
        """The route ids a measure of this metric may carry, from `subject`.

        ONE row, because a `Measure` carries ONE route, and the row names the
        concept that route must land on — the compiler's rule, restated here so
        the model can obey it instead of discovering it as a refusal.

        Which concept that is depends on the metric's shape, and this is exactly
        where the tool used to mislead. A LEAF metric is defined over one edge,
        so its route lands on the operand concept and must ARRIVE by that edge:
        a different edge to the same column is a different quantity. A COMPOSITE
        has one edge per component and cannot have them all in one route, so its
        route lands on the concept the composite is a quantity OF
        (`Ontology.measured_over`) and the compiler appends each component's own
        edge — total loss OF a claim, however you reached the claim. Where a
        composite is a quantity of more than one concept, as a ratio of two
        differently-grained metrics is, no route can name where it is measured
        and only SELF is offered: the compiler then derives each component's
        route itself.

        Previously this walked into the components and offered a route id per
        component operand — four route ids for a measure that can carry one, all
        of them landing on the wrong concept for a composite. `find_metric` said
        `Policy>ClaimAmount#1` for TotalLoss; that compiled only because the
        compiler was discarding the route.
        """
        name = str(definition.get("metric") or "")
        bases = self.ontology.measured_over(name) if name else ()
        composite = bool(definition.get("components"))
        operand = definition.get("operand")
        via = (operand.get("via_edge")
               if isinstance(operand, Mapping) and not composite else None)

        if composite and len(bases) != 1:
            return [{
                "metric": name,
                "operand_concept": None,
                "via_edge": None,
                "compatible_route_ids": [SUBJECT_ID],
                "note": (
                    f"{name} is a quantity of more than one concept "
                    f"({', '.join(bases)}), so no single route says where it is "
                    f"measured. Leave the route as {SUBJECT_ID}: each part is then "
                    f"measured over the subject, at the grain the subject fixes."
                ),
            }]

        concept = (bases[0] if composite
                   else (str(operand["concept"])
                         if isinstance(operand, Mapping) and operand.get("concept")
                         else None))
        if concept is None:                                   # pragma: no cover
            return []

        component_edges = tuple(_collect_edges(definition)) if composite else ()

        identifiers: list[str] = []
        if concept == subject and via is None:
            identifiers.append(SUBJECT_ID)
        for index, path in enumerate(
            self.graph.paths(subject, concept, max_hops=DEFAULT_MAX_HOPS), 1
        ):
            if via is not None and not (path.steps and path.steps[-1].edge == via
                                        and path.steps[-1].forward):
                continue
            if any(self.graph.extend(subject, path, edge) is None
                   for edge in component_edges):
                continue
            identifiers.append(route_id(subject, concept, index))
        if composite:
            note = (
                f"{name} is a quantity of a {concept}; any of these routes reaches one, "
                f"and each part of the definition then appends its own edge"
                if identifiers else
                f"NO governed route from {subject} arrives at {concept} within "
                f"{DEFAULT_MAX_HOPS} hops, so {name} cannot be measured at this "
                f"subject's grain — choose a different subject."
            )
        else:
            note = (
                f"any of these may carry {name}"
                if identifiers else
                f"NO governed route from {subject} arrives at {concept}"
                + (f" by {via}" if via else "")
                + f" within {DEFAULT_MAX_HOPS} hops, so {name} cannot be "
                  f"measured at this subject's grain — choose a different subject."
            )
        return [{
            "metric": name,
            "operand_concept": concept,
            "via_edge": via,
            "compatible_route_ids": identifiers,
            "note": note,
        }]

    def _expand_metric(self, name: str, seen: tuple[str, ...] = ()) -> dict[str, Any]:
        """The governed definition, fully expanded, with the edges it traverses.

        Composition is validated acyclic at load, so this recursion terminates;
        `seen` is carried only so the structure records the expansion chain.
        """
        metric: Metric = self.ontology.metric(name)
        node: dict[str, Any] = {
            "metric": name,
            "description": " ".join(str(metric.description).split()) if metric.description else None,
        }
        if metric.is_composite:
            node["kind"] = "composite"
            node["combine"] = metric.combine
            node["operator"] = _COMBINE_SYMBOL.get(str(metric.combine), str(metric.combine))
            node["components"] = [
                self._expand_metric(c, seen + (name,)) for c in metric.components
            ]
        else:
            operand = metric.operand
            concept = self.ontology.concept(str(operand.concept)) if operand else None
            attribute = (
                concept.attributes.get(str(operand.attribute)) if concept and operand else None
            )
            node["kind"] = "aggregation"
            node["op"] = metric.op
            node["operand"] = {
                "concept": operand.concept if operand else None,
                "attribute": operand.attribute if operand else None,
                "table": concept.table if concept else None,
                "column": attribute.column if attribute else None,
                "type": attribute.type if attribute else None,

                "via_edge": operand.via if operand else None,
            }
            if operand and operand.via:
                edge = self.ontology.edge(operand.via)
                node["operand"]["via_declared_as"] = f"{edge.source} -> {edge.target}"
                node["operand"]["via_description"] = edge.description
            node["sql_sketch"] = (
                f"{str(metric.op).upper().replace('COUNT_DISTINCT', 'COUNT(DISTINCT')}"
                f"({concept.table}.{attribute.column})"
                if concept and attribute and metric.op != "count_distinct"
                else (f"COUNT(DISTINCT {concept.table}.{attribute.column})"
                      if concept and attribute else None)
            )
        node["expansion_edges"] = sorted(_collect_edges(node))
        return node

    def _render_metric(self, data: Mapping[str, Any]) -> str:
        lines = [f"find_metric({data['phrase']!r}): {data['status']}"]
        if data["definition"]:
            lines.extend(_render_definition(data["definition"], indent=2))
            edges = data["definition"]["expansion_edges"]
            lines.append(
                f"  edges traversed by this definition: {', '.join(edges) or '(none)'}"
            )
            lines.append(
                "  This is the ONE authored definition. Compute it exactly; do not "
                "substitute an improvised aggregation."
            )
        for row in data.get("routes_from_subject") or ():
            where = f" arriving by {row['via_edge']}" if row["via_edge"] else ""
            lines.append(
                f"  {row['metric']} is measured on {row['operand_concept']}{where}; from "
                f"{data['subject']} use route: "
                f"{', '.join(row['compatible_route_ids']) or 'NONE — ' + row['note']}"
            )
        for hit in data["matches"][:5]:
            lines.append(f"  candidate {hit['score']:3d}  {hit['metric']}  ({hit['matched_on']})")
        for link in data["links"]:
            lines.append(
                f"  note: {link['matched_on']} names the business LINK {link['link']} "
                f"({link['from_concept']} -> {link['to_concept']}, "
                f"backed by {', '.join(link['backed_by'])}), not a metric."
            )
        for hit in data.get("attributes") or ():
            lines.append(
                f"  note: {hit['matched_on']} names the ATTRIBUTE {hit['attribute']}, "
                f"not a metric."
            )
        if data.get("note"):
            lines.append(f"  {data['note']}")
        return "\n".join(lines)

    def attribute_labels(self) -> tuple[tuple[str, str, str], ...]:
        """`(qualified_attribute, phrase, role)` for every authored label.

        Ordered by concept, then attribute, then authored position, so two
        processes agree. `role` is `primary` for the attribute's own label and
        `alias` for the rest: the distinction is authored, never inferred, and a
        resolver may use it to break an otherwise exact tie.
        """
        rows: list[tuple[str, str, str]] = []
        for cname in self.ontology.concept_names():
            concept = self.ontology.concept(cname)
            for aname in sorted(concept.attributes):
                attr = concept.attributes[aname]
                for index, text in enumerate(attr.labels):
                    rows.append(
                        (attr.qualified, text, "primary" if index == 0 else "alias")
                    )
        return tuple(rows)

    def unlabelled_attributes(self) -> tuple[str, ...]:
        """Attributes no phrase can reach. Empty is the property a lint asserts."""
        return tuple(
            attr.qualified
            for cname in self.ontology.concept_names()
            for _aname, attr in sorted(self.ontology.concept(cname).attributes.items())
            if not attr.labels
        )

    def label_collisions(self) -> tuple[dict[str, Any], ...]:
        """Phrases that name more than one attribute — REPORTED, not resolved.

        A collision is not automatically a defect: two concepts may legitimately
        both be called "amount", and `find_attribute` answers `ambiguous` for
        such a phrase rather than picking the first. What must never happen is
        silence, so this is a first-class query and a lint asserts on it.
        """
        by_phrase: dict[str, list[tuple[str, str, str]]] = {}
        for qualified, text, role in self.attribute_labels():
            by_phrase.setdefault(_phrase(text), []).append((qualified, text, role))
        out: list[dict[str, Any]] = []
        for normalised in sorted(by_phrase):
            entries = by_phrase[normalised]
            owners = sorted({q for q, _t, _r in entries})
            if len(owners) < 2:
                continue
            out.append({
                "normalised": normalised,
                "attributes": owners,
                "labels": [
                    {"attribute": q, "phrase": t, "role": r}
                    for q, t, r in sorted(entries)
                ],
            })
        return tuple(out)

    def _attribute_hits(
        self, query: Sequence[str], concept: str | None = None
    ) -> list[tuple[str, int, str, str]]:
        """Ranked `(qualified, score, why, text)`, best first, ties by name.

        Scores each authored label separately and the attribute's own
        identifier alongside them, exactly as `find_metric` scores each glossary
        phrase and the metric's own name. Total order, so the ranking is a
        function of the ontology and not of dict iteration.
        """
        best: dict[str, tuple[int, str, str]] = {}

        def bump(key: str, score: int, why: str, text: str) -> None:
            if score <= 0:
                return
            current = best.get(key)
            if current is None or score > current[0]:
                best[key] = (score, why, text)

        for cname in self.ontology.concept_names():
            if concept is not None and cname != concept:
                continue
            for aname in sorted(self.ontology.concept(cname).attributes):
                attr = self.ontology.concept(cname).attributes[aname]
                for index, text in enumerate(attr.labels):
                    bump(attr.qualified, _score(query, text),
                         f"{'label' if index == 0 else 'alias'} {text!r}", text)

                bump(attr.qualified, _score(query, attr.name),
                     "attribute name", attr.name)
        return [
            (key, score, why, text)
            for key, (score, why, text) in sorted(
                best.items(), key=lambda kv: (-kv[1][0], kv[0])
            )
        ]

    def find_attribute(self, phrase: str, concept: str | None = None) -> SkillResult:
        """A business phrase -> the ONE attribute it names, or an honest refusal.

        The statuses mirror `find_metric` exactly, and mean the same things:

          `unique`     one attribute outscores every other, and every word of
                       the phrase is accounted for by the label that won.
          `partial`    one attribute wins, but words are left over. The winner
                       is returned and the leftovers are named — the difference
                       between "claim open date" and "claim open date in 2019".
          `ambiguous`  two or more attributes tie at the top. Reported with all
                       of them; the caller narrows with `concept=`, and NOTHING
                       is picked on the caller's behalf.
          `absent`     no label matched at all.

        `concept` restricts the search to one concept, which is how a caller
        resolves a phrase it already knows the subject of.
        """
        skill = "find_attribute"
        query = _tokens(phrase)
        if not query:
            return self._fail(skill, "BAD_ARGUMENT", "phrase is empty after normalisation")
        if concept is not None and concept not in self.ontology.concepts:
            return self._fail(
                skill, "UNKNOWN_CONCEPT", f"concept {concept!r} is not in this ontology",
                did_you_mean=self._did_you_mean(str(concept)),
            )

        ranked = self._attribute_hits(query, concept)
        unmatched: list[str] = []
        if ranked:
            accounted = set(_tokens(ranked[0][3])) | set(_tokens(ranked[0][0]))
            unmatched = [w for w in dict.fromkeys(query) if w not in accounted]

        if not ranked:
            status, winner = "absent", None
        elif len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            status, winner = "ambiguous", None
        elif unmatched:
            status, winner = "partial", ranked[0][0]
        else:
            status, winner = "unique", ranked[0][0]

        def row(qualified: str, score: int, why: str) -> dict[str, Any]:
            cname, aname = qualified.split(".", 1)
            attr = self.ontology.concept(cname).attributes[aname]
            return {
                "attribute": qualified,
                "concept": cname,
                "name": aname,
                "score": score,
                "matched_on": why,
                "label": attr.label,
                "aliases": list(attr.aliases),
                "type": attr.type,
                "value_type": attr.value_type,
                "searchable": attr.searchable,
                "next_call": f"describe_concept({cname!r})",
            }

        data: dict[str, Any] = {
            "phrase": phrase,
            "concept": concept,
            "status": status,
            "unmatched_words": unmatched,
            "attribute": winner,
            "matches": [row(q, s, w) for q, s, w, _t in ranked[:8]],
            "match_count": len(ranked),
            "declared_in": self._semantic_source,
        }
        if status == "absent":
            data["note"] = (
                f"No attribute is labelled {phrase!r}. `search_concepts` searches "
                "concepts, edges, metrics and links as well; an unlabelled attribute "
                "is a gap in the ontology, not a phrase to guess a column for."
            )
        elif status == "ambiguous":
            tied = [q for q, s, _w, _t in ranked if s == ranked[0][1]]
            data["tied"] = tied
            data["note"] = (
                f"{phrase!r} names {len(tied)} attributes equally well "
                f"({', '.join(tied)}). Say which concept you mean — pass `concept=` "
                "— rather than letting one be chosen for you."
            )
        elif status == "partial":
            data["note"] = (
                f"{winner} is the closest labelled attribute, but "
                f"{', '.join(repr(w) for w in unmatched)} "
                f"{'is' if len(unmatched) == 1 else 'are'} not accounted for by its "
                "labels. Whatever those words ask for is not part of this attribute."
            )
        return SkillResult(skill, True, data, self._render_attribute(data))

    @staticmethod
    def _render_attribute(data: Mapping[str, Any]) -> str:
        lines = [f"find_attribute({data['phrase']!r}): {data['status']}"]
        if data["attribute"]:
            top = data["matches"][0]
            labels = ", ".join(
                [repr(top["label"])] if top["label"] else []
            ) + "".join(f", {alias!r}" for alias in top["aliases"])
            lines.append(
                f"  {top['attribute']}  ({top['type'] or 'untyped'}"
                f"{', searchable' if top['searchable'] else ''})  labels: {labels}"
            )
        for hit in data["matches"][:5]:
            lines.append(
                f"  candidate {hit['score']:3d}  {hit['attribute']}  ({hit['matched_on']})"
            )
        if data.get("note"):
            lines.append(f"  {data['note']}")
        return "\n".join(lines)

    def search_values(
        self, concept: str, attribute: str, phrase: str, follow_via: bool = True
    ) -> SkillResult:
        """Resolve a phrase to the canonical value the data holds.

        WHAT MAY BE SEARCHED. Only attributes the ontology declares
        `searchable`. That list is the ontology's and it is closed: without it,
        "resolve this phrase" becomes an arbitrary read over the warehouse, and
        the guarantee that a governed surface has no escape hatch (DESIGN rule
        4) goes with it.

        `via` ATTRIBUTES ARE FOLLOWED, and the choice is deliberate. A role
        object's `name` is declared as `via: <edge>, attribute: <far attr>`, and
        that far attribute IS declared searchable — so reading it through the
        declared edge is inside the governed surface; what the FORMAT lacks is
        a way to say the searchability travels (there is no `searchable:
        inherit`). Refusing would leave the most natural question about a role
        object — "which policies does <name> hold?" — with no tool that can
        answer it, now that there is no ingress stage to resolve it beforehand.
        So it is followed, `inherited` is set, and the rung records the edge, so
        the inference is never invisible in an audit. `follow_via=false`
        restores the strict refusal.
        """
        skill = "search_values"
        if concept not in self.ontology.concepts:
            return self._fail(
                skill, "UNKNOWN_CONCEPT", f"{concept!r} is not a concept in this ontology",
                did_you_mean=self._did_you_mean(str(concept)),
            )
        target = self.ontology.concept(concept)
        if attribute not in target.attributes:
            return self._fail(
                skill, "UNKNOWN_ATTRIBUTE", f"{concept} has no attribute {attribute!r}",
                known_attributes=sorted(target.attributes),
                searchable_attributes=[a.name for a in target.searchable_attributes],
            )
        snapshot, reason = self.snapshot()
        if snapshot is None:
            return self._fail(
                skill, "SNAPSHOT_UNAVAILABLE",
                f"canonical values cannot be read: {reason}",
                consequence="A literal written without grounding may match zero rows silently.",
            )

        attr = target.attributes[attribute]
        inherited = self._inherits_searchable(target, attribute)
        data: dict[str, Any] = {
            "concept": concept,
            "attribute": attribute,
            "phrase": phrase,
            "status": "",
            "value": None,
            "matches": [],
            "match_rung": "",
            "inherited": None,
            "searchable_attributes": [a.name for a in target.searchable_attributes],
            "database": _rel(self.db_path),
            "snapshot": snapshot.digest,
        }

        if attr.searchable and attr.column:
            read_concept, read_attr = target, attr
        elif inherited and follow_via:
            far_concept, far_attribute = inherited
            read_concept = self.ontology.concept(far_concept)
            read_attr = read_concept.attributes[far_attribute]
            data["inherited"] = {
                "via_edge": attr.via,
                "concept": far_concept,
                "attribute": far_attribute,
                "column": read_attr.column,
                "why": "this attribute is not itself declared searchable; it RESOLVES "
                       "through the declared edge to one that is. The ontology format "
                       "cannot declare inherited searchability, so the inference is "
                       "recorded here rather than assumed.",
            }
        elif inherited:
            data["status"] = "unsearchable_via"
            data["matches"] = [f"{inherited[0]}.{inherited[1]}"]
            return self._value_result(data, target)
        else:
            data["status"] = "unsearchable"
            return self._value_result(data, target)

        rung, matches = snapshot.values(
            read_concept.table, str(read_attr.column), str(phrase),
            where=self._where(read_concept, "t"),
        )
        data["match_rung"] = rung if not data["inherited"] else f"{rung} (via {attr.via})"
        data["matches"] = list(matches)
        if not matches:
            data["status"] = "absent"

            facts = self._column_facts(read_concept, read_attr)
            data["column_is_empty"] = bool(facts.get("empty_column"))
            if facts.get("empty_column"):
                data["reason"] = facts["empty_column"]
        elif len(matches) == 1:
            data["status"] = "unique"
            data["value"] = matches[0]
        else:
            data["status"] = "ambiguous"
        return self._value_result(data, target)

    def _value_result(self, data: dict[str, Any], concept: Concept) -> SkillResult:
        notes = {
            "unique": "Use this exact value as the literal, spelled exactly as returned.",
            "ambiguous": "Several values match. This is a question to ask, not a value to guess.",
            "absent": "Nothing in this snapshot matches. Any predicate built from this phrase "
                      "returns zero rows — the phrase, the attribute or the concept is wrong.",
            "unsearchable": "The ontology declares no `searchable` for this attribute, so no "
                            "literal can be grounded against it and a filter on it can only be "
                            "guessed. Searchable attributes of this concept are listed; if none "
                            "serves, this is a gap in the ontology, not a question you can answer "
                            "safely.",
            "unsearchable_via": "Not searchable itself, but it RESOLVES to an attribute that is "
                                "(named in `matches`). Re-issue against that concept and "
                                "attribute, or pass follow_via=true.",
        }
        data["note"] = data.get("reason") or notes.get(str(data["status"]), "")
        if data["inherited"]:

            table = self.ontology.concept(str(data["inherited"]["concept"])).table
            column = data["inherited"]["column"]
        else:
            table = concept.table
            column = concept.attributes[str(data["attribute"])].column
        where = (
            f"{table}.{column} = '{data['value']}'"
            if data["status"] == "unique" and column else None
        )
        data["filter_sql"] = where
        head = f"search_values({data['concept']}.{data['attribute']}, {data['phrase']!r}) -> " \
               f"{data['status']}"
        lines = [head]
        if data["inherited"]:
            lines.append(
                f"  resolved through edge {data['inherited']['via_edge']} to "
                f"{data['inherited']['concept']}.{data['inherited']['attribute']} "
                f"(inherited searchability — recorded, not declared)"
            )
        lines.append(
            f"  matches: {', '.join(data['matches']) or '(none)'}"
            + (f"  [rung {data['match_rung']}]" if data["match_rung"] else "")
        )
        if where:
            lines.append(f"  filter: {where}")
        lines.append(f"  {data['note']}")
        return SkillResult("search_values", True, data, "\n".join(lines))

    def check_sql(self, sql: str, expect_metrics: Sequence[str] | None = None) -> SkillResult:
        skill = "check_sql"
        if not str(sql or "").strip():
            return self._fail(skill, "BAD_ARGUMENT", "sql is empty")
        checker, reason = self._checker()
        if checker is None:
            return self._fail(
                skill, "CHECKER_UNAVAILABLE",
                f"SQL cannot be governed: {reason}",
                consequence="Nothing verifies that the joins in this SQL are declared edges.",
            )
        db = self.db_path if self.db_path and FsPath(self.db_path).exists() else None
        try:
            violations = checker(
                sql, self.ontology, self.graph, db=db,
                expect_metrics=list(expect_metrics) if expect_metrics else None,
            )
        except Exception as exc:  # pragma: no cover
            return self._fail(skill, "CHECKER_FAILED", f"the checker raised: {exc}")

        rows = [
            {
                "code": str(getattr(v, "code", "")),
                "severity": str(getattr(v, "severity", "violation")),
                "message": str(getattr(v, "message", "")),
                "fragment": str(getattr(v, "fragment", "")),
                "detail": _jsonable(getattr(v, "detail", {}) or {}),
            }
            for v in violations
        ]

        rows.sort(key=lambda r: (r["severity"], r["code"], r["message"], r["fragment"],
                                 json.dumps(r["detail"], sort_keys=True)))
        proven = [r for r in rows if r["severity"] != "undecidable"]
        undecidable = [r for r in rows if r["severity"] == "undecidable"]
        data: dict[str, Any] = {
            "governed": not rows,
            "violation_count": len(proven),
            "undecidable_count": len(undecidable),
            "violations": rows,
            "database": _rel(db) if db else None,
            "expect_metrics": sorted(expect_metrics) if expect_metrics else [],
        }
        if not rows:
            data["note"] = "Certified: every join is a declared edge, roles are committed, " \
                           "grain is safe, metrics match the registry and literals are grounded."
            text = "check_sql: PASS — no violations."
        else:
            data["note"] = (
                "Repair the SQL against each violation below and re-check. An `undecidable` "
                "severity is NOT a pass: it means this shape could not be certified."
            )
            lines = [
                f"check_sql: {len(proven)} violation(s), {len(undecidable)} undecidable."
            ]
            for row in rows:
                tag = "UNDECIDABLE" if row["severity"] == "undecidable" else "VIOLATION"
                frag = f"  [{row['fragment']}]" if row["fragment"] else ""
                lines.append(f"  {tag} {row['code']}: {row['message']}{frag}")
            text = "\n".join(lines)
        return SkillResult(skill, True, data, text)

def _collect_edges(node: Mapping[str, Any]) -> list[str]:
    """Every governed edge a metric expansion traverses, duplicates collapsed."""
    found: list[str] = []
    operand = node.get("operand")
    if isinstance(operand, Mapping) and operand.get("via_edge"):
        found.append(str(operand["via_edge"]))
    for component in node.get("components") or ():
        found.extend(_collect_edges(component))
    return sorted(set(found))

def _render_definition(node: Mapping[str, Any], indent: int) -> list[str]:
    pad = " " * indent
    if node["kind"] == "composite":
        lines = [f"{pad}{node['metric']} = {f' {node['operator']} '.join(c['metric'] for c in node['components'])}"]
        for component in node["components"]:
            lines.extend(_render_definition(component, indent + 2))
        return lines
    operand = node["operand"]
    via = f" via edge {operand['via_edge']}" if operand["via_edge"] else ""
    return [
        f"{pad}{node['metric']} = {node['op']}({operand['concept']}.{operand['attribute']})"
        f"{via}   [{operand['table']}.{operand['column']}]"
    ]

def _schema(properties: Mapping[str, Any], required: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }

SKILL_SPECS: tuple[SkillSpec, ...] = (
    SkillSpec(
        name="find_paths",
        description=(
            "Return EVERY governed route between two concepts, composed on demand from the "
            "ontology's typed edges — including routes nobody pre-authored. Call this before "
            "writing any query that touches more than one concept.\n\n"
            "Each path gives you, per step: the declared edge, the direction it is traversed, "
            "the role it commits to, the declared fan-out (whether that step DUPLICATES rows), "
            "and the exact physical join keys — junction tables, subset restriction tables and "
            "role predicates included — as ready-to-paste JOIN clauses in traversal order.\n\n"
            "Every clause names the edge that authorised it (`authorised_by`), so a join you "
            "write can always be traced to a declared edge. A join between two concepts with "
            "NO path returned is not authorised: do not write it.\n\n"
            "Read `role_signature` and `subject_predicates`: if a concept is a role object, its "
            "predicate is part of its identity and MUST appear in your WHERE clause. Read "
            "`max_fan_out`: when it is not `none`, rows are duplicated along the path and a "
            "plain SUM or COUNT over it silently returns an inflated number.\n\n"
            "Each step also carries `measured_fan_out` — the same question asked of the DATA: "
            "the largest and mean number of rows one origin row actually reaches, and how many "
            "origin rows reach none (so you know whether an inner join silently drops them). "
            "Where the measurement contradicts the declaration it is listed in "
            "`declaration_contradicted_by_data`; trust the measurement and aggregate defensively."
        ),
        input_schema=_schema(
            {
                "from_concept": {"type": "string",
                                 "description": "Concept the query starts from (its subject)."},
                "to_concept": {"type": "string",
                               "description": "Concept the query needs to reach."},
                "max_hops": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_HOPS,
                             "description": "Longest route to consider. Raise it if no path "
                                            "is returned."},
            },
            ["from_concept", "to_concept"],
        ),
    ),
    SkillSpec(
        name="describe_concept",
        description=(
            "Everything needed to write the SELECT and WHERE clauses for ONE concept: its "
            "physical table, its key columns, every attribute with its physical column and "
            "type, which attribute is the title (what a bare mention of the concept means), "
            "which attributes may be searched for values, and its one-hop neighbours.\n\n"
            "If `is_role_object` is true the concept is a FILTERED SUBSET of its table and "
            "`role_predicates` MUST be applied — selecting the concept IS the role commitment. "
            "Attributes with `resolved_via` do not live on this table; reach them through the "
            "named edge (use find_paths).\n\n"
            "It also reports what only the DATA can say: how many rows the concept has in this "
            "snapshot (a concept with none makes every query over it vacuous), and per attribute "
            "the STORAGE type, null and blank counts and distinct-value count. Read `hazards`: a "
            "column declared numeric but stored as INTEGER makes a ratio truncate to 0, and a "
            "date stored as a number will not compare against an ISO string. "
            "`ungroundable_attributes` lists attributes no literal can be grounded against."
        ),
        input_schema=_schema(
            {"name": {"type": "string", "description": "Exact concept name from list_concepts."}},
            ["name"],
        ),
    ),
    SkillSpec(
        name="list_concepts",
        description=(
            "The entry points: every concept with its physical table, attribute names, whether "
            "it is a role object, and how many neighbours it has — plus the names of every "
            "governed edge, metric, link and party role. Call this first when you do not yet "
            "know the vocabulary. It is a map, not the territory: use describe_concept and "
            "find_paths for detail."
        ),
        input_schema=_schema({}, []),
    ),
    SkillSpec(
        name="search_concepts",
        description=(
            "Map a phrase from the question onto ontology symbols — concepts, attributes, "
            "edges, governed metrics and business links — ranked, with the field that matched "
            "and the tool call to make next. Use it to turn the words of a question into names "
            "the other tools accept. Recall-oriented: it may return more than one reading, and "
            "choosing between them is your job."
        ),
        input_schema=_schema(
            {
                "phrase": {"type": "string", "description": "Words taken from the question."},
                "limit": {"type": "integer", "minimum": 1, "default": 10,
                          "description": "Maximum results to return."},
            },
            ["phrase"],
        ),
    ),
    SkillSpec(
        name="find_metric",
        description=(
            "Resolve a business term to the ONE authored definition of it, fully expanded: the "
            "aggregation, the concept, attribute and physical column it is computed over, the "
            "edge each operand must be reached BY, and — for a composite — every component "
            "metric, recursively.\n\n"
            "The registry is CLOSED and is returned with every answer. If a term is in it, "
            "compute it exactly as defined: an improvised sum over the same column reached by a "
            "different edge is a DIFFERENT quantity and will be rejected by check_sql. If the "
            "term is not in the registry, say so and aggregate an attribute directly instead of "
            "inventing a variant. When a phrase names a business relationship rather than a "
            "quantity, that is reported under `links`."
        ),
        input_schema=_schema(
            {"phrase": {"type": "string",
                        "description": "The business term as the question words it."}},
            ["phrase"],
        ),
    ),
    SkillSpec(
        name="search_values",
        description=(
            "Resolve a phrase to the CANONICAL value the database actually holds, before you "
            "put it in a WHERE clause. This is how 'deputy' becomes 'Deputy' instead of "
            "silently matching zero rows. Never invent a string literal for a filter — probe "
            "it.\n\n"
            "The ladder is exact -> case-insensitive -> prefix -> contains and stops at the "
            "first rung that matches, so an exact hit never drags in everything it is a "
            "substring of. `status` decides what you may do: `unique` — use `value` verbatim "
            "and `filter_sql` as written; `ambiguous` — several values match, ask rather than "
            "guess; `absent` — nothing matches, so any predicate you build returns no rows; "
            "`unsearchable` / `unsearchable_via` — the ontology does not authorise searching "
            "that attribute, and the answer names the attributes that would serve.\n\n"
            "Only attributes the ontology declares searchable can be resolved. An attribute "
            "that resolves through an edge to a searchable one IS followed, and the answer "
            "records that it was (`inherited`) — the value then lives on the far concept, so "
            "write the predicate there and reach it with find_paths."
        ),
        input_schema=_schema(
            {
                "concept": {"type": "string", "description": "Concept owning the attribute."},
                "attribute": {"type": "string",
                              "description": "Attribute to resolve the phrase against."},
                "phrase": {"type": "string", "description": "The value as the question spells it."},
                "follow_via": {"type": "boolean", "default": True,
                               "description": "Resolve through a `via` attribute to the far "
                                              "attribute it inherits searchability from. Set "
                                              "false to refuse instead."},
            },
            ["concept", "attribute", "phrase"],
        ),
    ),
    SkillSpec(
        name="check_sql",
        description=(
            "Judge SQL you have written against the ontology BEFORE running it. Returns the "
            "violations, each with a stable machine code, one precise human sentence, and the "
            "offending SQL fragment — so you can repair the specific defect rather than "
            "rewriting blindly.\n\n"
            "Codes: UNDECLARED_JOIN (a join no declared edge sanctions), UNCOMMITTED_ROLE (a "
            "traversal that could be any of several roles and commits to none), GRAIN_FANOUT / "
            "ROW_FANOUT (an aggregate over rows a join duplicated — a plausible wrong NUMBER, "
            "the failure nobody notices), GRAIN_MISMATCH (an aggregate whose operand was "
            "pre-aggregated at a grain that is not the answer's — nothing was duplicated and "
            "the number is still wrong), METRIC_FIDELITY (a governed metric computed as an "
            "improvised variant), UNGROUNDED_LITERAL (a literal matching no row), plus "
            "CARTESIAN, UNKNOWN_TABLE and PARSE_ERROR.\n\n"
            "An empty list means certified. A `severity` of `undecidable` is NOT a pass: the "
            "shape could not be analysed, so nothing about it is guaranteed. Pass "
            "`expect_metrics` with any governed metric names the question resolved to, so "
            "metric fidelity is checked against the question rather than only against aliases."
        ),
        input_schema=_schema(
            {
                "sql": {"type": "string", "description": "The SELECT statement to govern."},
                "expect_metrics": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Governed metric names the question resolved to "
                                   "(from find_metric).",
                },
            },
            ["sql"],
        ),
    ),
)

SKILL_NAMES: tuple[str, ...] = tuple(spec.name for spec in SKILL_SPECS)

_PICK_TAIL = (
    "\n\nYou will NOT write SQL. When you have retrieved enough, emit ONE JSON pick "
    "naming: the subject concept, the measures, the dimensions, the filters, and for "
    "each of them the `route` id that reaches it (or `SELF` for the subject's own "
    "columns). The SQL is compiled from that pick."
)

PICK_SKILL_SPECS: tuple[SkillSpec, ...] = (
    SkillSpec(
        name="find_paths",
        description=(

            "THE ONE YOU CALL MOST. Return every governed route between two concepts, "
            "composed on demand from the ontology's typed edges — including routes nobody "
            "pre-authored, and routes of three or four hops.\n\n"
            "Each route comes back with a STABLE ID written `<from>><to>#<n>` — the two "
            "concept names, then the route's number among the routes between them. That id "
            "is the only way to name a join in your pick, it means the same route every "
            "time, and a pick may mix ids retrieved from different calls. Call this once "
            "per (subject, target) pair your question needs, then reference the ids.\n\n"
            "Per route you get: the hop count; the `role_signature` it commits to; whether "
            "it duplicates rows (`fans-out`); and the ordered list of declared edges that "
            "authorise it. READ THE ROLE SIGNATURE. Two concepts are often joinable by "
            "several different relationships — who signed an agreement, who sold it, who "
            "underwrote it can all be the same pair of tables — and the role signature is "
            "the only thing that separates them. Two routes with the same endpoints and "
            "different roles answer DIFFERENT questions; pick the one the question means.\n\n"
            "If a pair of concepts returns NO route, no join between them is authorised — "
            "raise max_hops once, and otherwise re-read the question, because that reading "
            "is not expressible. Fan-out is not a reason to avoid a route: the compiler "
            "pre-aggregates across it, so choose the route that means what the question "
            "asks and let grain be handled for you."
            + _PICK_TAIL
        ),
        input_schema=_schema(
            {
                "from_concept": {"type": "string",
                                 "description": "The subject concept your pick starts from."},
                "to_concept": {"type": "string",
                               "description": "The concept the question needs to reach."},
                "max_hops": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_HOPS,
                             "description": "Longest route to consider. The default of 4 "
                                            "reaches everything in this ontology; raise it "
                                            "only if nothing came back."},
            },
            ["from_concept", "to_concept"],
        ),
    ),
    SkillSpec(
        name="describe_concept",
        description=(
            "One concept's vocabulary: its attribute names, which is its key, which is its "
            "title (what a bare mention of the concept means), and whether it is a ROLE "
            "OBJECT.\n\n"
            "Call it to learn the exact attribute names for your pick's dimensions and "
            "filters — the pick is validated against them, and a near-miss is rejected.\n\n"
            "`is_role_object` is the one that changes an answer: such a concept is a "
            "filtered subset of a shared table, and choosing it as your subject IS the role "
            "commitment — the predicate is applied for you and must not be restated as a "
            "filter. `neighbours` shows what the concept reaches in one hop, which tells you "
            "which find_paths call to make next. Attributes marked `resolved_via` live on "
            "another concept and are reached through a declared edge; you may still name "
            "them."
            + _PICK_TAIL
        ),
        input_schema=_schema(
            {"name": {"type": "string", "description": "Exact concept name from "
                                                       "list_concepts or search_concepts."}},
            ["name"],
        ),
    ),
    SkillSpec(
        name="list_subjects",
        description=(
            "START HERE. The subjects you may anchor a question on, one short line each, "
            "and the names of the governed measures. Pass `question` and the list is "
            "ranked against it.\n\n"
            "One of these subjects is your pick's subject, and the subject fixes the GRAIN "
            "of the answer — the single choice that changes what a number means.\n\n"
            "It deliberately does NOT return tables, attributes, edges or neighbour counts. "
            "Those are fetched per subject, by describe_concept and find_paths, for the few "
            "the question actually reaches. If a phrase in the question matches nothing "
            "here, search_concepts resolves it."
        ),
        input_schema=_schema(
            {"question": {"type": "string",
                          "description": "The question being answered. Ranks the list; "
                                         "required once an ontology exceeds the cap."}},
            [],
        ),
    ),
    SkillSpec(
        name="search_concepts",
        description=(
            "Turn words from the question into the exact ontology names the other tools "
            "accept, without dumping the whole ontology. Ranked across concepts, "
            "attributes, edges, governed metrics and business links, each hit carrying the "
            "field that matched and the tool call to make next.\n\n"
            "Use it when a phrase in the question does not obviously correspond to anything "
            "list_subjects returned. It is recall-oriented: more than one reading may come "
            "back, and choosing between them is your job."
        ),
        input_schema=_schema(
            {
                "phrase": {"type": "string", "description": "Words taken from the question."},
                "limit": {"type": "integer", "minimum": 1, "default": 10,
                          "description": "Maximum results to return."},
            },
            ["phrase"],
        ),
    ),
    SkillSpec(
        name="find_metric",
        description=(
            "Resolve a business term the question uses — a named quantity, a ratio, a count "
            "— to the ONE authored definition of it, fully expanded: the aggregation, the "
            "concept and attribute it is computed over, the declared edge each operand must "
            "be reached by, and for a composite every component metric recursively.\n\n"
            "Name the metric in your pick and it is computed exactly as authored; the "
            "definition is shown so your choice is informed, NOT so you can restate it. The "
            "registry is closed and comes back with every answer. If the question needs a "
            "quantity that is not in it, do not invent a variant — put a plain aggregation "
            "in the pick instead (`aggregation` over a route and attribute).\n\n"
            "PASS `subject` — the concept your pick is anchored on — and the answer names "
            "the exact route ids that may carry this metric from that subject. A metric is "
            "defined over a PARTICULAR declared edge, and the same column reached by a "
            "different edge is a different quantity: a route that lands on the right concept "
            "by the wrong edge is refused. This is the one pairing you cannot guess.\n\n"
            "Read `status`. `unique` — the term is governed, use it. `ambiguous` — several "
            "definitions fit equally, so ask rather than choose. `partial` — the registry "
            "covers only PART of what you asked (`unmatched_words` says which part is "
            "missing, typically a qualifier like an average or a rate); the definition shown "
            "is exact for what it covers and the rest is yours to express. `absent` — not "
            "governed at all."
            + _PICK_TAIL
        ),
        input_schema=_schema(
            {
                "phrase": {"type": "string",
                           "description": "The business term as the question words it."},
                "subject": {"type": "string",
                            "description": "The concept your pick is anchored on. Pass it to "
                                           "be told which route ids may carry this metric "
                                           "from there."},
            },
            ["phrase"],
        ),
    ),
    SkillSpec(
        name="search_values",
        description=(
            "Resolve a phrase to the CANONICAL value the database actually holds, before "
            "you put it in a filter. This is how 'deputy' becomes 'Deputy' instead of "
            "silently matching zero rows, and it is the ONLY free text in a pick — every "
            "other field is an id. Never invent a string literal; probe it.\n\n"
            "The ladder is exact -> case-insensitive -> prefix -> contains and stops at the "
            "first rung that matches, so an exact hit never drags in everything it is a "
            "substring of. `status` decides what you may do: `unique` — copy `value` into "
            "the filter verbatim; `ambiguous` — several values match, so this is a question "
            "to ask rather than a value to guess; `absent` — nothing matches, so any filter "
            "built from the phrase returns no rows; `unsearchable` — the ontology does not "
            "authorise searching that attribute, and the answer names the ones that would "
            "serve.\n\n"
            "You do not need this for numbers, dates or identifiers the question states "
            "literally — only for names and codes whose spelling in the data is unknown."
            + _PICK_TAIL
        ),
        input_schema=_schema(
            {
                "concept": {"type": "string", "description": "Concept owning the attribute."},
                "attribute": {"type": "string",
                              "description": "Attribute to resolve the phrase against."},
                "phrase": {"type": "string", "description": "The value as the question spells it."},
            },
            ["concept", "attribute", "phrase"],
        ),
    ),
)

PICK_SKILL_NAMES: tuple[str, ...] = tuple(spec.name for spec in PICK_SKILL_SPECS)

_PICK_BOUND_ARGUMENTS: Mapping[str, Mapping[str, Any]] = {
    "find_paths": {"detail": "routes"},
    "describe_concept": {"detail": "declared"},
}

def registry(arm: str = "pick") -> tuple[SkillSpec, ...]:
    """The advertised tool set for one arm. `spc/mcp_server.py`'s only choice."""
    if arm == "pick":
        return PICK_SKILL_SPECS
    if arm == "sql":
        return SKILL_SPECS
    raise ValueError(f"unknown arm {arm!r}; known: 'pick', 'sql'")

_DEFAULT: Skills | None = None

def default_skills() -> Skills:
    """The process-wide instance. Built once; the graph memo is what makes
    repeated `find_paths` calls a dict lookup."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Skills()
    return _DEFAULT

def _bind(skills: Skills | None) -> Skills:
    return skills if skills is not None else default_skills()

def call(name: str, arguments: Mapping[str, Any] | None = None,
         *, skills: Skills | None = None, arm: str = "sql") -> SkillResult:
    """Invoke a tool by name with a mapping of arguments.

    One entry point, so an in-process caller and an MCP client run exactly the
    same code. Unknown names and bad argument shapes come back as `ok=False`
    results, never exceptions — a tool call that raises is a transport failure a
    model cannot repair.

    `arm` selects the registry a refusal is explained against, and applies that
    arm's bound arguments (`_PICK_BOUND_ARGUMENTS`). It never changes which
    Python function runs.
    """
    bound = _bind(skills)
    handlers: Mapping[str, Callable[..., SkillResult]] = {
        "find_paths": bound.find_paths,
        "describe_concept": bound.describe_concept,
        "list_concepts": bound.list_concepts,
        "list_subjects": bound.list_subjects,
        "search_concepts": bound.search_concepts,
        "find_metric": bound.find_metric,
        "search_values": bound.search_values,
        "check_sql": bound.check_sql,
    }
    specs = registry(arm)
    known = tuple(spec.name for spec in specs)
    handler = handlers.get(name) if name in known else None
    if handler is None:
        return SkillResult(
            name=str(name), ok=False,
            data={"code": "UNKNOWN_SKILL", "error": f"no skill named {name!r}",
                  "known_skills": list(known)},
            text=f"UNKNOWN_SKILL: no skill named {name!r}. Known: {', '.join(known)}",
        )
    kwargs = {str(k): v for k, v in (arguments or {}).items()}
    if arm == "pick":
        kwargs.update(_PICK_BOUND_ARGUMENTS.get(name, {}))
    try:
        return handler(**kwargs)
    except TypeError as exc:
        return SkillResult(
            name=str(name), ok=False,
            data={"code": "BAD_ARGUMENTS", "error": str(exc),
                  "expected": dict(next(s.input_schema for s in specs if s.name == name))},
            text=f"BAD_ARGUMENTS calling {name}: {exc}",
        )

def call_pick(name: str, arguments: Mapping[str, Any] | None = None,
              *, skills: Skills | None = None) -> SkillResult:
    """`call` for the PICK arm: six tools, `find_paths` in route form.

    The arm's whole dispatch surface, so the harness and an MCP client bind one
    name and cannot drift apart.
    """
    return call(name, arguments, skills=skills, arm="pick")

def find_paths(from_concept: str, to_concept: str,
               max_hops: int = DEFAULT_MAX_HOPS) -> SkillResult:
    return default_skills().find_paths(from_concept, to_concept, max_hops)

def describe_concept(name: str) -> SkillResult:
    return default_skills().describe_concept(name)

def list_concepts() -> SkillResult:
    return default_skills().list_concepts()

def list_subjects(question: str | None = None) -> SkillResult:
    return default_skills().list_subjects(question)

def search_concepts(phrase: str, limit: int = 10) -> SkillResult:
    return default_skills().search_concepts(phrase, limit)

def find_metric(phrase: str) -> SkillResult:
    return default_skills().find_metric(phrase)

def find_attribute(phrase: str, concept: str | None = None) -> SkillResult:
    return default_skills().find_attribute(phrase, concept)

def search_values(concept: str, attribute: str, phrase: str,
                  follow_via: bool = True) -> SkillResult:
    return default_skills().search_values(concept, attribute, phrase, follow_via)

def check_sql(sql: str, expect_metrics: Sequence[str] | None = None) -> SkillResult:
    return default_skills().check_sql(sql, expect_metrics)

def _main() -> None:  # pragma: no cover
    import sys
    import time

    args = sys.argv[1:]
    if not args:
        print("usage: python -m spc.skills <skill> [json-arguments]")
        for spec in SKILL_SPECS:
            print(f"\n{spec.name}\n  {' '.join(spec.description.split())[:160]}...")
        return
    arguments = json.loads(args[1]) if len(args) > 1 else {}
    start = time.perf_counter()
    result = call(args[0], arguments)
    elapsed = (time.perf_counter() - start) * 1000
    print(result.text)
    print(f"\n[{result.name} ok={result.ok} {elapsed:.3f} ms]")

if __name__ == "__main__":  # pragma: no cover
    _main()
