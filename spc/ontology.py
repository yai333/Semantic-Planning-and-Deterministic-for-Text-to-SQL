from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path as FsPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

__all__ = [
    "OntologyError",
    "Attribute",
    "Concept",
    "Edge",
    "PartyRole",
    "ValueType",
    "MetricOperand",
    "Metric",
    "BusinessLink",
    "GlossaryEntry",
    "Ontology",
    "load_ontology",
    "ontology_from_mapping",
    "join_layers",
    "read_ddl_schema",
    "metadata_report",
    "DEFAULT_ONTOLOGY",
    "DEFAULT_SEMANTIC",
    "DEFAULT_MAPPING",
    "DEFAULT_DDL",
]

class OntologyError(ValueError):
    """An inconsistent ontology. Raised at load; never downgraded to a warning."""

CARDINALITIES = frozenset({"one_to_one", "one_to_many", "many_to_one", "many_to_many"})
FAN_OUTS = frozenset({"none", "bounded", "multiplicative"})
AGGREGATIONS = frozenset({"sum", "count", "count_distinct", "avg", "min", "max"})
COMBINES = frozenset({"add", "subtract", "multiply", "divide"})

ATTRIBUTE_TYPES = frozenset(
    {"string", "numeric", "integer", "date", "datetime", "boolean"}
)
NUMERIC_TYPES = frozenset({"numeric", "integer"})

_REPO_ROOT = FsPath(__file__).resolve().parent.parent

DEFAULT_SEMANTIC = _REPO_ROOT / "ontology" / "acme.semantic.yaml"
DEFAULT_ONTOLOGY = DEFAULT_SEMANTIC
DEFAULT_MAPPING = _REPO_ROOT / "ontology" / "acme.mapping.yaml"

DEFAULT_DDL = _REPO_ROOT / "database" / "acme" / "ACME_small.ddl"

_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

def _read_yaml(path: FsPath) -> Any:
    return yaml.load(path.read_text(), Loader=_LOADER)

def _pairs(mapping: Any, where: str) -> tuple[tuple[str, str], ...]:
    """A predicate/join mapping as ordered, hashable pairs."""
    if mapping is None:
        return ()
    if not isinstance(mapping, Mapping):
        raise OntologyError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    return tuple((str(k), str(v)) for k, v in mapping.items())

def _str_tuple(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value)
    raise OntologyError(f"{where}: expected a string or list of strings")

def predicate_code(pairs: Sequence[tuple[str, str]]) -> str | None:
    """Canonical short form of a role predicate.

    One column -> the bare value (`PH`), which is what the party-role vocabulary
    calls it. More than one -> `Col=Val` pairs, column-sorted, so the code is a
    function of the predicate and not of YAML key order.
    """
    if not pairs:
        return None
    if len(pairs) == 1:
        return str(pairs[0][1])
    return ",".join(f"{c}={v}" for c, v in sorted(pairs))

@dataclass(frozen=True)
class Attribute:
    """A concept property.

    Either `column`-backed (physical) or `via`-backed (resolved through an edge,
    Foundry's interface property resolution).

    `label`, `description`, `aliases` and `searchable` are the ADDITIONS. They
    are optional so an ontology that declares none still loads, and
    `metadata_report()` reports what their absence costs.

    `label` + `aliases` are the attribute half of the glossary: the phrases a
    business user says for this property, which a phrase resolver scores a
    question against. `label` is the PRIMARY one -- the display name -- and by
    convention it is the shortest phrase unique across the ontology, so that
    resolving it returns this attribute and no other. `aliases` are further
    accepted phrasings with no ordering commitment among them.
    """

    name: str
    concept: str
    column: str | None = None
    via: str | None = None
    via_attribute: str | None = None
    type: str | None = None
    value_type: str | None = None

    label: str | None = None
    description: str | None = None
    aliases: tuple[str, ...] = ()
    searchable: bool = False

    @property
    def is_resolved(self) -> bool:
        """True when the value lives on another concept, reached through `via`."""
        return self.via is not None

    @property
    def qualified(self) -> str:
        return f"{self.concept}.{self.name}"

    @property
    def labels(self) -> tuple[str, ...]:
        """Every natural-language name for this attribute, PRIMARY FIRST.

        Deduplicated verbatim and order-preserving, so a resolver's tie-break on
        position is a function of the authored file and not of set iteration.
        Empty when the ontology declares neither a label nor an alias -- which
        `metadata_report()` reports as `unlabelled` and a lint may refuse.
        """
        out: list[str] = []
        for text in ((self.label,) if self.label else ()) + self.aliases:
            if text and text not in out:
                out.append(text)
        return tuple(out)

@dataclass(frozen=True, eq=False)
class Concept:
    """A business entity mapped to one physical table.

    A *role object* (`backed_where` non-empty) is a filtered subset of its table
    whose predicate is part of the concept's identity: selecting the concept IS
    the role commitment, so there is no role declaration to omit.
    """

    name: str
    table: str
    key: tuple[str, ...] = ()
    display: str | None = None
    description: str | None = None
    grain: str | None = None
    implements: str | None = None
    backed_where: tuple[tuple[str, str], ...] = ()
    attributes: Mapping[str, Attribute] = field(default_factory=dict)

    searchable_declared: tuple[str, ...] = ()

    @property
    def is_role_object(self) -> bool:
        return bool(self.backed_where)

    @property
    def role_code(self) -> str | None:
        """The role this concept commits to, or None. Derived, never supplied."""
        return predicate_code(self.backed_where)

    @property
    def title_attribute(self) -> Attribute | None:
        """The declared display key -- the attribute a bare projection means."""
        return self.attributes.get(self.display) if self.display else None

    @property
    def searchable_attributes(self) -> tuple[Attribute, ...]:
        return tuple(a for a in self.attributes.values() if a.searchable)

    def attribute(self, name: str) -> Attribute:
        try:
            return self.attributes[name]
        except KeyError:
            raise OntologyError(f"concept {self.name!r} has no attribute {name!r}") from None

@dataclass(frozen=True)
class Edge:
    """A governed relationship. Traversable in either direction.

    Three physical backings, in the ontology's own vocabulary:
      `join`      direct FK           {from_column: to_column}
      `via_*`     junction table      from_join / to_join around `via_table`
      `restrict`  subset membership   an extra semijoin (`HAS_LOSS_PAYMENT`)

    `role_predicate` makes several role-typed edges over one junction distinct
    relationships, which is why "party related to policy" is not expressible.
    """

    name: str
    source: str
    target: str
    cardinality: str
    fan_out: str = "bounded"
    fan_out_reverse: str = "bounded"
    join: tuple[tuple[str, str], ...] = ()
    via_table: str | None = None
    via_from_join: tuple[tuple[str, str], ...] = ()
    via_to_join: tuple[tuple[str, str], ...] = ()
    restrict_table: str | None = None
    restrict_columns: tuple[tuple[str, str], ...] = ()
    role_predicate: tuple[tuple[str, str], ...] = ()
    description: str | None = None
    evidence: str | None = None

    @property
    def is_junction(self) -> bool:
        return self.via_table is not None

    @property
    def is_restricted(self) -> bool:
        return self.restrict_table is not None

    @property
    def role_code(self) -> str | None:
        return predicate_code(self.role_predicate)

    def endpoint(self, *, forward: bool) -> str:
        """The concept a traversal LANDS on when read in this direction."""
        return self.target if forward else self.source

    def origin(self, *, forward: bool) -> str:
        return self.source if forward else self.target

    def fan_out_in(self, *, forward: bool) -> str:
        """Declared row multiplication when read in this direction."""
        return self.fan_out if forward else self.fan_out_reverse

@dataclass(frozen=True)
class PartyRole:
    code: str
    name: str
    description: str | None = None

@dataclass(frozen=True)
class ValueType:
    name: str
    type: str
    format: str | None = None
    units: str | None = None

@dataclass(frozen=True)
class MetricOperand:
    concept: str
    attribute: str
    via: str | None = None

@dataclass(frozen=True)
class Metric:
    """A governed metric. Either a raw aggregation or a composition -- never both."""

    name: str
    op: str | None = None
    operand: MetricOperand | None = None
    combine: str | None = None
    components: tuple[str, ...] = ()

    per: str | None = None
    description: str | None = None

    @property
    def is_composite(self) -> bool:
        return self.combine is not None

    @property
    def has_row_identity(self) -> bool:
        """Whether a non-distributing aggregate over this metric is well posed.

        `sum` is safe on any composition because it distributes; `avg`, `min`
        and `max` are not, and asking for one without this is asking for a
        number that depends on which satellite happened to hold which rows."""
        return self.per is not None

@dataclass(frozen=True)
class BusinessLink:
    """A business relationship backed by one or more join-graph edges."""

    name: str
    predicate: str
    source: str
    target: str
    backed_by: tuple[str, ...] = ()
    description: str | None = None

@dataclass(frozen=True)
class GlossaryEntry:
    phrases: tuple[str, ...]
    kind: str
    target: str

@dataclass(frozen=True, eq=False)
class Ontology:
    """Everything the enumerator is allowed to know."""

    version: int
    domain: str
    concepts: Mapping[str, Concept]
    edges: tuple[Edge, ...]
    party_roles: Mapping[str, PartyRole] = field(default_factory=dict)
    value_types: Mapping[str, ValueType] = field(default_factory=dict)
    metrics: Mapping[str, Metric] = field(default_factory=dict)
    links: Mapping[str, BusinessLink] = field(default_factory=dict)
    glossary: tuple[GlossaryEntry, ...] = ()
    sources: tuple[str, ...] = ()

    raw: Mapping[str, Any] = field(default_factory=dict)

    mapping: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """A shallow copy of the merged document. See `raw`."""
        return dict(self.raw)

    def mapping_entry(self, kind: str, name: str) -> Mapping[str, Any] | None:
        """The mapping-layer entry that binds `name` to physical storage.

        `kind` is "concepts" or "edges". Returns None when the ontology carries
        no mapping layer. This is the provenance hook the split buys: a compiled
        query can cite the mapping entry a table or column came from without
        citing the concept that gave it meaning.
        """
        section = self.mapping.get(kind) or {}
        entry = section.get(name)
        return None if entry is None else MappingProxyType(dict(entry))

    @property
    def doc(self) -> Mapping[str, Any]:
        return self.raw

    def concept(self, name: str) -> Concept:
        try:
            return self.concepts[name]
        except KeyError:
            raise OntologyError(f"unknown concept {name!r}") from None

    def edge(self, name: str) -> Edge:
        for e in self.edges:
            if e.name == name:
                return e
        raise OntologyError(f"unknown edge {name!r}")

    @property
    def edges_by_name(self) -> Mapping[str, Edge]:
        return MappingProxyType({e.name: e for e in self.edges})

    def outgoing(self, concept: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.source == concept)

    def incoming(self, concept: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.target == concept)

    def incident(self, concept: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if concept in (e.source, e.target))

    @property
    def role_objects(self) -> tuple[Concept, ...]:
        return tuple(c for c in self.concepts.values() if c.is_role_object)

    def concept_names(self) -> tuple[str, ...]:
        """Sorted -- callers that iterate concepts must not inherit YAML order."""
        return tuple(sorted(self.concepts))

    def metric(self, name: str) -> Metric:
        try:
            return self.metrics[name]
        except KeyError:
            raise OntologyError(f"unknown metric {name!r}") from None

    def summands(self, name: str) -> tuple[str, ...]:
        """Every metric ADDED INTO `name`, transitively. Ordered, distinct.

        ONLY `add` EDGES ARE CROSSED, and that is the whole precision of it. A
        summand is contained in its total, so projecting both states one number
        twice; a DIVISOR is not, and a ratio beside its own denominator is a
        normal thing to want. `LossRatio` therefore returns the four amounts
        nested inside its numerator and never `PremiumAmount`.

        HERE, not in the skill step that first needed it, for the reason
        `measured_over` gives above: the planner is ADVISED of this by
        `review_plan` and the engine ENFORCES it in `_certify`, and an advice
        and an enforcement computed by two functions are two chances to
        disagree. A plan that projects a total beside its parts compiles
        cleanly and certifies -- nothing downstream can catch it -- so the
        enforcement has to be exact.
        """
        seen: list[str] = []

        def walk(metric_name: str, crossing_add: bool, guard: frozenset[str]) -> None:

            if metric_name in guard:
                return
            entry = self.metrics.get(metric_name)
            if entry is None or not entry.is_composite:
                return
            adds = entry.combine == "add"
            for component in entry.components:
                if (adds or crossing_add) and component not in seen:
                    if adds:
                        seen.append(component)
                walk(component, crossing_add or adds, guard | {metric_name})

        walk(name, False, frozenset())
        return tuple(seen)

    def measured_over(self, name: str) -> tuple[str, ...]:
        """The concept(s) a governed metric is a quantity OF. Ordered, distinct.

        A leaf metric names its last edge -- `via`, which for these amount
        metrics IS the kind -- so the concept the metric is a quantity of is the
        one that edge LEAVES: `LossPayment` is a quantity of a Claim, whose
        value happens to live on a ClaimAmount row. Without a `via` the operand
        concept is itself that concept.

        A composite is a quantity of whatever its components are quantities of,
        and the answer is a TUPLE because they need not agree. `TotalLoss`
        returns `('Claim',)` -- all four kinds leave Claim -- so "total loss OF
        an X" names one concept and a route from X to it is meaningful.
        `LossRatio` returns `('Claim', 'PolicyCoverageDetail')`: there is no
        single place a route could reach, which is the structural form of the
        registry's own note that its denominator follows the plan's SUBJECT.

        The compiler reads this to decide what `Measure.path` means for a
        governed metric, and `spc/skills.py` reads it to say which routes a
        model may pair with the metric. One definition, so the advice and the
        enforcement cannot drift.
        """
        metric = self.metric(name)
        if not metric.is_composite:
            operand = metric.operand
            if operand is None:                              # pragma: no cover
                raise OntologyError(f"metric {name!r} has neither operand nor components")
            if operand.via is None:
                return (operand.concept,)
            edge = self.edge(operand.via)
            if edge.target == operand.concept and edge.source != operand.concept:
                return (edge.source,)
            if edge.source == operand.concept and edge.target != operand.concept:
                return (edge.target,)
            raise OntologyError(                             # pragma: no cover
                f"metric {name!r}: edge {operand.via!r} does not determine one concept "
                f"that {operand.concept!r} is measured over"
            )
        bases: list[str] = []
        for component in metric.components:
            for base in self.measured_over(component):
                if base not in bases:
                    bases.append(base)
        return tuple(bases)

def _load_attribute(concept: str, name: str, spec: Any, declared_searchable: set[str]) -> Attribute:
    where = f"{concept}.{name}"
    if not isinstance(spec, Mapping):
        raise OntologyError(f"attribute {where}: expected a mapping")
    unknown = set(spec) - {
        "column", "via", "attribute", "type", "value_type",
        "label", "description", "aliases", "searchable",
    }
    if unknown:
        raise OntologyError(f"attribute {where}: unknown keys {sorted(unknown)}")

    label = spec.get("label")
    if label is not None and (not isinstance(label, str) or not label.strip()):
        raise OntologyError(f"attribute {where}: `label` must be a non-empty string")

    column = spec.get("column")
    via = spec.get("via")
    if column is None and via is None:
        raise OntologyError(f"attribute {where}: needs either `column` or `via`")
    if column is not None and via is not None:
        raise OntologyError(f"attribute {where}: `column` and `via` are mutually exclusive")
    if via is not None and not spec.get("attribute"):
        raise OntologyError(f"attribute {where}: `via` requires `attribute`")

    searchable = spec.get("searchable")
    if searchable is None:
        searchable = name in declared_searchable
    elif not isinstance(searchable, bool):
        raise OntologyError(f"attribute {where}: `searchable` must be a boolean")

    declared_type = spec.get("type")
    if declared_type is not None and declared_type not in ATTRIBUTE_TYPES:
        raise OntologyError(
            f"attribute {where}: type {declared_type!r} not in {sorted(ATTRIBUTE_TYPES)}"
        )

    def text(key: str) -> str | None:
        value = spec.get(key)
        return None if value is None else str(value)

    return Attribute(
        name=name,
        concept=concept,
        column=None if column is None else str(column),
        via=None if via is None else str(via),
        via_attribute=text("attribute"),
        type=declared_type,
        value_type=text("value_type"),
        label=text("label"),
        description=spec.get("description"),
        aliases=_str_tuple(spec.get("aliases"), f"attribute {where}.aliases"),
        searchable=bool(searchable),
    )

def _load_concept(name: str, spec: Any) -> Concept:
    if not isinstance(spec, Mapping):
        raise OntologyError(f"concept {name}: expected a mapping")
    unknown = set(spec) - {
        "table", "key", "display", "description", "grain", "implements",
        "backed_where", "attributes", "searchable",
    }
    if unknown:
        raise OntologyError(f"concept {name}: unknown keys {sorted(unknown)}")
    table = spec.get("table")
    if not table:
        raise OntologyError(f"concept {name}: missing `table`")

    declared = _str_tuple(spec.get("searchable"), f"concept {name}.searchable")
    attrs_spec = spec.get("attributes") or {}
    if not isinstance(attrs_spec, Mapping):
        raise OntologyError(f"concept {name}: `attributes` must be a mapping")
    attributes = {
        attr_name: _load_attribute(name, attr_name, attr_spec, set(declared))
        for attr_name, attr_spec in attrs_spec.items()
    }
    missing = [s for s in declared if s not in attributes]
    if missing:
        raise OntologyError(f"concept {name}: `searchable` names unknown attributes {missing}")

    return Concept(
        name=name,
        table=str(table),
        key=_str_tuple(spec.get("key"), f"concept {name}.key"),
        display=spec.get("display"),
        description=spec.get("description"),
        grain=spec.get("grain"),
        implements=spec.get("implements"),
        backed_where=_pairs(spec.get("backed_where"), f"concept {name}.backed_where"),
        attributes=MappingProxyType(attributes),
        searchable_declared=declared,
    )

def _load_edge(spec: Any, index: int) -> Edge:
    if not isinstance(spec, Mapping):
        raise OntologyError(f"edges[{index}]: expected a mapping")
    name = spec.get("name")
    if not name:
        raise OntologyError(f"edges[{index}]: missing `name`")
    unknown = set(spec) - {
        "name", "from", "to", "join", "via", "restrict", "role_predicate",
        "cardinality", "fan_out", "fan_out_reverse", "evidence", "description",
    }
    if unknown:
        raise OntologyError(f"edge {name}: unknown keys {sorted(unknown)}")
    source, target = spec.get("from"), spec.get("to")
    if not source or not target:
        raise OntologyError(f"edge {name}: needs both `from` and `to`")

    via = spec.get("via")
    via_table = via_from = via_to = None
    if via is not None:
        if not isinstance(via, Mapping) or "table" not in via:
            raise OntologyError(f"edge {name}: `via` needs a `table`")
        via_table = via["table"]
        via_from = _pairs(via.get("from_join"), f"edge {name}.via.from_join")
        via_to = _pairs(via.get("to_join"), f"edge {name}.via.to_join")
        if not via_from or not via_to:
            raise OntologyError(f"edge {name}: `via` needs both `from_join` and `to_join`")
    join = _pairs(spec.get("join"), f"edge {name}.join")
    if not join and via_table is None:
        raise OntologyError(f"edge {name}: needs either `join` or `via`")
    if join and via_table is not None:
        raise OntologyError(f"edge {name}: `join` and `via` are mutually exclusive")

    restrict = spec.get("restrict")
    restrict_table = restrict_columns = None
    if restrict is not None:
        if not isinstance(restrict, Mapping) or "table" not in restrict:
            raise OntologyError(f"edge {name}: `restrict` needs a `table`")
        restrict_table = restrict["table"]
        restrict_columns = _pairs(restrict.get("columns"), f"edge {name}.restrict.columns")

    cardinality = spec.get("cardinality")
    if cardinality not in CARDINALITIES:
        raise OntologyError(f"edge {name}: cardinality {cardinality!r} not in {sorted(CARDINALITIES)}")
    for key in ("fan_out", "fan_out_reverse"):
        value = spec.get(key, "bounded")
        if value not in FAN_OUTS:
            raise OntologyError(f"edge {name}: {key} {value!r} not in {sorted(FAN_OUTS)}")

    return Edge(
        name=str(name),
        source=str(source),
        target=str(target),
        cardinality=str(cardinality),
        fan_out=str(spec.get("fan_out", "bounded")),
        fan_out_reverse=str(spec.get("fan_out_reverse", "bounded")),
        join=join,
        via_table=via_table,
        via_from_join=via_from or (),
        via_to_join=via_to or (),
        restrict_table=restrict_table,
        restrict_columns=restrict_columns or (),
        role_predicate=_pairs(spec.get("role_predicate"), f"edge {name}.role_predicate"),
        description=spec.get("description"),
        evidence=spec.get("evidence"),
    )

def _load_metric(name: str, spec: Any) -> Metric:
    if not isinstance(spec, Mapping):
        raise OntologyError(f"metric {name}: expected a mapping")
    unknown = set(spec) - {"op", "operand", "combine", "components", "description",
                           "per"}
    if unknown:
        raise OntologyError(f"metric {name}: unknown keys {sorted(unknown)}")
    op, combine = spec.get("op"), spec.get("combine")
    if (op is None) == (combine is None):
        raise OntologyError(f"metric {name}: needs exactly one of `op` or `combine`")

    operand = None
    if op is not None:
        if op not in AGGREGATIONS:
            raise OntologyError(f"metric {name}: op {op!r} not in {sorted(AGGREGATIONS)}")
        raw = spec.get("operand")
        if not isinstance(raw, Mapping):
            raise OntologyError(f"metric {name}: `op` requires an `operand` mapping")
        if "concept" not in raw or "attribute" not in raw:
            raise OntologyError(f"metric {name}: operand needs `concept` and `attribute`")
        operand = MetricOperand(
            concept=str(raw["concept"]),
            attribute=str(raw["attribute"]),
            via=raw.get("via"),
        )
    else:
        if combine not in COMBINES:
            raise OntologyError(f"metric {name}: combine {combine!r} not in {sorted(COMBINES)}")
        if not spec.get("components"):
            raise OntologyError(f"metric {name}: `combine` requires `components`")
    if spec.get("per") is not None and combine is None:
        raise OntologyError(
            f"metric {name}: `per` names the grain a COMPOSITION is formed on; a "
            f"leaf metric is already formed on the concept its operand's edge leaves"
        )

    return Metric(
        name=name,
        op=op,
        operand=operand,
        combine=combine,
        components=_str_tuple(spec.get("components"), f"metric {name}.components"),
        per=None if spec.get("per") is None else str(spec["per"]),
        description=spec.get("description"),
    )

def _load_link(spec: Any, index: int) -> BusinessLink:
    if not isinstance(spec, Mapping):
        raise OntologyError(f"links[{index}]: expected a mapping")
    name = spec.get("name")
    if not name:
        raise OntologyError(f"links[{index}]: missing `name`")
    for key in ("predicate", "from", "to"):
        if not spec.get(key):
            raise OntologyError(f"link {name}: missing `{key}`")
    return BusinessLink(
        name=str(name),
        predicate=str(spec["predicate"]),
        source=str(spec["from"]),
        target=str(spec["to"]),
        backed_by=_str_tuple(spec.get("backed_by"), f"link {name}.backed_by"),
        description=spec.get("description"),
    )

def _load_glossary(raw: Any) -> tuple[GlossaryEntry, ...]:
    entries: list[GlossaryEntry] = []
    for index, spec in enumerate(raw or ()):
        if not isinstance(spec, Mapping):
            raise OntologyError(f"glossary[{index}]: expected a mapping")
        means = spec.get("means")
        if not isinstance(means, Mapping) or len(means) != 1:
            raise OntologyError(f"glossary[{index}]: `means` must name exactly one metric or link")
        kind, target = next(iter(means.items()))
        if kind not in ("metric", "link"):
            raise OntologyError(f"glossary[{index}]: `means` kind {kind!r} must be metric or link")
        phrases = _str_tuple(spec.get("phrases"), f"glossary[{index}].phrases")
        if not phrases:
            raise OntologyError(f"glossary[{index}]: needs `phrases`")
        entries.append(GlossaryEntry(phrases=phrases, kind=str(kind), target=str(target)))
    return tuple(entries)

def ontology_from_mapping(
    data: Mapping[str, Any],
    semantics: Mapping[str, Any] | None = None,
    *,
    sources: Sequence[str] = (),
) -> Ontology:
    """Build and VALIDATE an `Ontology` from already-parsed mappings."""
    if not isinstance(data, Mapping):
        raise OntologyError("ontology: expected a mapping at the document root")

    concepts_spec = data.get("concepts")
    if not concepts_spec:
        raise OntologyError("ontology: no `concepts`")
    concepts = {name: _load_concept(name, spec) for name, spec in concepts_spec.items()}

    edges_spec = data.get("edges") or ()
    edges = tuple(_load_edge(spec, i) for i, spec in enumerate(edges_spec))

    value_types = {}
    for name, spec in (data.get("value_types") or {}).items():
        if not isinstance(spec, Mapping) or "type" not in spec:
            raise OntologyError(f"value_type {name}: needs a `type`")
        value_types[name] = ValueType(
            name=name, type=str(spec["type"]),
            format=spec.get("format"), units=spec.get("units"),
        )

    party_roles = {}
    for code, spec in (data.get("party_roles") or {}).items():
        if not isinstance(spec, Mapping) or "name" not in spec:
            raise OntologyError(f"party_role {code}: needs a `name`")
        party_roles[str(code)] = PartyRole(
            code=str(code), name=str(spec["name"]), description=spec.get("description"),
        )

    semantics = semantics or {}
    metrics = {n: _load_metric(n, s) for n, s in (semantics.get("metrics") or {}).items()}
    links_seq = [_load_link(s, i) for i, s in enumerate(semantics.get("links") or ())]
    links: dict[str, BusinessLink] = {}
    for link in links_seq:
        if link.name in links:
            raise OntologyError(f"duplicate link name {link.name!r}")
        links[link.name] = link
    glossary = _load_glossary(semantics.get("glossary"))

    ontology = Ontology(
        version=int(data.get("version", 1)),
        domain=str(data.get("domain", "unknown")),
        concepts=MappingProxyType(concepts),
        edges=edges,
        party_roles=MappingProxyType(party_roles),
        value_types=MappingProxyType(value_types),
        metrics=MappingProxyType(metrics),
        links=MappingProxyType(links),
        glossary=glossary,
        sources=tuple(str(s) for s in sources),
        raw=MappingProxyType(_merged_document(data, semantics)),
    )
    _validate(ontology)
    return ontology

def _merged_document(
    data: Mapping[str, Any], semantics: Mapping[str, Any] | None
) -> dict[str, Any]:
    """The two YAML files as one mapping: the join graph, with the semantic
    layer's three sections laid over it. Matches what `spc/onto_shim.py`
    produced, so replacing the shim with this loader is a one-line change."""
    merged = dict(data)
    for key in ("metrics", "links", "glossary"):
        if semantics and key in semantics:
            merged[key] = semantics[key]
    return merged

_DDL_TABLE = "CREATE TABLE"

def read_ddl_schema(path: str | FsPath = DEFAULT_DDL) -> dict[str, dict[str, str]]:
    """`{table: {column: declared_type}}` from a DDL file, case preserved.

    Deliberately a 30-line reader and not a SQL parser: it recognises
    `CREATE TABLE name (`, then any line whose first character is a tab as
    `column type ...`, and stops at the closing paren. Constraint lines start
    with a space, not a tab, which is what separates them from columns.

    PARTIAL BY CONSTRUCTION. `ACME_small.ddl` is an extract: it declares 14 of
    the 18 tables this ontology maps (`Party`, `Insurable_Object`, `Occurrence`
    and `Geographic_Location` are referenced by FOREIGN KEY but never declared).
    A partial DDL cannot refute a column in a table it does not declare, so
    `join_layers` checks columns only for tables that ARE declared and reports
    the rest as unverified rather than pretending to a check it cannot make.
    """
    schema: dict[str, dict[str, str]] = {}
    table: str | None = None
    for line in FsPath(path).read_text().splitlines():
        stripped = line.strip()
        if stripped.upper().startswith(_DDL_TABLE):
            table = stripped[len(_DDL_TABLE):].strip().rstrip("(").strip()
            schema.setdefault(table, {})
            continue
        if table is None:
            continue
        if stripped.startswith(")"):
            table = None
            continue
        if not line.startswith("\t"):
            continue
        parts = stripped.split()
        if len(parts) >= 2 and parts[0].upper() not in ("PRIMARY", "FOREIGN", "UNIQUE"):
            schema[table][parts[0]] = parts[1]
    return schema

_SEMANTIC_TOP = {
    "version", "domain", "value_types", "concepts", "party_roles", "edges",
    "metrics", "links", "glossary",
}
_MAPPING_TOP = {"version", "warehouse", "source", "concepts", "edges"}
_SEMANTIC_CONCEPT = {
    "key", "display", "description", "grain", "implements", "role",
    "attributes", "searchable",
}
_MAPPING_CONCEPT = {"table", "key_columns", "columns", "predicate"}
_SEMANTIC_ATTRIBUTE = {
    "via", "attribute", "type", "value_type",
    "label", "description", "aliases", "searchable",
}
_SEMANTIC_EDGE = {
    "from", "to", "role", "cardinality", "fan_out", "fan_out_reverse",
    "evidence", "description",
}
_MAPPING_EDGE = {"join", "via_table", "from_join", "to_join", "restrict", "predicate"}

_CONCEPT_ORDER = (
    "table", "implements", "backed_where", "key", "display", "grain",
    "description", "attributes", "searchable",
)
_ATTRIBUTE_ORDER = (
    "column", "via", "attribute", "type", "value_type",
    "label", "description", "aliases", "searchable",
)
_EDGE_ORDER = (
    "name", "from", "to", "join", "via", "restrict", "role_predicate",
    "cardinality", "fan_out", "fan_out_reverse", "evidence", "description",
)

def _ordered(spec: Mapping[str, Any], order: Sequence[str]) -> dict[str, Any]:
    out = {k: spec[k] for k in order if k in spec}
    out.update({k: v for k, v in spec.items() if k not in out})
    return out

def _reject_keys(spec: Mapping[str, Any], allowed: set[str], where: str, layer: str) -> None:
    unknown = set(spec) - allowed
    if unknown:
        raise OntologyError(
            f"{layer} layer, {where}: keys {sorted(unknown)} do not belong here "
            f"(allowed: {sorted(allowed)})"
        )

def join_layers(
    semantic: Mapping[str, Any],
    mapping: Mapping[str, Any],
    *,
    schema: Mapping[str, Mapping[str, str]] | None = None,
    strict_tables: bool = False,
) -> dict[str, Any]:
    """Join the two layers into the single merged document the loader types.

    Eight validation rules, every one of them an `OntologyError` and never a
    warning -- a half-joined ontology compiles to SQL against the wrong column,
    which is the failure mode that has to be impossible:

      1. a concept declared in the semantic layer with NO mapping entry;
      2. a mapping entry naming NO semantic concept (likewise for edges);
      3. a column-backed attribute with no column in the mapping -- i.e. any
         attribute that is not resolved through an edge (`via:`);
      4. a mapping `columns:` entry for an attribute the semantic layer does not
         declare, or for a `via:` attribute, which by definition has no column;
      5. a column (or table) that does not exist in the physical schema, for
         every table the schema actually declares -- see `read_ddl_schema` on
         why a partial DDL checks only what it covers;
      6. a `role:` with no realising `predicate:` in the mapping, a `predicate:`
         with no `role:`, or the two disagreeing on the code;
      7. a semantic `key:` naming an attribute whose column is not in the
         mapping's `key_columns` (the business key must sit inside the storage
         key; it may be narrower, as it is for the role objects);
      8. an edge bound to neither `join:` nor `via_table:`, or to both.

    `schema=None` skips rule 5 only. `strict_tables=True` additionally requires
    every mapped table to be declared in the schema, which is right for a
    complete DDL and wrong for this repo's extract.
    """
    if not isinstance(semantic, Mapping) or not isinstance(mapping, Mapping):
        raise OntologyError("both layers must be mappings at the document root")
    _reject_keys(semantic, _SEMANTIC_TOP, "document root", "semantic")
    _reject_keys(mapping, _MAPPING_TOP, "document root", "mapping")

    sem_concepts: Mapping[str, Any] = semantic.get("concepts") or {}
    map_concepts: Mapping[str, Any] = mapping.get("concepts") or {}
    sem_edges: Mapping[str, Any] = semantic.get("edges") or {}
    map_edges: Mapping[str, Any] = mapping.get("edges") or {}
    if not sem_concepts:
        raise OntologyError("semantic layer: no `concepts`")

    for kind, sem_side, map_side in (
        ("concept", sem_concepts, map_concepts), ("edge", sem_edges, map_edges)
    ):
        missing = [n for n in sem_side if n not in map_side]
        if missing:
            raise OntologyError(
                f"{kind}s declared in the semantic layer with no mapping entry: {missing}"
            )
        orphan = [n for n in map_side if n not in sem_side]
        if orphan:
            raise OntologyError(
                f"mapping entries for {kind}s the semantic layer does not declare: {orphan}"
            )

    roles = set(semantic.get("party_roles") or {})
    unverified_tables: set[str] = set()

    def check_column(table: str, column: str, where: str) -> None:
        if schema is None:
            return
        columns = schema.get(table)
        if columns is None:
            if strict_tables:
                raise OntologyError(f"{where}: table {table!r} is not in the physical schema")
            unverified_tables.add(table)
            return
        if column not in columns:
            raise OntologyError(
                f"{where}: column {table}.{column} does not exist in the physical schema"
            )

    concepts: dict[str, Any] = {}
    for name, sem in sem_concepts.items():
        if not isinstance(sem, Mapping):
            raise OntologyError(f"semantic layer, concept {name}: expected a mapping")
        phys = map_concepts[name]
        if not isinstance(phys, Mapping):
            raise OntologyError(f"mapping layer, concept {name}: expected a mapping")
        _reject_keys(sem, _SEMANTIC_CONCEPT, f"concept {name}", "semantic")
        _reject_keys(phys, _MAPPING_CONCEPT, f"concept {name}", "mapping")

        table = phys.get("table")
        if not table:
            raise OntologyError(f"mapping layer, concept {name}: missing `table`")
        table = str(table)
        if schema is not None and table not in schema:
            if strict_tables:
                raise OntologyError(
                    f"mapping layer, concept {name}: table {table!r} is not in the physical schema"
                )
            unverified_tables.add(table)

        columns = phys.get("columns") or {}
        if not isinstance(columns, Mapping):
            raise OntologyError(f"mapping layer, concept {name}: `columns` must be a mapping")
        attrs_spec = sem.get("attributes") or {}
        if not isinstance(attrs_spec, Mapping):
            raise OntologyError(f"semantic layer, concept {name}: `attributes` must be a mapping")

        merged_attrs: dict[str, Any] = {}
        for attr_name, attr_spec in attrs_spec.items():
            if not isinstance(attr_spec, Mapping):
                raise OntologyError(f"semantic layer, attribute {name}.{attr_name}: expected a mapping")
            _reject_keys(attr_spec, _SEMANTIC_ATTRIBUTE, f"attribute {name}.{attr_name}", "semantic")
            resolved = attr_spec.get("via") is not None
            column = columns.get(attr_name)
            if resolved:
                if column is not None:
                    raise OntologyError(
                        f"mapping layer, concept {name}: attribute {attr_name!r} is resolved "
                        f"through edge {attr_spec['via']!r} and cannot have a column"
                    )
                merged_attrs[attr_name] = _ordered(dict(attr_spec), _ATTRIBUTE_ORDER)
                continue
            if column is None:
                raise OntologyError(
                    f"mapping layer, concept {name}: attribute {attr_name!r} has no column "
                    f"(and is not resolved through an edge)"
                )
            check_column(table, str(column), f"concept {name}, attribute {attr_name}")
            merged_attrs[attr_name] = _ordered(
                {"column": str(column), **dict(attr_spec)}, _ATTRIBUTE_ORDER
            )

        stray = [c for c in columns if c not in attrs_spec]
        if stray:
            raise OntologyError(
                f"mapping layer, concept {name}: `columns` names attributes the semantic "
                f"layer does not declare: {sorted(stray)}"
            )

        key_columns = _str_tuple(phys.get("key_columns"), f"mapping concept {name}.key_columns")
        for column in key_columns:
            check_column(table, column, f"concept {name}, key column")
        business_key = _str_tuple(sem.get("key"), f"semantic concept {name}.key")
        for attr_name in business_key:
            if attr_name not in attrs_spec:
                raise OntologyError(
                    f"semantic layer, concept {name}: `key` names unknown attribute {attr_name!r}"
                )
            column = columns.get(attr_name)
            if column is None:
                raise OntologyError(
                    f"semantic layer, concept {name}: key attribute {attr_name!r} is resolved "
                    f"through an edge and cannot be part of an identity"
                )
            if key_columns and str(column) not in key_columns:
                raise OntologyError(
                    f"concept {name}: business key attribute {attr_name!r} maps to column "
                    f"{column!r}, which is not in the mapping's key_columns {list(key_columns)}"
                )

        role = sem.get("role")
        predicate = phys.get("predicate")
        backed_where = _role_predicate(role, predicate, roles, f"concept {name}")
        for column in dict(backed_where or {}):
            check_column(table, column, f"concept {name}, role predicate")

        merged: dict[str, Any] = {"table": table}
        if sem.get("implements") is not None:
            merged["implements"] = sem["implements"]
        if backed_where:
            merged["backed_where"] = backed_where
        if key_columns:

            merged["key"] = key_columns[0] if len(key_columns) == 1 else list(key_columns)
        for key in ("display", "grain", "description"):
            if sem.get(key) is not None:
                merged[key] = sem[key]
        merged["attributes"] = merged_attrs
        if sem.get("searchable") is not None:
            merged["searchable"] = sem["searchable"]
        concepts[name] = _ordered(merged, _CONCEPT_ORDER)

    edges: list[dict[str, Any]] = []
    for name, sem in sem_edges.items():
        if not isinstance(sem, Mapping):
            raise OntologyError(f"semantic layer, edge {name}: expected a mapping")
        phys = map_edges[name]
        if not isinstance(phys, Mapping):
            raise OntologyError(f"mapping layer, edge {name}: expected a mapping")
        _reject_keys(sem, _SEMANTIC_EDGE, f"edge {name}", "semantic")
        _reject_keys(phys, _MAPPING_EDGE, f"edge {name}", "mapping")

        join = phys.get("join")
        via_table = phys.get("via_table")
        if (join is None) == (via_table is None):
            raise OntologyError(
                f"mapping layer, edge {name}: needs exactly one of `join` or `via_table`"
            )

        merged = {"name": name, "from": sem.get("from"), "to": sem.get("to")}
        if not merged["from"] or not merged["to"]:
            raise OntologyError(f"semantic layer, edge {name}: needs both `from` and `to`")
        source_table = _table_of(map_concepts, str(merged["from"]))
        target_table = _table_of(map_concepts, str(merged["to"]))

        if join is not None:
            if not isinstance(join, Mapping) or not join:
                raise OntologyError(f"mapping layer, edge {name}: `join` must be a non-empty mapping")
            for left, right in join.items():
                check_column(source_table, str(left), f"edge {name}, join left")
                check_column(target_table, str(right), f"edge {name}, join right")
            merged["join"] = dict(join)
        else:
            from_join, to_join = phys.get("from_join"), phys.get("to_join")
            if not isinstance(from_join, Mapping) or not from_join \
                    or not isinstance(to_join, Mapping) or not to_join:
                raise OntologyError(
                    f"mapping layer, edge {name}: `via_table` needs both `from_join` and `to_join`"
                )
            for left, right in from_join.items():
                check_column(source_table, str(left), f"edge {name}, from_join source")
                check_column(str(via_table), str(right), f"edge {name}, from_join junction")
            for left, right in to_join.items():
                check_column(str(via_table), str(left), f"edge {name}, to_join junction")
                check_column(target_table, str(right), f"edge {name}, to_join target")
            merged["via"] = {
                "table": via_table, "from_join": dict(from_join), "to_join": dict(to_join),
            }

        restrict = phys.get("restrict")
        if restrict is not None:
            if not isinstance(restrict, Mapping) or "table" not in restrict:
                raise OntologyError(f"mapping layer, edge {name}: `restrict` needs a `table`")
            restrict_columns = restrict.get("columns") or {}
            for left, right in dict(restrict_columns).items():
                check_column(target_table, str(left), f"edge {name}, restrict target")
                check_column(str(restrict["table"]), str(right), f"edge {name}, restrict marker")
            merged["restrict"] = {
                "table": restrict["table"], "columns": dict(restrict_columns),
            }

        role_predicate = _role_predicate(
            sem.get("role"), phys.get("predicate"), roles, f"edge {name}"
        )
        if role_predicate:
            merged["role_predicate"] = role_predicate
            junction = str(via_table) if via_table is not None else source_table
            for column in role_predicate:
                check_column(junction, column, f"edge {name}, role predicate")

        for key in ("cardinality", "fan_out", "fan_out_reverse", "evidence", "description"):
            if sem.get(key) is not None:
                merged[key] = sem[key]
        edges.append(_ordered(merged, _EDGE_ORDER))

    document: dict[str, Any] = {
        "version": semantic.get("version", 1),
        "domain": semantic.get("domain", "unknown"),
    }
    if mapping.get("source") is not None:
        document["source"] = mapping["source"]
    for key in ("value_types",):
        if semantic.get(key) is not None:
            document[key] = semantic[key]
    document["concepts"] = concepts
    if semantic.get("party_roles") is not None:
        document["party_roles"] = semantic["party_roles"]
    document["edges"] = edges
    for key in ("metrics", "links", "glossary"):
        if semantic.get(key) is not None:
            document[key] = semantic[key]
    if unverified_tables:

        document["_unverified_tables"] = sorted(unverified_tables)
    return document

def _table_of(map_concepts: Mapping[str, Any], concept: str) -> str:
    entry = map_concepts.get(concept)
    if not isinstance(entry, Mapping) or not entry.get("table"):
        raise OntologyError(f"mapping layer: concept {concept!r} has no table")
    return str(entry["table"])

def _role_predicate(
    role: Any, predicate: Any, roles: set[str], where: str
) -> dict[str, str]:
    """Rule 6: `role:` (semantic) and `predicate:` (mapping) must agree.

    The role code is the MEANING -- a commitment drawn from `party_roles`. The
    predicate is how one warehouse realises it. Either both are present and
    consistent, or neither is; a predicate with no declared role would be a
    filter the semantic layer cannot see, which is exactly the omission this
    ontology exists to make unrepresentable.
    """
    if role is None and predicate is None:
        return {}
    if role is None:
        raise OntologyError(
            f"{where}: the mapping declares a predicate {predicate!r} but the semantic "
            f"layer declares no `role` -- a filter the semantic layer cannot see"
        )
    if predicate is None:
        raise OntologyError(
            f"{where}: the semantic layer commits to role {role!r} but the mapping "
            f"declares no `predicate` to realise it"
        )
    if not isinstance(predicate, Mapping) or len(predicate) != 1:
        raise OntologyError(
            f"{where}: `predicate` must name exactly one column and one literal, got {predicate!r}"
        )
    (column, value), = predicate.items()
    if str(value) != str(role):
        raise OntologyError(
            f"{where}: semantic role {role!r} and mapping predicate "
            f"{column}={value!r} disagree"
        )
    if roles and str(role) not in roles:
        raise OntologyError(
            f"{where}: role {role!r} is not in `party_roles` {sorted(roles)}"
        )
    return {str(column): str(value)}

def load_ontology(
    path: str | FsPath = DEFAULT_SEMANTIC,
    mapping: str | FsPath = DEFAULT_MAPPING,
    *,
    ddl: str | FsPath | None = DEFAULT_DDL,
) -> Ontology:
    """Load the semantic layer + its mapping layer into a validated `Ontology`.

    The two files are joined by `join_layers` (which raises on any mismatch),
    then typed by `ontology_from_mapping`. The returned object has exactly the
    interface it had when the ontology was one file -- `Concept.table`,
    `Attribute.column`, `Edge.via_table` and `Ontology.raw` are all unchanged --
    plus `mapping_entry()` for provenance.

    `ddl=None` skips the column-existence check (rule 5).
    """
    path, mapping_path = FsPath(path), FsPath(mapping)
    semantic_doc = _read_yaml(path)
    mapping_doc = _read_yaml(mapping_path)
    schema = None if ddl is None else read_ddl_schema(ddl)
    merged = join_layers(semantic_doc, mapping_doc, schema=schema)
    merged.pop("_unverified_tables", None)

    business = {k: merged.pop(k) for k in ("metrics", "links", "glossary") if k in merged}
    ontology = ontology_from_mapping(
        merged, business or None, sources=(str(path), str(mapping_path))
    )
    object.__setattr__(ontology, "mapping", MappingProxyType(dict(mapping_doc)))
    return ontology

def _validate(o: Ontology) -> None:
    seen_edges: set[str] = set()
    for edge in o.edges:
        if edge.name in seen_edges:
            raise OntologyError(f"duplicate edge name {edge.name!r}")
        seen_edges.add(edge.name)
        for role, cname in (("from", edge.source), ("to", edge.target)):
            if cname not in o.concepts:
                raise OntologyError(f"edge {edge.name}: `{role}` names unknown concept {cname!r}")

    for concept in o.concepts.values():
        if concept.display and concept.display not in concept.attributes:
            raise OntologyError(
                f"concept {concept.name}: display key {concept.display!r} is not an attribute"
            )
        if concept.implements:
            if concept.implements not in o.concepts:
                raise OntologyError(
                    f"concept {concept.name}: implements unknown concept {concept.implements!r}"
                )

            seen, node = [concept.name], concept.implements
            while node:
                if node in seen:
                    raise OntologyError(
                        f"concept implements cycle: {' -> '.join(seen + [node])}"
                    )
                seen.append(node)
                node = o.concepts[node].implements if node in o.concepts else None
        for attr in concept.attributes.values():
            if attr.value_type and attr.value_type not in o.value_types:
                raise OntologyError(
                    f"attribute {attr.qualified}: unknown value_type {attr.value_type!r}"
                )
            if attr.via is None:
                continue
            try:
                via_edge = o.edge(attr.via)
            except OntologyError:
                raise OntologyError(
                    f"attribute {attr.qualified}: `via` names unknown edge {attr.via!r}"
                ) from None
            if concept.name not in (via_edge.source, via_edge.target):
                raise OntologyError(
                    f"attribute {attr.qualified}: edge {attr.via!r} is not incident to {concept.name}"
                )
            far = via_edge.target if via_edge.source == concept.name else via_edge.source
            far_concept = o.concepts[far]
            if attr.via_attribute not in far_concept.attributes:
                raise OntologyError(
                    f"attribute {attr.qualified}: {far}.{attr.via_attribute} does not exist"
                )

    _validate_role_codes(o)
    _validate_metrics(o)
    _validate_links(o)

    for entry in o.glossary:
        table = o.metrics if entry.kind == "metric" else o.links
        if entry.target not in table:
            raise OntologyError(
                f"glossary: {entry.kind} {entry.target!r} is not defined "
                f"(phrases {list(entry.phrases)})"
            )

def _validate_role_codes(o: Ontology) -> None:
    """Typo guard on role predicates.

    Generic rule, no domain knowledge: group every declared predicate by column;
    if ANY value on a column is a declared party-role code, then EVERY value on
    that column must be. `{Party_Role_Code: PG}` next to `PH`/`AG`/`UW` is a typo,
    and a typo here silently produces a traversal that returns nothing.
    """
    by_column: dict[str, list[tuple[str, str]]] = {}
    for edge in o.edges:
        for column, value in edge.role_predicate:
            by_column.setdefault(column, []).append((value, f"edge {edge.name}"))
    for concept in o.concepts.values():
        for column, value in concept.backed_where:
            by_column.setdefault(column, []).append((value, f"concept {concept.name}"))

    for column, uses in by_column.items():
        values = {v for v, _ in uses}
        if not (values & set(o.party_roles)):
            continue
        for value, where in uses:
            if value not in o.party_roles:
                raise OntologyError(
                    f"{where}: role code {value!r} on column {column!r} is not in `party_roles` "
                    f"{sorted(o.party_roles)}"
                )

def _validate_metrics(o: Ontology) -> None:
    for metric in o.metrics.values():
        if metric.operand is not None:
            operand = metric.operand
            if operand.concept not in o.concepts:
                raise OntologyError(
                    f"metric {metric.name}: unknown concept {operand.concept!r}"
                )
            concept = o.concepts[operand.concept]
            if operand.attribute not in concept.attributes:
                raise OntologyError(
                    f"metric {metric.name}: {operand.concept} has no attribute "
                    f"{operand.attribute!r}"
                )
            if operand.via is not None:
                edge = o.edge(operand.via)
                if operand.concept not in (edge.source, edge.target):
                    raise OntologyError(
                        f"metric {metric.name}: edge {operand.via!r} does not reach "
                        f"{operand.concept}"
                    )

            attr = concept.attributes[operand.attribute]
            if metric.op in ("sum", "avg") and attr.type is not None \
                    and attr.type not in NUMERIC_TYPES:
                raise OntologyError(
                    f"metric {metric.name}: {metric.op} over {attr.qualified}, "
                    f"declared type {attr.type!r}"
                )
        for component in metric.components:
            if component not in o.metrics:
                raise OntologyError(
                    f"metric {metric.name}: component {component!r} is not a metric"
                )

        if metric.per is not None:
            bases = o.measured_over(metric.name)
            if len(bases) != 1:
                raise OntologyError(
                    f"metric {metric.name}: per={metric.per!r}, but it is measured "
                    f"over {list(bases)} -- a composition spanning several grains "
                    f"has no single row to be formed on, and an aggregate over it "
                    f"would depend on which grain happened to be chosen"
                )
            if metric.per != bases[0]:
                raise OntologyError(
                    f"metric {metric.name}: per={metric.per!r} but its components' "
                    f"edges all leave {bases[0]!r}"
                )

    state: dict[str, int] = {}

    def visit(name: str, stack: tuple[str, ...]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise OntologyError(f"metric composition cycle: {' -> '.join(stack + (name,))}")
        state[name] = 1
        for component in o.metrics[name].components:
            visit(component, stack + (name,))
        state[name] = 2

    for name in o.metrics:
        visit(name, ())

def _validate_links(o: Ontology) -> None:
    for link in o.links.values():
        for role, cname in (("from", link.source), ("to", link.target)):
            if cname not in o.concepts:
                raise OntologyError(f"link {link.name}: `{role}` names unknown concept {cname!r}")
        if not link.backed_by:
            raise OntologyError(f"link {link.name}: no `backed_by` edges")
        node = link.source
        for edge_name in link.backed_by:
            edge = o.edge(edge_name)
            if node == edge.source:
                node = edge.target
            elif node == edge.target:
                node = edge.source
            else:
                raise OntologyError(
                    f"link {link.name}: edge {edge_name!r} is not incident to {node!r} "
                    f"-- `backed_by` is not a connected chain"
                )
        if node != link.target:
            raise OntologyError(
                f"link {link.name}: `backed_by` chain ends at {node!r}, not {link.target!r}"
            )

def metadata_report(o: Ontology) -> dict[str, Any]:
    """What the three attribute additions would have to carry, per attribute.

    Reported, never written: `ontology/acme.semantic.yaml` is authored under a
    provenance rule (see its header) and this module does not edit it.

    Since the two-layer split these gaps read more sharply. An attribute name is
    now purely semantic -- no column's spelling stands behind it -- so
    `InsurableObject.object_type` appearing under `ungroundable` is a MISSING
    DESCRIPTION and not a physical name that happened to look self-explanatory.
    It is also the one attribute binding `spc/derive_mapping.py` cannot recover
    from the schema, which is the same defect seen from the other side.

    Three buckets:
      `undescribed`  attributes with no `description`. A planner grounding a
                     question against `object_type` or `location_code` has only
                     the identifier to go on.
      `unlabelled`   attributes carrying NEITHER `label` nor `aliases`. These
                     are the ones no question phrase can reach: the resolver has
                     nothing but the identifier's own spelling to score against.
                     Completeness here is the anti-bias property a lint asserts
                     -- labelling a subset would make the subset a hidden input.
      `unaliased`    string/date attributes with no `aliases`. Weaker than
                     `unlabelled`: one accepted phrasing exists, alternatives do
                     not.
      `ungroundable` non-key string attributes not marked searchable, i.e. no
                     column a value literal may be resolved against. Palantir:
                     filtering only works on indexed (searchable) properties.
      `searchable_by_inheritance`
                     `via`-resolved attributes that are not marked searchable
                     but resolve to one that is. They are groundable in
                     practice, and the format has no way to SAY so -- the
                     inheritance is inferred here, not declared.
    """
    undescribed: list[str] = []
    unlabelled: list[str] = []
    unaliased: list[str] = []
    ungroundable: list[str] = []
    inherited: list[str] = []
    key_columns = {c.name: set(c.key) for c in o.concepts.values()}

    def resolves_to_searchable(attr: Attribute) -> bool:
        if attr.via is None:
            return False
        edge = o.edge(attr.via)
        far = edge.target if edge.source == attr.concept else edge.source
        target = o.concepts[far].attributes.get(attr.via_attribute or "")
        return bool(target and target.searchable)

    for name in o.concept_names():
        concept = o.concepts[name]
        for attr in concept.attributes.values():
            if not attr.description:
                undescribed.append(attr.qualified)
            if not attr.labels:
                unlabelled.append(attr.qualified)
            if not attr.aliases:
                unaliased.append(attr.qualified)
            is_identifier = attr.column in key_columns[name] or attr.value_type == "Identifier"
            if attr.type in ("string", None) and not attr.searchable and not is_identifier:
                if resolves_to_searchable(attr):
                    inherited.append(attr.qualified)
                else:
                    ungroundable.append(attr.qualified)

    total = sum(len(c.attributes) for c in o.concepts.values())
    return {
        "attributes": total,
        "concepts": len(o.concepts),
        "edges": len(o.edges),
        "undescribed": undescribed,
        "unlabelled": unlabelled,
        "unaliased": unaliased,
        "ungroundable": ungroundable,
        "searchable_by_inheritance": inherited,
        "searchable_declared": [
            a.qualified for c in o.concepts.values() for a in c.searchable_attributes
        ],
        "undescribed_concepts": [
            c.name for c in o.concepts.values() if not c.description
        ],
        "undescribed_edges": [e.name for e in o.edges if not e.description],
        "key_columns_without_attribute": [
            f"{c.name}.{column}"
            for c in o.concepts.values()
            for column in c.key
            if column not in {a.column for a in c.attributes.values() if a.column}
        ],
    }

def _main() -> None:  # pragma: no cover
    ontology = load_ontology()
    report = metadata_report(ontology)
    print(f"{ontology.domain} v{ontology.version}: "
          f"{report['concepts']} concepts, {report['edges']} edges, "
          f"{report['attributes']} attributes")
    for key in (
        "undescribed", "unlabelled", "unaliased",
        "ungroundable", "searchable_by_inheritance",
        "searchable_declared",
        "undescribed_concepts", "undescribed_edges", "key_columns_without_attribute",
    ):
        values = report[key]
        print(f"\n{key} ({len(values)}):")
        for value in values:
            print(f"  {value}")

if __name__ == "__main__":  # pragma: no cover
    _main()
