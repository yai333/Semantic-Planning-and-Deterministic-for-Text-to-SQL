from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from spc.graph import DEFAULT_MAX_HOPS, PathGraph
from spc.ontology import Metric, Ontology
from spc.plan import (
    SUBJECT,
    Filter,
    Measure,
    Path,
    Plan,
    Projection,
    Top,
)

__all__ = [
    "Menu",
    "MenuRoute",
    "build_menu",
    "parse_pick",
    "rooted_paths",
    "route_ids_named",
    "PickError",
    "SUBJECT_ID",
]

SUBJECT_ID = "SELF"

_AGGREGATIONS = ("sum", "count", "count_distinct", "avg", "min", "max")
_OPERATORS = ("=", "!=", "<", "<=", ">", ">=", "LIKE", "IN", "BETWEEN")
_COMBINES = ("+", "-", "*", "/")

class PickError(ValueError):
    """A pick that names something the menu does not offer."""

@dataclass(frozen=True)
class MenuRoute:
    """One governed path, offered under a stable id."""

    id: str
    subject: str
    target: str
    path: Path
    roles: tuple[str, ...]
    fans_out: bool

    @property
    def hops(self) -> int:
        return len(self.path.steps)

    def notation(self) -> str:
        return " ".join(
            f"{s.edge}{'>' if s.forward else '<'}" for s in self.path.steps
        ) or "-"

@dataclass(frozen=True)
class Menu:
    concepts: tuple[str, ...]
    routes: tuple[MenuRoute, ...]
    metrics: tuple[str, ...]
    ontology: Ontology
    graph: PathGraph
    _by_id: Mapping[str, MenuRoute] = field(default_factory=dict)

    def route(self, route_id: str) -> MenuRoute:
        try:
            return self._by_id[route_id]
        except KeyError:
            raise PickError(f"no route {route_id!r} on the menu") from None

    def routes_for(self, subject: str) -> tuple[MenuRoute, ...]:
        return tuple(r for r in self.routes if r.subject == subject)

    def render(self, *, subject: str | None = None) -> str:
        """The menu as the model sees it. Pure, ordered, stable."""
        lines: list[str] = []
        lines.append("CONCEPTS  (pick exactly one as `subject`; it fixes the grain)")
        for name in self.concepts:
            concept = self.ontology.concept(name)
            attributes = ", ".join(sorted(concept.attributes))
            note = f"  -- {_one_line(concept.description)}" if concept.description else ""
            role = f"  [role: {concept.role_code}]" if concept.role_code else ""
            lines.append(f"  {name}{role}{note}")
            lines.append(f"      attributes: {attributes}")

        if self.metrics:
            lines.append("")
            lines.append("METRICS  (governed; use by name -- a metric may not be redefined)")
            for name in self.metrics:
                lines.append(f"  {name} = {_definition(self.ontology, name)}")
                description = self.ontology.metric(name).description
                if description:
                    lines.append(f"      {_one_line(description)}")

        lines.append("")
        lines.append("ROUTES  (governed paths; use the id as `route`. "
                     f"`{SUBJECT_ID}` = the subject itself, no traversal)")
        shown = self.routes if subject is None else self.routes_for(subject)
        for route in shown:
            roles = "+".join(route.roles) if route.roles else "-"
            fan = "fans-out" if route.fans_out else "one-to-one"
            lines.append(
                f"  {route.id}  {route.subject} -> {route.target}"
                f"  {route.hops} hop(s)  roles:{roles}  {fan}"
            )
            lines.append(f"      {route.notation()}")
        return "\n".join(lines)

    def token_size(self, *, subject: str | None = None) -> int:
        """The arm's fixed prompt cost, measured not guessed.

        `tiktoken` when it is installed; otherwise the standard 4-characters-per
        -token approximation, which is flagged in the report so a number from one
        estimator is never compared with a number from the other.
        """
        text = self.render(subject=subject)
        try:                                    # pragma: no cover
            import tiktoken

            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:                       # noqa: BLE001
            return (len(text) + 3) // 4

    def token_estimator(self) -> str:
        try:                                    # pragma: no cover
            import tiktoken  # noqa: F401

            return "tiktoken/cl100k_base"
        except Exception:                       # noqa: BLE001
            return "chars/4 approximation"

    @property
    def _has_per_metric(self) -> bool:
        """Whether any metric on this menu declares the grain it is formed on.

        `over` is meaningless without one, and a field the model can emit but
        never use is a field it will eventually use wrongly.
        """
        return any(getattr(self.ontology.metric(name), "per", None)
                   for name in self.metrics)

    def pick_schema(self, *, subject: str | None = None) -> dict[str, Any]:
        """A JSON schema whose every structural field is an enum over menu ids.

        The point is not validation, it is EXPRESSIBILITY: a constrained decoder
        given this schema cannot emit a route, a metric, a concept or an operator
        that is not on the menu, so the failure mode "invalid plan" is removed and
        only "wrong reading" is left. Attribute NAMES are enumerated over the
        union of the candidate concepts' attributes; the concept-specific binding
        is enforced by `parse_pick`, which the schema cannot express.

        The shape is deliberately what a CONSTRAINED DECODER accepts, not merely
        what a validator accepts, because this schema is sent as the arm's
        `response_format` rather than only checked afterwards. That imposes three
        rules the looser earlier version broke:

          * every declared property is `required`; a field the model may omit is
            expressed as required-and-NULLABLE instead. `parse_pick` reads an
            explicit null exactly as it reads an absent key;
          * no numeric-range keywords (`minimum`), which a strict decoder rejects;
          * a composite measure's `parts` are the SAME schema recursively, via
            `$defs`/`$ref`. Left as a bare `{"type": "object"}` -- as it was --
            the nested parts of a composite were the one place an off-menu metric
            or route was still emissible, which defeats the whole construction.
        """
        routes = self.routes if subject is None else self.routes_for(subject)
        route_ids = [SUBJECT_ID] + [r.id for r in routes]
        attributes = sorted({
            name
            for concept in self.concepts
            for name in self.ontology.concept(concept).attributes
        })

        def nullable(enum: Sequence[str]) -> dict[str, Any]:
            """An enum the model may decline to use. Required, but null-able."""
            return {"anyOf": [{"type": "string", "enum": list(enum)},
                              {"type": "null"}]}

        reference = {
            "type": "object",
            "properties": {
                "route": {"type": "string", "enum": route_ids},
                "attribute": nullable(attributes),
            },
            "required": ["route", "attribute"],
            "additionalProperties": False,
        }
        measure = {
            "type": "object",
            "properties": {
                "metric": nullable(self.metrics),
                "aggregation": nullable(_AGGREGATIONS),
                "route": {"type": "string", "enum": route_ids},
                "attribute": nullable(attributes),
                "combine": nullable(_COMBINES),

                **({"over": nullable(_AGGREGATIONS)} if self._has_per_metric else {}),

                "parts": {"anyOf": [
                    {"type": "array", "items": {"$ref": "#/$defs/measure"}},
                    {"type": "null"},
                ]},
            },
            "required": ["metric", "aggregation", "route", "attribute",
                         "combine", "parts"]
                        + (["over"] if self._has_per_metric else []),
            "additionalProperties": False,
        }
        top = {
            "type": "object",
            "properties": {
                "by": {"type": "string", "enum": ["measure", "dimension"]},
                "index": {"type": "integer"},
                "descending": {"type": "boolean"},
                "n": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["by", "index", "descending", "n"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "$defs": {"measure": measure},
            "properties": {
                "subject": {"type": "string",
                            "enum": [subject] if subject else list(self.concepts)},
                "measures": {"type": "array", "items": {"$ref": "#/$defs/measure"}},
                "dimensions": {"type": "array", "items": reference},
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "route": {"type": "string", "enum": route_ids},
                            "attribute": {"type": "string", "enum": attributes},
                            "operator": {"type": "string", "enum": list(_OPERATORS)},

                            "value": {"anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "null"},
                                {"type": "array", "items": {"anyOf": [
                                    {"type": "string"},
                                    {"type": "number"},
                                    {"type": "boolean"},
                                ]}},
                            ]},
                        },
                        "required": ["route", "attribute", "operator", "value"],
                        "additionalProperties": False,
                    },
                },
                "top": {"anyOf": [top, {"type": "null"}]},
            },
            "required": ["subject", "measures", "dimensions", "filters", "top"],
            "additionalProperties": False,
        }

    def response_format(self, *, subject: str | None = None,
                        name: str = "governed_pick") -> dict[str, Any]:
        """`pick_schema` wrapped as an OpenAI chat-completions `response_format`.

        Built here rather than in the harness so the schema the arm SENDS and the
        schema `parse_pick` enforces are the same object, produced once.
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": True,
                "schema": self.pick_schema(subject=subject),
            },
        }

def _one_line(text: str | None) -> str:
    return " ".join((text or "").split())

def _definition(onto: Ontology, name: str) -> str:
    """A governed metric's AUTHORED definition, rendered structurally."""
    metric: Metric = onto.metric(name)
    if metric.is_composite:
        symbol = {"add": " + ", "subtract": " - ",
                  "multiply": " * ", "divide": " / "}[metric.combine or "add"]
        return symbol.join(metric.components)
    operand = metric.operand
    if operand is None:                                      # pragma: no cover
        return "?"
    via = f" via {operand.via}" if operand.via else ""
    return f"{metric.op}({operand.concept}.{operand.attribute}{via})"

def build_menu(
    concepts: Sequence[str],
    onto: Ontology,
    graph: PathGraph,
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    metrics: Sequence[str] | None = None,
) -> Menu:
    """The governed choices available over `concepts`.

    Ids are assigned in the enumerator's own total order over sorted concept
    pairs, so the same candidate set always yields the same menu -- an id is
    stable enough to appear in a logged pick and still mean the same route.
    """
    names = tuple(sorted(dict.fromkeys(concepts)))
    for name in names:
        onto.concept(name)

    routes: list[MenuRoute] = []
    for source in names:
        for target in names:
            if source == target:
                continue
            for path in graph.paths(source, target, max_hops=max_hops):
                routes.append(MenuRoute(
                    id="",
                    subject=source,
                    target=target,
                    path=path,
                    roles=graph.role_signature(source, path),
                    fans_out=_fans_out(onto, path),
                ))
    numbered = tuple(
        MenuRoute(id=f"R{index + 1:02d}", subject=r.subject, target=r.target,
                  path=r.path, roles=r.roles, fans_out=r.fans_out)
        for index, r in enumerate(routes)
    )
    chosen = tuple(sorted(metrics)) if metrics is not None else tuple(sorted(onto.metrics))
    return Menu(
        concepts=names,
        routes=numbered,
        metrics=chosen,
        ontology=onto,
        graph=graph,
        _by_id={r.id: r for r in numbered},
    )

def _fans_out(onto: Ontology, path: Path) -> bool:
    for step in path.steps:
        edge = onto.edge(step.edge)
        if edge.fan_out_in(forward=step.forward) != "none":
            return True
    return False

def route_ids_named(pick: str | Mapping[str, Any]) -> tuple[str, ...]:
    """Every route id a pick mentions, in document order, `SELF` dropped.

    Walked structurally rather than by regex so a nested composite measure
    cannot hide a route from the rooting pass below.
    """
    data = json.loads(pick) if isinstance(pick, str) else dict(pick)
    found: list[str] = []

    def note(value: Any) -> None:
        if value is None:
            return
        text = str(value)
        if text == SUBJECT_ID or text in found:
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

def rooted_paths(pick: Mapping[str, Any], menu: Menu, subject: str) -> dict[str, Path]:
    """Each route the pick names, expressed as a route FROM the subject.

    A route id names a governed route between its own two endpoints, and the
    tools that mint ids -- `find_paths(from, to)`, `find_metric(..., subject=)` --
    take those endpoints as arguments. Nothing about that surface says the
    `from` end must be the pick's SUBJECT, and the natural way to reach a
    third concept is to ask for each leg: `Agent>Policy`, then `Policy>Claim`,
    then `Claim>Catastrophe`. The plan normal form roots every route at the
    subject (`Plan.paths`), so those legs used to be rejected one by one --
    "route Policy>Claim#1 starts at 'Policy', not at the chosen subject
    'Agent'" -- for naming exactly the traversal the question asks for.

    They are COMPOSED here instead, and composition is governed the same way a
    single route is: a leg is accepted only when prefix ++ leg is itself in
    `PathGraph.paths(subject, target)`, i.e. a route the enumerator would have
    produced on its own. Nothing new becomes reachable; what changes is that a
    route already reachable may be NAMED in legs.

    Two rules keep it deterministic and unambiguous:

      * the prefix must be UNIQUE. Where the pick roots more than one distinct
        route at the leg's own start concept, "which of them does this leg hang
        off" has no answer and the leg is left unrooted (and then refused).
      * legs compose to a FIXPOINT rather than in document order, because a
        pick's measures are parsed before its dimensions and the leg that
        supplies a prefix is often written after the leg that needs it.

    Composition preserves the chain the model wrote, which is the point:
    `_Layout.walk` shares the prefix of two paths, so a Claim and the
    Catastrophe reached THROUGH it land on one branch. Rooting each leg
    independently at the subject would have paired every claim with every
    catastrophe of that subject instead.
    """
    identifiers = route_ids_named(pick)
    routes = {rid: menu.route(rid) for rid in identifiers}
    rooted = {rid: r.path for rid, r in routes.items() if r.subject == subject}
    pending = [rid for rid in identifiers if rid not in rooted]
    while pending:
        progress = False
        for route_id in list(pending):
            route = routes[route_id]
            prefixes = {p for p in rooted.values() if p.target == route.subject}
            if len(prefixes) != 1:
                continue
            prefix = next(iter(prefixes))
            composed = Path(steps=prefix.steps + route.path.steps, target=route.target)
            if composed not in menu.graph.paths(
                    subject, route.target, max_hops=len(composed.steps)):
                continue
            rooted[route_id] = composed
            pending.remove(route_id)
            progress = True
        if not progress:
            break
    return rooted

def parse_pick(pick: str | Mapping[str, Any], menu: Menu) -> Plan:
    """Turn the model's structured choice into a `Plan`.

    Every route, metric, concept and operator must be on the menu; anything else
    raises `PickError`. Route ids become INDICES into `Plan.paths` in first-use
    order, which is the normal form's rule that a relationship is named in
    exactly one place and referred to everywhere else by index.

    A route id whose own start concept is not the subject is COMPOSED onto the
    route that reaches it -- see `rooted_paths`, which also says why.

    An explicit `null` reads as an absent key throughout. A constrained decoder
    cannot omit a property -- `pick_schema` marks every field required and makes
    the optional ones nullable -- so `{"route": null}` is how such a decoder says
    "the subject itself", and `x.get(k, default)` would hand `None` on to the
    lookups instead.
    """
    data = json.loads(pick) if isinstance(pick, str) else dict(pick)
    subject = data.get("subject")
    if subject not in menu.concepts:
        raise PickError(f"subject {subject!r} is not on the menu {list(menu.concepts)}")

    rooted = rooted_paths(data, menu, subject)
    order: list[str] = []

    def reference(route_id: str) -> int:
        if route_id == SUBJECT_ID:
            return SUBJECT
        route = menu.route(route_id)
        if route_id not in rooted:
            raise PickError(
                f"route {route_id} starts at {route.subject!r}, not at the chosen "
                f"subject {subject!r}, and the pick names no single governed route "
                f"from {subject!r} to {route.subject!r} to compose it onto"
            )
        if route_id not in order:
            order.append(route_id)
        return order.index(route_id)

    def concept_of(route_id: str) -> str:
        return subject if route_id == SUBJECT_ID else menu.route(route_id).target

    def attribute(route_id: str, name: str | None) -> str | None:
        if name is None:
            return None
        concept = menu.ontology.concept(concept_of(route_id))
        if name not in concept.attributes:
            raise PickError(f"{concept.name} has no attribute {name!r}")
        return name

    def measure(spec: Mapping[str, Any]) -> Measure:
        stated = [k for k in ("metric", "aggregation", "combine") if spec.get(k)]
        if len(stated) != 1:
            raise PickError(
                "a measure names exactly one of `metric`, `aggregation`, `combine`; "
                f"got {stated}"
            )
        if "combine" in stated:
            operator = spec["combine"]
            if operator not in _COMBINES:
                raise PickError(f"unknown combine {operator!r}")
            parts = spec.get("parts") or ()
            if len(parts) < 2:
                raise PickError("a composite needs at least two parts")
            return Measure(combine=operator, parts=tuple(measure(p) for p in parts))
        route_id = spec.get("route") or SUBJECT_ID
        if "metric" in stated:
            name = spec["metric"]
            if name not in menu.metrics:
                raise PickError(f"metric {name!r} is not on the menu")

            over = spec.get("over")
            if over:
                if over not in _AGGREGATIONS:
                    raise PickError(f"unknown aggregation {over!r}")
                metric = menu.ontology.metric(name)
                if not getattr(metric, "per", None):
                    raise PickError(
                        f"`over` needs {name!r} to declare the grain it is formed "
                        f"on (`per`); without it there is no row to aggregate")
                if len(menu.ontology.measured_over(name)) != 1:
                    raise PickError(
                        f"{name!r} is measured over "
                        f"{list(menu.ontology.measured_over(name))} -- a composite "
                        f"spanning several grains has no single row to aggregate")
            return Measure(governed=name, over=over or None,
                           path=reference(route_id))
        aggregation = spec["aggregation"]
        if aggregation not in _AGGREGATIONS:
            raise PickError(f"unknown aggregation {aggregation!r}")
        return Measure(
            aggregation=aggregation,
            path=reference(route_id),
            attribute=attribute(route_id, spec.get("attribute")),
        )

    measures = tuple(measure(m) for m in data.get("measures") or ())
    projections = tuple(
        Projection(path=reference(d.get("route") or SUBJECT_ID),
                   attribute=attribute(d.get("route") or SUBJECT_ID, d.get("attribute")))
        for d in data.get("dimensions") or ()
    )
    filters = tuple(
        Filter(path=reference(f.get("route") or SUBJECT_ID),
               attribute=attribute(f.get("route") or SUBJECT_ID, f["attribute"]) or "",
               operator=_operator(f["operator"]),
               value=f["value"])
        for f in data.get("filters") or ()
    )

    top = None
    if data.get("top"):
        spec = data["top"]
        index = int(spec.get("index") or 0)
        by = spec.get("by")
        if by not in ("measure", "dimension"):
            raise PickError("top.by is 'measure' or 'dimension'")
        top = Top(
            by_measure=index if by == "measure" else None,
            by_projection=index if by == "dimension" else None,
            descending=bool(spec.get("descending", False)),
            n=int(spec["n"]) if spec.get("n") is not None else None,
        )

    paths = tuple(rooted[route_id] for route_id in order)
    return Plan(subject=subject, paths=paths, project=projections,
                measures=measures, filters=filters, top=top)

def _operator(operator: str) -> Any:
    if operator not in _OPERATORS:
        raise PickError(f"unknown operator {operator!r}")
    return operator

def _main() -> None:  # pragma: no cover
    from spc.ontology import load_ontology

    onto = load_ontology()
    graph = PathGraph(onto)
    for candidates in (
        ["Claim", "Policy", "ClaimAmount"],
        ["Claim", "Policy", "ClaimAmount", "Catastrophe"],
        ["Party", "Policy", "Claim", "ClaimAmount"],
    ):
        menu = build_menu(candidates, onto, graph)
        print(f"{candidates}: {len(menu.routes)} routes, "
              f"{menu.token_size()} tokens ({menu.token_estimator()})")

if __name__ == "__main__":  # pragma: no cover
    _main()
