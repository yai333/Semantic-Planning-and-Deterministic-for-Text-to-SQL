from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass(frozen=True, order=True)
class Step:
    """One governed edge traversal.

    `role` is not supplied by a planner; it is read off the ontology -- from the
    edge's `role_predicate`, or from the `backed_where` of a role-object concept
    the step lands on. Role commitment is therefore INTRINSIC to the path, which
    is what makes "role omission" unrepresentable instead of merely detectable.
    """

    edge: str
    forward: bool
    role: str | None = None

@dataclass(frozen=True, order=True)
class Path:
    """A governed route from the plan's subject to one concept.

    Branch identity is preserved -- two paths that reach the same concept by
    different routes are different paths, because ACME permits (say) PolicyHolder
    on one branch and Agent on another, and flattening roles into one global set
    loses exactly that distinction.
    """

    steps: tuple[Step, ...]
    target: str

    @property
    def roles(self) -> tuple[str, ...]:
        """The role signature. Derived, ordered, branch-preserving."""
        return tuple(s.role for s in self.steps if s.role)

PathRef = int
SUBJECT: PathRef = -1

@dataclass(frozen=True, order=True)
class Projection:
    """A row-grain column.

    `attribute` may be None, meaning "the concept's declared title key" -- the
    derivation replaces it at compile time. Two plans differing only by writing
    the title key explicitly are the same plan, so only one of them is
    representable: the canonical form leaves it None.
    """

    path: PathRef
    attribute: str | None = None

Aggregation = Literal["sum", "count", "count_distinct", "avg", "min", "max"]
Combine = Literal["+", "-", "*", "/"]

@dataclass(frozen=True, order=True)
class Measure:
    """An aggregate. One list, not two.

    The old IR split `metrics` and `composites` across two fields and then had to
    re-merge them; a governed name and a raw aggregation and a composite are all
    just measures. Exactly one of (`governed`, `aggregation`, `combine`) is set.

    A one-part composite is NOT representable -- it IS the bare measure.
    Operands of `+` and `*` are canonically ordered, because operand order is
    semantically inert and would otherwise multiply the candidate space by n!.
    That ordering is `canonical_operands` below, and it is applied in BOTH
    places or in neither: `canonicalise` orders the key's operands and
    `_Compiler._expand` orders the ones it lowers, using the same total order on
    the same remapped path references. Ordering only the key would merge two
    plans that compile differently, which trades SOUNDNESS -- the property the
    enumerator relies on -- for completeness.

    `governed` is expanded against `path` like any other measure. For a leaf
    metric `path` is the full route, ending in the metric's own edge; for a
    COMPOSITE it is the route to the concept the composite measures over, and
    each component appends its own edge (spc/compile.py, commitment 1b).
    """

    governed: str | None = None

    over: Aggregation | None = None
    aggregation: Aggregation | None = None
    path: PathRef = SUBJECT
    attribute: str | None = None
    combine: Combine | None = None
    parts: tuple["Measure", ...] = ()

MeasureRef = int

Operator = Literal["=", "!=", "<", "<=", ">", ">=", "LIKE", "IN", "BETWEEN"]

@dataclass(frozen=True, order=True)
class Filter:
    """A row predicate.

    `value` is a literal that has been GROUNDED against the database -- the
    canonical value the probe resolved, not a string a model invented. An
    unprobed literal is how a plan came to emit 'deputy' against data holding
    'Deputy' and return zero rows without complaint.
    """

    path: PathRef
    attribute: str
    operator: Operator
    value: object

@dataclass(frozen=True, order=True)
class Having:
    """An aggregate predicate. References a measure BY INDEX rather than
    restating it, so the two cannot drift apart."""

    measure: MeasureRef
    operator: Operator
    value: object

@dataclass(frozen=True, order=True)
class Top:
    """Ordering and truncation as ONE node.

    They were two fields with a cross-field rule ("a limit without an order is
    refused"), which is a constraint the type system should have carried instead.
    `n = None` means order without truncation.
    """

    by_projection: int | None = None
    by_measure: MeasureRef | None = None
    descending: bool = False
    n: int | None = None

SetOp = Literal["union", "intersect", "except"]

@dataclass(frozen=True)
class Plan:
    """A governed, grain-safe reading of a question.

    Absent by construction, because each is DERIVED:
      group_by  -- the non-aggregated projections
      roles     -- the role signatures of `paths`
      distinct  -- from the declared fan-out along each projection's path
      alias     -- from the governed name or attribute
      via       -- it IS the path index

    Anything a planner could state inconsistently with the rest of the plan is
    not stated at all.
    """

    subject: str
    paths: tuple[Path, ...] = ()
    project: tuple[Projection, ...] = ()
    measures: tuple[Measure, ...] = ()
    filters: tuple[Filter, ...] = ()
    having: tuple[Having, ...] = ()
    top: Top | None = None
    set_op: tuple[SetOp, "Plan"] | None = None

@dataclass(frozen=True)
class Clarify:
    """The answer when the question does not determine one.

    Returned when candidates that are identical after erasing role signature
    carry more than one distinct signature -- decided BEFORE pruning, because a
    fixed beam is deterministic and still unsafe: it can deterministically choose
    the same unjustified role forever.
    """

    question: str
    alternatives: tuple[Plan, ...]
    differ_by: str = "role"

Answer = Plan | Clarify

canonical_gaps: dict[str, str] = {
    "path order reversed":
        "reordering `Plan.paths` and remapping every reference changes alias "
        "allocation and therefore the SQL bytes in most cases (30 of 41 in the "
        "generated set) -- but not all: the remaining 11 compile identically "
        "under two different keys. Order cannot be canonicalised away without "
        "simulating `_Layout`, and a rewrite that is inert only sometimes is a "
        "rewrite that breaks SOUNDNESS the rest of the time.",
    "unreferenced route appended":
        "a route nothing points at is inert 777 times in 779 and NOT inert "
        "twice: walking a path is eager and `via` attribute resolution is lazy, "
        "so declaring a route through Party makes a role object's own `name` "
        "resolve to `party_2`. Exactly the rule that would have been adopted on "
        "inspection and been wrong.",
    "display attribute written out":
        "a projection naming its concept's display attribute and one leaving "
        "`attribute` None are the same projection -- the compiler substitutes "
        "the title key for None. The equivalence is NOT decidable from the plan "
        "alone, which is why `canonical_key(plan, onto)` takes an optional "
        "ontology and closes this gap when given one. Without it: 464 of 464 "
        "pairs same SQL, different key.",
    "projection / measure / filter order":
        "each is preserved into the SELECT and WHERE, so an n-item list has n! "
        "encodings that all mean one thing and none of which agree byte for "
        "byte. Not a gap in the key (the key follows the SQL exactly); a gap in "
        "the GRAMMAR, and a source of churn no compiler determinism can remove.",
}

canonical_gaps_closed: dict[str, str] = {
    "composite governed metric ignores Measure.path":
        "CLOSED 2026-08-12 in `spc/compile.py`. `_Compiler._governed` recursed "
        "into a composite metric's components with `SUBJECT`, discarding the "
        "`Measure.path` the plan stated, so `Measure(governed='TotalLoss', "
        "path=k)` compiled identically for every k -- 12 plans naming 12 "
        "different governed routes produced one SQL string. Re-measured with "
        "ONLY this branch reverted: it caused 15 of the 661 distinct SQL "
        "strings then reachable from more than one key, and those 15 carried "
        "23721 pairwise key-disagreements between plans compiling identically. "
        "(This entry first said '21987 completeness violations'. That was a "
        "PAIRWISE count reported in the unit `test_complete` reports, which is "
        "SQL strings reachable from more than one key -- two different units, "
        "three orders of magnitude apart. Both are given above, each named.) "
        "The path is the "
        "plan's one degree of freedom (DESIGN 3.3) and for the metric that "
        "carries most of the benchmark it was not exercised at all. Now a "
        "composite's route is the route to the concept it is a QUANTITY OF "
        "(`Ontology.measured_over`) and each component appends its own edge; a "
        "route to a component's operand, or to any other concept, is REFUSED, "
        "which is what the non-composite branch always did. The 24-plan "
        "determinism digest did not move: no gold plan routed a composite.",
    "composite operand order":
        "CLOSED 2026-08-12. `Measure.parts` was folded left to right, so "
        "`a + b` and `b + a` compiled differently in 140 of 140 measured cases "
        "while this file's own docstring claimed they were canonically ordered. "
        "`canonical_operands` now orders the operands of the commutative "
        "operators, and is applied in BOTH the key and the compiler over the "
        "same remapped path references -- ordering only the key would have "
        "merged plans that compile differently, trading soundness for "
        "completeness. `-` and `/` keep the order they were written in.",
}

def _measure_structure(node: Measure, ref) -> Any:
    """A measure as ordered JSON-able scalars, path references remapped.

    One definition serving three callers -- the digest, the operand order and the
    compiler's operand order -- because two definitions would be two chances for
    the key and the SQL to disagree.
    """
    return [node.governed, node.aggregation, ref(node.path), node.attribute,
            node.combine, [_measure_structure(p, ref) for p in node.parts]]

def canonical_operands(combine: Combine | None, parts: tuple[Measure, ...],
                       ref=None) -> tuple[Measure, ...]:
    """Operands of `+` and `*` in one fixed order; every other operator's kept.

    `a + b` and `b + a` are one measure -- the operators are commutative, so the
    order carries no meaning and an n-operand composite would otherwise have n!
    encodings. `-` and `/` are NOT commutative and are left exactly as written.

    `ref` remaps path indices before comparing, so that a plan and its
    canonicalisation (which collapses restated routes and therefore renumbers
    them) sort their operands the same way. Without it a restated route could
    reorder the operands of the canonical form only, and the key would name SQL
    the compiler does not emit.
    """
    if combine not in ("+", "*") or len(parts) < 2:
        return tuple(parts)
    identity = ref if ref is not None else (lambda value: value)
    return tuple(sorted(parts, key=lambda part: json.dumps(
        _measure_structure(part, identity), sort_keys=True,
        separators=(",", ":"), default=repr)))

def canonical_ref(plan: Plan):
    """The path-reference remap `canonicalise` applies to `plan`.

    Exported so the compiler can order operands by the SAME key the normal form
    uses, rather than by a second, nearly-identical one.
    """
    _paths, remap = _canonical_paths(plan)
    return lambda value: remap.get(value, value)

def _canonical_paths(plan: Plan) -> tuple[tuple[Path, ...], dict[int, int]]:
    """Roles erased, duplicates collapsed. Returns the paths and a ref remap.

    Two rewrites, both measured inert on every generated plan:

    ROLE ERASURE. `Step.role` is not planner-supplied and the compiler does not
    read it -- `_Compiler._govern` discards whatever is there and re-derives the
    role from the edge and the landing concept. So a plan carrying a role, a
    plan carrying the WRONG role, and a plan carrying none are one plan.

    DUPLICATE COLLAPSE. The same route stated twice is one route: `_Layout`
    memoises a traversal by its edge prefix, so the second copy allocates
    nothing and a reference to either copy resolves to the same alias. Order is
    otherwise preserved, because order is NOT inert (see the header).
    """
    seen: dict[Path, int] = {}
    kept: list[Path] = []
    remap: dict[int, int] = {SUBJECT: SUBJECT}
    for index, path in enumerate(plan.paths):
        bare = Path(steps=tuple(Step(edge=s.edge, forward=s.forward, role=None)
                                for s in path.steps),
                    target=path.target)
        first = seen.get(bare)
        if first is None:
            first = len(kept)
            seen[bare] = first
            kept.append(bare)
        remap[index] = first
    return tuple(kept), remap

def _title_key_of(plan: Plan, ref: PathRef, onto: Any) -> str | None:
    """The concept's declared display attribute, or None if there is not one."""
    concept = plan.subject if ref == SUBJECT else None
    if concept is None:
        if not 0 <= ref < len(plan.paths):
            return None
        concept = plan.paths[ref].target
    try:
        attribute = onto.concept(concept).title_attribute
    except Exception:                          # noqa: BLE001
        return None
    return getattr(attribute, "name", None)

def canonicalise(plan: Plan, onto: Any = None) -> Plan:
    """The plan, rewritten into its normal form. Compiles identically.

    `onto` is optional and buys exactly one more rewrite: a projection that
    names its concept's display attribute is the same projection as one that
    leaves `attribute` None, because the compiler substitutes the title key for
    None. That equivalence is not decidable from the plan alone -- which is
    itself worth knowing, and is why the parameter exists rather than the rule
    being dropped. Without it the two forms simply get different keys, and the
    invariant stays SOUND either way.
    """
    paths, remap = _canonical_paths(plan)

    def ref(value: PathRef) -> PathRef:
        return remap.get(value, value)

    def projection(node: Projection) -> Projection:
        attribute = node.attribute
        if onto is not None and attribute is not None:
            if attribute == _title_key_of(plan, node.path, onto):
                attribute = None
        return Projection(path=ref(node.path), attribute=attribute)

    def measure(node: Measure) -> Measure:
        parts = tuple(measure(p) for p in node.parts)
        return Measure(governed=node.governed, aggregation=node.aggregation,
                       path=ref(node.path), attribute=node.attribute,
                       combine=node.combine,
                       parts=canonical_operands(node.combine, parts))

    return Plan(
        subject=plan.subject,
        paths=paths,
        project=tuple(projection(p) for p in plan.project),
        measures=tuple(measure(m) for m in plan.measures),
        filters=tuple(Filter(path=ref(f.path), attribute=f.attribute,
                             operator=f.operator, value=f.value)
                      for f in plan.filters),
        having=plan.having,
        top=plan.top,
        set_op=((plan.set_op[0], canonicalise(plan.set_op[1], onto))
                if plan.set_op is not None else None),
    )

def _structure(plan: Plan) -> Any:
    """The canonicalised plan as JSON-able structure. No sets, no `hash()`.

    Everything is a list or a scalar in a fixed order, so the digest is stable
    across processes and `PYTHONHASHSEED` values -- the failure mode that made
    the graph's earlier sort key only accidentally total.
    """
    def path(node: Path) -> Any:
        return [node.target, [[s.edge, s.forward] for s in node.steps]]

    def measure(node: Measure) -> Any:
        return _measure_structure(node, lambda value: value)

    return [
        plan.subject,
        [path(p) for p in plan.paths],
        [[p.path, p.attribute] for p in plan.project],
        [measure(m) for m in plan.measures],
        [[f.path, f.attribute, f.operator, f.value] for f in plan.filters],
        [[h.measure, h.operator, h.value] for h in plan.having],
        (None if plan.top is None else
         [plan.top.by_projection, plan.top.by_measure, plan.top.descending,
          plan.top.n]),
        (None if plan.set_op is None else
         [plan.set_op[0], _structure(plan.set_op[1])]),
    ]

def canonical_key(plan: Plan, onto: Any = None) -> str:
    """A stable identity for a plan.

    Two plans with the same key MUST compile to the same SQL, and the enumerator
    keeps only one of each key. This is the mechanism that makes the normal form
    real rather than aspirational, so it is tested directly: enumerate, group by
    key, and assert that every group compiles to byte-identical SQL.

    The converse -- same SQL implies same key -- is measured in the same test and
    is FALSE; `canonical_gaps` lists the degrees of freedom that survive, each
    one found by generation rather than by inspection.
    """
    payload = json.dumps(_structure(canonicalise(plan, onto)),
                         sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]
