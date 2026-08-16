from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

from sqlglot import exp

from spc.graph import DEFAULT_MAX_HOPS, PathGraph, path_signature
from spc.ontology import Attribute, Ontology
from spc.plan import (
    SUBJECT,
    Filter,
    Having,
    Measure,
    Path,
    Plan,
    Projection,
    Step,
    Top,
    canonical_operands,
    canonical_ref,
)

__all__ = [
    "compile",
    "canonical_paths",
    "role_signatures",
    "CompileError",
    "GrainError",
    "DIALECT",
]

DIALECT = "sqlite"

_ONE = "one"
_MANY = "many"
_MULTIPLICITY = {"none": _ONE, "bounded": _MANY, "multiplicative": _MANY}

_RECOMBINE = {
    "sum": "sum",
    "count": "sum",
    "count_distinct": "sum",
    "min": "min",
    "max": "max",
}

_COMBINE_OF = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}

class CompileError(ValueError):
    """A plan that cannot be lowered. Always names the offending element."""

class GrainError(CompileError):
    """A plan whose number cannot be certified at the requested grain.

    Raised, never emitted. DESIGN rule 5: traversal errors are obvious, grain
    errors are plausible, and a compiler that guesses here ships false numbers.
    """

def _ident(name: str) -> exp.Identifier:
    return exp.to_identifier(name, quoted=True)

def _col(alias: str, column: str) -> exp.Column:
    return exp.Column(this=_ident(column), table=_ident(alias))

def _ref(alias: str) -> exp.Column:
    """A reference to an already-aliased output column of a derived table."""
    return exp.Column(this=_ident(alias))

def _table(name: str, alias: str) -> exp.Table:
    return exp.Table(this=_ident(name), alias=exp.TableAlias(this=_ident(alias)))

def _literal(value: object) -> exp.Expression:
    if value is None:
        return exp.null()
    if isinstance(value, bool):
        return exp.true() if value else exp.false()
    if isinstance(value, (int, float)):
        return exp.Literal.number(value)
    return exp.Literal.string(str(value))

def _eq(left: exp.Expression, right: exp.Expression) -> exp.EQ:
    return exp.EQ(this=left, expression=right)

def _all(conditions: Sequence[exp.Expression]) -> exp.Expression | None:
    if not conditions:
        return None
    node = conditions[0]
    for extra in conditions[1:]:
        node = exp.And(this=node, expression=extra)
    return node

_AGG_NODES = {
    "sum": exp.Sum,
    "count": exp.Count,
    "avg": exp.Avg,
    "min": exp.Min,
    "max": exp.Max,
}

def _aggregate(kind: str, operand: exp.Expression) -> exp.Expression:
    if kind == "count_distinct":
        return exp.Count(this=exp.Distinct(expressions=[operand]))
    try:
        node = _AGG_NODES[kind]
    except KeyError:
        raise CompileError(f"unknown aggregation {kind!r}") from None
    return node(this=operand)

def _safe_divide(numerator: exp.Expression, denominator: exp.Expression) -> exp.Expression:
    """Division that neither truncates nor raises.

    Two defects in one node. ACME's LossRatio returned 0 because 13600/20000 is
    INTEGER division in SQLite (DESIGN rule 2: the storage type is a fact, and
    the fact is that these columns are integers), so the numerator is CAST to
    REAL. And a zero denominator is a data condition, not a plan error, so it
    yields NULL instead of a failed query.
    """
    guarded = exp.Cast(this=numerator, to=exp.DataType.build("REAL", dialect=DIALECT))
    division = exp.Div(this=guarded, expression=denominator)
    return exp.Case(
        ifs=[exp.If(this=_eq(denominator, exp.Literal.number(0)), true=exp.null())],
        default=division,
    )

_ARITHMETIC = {"+": exp.Add, "-": exp.Sub, "*": exp.Mul}

@dataclass(frozen=True)
class _Node:
    """One aliased source in the join tree.

    `multiplicity` is the declared row multiplication of the traversal that
    REACHED this node from its parent -- the only fan-out fact the compiler
    consults, and it comes from the ontology, never from the data.

    `converges` is that same declaration read the OTHER way: many of the parent's
    rows may lead to ONE of this node's rows. On its own it is harmless. Under a
    step that already multiplied it is the shape that makes one row REACHABLE
    TWICE from one subject row -- see `_Layout.may_revisit`.
    """

    alias: str
    table: str
    parent: str | None
    on: tuple[tuple[str, str], ...]
    where: tuple[tuple[str, str], ...]
    multiplicity: str
    concept: str | None
    requires: tuple[str, ...] = ()
    converges: bool = False

    step: int = 0

class _Layout:
    """Alias allocation and the join tree, built once per compilation.

    Two paths that share a prefix share the aliases of that prefix -- the prefix
    IS the identity of a traversal -- so `CLAIMED_AGAINST` followed by two
    different edges joins the coverage detail once and branches from it. Two
    paths that reach the same concept by DIFFERENT edges get different aliases,
    which is what makes four amount kinds four sources rather than one source
    required to be all four kinds at once.
    """

    def __init__(self, onto: Ontology, subject: str) -> None:
        self._onto = onto
        self.nodes: dict[str, _Node] = {}
        self._landing: dict[tuple[tuple[str, bool], ...], tuple[str, str]] = {}
        self._resolved: dict[tuple[str, str], str] = {}
        self._step = 0
        concept = onto.concept(subject)
        self.root = self._allocate(concept.table)
        self.nodes[self.root] = _Node(
            alias=self.root,
            table=concept.table,
            parent=None,
            on=(),
            where=concept.backed_where,
            multiplicity=_ONE,
            concept=subject,
        )
        self._landing[()] = (self.root, subject)

    def _allocate(self, table: str) -> str:
        base = table.lower()
        if base not in self.nodes:
            return base
        index = 2
        while f"{base}_{index}" in self.nodes:
            index += 1
        return f"{base}_{index}"

    def _add(self, node: _Node) -> str:
        self.nodes[node.alias] = node
        return node.alias

    def _bump(self) -> int:
        self._step += 1
        return self._step

    def walk(self, path: Path) -> str:
        """Allocate (or reuse) every alias `path` needs; return its landing."""
        prefix: tuple[tuple[str, bool], ...] = ()
        alias, concept = self.root, self.nodes[self.root].concept or ""
        for step in path.steps:
            prefix = prefix + ((step.edge, step.forward),)
            seen = self._landing.get(prefix)
            if seen is not None:
                alias, concept = seen
                continue
            alias, concept = self._traverse(alias, concept, step)
            self._landing[prefix] = (alias, concept)
        return alias

    def _traverse(self, origin_alias: str, origin_concept: str, step: Step) -> tuple[str, str]:
        edge = self._onto.edge(step.edge)
        source = edge.origin(forward=step.forward)
        target = edge.endpoint(forward=step.forward)
        if source != origin_concept:
            raise CompileError(
                f"edge {edge.name!r} read {'forward' if step.forward else 'reverse'} leaves "
                f"{source!r}, not {origin_concept!r} -- the path is not connected"
            )
        landing = self._onto.concept(target)
        multiplicity = _MULTIPLICITY[edge.fan_out_in(forward=step.forward)]

        converges = _MULTIPLICITY[edge.fan_out_in(forward=not step.forward)] == _MANY
        self._step += 1
        ordinal = self._step

        parent = origin_alias
        if edge.is_junction:
            junction = self._allocate(edge.via_table or "")
            if step.forward:
                junction_on = tuple((s, j) for s, j in edge.via_from_join)
                landing_on = tuple((j, t) for j, t in edge.via_to_join)
            else:
                junction_on = tuple((t, j) for j, t in edge.via_to_join)
                landing_on = tuple((j, s) for s, j in edge.via_from_join)
            self._add(_Node(
                alias=junction,
                table=edge.via_table or "",
                parent=origin_alias,
                on=junction_on,
                where=edge.role_predicate,
                multiplicity=multiplicity,
                concept=None,
                step=ordinal,
            ))
            parent, multiplicity = junction, _ONE
        else:
            if step.forward:
                landing_on = tuple((s, t) for s, t in edge.join)
            else:
                landing_on = tuple((t, s) for s, t in edge.join)

        alias = self._allocate(landing.table)
        requires: tuple[str, ...] = ()
        node = _Node(
            alias=alias,
            table=landing.table,
            parent=parent,
            on=landing_on,
            where=landing.backed_where,
            multiplicity=multiplicity,
            concept=target,
            converges=converges,
            step=ordinal,
        )
        self._add(node)

        if edge.restrict_table:

            dst_alias = alias if step.forward else origin_alias
            restrict = self._allocate(edge.restrict_table)
            self._add(_Node(
                alias=restrict,
                table=edge.restrict_table,
                parent=dst_alias,
                on=tuple((d, r) for d, r in edge.restrict_columns),
                where=(),
                multiplicity=_ONE,
                concept=None,
                step=ordinal,
            ))
            requires = (restrict,)
            self.nodes[alias] = replace(node, requires=requires)

        return alias, target

    def resolve(self, alias: str, concept: str, attribute: Attribute) -> tuple[str, str]:
        """Where an attribute physically lives, following `via` resolution.

        A role object's `name` is Foundry interface-property resolution: the value
        is on Party, reached through IS_PARTY_*. The extra hop is allocated here
        rather than being something a planner must remember to request.
        """
        if attribute.column is not None:
            return alias, attribute.column
        edge = self._onto.edge(attribute.via or "")
        forward = edge.source == concept
        far = edge.target if forward else edge.source
        key = (alias, edge.name)
        landed = self._resolved.get(key)
        if landed is None:
            far_concept = self._onto.concept(far)
            if edge.is_junction:
                raise CompileError(
                    f"resolved attribute {attribute.qualified} uses junction edge "
                    f"{edge.name!r}; only direct edges resolve attributes"
                )
            if forward:
                on = tuple((s, t) for s, t in edge.join)
            else:
                on = tuple((t, s) for s, t in edge.join)
            landed = self._allocate(far_concept.table)
            self._add(_Node(
                alias=landed,
                table=far_concept.table,
                parent=alias,
                on=on,
                where=far_concept.backed_where,
                multiplicity=_MULTIPLICITY[edge.fan_out_in(forward=forward)],
                concept=far,
                converges=_MULTIPLICITY[edge.fan_out_in(forward=not forward)] == _MANY,
                step=self._bump(),
            ))
            self._resolved[key] = landed
        far_attribute = self._onto.concept(far).attribute(attribute.via_attribute or "")
        return self.resolve(landed, far, far_attribute)

    def closure(self, aliases: Iterable[str]) -> list[str]:
        """Every alias needed to bring `aliases` into a query, in join order."""
        needed: set[str] = set()
        stack = list(aliases)
        while stack:
            alias = stack.pop()
            if alias in needed:
                continue
            needed.add(alias)
            node = self.nodes[alias]
            if node.parent is not None:
                stack.append(node.parent)
            stack.extend(node.requires)
        return [a for a in self.nodes if a in needed]

    def ancestry(self, alias: str) -> list[str]:
        chain = [alias]
        node = self.nodes[alias]
        while node.parent is not None:
            chain.append(node.parent)
            node = self.nodes[node.parent]
        return list(reversed(chain))

    def fans_out(self, alias: str) -> bool:
        return any(self.nodes[a].multiplicity == _MANY for a in self.ancestry(alias)[1:])

    def may_revisit(self, alias: str) -> bool:
        """Can ONE subject row reach ONE of `alias`'s rows more than once?

        Fan-out alone never does that: a claim's four loss payments are four
        DIFFERENT rows, and summing them is the answer. Duplication needs two
        declarations to meet -- an earlier traversal that multiplies, and a later
        one that CONVERGES, so several of the rows the first produced lead back to
        the same row. `Policy -COVERED_BY-> PolicyCoverageDetail` multiplies, and
        `CLAIMED_AGAINST` read backwards converges (ACME declares it bounded in
        both directions), so a policy reaches one claim once per coverage detail
        and a naive SUM adds that claim's amounts once per detail.

        "Earlier" is per TRAVERSAL, not per node, which is why `_Node.step`
        exists: a many-to-many junction edge multiplies and converges inside its
        own single step (`Party -SOLD-> Policy`) without ever reaching one policy
        twice, because a junction row IS the pair. Counting that as duplication
        would refuse most of the ontology.

        Read from the ontology's declarations, never from the data -- the same
        commitment as `fans_out` (DESIGN rule 2).
        """
        fanned: set[int] = set()
        for name in self.ancestry(alias)[1:]:
            node = self.nodes[name]
            if node.converges and any(step < node.step for step in fanned):
                return True
            if node.multiplicity == _MANY:
                fanned.add(node.step)
        return False

    def peak(self, left: str, right: str) -> str | None:
        """The node where two branches BOTH multiply, or None.

        Below such a node neither side's row determines the other's, so every
        combined row is one arbitrary pairing -- an accidental cross product. If
        one alias is an ancestor of the other there is no peak: listing a policy
        beside each of its claims is what a join is FOR.
        """
        left_chain, right_chain = self.ancestry(left), self.ancestry(right)
        shared = 0
        while (shared < len(left_chain) and shared < len(right_chain)
               and left_chain[shared] == right_chain[shared]):
            shared += 1
        if shared == len(left_chain) or shared == len(right_chain):
            return None
        left_child, right_child = left_chain[shared], right_chain[shared]
        if (self.nodes[left_child].multiplicity == _MANY
                and self.nodes[right_child].multiplicity == _MANY):
            return left_chain[shared - 1]
        return None

@dataclass(frozen=True)
class _Leaf:
    aggregation: str
    alias: str
    column: str
    label: str

    dedupe: tuple[tuple[str, str], ...] = ()

@dataclass(frozen=True)
class _Composite:
    operator: str
    parts: tuple[object, ...]

@dataclass(frozen=True)
class _Over:
    """An aggregate applied to a composition AT THE GRAIN IT IS FORMED ON.

    `sum` distributes over a composition, so a total needs none of this:
    SUM(a)+SUM(b) is SUM(a+b) and the ordinary satellites give the right answer.
    `avg` does not distribute -- the average of a+b+c+d over claims is not the
    sum of four averages -- so the composition has to be formed per row FIRST
    and aggregated afterwards. `inner` is the composite; `aggregation` is what
    to do with its per-row values.
    """

    aggregation: str
    inner: _Composite

class _Compiler:
    def __init__(self, plan: Plan, onto: Ontology, graph: PathGraph) -> None:
        self.plan = plan
        self.onto = onto
        self.graph = graph
        self.subject = onto.concept(plan.subject)

        self.ref = canonical_ref(plan)

        self._per_keyed: set[str] = set()
        self._force_per_keyed = False
        self.paths: list[Path] = [self._govern(p) for p in plan.paths]
        self.layout = _Layout(onto, plan.subject)
        self.path_alias: list[str] = [self.layout.walk(p) for p in self.paths]
        self.filters: list[tuple[str, exp.Expression]] = []
        self._applied: set[int] = set()

        self.group_keys: tuple[tuple[str, str], ...] = ()
        self._satellite_index: dict[tuple, str] = {}
        self._satellite_selects: list[tuple[str, exp.Select]] = []

        self._grain_selects: list[tuple[str, exp.Select]] = []
        self.roles: tuple[tuple[str, ...], ...] = tuple(
            graph.role_signature(plan.subject, p) for p in self.paths
        )

    def _govern(self, path: Path) -> Path:
        """Re-derive the path's roles from the ontology and prove it governed.

        Role commitment is INTRINSIC (DESIGN rule 3), so whatever a caller put in
        `Step.role` is discarded and re-read here; then the canonicalised path
        must appear in the enumerator's own output, which is what "governed by
        construction" means operationally.
        """
        concept = self.plan.subject
        steps: list[Step] = []
        for step in path.steps:
            edge = self.onto.edge(step.edge)
            if edge.origin(forward=step.forward) != concept:
                raise CompileError(
                    f"path {path.target!r}: edge {edge.name!r} does not leave {concept!r}"
                )
            concept = edge.endpoint(forward=step.forward)
            landing = self.onto.concept(concept)
            codes = [c for c in (edge.role_code, landing.role_code) if c]
            ordered: list[str] = []
            for code in codes:
                if code not in ordered:
                    ordered.append(code)
            steps.append(Step(edge=edge.name, forward=step.forward,
                              role="+".join(ordered) if ordered else None))
        if concept != path.target:
            raise CompileError(
                f"path declares target {path.target!r} but lands on {concept!r}"
            )
        canonical = Path(steps=tuple(steps), target=path.target)
        if canonical.steps:
            governed = self.graph.paths(
                self.plan.subject, path.target, max_hops=max(len(canonical.steps), 1)
            )
            if canonical not in governed:
                raise CompileError(
                    f"path {path_signature(canonical)} is not a governed route from "
                    f"{self.plan.subject!r}"
                )
        return canonical

    def _concept_of(self, ref: int) -> str:
        if ref == SUBJECT:
            return self.plan.subject
        try:
            return self.paths[ref].target
        except IndexError:
            raise CompileError(f"path index {ref} is out of range") from None

    def _alias_of(self, ref: int) -> str:
        if ref == SUBJECT:
            return self.layout.root
        try:
            return self.path_alias[ref]
        except IndexError:
            raise CompileError(f"path index {ref} is out of range") from None

    def _attribute(self, ref: int, name: str | None) -> tuple[str, str, Attribute]:
        concept_name = self._concept_of(ref)
        concept = self.onto.concept(concept_name)
        if name is None:
            attribute = concept.title_attribute
            if attribute is None:
                raise CompileError(
                    f"concept {concept_name!r} declares no display attribute, so a bare "
                    f"projection has no meaning"
                )
        else:
            attribute = concept.attribute(name)
        alias, column = self.layout.resolve(self._alias_of(ref), concept_name, attribute)
        return alias, column, attribute

    def _expand(self, measure: Measure) -> object:
        stated = [measure.governed is not None,
                  measure.aggregation is not None,
                  measure.combine is not None]
        if sum(stated) != 1:
            raise CompileError(
                "a measure sets exactly one of `governed`, `aggregation`, `combine`"
            )
        if measure.governed is not None and measure.over:
            inner = self._governed(measure.governed, measure.path)
            if not isinstance(inner, _Composite):
                raise CompileError(
                    f"`over` applies to a composition; {measure.governed!r} is a "
                    f"single aggregation and takes `aggregation` instead")
            return _Over(aggregation=measure.over, inner=inner)
        if measure.combine is not None:
            if len(measure.parts) < 2:
                raise CompileError(
                    "a one-part composite is not representable -- it IS the bare measure"
                )
            parts = canonical_operands(measure.combine, measure.parts, self.ref)
            return _Composite(measure.combine,
                              tuple(self._expand(p) for p in parts))
        if measure.aggregation is not None:
            return self._leaf(measure.aggregation, measure.path, measure.attribute,
                              label=f"{measure.aggregation}({measure.attribute})")
        return self._governed(measure.governed or "", measure.path)

    def _governed(self, name: str, ref: int) -> object:
        metric = self.onto.metric(name)
        if metric.is_composite:
            operator = _COMBINE_OF[metric.combine or ""]
            if ref == SUBJECT:

                return _Composite(operator, tuple(
                    self._governed(component, SUBJECT)
                    for component in metric.components))
            base = self._measured_over(name)
            if not 0 <= ref < len(self.paths):
                raise CompileError(f"path index {ref} is out of range")
            path = self.paths[ref]
            if path.target != base:
                raise CompileError(
                    f"composite metric {name!r} is measured over {base!r} -- that is "
                    f"where its components' own edges leave from -- but path {ref} "
                    f"lands on {path.target!r}. A composite's route is the route to "
                    f"the concept it measures over, NOT to a component's operand"
                )
            return _Composite(operator, tuple(
                self._component(component, ref) for component in metric.components))
        operand = metric.operand
        if operand is None:                                  # pragma: no cover
            raise CompileError(f"metric {name!r} has neither operand nor components")
        target = self._route(name, operand.concept, operand.via, ref)
        return self._leaf(metric.op or "", target, operand.attribute, label=name,
                          grain_of=self._grain_alias(name, ref, target))

    def _grain_alias(self, metric: str, ref: int, target: int) -> str | None:
        """Where a GOVERNED metric's route is deduplicated, or None.

        The number a plan asks for is the metric's own quantity aggregated over
        the instances its route reaches -- each of them ONCE. Which instances
        those are is `Ontology.measured_over(metric)`, and that concept's layout
        alias lies on the route, so the prefix is found by walking back from
        where the route landed.

        WHY `ref` NO LONGER MATTERS (changed 2026-08-12). This used to return
        None for `ref == SUBJECT`, i.e. whenever the compiler DERIVED the route
        rather than the plan naming one, on the reasoning that a plan is only
        answerable for a route it stated. That reasoning was wrong, and wrong in
        the direction the project exists to prevent: `Plan(subject=Policy,
        TotalLoss)` -- no route named, the DEFAULT path, what all 24 gold plans
        and every realistic pick do -- read 26900 on `acme_cf_fanout` where
        claim-counted-once is 19200, because ACME declares `CLAIMED_AGAINST`
        bounded in BOTH directions and a policy reaches one claim once per
        coverage detail. A derived route is still a route, and the concept the
        metric is a quantity of does not depend on who chose the way there. Both
        cases now deduplicate.

        The correction is grain-only: it rewrites the SQL of 11 of the 24 gold
        plans and changes the RESULT of none of them, on `acme_N` or on
        `acme_cf_fanout` (measured before and after -- results/DECISIONS.md).
        It does move the determinism digest, which is a tripwire for UNINTENDED
        change; this change is intended, and both digests are recorded.

        None means "nothing to deduplicate at": either the metric is a quantity
        of more than one concept (`LossRatio`), or the measured-over concept is
        not on the route -- and note that `_leaf` only consults this at all when
        the operand does fan out relative to the subject.
        """
        bases = self.onto.measured_over(metric)
        if len(bases) != 1:
            return None
        landing = self._alias_of(target)
        for alias in reversed(self.layout.ancestry(landing)):
            if self.layout.nodes[alias].concept == bases[0]:
                return alias
        return None

    def _measured_over(self, name: str) -> str:
        """The one concept a composite is a quantity of, or a refusal.

        `Ontology.measured_over` owns the semantics -- the same answer that
        `spc/skills.py` shows a model, so the advice and the enforcement are one
        fact. Here it is only turned into a `CompileError` when the composite
        turns out to be a quantity of MORE THAN ONE concept, as `LossRatio` is:
        no single route can name where such a metric is measured, so a plan that
        supplies one is refused rather than resolved to one of the two.
        """
        bases = self.onto.measured_over(name)
        if len(bases) != 1:
            raise CompileError(
                f"composite metric {name!r} is measured over more than one concept "
                f"({', '.join(bases)}), so a route to it is not well defined; measure "
                f"it from the SUBJECT and let each component take its own route"
            )
        return bases[0]

    def _component(self, name: str, over: int) -> object:
        """One component of a composite that the plan routed explicitly.

        The plan's route reaches the concept the composite is measured over; the
        component's own `via` -- the edge that IS its identity -- is appended.
        Four amount kinds over one Agent->Claim route are four four-hop routes
        that share their first three hops, not one route four ways.
        """
        metric = self.onto.metric(name)
        if metric.is_composite:
            return _Composite(_COMBINE_OF[metric.combine or ""], tuple(
                self._component(component, over) for component in metric.components))
        operand = metric.operand
        if operand is None:                                  # pragma: no cover
            raise CompileError(f"metric {name!r} has neither operand nor components")
        target = self._extend(over, operand.via, operand.concept, metric=name)
        return self._leaf(metric.op or "", target, operand.attribute, label=name,
                          grain_of=self._alias_of(over))

    def _extend(self, over: int, via: str | None, concept: str, *, metric: str) -> int:
        """`paths[over]` followed by `via`, governed, deduplicated, walked."""
        prefix = self.paths[over]
        if via is None:
            if prefix.target != concept:                     # pragma: no cover
                raise CompileError(
                    f"metric {metric!r} measures {concept!r}; path {over} lands on "
                    f"{prefix.target!r}"
                )
            return over
        composed = self.graph.extend(self.plan.subject, prefix, via)
        if composed is None or composed.target != concept:
            raise CompileError(
                f"metric {metric!r} is defined over edge {via!r}, and path {over} "
                f"({path_signature(prefix)}) does not compose with it into a governed "
                f"route to {concept!r}"
            )
        extended = self._govern(composed)
        for index, existing in enumerate(self.paths):
            if existing == extended:
                return index
        self.paths.append(extended)
        self.path_alias.append(self.layout.walk(extended))
        self.roles = self.roles + (
            self.graph.role_signature(self.plan.subject, extended),)
        return len(self.paths) - 1

    def _route(self, metric: str, concept: str, via: str | None, ref: int) -> int:
        """Which governed path carries this metric's operand.

        The metric names its LAST edge (the amount kind); the route from the
        subject to that edge is a path, and a path is the plan's one degree of
        freedom -- so an explicit `Measure.path` wins, a path the plan already
        declared is reused, and only then is one derived. Derivation takes the
        first candidate in the enumerator's TOTAL order and refuses when the
        shortest length is reached by more than one route, because silently
        picking among equals is how a plan stops being a function of the question.
        """
        if ref != SUBJECT:
            path = self.paths[ref] if ref < len(self.paths) else None
            if path is None:
                raise CompileError(f"path index {ref} is out of range")
            if path.target != concept:

                bases = self.onto.measured_over(metric)
                if via is not None and len(bases) == 1 and path.target == bases[0]:
                    return self._extend(ref, via, concept, metric=metric)
                raise CompileError(
                    f"metric {metric!r} measures {concept!r}; path {ref} does not land there"
                )
            if via is not None and (not path.steps or path.steps[-1].edge != via):
                raise CompileError(
                    f"metric {metric!r} is defined over edge {via!r}; path {ref} arrives by "
                    f"{path.steps[-1].edge if path.steps else 'no edge'}"
                )
            return ref
        if concept == self.plan.subject and via is None:
            return SUBJECT
        for index, path in enumerate(self.paths):
            if path.target != concept:
                continue
            if via is None or (path.steps and path.steps[-1].edge == via):
                return index
        candidates = [
            p for p in self.graph.paths(self.plan.subject, concept, max_hops=DEFAULT_MAX_HOPS)
            if via is None or (p.steps and p.steps[-1].edge == via)
        ]
        if not candidates:
            raise CompileError(
                f"metric {metric!r} needs a governed route from {self.plan.subject!r} to "
                f"{concept!r}" + (f" ending in {via!r}" if via else "") + ", and none exists"
            )
        shortest = [p for p in candidates if len(p.steps) == len(candidates[0].steps)]
        if len(shortest) > 1:
            raise CompileError(
                f"metric {metric!r} has {len(shortest)} equally short governed routes to "
                f"{concept!r}; the plan must name one "
                f"({', '.join(path_signature(p) for p in shortest)})"
            )
        self.paths.append(candidates[0])
        self.path_alias.append(self.layout.walk(candidates[0]))
        self.roles = self.roles + (
            self.graph.role_signature(self.plan.subject, candidates[0]),)
        return len(self.paths) - 1

    def _leaf(self, aggregation: str, ref: int, attribute: str | None, *, label: str,
              grain_of: str | None = None) -> _Leaf:
        if aggregation not in _RECOMBINE and aggregation != "avg":
            raise CompileError(f"unknown aggregation {aggregation!r}")
        alias, column, attr = self._attribute(ref, attribute)
        fans_out = self.layout.fans_out(alias)
        if fans_out and grain_of is None and self.layout.may_revisit(alias):

            raise GrainError(
                f"{label}: the route to {attr.qualified} multiplies and then converges, "
                f"so it reaches one row more than once, and this measure names no "
                f"governed metric -- nothing declares what quantity it is OF, so there "
                f"is no grain to count each row once at. Refused rather than summed "
                f"over duplicated rows. Name a governed metric, or route to the "
                f"operand without passing through the converging traversal."
            )
        if fans_out and aggregation == "avg":

            raise GrainError(
                f"{label}: AVG does not decompose across the fan-out on the route to "
                f"{attr.qualified}. An average of per-subject averages is not the "
                f"average, so it is refused rather than approximated -- express it as a "
                f"sum divided by a count."
            )
        if fans_out and aggregation == "count_distinct" and not self._is_identifier(ref, attr):

            raise GrainError(
                f"{label}: COUNT(DISTINCT {attr.qualified}) cannot be recombined across "
                f"the fan-out on its route, because the same value may appear under two "
                f"subjects and would be counted twice. Only an identifying attribute "
                f"composes."
            )
        return _Leaf(aggregation=aggregation, alias=alias, column=column,
                     label=label, dedupe=self._grain_keys(grain_of, ref, label=label))

    def _grain_keys(self, over: str | None, operand: int, *, label: str
                    ) -> tuple[tuple[str, str], ...]:
        """The columns that identify ONE operand row of this measure.

        Two things can make the join tree hand the same operand row to a
        satellite twice, and one set of columns answers both:

        THE MEASURE'S OWN ROUTE may reach the concept the metric is a quantity
        OF more than once -- ACME declares `CLAIMED_AGAINST` bounded in both
        directions, so one Claim may hang off several coverage details of one
        policy, and a route through those details reaches that claim once per
        detail.

        THE GROUP KEYS bring their own traversals into the satellite, and those
        are not part of the measured quantity at all: a dimension that fans out
        beside the operand multiplies it for no reason a reader could see.

        Both are removed by naming the key of the measured-over concept and the
        key of the operand's own concept and deduplicating on them (together
        with the group keys) before aggregating. `over` is None when nothing
        declares what quantity the measure is OF -- a raw
        `Measure(aggregation=...)` -- and then the operand row's own key is all
        there is; that is enough for the group-key artefact, and the case where
        the measure's OWN route revisits is refused in `_leaf` rather than
        decided here. Where a concept declares no key there is nothing to
        deduplicate BY, and the plan is refused rather than answered over
        multiplied rows.
        """
        pairs: list[tuple[str, str]] = []
        aliases = [self._alias_of(operand)] if over is None else [over, self._alias_of(operand)]
        for alias in aliases:
            name = self.layout.nodes[alias].concept or ""
            concept = self.onto.concept(name)
            if not concept.key:
                raise GrainError(
                    f"{label}: the rows {name!r} contributes to this measure cannot be "
                    f"counted once, because {name!r} declares no key. Refused rather "
                    f"than summed over rows the join tree may have duplicated."
                )
            for column in concept.key:
                pair = (alias, column)
                if pair not in pairs:
                    pairs.append(pair)
        return tuple(pairs)

    def _is_identifier(self, ref: int, attribute: Attribute) -> bool:
        concept = self.onto.concept(self._concept_of(ref))
        return attribute.column in set(concept.key) or attribute.value_type == "Identifier"

    def _predicate(self, spec: Filter) -> tuple[str, exp.Expression]:
        alias, column, _attribute = self._attribute(spec.path, spec.attribute)
        column_ref = _col(alias, column)
        operator = spec.operator
        if operator == "BETWEEN":
            bounds = spec.value
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise CompileError("BETWEEN needs exactly two bounds")
            return alias, exp.Between(this=column_ref,
                                      low=_literal(bounds[0]), high=_literal(bounds[1]))
        if operator == "IN":
            values = spec.value
            if not isinstance(values, (list, tuple)) or not values:
                raise CompileError("IN needs a non-empty list of values")
            return alias, exp.In(this=column_ref,
                                 expressions=[_literal(v) for v in values])
        node = {
            "=": exp.EQ, "!=": exp.NEQ, "<": exp.LT, "<=": exp.LTE,
            ">": exp.GT, ">=": exp.GTE, "LIKE": exp.Like,
        }.get(operator)
        if node is None:
            raise CompileError(f"unknown operator {operator!r}")
        return alias, node(this=column_ref, expression=_literal(spec.value))

    def _source(self, aliases: Sequence[str]) -> exp.Select:
        """A SELECT over exactly `aliases`, joined in allocation order."""
        select = exp.Select()
        root = self.layout.nodes[aliases[0]]
        select = select.from_(_table(root.table, root.alias), copy=False)
        root_conditions = [_eq(_col(root.alias, c), _literal(v)) for c, v in root.where]
        for alias in aliases[1:]:
            node = self.layout.nodes[alias]
            conditions = [_eq(_col(node.parent or "", parent_column), _col(alias, own))
                          for parent_column, own in node.on]
            conditions += [_eq(_col(alias, c), _literal(v)) for c, v in node.where]
            select = select.join(
                _table(node.table, alias),
                on=_all(conditions),
                join_type="INNER",
                copy=False,
            )
        present = set(aliases)
        for index, (alias, condition) in enumerate(self.filters):
            if alias in present:
                root_conditions.append(condition)
                self._applied.add(index)
        combined = _all(root_conditions)
        if combined is not None:
            select = select.where(combined, copy=False)
        return select

    def _group_key_columns(
        self, dimensions: Sequence[tuple[str, str, Attribute]]
    ) -> tuple[tuple[str, str], ...]:
        """The identity of one output group: the projected columns themselves.

        The columns, NOT the keys of the concepts they belong to. Keying on the
        concept keys was tried and is wrong in the direction this whole change
        exists to close: a satellite keyed on the entity and an outer query
        grouped on a DISPLAY column means several entities can land in one
        group, and their subtotals are then added -- correct while the operand
        rows are disjoint, a double count the moment they are not (two coverage
        details priced the same, whose claims overlap). Grouping the satellite
        on the value the answer actually groups by makes each operand row
        belong to a group ONCE, whatever the entity structure above it.

        Deduplicated, because two projections may resolve to one column, and an
        output group is identified by the set of values, not by how many times
        the plan named them.

        With no projections this is empty, and the answer is one row: the
        satellites are single-row relations and there is nothing to key them on.
        """
        pairs: list[tuple[str, str]] = []
        for alias, column, _attribute in dimensions:
            pair = (alias, column)
            if pair not in pairs:
                pairs.append(pair)
        return tuple(pairs)

    def _satellite(self, leaf: _Leaf, *, per_keyed: bool = False) -> str:
        """Pre-aggregate one measure AT THE OUTPUT GRAIN, in two stages.

        THE RULE THIS EXISTS TO ENFORCE (DESIGN 9.4). Three grains are in play:
        the measure's own (`Ontology.measured_over`), the output's (the group
        keys) and the subject's. Until 2026-08-12 a satellite was keyed on the
        SUBJECT key, which is neither of the first two, and that is eager
        aggregation (Yan & Larson, VLDB 1995) without its correctness
        precondition -- the pushed-down grouping columns must functionally
        determine the join key. It failed in both directions and both were
        executed:

            output FINER than the measure   subject=Policy, dims=(policy,
                claim), TotalLoss read 13600 beside EVERY claim of the policy
                (the policy total repeated) where the claims are 4600 and 9000.
            output COARSER than the measure  subject=Claim, dim=policy,
                governed LossRatio read 0.34, because one coverage detail's
                20000 premium was added once per claim -- a 40000 denominator
                for a policy whose premium is 20000. The truth is 0.68.

        Keying on the group keys fixes both without a rule about question
        shapes: where the output is finer, the measure is recomputed at that
        finer grain because the finer key is part of the satellite's own GROUP
        BY; where it is coarser, each operand row is deduplicated once per
        group before anything is summed, so a row several subjects share is
        added once.

        THREE STAGES, all CTEs because `spc/check.py` reads CTEs and must be
        able to certify what is emitted:

            __g<i>  THE BRIDGE. SELECT DISTINCT (group keys, measured-over key,
                    operand key) -- which operand rows belong to which output
                    group, each pairing once. No VALUE column. This stage used
                    to exist only for a governed metric whose route could
                    revisit; it now runs for EVERY measure, because the group
                    keys bring traversals of their own into the satellite and
                    those can duplicate the operand just as a converging route
                    can.
            __o<i>  THE MEASURE AT ITS OWN GRAIN. SELECT DISTINCT (measured-over
                    key, operand key, value). One row per operand row, and
                    nothing else in it.
            __m<i>  SELECT group keys, agg(value) FROM bridge JOIN operand ON
                    the operand identity GROUP BY group keys.

        WHY THE VALUE IS NOT IN THE BRIDGE, which is the whole reason there are
        three stages and not two. A bridge relates a dimension to an operand
        across a fan-out, so its rows ARE an arbitrary pairing in the row sense
        -- a claim's loss payment beside its policy's premium. That is exactly
        what `check.py`'s ROW_FANOUT rule exists to reject, and it is right to:
        such a row is only meaningful as an association, never as a reported
        value. Carrying the amount in the same relation made the compiler emit
        a row the checker had to be TAUGHT to accept, and the first attempt at
        teaching it (exempting any source that also projects its own key) put a
        hole in the rule big enough to certify a genuine 20-row cross product
        of loss amounts against premium amounts. Splitting the stages removes
        the need for the exemption instead of arguing for it: the bridge
        projects identities only, so no pair of its sources is two quantities,
        and the rule goes back to what it was.

        AVG is lowered as a SUM and a COUNT so the outer query can recombine it
        as SUM(sum)/SUM(count); an average of averages is never formed.
        """

        per_keyed = per_keyed or getattr(self, "_force_per_keyed", False)
        signature = (leaf.aggregation, leaf.alias, leaf.column, leaf.dedupe,
                     per_keyed)
        existing = self._satellite_index.get(signature)
        if existing is not None:
            return existing
        name = f"__m{len(self._satellite_selects)}"

        bridge_aliases = self.layout.closure(
            [self.layout.root, leaf.alias]
            + [alias for alias, _column in self.group_keys]
            + [alias for alias, _condition in self.filters])
        bridge_select = self._source(bridge_aliases)
        bridge_projections: list[exp.Expression] = [
            exp.alias_(_col(alias, column), _ident(f"__k{i}"))
            for i, (alias, column) in enumerate(self.group_keys)
        ]
        bridge_projections += [exp.alias_(_col(alias, column), _ident(f"__g{i}"))
                               for i, (alias, column) in enumerate(leaf.dedupe)]
        bridge = f"__g{len(self._grain_selects)}"
        self._grain_selects.append(
            (bridge, bridge_select.select(*bridge_projections, copy=False)
             .distinct(copy=False)))

        operand_aliases = self.layout.closure([self.layout.root, leaf.alias])
        operand_select = self._source(operand_aliases)
        operand_projections: list[exp.Expression] = [
            exp.alias_(_col(alias, column), _ident(f"__g{i}"))
            for i, (alias, column) in enumerate(leaf.dedupe)
        ]
        operand_projections.append(
            exp.alias_(_col(leaf.alias, leaf.column), _ident("__u")))
        operand = f"__o{len(self._grain_selects)}"
        self._grain_selects.append(
            (operand, operand_select.select(*operand_projections, copy=False)
             .distinct(copy=False)))

        select = exp.Select().from_(_table(bridge, bridge), copy=False)
        select = select.join(
            _table(operand, operand),
            on=_all([_eq(_col(bridge, f"__g{i}"), _col(operand, f"__g{i}"))
                     for i in range(len(leaf.dedupe))]),
            join_type="INNER", copy=False)
        key_refs = [_col(bridge, f"__k{i}") for i in range(len(self.group_keys))]
        value = _col(operand, "__u")
        projections: list[exp.Expression] = [
            exp.alias_(reference, _ident(f"__k{i}"))
            for i, reference in enumerate(key_refs)
        ]
        if per_keyed:
            if not leaf.dedupe:
                raise GrainError(
                    "a per-grain aggregate needs the measured-over key, and this "
                    "operand declares no dedupe key to carry it")
            key_refs = key_refs + [_col(bridge, "__g0")]
            projections.append(exp.alias_(_col(bridge, "__g0"), _ident("__p")))
            self._per_keyed.add(name)
        if leaf.aggregation == "avg":
            projections.append(exp.alias_(_aggregate("sum", value), _ident("__v")))
            projections.append(exp.alias_(_aggregate("count", value), _ident("__n")))
        else:
            projections.append(
                exp.alias_(_aggregate(leaf.aggregation, value), _ident("__v")))
        select = select.select(*projections, copy=False)
        if key_refs:
            select = select.group_by(*key_refs, copy=False)
        self._satellite_index[signature] = name
        self._satellite_selects.append((name, select))
        return name

    def build(self) -> exp.Expression:
        for spec in self.plan.filters:
            self.filters.append(self._predicate(spec))
        measures = [self._expand(m) for m in self.plan.measures]
        tree = self._listing() if not measures else self._aggregation(measures)
        self._check_filters_applied()
        return tree

    def _check_filters_applied(self) -> None:
        """A predicate that reached no sub-query is a SILENTLY DROPPED filter.

        It happens in a LISTING (no measures) when a filter names a route the plan
        declares but does not project: the route is never joined, so the predicate
        has nowhere to land. Joining it would turn one row per subject into one
        row per qualifying child, which is a different answer, and the semi-join
        that would be correct (EXISTS) is not part of this surface. Refused loudly
        rather than answered over the wrong rows. With measures present the same
        filter is a legitimate semi-join and is folded into the spine instead.
        """
        missing = [index for index in range(len(self.filters))
                   if index not in self._applied]
        if missing:
            raise CompileError(
                "filter(s) " + ", ".join(
                    f"{self.plan.filters[i].attribute!r}" for i in missing)
                + " name a route the plan never projects or measures, so they would be "
                  "silently dropped"
            )

    def _listing(self) -> exp.Expression:
        if not self.plan.project:
            raise CompileError("a plan with no measures must project something")
        columns = [self._attribute(p.path, p.attribute) for p in self.plan.project]
        self._certify_row_grain([alias for alias, _c, _a in columns])
        aliases = self.layout.closure([self.layout.root] + [a for a, _c, _x in columns])
        select = self._source(aliases)
        labels = _labels([attribute.name for _a, _c, attribute in columns])
        select = select.select(
            *[exp.alias_(_col(alias, column), _ident(label))
              for (alias, column, _attr), label in zip(columns, labels)],
            copy=False,
        )
        if self._listing_is_distinct(columns):
            select = select.distinct(copy=False)
        order = self._ordering(
            projection_refs=[_col(a, c) for a, c, _x in columns],
            measure_refs=[],
            identity=[_col(self.layout.root, k) for k in (self.subject.key or ())],
        )
        return self._finish(select, order)

    def _listing_is_distinct(self, columns: Sequence[tuple[str, str, Attribute]]) -> bool:
        """Whether a measure-free listing must be DISTINCT. DERIVED, per DESIGN 3.4.

        `distinct` is not a field of `Plan` -- "a field a planner can only get
        wrong should not exist" -- so it has to be a function of the declared
        fan-out, and this is that function. The spine of the aggregate shape has
        been DISTINCT all along; the listing shape emitted none, so a row reached
        twice through a fanning intermediate came back twice. `Policy ->
        PolicyCoverageDetail -> Claim` is the shape: `COVERED_BY` multiplies,
        `CLAIMED_AGAINST` read backwards converges, so a policy reaches one claim
        once per coverage detail and `(policy_number, company_claim_number)` is
        emitted once per detail. On `acme_cf_fanout` the gold plan for
        `query-c70e3c4c` returned 5 rows where 4 are distinct.

        The predicate is `_Layout.may_revisit`, not `fans_out`, and the
        difference is the whole of the argument. FAN-OUT alone produces
        DIFFERENT rows -- a claim's four loss payments are four amounts, and
        listing them four times is the answer, so DISTINCT there would DELETE
        data. REVISITING produces the SAME row twice, and there the multiset is
        an artefact of the join tree rather than a fact about the domain: the
        repetition count is the fan-out of an intermediate the projection does
        not even name. Where a row can be reached twice the multiset is not
        certifiable and the set is (rule 4), so the set is what is emitted.

        Read from the ontology's declarations, never from the data.
        """
        watched = [self.layout.root] + [alias for alias, _column, _attribute in columns]
        return any(self.layout.may_revisit(alias) for alias in watched)

    def _certify_row_grain(self, aliases: Sequence[str]) -> None:
        for i, left in enumerate(aliases):
            for right in aliases[i + 1:]:
                peak = self.layout.peak(left, right)
                if peak is None:
                    continue
                raise GrainError(
                    f"`{left}` and `{right}` are projected on the same row but neither "
                    f"determines the other: `{peak}` fans out towards both, so every output "
                    f"row is one arbitrary pairing. Aggregate each branch at its own grain "
                    f"(the compiler will pre-aggregate a MEASURE across this shape; a raw "
                    f"row-grain projection across it has no certified value)."
                )

    def _aggregation(self, measures: Sequence[object]) -> exp.Expression:
        dimensions = [self._attribute(p.path, p.attribute) for p in self.plan.project]
        self._certify_row_grain([alias for alias, _c, _a in dimensions])

        self.group_keys = self._group_key_columns(dimensions)

        self._force_per_keyed = any(getattr(m, "over", None)
                                    for m in self.plan.measures)
        outer = [self._outer(m) for m in measures]

        keys = self.group_keys

        spine_aliases = self.layout.closure(
            [self.layout.root] + [a for a, _c, _x in dimensions])
        spine = self._source(spine_aliases)
        spine_projections: list[exp.Expression] = [
            exp.alias_(_col(alias, column), _ident(f"__k{i}"))
            for i, (alias, column) in enumerate(keys)
        ]
        spine_projections += [
            exp.alias_(_col(alias, column), _ident(f"__d{i}"))
            for i, (alias, column, _attr) in enumerate(dimensions)
        ]

        if keys:
            spine = spine.select(*spine_projections, copy=False).distinct(copy=False)
            select = exp.Select().from_(_table("__spine", "__spine"), copy=False)

            anchor: str | None = None
            for name, _satellite_select in self._satellite_selects:
                conditions = [_eq(_col("__spine", f"__k{i}"), _col(name, f"__k{i}"))
                              for i in range(len(keys))]
                if name in self._per_keyed:
                    if anchor is None:
                        anchor = name
                    else:
                        conditions.append(_eq(_col(anchor, "__p"), _col(name, "__p")))
                select = select.join(_table(name, name), on=_all(conditions),
                                     join_type="INNER", copy=False)
        else:

            spine = None
            names = [name for name, _select in self._satellite_selects]
            select = exp.Select().from_(_table(names[0], names[0]), copy=False)
            for name in names[1:]:
                select = select.join(_table(name, name), join_type="CROSS", copy=False)

        labels = _labels(
            [attribute.name for _a, _c, attribute in dimensions]
            + [_measure_label(m, self.ref) for m in self.plan.measures]
        )
        dimension_refs = [_col("__spine", f"__d{i}") for i in range(len(dimensions))]
        projections = [exp.alias_(ref, _ident(label))
                       for ref, label in zip(dimension_refs, labels)]
        measure_labels = labels[len(dimensions):]
        projections += [exp.alias_(expression, _ident(label))
                        for expression, label in zip(outer, measure_labels)]
        select = select.select(*projections, copy=False)
        if dimension_refs:
            select = select.group_by(*dimension_refs, copy=False)

        for clause in self.plan.having:
            select = select.having(self._having(clause, outer), copy=False)

        order = self._ordering(projection_refs=dimension_refs, measure_refs=outer,
                               identity=dimension_refs)
        select = self._finish(select, order)
        if spine is not None:
            select = select.with_("__spine", as_=spine, copy=False)
        for name, grain_select in self._grain_selects:
            select = select.with_(name, as_=grain_select, copy=False)
        for name, satellite_select in self._satellite_selects:
            select = select.with_(name, as_=satellite_select, copy=False)
        return select

    def _outer(self, node: object) -> exp.Expression:
        if isinstance(node, _Composite):
            parts = [self._outer(p) for p in node.parts]
            if node.operator == "/":
                if len(parts) != 2:
                    raise CompileError("division takes exactly two operands")
                return _safe_divide(parts[0], parts[1])
            if node.operator == "-":
                if len(parts) != 2:
                    raise CompileError("subtraction takes exactly two operands")
                return exp.Sub(this=parts[0], expression=parts[1])
            builder = _ARITHMETIC[node.operator]
            expression = parts[0]
            for part in parts[1:]:
                expression = builder(this=expression, expression=part)
            return exp.Paren(this=expression)
        if isinstance(node, _Over):

            values: list[exp.Expression] = []
            for part in node.inner.parts:
                if not isinstance(part, _Leaf):
                    raise CompileError(
                        "`over` needs every part of the composition to be a "
                        "single aggregation; nested compositions are not "
                        "supported at a declared grain")
                values.append(_col(self._satellite(part, per_keyed=True), "__v"))
            combined = values[0]
            for value in values[1:]:
                combined = exp.Add(this=combined, expression=value)
            return _aggregate(node.aggregation, exp.Paren(this=combined))
        leaf = node
        assert isinstance(leaf, _Leaf)

        name = self._satellite(leaf)
        if leaf.aggregation == "avg":
            return _safe_divide(_aggregate("sum", _col(name, "__v")),
                                _aggregate("sum", _col(name, "__n")))
        return _aggregate(_RECOMBINE[leaf.aggregation], _col(name, "__v"))

    def _having(self, clause: Having, outer: Sequence[exp.Expression]) -> exp.Expression:
        if not 0 <= clause.measure < len(outer):
            raise CompileError(f"having references measure {clause.measure}, which does not exist")
        node = {
            "=": exp.EQ, "!=": exp.NEQ, "<": exp.LT, "<=": exp.LTE,
            ">": exp.GT, ">=": exp.GTE, "LIKE": exp.Like,
        }.get(clause.operator)
        if node is None:
            raise CompileError(f"unknown operator {clause.operator!r}")
        return node(this=outer[clause.measure], expression=_literal(clause.value))

    def _ordering(
        self,
        *,
        projection_refs: Sequence[exp.Expression],
        measure_refs: Sequence[exp.Expression],
        identity: Sequence[exp.Expression],
    ) -> list[exp.Ordered]:
        """ORDER BY with an identity tiebreak, always.

        A truncation whose ordering has ties is nondeterministic even under a
        deterministic compiler, because the engine breaks the tie. The identity
        columns are appended so the order is total on the data as well as on the
        plan.
        """
        top = self.plan.top
        ordered: list[exp.Ordered] = []
        if top is not None:
            stated = [top.by_projection is not None, top.by_measure is not None]
            if sum(stated) != 1:
                raise CompileError("`top` orders by exactly one of a projection or a measure")
            if top.by_projection is not None:
                if not 0 <= top.by_projection < len(projection_refs):
                    raise CompileError(
                        f"top orders by projection {top.by_projection}, which does not exist")
                ordered.append(exp.Ordered(this=projection_refs[top.by_projection],
                                           desc=top.descending))
            else:
                index = top.by_measure or 0
                if not 0 <= index < len(measure_refs):
                    raise CompileError(
                        f"top orders by measure {index}, which does not exist")
                ordered.append(exp.Ordered(this=measure_refs[index], desc=top.descending))
        if not ordered:
            return []
        chosen = {o.this.sql(dialect=DIALECT) for o in ordered}
        for reference in list(projection_refs) + list(identity):
            text = reference.sql(dialect=DIALECT)
            if text in chosen:
                continue
            chosen.add(text)
            ordered.append(exp.Ordered(this=reference, desc=False))
        return ordered

    def _finish(self, select: exp.Select, order: Sequence[exp.Ordered]) -> exp.Select:
        if order:
            select = select.order_by(*order, copy=False)
        top = self.plan.top
        if top is not None and top.n is not None:
            if not order:
                raise CompileError("a truncation without an ordering is not deterministic")
            select = select.limit(exp.Literal.number(top.n), copy=False)
        return select

def _labels(names: Sequence[str]) -> list[str]:
    """Output column names, derived and deduplicated.

    Derived, never chosen: two plans differing only by an alias are the same plan
    (DESIGN rule 1), so there is no alias field to differ on.
    """
    out: list[str] = []
    seen: dict[str, int] = {}
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        out.append(name if count == 1 else f"{name}_{count}")
    return out

def _measure_label(measure: Measure, ref=None) -> str:
    """The derived output name. Operands are named in the order they are LOWERED,
    which for `+` and `*` is the canonical one, so the label cannot disagree with
    the expression under it."""
    if measure.governed:
        return measure.governed
    if measure.aggregation:
        return f"{measure.aggregation}_{measure.attribute or 'id'}"
    ordered = canonical_operands(measure.combine, measure.parts, ref)
    parts = [_measure_label(p, ref) for p in ordered]
    tag = {"+": "plus", "-": "minus", "*": "times", "/": "per"}[measure.combine or "+"]
    return f"_{tag}_".join(parts)

def compile(plan: Plan, onto: Ontology, graph: PathGraph) -> str:  # noqa: A001
    """Lower a governed `Plan` to SQL. Deterministic; raises rather than guessing.

    Raises `CompileError` for a plan the ontology does not sanction and
    `GrainError` for one whose number cannot be certified at the requested grain.
    """
    tree = _compile_tree(plan, onto, graph)
    return tree.sql(dialect=DIALECT, pretty=True)

def canonical_paths(plan: Plan, onto: Ontology, graph: PathGraph) -> tuple[Path, ...]:
    """The plan's routes as the compiler reads them -- roles re-derived, order
    fixed, and any route a governed metric had to derive appended at the end.

    The audit surface: what a plan MEANS is its canonical routes, not the tuple a
    caller happened to write, because a caller cannot supply a role. The plan is
    compiled to populate it -- a route a metric derives is only known once the
    metric has been expanded.
    """
    compiler = _Compiler(plan, onto, graph)
    compiler.build()
    return tuple(compiler.paths)

def role_signatures(plan: Plan, onto: Ontology, graph: PathGraph) -> tuple[tuple[str, ...], ...]:
    """The role commitment of each canonical route, from the GRAPH.

    `PathGraph.role_signature` and not `Path.roles`: a role-object subject is not
    a `Step`, so the property alone reads a policy-holder plan as role-free.
    """
    compiler = _Compiler(plan, onto, graph)
    compiler.build()
    return compiler.roles

def _compile_tree(plan: Plan, onto: Ontology, graph: PathGraph) -> exp.Expression:
    tree = _Compiler(plan, onto, graph).build()
    if plan.set_op is None:
        return tree
    operator, other = plan.set_op
    right = _compile_tree(other, onto, graph)
    node = {"union": exp.Union, "intersect": exp.Intersect, "except": exp.Except}[operator]
    return node(this=tree, expression=right, distinct=True)
