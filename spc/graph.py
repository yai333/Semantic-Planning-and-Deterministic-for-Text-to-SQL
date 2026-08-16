from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

from spc.ontology import Concept, Edge, Ontology, load_ontology, predicate_code
from spc.plan import Path, Step

__all__ = [
    "PathGraph",
    "Traversal",
    "path_sort_key",
    "path_signature",
    "DEFAULT_MAX_HOPS",
]

DEFAULT_MAX_HOPS = 4

def path_sort_key(path: Path) -> tuple:
    """The total order paths are returned in.

    Hop count first (shorter routes are the more likely reading and the cheaper
    SQL), then the step sequence. Direction and role are IN the key because
    edge names alone do not separate two distinct paths, and a partial key plus
    a stable sort is exactly how arrival order survived the previous fix.

    `role or ""` avoids comparing `None` against `str`, which raises -- the
    dataclass-generated ordering on `Step` cannot be used for this reason.
    """
    return (
        len(path.steps),
        tuple((s.edge, not s.forward, s.role or "") for s in path.steps),
        path.target,
    )

def path_signature(path: Path) -> str:
    """A short stable text form. Used for hashing a whole result set in tests."""
    body = "|".join(
        f"{s.edge}{'>' if s.forward else '<'}{s.role or '-'}" for s in path.steps
    )
    return f"{body}=>{path.target}"

def _assert_total_order(paths: Sequence[Path]) -> None:
    keys = [path_sort_key(p) for p in paths]
    for previous, current in zip(keys, keys[1:]):
        if previous >= current:
            raise AssertionError(
                "path ordering is not total -- two paths share a sort key; "
                "ordering would fall back to enumeration order: "
                f"{previous!r} vs {current!r}"
            )

@dataclass(frozen=True)
class Traversal:
    """One directed reading of an edge, with everything the walk needs.

    Precomputed once at build time: the walk itself allocates nothing but the
    `Step` tuple it is about to record.
    """

    step: Step
    origin: str
    landing: str
    edge: Edge

    @property
    def fan_out(self) -> str:
        """Declared row multiplication in this direction (grain certification)."""
        return self.edge.fan_out_in(forward=self.step.forward)

def _derive_role(edge: Edge, landing: Concept) -> str | None:
    """The role a step commits to. DESIGN rule 3.

    Two sources, both intrinsic to the ontology:
      * the edge's `role_predicate`  (`HOLDS` -> PH)
      * the `backed_where` of a role-object concept the step LANDS ON
        (`Policy -HOLDS_R-> PolicyHolder` read in reverse -> PH)

    Both may apply; the codes are concatenated in a fixed order (edge, then
    concept) with duplicates collapsed, so the role of a step is a pure function
    of the ontology and never of traversal history.
    """
    codes: list[str] = []
    for code in (edge.role_code, landing.role_code):
        if code and code not in codes:
            codes.append(code)
    if not codes:
        return None
    return "+".join(codes)

class PathGraph:
    """In-memory governed graph over an `Ontology`.

    Immutable after construction. `paths()` memoises per
    `(source, target, max_hops, revisit_concepts)`, so repeated enumeration --
    the inner loop of candidate generation -- costs a dict lookup.
    """

    def __init__(self, ontology: Ontology) -> None:
        self._ontology = ontology
        self._adjacency: dict[str, tuple[Traversal, ...]] = _build_adjacency(ontology)
        self._cache: dict[tuple[str, str, int, bool], tuple[Path, ...]] = {}

    @classmethod
    def from_ontology(cls, ontology: Ontology) -> "PathGraph":
        return cls(ontology)

    @classmethod
    def load(cls, *args, **kwargs) -> "PathGraph":
        """Load the ontology from disk and build the graph. Arguments as
        `spc.ontology.load_ontology`."""
        return cls(load_ontology(*args, **kwargs))

    @property
    def ontology(self) -> Ontology:
        return self._ontology

    @property
    def concepts(self) -> tuple[str, ...]:
        """Every concept, sorted. Iteration order is never YAML order."""
        return tuple(sorted(self._adjacency))

    def traversals_from(self, concept: str) -> tuple[Traversal, ...]:
        """Every directed reading leaving `concept`, in the canonical order the
        walk visits them."""
        if concept not in self._adjacency:
            raise KeyError(f"unknown concept {concept!r}")
        return self._adjacency[concept]

    def degree(self, concept: str) -> int:
        return len(self.traversals_from(concept))

    def role_signature(self, subject: str, path: Path) -> tuple[str, ...]:
        """The FULL role commitment of traversing `path` from `subject`.

        `Path.roles` covers the steps, and the subject concept is not a step --
        so a plan whose subject is `PolicyHolder` reads as role-free through
        that property alone, even though selecting the concept IS the role
        commitment (`backed_where`). Role closure per DESIGN rule 3 must bucket
        candidates on THIS, not on `Path.roles`, or a role-object subject and a
        role-typed-edge subject are wrongly judged to differ by role.

        Derived and ordered: subject role first, then one entry per role-bearing
        step, in traversal order, preserving branch identity.
        """
        subject_role = self._ontology.concept(subject).role_code
        return ((subject_role,) if subject_role else ()) + path.roles

    def paths(
        self,
        source: str,
        target: str,
        *,
        max_hops: int = DEFAULT_MAX_HOPS,
        revisit_concepts: bool = False,
    ) -> tuple[Path, ...]:
        """Every governed path from `source` to `target`, in total order.

        `revisit_concepts=False` (default) enumerates simple paths: a concept
        appears at most once, so the plan has one alias per concept and no
        self-join.

        `revisit_concepts=True` reproduces the old Cypher relationship-
        uniqueness semantics and exists ONLY so the difference can be measured
        rather than asserted (`test_old_semantics` reproduces the recorded 465).
        **It is not a candidate-generation setting.** Turning it on for real
        enumeration multiplies Claim->ClaimAmount from 5 to 465, of which 460
        are self-joins -- which is the duplication crisis DESIGN rule 1 exists
        to prevent, arriving by a different door. A caller that wants self-join
        readings should say so as an explicit, separately governed construct.
        """
        if source not in self._adjacency:
            raise KeyError(f"unknown concept {source!r}")
        if target not in self._adjacency:
            raise KeyError(f"unknown concept {target!r}")
        if max_hops < 0:
            raise ValueError("max_hops must be >= 0")

        cache_key = (source, target, max_hops, revisit_concepts)
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit

        found = list(self._walk(source, target, max_hops, revisit_concepts))
        found.sort(key=path_sort_key)
        _assert_total_order(found)
        result = tuple(found)
        self._cache[cache_key] = result
        return result

    def concepts_visited(self, source: str, path: Path) -> tuple[str, ...]:
        """The concepts `path` occupies, starting at `source`. In traversal order."""
        visited = [source]
        node = source
        for step in path.steps:
            edge = self._ontology.edge(step.edge)
            if edge.origin(forward=step.forward) != node:
                raise ValueError(
                    f"edge {step.edge!r} does not leave {node!r} -- the path is not "
                    f"connected from {source!r}"
                )
            node = edge.endpoint(forward=step.forward)
            visited.append(node)
        return tuple(visited)

    def extend(self, source: str, path: Path, edge_name: str) -> Path | None:
        """`path` with one more governed step over `edge_name`, or None.

        Governed means exactly what `paths()` enumerates, and this composes ONE
        route out of two decisions the plan makes separately: a governed metric
        names its last edge, a plan names the route to the concept the metric is
        measured over, and the two must compose into a route the enumerator
        itself would produce. It returns None when they do not -- when the edge
        does not leave where the path landed, or lands on a concept the route
        already visited. That second case is a SELF-JOIN reading, which
        `paths()` excludes from candidate generation (DESIGN rule 1) and which
        this must exclude identically, or the compiler would refuse routes the
        knowledge layer offered.

        The role is derived here, exactly as the walk derives it, so an extended
        route is byte-identical to the enumerated one it must equal.
        """
        for traversal in self._adjacency.get(path.target, ()):
            if traversal.edge.name != edge_name:
                continue
            if traversal.landing in self.concepts_visited(source, path):
                return None
            return Path(steps=path.steps + (traversal.step,), target=traversal.landing)
        return None

    def path_count(self, source: str, target: str, **kwargs) -> int:
        return len(self.paths(source, target, **kwargs))

    def shortest(self, source: str, target: str, **kwargs) -> Path | None:
        """The first path in the total order, or None. Deterministic by
        construction -- there is no tie to break."""
        found = self.paths(source, target, **kwargs)
        return found[0] if found else None

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _walk(
        self, source: str, target: str, max_hops: int, revisit_concepts: bool
    ) -> Iterator[Path]:
        adjacency = self._adjacency
        steps: list[Step] = []
        visited: set[str] = {source}
        used_edges: set[str] = set()

        def descend(node: str) -> Iterator[Path]:
            if len(steps) >= max_hops:
                return
            for traversal in adjacency[node]:
                if revisit_concepts:

                    if traversal.edge.name in used_edges:
                        continue
                    used_edges.add(traversal.edge.name)
                else:
                    if traversal.landing in visited:
                        continue
                    visited.add(traversal.landing)

                steps.append(traversal.step)
                if traversal.landing == target:
                    yield Path(steps=tuple(steps), target=target)
                yield from descend(traversal.landing)
                steps.pop()

                if revisit_concepts:
                    used_edges.discard(traversal.edge.name)
                else:
                    visited.discard(traversal.landing)

        yield from descend(source)

def _build_adjacency(ontology: Ontology) -> dict[str, tuple[Traversal, ...]]:
    """Both directed readings of every edge, bucketed by origin and sorted.

    The sort here is what makes the walk's emission order deterministic before
    anything is sorted downstream: `(edge name, forward-first)`. Two edges cannot
    share a name -- the loader rejects that -- so the key is total.
    """
    buckets: dict[str, list[Traversal]] = {name: [] for name in ontology.concepts}
    for edge in ontology.edges:
        for forward in (True, False):
            origin = edge.origin(forward=forward)
            landing = edge.endpoint(forward=forward)
            role = _derive_role(edge, ontology.concepts[landing])
            buckets[origin].append(
                Traversal(
                    step=Step(edge=edge.name, forward=forward, role=role),
                    origin=origin,
                    landing=landing,
                    edge=edge,
                )
            )
    return {
        name: tuple(sorted(items, key=lambda t: (t.step.edge, not t.step.forward)))
        for name, items in buckets.items()
    }

def _main() -> None:  # pragma: no cover
    import time

    graph = PathGraph.load()
    pairs = [
        ("Party", "Policy"),
        ("Claim", "ClaimAmount"),
        ("Policy", "ClaimAmount"),
        ("Claim", "Occurrence"),
    ]
    print(f"{len(graph.concepts)} concepts, {len(graph.ontology.edges)} edges")
    for source, target in pairs:
        start = time.perf_counter()
        found = graph.paths(source, target)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"\n{source} -> {target}: {len(found)} paths in {elapsed:.3f} ms")
        for path in found[:6]:
            print(f"  {path_signature(path)}")

if __name__ == "__main__":  # pragma: no cover
    _main()
