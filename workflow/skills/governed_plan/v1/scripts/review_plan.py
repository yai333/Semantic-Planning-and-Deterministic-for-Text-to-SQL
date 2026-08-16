from __future__ import annotations

import json
import sys
from typing import Any

_ONTOLOGY = None

def _ontology():
    """The ontology, resolved once.

    `sys.path.insert` ran on EVERY call here, which was free when each step was
    its own short-lived subprocess and is not now that the engine calls these
    functions directly: this is invoked per attribute, so the path grew without
    bound inside one run. Guarded, and the result held.
    """
    global _ONTOLOGY
    if _ONTOLOGY is None:
        root = __file__.split("/workflow/")[0]
        if root not in sys.path:
            sys.path.insert(0, root)
        from spc.skills import default_skills  # noqa: PLC0415
        _ONTOLOGY = default_skills().ontology
    return _ONTOLOGY

_SKILLS = None

def _skills(given: Any = None) -> Any:
    """The resolver registry, resolved once.

    `review()` is declared in SKILL.md with `plan` and `resolved` only, so the
    engine never passes `skills` -- and any check needing the path graph would
    silently never run. Same lazy pattern as `_ontology()` above, and the caller
    may still inject one.
    """
    global _SKILLS
    if given is not None:
        return given
    if _SKILLS is None:
        root = __file__.split("/workflow/")[0]
        if root not in sys.path:
            sys.path.insert(0, root)
        from spc.skills import default_skills  # noqa: PLC0415
        _SKILLS = default_skills()
    return _SKILLS

def _selectable(qualified: str) -> bool:
    """`Concept.attr` has a backing column, so it can appear in SQL at all."""
    try:
        concept, attr = qualified.split(".", 1)
        spec = _ontology().concept(concept).attributes[attr]
    except Exception:                           # noqa: BLE001
        return False
    return getattr(spec, "column", None) is not None

def _concept_named_by(phrase: str) -> str | None:
    """The concept this phrase IS, if it is one."""
    want = phrase.casefold().replace(" ", "").replace("_", "")
    try:
        for concept in _ontology().concept_names():
            if concept.casefold().replace("_", "") == want:
                return concept
    except Exception:                           # noqa: BLE001
        return None
    return None

def _display_of(concept: str) -> str | None:
    try:
        return getattr(_ontology().concept(concept), "display", None)
    except Exception:                           # noqa: BLE001
        return None

def _edges_of(skills: Any, route_id: str) -> tuple[str, ...] | None:
    """The edge chain a route follows, or None if it cannot be read."""
    try:
        source, rest = route_id.split(">", 1)
        target = rest.split("#", 1)[0]
        got = skills.find_paths(source, target)
        for path in (got.data if hasattr(got, "data") else got).get("paths") or ():
            if path.get("route_id") == route_id:
                return tuple(e if isinstance(e, str) else str(e.get("edge"))
                             for e in (path.get("edges") or ()))
    except Exception:                            # noqa: BLE001
        return None
    return None

def _concepts_on(skills: Any, route_id: str) -> tuple[str, ...]:
    """Every concept `route_id` occupies, intermediates included. () if unreadable.

    `_edges_of`'s sibling: same `find_paths` lookup, reading the concepts the
    steps pass through rather than the edges they follow. Returns () rather than
    raising, so a route the graph cannot explain narrows nothing -- the caller
    still has the endpoints it parsed out of the id.
    """
    try:
        source, rest = route_id.split(">", 1)
        target = rest.split("#", 1)[0]
        got = skills.find_paths(source, target)
        for path in (got.data if hasattr(got, "data") else got).get("paths") or ():
            if path.get("route_id") != route_id:
                continue
            seen = {source, target}
            for step in path.get("steps") or ():
                for side in ("from_concept", "to_concept"):
                    if step.get(side):
                        seen.add(str(step[side]))
            return tuple(sorted(seen))
    except Exception:                            # noqa: BLE001
        return ()
    return ()

def _diverge(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Whether two edge chains are genuinely different BRANCHES.

    They are not, if one is a prefix of the other: `Claim>PolicyHolder#1` is
    CLAIMED_AGAINST > COVERED_BY > HOLDS_R and `Claim>Policy#1` is its first two
    edges, so the holder is reached THROUGH the policy and the two projections
    sit on one chain. Flagging that pair was this check's first bug -- it made
    the planner drop a dimension the compiler had no objection to, which is the
    "narrow the question until it compiles" failure this file exists to catch.
    """
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return longer[:len(shorter)] != shorter

def _step_fans(skills: Any, route_id: str) -> tuple[bool, ...] | None:
    """Whether EACH step along `route_id` fans out, in order -- one boolean per
    edge in `_edges_of`'s chain, read off the same path object `find_paths`
    already computed.

    NOT the route's whole-route `max_fan_out`. That field is the MAX over every
    step, so a junction shared by every multi-hop route off one subject (ACME:
    `CLAIMED_AGAINST`, Claim to its coverage details, declared `bounded`) makes
    EVERY such route report `bounded` regardless of what that route's own
    divergent edge does -- `Claim>Agent#1` read `bounded` even after `SOLD_R`
    was declared `fan_out_reverse: none`, purely because of a hop the Agent
    branch shares with every sibling dimension. `spc/compile.py::_Layout.peak`,
    the compiler's own certifier for this shape, does not make that mistake: it
    asks whether the immediate step where two branches SPLIT is MANY on both
    sides, nothing upstream of the split and nothing further down either.
    Matching what it actually checks needs the fan of each step, not one
    max over the route.
    """
    try:
        source, rest = route_id.split(">", 1)
        target = rest.split("#", 1)[0]
        got = skills.find_paths(source, target)
        paths = (got.data if hasattr(got, "data") else got).get("paths") or ()
        for path in paths:
            if path.get("route_id") == route_id:
                return tuple(str(s.get("fan_out", "none")) != "none"
                             for s in (path.get("steps") or ()))
    except Exception:                            # noqa: BLE001
        return None
    return None

def _routes_in(plan: dict) -> set[str]:
    seen: set[str] = set()
    for key in ("measures", "dimensions", "filters"):
        for item in plan.get(key) or []:
            if isinstance(item, dict) and item.get("route"):
                seen.add(str(item["route"]))
    return seen

def review(plan: dict, resolved: dict, skills: Any = None) -> dict:
    """`{ok, findings: [{severity, what, recommend}]}`.

    A MISSING ARGUMENT IS A FINDING, NEVER A CRASH. Both of these were `None` in
    practice and raised `AttributeError: 'NoneType' object has no attribute
    'get'`, which reaches the planner as `error: review_plan failed: ...` --
    indistinguishable from the step having no opinion. Three questions produced
    no SQL that way: the planner read a crash as a dead end and answered in
    prose. `resolved` goes missing legitimately, because the agent may call
    review before it has resolved anything.

    (This used to cite `on_compile_failure: resolve` as the reason. That key was
    declared in SKILL.md and never read by anything -- see the note where it
    used to sit. A comment is not evidence that a mechanism exists.)
    """
    if not isinstance(plan, dict):
        return {"ok": False, "findings": [{
            "severity": "error",
            "what": f"no plan was given to review (got {type(plan).__name__})",
            "recommend": "pass the draft plan as `plan`"}]}
    if not isinstance(resolved, dict):
        return {"ok": False, "findings": [{
            "severity": "error",
            "what": f"no resolution was given to review (got "
                    f"{type(resolved).__name__}) -- nothing can be checked "
                    f"against it",
            "recommend": "run `resolve` and pass its whole output as `resolved`. "
                         "If a compile refusal sent you back, the resolution was "
                         "discarded on purpose and must be redone"}]}
    report = resolved.get("report", resolved)
    if not isinstance(report, dict):
        report = resolved
    findings: list[dict] = []

    def _metrics_in(spec: Any) -> set[str]:
        if not isinstance(spec, dict):
            return set()
        found = {str(spec["metric"])} if spec.get("metric") else set()
        for part in spec.get("parts") or ():
            found |= _metrics_in(part)
        return found

    used_metrics = set()
    for m in plan.get("measures") or []:
        used_metrics |= _metrics_in(m)

    summands = report.get("summands") or {}
    contained = {part for whole in used_metrics for part in summands.get(whole, ())}

    _defined = {m for m in ((report.get("definitions") or {}).get("metrics")
                            or ()) if m}
    _redefined = {name for name, parts in summands.items()
                  if _defined and parts and set(parts) != _defined}
    for metric in report.get("metrics") or []:
        if metric in used_metrics or metric in contained or metric in _redefined:
            continue
        findings.append({
            "severity": "error", "what": f"resolved metric {metric!r} is not in the plan",
            "recommend": f"add a measure {{'metric': '{metric}', 'route': ...}} "
                         f"unless the question genuinely does not ask for it"})

    placed = {(str(m.get("metric")), str(m.get("route") or "SELF"))
              for m in (plan.get("measures") or [])
              if isinstance(m, dict) and m.get("metric")}
    for whole, route in sorted(placed):
        for part in summands.get(whole, ()):
            if (part, route) in placed:
                findings.append({
                    "severity": "error",
                    "what": f"{part!r} is added into {whole!r}; the plan projects both",
                    "recommend": f"drop the measure {part!r} -- it is already inside "
                                 f"{whole!r} -- unless the question asks to see the "
                                 f"breakdown as well as the total"})

    subject = str(plan.get("subject") or "")
    used_attrs = set()
    for key in ("dimensions", "filters"):
        for item in plan.get(key) or []:
            if not isinstance(item, dict) or not item.get("attribute"):
                continue
            name = str(item["attribute"])
            used_attrs.add(name)
            route = str(item.get("route") or "SELF")
            concept = (subject if route == "SELF"
                       else route.split(">", 1)[-1].split("#", 1)[0]
                       if ">" in route else "")
            if concept:
                used_attrs.add(f"{concept}.{name}")
    for qualified in report.get("attributes") or []:
        bare = qualified.split(".", 1)[-1]

        same_name = [q for q in (report.get("attributes") or [])
                     if q.split(".", 1)[-1] == bare]
        satisfied = (qualified in used_attrs
                     or (len(same_name) == 1 and bare in used_attrs))
        if not satisfied:
            findings.append({
                "severity": "error",
                "what": f"resolved attribute {qualified!r} is not used",
                "recommend": f"add {{'attribute': '{bare}', 'route': ...}} to dimensions "
                             f"if the question asks to see or group by it"})

    resolved_attrs = set(report.get("attributes") or [])

    grain_identity: set[str] = set()
    if skills and subject:
        try:
            spec = skills.ontology.concept(str(subject))
        except Exception:                        # noqa: BLE001
            spec = None
        if spec is not None and getattr(spec, "grain", None):
            for ident in (*(spec.key or ()), getattr(spec, "display", None)):
                if ident:
                    grain_identity.add(f"{subject}.{ident}")

    for key in ("dimensions", "filters"):
        for item in plan.get(key) or []:
            if not isinstance(item, dict) or not item.get("attribute"):
                continue
            name = str(item["attribute"])
            route = str(item.get("route") or "SELF")
            concept = (subject if route == "SELF"
                       else route.split(">", 1)[-1].split("#", 1)[0]
                       if ">" in route else "")
            if not concept:
                continue

            qualified = name if "." in name else f"{concept}.{name}"
            if qualified in resolved_attrs:
                continue

            if qualified in grain_identity:
                continue

            candidates = {str(c.get("attribute"))
                          for entry in (report.get("ambiguous") or [])
                          for c in (entry.get("candidates") or [])}
            subject_key = {f"{subject}.{k}" for k in
                           ((skills.ontology.concept(subject).key or ())
                            if skills and subject else ())}
            display = getattr(skills.ontology.concept(subject), "display", None) \
                if skills and subject else None
            if display:
                subject_key.add(f"{subject}.{display}")
            if qualified in candidates and not (
                    route == "SELF" and qualified in subject_key):
                continue
            findings.append({
                "severity": "error",
                "what": f"{key[:-1]} {qualified} was never resolved -- "
                        f"resolution settled {sorted(resolved_attrs) or 'nothing'}",

                "recommend": (
                    (f"{concept} is this plan's SUBJECT -- it is already what "
                     f"each row is about, and any route you take from it "
                     f"already restricts to the rows the question describes. A "
                     f"phrase that SCOPES the question needs no dimension; "
                     f"projecting one adds a column nobody asked for and splits "
                     f"the grouping. Drop the dimension whose attribute is "
                     f"`{name}` on route SELF and resubmit the REST of the plan "
                     f"unchanged -- it is the only thing wrong here. "
                     if concept and concept == plan.get("subject") else

                     f"Drop {qualified} and resubmit the rest of the plan "
                     f"unchanged -- every other dimension here resolved. ")
                    + "Project what resolution found and nothing else. If a "
                      "phrase came back ambiguous, say so in your answer rather "
                      "than choosing a reading for it -- and note that an extra "
                      "dimension changes the grouping, so it splits the very "
                      "number the question asked for")})

    for item in report.get("ambiguous") or []:
        phrase = str(item.get("phrase", ""))
        between = list(item.get("between") or [])
        live = [c for c in between if _selectable(c)]
        dropped = [c for c in between if c not in live]

        named = _concept_named_by(phrase)
        if named:
            display = _display_of(named)
            if display and _selectable(f"{named}.{display}"):
                findings.append({
                    "severity": "info",
                    "what": f"{phrase!r} names the concept {named}, not one of its fields",
                    "recommend": f"use {{'attribute': '{display}', 'route': ...}} -- "
                                 f"the ontology declares `display: {display}` as what "
                                 f"identifies {named}"})
                continue
        if len(live) == 1:
            attr = live[0].split(".", 1)[-1]
            findings.append({
                "severity": "info",
                "what": f"{phrase!r} looked ambiguous but only {live[0]} is selectable"
                        + (f" ({', '.join(dropped)} have no column)" if dropped else ""),
                "recommend": f"use {{'attribute': '{attr}', 'route': ...}}"})
            continue
        findings.append({
            "severity": "warn",
            "what": f"{phrase!r} matched several readings equally: {live or between}",
            "recommend": "a genuine semantic choice -- say the reading is ambiguous "
                         "rather than dropping the phrase or picking silently"})

    if "route_ids" not in resolved:
        return {"ok": False, "findings": [{
            "severity": "error",
            "what": "the `resolved` given to review_plan has no `route_ids`; it is "
                    "not the object resolve returned",
            "recommend": "pass resolve's whole output, not a part of it -- no "
                         "route can be checked without it"}]}

    if skills is not None:
        routed: dict[str, list[str]] = {}
        for item in plan.get("dimensions") or []:
            if not isinstance(item, dict):
                continue
            route = str(item.get("route") or "SELF")
            if route == "SELF" or ">" not in route or "#" not in route:
                continue
            routed.setdefault(route, []).append(str(item.get("attribute")))
        chains = {r: _edges_of(skills, r) for r in routed}
        fans = {r: _step_fans(skills, r) for r in routed}
        routes = sorted(routed)
        offending: set[str] = set()
        for i, r1 in enumerate(routes):
            for r2 in routes[i + 1:]:
                c1, c2, f1, f2 = chains.get(r1), chains.get(r2), fans.get(r1), fans.get(r2)
                if not c1 or not c2 or not f1 or not f2 or not _diverge(c1, c2):
                    continue
                shared = 0
                while (shared < len(c1) and shared < len(c2)
                       and c1[shared] == c2[shared]):
                    shared += 1
                if shared < len(f1) and shared < len(f2) and f1[shared] and f2[shared]:
                    offending.add(r1)
                    offending.add(r2)
        if offending:
            where = "; ".join(f"{', '.join(routed[route])} via {route}"
                              for route in sorted(offending))
            findings.append({
                "severity": "error",
                "what": f"two branches that each fan out are projected on the same "
                        f"row: {where}",
                "recommend": "the subject reaches several rows down each of these, "
                             "so pairing them makes every output row one arbitrary "
                             "combination. Keep one of them as a dimension, or ask "
                             "for the others as measures so each is aggregated at "
                             "its own grain"})

    if ((resolved.get("report") or {}).get("averages")):
        computes = any(
            (m.get("over") == "avg") or (m.get("combine"))
            for m in plan.get("measures") or [] if isinstance(m, dict))
        if not computes:
            findings.append({
                "severity": "error",
                "what": "the question asks for an AVERAGE but the plan "
                        "projects its parts without the division",
                "recommend": "wrap the parts in a combine (divide the total "
                             "by the count) or project the metric with "
                             "`over: avg` -- a sum and a count side by side "
                             "are two numbers, not their average"})

    retrieved = set(resolved.get("route_ids") or [])
    for route in _routes_in(plan):
        if route != "SELF" and route not in retrieved:

            target = str(route.split("#", 1)[0]).split(">")[-1]
            options: list[str] = []
            for item in plan.get("measures") or []:
                if isinstance(item, dict) and str(item.get("route")) == route:
                    options = [str(r) for r in
                               (((resolved.get("report") or {})
                                 .get("metric_routes") or {})
                                .get("by_metric") or {}).get(
                                   str(item.get("metric"))) or []]
                    break
            if options:
                rec = (f"{route!r} does not exist. This measure can use: "
                       f"{', '.join(options[:4])}"
                       + (" -- pick one exactly as written" if options else ""))
            else:
                rec = "use SELF or one of the routes resolve returned"
            findings.append({
                "severity": "error",
                "what": f"route {route!r} was never retrieved",
                "recommend": rec})

    _OVER_OK = ("avg", "min", "max")
    for spec in plan.get("measures") or ():
        over = (spec or {}).get("over")
        if not over:
            continue
        named = str((spec or {}).get("metric") or "")
        if str(over).casefold() not in _OVER_OK:
            findings.append({
                "severity": "error",
                "what": f"`over: {over!r}` is not an aggregation -- it must be "
                        f"one of {list(_OVER_OK)}",
                "recommend": (
                    f"if you meant to combine {named!r} with {over!r} "
                    f"arithmetically, that is not what `over` does: use "
                    f"`combine` with `parts`, one part per metric. `over` only "
                    f"says HOW to aggregate a single composite across its own "
                    f"rows")})
            continue
        try:
            per = getattr(_ontology().metrics.get(named), "per", None)
        except Exception:                                # noqa: BLE001
            per = None
        if not per:
            findings.append({
                "severity": "error",
                "what": f"`over` was set on {named!r}, which declares no `per` "
                        f"-- there is no grain for it to aggregate across",
                "recommend": (
                    f"drop `over` from {named!r}. Only a metric whose ontology "
                    f"entry declares `per` is formed on rows that can be "
                    f"averaged or bounded; for anything else the metric's own "
                    f"definition already says how it totals")})

    for spec in plan.get("measures") or ():
        if not isinstance(spec, dict):
            continue
        agg = str(spec.get("aggregation") or "").casefold()
        if (agg and agg != "count" and not spec.get("attribute")
                and not spec.get("metric")):
            findings.append({
                "severity": "error",
                "what": f"`aggregation: {agg}` with no attribute and no metric "
                        f"aggregates the subject's own key -- a number that "
                        f"means nothing",
                "recommend": "name the thing being aggregated. If the question "
                             "states a quantity (an amount, a total), quote its "
                             "words in `quantity_phrases` and resolve again -- "
                             "a governed metric will then be nameable as "
                             "{'metric': ...} -- rather than summing whatever "
                             "the subject's key happens to be"})

    for spec in plan.get("measures") or ():
        if not isinstance(spec, dict):
            continue
        named = [k for k in ("metric", "aggregation", "combine") if spec.get(k)]
        if len(named) > 1:
            findings.append({
                "severity": "error",
                "what": f"one measure names {named} -- a measure may name "
                        f"exactly one of metric, aggregation or combine",
                "recommend": "keep `combine` with its `parts` if the question "
                             "stated the arithmetic, and move the metric INSIDE "
                             "a part ({'metric': ...}); otherwise drop "
                             "`combine`/`parts` and keep the metric alone. The "
                             "compiler refuses this after submission, so fixing "
                             "it here costs nothing"})

    _sk = _skills(skills)
    retrieved_routes = [r for r in (resolved.get("route_ids") or ()) if ">" in r]
    for spec in plan.get("measures") or ():
        route = (spec or {}).get("route")
        if not route or route == "SELF" or route not in retrieved_routes:
            continue
        mine = _edges_of(_sk, route)
        if not mine:
            continue
        shorter = []
        for other in retrieved_routes:
            if other == route:
                continue
            chain = _edges_of(_sk, other)
            if (chain and len(chain) < len(mine)
                    and chain[-1] == mine[-1]
                    and other.split(">", 1)[0] == route.split(">", 1)[0]
                    and other.split(">", 1)[1].split("#", 1)[0]
                    == route.split(">", 1)[1].split("#", 1)[0]):
                shorter.append((len(chain), other, chain))
        if shorter:
            hops, best, chain = sorted(shorter)[0]
            findings.append({
                "severity": "error",
                "what": f"route {route!r} reaches the operand in {len(mine)} "
                        f"edges {list(mine)} where {best!r} does it in {hops} "
                        f"{list(chain)}, ending on the same edge",
                "recommend": f"use {best!r}. Both arrive at the same concept by "
                             f"{mine[-1]}, so the extra leg cannot reach "
                             f"anything the shorter route misses -- it can only "
                             f"widen what is counted, which is how a "
                             f"per-coverage amount becomes a policy-wide one"})

    defined = {m for m in ((report.get("definitions") or {}).get("metrics")
                           or ()) if m}
    summands = {k: set(v or ()) for k, v in
                (resolved.get("summands") or {}).items() if v}
    if defined:
        for spec in plan.get("measures") or ():
            named = (spec or {}).get("metric")
            parts = summands.get(named)
            if parts and parts != defined:
                findings.append({
                    "severity": "error",
                    "what": f"the question defines its quantity as "
                            f"{sorted(defined)}, but metric {named!r} adds "
                            f"{sorted(parts)}",
                    "recommend": f"{named!r} is a different quantity with the "
                                 f"same name -- using it answers a question "
                                 f"nobody asked. Build what was defined: set "
                                 f"`combine` to the operator the question used "
                                 f"and `parts` to {sorted(defined)}"})

    if defined:
        top_level = {str((m or {}).get("metric")) for m in plan.get("measures") or ()
                     if isinstance(m, dict) and m.get("metric")}

        composes = any((m or {}).get("combine")
                       for m in plan.get("measures") or ()
                       if isinstance(m, dict))
        shown = defined & top_level
        if shown and not composes and (len(shown) > 1 or len(defined) > 1):
            findings.append({
                "severity": "error",
                "what": f"the question DEFINES its quantity as {sorted(defined)}, "
                        f"and this plan returns {sorted(shown)} as separate "
                        f"measures -- that shows the parts, not the thing they "
                        f"define",
                "recommend": (
                    "build the defined quantity with ONE measure: set `combine` "
                    "to the operator the question used and `parts` to those "
                    "metrics, one part each. The parts belong inside it, not "
                    "beside it -- the question asked for what they make, not "
                    "for them")})

    named = [s for s in (resolved.get("subjects") or [])
             if isinstance(s, str) and s != plan.get("subject")]
    if named:

        reached: set[str] = set()
        for route in _routes_in(plan):
            if not route or route == "SELF":
                continue
            reached.update(route.split("#")[0].split(">"))
            reached.update(_concepts_on(skills, route))
        for subject in named:
            if subject not in reached:
                findings.append({
                    "severity": "error",
                    "what": f"the question named {subject!r} but no route in "
                            f"this plan reaches it",
                    "recommend": f"the plan answers about every row, not the "
                                 f"ones {subject!r} identifies -- any role or "
                                 f"restriction that concept carries is applied "
                                 f"by the JOIN a route to it makes. Project it "
                                 f"as a dimension, reach it on the route of a "
                                 f"measure, or MAKE IT THE SUBJECT and measure "
                                 f"from there -- anchoring works when the other "
                                 f"two do not, because a subject may always "
                                 f"project its own identifier. IF ITS ATTRIBUTES DID NOT "
                                 f"RESOLVE, that is because no span named it: "
                                 f"resolve again with the words the question "
                                 f"uses for {subject!r} in `attribute_phrases`, "
                                 f"which is what makes its identifier "
                                 f"projectable. If it is genuinely not part of "
                                 f"the answer, say so rather than dropping it"})

    subject_name = plan.get("subject")
    if subject_name and (plan.get("measures") or ()) and not any(
            (d or {}).get("route") in (None, "SELF")
            for d in plan.get("dimensions") or ()):
        try:
            import sys as _sys
            _root = __file__.split("/workflow/")[0]
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from spc.ontology import load_ontology
            grain = getattr(load_ontology().concept(subject_name), "grain", None)
        except Exception:                      # noqa: BLE001
            grain = None
        if grain:
            findings.append({
                "severity": "error",
                "what": f"subject {subject_name!r} is {grain} and no dimension "
                        f"identifies which one each row is about",
                "recommend": f"project an identifying attribute of "
                             f"{subject_name!r} on route SELF. Without it, two "
                             f"{subject_name}s sharing a value of your other "
                             f"dimensions become one row and their quantities "
                             f"are added together. IF NO ATTRIBUTE OF "
                             f"{subject_name!r} RESOLVED, resolve again with "
                             f"the words the question uses for it in "
                             f"`attribute_phrases` -- you cannot project what "
                             f"was never resolved, and dropping the subject "
                             f"instead is the one move that is not available"})

    for item in report.get("unresolved") or []:
        findings.append({
            "severity": "warn",
            "what": f"{item.get('phrase')!r} ({item.get('slot')}) resolved to nothing",
            "recommend": "the question asked for it and nothing matched; say so"})

    return {"ok": not any(f["severity"] == "error" for f in findings),
            "findings": findings}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: review_plan.py '<plan json>' '<resolve output json>'", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(review(json.loads(sys.argv[1]), json.loads(sys.argv[2])), indent=2))
