from __future__ import annotations

from typing import Any, Mapping

def _drop(obj: dict, *names: str) -> None:
    """Remove properties AND their `required` entries, together.

    Strict mode requires the two to agree; dropping a property while leaving it
    required produces a schema the provider rejects outright, which would turn
    every question into SCHEMA_REFUSED -- worse than the bug being fixed.
    """
    for name in names:
        obj.get("properties", {}).pop(name, None)
    if "required" in obj:
        obj["required"] = [r for r in obj["required"] if r in obj.get("properties", {})]

def narrow(schema: dict, resolved: Mapping[str, Any],
           decomposition: Mapping[str, Any]) -> dict:
    """Cut the pick schema down to what stage 2 resolved and the question asked."""
    decomp = decomposition

    measure = (schema.get("$defs") or {}).get("measure")
    if measure:

        defined = {m for m in (((resolved.get("report") or {}).get("definitions")
                                or {}).get("metrics") or ()) if m}
        summands = {k: set(v or ()) for k, v in
                    (resolved.get("summands") or {}).items() if v}

        redefined = bool(defined) and not any(
            parts == defined for parts in summands.values())

        if resolved.get("metrics") and not redefined:

            _drop(measure, "aggregation", "attribute", "combine", "parts")
        elif redefined:

            _drop(measure, "aggregation", "attribute")
        else:

            _drop(measure, "metric", "combine", "parts")

        for branch in ((measure.get("properties") or {}).get("over")
                       or {}).get("anyOf") or ():
            if isinstance(branch, dict) and branch.get("enum"):
                branch["enum"] = [a for a in branch["enum"]
                                  if a in ("avg", "min", "max")]

    if not (decomp.get("rank_phrases") or ()):
        _drop(schema, "top")

    if not ((decomp.get("literal_phrases") or ())
            or (decomp.get("comparison_phrases") or ())
            or (decomp.get("time_phrases") or ())):
        _drop(schema, "filters")
    if not ((decomp.get("attribute_phrases") or ()) or resolved.get("concepts")):
        _drop(schema, "dimensions")

    qualified = [str(a) for a in
                 ((resolved.get("report") or {}).get("attributes") or ()) if a]
    settled = {q.rpartition(".")[2] for q in qualified}
    if settled:
        dims = (schema.get("properties") or {}).get("dimensions") or {}
        target = ((dims.get("items") or {}).get("properties") or {})
        attr = target.get("attribute") or {}
        for branch in attr.get("anyOf") or ():
            if isinstance(branch, dict) and branch.get("enum"):
                kept = [a for a in branch["enum"] if a in settled]
                if kept:
                    branch["enum"] = kept

        subjects = [str(s) for s in (resolved.get("subjects") or ()) if s]
        candidates = {str(c.get("attribute"))
                      for entry in ((resolved.get("report") or {}).get("ambiguous") or ())
                      for c in (entry.get("candidates") or ()) if c.get("attribute")}
        owners = ({q.rpartition(".")[0] for q in qualified if "." in q}
                  | {c.rpartition(".")[0] for c in candidates if "." in c})
        route = target.get("route") or {}
        if subjects and owners and route.get("enum"):

            def lands(route_id: str) -> set[str]:
                return (set(subjects) if route_id == "SELF"
                        else {route_id.split(">", 1)[-1].split("#", 1)[0]})
            kept_routes = [r for r in route["enum"] if lands(r) & owners]

            if kept_routes:
                route["enum"] = kept_routes
    return schema
