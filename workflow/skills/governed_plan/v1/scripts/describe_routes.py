from __future__ import annotations

from typing import Any

RESOLVERS = frozenset({"find_paths"})

def describe_routes(source: str, target: str, skills: Any) -> dict:
    """Every governed route from `source` to `target`, with its edges.

    `fans_out` is per STEP, in order, matching `edges`. A True means that step
    may reach several rows, so an extra True is an extra multiplication -- which
    is the difference between the two routes in this module's docstring.
    """
    try:
        got = skills.find_paths(source, target)
        payload = got.data if hasattr(got, "data") else got
    except Exception as exc:                                # noqa: BLE001
        return {"source": source, "target": target, "routes": [],
                "error": f"no routes could be read: {exc}"}

    routes = []
    for path in (payload.get("paths") or ()):
        edges = tuple(e if isinstance(e, str) else str(e.get("edge"))
                      for e in (path.get("edges") or ()))
        fans = tuple(str(s.get("fan_out", "none")) != "none"
                     for s in (path.get("steps") or ()))
        routes.append({
            "route": path.get("route_id"),
            "edges": list(edges),
            "hops": len(edges),
            "fans_out": list(fans),
            "multiplying_steps": sum(1 for f in fans if f),
        })
    routes.sort(key=lambda r: (r["hops"], r["route"] or ""))

    note = ("Routes are listed shortest first. Where two share the SAME FINAL "
            "EDGE, the shorter one reaches the same concept by the same "
            "relationship and the extra legs can only widen what is counted -- "
            "prefer it. Where the final edges DIFFER, they are different "
            "relationships and the choice is a semantic one: read the edge "
            "names. Use a `route` value from this list verbatim; there are no "
            "other routes and the ids are not composable.")
    return {"source": source, "target": target,
            "route_count": len(routes), "routes": routes, "note": note}
