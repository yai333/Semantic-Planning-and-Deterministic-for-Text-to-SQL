from __future__ import annotations

import json
import sys

def plan_schema(resolved: dict) -> dict:

    _root = __file__.split("/workflow/")[0]
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from spc.skills import default_skills
    import importlib.util as u
    from pathlib import Path

    ref = Path(__file__).resolve().parent / "narrow.py"
    spec = u.spec_from_file_location("_narrow", ref)
    narrow_mod = u.module_from_spec(spec)
    spec.loader.exec_module(narrow_mod)

    skills = default_skills()
    subjects = list(dict.fromkeys(resolved.get("subjects") or []))
    route_ids = list(dict.fromkeys(resolved.get("route_ids") or []))

    if len(subjects) > 1:
        base = skills.menu(
            route_ids, subject=subjects[0],
            metrics=resolved.get("metrics") or None,
            concepts=resolved.get("concepts") or ()).pick_schema(subject=None)
        prop = (base.get("properties") or {}).get("subject") or {}
        if set(subjects) <= set(prop.get("enum") or ()):
            prop["enum"] = subjects
    else:
        base = skills.pick_schema(
            route_ids,
            subject=subjects[0] if subjects else None,
            metrics=resolved.get("metrics") or None,
            concepts=resolved.get("concepts") or ())
    return narrow_mod.narrow(base, resolved, resolved.get("decomposition") or {})

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: plan_schema.py '<resolve output json>'", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(plan_schema(json.loads(sys.argv[1])), indent=2))
