from __future__ import annotations

import argparse
import json
import sys

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m workflow")
    p.add_argument("question", nargs="+")
    p.add_argument("--model", default="gpt-5.6-luna", help="agent 1, the planner")
    p.add_argument("--critic-model", default="gpt-5.4-mini", help="agent 2")
    p.add_argument("--cap", type=int, default=1, help="review revisions; 0 disables")
    p.add_argument("--max-turns", type=int, default=14)
    p.add_argument("--trace", action="store_true", help="trace to MLflow")
    p.add_argument("--json", action="store_true", dest="as_json")
    p.add_argument("--sql-only", action="store_true")
    args = p.parse_args(argv)

    if args.trace:
        from spc import trace
        print(f"trace: {trace.configure('mlflow')}", file=sys.stderr)

    from workflow import Agent
    res = Agent(model=args.model, critic_model=args.critic_model,
                max_turns=args.max_turns, cap=args.cap).run(" ".join(args.question))

    if args.sql_only:
        if not res.ok:
            print(res.error or "no sql", file=sys.stderr)
            return 1
        print(res.sql)
        return 0

    tin, tout = res.cost()
    if args.as_json:
        print(json.dumps({
            "question": res.question, "ok": res.ok, "sql": res.sql,
            "skill": res.skill, "answer": res.answer, "error": res.error,
            "refusals": res.refusals, "verdict": res.verdict, "why": res.why,
            "revisions": res.revisions, "tokens": {"in": tin, "out": tout, "cached": res.cached()},
            "turns": [t.__dict__ for t in res.turns]}, indent=2))
        return 0 if res.ok else 1

    print(f"question : {res.question}")
    print(f"skill    : {res.skill or '-'}")
    for i, t in enumerate(res.turns, 1):
        print(f"  {i:2}. [{t.agent:7}] {t.tool:11} {t.detail[:76]}")
    for r in res.refusals:
        print(f"  refused: {r[:100]}")
    if res.verdict:
        print(f"review   : {res.verdict} -- {res.why[:88]}")
    cached = res.cached()
    share = f" ({cached/tin:.0%} cached)" if tin else ""
    print(f"turns    : {len(res.turns)}  revisions {res.revisions}  "
          f"tokens {tin}/{tout}{share}")
    if res.error:
        print(f"error    : {res.error}")
    if res.sql:
        print()
        print(res.sql)
    return 0 if res.ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
