from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from spc import trace
from workflow import loader

ROOT = Path(__file__).resolve().parent.parent

PLANNER_SYSTEM = """You are an analytics agent over a governed insurance ontology.

You answer questions by loading the skill that fits and following its steps. You
never write SQL: the skill's `certify` step compiles it from the plan you build,
and a plan naming anything that was not retrieved is refused.

Call one tool at a time and think briefly about what each result tells you.
Follow the loaded skill's steps in order.

Your work ends when you submit the plan. It is reviewed, compiled and checked
after you -- none of that is yours to do, and there is no second model reading
your work.

If it is rejected, THIS ATTEMPT ENDS. You do not get to patch the plan in place:
the whole run starts again from the question, and you will be told what went
wrong last time. Attempts are limited, so treat the first one as the one that
counts. If it cannot be certified, say so rather than claiming an answer.

If the question is out of scope for the ontology, say so plainly and stop.
Declining is a correct answer, and better than a confident one at the wrong grain.
"""

RETRY_CONTEXT = """

## A previous attempt at this question failed

    {failure}

That attempt is over and its reasoning is not available to you. Start from the
question again. If the reason above says the plan named something that could not
be built, consider cutting the question differently -- different spans, or
different subjects -- rather than rebuilding the same plan and expecting a
different answer. If it says the question cannot be answered at a single grain,
say so plainly instead of trying again."""

@dataclass
class Turn:
    agent: str = "planner"
    tool: str = ""
    detail: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    cached_tokens: int = 0

@dataclass
class Result:
    question: str
    sql: str = ""
    plan: dict | None = None
    answer: str = ""
    skill: str = ""
    verdict: str = ""
    why: str = ""
    revisions: int = 0
    refusals: list[str] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.sql)

    def cost(self) -> tuple[int, int]:
        return (sum(t.input_tokens for t in self.turns),
                sum(t.output_tokens for t in self.turns))

    def cached(self) -> int:
        return sum(t.cached_tokens for t in self.turns)

TRACE_FIELD = 12_000

def _traceable_value(value: Any) -> Any:
    """A span-safe view of `value`, WITH ANY ELISION DECLARED.

    Silently dropping the tail is the defect this repo spent a day on; a trace
    that quietly shortens a payload would reproduce it in the one place built to
    diagnose it. So an oversized field is replaced by a marker that says what it
    was and where the whole thing lives.
    """
    try:
        text = value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, default=str)
    except Exception:                            # noqa: BLE001
        return f"<unserialisable {type(value).__name__}>"
    if len(text) <= TRACE_FIELD:
        return value
    return (f"<{len(text)} chars, over the {TRACE_FIELD} trace field limit; "
            f"the whole value is in results/steps/> " + text[:TRACE_FIELD])

def _traceable(args: dict) -> dict:
    return {k: _traceable_value(v) for k, v in (args or {}).items()}

class SkillStepError(RuntimeError):
    """A step that cannot be called at all -- absent, or missing its entry."""

class Agent:
    def __init__(self, *, llm: Callable[..., Any] | None = None,
                 model: str = "gpt-5.6-luna",
                 skills: list[str] | None = None, temperature: float = 0.0,
                 max_turns: int = 14, cap: int = 1, samples: int = 1,
                 timeout: int = 120,
                 max_step_output: int = 64_000,
                 artifacts: Path | None = None) -> None:
        if llm is None:

            if str(model).startswith("claude-"):
                from spc.anthropic_proxy import anthropic_complete
                llm = anthropic_complete
            else:
                from spc.bench import complete
                llm = complete
        self.llm = llm
        self.model = model
        self.temperature, self.max_turns, self.cap = temperature, max_turns, cap

        self.samples = samples
        self.max_step_output = max_step_output
        self._loaded: dict[tuple[str, str], Any] = {}
        self._resolved: dict | None = None
        self._effective: dict | None = None
        self._ran_since_submit: set[str] = set()
        self._last_review_errors: list[str] = []
        self.artifacts = artifacts or (ROOT / 'results' / 'steps')
        self.timeout = timeout
        self.catalogue = {s.key: s for s in loader.catalogue(skills or ["governed_plan"])}
        from spc.skills import default_skills
        self.skills = default_skills()
        self._checker: Callable[[str], list] | None = None

    def _tools(self, loaded: str, resolved: dict | None = None) -> list[dict]:
        """`load_skill`, then ONE TYPED TOOL PER DECLARED STEP of that skill."""
        tools = [{"type": "function", "function": {
            "name": "load_skill",
            "description": "Load a skill's instructions before using it.",
            "parameters": {"type": "object", "additionalProperties": False,
                           "required": ["skill"],
                           "properties": {"skill": {"type": "string",
                                                    "enum": sorted(self.catalogue)}}}}}]
        if not loaded:
            return tools
        skill = self.catalogue[loaded]
        for name, step in dict(skill.spec.get("steps") or {}).items():
            props: dict[str, Any] = {}
            for arg, spec in dict(step.get("args") or {}).items():
                if spec.get("schema"):

                    props[arg] = dict(skill.resource(str(spec["schema"])))
                else:
                    props[arg] = {"type": str(spec.get("type", "string")),
                                  "description": str(spec.get("for", ""))}
            tools.append({"type": "function", "function": {
                "name": name, "description": str(step.get("for", "")),
                "parameters": {"type": "object", "additionalProperties": False,
                               "required": list(props), "properties": props}}})

        answer = dict(skill.spec.get("answer") or {})

        required = list(answer.get("requires") or ()) if answer else []
        satisfied = all(step in self._ran_since_submit for step in required)
        if answer and resolved is not None and satisfied:
            tools.append({"type": "function", "function": {
                "name": str(answer["name"]),
                "description": str(answer.get("for", "")),
                "parameters": self._answer_schema(skill, resolved)}})
        return tools

    def _answer_schema(self, skill: Any, resolved: dict) -> dict:
        """Build the plan schema by running the skill's declared script."""
        answer = dict(skill.spec.get("answer") or {})
        source = str(answer.get("schema", ""))
        if not source.startswith("script:"):
            return dict(skill.resource(source))
        out = self._run_step(skill.key, source.split(":", 1)[1],
                             {"resolved": resolved}, declared=False)

        self._effective = None
        if not isinstance(out, dict):

            raise RuntimeError(f"{skill.key}: plan schema did not build: {str(out)[:200]}")
        return out

    def _step_fn(self, skill: Any, name: str, step: dict):
        """Import a step's script once and return its declared entry function.

        The module is loaded by PATH and namespaced by skill directory, so
        `v1/resolve.py` and `v2/resolve.py` are two modules and a campaign can
        run both. It is deliberately NOT registered under a name a plain
        `import` could reach: a step is called through the skill that declares
        it or not at all.
        """
        import importlib.util  # noqa: PLC0415

        script = (skill.path / str(step["script"])).resolve()
        if not script.is_relative_to(skill.path.resolve()) or not script.exists():
            raise SkillStepError(
                f"{skill.key} declares a script outside itself, or absent")
        entry = str(step.get("entry") or name)
        key = (str(script), entry)
        cached = self._loaded.get(key)
        if cached is not None:
            return cached
        spec = importlib.util.spec_from_file_location(
            f"_skillstep.{skill.key}_v{skill.version}.{script.stem}", script)
        if spec is None or spec.loader is None:      # pragma: no cover
            raise SkillStepError(f"cannot load {script.name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, entry, None)
        if not callable(fn):
            raise SkillStepError(
                f"{script.name} declares entry {entry!r}, which it does not define")
        self._loaded[key] = fn
        return fn

    def _run_step(self, loaded: str, name: str, args: dict,
                  declared: bool = True) -> Any:
        """Call a declared step IN PROCESS and return what it returns.

        THIS USED TO BE A SUBPROCESS. Each step was `python scripts/<step>.py`
        with its arguments JSON-encoded into argv and its result parsed back off
        stdout -- originally so the scripts could run in the skill's own uv
        sandbox. With that sandbox gone (it declared the repo as a path
        dependency, so the isolation was nominal) the subprocess bought nothing
        and cost three things:

          * A SERIALISATION BOUNDARY WHERE NONE WAS NEEDED. Every value crossed
            as text, and `stdout` was capped at 8000 characters -- which
            silently truncated `resolve`'s 11,963-char JSON mid-token and cost a
            day of misdirected debugging. A returned dict cannot be truncated.
          * ~150ms of interpreter startup per step, several steps per turn.
          * A fresh `Skills()` per call: one ontology parse and one path-graph
            build per step, thrown away each time. The engine's own instance is
            passed in instead.

        `declared=False` is for scripts the ENGINE runs on the skill's behalf --
        the plan schema builder -- which the agent may not call and which is
        therefore not in `steps:`.
        """
        skill = self.catalogue[loaded]
        steps = dict(skill.spec.get("steps") or {})
        if declared:
            if name not in steps:
                return f"error: {name!r} is not a step of {skill.key}; has {sorted(steps)}"
            step = steps[name]
        else:
            step = {"script": f"scripts/{name}.py", "args": {k: {} for k in args},
                    "entry": name}

        try:
            fn = self._step_fn(skill, name, step)
        except SkillStepError as exc:
            return f"error: {exc}"

        import inspect  # noqa: PLC0415

        wanted = list(dict(step.get("args") or {}))
        call: dict[str, Any] = {k: args.get(k) for k in wanted}
        if "skills" in inspect.signature(fn).parameters:
            call["skills"] = self.skills

        if "resolved" in wanted and self._resolved is not None:
            call["resolved"] = self._resolved

        self._effective = {k: ("<engine: Skills>" if k == "skills" else v)
                           for k, v in call.items()}
        try:
            return fn(**call)
        except Exception as exc:                     # noqa: BLE001

            return f"error: {name} failed: {type(exc).__name__}: {exc}"

    def _stash(self, name: str, body: str) -> Path:
        """Write a step's full output where a human can read it afterwards.

        The conversation gets the part addressed to the model; this keeps the
        whole thing, so a run stays reconstructable without re-running it and
        without paying for the payload on every turn.
        """
        self._steps = getattr(self, "_steps", 0) + 1
        directory = self.artifacts / f"{id(self):x}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self._steps:02d}-{name}.json"
        path.write_text(body)
        return path

    def _certify(self, plan: dict, route_ids: list[str]) -> tuple[str, str]:
        """Compile the plan and certify its grain. Returns `(sql, refusal)`.

        THE ENGINE'S JOB, NOT A STEP OF THE SKILL. It was a tool the agent
        called, which meant the agent could certify repeatedly, decide what a
        refusal meant, and declare itself finished. Compiling and checking are
        deterministic and happen to every plan exactly once; leaving them to the
        agent gave it a decision it has no information to make.
        """
        from spc import check as check_module
        from spc import compile as compile_module
        from spc.bench import DB, load_checker

        skills = self.skills
        stray = [r for r in skills.route_ids_in(plan) if r not in set(route_ids)]
        if stray:
            return "", (f"pick rejected: UnretrievedRouteError: {stray[0]!r} is a "
                        f"governed route this question never retrieved")

        def _placed(measure: dict) -> tuple[str, str]:
            return str(measure.get("metric")), str(measure.get("route") or "SELF")

        placed = [_placed(m) for m in (plan.get("measures") or [])
                  if isinstance(m, dict) and m.get("metric")]
        doubled = sorted({
            (part, route)
            for metric, route in placed
            for part in skills.ontology.summands(metric)
            if (part, route) in placed
        })
        if doubled:
            part, route = doubled[0]
            inside = next(metric for metric, r in placed
                          if r == route and part in skills.ontology.summands(metric))
            where = "" if route == "SELF" else f" by route {route!r}"
            return "", (f"pick rejected: DoubleCountedMeasure: {part!r} is "
                        f"added into {inside!r}, and the plan projects both"
                        f"{where} -- drop it unless the question asks for the "
                        f"breakdown as well as the total")

        if not plan.get("subject"):
            roots = {str(r).split(">", 1)[0]
                     for r in skills.route_ids_in(plan)
                     if r and str(r) != "SELF"}
            if len(roots) == 1:
                plan = {**plan, "subject": roots.pop()}

        try:
            parsed = skills.parse_pick(plan)
        except Exception as exc:                # noqa: BLE001
            return "", f"pick rejected: {type(exc).__name__}: {exc}"
        try:
            sql = compile_module.compile(parsed, skills.ontology, skills.graph)
        except Exception as exc:                # noqa: BLE001
            return "", f"compile refused: {type(exc).__name__}: {exc}"
        if self._checker is None:
            self._checker = load_checker(DB)

        blocking = check_module.blocking(self._checker(sql))
        if blocking:
            codes = ", ".join(sorted({getattr(v, "code", "?") for v in blocking}))
            return "", f"checker rejected: {codes}"
        return sql, ""

    def run(self, question: str) -> Result:
        """One question, retried as a WHOLE RUN rather than patched in place.

        A rejected plan used to be handed back inside the same conversation --
        "fix what it names and submit again" -- with the failed plan, the
        reasoning that produced it and every earlier observation still in
        context. The planner then edited that plan, which is the one thing it
        should not do when the plan's premise is what failed: on query-08e96beb
        it alternated between two illegal plans until the cap stopped it.

        A retry is now a fresh run. The conversation is discarded and only the
        REASON survives, appended to the system prompt (`RETRY_CONTEXT`), so the
        planner starts from the question with one new fact rather than from a
        transcript arguing for the plan that failed.

        `cap` is the number of retries, not the number of submissions.
        """

        if self.samples > 1:
            tried: list[Result] = []
            for _ in range(self.samples):
                got = self._attempt(question)
                tried.append(got)
                if got.ok:
                    got.revisions = len(tried) - 1
                    got.refusals = [r for t in tried for r in t.refusals]
                    return got
            worst = tried[-1]
            worst.refusals = [r for t in tried for r in t.refusals]
            return worst

        attempt = self._attempt(question)
        for _ in range(self.cap):
            if attempt.ok or not attempt.refusals:
                break
            retry = self._attempt(question, failure=attempt.refusals[-1])

            retry.refusals = attempt.refusals + retry.refusals
            retry.revisions = attempt.revisions + 1
            attempt = retry
        return attempt

    def _attempt(self, question: str, failure: str = "") -> Result:
        """One run of the agent, start to finish. No in-place revision."""
        advert = loader.advertise(list(self.catalogue.values()))
        system = f"{PLANNER_SYSTEM}\n\n{advert}"
        if failure:
            system += RETRY_CONTEXT.format(failure=failure)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": question}]

        self._resolved = None
        self._ran_since_submit: set[str] = set()
        self._last_review_errors: list[str] = []
        res, loaded, resolved = Result(question=question), "", None

        with trace.span("agent", question=question, model=self.model,
                        commit=_commit(), is_retry=bool(failure),

                        carried_failure=failure or "(first attempt -- nothing to carry)"
                        ) as run_span:
            for _ in range(self.max_turns):
                try:
                    c = self.llm(messages, model=self.model, temperature=self.temperature,
                                 tools=self._tools(loaded, resolved))
                except Exception as exc:        # noqa: BLE001
                    res.error = f"provider: {type(exc).__name__}: {exc}"
                    break
                turn = Turn(input_tokens=c.input_tokens, output_tokens=c.output_tokens,
                            cached_tokens=getattr(c, "cached_tokens", 0))

                if not c.tool_calls:
                    turn.tool, turn.detail = "(final)", "answered"
                    res.turns.append(turn)
                    res.answer = (c.text or "").strip()
                    break

                call = c.tool_calls[0]
                name = turn.tool = call["name"]
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [{"id": call.get("id", "call_0"),
                                                 "type": "function",
                                                 "function": {"name": name,
                                                              "arguments": call["arguments"]}}]})
                try:
                    args = json.loads(call["arguments"] or "{}")
                except ValueError:
                    args = {}

                answer_name = str((self.catalogue.get(loaded).spec.get("answer") or {}
                                   ).get("name", "")) if loaded else ""
                recorded = False

                with trace.span(name, **_traceable(args)) as tool_span:
                    if name == "load_skill":
                        key = str(args.get("skill", ""))
                        if key in self.catalogue:
                            loaded = res.skill = key
                            observation = (f"# Skill loaded: {self.catalogue[key].name}"
                                           f"\n\n{self.catalogue[key].body}")
                            turn.detail = key
                        else:
                            observation = (f"error: no skill {key!r}; "
                                           f"available: {sorted(self.catalogue)}")
                    elif not loaded:
                        observation = "error: load a skill before running its steps"
                    elif name == answer_name:

                        turn.detail = "plan submitted"

                        self._ran_since_submit.clear()
                        res.plan = args

                        res.turns.append(turn)
                        recorded = True
                        observation = self._finish(res, resolved, messages)

                        tool_span["called_with"] = _traceable({"plan": args})
                        tool_span["certified"] = bool(res.sql)
                        tool_span["verdict"] = res.verdict or ""
                        if res.refusals:
                            tool_span["refusal"] = res.refusals[-1][:400]
                        if res.sql:
                            tool_span["sql"] = res.sql[:2000]
                        if observation is None:
                            break
                    else:
                        turn.detail = f"{name}({', '.join(sorted(args))})"[:90]
                        observation = self._run_step(loaded, name, args)
                        self._ran_since_submit.add(name)

                        if isinstance(observation, dict):

                            errs = [
                                " -- ".join(part for part in
                                            (f.get("what", ""),
                                             f.get("recommend", "")) if part)
                                for f in (observation.get("findings") or [])
                                if isinstance(f, dict)
                                and f.get("severity") == "error"]
                            self._last_review_errors = errs
                        if name == "resolve":

                            if isinstance(observation, dict):
                                got = dict(observation)
                                got["decomposition"] = args.get("decomposition", {})
                                resolved = self._resolved = got
                                artifact = self._stash(
                                    name, json.dumps(got, ensure_ascii=False, indent=1))

                                shown = dict(got.get("report") or {})
                                shown["route_count"] = len(got.get("route_ids") or [])
                                shown["_artifact"] = str(artifact)
                                observation = json.dumps(shown, ensure_ascii=False)
                    tool_span["detail"] = turn.detail
                    if self._effective is not None:
                        tool_span["called_with"] = _traceable(self._effective)
                        self._effective = None
                    if observation is not None:
                        tool_span["result"] = _traceable_value(observation)

                if not recorded:
                    res.turns.append(turn)

                shown_text = (observation if isinstance(observation, str)
                              else json.dumps(observation, ensure_ascii=False,
                                              default=str))
                if len(shown_text) > self.max_step_output:
                    shown_text = (
                        f"error: {name} produced {len(shown_text)} characters, over "
                        f"the {self.max_step_output} limit. Its output is structured "
                        f"and cannot be truncated without corrupting it. Narrow the "
                        f"request and call it again.")
                messages.append({"role": "tool",
                                 "tool_call_id": call.get("id", "call_0"),
                                 "content": shown_text})
            else:
                res.error = f"did not finish in {self.max_turns} turns"

            if not res.sql and not res.refusals:
                why = "; ".join(self._last_review_errors[:2])
                if not why:
                    report = (self._resolved or {}).get("report") or {}
                    stuck = [f"{a.get('phrase')!r} was ambiguous between "
                             f"{a.get('between')}"
                             for a in (report.get("ambiguous") or ())[:2]]
                    stuck += [f"{u.get('phrase')!r} ({u.get('slot')}) resolved "
                              f"to nothing"
                              for u in (report.get("unresolved") or ())[:2]]
                    why = ("; ".join(stuck) if stuck else
                           "no reason was recorded -- resolution settled "
                           "nothing the plan could be built from")
                res.refusals.append(
                    "the attempt ended without submitting a plan: " + why)
                if not res.error:
                    res.error = "ended without submitting a plan"

            run_span["turns"] = len(res.turns)
            run_span["outcome"] = ("certified" if res.sql else
                                   (res.error or res.verdict or "no sql"))
        return res

    def _finish(self, res: Result, resolved: dict | None,
                messages: list[dict]) -> str | None:
        """Certify, then ALWAYS review, then decide. `None` ends the run.

        THE REVIEWER SEES FAILURES TOO. A checker refusal used to go straight
        back to the planner as raw text -- `checker rejected: GRAIN_FANOUT` --
        which names a violation and not a mistake, and leaves the planner to
        guess which part of its plan caused it. The reviewer's job is exactly
        that translation, so it gets the refusal along with the plan and returns
        something actionable. A deterministic failure is still a finding about
        the plan; it just arrives as an exception rather than as a wrong answer.

        The returned string is what the planner is shown, so the feedback the
        system prompt promised reads the same whether it came from the checker
        or from the reviewer.
        """
        route_ids = list((resolved or {}).get("route_ids") or [])

        skill = self.catalogue.get(res.skill)
        spec_answer = dict((skill.spec.get("answer") or {})) if skill else {}
        gate = str(spec_answer.get("gate") or "")
        if resolved is not None and gate:
            verdict = self._run_step(res.skill, gate,
                                     {"plan": res.plan or {}, "resolved": resolved})

            if not isinstance(verdict, dict):
                res.refusals.append(f"review unavailable: {verdict}")
                res.verdict = "refused"
                res.error = f"review step failed, plan not reviewed: {verdict}"
                return None
            errors = [f for f in verdict.get("findings", [])
                      if isinstance(f, dict) and f.get("severity") == "error"]
            if errors:
                refusal = ("review rejected: " + errors[0].get("what", "")
                           + " -- " + errors[0].get("recommend", ""))

                res.refusals.append(refusal)
                res.verdict = "refused"
                res.error = f"rejected by review: {refusal}"
                return None

        sql, refusal = self._certify(res.plan or {}, route_ids)

        if not refusal:
            res.sql = sql
            res.verdict = "certified"
            return None

        res.refusals.append(refusal)
        res.verdict = "refused"
        res.error = f"could not be compiled or certified: {refusal}"
        return None

_COMMIT: str | None = None

def _commit() -> str:
    """The HEAD sha, with `+dirty` when the tree has uncommitted changes.

    Read once. A trace without it cannot be compared to another trace, because
    nothing says whether they came from the same system.
    """
    global _COMMIT
    if _COMMIT is None:
        import subprocess  # noqa: PLC0415
        try:
            sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, cwd=ROOT,
                                 timeout=5).stdout.strip() or "unknown"
            dirty = subprocess.run(["git", "status", "--porcelain"],
                                   capture_output=True, text=True, cwd=ROOT,
                                   timeout=5).stdout.strip()
            _COMMIT = f"{sha}+dirty" if dirty else sha
        except Exception:                        # noqa: BLE001
            _COMMIT = "unknown"
    return _COMMIT

def _json_field(text: str, key: str) -> str:
    try:
        got = json.loads(text)
    except ValueError:
        return ""
    return str(got.get(key, "")) if isinstance(got, dict) else ""
