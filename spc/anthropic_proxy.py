from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from spc.bench import Completion, LLMUnavailable

_VERSION = "2023-06-01"
_MAX_TOKENS = 8192

def _credentials() -> tuple[str, str]:
    key, base = os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_API_BASE")
    if key and base:
        return key, base
    local = Path(__file__).resolve().parent.parent / ".env"
    if local.exists():
        env = {}
        for line in local.read_text().splitlines():
            m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*)\s*", line)
            if m:
                env[m.group(1)] = m.group(2).strip("\"'")
        if env.get("OPENAI_API_KEY") and env.get("OPENAI_API_BASE"):
            return env["OPENAI_API_KEY"], env["OPENAI_API_BASE"]
    raise LLMUnavailable("anthropic_complete needs OPENAI_API_BASE (the "
                         "reseller endpoint that serves /v1/messages)")

def _to_blocks(messages: Sequence[dict]) -> tuple[str, list[dict]]:
    """OpenAI-style history -> (system, Anthropic messages).

    role:tool becomes a USER message carrying tool_result blocks; consecutive
    user-turns (only tool results, in our loop) merge into one message, which
    keeps the tool_result immediately after the assistant tool_use that
    spawned it -- the exact invariant the translated endpoint violated.
    """
    system: list[str] = []
    out: list[dict] = []

    def flush_user(blocks: list[dict]) -> None:
        if blocks:
            out.append({"role": "user", "blocks": blocks})

    pending: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content"):
                system.append(str(m["content"]))
        elif role == "user":
            flush_user(pending); pending = []
            out.append({"role": "user", "blocks": [
                {"type": "text", "text": str(m.get("content") or "")}]})
        elif role == "assistant":
            flush_user(pending); pending = []
            blocks = []
            if m.get("content"):
                blocks.append({"type": "text", "text": str(m["content"])})
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:                    # noqa: BLE001
                    args = {}
                blocks.append({"type": "tool_use", "id": call.get("id") or "call_0",
                               "name": fn.get("name") or "", "input": args})
            out.append({"role": "assistant", "blocks": blocks})
        elif role == "tool":

            pending.append({"type": "tool_result",
                            "tool_use_id": m.get("tool_call_id") or "call_0",
                            "content": str(m.get("content") or "")})
    flush_user(pending)
    return "\n\n".join(system), [{"role": m["role"], "content": m["blocks"]}
                                 for m in out]

def _to_anthropic_tools(tools: Sequence[dict]) -> list[dict]:
    out = []
    for t in tools:
        fn = t.get("function") if t.get("type") == "function" else t
        if not fn:
            continue
        out.append({"name": fn.get("name") or "",
                    "description": fn.get("description") or "",
                    "input_schema": fn.get("parameters")
                    or {"type": "object", "properties": {}}})
    return out

def anthropic_complete(
    messages: Sequence[dict],
    *,
    model: str,
    temperature: float = 0.0,
    tools: Sequence[dict] | None = None,
    response_format: dict | None = None,   # noqa: ARG001
    tool_choice: dict | str | None = None,
) -> Completion:
    """`complete`'s signature and return type, over /v1/messages.

    `response_format` is accepted and ignored ON PURPOSE: on this proxy it is
    not enforced for Anthropic models (measured; see complete's docstring),
    and every constrained output here already travels as a forced tool call.
    """
    key, base = _credentials()
    system, conv = _to_blocks(messages)
    payload: dict[str, Any] = {"model": model, "max_tokens": _MAX_TOKENS,
                               "messages": conv, "temperature": temperature}
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = _to_anthropic_tools(tools)
        if isinstance(tool_choice, dict):
            name = (tool_choice.get("function") or {}).get("name") \
                or tool_choice.get("name")
            payload["tool_choice"] = ({"type": "tool", "name": name}
                                      if name else {"type": "auto"})
        elif tool_choice == "required":
            payload["tool_choice"] = {"type": "any"}

    def _post(pl: dict) -> dict:
        req = urllib.request.Request(
            base.rstrip("/").removesuffix("/v1") + "/v1/messages",
            data=json.dumps(pl).encode(),
            headers={"x-api-key": key, "anthropic-version": _VERSION,
                     "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"anthropic /v1/messages {exc.code}: "
                               f"{exc.read().decode()[:400]}") from exc

    dropped = False
    try:
        data = _post(payload)
    except RuntimeError as exc:
        if "temperature" not in str(exc) or "deprecat" not in str(exc):
            raise
        payload.pop("temperature", None)
        data = _post(payload)
        dropped = True

    text_parts, calls = [], []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text") or "")
        elif block.get("type") == "tool_use":
            calls.append({"id": block.get("id") or "call_0",
                          "name": block.get("name") or "",
                          "arguments": json.dumps(block.get("input") or {})})
    usage = data.get("usage") or {}
    return Completion(
        text="".join(text_parts),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cached_tokens=usage.get("cache_read_input_tokens", 0),
        tool_calls=tuple(calls),
        temperature_dropped=dropped,
        raw=data)
