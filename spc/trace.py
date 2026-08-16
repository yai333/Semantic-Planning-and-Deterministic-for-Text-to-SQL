from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent

_SINK: str = os.environ.get("SPC_TRACE", "none").lower()
_MLFLOW: Any = None
_JSONL: Any = None
_ACTIVE: list[dict] = []

def _jsonable(value: Any) -> Any:
    """Traces must never raise. Anything unserialisable becomes its repr."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_jsonable(v) for v in value]
        return repr(value)[:2000]

def _load_env() -> None:
    """Read `.env` if present. Gitignored, so credentials never reach a commit."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

def configure(sink: str | None = None, *, experiment: str = "spc-planner") -> str:
    """Choose a sink. Returns the sink actually in use."""
    global _SINK, _MLFLOW, _JSONL
    _load_env()
    _SINK = (sink or os.environ.get("SPC_TRACE", "none")).lower()

    if _SINK == "mlflow":
        try:
            import mlflow  # noqa: PLC0415

            mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
            mlflow.set_experiment(experiment)
            _MLFLOW = mlflow
        except Exception as exc:  # noqa: BLE001
            print(f"trace: mlflow unavailable ({exc}); falling back to jsonl")
            _SINK = "jsonl"

    if _SINK == "wandb":
        try:
            import wandb  # noqa: PLC0415
            if not os.environ.get("WANDB_API_KEY"):
                raise RuntimeError("WANDB_API_KEY is not set")
            wandb.init(project=experiment, reinit=True)
            _MLFLOW = wandb
        except Exception as exc:  # noqa: BLE001
            print(f"trace: wandb unavailable ({exc}); falling back to jsonl")
            _SINK = "jsonl"

    if _SINK == "jsonl":
        path = ROOT / "results" / "traces.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        _JSONL = path.open("a")

    return _SINK

def enabled() -> bool:
    return _SINK not in ("none", "")

@contextmanager
def span(stage: str, **inputs: Any) -> Iterator[dict]:
    """Record one stage. Yields a dict the caller fills with its outputs.

        with span("decompose", question=q) as s:
            d = decompose(q)
            s["subjects"] = d.subjects

    A no-op when tracing is off, so instrumentation costs nothing in the
    deterministic suites and cannot alter a measured result.
    """
    record: dict[str, Any] = {"stage": stage, "inputs": _jsonable(inputs)}
    if not enabled():
        yield record
        return

    start = time.perf_counter()
    _ACTIVE.append(record)

    stack = ExitStack()
    live = None
    if _SINK == "mlflow" and _MLFLOW is not None:
        try:
            live = stack.enter_context(_MLFLOW.start_span(name=stage))
        except Exception as exc:  # noqa: BLE001
            print(f"trace: span open failed ({exc})")
            live = None
    try:
        yield record
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["ms"] = round((time.perf_counter() - start) * 1000, 2)
        _ACTIVE.pop()
        if live is not None:
            payload = _jsonable(record)
            try:
                live.set_inputs(payload.get("inputs", {}))
                live.set_outputs({k: v for k, v in payload.items()
                                  if k not in ("stage", "inputs")})
            except Exception as exc:  # noqa: BLE001
                print(f"trace: emit failed ({exc})")
            stack.close()
        else:
            stack.close()
            _emit(record)

def _emit(record: dict) -> None:
    payload = _jsonable(record)
    try:
        if _SINK == "mlflow" and _MLFLOW is not None:
            with _MLFLOW.start_span(name=record["stage"]) as s:
                s.set_inputs(payload.get("inputs", {}))
                s.set_outputs({k: v for k, v in payload.items()
                               if k not in ("stage", "inputs")})
        elif _SINK == "wandb" and _MLFLOW is not None:
            _MLFLOW.log({f"{record['stage']}/{k}": v for k, v in payload.items()
                         if isinstance(v, (int, float))})
            _MLFLOW.log({f"{record['stage']}_trace": json.dumps(payload)[:8000]})
        elif _SINK == "jsonl" and _JSONL is not None:
            _JSONL.write(json.dumps(payload) + "\n")
            _JSONL.flush()
    except Exception as exc:  # noqa: BLE001
        print(f"trace: emit failed ({exc})")
