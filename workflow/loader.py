from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parent
SKILLS = ROOT / "skills"
SPEC_FIELDS = {"name", "description", "license", "compatibility",
               "metadata", "allowed-tools"}

class SkillError(ValueError):
    """A skill that cannot be trusted to run. Raised at load, never mid-run."""

def split_frontmatter(text: str, where: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        raise SkillError(f"{where}: must begin with a `---` frontmatter block")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillError(f"{where}: frontmatter is not closed with `---`")
    meta = yaml.safe_load(parts[1])
    if not isinstance(meta, Mapping):
        raise SkillError(f"{where}: frontmatter must be a mapping")
    body = parts[2].strip()
    if not body:
        raise SkillError(f"{where}: the body is empty -- it IS the procedure")
    illegal = sorted(set(meta) - SPEC_FIELDS)
    if illegal:
        raise SkillError(
            f"{where}: non-spec frontmatter keys {illegal} -- the Agent Skills "
            f"standard allows {sorted(SPEC_FIELDS)}; put your own data under "
            f"`metadata:`")
    return dict(meta), body

@dataclass(frozen=True)
class Skill:
    """One workflow. `description` is what an agent chooses on; `body` is what
    it gets once it has chosen."""

    key: str
    name: str
    description: str
    version: int
    path: Path
    body: str
    spec: Mapping[str, Any]
    resources: Mapping[str, Any]

    @property
    def workflow(self) -> list[dict]:
        return list(self.spec.get("workflow") or ())

    def resource(self, key: str) -> Any:
        if key not in self.resources:
            raise SkillError(f"{self.key}: no resource {key!r}; "
                             f"has {sorted(self.resources)}")
        return self.resources[key]

def _import(ref: Path, mod_name: str) -> Any:
    """Import a skill file by path.

    NAMESPACED BY SKILL AND VERSION so two versions coexist, and absent from
    `sys.modules` so a skill's code is reached through the skill that ships it
    or not at all -- an engine that could import it directly could run a step no
    artifact named, and a result could not cite what produced it.
    """
    spec = importlib.util.spec_from_file_location(mod_name, ref)
    module = importlib.util.module_from_spec(spec)      # type: ignore[arg-type]
    try:
        spec.loader.exec_module(module)                 # type: ignore[union-attr]
    except Exception as exc:                            # noqa: BLE001
        raise SkillError(f"{ref.name}: does not import -- "
                         f"{type(exc).__name__}: {exc}") from exc
    return module

def load(key: str, version: int = 1) -> Skill:
    """Read `workflow/skills/<key>/v<version>/SKILL.md`, or raise."""
    directory = SKILLS / key / f"v{version}"
    path = directory / "SKILL.md"
    if not path.exists():
        available = sorted(p.name for p in SKILLS.iterdir()) if SKILLS.exists() else []
        raise SkillError(f"no skill {key!r} at {path}; available: {available}")

    meta, body = split_frontmatter(path.read_text(), f"{key}/SKILL.md")
    for field in ("name", "description"):
        if field not in meta:
            raise SkillError(f"{key}/SKILL.md: frontmatter missing {field!r}")
    spec = dict(meta.get("metadata") or {})

    resources: dict[str, Any] = {}
    for rkey, value in spec.items():
        if not isinstance(value, str) or not value.endswith((".json", ".py")):
            continue
        ref = directory / value
        if not ref.exists():
            raise SkillError(f"{key}/SKILL.md: declares {rkey}={value!r}, absent")
        if value.endswith(".json"):
            resources[rkey] = json.loads(ref.read_text())
            continue
        resources[rkey] = _import(ref, f"spc_skill.{key}_v{version}.{ref.stem}")

    return Skill(key=key, name=str(meta["name"]), description=str(meta["description"]),
                 version=version, path=directory, body=body, spec=spec,
                 resources=resources)

def catalogue(keys: list[str]) -> list[Skill]:
    """Load several, for advertising in a system prompt."""
    return [load(k) for k in keys]

def advertise(skills: list[Skill]) -> str:
    """The `## Available skills` block for a system prompt.

    NAME AND DESCRIPTION ONLY. The body is what loading buys; putting it here
    would make the skill a prompt and the choice meaningless.
    """
    if not skills:
        return ""
    lines = ["## Available skills", ""]
    for s in skills:
        lines.append(f"- **`{s.key}`** — {s.description}")
    lines += ["", "Load the one the question needs before you act. A skill's "
                  "steps are not available to you until you have loaded it."]
    return "\n".join(lines)
