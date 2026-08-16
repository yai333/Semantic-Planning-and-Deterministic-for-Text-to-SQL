from __future__ import annotations

import json
import sys
from typing import Any

def subjects(question: str, skills: Any) -> dict:
    """The subject menu for `question`."""
    got = skills.list_subjects(question)
    return got.data if hasattr(got, "data") else got

def _main(argv: list[str]) -> int:
    sys.path.insert(0, __file__.split("/workflow/")[0])
    from spc.skills import default_skills
    print(json.dumps(subjects(argv[1] if len(argv) > 1 else "",
                              default_skills()), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
