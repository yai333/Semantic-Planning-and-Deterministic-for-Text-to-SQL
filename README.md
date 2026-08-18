# Bounded Semantic Planning and Deterministic Compilation for Reliable Enterprise Text-to-SQL

[![arXiv](https://img.shields.io/badge/arXiv-2608.16663-b31b1b.svg)](https://arxiv.org/abs/2608.16663)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reference source for the preprint
[Bounded Semantic Planning and Deterministic Compilation for Reliable
Enterprise Text-to-SQL](https://arxiv.org/abs/2608.16663) (arXiv:2608.16663,
cs.DB).

Semantic path compilation (SPC) relocates relational realization out of the
language model. A multi-turn planner grounds phrases taken verbatim from the
question and selects among governed alternatives that a deterministic resolver
has enumerated for that question. Graph traversal, role predicates, grain
lowering, SQL construction, and static checking are implemented in code, and
the planner never names a table, column, join predicate, or SQL fragment.

## Scope

This is a source-reference snapshot for readers of the paper. It is not a
runnable distribution and not a reproduction package.

Included: the semantic model loader, route resolver, plan representation,
grain-aware compiler, static checker, planner workflow, and the authored ACME
semantic model with its physical mapping.

Not included: benchmark questions, gold labels, database fixtures, the
evaluation harness and judge, tests, run traces, verdict artifacts, credentials,
and the original workflow declaration and decomposition schema. `spc/bench.py`
is retained for reference but imports an `evaluation` package that is outside
this snapshot. Sections 5.6 and 9 of the paper state the reproducibility scope:
the reported measurements cannot be reproduced end to end from public artifacts
alone.

## Layout

| Path | Role | Paper |
| --- | --- | --- |
| `ontology/acme.semantic.yaml` | Concepts, party roles, typed edges carrying business role and per-direction fan-out, and metrics | 4.2 |
| `ontology/acme.mapping.yaml` | Tables, key columns, marker tables, and join and role predicates | 4.2 |
| `spc/ontology.py` | Loads the two layers and merges them under eight validation rules that raise rather than warn | 4.2 |
| `spc/graph.py` | Deterministic route enumeration over the governed graph, with a canonical total order | 4.4 |
| `spc/menu.py` | The question-specific choice set and submission schema exposed to the planner | 4.4, 4.5 |
| `spc/plan.py` | Canonical form of a submitted plan | 4.5 |
| `spc/compile.py` | Grain-aware spine and satellite lowering to a SQLGlot syntax tree | 4.7 |
| `spc/check.py` | Static checks over emitted SQL, with undecidable findings separated from violations | 4.8 |
| `workflow/agent.py` | The multi-turn planner loop and its typed tool surface | 4.1 |
| `workflow/skills/governed_plan/` | The governed planning skill | 4.1, 4.5 |

## Requirements

Python 3.12. The deterministic core depends on `sqlglot` and `PyYAML`. The
model adapter additionally imports `openai`. SQL is rendered for SQLite in the
evaluated configuration.

## Citation

```bibtex
@misc{ai2026bounded,
  title         = {Bounded Semantic Planning and Deterministic Compilation
                   for Reliable Enterprise Text-to-SQL},
  author        = {Ai, Yi},
  year          = {2026},
  eprint        = {2608.16663},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DB},
  doi           = {10.48550/arXiv.2608.16663},
  url           = {https://arxiv.org/abs/2608.16663}
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
