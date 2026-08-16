# Semantic Planning and Deterministic Compilation for Text-to-SQL

This repository contains an illustrative reference implementation of semantic
planning and deterministic SQL compilation. The research paper is not included.
This snapshot is provided for source-code reference and is not a standalone
distribution.

## Contents

- `spc/`: ontology loading, path resolution, semantic planning, SQL compilation,
  grain checking, and model adapter code.
- `workflow/`: the governed planning agent and its deterministic tool scripts.
- `ontology/acme.semantic.yaml`: portable ACME insurance semantics.
- `ontology/acme.mapping.yaml`: physical warehouse mappings.

The original workflow declaration and decomposition schema are omitted from
this reference snapshot. Tests, benchmark questions, gold labels, databases,
evaluation results, traces, and credentials are also intentionally excluded.

## License

Released under the MIT License. See `LICENSE`.
