# Result schema

Systems must emit **`nexus-eval-result-v1`** as implemented in:

- `nexus/evaluation/schema.py`
- `nexus/evaluation/validate.py`

Minimum guarantees:

- one mutually exclusive terminal outcome per question
- regenerable aggregates from per-question rows
- metric denominators exposed
- `comparison_mode` ∈ {`controlled`, `system_level`, ``}
