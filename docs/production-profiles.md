# NEXUS production profiles vs library defaults

**Canonical status:** [`CURRENT_STATE.md`](CURRENT_STATE.md)

## Rule

`NEXUSConfig.realizer_backend` defaults to **`synth`** for backward
compatibility with registered Stage 2 experiments. That default is **not** the
recommended production QA profile.

Public entry points (`python -m nexus ask`, `nexus.api.ask`) default to
**`grounded`**. `NEXUSRunner` requires an explicit `ProductionNEXUSConfig`
(no silent `lexical_only`/`synth` default). Callers must select a named
profile when they need historical `synth` semantics — never rely on an
omitted profile to mean production-safe behavior at the low-level
`NEXUSConfig()` constructor.

## Recommended production profile

```python
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner

config = ProductionNEXUSConfig.grounded()
# Optional: inject ER3 / dialogue via stack.pipeline resolvers
runner = NEXUSRunner(graph=graph, config=config)
```

| Factory | Backend | `allow_synth_fallback` | Use when |
|---------|---------|------------------------|----------|
| `lexical_only()` | synth | `true` (historical) | Pure lexical baseline / Stage-2 compatibility |
| `pointer_copy()` | pointer_copy | `false` | Extractive factual answers only |
| `comparison_plan()` | abstractive_plan_v3 | `false` | Comparison questions only |
| `deterministic_render()` | deterministic_render | `false` | Pure proof→statement path render |
| `l1_acceptance()` | l1_acceptance | `false` | **L1 paired publish** — relation path render + node-fact copy + qualitative dual-compare + metric/comparison-plan |
| `grounded()` | grounded_v1 | `false` | Factual + comparison routing with path-render fallback; `require_structured_provenance=True` |

Safe profiles **abstain** when deterministic/constrained realization cannot produce a
supported answer. They do not silently call `SynthesizingModel` / `get_available_model()`.
Experimental overrides may pass `allow_synth_fallback=True` (different `config_hash`).

## Library / experiment default

```python
from nexus.utils.config import NEXUSConfig

config = NEXUSConfig()  # realizer_backend == "synth"
```

Keep `synth` when reproducing historical Stage 2 naturalness/relevance gates.

## Training architectures

See [`training/REJECTED_ARCHITECTURES.json`](../training/REJECTED_ARCHITECTURES.json)
and [`training/architecture_registry.py`](../training/architecture_registry.py).
Rejected sequence-to-sequence Realizers must not be relaunched from loss alone.
