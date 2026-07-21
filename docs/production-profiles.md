# NEXUS production profiles vs library defaults

## Rule

`NEXUSConfig.realizer_backend` defaults to **`synth`** for backward
compatibility with registered Stage 2 experiments. That default is **not** the
recommended production QA profile.

## Recommended production profile

```python
from nexus.pipeline.config import ProductionNEXUSConfig
from nexus.pipeline.runner import NEXUSRunner

config = ProductionNEXUSConfig.grounded()
# Optional: inject ER3 / dialogue via stack.pipeline resolvers
runner = NEXUSRunner(graph=graph, config=config)
```

| Factory | Backend | Use when |
|---------|---------|----------|
| `lexical_only()` | synth | Pure lexical baseline |
| `pointer_copy()` | pointer_copy | Extractive factual answers only |
| `comparison_plan()` | abstractive_plan_v3 | Comparison questions only |
| `grounded()` | grounded_v1 | **Recommended** factual + comparison routing |

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
