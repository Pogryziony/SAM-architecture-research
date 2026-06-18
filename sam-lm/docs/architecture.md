# Architecture

How the current SAM implementation works, module by module.

> See [thesis.md](thesis.md) for *why* SAM exists. This page covers *how* the
> current code implements it.

## High-level overview

```
Input question ─► Tokenizer ─► [Core Transformer with Memory Layers] ─► Answer
                                   │
                      Retriever ◄──┤ (external text query or hidden-state query)
                          │        │
                      Top-K slots   │
                          │        │
                      Aggregation ─┘ (uniform_mean, score_weighted, learned_selector, etc.)
                          │
                      Gate ───────── (learned scalar or forced)
                          │
                      Add to residual stream
```

SAM is a small transformer where certain layers have an attached **memory
block**. At each memory-block layer:

1. The transformer's hidden state is projected into a **query vector**
2. The query is used to **retrieve** relevant slots from the memory bank
3. Retrieved slot values are **aggregated** into a single memory vector
4. A learned **gate** controls how much the memory vector influences the output
5. The gated memory is **added** to the residual stream

## Thesis

SAM (Sparse Associative Memory) decouples knowledge from computation in language models.

Dense LLMs store reasoning capability and factual knowledge in the same streamed weights. Every parameter participates in every forward pass. This creates a fundamental tension:

- To improve reasoning, you want deep, expressive computation.
- To store more facts, you need more parameters.
- Every additional fact increases interference with learned reasoning patterns.

SAM separates these concerns:

1. **Reasoning core**: A small dense Transformer (tens of millions of parameters) that implements
   language understanding, attention, and multi-step inference.
2. **Associative memory**: A large sparse product-key memory (millions to billions of slots)
   that stores facts as key-value pairs. Memory is read-only during inference and retrieved
   by learned product-key lookup.

## Why Dense Models Entangle Knowledge and Computation

In a standard Transformer, the feed-forward layers (MLPs) serve as a form of associative memory.
Empirical evidence (Geva et al., 2021; Meng et al., 2022) shows that MLP layers store factual
knowledge in their weight matrices. But these same layers also implement the non-linear
transformations needed for reasoning -- composing concepts, resolving references, tracking state.

This entanglement means:

- **Scaling knowledge requires scaling compute**: More facts -> larger MLPs -> more FLOPs per token.
- **Knowledge updates require retraining**: Changing a fact means modifying distributed weights.
- **Interference**: New facts can degrade existing reasoning patterns.

## Why Product-Key Memory

Product-key memory (Lample et al., 2019) addresses the core challenge of large-scale
associative retrieval:

1. **O(sqrt(N)) lookup cost**: With N total slots, retrieval requires scoring only 2 * sqrt(N)
   key vectors, not N. This makes billion-slot memories tractable.

2. **Differentiable**: The top-k selection and softmax weighting are fully differentiable,
   enabling end-to-end training with gradient descent.

3. **Structured addressing**: The cartesian product decomposition
   `slot_id = k1 * num_subkeys + k2` provides a natural addressing scheme that can be
   learned through standard optimization.

4. **Content-addressable**: Retrieval is driven by the query, not a fixed index. The model
   learns what to retrieve based on the current context.

## How the POC Differs from the Final Architecture

This proof-of-concept implements the minimal version needed to test the core thesis.
The following are **intentionally excluded**:

| Component          | POC                                 | Final Architecture                      |
|--------------------|-------------------------------------|-----------------------------------------|
| Memory size        | 1M slots (1024 subkeys)             | 32M+ slots                              |
| Value storage      | Embedding table (vocab x value_dim) | Per-slot trainable vectors, int4 quantized |
| Memory backend     | In-memory PyTorch tensors          | mmap-backed, persistent                 |
| Retrieval          | Single query per sequence           | Tokenwise, adaptive re-query            |
| Attention          | Full causal                         | Local + global, memory-augmented cross-attention |
| Weight format      | fp32                                | Ternary (1.58-bit) core weights         |
| Training           | Single GPU                          | Multi-GPU, pipeline parallelism         |
| Application        | Synthetic facts                     | Real codebases, API docs, error logs    |
| Agent loop         | None                                | Playwright, code repair, tool use       |

## Memory Modes

1. **core_only**: No memory layers enabled. The model must store all knowledge in its
   dense weights. This is the capacity floor -- it should be weak on knowledge-intensive tasks.

2. **oracle_memory**: The correct required slot values are injected directly, bypassing
   retrieval. This is the upper bound -- it tells us whether the core *can* use memory
   when retrieval is perfect.

3. **retrieved_memory**: Memory is retrieved by learned product-key lookup. This is the
   real SAM mechanism. The core learns what to query and how to integrate retrieved values.

4. **random_memory**: Random live slot values are injected. This is a placebo control --
   if random memory improves performance, the gating mechanism alone (not content) is
   providing benefit.

5. **retrieved_memory_external_text_query**: A standalone retriever (dual encoder or
   chain-set) encodes raw question text into a query vector. Retrieved slots provide
   fact text as input tokens. Decouples retrieval training from SAM core training.

6. **oracle_text_memory**: Oracle facts injected as input text tokens. Upper bound
   for text-based memory without product-key retrieval.

## Retriever Architectures

### Dual Encoder (Experiment 0.6)

Question → query_encoder → query_vector · slot_embedding → topK slots.

Trained with InfoNCE on the first required slot only. Achieves 99.3% any_required@8 but
fails catastrophically on all_required@K for multi-hop tasks (3-hop = 0%).

### Chain-Set BCE Retriever (Experiment 0.11)

Question → query_encoder → query_vector · slot_embedding → topK slots.

Trained with multi-positive BCE loss treating ALL required slots as positives against
all live slots. This directly optimizes complete chain retrieval.

### Slot Graph Expander (Experiment 0.11)

Anchor slots → MLP transition scorer → neighbor slots → union.

Two-stage: retrieve anchor slots from question, expand to related chain slots via
learned slot-to-slot transitions.

## Validation Results

### Experiment 0.6 — Oracle Memory Validation

**SAM oracle_memory achieves 99.9% accuracy vs 68.7% core-only (+31pp).**

| Model | Overall | Single | Two-hop | Three-hop |
|-------|---------|--------|---------|-----------|
| Dense baseline (14.6M) | 68.7% | 91.5% | 71.1% | 22.0% |
| SAM core_only (15.7M) | 68.7% | 91.5% | 71.1% | 22.0% |
| **SAM oracle_memory** | **99.9%** | **99.5%** | **100%** | **100%** |

### Experiment 0.10 — Required-Set Retrieval Diagnostic

The dual encoder retriever (trained on single-slot InfoNCE) achieves near-perfect
any_required@K but fails on all_required@K for multi-hop tasks:

| K | all_required@K | 1-hop | 2-hop | 3-hop |
|---|---------------|-------|-------|-------|
| 1 | 0.2408 | 0.915 | 0.000 | 0.000 |
| 8 | 0.2634 | 1.000 | 0.001 | 0.000 |
| 64 | 0.2729 | 1.000 | 0.017 | 0.000 |

73% of examples had required slots completely absent from top64. The retriever
finds the output slot but misses intermediate chain slots.

### Experiment 0.11 — Chain-Aware Retrieval

The chain-set BCE retriever eliminates the multi-hop retrieval bottleneck:

| K | all_required@K | 2-hop all@K | 3-hop all@K |
|---|---------------|-------------|-------------|
| 8 | 0.8103 | 0.8514 | 0.3433 |
| 16 | 0.9653 | 0.9600 | 0.9267 |
| 32 | **1.0000** | **1.0000** | **1.0000** |

**Retrieval is solved.** All three retrieval gates (A/B/C) pass at K=32.

However, SAM with chain-aware retrieved memory achieves identical accuracy to
core_only and random_memory (68.7%). The model does not benefit from perfect
retrieval — reasoning capacity or memory integration is the new bottleneck.

### Full SAM Mode Comparison (val set, 3800 examples)

| Mode | Overall | 1-hop | 2-hop | 3-hop | Recall@32 |
|------|---------|-------|-------|-------|-----------|
| core_only | 0.6874 | 0.915 | 0.711 | 0.22 | — |
| random_memory | 0.6874 | 0.915 | 0.711 | 0.22 | — |
| dual_encoder retrieved | 0.6868 | 0.915 | 0.711 | 0.22 | 1.0* |
| **chain_set retrieved** | **0.6866** | **0.915** | **0.710** | **0.22** | **1.0†** |
| oracle_memory | 0.9987 | 0.995 | 1.000 | 1.00 | — |

\* any_required@32 = 1.0 (all_required@32 = 0.27)\
† any_required@32 = 1.0, all_required@32 = 1.0

### Key Findings:

- **Core CAN compose** — Oracle memory proves the architecture works (99.9%)
- **Retrieval is solved** — Chain-set BCE achieves 100% all_required@32
- **Retrieval ≠ Accuracy** — Perfect retrieval doesn't improve SAM QA accuracy
- **SAM is the new bottleneck** — Memory integration or capacity limits benefit

## Memory Integration Modes (Gate Stress)

How the retrieved memory vector is combined with the transformer's hidden state.
Controlled by `memory_integration_mode`:

| Mode | How it works | When used |
|------|-------------|-----------|
| `integrate_gated` (normal_gate) | `out = x + σ(gate) * mem` — learned scalar gate between 0 and 1 | Default mode. The core can learn to use or ignore memory. |
| `forced_gate_1` | gate = 1.0 always — memory always forced in | Stress test: does forcing memory use help? |
| `forced_gate_scalar` | gate = fixed value (e.g., 0.5) | Test partial gate thresholds |
| `concat_projection` | Concatenate [hidden, memory], project back to hidden size | Alternative integration architecture |

**From Experiment 0.13A:** Forced gate did not significantly change results with
controlled random distractors — gate suppression is NOT the primary bottleneck.

## Controlled Noisy Memory Path (0.13A/0.13B)

Added to test noise tolerance. Controlled by `memory_noise_mode`:

| Mode | What it injects |
|------|----------------|
| `oracle_plus_distractors` | Required slots + N random distractors from live slots |
| `oracle_plus_realistic_distractors` | Required slots + N distractors from the retriever's top-K results |

The `_build_noisy_memory_slots()` method:
1. Takes the gold required slot IDs
2. Samples N random live slots (excluding required ones) — or N realistic distractors
3. Combines required + distractors
4. Queries PKM for their value vectors
5. Aggregates and injects **through the same memory integration code** as normal retrieval

This is NOT a separate oracle shortcut. It uses the identical memory path with
controlled slot content — making it a fair test of the integration step.

## Slot Selector (`sam/model/slot_selector.py`)

A small (3-layer MLP) neural network that looks at retrieved candidate slots
and predicts which ones are actually needed for the question.

**Input features per slot:**
- Query embedding (from retriever or SAM hidden state)
- Slot embedding (from retriever's slot table)
- Slot value vector (from PKM)
- Retrieval score, rank position, score margin from top
- Optional: hop count embedding

**Output:** One logit per candidate slot (probability that slot is required).

**Training:** BCE (Binary Cross-Entropy) loss — each slot is a binary
classification problem (required or not).

**Results (0.12):**
- Recall: 96.6% — finds nearly all required slots
- Precision: 50% — selects about twice as many slots as needed
- ~1.75 distractors injected (vs ~1.89 required)
- But QA accuracy = core_only (68.74%) — the reason for 0.13A investigation

## Current Status of Components

| Component | Status | Experiment |
|-----------|--------|-----------|
| Product-key memory | Working, used in all experiments | Since 0.6 |
| Dual encoder retriever | Working (99% Rec@8 on dense data) | 0.5-0.6 |
| Chain-set retriever | Working (all_required@32 = 100%) | 0.11 |
| Oracle memory path | Validated (99.87%) | 0.6 |
| Controlled noise path | Implemented | 0.13A |
| Learned selector | Partial (96.6% recall, 50% precision) | 0.12 |
| Realistic distractor path | Implemented | 0.13B (in progress) |
| Efficiency measurements | Not implemented | — |
