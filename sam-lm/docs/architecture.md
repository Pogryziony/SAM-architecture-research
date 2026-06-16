# SAM Architecture

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

## Validation Results (Experiment 0.6)

The POC was validated through a series of controlled experiments. Key result:

**SAM oracle_memory achieves 99.9% accuracy vs 68.7% core-only (+31pp).**

| Model | Overall | Single | Two-hop | Three-hop |
|-------|---------|--------|---------|-----------|
| Dense baseline (14.6M) | 68.7% | 91.5% | 71.1% | 22.0% |
| SAM core_only (15.7M) | 68.7% | 91.5% | 71.1% | 22.0% |
| **SAM oracle_memory** | **99.9%** | **99.5%** | **100%** | **100%** |

### Findings:
- **SAM core matches dense** at equal parameter count — no architecture disadvantage
- **Memory provides +31pp** — the core CAN compose retrieved latent vectors
- **Three-hop solved** — goes from 22% → 100% with memory injection
- **Retrieval solved** — dual encoder achieves 99.3% Rec@8 on dense shared-slot dataset
- **Product-key candidate generation works** (100% subkey accuracy) but ranking needs more data
- The initial failure mode ("core cannot compose" at 60-70% prior) is NOT confirmed

### Next: SAM retrieved-memory with dual encoder backend, then memory scaling.
