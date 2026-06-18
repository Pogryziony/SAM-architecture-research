# The SAM Thesis

*Why SAM exists and what problem it tries to solve.*

## Simple version

Most large language models store **all knowledge inside dense weights** — giant
matrices of numbers that must be loaded from memory and processed for every
single prediction. This is expensive: it uses a lot of RAM (or VRAM on GPUs),
a lot of bandwidth, and a lot of energy.

SAM tries a different approach: use a **small active system** that does the
thinking, and a **large memory bank** that stores knowledge. The thinking part
stays the same size regardless of how much knowledge is stored. When the model
needs a fact, it looks up only the relevant pieces from the memory bank, not
everything.

Instead of making one huge model that keeps most knowledge inside dense
weights, SAM tries to use a smaller active system that can select and read
only the memory it needs.

## Why huge dense weights can be expensive

A standard transformer language model stores knowledge in its weight matrices.
Every parameter participates in every computation. This means:

1. **Streaming cost**: For each token generated, the entire model's weights
   must be streamed from RAM/VRAM into the processor. A 350M parameter model
   in FP16 format streams about 700 MB per token.

2. **Scaling cost**: To store more knowledge, you need more parameters.
   More parameters → more streaming → more cost. Knowledge and computation
   are locked together.

3. **Update cost**: Changing one fact means retraining (or fine-tuning) the
   entire model. There's no clean way to edit a single piece of knowledge.

## Why RAM/VRAM bandwidth matters

The speed of moving data from memory to the processor is often the bottleneck,
not the speed of the processor itself. This is called the **memory wall**.

- **DDR5 RAM** (typical in consumer PCs): ~40-60 GB/s
- **HBM2e VRAM** (high-end GPUs): ~1.5-2 TB/s
- **NVMe SSD**: ~3-7 GB/s (sequential), much slower for random access

A dense 120M-parameter model streaming FP16 weights at 240 MB/token on DDR5
RAM would max out around 170-250 tokens per second — before any computation.

SAM's approach: stream the same 240 MB for the core, but only read a few
kilobytes of memory per token (the selected slots). The memory bank is
RAM-resident but sparsely accessed.

## What "CPU-first sparse-memory architecture" means

**CPU-first**: Designed to run efficiently on consumer CPUs with DDR5 RAM,
not requiring expensive GPUs with HBM VRAM. The target is a system you can
run on a normal desktop or laptop.

**Sparse memory**: Only a tiny fraction of the memory bank is touched for
each computation. With 1,650 slots and topK=8, only 0.5% of memory is read
per token. With a larger bank (millions of slots), the fraction drops further.

**Active compute stays small**: The core model that does reasoning is modest
(~16M parameters in current experiments). It doesn't grow when you add more
knowledge.

## Why SAM is not "just RAG"

RAG (Retrieval-Augmented Generation) is a popular technique where a search
system finds relevant documents and prepends them to the model's input text.
The model then reads the text and generates an answer.

SAM is different in several critical ways:

1. **Integration depth**: RAG prepends text to the input. SAM injects memory
   **into the model's internal computation** at specific layers via learned
   gating. Memory is part of the model's reasoning process, not just extra
   context.

2. **Memory format**: RAG uses text documents. SAM uses **latent vectors**
   (numeric embeddings) that are combined mathematically inside the model.
   This is more compact and allows the model to blend information from
   multiple memory slots.

3. **Training**: In RAG, the retriever is typically separate from the model.
   In SAM, the retrieval keys, slot values, and memory gate are all trained
   together (or in coordinated stages), creating a tighter coupling.

4. **Efficiency target**: RAG still requires the model to process all retrieved
   text through its full computation. SAM's memory is pre-encoded as latent
   vectors and injected with a lightweight integration step.

## Why SAM is not trying to beat GPT at small scale

SAM's current experiments use:
- ~16M parameters (GPT-2 small was 124M, GPT-3 was 175B)
- A synthetic dataset with 853 vocabulary tokens
- Template-generated questions about made-up entities

The goal is **not** to achieve state-of-the-art benchmark scores. The goal is
to test architectural hypotheses under controlled conditions:

1. Can a small core use external memory for reasoning? (Yes — oracle experiments)
2. Can memory be retrieved correctly? (Yes — chain-set retriever)
3. Can the model tolerate noisy memory? (Yes — controlled distractors up to +8)
4. Can realistic retrieval work end-to-end? (Not yet)

## What success would mean at small scale

Success at the current scale would mean demonstrating:

- A ~16M-parameter model that achieves near-perfect multi-hop QA accuracy
  when realistic retrieval provides clean enough memory.
- Learned selection that can distinguish required slots from realistic distractors.
- Memory integration that is robust to the types of noise that actual
  retrievers produce.

This would validate the **architectural pattern**, not the final product.
Scaling to practical sizes would be a separate, future effort.

## What success would NOT mean

- It would NOT mean SAM beats GPT or DeepSeek at any practical task.
- It would NOT mean the architecture is proven at scale.
- It would NOT mean SAM is ready for production use.
- It would NOT mean the specific mechanisms (latent averaging, gated sum,
  product-key lookup) are the final design.

Success at this stage means: the core ideas are not fundamentally broken,
and it is worth investing more effort in scaling and improving them.

## The honest starting point

The thesis starts from an observation about cost:

- Dense models lock knowledge and computation together. To know more, you
  must compute more. To update knowledge, you must retrain.

This observation is well-established. Product-key memory and sparse
mixture-of-experts architectures have shown that separating knowledge from
computation can work at scale (Meta's Memory Layers at Scale, 2024).

SAM's novel bets are:
1. Can this work efficiently on **CPU + DDR5 RAM** (not GPU + HBM)?
2. Can the core do **multi-hop reasoning** over latent retrieved memory
   (not just single-step factual recall)?
3. Can learned selection provide clean enough memory for the gate to open?

These questions are being tested one experiment at a time.

---

*Last updated: 2026-06-18*
