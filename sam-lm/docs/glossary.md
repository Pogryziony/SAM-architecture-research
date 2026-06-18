# Glossary

Simple explanations of terms used in the SAM project.

---

## A

### all_required@K
The fraction of examples where **all** required slots (facts needed to answer
the question) appear in the top-K retrieval results. A value of 1.0 means every
required slot is present in the top K for every example.

*Example:* For a 3-hop question needing slots [42, 117, 203],
all_required@32 = 1.0 means all three slots are in the top-32 results.

### any_required@K
The fraction of examples where **at least one** required slot appears in the
top-K retrieval results. Easier to achieve than all_required@K.

### attention
A mechanism in transformer models that lets each position in a sequence look at
all other positions. In SAM, attention is used inside the core for local
syntax and symbol matching.

---

## C

### chain-set retrieval
A retrieval approach that retrieves **all** slots needed for a reasoning chain
at once, rather than retrieving slots one at a time. Trained with a
multi-positive loss that rewards finding the complete set of required slots.

### controlled distractor
A randomly-selected slot (from live memory slots) that is injected into memory
alongside the correct required slots. Used in Experiment 0.13A to measure
noise tolerance under controlled conditions. **Not the same as a realistic
distractor** — controlled distractors are random, while realistic distractors
are top-ranked by an actual retriever and may be semantically misleading.

### core-only
A SAM run mode where **zero memory slots** are used. The model answers
questions using only its own trained weights (no external memory). This is the
**baseline**: if memory cannot beat core-only, memory provides no benefit.

### coverage@K
The fraction of required facts (individual slots) that appear in the top-K
retrieval results, averaged over all examples. Unlike all_required@K, this
counts partial success.

---

## D

### dense weights
The standard way neural networks store knowledge: large matrices of numbers
(floating-point values) where every value is used on every computation.
Contrast with **sparse memory** where only a small subset of values is accessed
per computation.

### distractor
A memory slot that is **not needed** to answer the current question, but gets
included in the retrieved memory anyway. Distractors are noise that the model
must learn to ignore or the gate must learn to suppress.

### dual encoder
A retrieval model with two separate encoders: one for the question and one for
each memory slot. The encoders map both to the same space so similarity can be
computed with dot products.

---

## E

### embedding
A numeric vector (list of numbers) that represents a word, fact, or concept in
a way computers can work with. Similar concepts have similar embeddings.

---

## G

### gate
In SAM's memory integration, a learned scaling factor (between 0 and 1) that
controls how much the retrieved memory influences the model's output. A gate
of 0 means "ignore memory completely." A gate of 1 means "fully use memory."
SAM learns when to open or close the gate during training.

---

## H

### hard negative
A distractor slot that is **semantically similar** to the question but
**factually wrong**. Hard negatives are more dangerous than random distractors
because they look plausible. Realistic retrievers often return hard negatives.

### 1-hop / 2-hop / 3-hop reasoning
The number of separate facts a question needs to answer correctly.
- **1-hop:** Question needs 1 fact (e.g., "What color is X?")
- **2-hop:** Question needs facts A → B (e.g., "X is connected to Y, what color is Y?")
- **3-hop:** Question needs facts A → B → C (e.g., "X → Y → Z, what color is Z?")

Multi-hop reasoning is harder because all required facts must be retrieved
correctly.

---

## L

### latent memory
Memory stored as numeric vectors (embeddings) rather than readable text.
The model combines these vectors mathematically. This is the primary memory
form in SAM — as opposed to text/payload memory where the model reads actual
words.

### learned selector
A small neural network in SAM that looks at retrieved candidate slots and
decides which ones are actually needed. It acts as a filter between retrieval
(which may return many slots) and memory injection (which should only get the
useful ones).

---

## M

### memory integration
How retrieved memory values are combined with the model's own computation.
SAM uses **gated integration**: `output = core_computation + gate * memory_vector`.
Other modes include forced-gate (gate = fixed value) and concat_projection
(concatenate memory instead of adding).

### memory vector / memory norm
The combined value from retrieved slots (aggregated by averaging or
weighted sum). **Memory norm** is a measure of how large this vector is (its
mathematical magnitude). A near-zero memory norm suggests the gate is
suppressing memory.

### mmap
Memory-mapped file I/O. A technique for accessing large files on disk as if
they were in RAM, without loading everything at once. In SAM's longer-term
vision, the memory bank would be mmap-backed to handle knowledge stores
larger than available RAM.

---

## O

### oracle memory
A SAM run mode where the model receives **exactly the correct required slots**
with no distractors. This is an **upper bound** — the best the model could
possibly do if retrieval were perfect. Oracle memory proves the core **can**
use memory, but does not prove the full pipeline works.

### oracle filter
A diagnostic mode where retrieved slots are filtered to keep only those that
are actually required. Used to measure how much accuracy is lost purely
because of distractor slots (separate from retrieval failures).

---

## P

### PKM (Product-Key Memory)
The specific memory addressing scheme used in SAM. A query vector is split
into two parts, each part scored against a codebook. The Cartesian product of
the top results from each codebook gives candidate slots. This allows
efficient lookup in very large memory banks (O(√N) rather than O(N)).

### precision
In the context of slot selection: what fraction of the slots the **selector
chose** are actually required? Precision of 50% means half the selected slots
are distractors.

### prompt
The input text given to a model. In SAM's QA setup, the prompt is the question
plus possible answer choices.

---

## R

### RAG (Retrieval-Augmented Generation)
The common technique of searching a document database and inserting the found
text into a language model's input. SAM is different from RAG because memory
is **integrated into the model's internal computation** (latent vectors,
learned gating), not simply prepended to the input text.

### recall
In the context of slot selection: what fraction of the **required** slots did
the selector find? Recall of 96% means the selector found nearly all needed
slots.

### Recall@K (Rec@K)
In retrieval: what fraction of examples have the correct slot in the top-K
retrieval results? Rec@8 = 99% means the correct slot is in the top 8 results
for 99% of examples.

### residual stream
In transformer models, the main pathway of information flow. Each layer
**adds** its contribution to the stream rather than replacing it. SAM's memory
also adds to the residual stream through the gated integration.

### retrieval
The process of searching the memory bank to find slots that might be relevant
to the current question. In SAM, retrieval happens using learned query
vectors that are matched against slot keys.

### router / routing
The mechanism that decides which memory slots to access. In SAM's product-key
memory, the router is the two-stage codebook matching process.

---

## S

### selector
See **learned selector**.

### slot
One entry in SAM's memory bank. Each slot stores a fact as a value vector
(a learned embedding of the fact's "answer" or content). The memory bank in
current experiments has 1,650 live slots.

### sparse memory
A memory system where only a tiny fraction of the stored values are accessed
for any given computation. Contrast with **dense weights** where all values
are used every time.

### synthetic dataset
Artificially generated data rather than real-world text. SAM's current
experiments use synthetic multi-hop QA: template-generated questions and
fact-chains about made-up entities. This provides clean ground-truth for
controlled experiments.

---

## T

### topK
The number of highest-scoring results to keep from a retrieval or selection
step. topK=8 means "keep the 8 best-matching slots."
