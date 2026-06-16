# SAM — Sparse Associative Memory Language Model
### Architecture, falsification plan, and staged implementation roadmap

Status: research/engineering plan, not a hype document. Every load-bearing claim is tagged:
`[proven]` (established by prior work), `[plausible]` (reasonable, unproven at this config), `[speculative]` (weak/no precedent), `[impl-risk]` (engineering feasibility risk), `[likely-false]`.

---

## 1. Executive verdict

**Is SAM worth pursuing?** Yes, as a falsifiable POC — but the bar for "success" must be set on multi-hop reasoning, not factual recall. The recall win is nearly free and would be a *false positive* for the broader thesis.

**Why the thesis is partly de-risked already.** The "decouple knowledge from computation" half is not novel and is largely validated. Meta's *Memory Layers at Scale* (Berges et al., Dec 2024) showed trainable product-key key-value memory layers beating dense models with **>2× the compute budget** and matching MoE at equal compute/params, with gains *most pronounced on factual tasks*, scaling to 128B memory params. Product-key memory itself dates to Lample et al. 2019; PEER (He, 2024) extended it. So "small active core + large sparse memory → more knowledge per FLOP" is `[proven]` for factual recall.

**What is genuinely novel in SAM (the unproven bets):**
1. CPU-native, mmap-backed runtime with a hard 64 GB RAM budget (prior work is GPU-resident). `[plausible]`
2. Adaptive multi-hop re-query over *latent* memory (retrieve → integrate → re-query). `[speculative]`
3. Dual latent + exact-payload memory in one addressing scheme. `[plausible]`
4. Surprise-gated input (BLT-style) wired to a memory model. `[plausible]`, deferred.
5. Ternary core at ~120M. `[impl-risk]` — see failure mode 3.

**Strongest defensible version of the thesis.** *At a fixed streamed-bytes/token budget equal to a ~120M dense core, SAM matches the knowledge-task accuracy of a 2–4× larger dense Transformer, on CPU, while keeping knowledge editable and provenance-tracked.* This is mostly proven for recall; SAM's bet is extending it to (a) CPU runtime economics and (b) reasoning, not just lookup.

**Most likely failure mode (≈60–70% prior).** Memory lifts factual recall sharply, but the tiny core cannot *compose* 2–4 retrieved latent vectors into a reasoning chain. The **oracle-memory ablation** reveals the bottleneck is core reasoning capacity, not retrieval — which more memory cannot fix. SAM then collapses to exactly the prompt's stated failure case: *a retrieval-heavy specialist*, not a general base architecture.

**Secondary failure modes.**
- **Training-time optimizer memory + cold slots** `[impl-risk]`: 32M slots × 256-dim FP16 values + Adam moments ≈ 48 GB of optimizer state; and under a realistic POC token budget most slots receive ~0 gradient (Zipfian routing), so effective capacity ≪ nominal. Mitigation: sparse/embedding-bag optimizer, start at 8–16M slots.
- **Ternary core at 120M** `[impl-risk]`: BitNet b1.58 parity with FP16 is a **≥3B-parameter** phenomenon; below that, reported degradation is material (≈2–3 pts with a 16→1.58 schedule, larger without). Treat A8 as a late, optional, separately-gated experiment — never assume it.

**Bottom line:** Build Phases 1–6. Gate the program on the multi-hop result and the oracle-memory diagnostic. Do **not** scale on a recall win alone.

---

## 2. Architecture summary (one page)

```
                              token t
                                │
              ┌─────────────────▼──────────────────┐
              │  (optional) surprise-gated patcher  │  small byte/token entropy LM
              │  low-entropy → compress, high → keep│  allocates core compute (BLT-style)
              └─────────────────┬──────────────────┘
                                │ residual stream  x_t  (d = 512–768)
   ┌────────────────────────────────────────────────────────────────┐
   │  REASONING CORE  (12–18 blocks, ~120M params, runs every token) │
   │                                                                  │
   │  every block:   selective SSM / linear-recurrent mixer (Mamba-2) │
   │  ~every 4:      sliding-window local attention (exact copy/syntax)│
   │  ~every 3:      MEMORY LAYER  ───────────────┐                   │
   │  every block:   RMSNorm, residual, FF/SwiGLU │                   │
   └──────────────────────────────────────────────┼───────────────────┘
                                                   │ query q_t
                          ┌────────────────────────▼─────────────────────────┐
                          │   PRODUCT-KEY ASSOCIATIVE MEMORY (sparse)         │
                          │   split q → (q1,q2); top-A vs C1, top-B vs C2;     │
                          │   A×B candidates → exact top-k slots               │
                          │                                                   │
                          │   mode A latent:  read int8/int4 value vectors    │
                          │   mode B payload: read exact strings via pointer  │
                          │   gate g·Σ wᵢ vᵢ  ──► back into residual stream    │
                          │   mmap bank (RAM-resident) + hot-slot cache        │
                          └────────────────────────┬─────────────────────────┘
                                                   │
                  ┌────────────────────────────────▼─────────────────────────┐
                  │  ADAPTIVE RE-QUERY CONTROLLER (halt head, PonderNet-style) │
                  │  decide: no-lookup / lookup / re-query / halt / abstain    │
                  └────────────────────────────────┬─────────────────────────┘
                                                   │
                                          next-token / action head
                                  (LM head; optional READ/SEARCH/PATCH/RUN/DIAGNOSE/ABSTAIN)
```

**Data flow per token:** input → (optional patcher) → core blocks; at each memory layer the block emits a query, the product-key memory returns top-k latent values (and/or exact payloads), a learned gate injects them into the residual stream; the halt head decides whether to re-query or proceed; final head emits a token or a structured action. Knowledge lives in the bank (tens of GB, sparsely touched); computation lives in the core (small, dense, every token).

---

## 3. Detailed architecture

### 3.1 Reasoning core

Hybrid block, repeated 12–18× (POC: 12), `d` = 768, designed so the core *cannot* hold broad world knowledge (that is the point — if a core-only model does well on knowledge tasks, the thesis is untested).

| Component | Choice | Role | Status |
|---|---|---|---|
| Sequence mixer (every block) | Selective SSM, Mamba-2 style | linear-time state, long local context cheaply | `[proven]` primitive |
| Local attention (~every 4th block) | sliding window 128–512 | exact copy, local syntax, symbol agreement | `[proven]` (Mistral/Longformer) |
| Memory layer (~every 3rd block) | product-key (§3.2) | knowledge injection | `[proven]` primitive |
| Channel mixer | SwiGLU FF (or small sparse-expert) | local composition | `[proven]` |
| Norm | RMSNorm, pre-norm | stability | `[proven]` |
| Heads | LM head; optional verifier/value + action heads | generation + dev tasks | mixed |

Deliberate capacity ceiling: keep FF width modest and rely on memory for facts. Core-only baseline (A1) must be *weak* on knowledge or the experiment is invalid.

### 3.2 Product-key associative memory

Addressing (per memory layer, per token):
1. Project residual → query `q ∈ R^{d_k}`; split into `q1, q2 ∈ R^{d_k/2}`.
2. Two sub-key codebooks `C1, C2`, each with `C = ceil(√N)` entries (padded to a power of two, e.g. 8192). Score `q1·C1`, `q2·C2`.
3. Take top-A from C1, top-B from C2. Candidate set = A×B pairs. Combined score is **additive**: `s(i,j) = s1[i] + s2[j]`, so exact top-k over the candidate grid needs no extra key reads.
4. Final top-k slots → read value vectors `vᵢ` (quantized) and/or payload pointers.
5. Inject: `x ← x + g ⊙ Σ_i softmax(s)_i · dequant(vᵢ)`, `g` a learned per-channel gate.

Note: exact global top-k is only *guaranteed* inside the A×B grid if A,B are large enough; with A=B=32 the true top-8 is recovered with high probability, not certainty `[plausible]`. Increase A,B if Recall@k of an oracle probe is low.

Required parameters (POC defaults):

| Param | Value | Notes |
|---|---|---|
| key dim `d_k` | 256 (q1,q2 = 128 each) | |
| value dim `V` | 256 | |
| codebooks | 2 | product structure |
| codebook size `C` | 8192 (padded) | √N for N≈32M is ~5657 |
| candidates A×B | 32×32 = 1024 | |
| final top-k | 8 | |
| value precision | BF16 train → int8/int4 infer | |
| layout | struct-of-arrays, values contiguous by slot id | SIMD-friendly dequant |
| mmap | `MAP_SHARED` + `madvise(RANDOM)`; bank RAM-resident | no host swap |
| slot metadata | v0: none; v2: 32 B/slot provenance | see §3.6 |
| load balancing | aux loss §5.1 + optional slot dropout | |
| hot-slot mitigation | LFU/LRU hot-slot cache; temperature on gate | |
| dead-slot mitigation | periodic re-init of never-retrieved slots toward live query centroids | |

### 3.3 Dual memory mode

- **Mode A (latent):** value vectors injected into the stream. Good for soft associations, idioms, distributional knowledge. `[proven]` useful.
- **Mode B (exact payload):** the slot stores a pointer into a separate payload store (mmap blob) holding the *exact* string: API signature, code span, type, enum, error example, prior successful patch, factual record. Returned verbatim to the action layer / context. Required because software tasks need exact tokens, not approximate latent recall. This is the RAG-like escape hatch, but addressed by the *same* learned keys and integrated end-to-end. `[plausible]` as a unified scheme.

Routing between modes: a small per-slot type flag + a gate decides whether a retrieved slot contributes a latent vector, a payload, or both.

### 3.4 Adaptive re-query

A halt head (PonderNet/ACT lineage) emits, after each integration step, a halt probability and an action in {no-lookup, lookup, re-query, halt, abstain, request-context}. Loop:

```
q0 = project(x);  for step s in 1..S_max:
    retrieve(qs) → integrate → update x
    p_halt = halt_head(x)
    if sample/threshold(p_halt): break
    qs+1 = requery_head(x)         # next-hop query conditioned on integrated state
emit token/action (or abstain)
```

This is the only mechanism that can do **multi-hop**: hop-1 retrieves fact A; the integrated state forms a query for fact B; etc. It is also the riskiest component `[speculative]` — adaptive halting is finicky and multi-hop over *latent* (not text) memory has weak precedent. Hard cap `S_max` (e.g. 4) for the POC.

### 3.5 Surprise-gated hierarchical input (deferred past first POC)

BLT-style: a small byte/token entropy LM scores next-symbol surprise; predictable spans (logs, boilerplate, generated files) are compressed into latent patches getting less core compute; high-surprise spans (errors, diffs, novel symbols) are expanded. BLT showed up to ~50% inference-FLOP savings at 8B `[proven]` at byte level; transfer to a small memory model is `[plausible]`. Train the entropy model first, freeze, then patch. Ablate as A6 — do not entangle it with the core memory experiment.

### 3.6 Editable memory (staged)

| Stage | Memory | When |
|---|---|---|
| v0 | frozen, offline-built | first POC |
| v1 | trainable values/keys | Phase 5 |
| v2 | editable + provenance | Phase 7 |
| v3 | local per-user/per-repo | product |

Per-slot provenance (v2, ~32 B): source_type(1) · source_id(8) · span_hash(8) · timestamp(4) · confidence(int8,1) · access_count(4) · last_updated(4) · payload_ptr(8) · invalidation_policy(1). Enables edit/invalidate without retraining the core — a concrete advantage over dense weights.

---

## 4. Mathematical / computational specification

### 4.1 Per-token lookup cost (one memory layer)

| Step | Cost |
|---|---|
| codebook scoring | `2 · C · (d_k/2)` MAC = `2·8192·128 ≈ 2.1 M` MAC |
| top-A/top-B select | 2 partial sorts of 8192 |
| candidate top-k | partial sort of A·B = 1024 (additive scores, no key reads) |
| value reads | `k` random reads × `V·p` B = `8 × 256 B = 2 KB` (int8) |
| integrate/gate | `~k·V·d` MAC, small |

Per token across 4 memory layers (12-layer POC, memory every 3): **~8.4 M MAC** for codebooks + **32 random value reads ≈ 8 KB** random-access bytes. Codebooks (~4.2 MB/layer FP16) stay resident/cached.

### 4.2 Streamed vs random-access bytes/token (the actual economic claim)

| Model | Streamed weights/token (FP16) | Random-access bytes/token | Knowledge capacity touched |
|---|---|---|---|
| Dense 120M | ~240 MB | 0 | in-weights |
| Dense 350M | ~700 MB | 0 | in-weights |
| SAM 120M core + 16 GB bank (int8) | ~240 MB (core) | ~8 KB | 16 GB resident, 8 KB read |
| SAM, **ternary** core | ~24 MB (≈10×) | ~8 KB | as above |

The claim is **not** "fewer streamed bytes than dense-120M" (the core is the same size). The claim is **350M-level knowledge at 120M-level streamed bytes** — i.e. ~3× lower streamed bytes/token than the 350M dense model it aims to match on knowledge tasks, with ternary pushing that toward ~30×. Falsifiable: if SAM cannot reach 350M-dense knowledge accuracy, the streamed-bytes advantage is moot.

### 4.3 CPU latency model

- DDR5 dual-channel realistic ≈ 40–60 GB/s. Streaming 240 MB/token ⇒ ~4–6 ms/token ⇒ **~170–250 tok/s** ceiling from core streaming alone.
- Random RAM read ≈ 80–100 ns. 32 reads/token ⇒ ~3 µs — **negligible vs core streaming, IF bank is RAM-resident.** `[proven]` by arithmetic.
- NVMe random 4 KB QD1 ≈ 20–80 µs. 32 reads/token ⇒ 0.6–2.6 ms + page-fault storms ⇒ comparable to or worse than core streaming. **Conclusion: the touched bank must be RAM-resident; NVMe-backed billion-slot banks break the low-latency target.** `[proven]` by arithmetic.

### 4.4 Memory sizing table (value_dim = 256)

| Slots | Codebook √N (pad) | Keys (FP16) | Values FP16 (512 B) | Values int8 (256 B) | Values int4 (128 B) | +32B meta |
|---|---|---|---|---|---|---|
| 1M | 1024 | ~0.5 MB | 0.5 GB | 0.25 GB | 0.13 GB | +32 MB |
| 16M | 4096 | ~2 MB | 8 GB | 4 GB | 2 GB | +0.5 GB |
| 64M | 8192 | ~4 MB | 32 GB | **16 GB** | 8 GB | +2 GB |
| 256M | 16384 | ~8 MB | 128 GB | 64 GB | **32 GB** | +8 GB |
| 1B | 32768 | ~16 MB | 512 GB | 256 GB | 128 GB | +32 GB |

Keys are never the bottleneck; values dominate. **Practical on 64 GB RAM:** 64M int8 (16 GB) is comfortable (recommended ceiling for 64 GB); 256M int4 (32 GB) is feasible but tight and forces payload store onto NVMe; 1B is RAM-resident-impossible on 64 GB and hits the §4.3 latency wall.

### 4.5 Training-time memory (the under-discussed constraint) `[impl-risk]`

32M slots × 256 × 2 B (BF16 values) = 16 GB params; Adam moments (m,v) ≈ +32 GB ⇒ ~48 GB optimizer state for memory alone, before the core. Mitigations (all `[proven]` in recsys-scale embedding training): sparse gradients (only retrieved rows), sparse/embedding-bag optimizer, FP32 master on host/CPU, sharded memory. Consequence: only retrieved slots get gradient ⇒ under a small token budget most slots stay cold. Effective utilization, not nominal slot count, is the metric — **start the POC at 8–16M slots** and grow only if slot-entropy stays high.

---

## 5. Training design

### 5.1 Loss

```
L = L_lm
  + λ_mem  · L_mem_contrastive      # query↔correct-slot InfoNCE
  + λ_bal  · L_load_balance         # Switch-style aux: N·Σ fᵢ·Pᵢ
  + λ_ent  · L_slot_entropy         # anti-collapse on usage distribution
  + λ_recon· L_retrieval_recon      # value reconstructs teacher hidden/payload emb
  + λ_halt · L_adaptive_requery     # PonderNet geometric prior + correctness
  + λ_verify· L_verifier            # pass/fail of executed code action
```

| Term | Why it exists | Supervision | How it fails |
|---|---|---|---|
| `L_lm` | the actual objective | next token CE | dominates; can ignore memory if gate collapses |
| `L_mem_contrastive` | LM gradient through hard top-k is sparse/weak; need direct retrieval signal | "correct slot" = slot that most reduces LM loss (EM bootstrap) or teacher | degenerate targets; needs warm start |
| `L_load_balance` | prevent a few hot slots absorbing all traffic | batch routing stats | too strong ⇒ fights real Zipfian knowledge ⇒ hurts recall; keep λ low |
| `L_slot_entropy` | prevent collapse to few slots / dead mass | usage histogram entropy | overlaps L_bal; use one primary + one light regularizer |
| `L_retrieval_recon` | dense gradient to value vectors | reconstruct teacher hidden state or payload embedding | wrong target ⇒ values learn the wrong thing |
| `L_adaptive_requery` | learn when to stop/hop | geometric step prior + answer correctness | collapses to always-1-step or always-max; needs multi-hop data |
| `L_verifier` | reward executable correctness for code | executed pass/fail | reward sparsity; needs harness |

### 5.2 Gradient strategy for discrete top-k

- **Early:** soft retrieval over the 1024-candidate set (softmax-weighted value sum) → fully differentiable; trains query, keys, values densely. `[proven]` (PKM/MoE).
- **Mid:** hard top-k forward + straight-through estimator; candidate-softmax supplies surrogate gradient to keys/query; selected values get true gradient.
- **Inference:** hard top-k, no softmax.
- **Risks:** soft→hard train/inference gap; STE bias; candidate set may miss true top-k (raise A,B). Alternative: Gumbel-softmax relaxation early; or keep a permanent small soft "candidate tail" during training.

### 5.3 Curriculum (with gates)

| Stage | Goal | Measure | Go/no-go |
|---|---|---|---|
| 0 Core-only baseline | weak baseline | ppl, knowledge acc | core-only must be *weak* on knowledge (else thesis untestable) |
| 1 Retrieval pretrain | query learns to hit known slots | Recall@1/8/32 | Recall@8 ≥ 0.8 on synthetic, else fix keys |
| 2 Memory-augmented LM | next-token with memory | core-only vs random vs retrieved vs **oracle** | retrieved ≫ random; gap to oracle quantified |
| 3 Knowledge-heavy | API/lib/fact recall | acc vs dense-120M/350M | beat 120M; approach 350M |
| 4 Multi-hop synthetic | compose 2–4 retrieved facts | multi-hop acc; **oracle-mem multi-hop** | **decisive gate** (see §10) |
| 5 Code/dev tasks | compile/test repair | verified patch rate | beat 120M dense + RAG baseline |
| 6 Adaptive re-query | halt/hop behavior | hop count vs accuracy | re-query helps multi-hop without runaway loops |
| 7 Quantization | int4 mem / ternary core | Δacc, tok/s, RAM | int4 mem Δ small; ternary core gated separately |

### 5.4 Data sources

Synthetic knowledge graphs (controllable multi-hop with known hop structure) for Stages 1–4; permissively licensed code + package docs + type stubs + real CI logs / stack traces for Stages 5–7; held-out fact sets edited post-hoc to test editable memory. Synthetic-first because it gives ground-truth hop structure to supervise re-query and to build oracle memory.

---

## 6. Evaluation design

### 6.1 Primary metric

```
efficiency = verified_success_rate / (peak_RAM_GB × wall_clock_seconds)
```

Knowledge/code tasks use *verified* success (execution or exact-match), not perplexity.

### 6.2 Secondary metrics

perplexity · Recall@k · multi-hop accuracy · patch-apply rate · verified-patch rate · token latency · lookup latency · slot entropy · slot load balance · CPU utilization · RAM bandwidth · page-fault rate · hot-slot ratio · dead-slot ratio · streamed MB/token · random-read bytes/token.

### 6.3 Categories
A. Knowledge recall (API signatures, package behavior, framework config, factual QA).
B. Reasoning over retrieved facts (multi-hop QA, dependency/symbol/config-implication reasoning).
C. Code generation (standalone, repo-local completion, library-specific usage).
D. QA/dev automation (failing Playwright/Cypress repair, API test gen, compile/type-error repair, stack-trace diagnosis, flaky-wait repair, minimal patch).
E. CPU/runtime (tok/s, peak RAM, streamed MB/token, random-read latency, cache-miss rate, page faults, hot/dead-slot ratios).

### 6.4 Ablations and pass conditions

| ID | Config | Tests | Pass condition |
|---|---|---|---|
| A0 | same-size dense Transformer (120M) + 350M dense | the real baselines | reference |
| A1 | SAM core only, no memory | core capacity floor | must be *weak* on knowledge |
| A2 | SAM + random memory | placebo | ≈ A1 (else metric leak) |
| A3 | SAM + frozen retrieval memory | retrieval value | > A1 on recall |
| A4 | SAM + trainable memory | learned values | > A3; ≥ 350M dense on **knowledge** |
| A5 | A4 + adaptive re-query | multi-hop | > A4 on **multi-hop**; ≈ 350M dense |
| A6 | + surprise-gated input | compute savings | ≥ accuracy at lower FLOPs |
| A7 | + int4 memory | quant loss | Δacc ≤ ~1 pt |
| A8 | + ternary core | CPU deploy | gated separately; expect loss at 120M |

**Critical criterion (restated, sharpened):** SAM must beat the **same-active-compute dense Transformer** on knowledge *and* reasoning-over-retrieved-facts. `A4 > A1` (memory helps a weak model) is **not** sufficient and must not be reported as success.

---

## 7. Implementation roadmap

Minimum viable POC = Phases 1–5 (frozen→trainable memory + first knowledge/multi-hop read). Decision to scale lives at Phase 6/9.

| Phase | Objective | Key outputs | Success | Failure | Go/no-go gate |
|---|---|---|---|---|---|
| 1 Lit & arch validation | separate novel composition from known primitives | primitive inventory, risk table | this document's claims hold | a primitive is unavailable on CPU | proceed only if all primitives have CPU paths |
| 2 Dense baseline + synthetic bench | build A0 + synthetic multi-hop suite | 120M/350M dense, KG tasks, oracle memory | baselines + oracle reproduce | bench not separable from recall | bench discriminates recall vs multi-hop |
| 3 Product-key memory prototype | lookup, storage, metrics | mmap bank, Recall@k harness | Recall@8 ≥ 0.8 synthetic | Recall@8 < threshold | retrieval is learnable |
| 4 SAM-v0 frozen memory | core queries offline bank | A3 results | retrieved ≫ random; LM ppl ↓ | no LM gain | memory measurably helps |
| 5 SAM-v1 trainable memory | train values/keys, fix load balance | A4 results | ≥ 350M dense on **knowledge**, lower streamed bytes | only beats 120M, not 350M | knowledge parity with 2–3× dense |
| 6 Adaptive re-query | halt head, multi-hop | A5 + oracle-mem multi-hop | ≥ 350M dense on **multi-hop** | recall-only win, multi-hop flat | **PROGRAM GATE** (§10) |
| 7 Code/dev specialization | repo/task memory, payloads | verified patch rates | > 120M dense + RAG on dev tasks | exact-payload mode needed for everything (latent adds nothing) | latent memory earns its place |
| 8 Quantization & runtime | int4 mem, ternary core, CPU profiling | tok/s, RAM, page faults | int4 Δ ≤ 1 pt; meets RAM profile | quant erases the win | deployable on target RAM |
| 9 Scale decision | scale slots / core / or stop | decision memo | clear scaling law | flat/negative scaling | scale, respecialize, or abandon |

### RAM profiles (runtime)

| RAM | Core | Slots | top-k | Modes |
|---|---|---|---|---|
| 8 GB | tiny | 1–4M | 4 | shallow, no/limited multi-hop |
| 16 GB | 100–150M | 16–32M | 4–8 | single-hop + light re-query |
| 32 GB | 120–180M | 64M int8 | 8 | adaptive re-query, larger window |
| 64 GB | 120–180M | 64M int8 / 256M int4 | 8–16 | adaptive multi-hop, hot-slot cache, exact payloads |

---

## 8. Engineering risks

| Risk | Severity | Class | Mitigation |
|---|---|---|---|
| Tiny core can't multi-hop over latent | **critical** | speculative | oracle-mem diagnostic; if oracle still ≪ 350M dense, bottleneck is core → don't scale as base |
| Routing collapse / hot slots | high | proven-problem | load-balance + entropy loss, gate temperature, slot dropout |
| Dead slots (cold under POC budget) | high | impl-risk | start 8–16M slots; re-init dead slots; track utilization |
| Cache miss / page-fault storms | high | proven (NVMe) | keep touched bank RAM-resident; hot-slot cache; `madvise(RANDOM)`; no swap |
| Top-k non-differentiability | medium | known | soft candidates early, STE + candidate-softmax, raise A,B |
| Training-time optimizer memory | high | impl-risk | sparse optimizer, FP32 master on host, sharding |
| Quantization loss (int4 / ternary) | medium–high | known | int4 mem usually safe (Δ≤1pt); ternary core only ≥3B-like regimes — gate A8 |
| Stale / corrupt knowledge | medium | design | provenance + invalidation policy + confidence; checksum payload store |
| Adaptive halting instability | medium | speculative | PonderNet geometric prior; hard S_max; penalize loops |
| RAG matches latent for cheaper | medium | strategic | if A7/Phase-7 shows payloads alone suffice, reposition as exact-retrieval specialist |

---

## 9. Product implications

**Where SAM can win.** Local, CPU-first, *editable*, knowledge-dense assistant where verification is cheap — i.e. **software/QA automation**: API/library knowledge, framework config, error-message diagnosis, Playwright/Cypress/Selenium repair, API test generation, compile/type-error repair, stack-trace reasoning, minimal-patch generation, with **per-repo memory** that updates without retraining. Provenance/invalidation is a real differentiator over dense weights (you can *delete* a wrong fact). This maps directly onto a local developer/QA tool: SAM is the model, the agent harness (READ/SEARCH/PATCH/RUN/DIAGNOSE/ABSTAIN) is a *separate* product layer — keep them decoupled. (For a local CLI assistant, SAM is a candidate engine; the CLI is the harness, not the architecture.)

**Where it probably loses.** Deep, novel multi-hop reasoning that needs a large core; open-ended generation where knowledge isn't the bottleneck; any setting where RAG + a small context window already matches latent memory at far lower complexity. On NVMe-spilled banks the latency target breaks.

**Positioning.** "350M-class knowledge at 120M-class streamed bytes, on CPU, editable" — *not* "a better general LLM." Sell the knowledge-per-RAM-second and editability, prove the reasoning claim before claiming it.

---

## 10. Final recommendation

**Build — Phases 1–6 — with the program gate at multi-hop, not recall.**

- **Exact first experiment:** SAM-32M-Mem as specified (120M core, 12 layers, d=768, local attn every 4, mixer every layer, memory every 3; 32M slots, padded codebooks 8192, V=256, topA=topB=32, candidates=1024, topK=8, BF16 train / int8–int4 infer). Two amendments: (a) **fall back to 8–16M slots** if slot entropy collapses under the POC token budget; (b) the **oracle-memory SAM run is mandatory**, not optional.
- **Exact baseline to beat:** the **350M dense Transformer on multi-hop-over-retrieved-facts**, with the 120M dense as the floor and a same-payload RAG baseline as the alternative explanation.
- **Exact success metric:** `verified_success_rate / (peak_RAM_GB × wall_clock_s)` strictly greater than all dense baselines, **conditioned on** multi-hop accuracy ≥ 350M dense. Recall ≥ 350M dense is necessary but **not** sufficient.

**The decisive read (oracle diagnostic):**
- Oracle-memory SAM ≈ 350M dense on multi-hop → bottleneck is *retrieval* (fixable: better keys/contrastive loss/re-query) → **scale.**
- Oracle-memory SAM ≪ 350M dense on multi-hop → bottleneck is the *core's reasoning*, which memory cannot fix → **reclassify SAM as a retrieval-heavy specialist** (the prompt's own failure criterion) and do not scale it as a general base architecture.

**Failure criterion (binding):** if SAM only improves factual recall but loses substantially on multi-hop reasoning, it is a specialist, not a base. Ship it as an editable knowledge/QA-automation engine; do not market it as a general LLM replacement.

---

### Prior-work anchors (for Phase 1)
Product-key memory: Lample et al. 2019. Memory layers at scale (beats >2× compute dense, MoE-parity, factual gains, 128B mem params): Berges et al., Meta FAIR, Dec 2024. PEER (rank-one expert values via product keys): He 2024. Selective SSM core: Mamba / Mamba-2. Local/sliding-window attention: Mistral, Longformer. Ternary weights (FP16 parity from ~3B): BitNet b1.58 (Ma et al. 2024). Surprise/entropy patching (~50% inference FLOPs at 8B): Byte Latent Transformer, Meta Dec 2024. Adaptive computation/halting: ACT (Graves), PonderNet, Universal Transformer. Load-balancing aux loss: Switch Transformer. Text retrieval precedents (contrast with latent memory): kNN-LM, RETRO.
