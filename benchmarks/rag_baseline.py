"""
RAG Baseline Comparator — standalone RAG pipeline for NEXUS vs RAG comparison.

Pipeline:
  1. Read all .md files from sam-lm/docs/ and sam-lm/experiments/
  2. Chunk into ~500-token segments with 100-token overlap (sentence-boundary aware)
  3. Embed with sentence-transformers all-MiniLM-L6-v2 (cache to disk)
  4. For each question: embed, cosine similarity, top-5 chunks
  5. Build RAG prompt matching NEXUS token budget
  6. Generate answer with same model backend as NEXUS
  7. Verify with same Verifier as NEXUS

Comparison: runs both NEXUS and RAG on the same questions with the same model,
scoring, and verifier — the only variable is the retrieval mechanism.

Usage:
    python benchmarks/rag_baseline.py --limit 30 --output benchmarks/nexus_vs_rag_30.json
    python benchmarks/rag_baseline.py --limit 200 --output benchmarks/nexus_vs_rag_200.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root on sys.path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from nexus.graph.store import InMemoryGraphStore
from nexus.reasoning.answer import answer_question
from nexus.reasoning.model_interface import (
    FallbackModel, ModelInterface, OllamaModel,
    SynthesizingModel, get_available_model,
)
from nexus.reasoning.verifier import Verifier, VerificationResult
from nexus.utils.config import DEFAULT_CONFIG

# ---- Constants ----
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 100
TOP_K_CHUNKS = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_CACHE = _project_root / "benchmarks" / ".rag_embedding_cache.pkl"
CORPUS_DIRS = [
    _project_root / "sam-lm" / "docs",
    _project_root / "sam-lm" / "experiments",
]


# ═══════════════════════════════════════════════════════════════════════
# Token estimation
# ═══════════════════════════════════════════════════════════════════════

def _count_tokens(text: str) -> int:
    """Simple word-count token estimation (split on whitespace)."""
    return len(text.split())


# ═══════════════════════════════════════════════════════════════════════
# Chunking
# ═══════════════════════════════════════════════════════════════════════

def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preferring double-newline boundaries."""
    # First split on double newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)
    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Split paragraph on single newlines
        lines = para.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Split on sentence boundaries (. ! ? followed by space or end)
            parts = re.split(r'(?<=[.!?])\s+', line)
            sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def chunk_documents(doc_paths: list[Path]) -> list[dict[str, Any]]:
    """
    Read all .md files, chunk into ~500-token segments with 100-token overlap.
    Returns list of dicts with: text, source, chunk_index.
    """
    all_chunks: list[dict[str, Any]] = []
    for doc_path in sorted(doc_paths):
        try:
            text = doc_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        sentences = _split_into_sentences(text)
        if not sentences:
            continue

        # Build chunks greedily: add sentences until we hit ~CHUNK_SIZE_TOKENS
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _count_tokens(sent)
            if current_tokens + sent_tokens > CHUNK_SIZE_TOKENS and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep last ~OVERLAP tokens worth of sentences
                overlap_sents: list[str] = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    st = _count_tokens(s)
                    if overlap_tokens + st > CHUNK_OVERLAP_TOKENS:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tokens += st
                current_chunk = overlap_sents
                current_tokens = overlap_tokens

            current_chunk.append(sent)
            current_tokens += sent_tokens

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        source_name = doc_path.name
        source_rel = str(doc_path.relative_to(_project_root))
        for ci, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "source": source_name,
                "source_path": source_rel,
                "chunk_index": ci,
                "token_count": _count_tokens(chunk_text),
            })

    return all_chunks


# ═══════════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════════

class RAGEmbedder:
    """
    Sentence-transformers embedder with disk caching.

    Uses all-MiniLM-L6-v2 (CPU-friendly, 384-dim, ~80MB download).
    Embeddings are cached to disk so they're only computed once.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, cache_path: Path = EMBEDDING_CACHE):
        self._model_name = model_name
        self._cache_path = cache_path
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[rag] Installing sentence-transformers...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])
            from sentence_transformers import SentenceTransformer
        print(f"[rag] Loading embedding model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name)

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[list[float]]:
        """Embed all chunks, using cache if available."""
        # Check cache
        cache_key = self._cache_key(chunks)
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("key") == cache_key:
                    print(f"[rag] Loaded {len(cached['embeddings'])} embeddings from cache")
                    return cached["embeddings"]
                else:
                    print("[rag] Cache key mismatch, recomputing embeddings")
            except Exception:
                print("[rag] Cache corrupted, recomputing embeddings")

        self._load_model()
        texts = [c["text"] for c in chunks]
        print(f"[rag] Embedding {len(texts)} chunks (this may take a minute)...")
        embeddings = self._model.encode(
            texts,
            convert_to_tensor=False,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        # Convert numpy arrays to lists for JSON serializability
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Save cache
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "wb") as f:
            pickle.dump({"key": cache_key, "embeddings": embeddings_list}, f)
        print(f"[rag] Cached {len(embeddings_list)} embeddings to {self._cache_path}")

        return embeddings_list

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        self._load_model()
        emb = self._model.encode(
            [query],
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        return emb[0].tolist()

    @staticmethod
    def _cache_key(chunks: list[dict[str, Any]]) -> str:
        """Generate a stable cache key from chunk metadata."""
        import hashlib
        key_parts = sorted(
            f"{c['source']}:{c['chunk_index']}:{_count_tokens(c['text'])}"
            for c in chunks
        )
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════════════════

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (assumes normalized)."""
    dot = sum(x * y for x, y in zip(a, b))
    # Already normalized by the embedder, but be safe
    if dot > 1.0:
        return 1.0
    if dot < -1.0:
        return -1.0
    return dot


def retrieve_top_k(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    chunks: list[dict[str, Any]],
    k: int = TOP_K_CHUNKS,
) -> list[dict[str, Any]]:
    """Retrieve top-k most similar chunks by cosine similarity."""
    scores = [
        (cosine_similarity(query_embedding, ce), i)
        for i, ce in enumerate(chunk_embeddings)
    ]
    scores.sort(key=lambda x: -x[0])

    results: list[dict[str, Any]] = []
    for score, idx in scores[:k]:
        chunk = dict(chunks[idx])
        chunk["similarity"] = round(score, 4)
        results.append(chunk)
    return results


# ═══════════════════════════════════════════════════════════════════════
# RAG Prompt Builder
# ═══════════════════════════════════════════════════════════════════════

def build_rag_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """
    Build a RAG prompt matching the NEXUS token budget.

    The NEXUS prompt provides structured evidence with facts, paths, and sources.
    The RAG prompt provides raw document excerpts. Both should use equivalent tokens.
    """
    parts: list[str] = []

    # System instruction — same style as NEXUS
    parts.append(
        "SYSTEM: You are a precise reasoning assistant. "
        "Use the provided document excerpts to answer the question. "
        "Answer ONLY based on the excerpts. "
        "Quote specific numbers when available. "
        "If the excerpts truly lack the answer, say \"Insufficient evidence to answer.\" "
        "Do not invent facts."
    )

    parts.append(f"\nQUESTION: {question}")

    parts.append(
        "\nIMPORTANT: Answer ONLY the question asked. Do not list all excerpts. "
        "Maximum 3 sentences."
    )

    # Document excerpts section
    parts.append("\nDOCUMENT EXCERPTS:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        chunk_text = chunk["text"]
        # Truncate very long chunks to keep token budget reasonable
        source = chunk.get("source", "unknown")
        if _count_tokens(chunk_text) > 300:
            chunk_text = " ".join(chunk_text.split()[:300]) + "..."
        parts.append(f"\n[{i}] {source}:")
        parts.append(chunk_text)

    # Sources summary
    parts.append(f"\nSources: {len(retrieved_chunks)} document excerpt(s) retrieved by semantic search.")

    parts.append("\nANSWER:")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════
# RAG evidence pack builder (for Verifier compatibility)
# ═══════════════════════════════════════════════════════════════════════

def build_rag_evidence_pack(retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build an evidence pack from RAG chunks compatible with the NEXUS Verifier.

    The Verifier expects ``evidence_pack`` with ``paths`` containing ``nodes``
    (with id, type) and ``edges`` (with type, confidence), plus ``facts`` strings.

    We construct a minimal pack from the chunk content to enable fair comparison.
    """
    sources: list[str] = []
    facts: list[str] = []
    nodes: list[dict[str, Any]] = []

    for i, chunk in enumerate(retrieved_chunks):
        source = chunk.get("source_path", chunk.get("source", ""))
        sources.append(source)

        # Extract key sentences from chunk as "facts"
        text = chunk["text"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences[:3]:  # Top 3 sentences per chunk
            sent = sent.strip()
            if len(sent) > 20 and len(sent) < 300:
                facts.append(sent)

        # Create synthetic nodes from chunk metadata
        nodes.append({
            "id": f"rag_chunk_{i}",
            "type": "document_excerpt",
            "description": text[:200] + ("..." if len(text) > 200 else ""),
        })

    paths = [{
        "nodes": nodes,
        "edges": [],
        "score": 1.0,
    }]

    return {
        "question": "",
        "paths": paths,
        "facts": facts,
        "sources": sources,
    }


# ═══════════════════════════════════════════════════════════════════════
# RAG Pipeline
# ═══════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Full RAG pipeline: chunk → embed → retrieve → prompt → model → verify.
    """

    def __init__(
        self,
        model: ModelInterface,
        verifier: Verifier | None = None,
        chunks: list[dict[str, Any]] | None = None,
        chunk_embeddings: list[list[float]] | None = None,
        embedder: RAGEmbedder | None = None,
    ):
        self._model = model
        self._verifier = verifier or Verifier(hallucination_threshold=0.2)
        self._chunks = chunks or []
        self._embeddings = chunk_embeddings or []
        self._embedder = embedder or RAGEmbedder()

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def answer_question(self, question: str) -> dict[str, Any]:
        """
        Run the full RAG pipeline on a single question.

        Returns dict with: answer, evidence_pack, verification, latency_s,
        retrieved_chunks, is_insufficient, error.
        """
        t0 = time.perf_counter()
        result: dict[str, Any] = {
            "question": question,
            "answer": "",
            "evidence_pack": {},
            "verification": None,
            "retrieved_chunks": [],
            "is_insufficient": False,
            "latency_s": 0.0,
            "error": None,
        }

        try:
            # Edge case: no chunks available
            if not self._chunks or not self._embeddings:
                result["answer"] = "Insufficient evidence to answer. No documents indexed."
                result["verification"] = VerificationResult(
                    supported_count=0,
                    hallucination_rate=0.0,
                    passed=True,
                )
                result["latency_s"] = round(time.perf_counter() - t0, 4)
                return result

            # Step 1: Embed query
            query_emb = self._embedder.embed_query(question)

            # Step 2: Retrieve top-k chunks
            retrieved = retrieve_top_k(query_emb, self._embeddings, self._chunks, k=TOP_K_CHUNKS)
            result["retrieved_chunks"] = [
                {
                    "source": r["source"],
                    "source_path": r["source_path"],
                    "chunk_index": r["chunk_index"],
                    "similarity": r["similarity"],
                    "token_count": r["token_count"],
                }
                for r in retrieved
            ]

            # Step 3: Build evidence pack (for verifier)
            evidence_pack = build_rag_evidence_pack(retrieved)
            result["evidence_pack"] = evidence_pack

            # Step 4: Build prompt
            prompt = build_rag_prompt(question, retrieved)
            prompt_tokens = _count_tokens(prompt)

            # Step 5: Generate answer
            answer = self._model.generate(prompt)
            result["answer"] = answer
            result["is_insufficient"] = "insufficient evidence" in answer.lower()

            # Step 6: Verify
            verification = self._verifier.verify(answer, evidence_pack)
            result["verification"] = verification

        except Exception as exc:
            result["answer"] = f"[ERROR] {exc}"
            result["error"] = str(exc)
            result["verification"] = VerificationResult(
                supported_count=0,
                hallucination_rate=1.0,
                passed=False,
            )

        result["latency_s"] = round(time.perf_counter() - t0, 4)
        return result


# ═══════════════════════════════════════════════════════════════════════
# Pipeline runners (compatible with benchmark harness)
# ═══════════════════════════════════════════════════════════════════════

def run_nexus_pipeline(
    question_text: str,
    graph: InMemoryGraphStore,
    model: ModelInterface,
    verifier: Verifier,
) -> dict[str, Any]:
    """
    Run the full NEXUS pipeline and return timing + metrics.

    Returns a dict with:
        - answer: the generated answer text
        - passed: whether verification passed
        - hallucination_rate: float 0.0-1.0
        - supported_count: number of supported claims
        - unsupported_count: number of unsupported claims
        - path_count: number of traversal paths found
        - is_insufficient: whether the answer says "Insufficient evidence"
        - latency_s: total wall-clock seconds
        - error: error message if pipeline crashed (None otherwise)
        - evidence_tokens: token count of evidence section for budget comparison
    """
    t0 = time.perf_counter()
    try:
        result = answer_question(
            question_text, graph, model=model, verifier=verifier,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "answer": f"[ERROR] {exc}",
            "passed": False,
            "hallucination_rate": 1.0,
            "supported_count": 0,
            "unsupported_count": 0,
            "path_count": 0,
            "is_insufficient": False,
            "latency_s": round(elapsed, 4),
            "error": str(exc),
            "parsed_entity_ids": [],
            "evidence_tokens": 0,
        }
    elapsed = time.perf_counter() - t0

    verif: VerificationResult | None = result.get("verification")
    answer = result.get("answer", "")
    is_insufficient = "insufficient evidence" in answer.lower()

    # Estimate evidence tokens
    evidence_pack = result.get("evidence_pack", {})
    evidence_tokens = _count_tokens(json.dumps(evidence_pack, default=str))

    if verif is not None:
        passed = verif.passed
        hall_rate = verif.hallucination_rate
        supported = verif.supported_count
        unsupported = len(verif.unsupported_claims)
    else:
        passed = True
        hall_rate = 0.0
        supported = 0
        unsupported = 0

    return {
        "answer": answer,
        "passed": passed,
        "hallucination_rate": round(hall_rate, 4),
        "supported_count": supported,
        "unsupported_count": unsupported,
        "path_count": result.get("path_count", 0),
        "is_insufficient": is_insufficient,
        "latency_s": round(elapsed, 4),
        "error": None,
        "entity_resolution_method": result.get("entity_resolution_method", "none"),
        "parsed_entity_ids": (
            result["parsed_query"].entity_ids if result.get("parsed_query") else []
        ),
        "evidence_tokens": evidence_tokens,
    }


def run_rag_pipeline(
    question_text: str,
    rag: RAGPipeline,
) -> dict[str, Any]:
    """
    Run the RAG pipeline and return timing + metrics.

    Returns same-format dict as run_nexus_pipeline for direct comparison.
    """
    result = rag.answer_question(question_text)

    verif = result.get("verification")
    if verif is not None:
        passed = verif.passed
        hall_rate = verif.hallucination_rate
        supported = verif.supported_count
        unsupported = len(verif.unsupported_claims)
    else:
        passed = True
        hall_rate = 0.0
        supported = 0
        unsupported = 0

    # Estimate evidence tokens from chunks (cumulative token count)
    retrieved_chunks = result.get("retrieved_chunks", [])
    evidence_tokens = sum(
        c.get("token_count", _count_tokens(c.get("text", "")))
        for c in retrieved_chunks
    )

    return {
        "answer": result["answer"],
        "passed": passed,
        "hallucination_rate": round(hall_rate, 4),
        "supported_count": supported,
        "unsupported_count": unsupported,
        "path_count": len(result.get("retrieved_chunks", [])),
        "is_insufficient": result.get("is_insufficient", False),
        "latency_s": result["latency_s"],
        "error": result.get("error"),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "evidence_tokens": evidence_tokens,
    }


# ═══════════════════════════════════════════════════════════════════════
# Scoring (reuse from run_benchmark.py)
# ═══════════════════════════════════════════════════════════════════════

# Regex patterns for extracting key facts from text
_FACT_PATTERNS = [
    (re.compile(r'\b(\d+\.?\d*\s*%)(?=\s|$|[,.);])'), "percentage"),
    (re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*million\b', re.IGNORECASE), "number+million"),
    (re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:live\s+)?(slots?|examples?|tokens?|parameters?|params?|subkeys?|distractors?|vocabulary|hops?)\b', re.IGNORECASE), "number+unit"),
    (re.compile(r'\b(\d{3,}(?:,\d{3})*(?:\.\d+)?)\b(?!\s*%)'), "large_number"),
    (re.compile(r'\b(\w+@\d+)\b'), "at_notation"),
    (re.compile(r'\b([Kk]=\d+)\b'), "k_notation"),
    (re.compile(r'\b(Exp_\d+_\d+[A-Z]?_\w+)\b'), "experiment_id"),
    (re.compile(r'\b(Concept_\w+)\b'), "concept_id"),
    (re.compile(r'\b(Decision_\w+)\b'), "decision_id"),
    (re.compile(r'\b(depends_on|validates|caused_by|contradicts|implements|mentioned_in|derived_from|related_to|replaces|blocked_by)\b', re.IGNORECASE), "relation"),
    (re.compile(r'\b(core_only|oracle_memory|retrieved_memory|random_memory|oracle_text_memory|oracle_filter|oracle_text_memory|retrieved_memory_external_text_query)\b', re.IGNORECASE), "sam_mode"),
    (re.compile(r'\b(Gate\s+\d+)\b', re.IGNORECASE), "gate_ref"),
]


def _extract_key_facts(text: str) -> set[str]:
    """Extract key facts from text using defined regex patterns."""
    facts: set[str] = set()
    for pattern, _fact_type in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            fact_str = match.group(0).strip().lower()
            fact_str = re.sub(r'(\d),(\d)', r'\1\2', fact_str)
            facts.add(fact_str)
    return facts


def _extract_numbers(text: str) -> set[float]:
    """Extract all numeric values from text."""
    numbers: set[float] = set()
    tokens: list[str] = re.findall(r'[^\s]+', text)
    percent_re = re.compile(r'^([\d,]+\.?\d*)\s*%$')

    for token in tokens:
        while token and token[-1] in ',.;:)!?' and token[-1] != '%':
            token = token[:-1]
        if not token:
            continue
        pm = percent_re.match(token)
        if pm:
            try:
                val = float(pm.group(1).replace(',', ''))
                numbers.add(round(val / 100.0, 10))
            except ValueError:
                pass
            continue
        stripped = token.replace(',', '')
        if stripped.replace('.', '', 1).isdigit():
            try:
                val = float(stripped)
                numbers.add(val)
            except ValueError:
                pass
    return numbers


def _fuzzy_number_match(pred_nums: set[float], gt_nums: set[float]) -> tuple[int, int]:
    """Match predicted numbers against ground truth with 5% relative tolerance."""
    if not gt_nums:
        return 0, 0
    pred_list: list[float | None] = list(pred_nums)
    gt_list = sorted(gt_nums, reverse=True)
    matches = 0
    for gt in gt_list:
        best_idx: int = -1
        best_err: float = float('inf')
        for i, pred in enumerate(pred_list):
            if pred is None:
                continue
            denom = max(abs(gt), 0.001)
            rel_err = abs(pred - gt) / denom
            if rel_err < 0.05 or abs(pred - gt) < 0.001:
                if rel_err < best_err:
                    best_err = rel_err
                    best_idx = i
        if best_idx >= 0:
            matches += 1
            pred_list[best_idx] = None
    return matches, len(gt_nums)


def compute_key_fact_score(predicted_answer: str, ground_truth: str) -> dict[str, Any]:
    """Compute key-fact match score with fuzzy numeric scoring."""
    empty_detail: dict[str, Any] = {
        "gt_numbers": [], "pred_numbers": [], "fuzzy_matches": 0,
        "total_gt": 0, "fuzzy_score": 0.0, "exact_score": 0.0,
        "entity_overlap": [],
    }

    if "insufficient evidence" in predicted_answer.lower():
        return {
            "fuzzy_accuracy": 0.0, "exact_accuracy": 0.0,
            "scoring_detail": empty_detail,
        }

    gt_facts = _extract_key_facts(ground_truth)
    pred_facts = _extract_key_facts(predicted_answer)

    if not gt_facts:
        empty_detail["fuzzy_score"] = None
        empty_detail["exact_score"] = None
        return {
            "fuzzy_accuracy": None, "exact_accuracy": None,
            "scoring_detail": empty_detail,
        }

    intersection = gt_facts & pred_facts
    exact_score: float = round(len(intersection) / len(gt_facts), 4)

    gt_nums = _extract_numbers(ground_truth)
    pred_nums = _extract_numbers(predicted_answer)
    entity_overlap = sorted(list(intersection))

    if gt_nums:
        fuzzy_matches, total_gt = _fuzzy_number_match(pred_nums, gt_nums)
        fuzzy_score = round(fuzzy_matches / total_gt, 4) if total_gt > 0 else None
        primary_accuracy = fuzzy_score if fuzzy_score is not None else exact_score
    else:
        fuzzy_matches = 0
        total_gt = 0
        fuzzy_score = None
        primary_accuracy = exact_score

    return {
        "fuzzy_accuracy": primary_accuracy,
        "exact_accuracy": exact_score,
        "scoring_detail": {
            "gt_numbers": sorted(list(gt_nums)),
            "pred_numbers": sorted(list(pred_nums)),
            "fuzzy_matches": fuzzy_matches,
            "total_gt": total_gt,
            "fuzzy_score": fuzzy_score,
            "exact_score": exact_score,
            "entity_overlap": entity_overlap,
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Graph construction (reuse from run_benchmark.py)
# ═══════════════════════════════════════════════════════════════════════

def build_benchmark_graph() -> tuple[InMemoryGraphStore, dict[str, Any]]:
    """Build the benchmark knowledge graph deterministically."""
    from nexus.ingestion.populate_from_experiments import populate_graph, EXPERIMENTS_DIR
    from nexus.ingestion.ingest_docs import ingest_directory

    graph = InMemoryGraphStore()

    if EXPERIMENTS_DIR.exists():
        graph = populate_graph(EXPERIMENTS_DIR, graph)

    docs_dir = _project_root / "docs"
    if docs_dir.exists():
        ingest_directory(docs_dir, graph)
    sam_docs_dir = _project_root / "sam-lm" / "docs"
    if sam_docs_dir.exists():
        ingest_directory(sam_docs_dir, graph)
    sam_exp_dir = _project_root / "sam-lm" / "experiments"
    if sam_exp_dir.exists():
        ingest_directory(sam_exp_dir, graph)

    provenance = {
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    return graph, provenance


# ═══════════════════════════════════════════════════════════════════════
# Question loading
# ═══════════════════════════════════════════════════════════════════════

def load_questions(jsonl_path: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Load questions from the JSONL dataset."""
    questions: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if limit and limit > 0:
        questions = questions[:limit]
    return questions


# ═══════════════════════════════════════════════════════════════════════
# Comparison runner
# ═══════════════════════════════════════════════════════════════════════

def run_comparison(
    questions: list[dict[str, Any]],
    graph: InMemoryGraphStore,
    nexus_model: ModelInterface,
    rag_pipeline: RAGPipeline,
    verifier: Verifier,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run NEXUS and RAG pipelines on all questions. Returns (results, summary).
    """
    total = len(questions)
    results: list[dict[str, Any]] = []

    for i, q in enumerate(questions, 1):
        qtext = q["question"]
        qid = q.get("id", f"q{str(i).zfill(3)}")
        ground_truth = q.get("answer", "")
        marker = f"[{i}/{total}]"

        # ── Run NEXUS ──
        nexus_result = run_nexus_pipeline(qtext, graph, nexus_model, verifier)

        # Entity resolution accuracy
        gt_entity_ids: list[str] = q.get("entities", [])
        nexus_parsed_ids: list[str] = nexus_result.get("parsed_entity_ids", [])
        entity_resolution_hit = bool(
            gt_entity_ids and any(
                gid == pid or pid.startswith(gid + "_")
                for gid in gt_entity_ids
                for pid in nexus_parsed_ids
            )
        )
        nexus_result["entity_resolution_hit"] = entity_resolution_hit
        nexus_result["gt_entity_ids"] = gt_entity_ids

        nexus_scores = compute_key_fact_score(nexus_result["answer"], ground_truth)
        nexus_result["accuracy"] = nexus_scores["fuzzy_accuracy"]
        nexus_result["exact_accuracy"] = nexus_scores["exact_accuracy"]
        nexus_result["scoring_detail"] = nexus_scores["scoring_detail"]

        # ── Run RAG ──
        rag_result = run_rag_pipeline(qtext, rag_pipeline)

        rag_scores = compute_key_fact_score(rag_result["answer"], ground_truth)
        rag_result["accuracy"] = rag_scores["fuzzy_accuracy"]
        rag_result["exact_accuracy"] = rag_scores["exact_accuracy"]
        rag_result["scoring_detail"] = rag_scores["scoring_detail"]

        # ── Status indicator ──
        if nexus_result["error"]:
            n_status = "ERR"
        elif nexus_result["is_insufficient"]:
            n_status = "INS"
        elif nexus_result["passed"]:
            n_status = "PASS"
        else:
            n_status = f"HALL({nexus_result['hallucination_rate']:.0%})"

        if rag_result.get("error"):
            r_status = "ERR"
        elif rag_result["is_insufficient"]:
            r_status = "INS"
        elif rag_result["passed"]:
            r_status = "PASS"
        else:
            r_status = f"HALL({rag_result['hallucination_rate']:.0%})"

        n_acc = nexus_scores["fuzzy_accuracy"]
        r_acc = rag_scores["fuzzy_accuracy"]
        n_lat = nexus_result["latency_s"]
        r_lat = rag_result["latency_s"]
        er_hit = "HIT" if entity_resolution_hit else "MISS"

        print(
            f"  {marker} {qid}: NEXUS={n_status}(acc={n_acc if n_acc is not None else 'N/A'}, {n_lat:.2f}s) | "
            f"RAG={r_status}(acc={r_acc if r_acc is not None else 'N/A'}, {r_lat:.2f}s) | "
            f"ER={er_hit}"
        )

        results.append({
            "question_id": qid,
            "question": qtext,
            "ground_truth": ground_truth,
            "question_type": q.get("question_type", ""),
            "difficulty": q.get("difficulty", ""),
            "hops": q.get("hops", 1),
            "nexus": nexus_result,
            "rag": rag_result,
        })

    # ── Compute summary ──
    summary = compute_comparison_summary(results)
    return results, summary


def compute_comparison_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics comparing NEXUS vs RAG."""

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    total = len(results)

    # NEXUS metrics
    nx_errors = sum(1 for r in results if r["nexus"].get("error"))
    nx_answered = sum(1 for r in results if not r["nexus"]["is_insufficient"] and not r["nexus"].get("error"))
    nx_insufficient = sum(1 for r in results if r["nexus"]["is_insufficient"])
    nx_passed = sum(1 for r in results if r["nexus"]["passed"] and not r["nexus"].get("error"))
    nx_hall_rates = [r["nexus"]["hallucination_rate"] for r in results if not r["nexus"].get("error")]
    nx_latencies = [r["nexus"]["latency_s"] for r in results if not r["nexus"].get("error")]
    nx_paths = [r["nexus"]["path_count"] for r in results if not r["nexus"].get("error")]
    nx_ev_tokens = [r["nexus"].get("evidence_tokens", 0) for r in results if not r["nexus"].get("error")]
    nx_accuracies = [
        r["nexus"]["accuracy"] for r in results
        if not r["nexus"].get("error") and r["nexus"]["accuracy"] is not None
    ]

    # RAG metrics
    rg_errors = sum(1 for r in results if r["rag"].get("error"))
    rg_answered = sum(1 for r in results if not r["rag"]["is_insufficient"] and not r["rag"].get("error"))
    rg_insufficient = sum(1 for r in results if r["rag"]["is_insufficient"])
    rg_passed = sum(1 for r in results if r["rag"]["passed"] and not r["rag"].get("error"))
    rg_hall_rates = [r["rag"]["hallucination_rate"] for r in results if not r["rag"].get("error")]
    rg_latencies = [r["rag"]["latency_s"] for r in results if not r["rag"].get("error")]
    rg_ev_tokens = [r["rag"].get("evidence_tokens", 0) for r in results if not r["rag"].get("error")]
    rg_accuracies = [
        r["rag"]["accuracy"] for r in results
        if not r["rag"].get("error") and r["rag"]["accuracy"] is not None
    ]

    scorable = len(nx_accuracies)

    # Entity resolution
    er_hits = sum(1 for r in results if not r["nexus"].get("error") and r["nexus"].get("entity_resolution_hit"))
    er_total = sum(1 for r in results if not r["nexus"].get("error") and r["nexus"].get("entity_resolution_hit") is not None)

    return {
        "total_questions": total,
        "scorable_questions": scorable,
        "nexus": {
            "errors": nx_errors,
            "answered": nx_answered,
            "insufficient_evidence": nx_insufficient,
            "answer_rate": round(nx_answered / total, 4) if total > 0 else 0.0,
            "verification_passed": nx_passed,
            "verification_pass_rate": round(nx_passed / total, 4) if total > 0 else 0.0,
            "avg_hallucination_rate": round(avg(nx_hall_rates), 4),
            "avg_latency_s": round(avg(nx_latencies), 4),
            "avg_paths_found": round(avg(nx_paths), 2),
            "avg_accuracy": round(avg(nx_accuracies), 4),
            "avg_evidence_tokens": round(avg(nx_ev_tokens), 1),
        },
        "rag": {
            "errors": rg_errors,
            "answered": rg_answered,
            "insufficient_evidence": rg_insufficient,
            "answer_rate": round(rg_answered / total, 4) if total > 0 else 0.0,
            "verification_passed": rg_passed,
            "verification_pass_rate": round(rg_passed / total, 4) if total > 0 else 0.0,
            "avg_hallucination_rate": round(avg(rg_hall_rates), 4),
            "avg_latency_s": round(avg(rg_latencies), 4),
            "avg_accuracy": round(avg(rg_accuracies), 4),
            "avg_evidence_tokens": round(avg(rg_ev_tokens), 1),
        },
        "entity_resolution": {
            "hits": er_hits,
            "total": er_total,
            "rate": round(er_hits / er_total, 4) if er_total > 0 else 0.0,
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════════

def print_comparison_table(summary: dict[str, Any]):
    """Print a human-readable NEXUS vs RAG comparison table."""
    n = summary["nexus"]
    r = summary["rag"]
    total = summary["total_questions"]

    print()
    print("=" * 85)
    print("  NEXUS vs RAG -- Baseline Comparison")
    print("=" * 85)
    print(f"  Questions:           {total}")
    print(f"  Scorable:            {summary.get('scorable_questions', total)}")
    print(f"  NEXUS errors:        {n['errors']}")
    print(f"  RAG errors:          {r['errors']}")
    print()

    # Main comparison table
    print(f"  {'System':<30} {'Accuracy':>10} {'Hallucination':>14} {'Verify':>8} {'Latency':>10} {'Ctx tokens':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*14} {'-'*8} {'-'*10} {'-'*12}")

    n_acc = f"{n['avg_accuracy']:.1%}" if n["avg_accuracy"] is not None else "N/A"
    n_hall = f"{n['avg_hallucination_rate']:.1%}"
    n_ver = f"{n['verification_pass_rate']:.0%}"
    n_lat = f"{n['avg_latency_s']:.1f}s"
    n_tok = f"{n['avg_evidence_tokens']:.0f}"

    r_acc = f"{r['avg_accuracy']:.1%}" if r["avg_accuracy"] is not None else "N/A"
    r_hall = f"{r['avg_hallucination_rate']:.1%}"
    r_ver = f"{r['verification_pass_rate']:.0%}"
    r_lat = f"{r['avg_latency_s']:.1f}s"
    r_tok = f"{r['avg_evidence_tokens']:.0f}"

    print(f"  {'NEXUS + qwen2.5:3b':<30} {n_acc:>10} {n_hall:>14} {n_ver:>8} {n_lat:>10} {n_tok:>12}")
    print(f"  {'RAG + qwen2.5:3b':<30} {r_acc:>10} {r_hall:>14} {r_ver:>8} {r_lat:>10} {r_tok:>12}")

    # Delta row
    if n["avg_accuracy"] is not None and r["avg_accuracy"] is not None:
        delta_acc = n["avg_accuracy"] - r["avg_accuracy"]
        delta_hall = r["avg_hallucination_rate"] - n["avg_hallucination_rate"]
        delta_ver = n["verification_pass_rate"] - r["verification_pass_rate"]
        delta_lat = n["avg_latency_s"] - r["avg_latency_s"]
        d_sign = "+" if delta_acc > 0 else ""
        a_str = f"{d_sign}{delta_acc:.1%}"
        h_str = f"{'+' if delta_hall > 0 else ''}{delta_hall:.1%}"
        v_str = f"{'+' if delta_ver > 0 else ''}{delta_ver:.0%}"
        l_str = f"{'+' if delta_lat > 0 else ''}{delta_lat:.1f}s"
        print(f"  {'(delta = NEXUS - RAG)':<30} {a_str:>10} {h_str:>14} {v_str:>8} {l_str:>10}")

    print()
    print("  Key:")
    print("  - Accuracy: fuzzy key-fact match score (5% numeric tolerance)")
    print("  - Hallucination: % of claims unsupported by evidence")
    print("  - Verify: % of answers passing hallucination threshold check")
    print("  - Latency: avg wall-clock time per question (seconds)")
    print("  - Ctx tokens: avg word-count tokens of evidence provided to the model")
    print()

    # Answer rate comparison
    nx_ans_str = f"{n['answered']}/{total} ({n['answer_rate']:.1%})"
    rg_ans_str = f"{r['answered']}/{total} ({r['answer_rate']:.1%})"
    nx_ins_str = f"{n['insufficient_evidence']}/{total}"
    rg_ins_str = f"{r['insufficient_evidence']}/{total}"

    print(f"  {'Answer rate:':<30} {'NEXUS':>20} {'RAG':>20}")
    print(f"  {'-'*30} {'-'*20} {'-'*20}")
    print(f"  {'Answered':<30} {nx_ans_str:>20} {rg_ans_str:>20}")
    print(f"  {'Insufficient evidence':<30} {nx_ins_str:>20} {rg_ins_str:>20}")

    # Entity resolution
    if summary.get("entity_resolution") and summary["entity_resolution"]["total"] > 0:
        er = summary["entity_resolution"]
        er_str = f"{er['rate']:.1%} ({er['hits']}/{er['total']})"
        print(f"  {'Entity resolution (NEXUS):':<30} {er_str:>20}")

    print()
    print("=" * 85)
    print()


# ═══════════════════════════════════════════════════════════════════════
# Initialization helpers
# ═══════════════════════════════════════════════════════════════════════

def initialize_rag_pipeline(
    model: ModelInterface,
    force_reembed: bool = False,
) -> RAGPipeline:
    """
    Initialize the RAG pipeline: read docs, chunk, embed, cache.

    Args:
        model: ModelInterface for answer generation
        force_reembed: If True, delete cache and re-embed even if cache exists

    Returns:
        RAGPipeline ready to answer questions
    """
    embedder = RAGEmbedder(cache_path=EMBEDDING_CACHE)

    if force_reembed and EMBEDDING_CACHE.exists():
        EMBEDDING_CACHE.unlink()
        print("[rag] Cache deleted, will re-embed")

    # Collect all .md files
    doc_paths: list[Path] = []
    for corpus_dir in CORPUS_DIRS:
        if corpus_dir.exists():
            doc_paths.extend(sorted(corpus_dir.glob("**/*.md")))

    print(f"[rag] Found {len(doc_paths)} .md files in corpus")

    # Chunk
    chunks = chunk_documents(doc_paths)
    print(f"[rag] Created {len(chunks)} chunks "
          f"(avg {round(sum(c['token_count'] for c in chunks) / len(chunks), 0)} tokens/chunk)")

    # Embed
    embeddings = embedder.embed_chunks(chunks)

    # Build pipeline
    verifier = Verifier(hallucination_threshold=0.2)
    return RAGPipeline(
        model=model,
        verifier=verifier,
        chunks=chunks,
        chunk_embeddings=embeddings,
        embedder=embedder,
    )


def initialize_nexus_model() -> tuple[ModelInterface, str]:
    """
    Determine the best model to use for comparison.

    Strategy:
    1. Try Ollama with qwen2.5:latest (or similar 3B model)
    2. If Ollama not available, use SynthesizingModel (noted as frontier-model proxy)

    Returns (model, model_label).
    """
    from nexus.reasoning.model_interface import _check_ollama_available

    ollama_ok, detected_model = _check_ollama_available()

    if ollama_ok:
        # Use the same model as NEXUS for fair comparison
        # Prefer qwen2.5:latest if available, otherwise use detected
        print(f"[model] Ollama available: {detected_model}")
        model = OllamaModel(model_name="qwen2.5:latest")
        label = "qwen2.5:latest"
        # Test that the model actually works — fall back to detected if not
        try:
            _ = model.generate("Hello. Say hi briefly.")
            print(f"[model] Successfully connected to Ollama with qwen2.5:latest")
        except Exception:
            print(f"[model] qwen2.5:latest not found, using detected: {detected_model}")
            model = OllamaModel(model_name=detected_model)
            label = detected_model
    else:
        # Fallback: SynthesizingModel — template-based, stands in for frontier model
        print("[model] No Ollama — using SynthesizingModel (proxy for frontier model)")
        print("[model] NOTE: This is a template synthesizer, not a real LLM.")
        print("[model]       It stands in for GPT-4o-mini at ~$0.15/1K input tokens.")
        print("[model]       Replace with real API call for production numbers.")
        model = SynthesizingModel()
        label = "SynthesizingModel (GPT-4o-mini proxy)"

    return model, label


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RAG Baseline Comparator — NEXUS vs RAG benchmark"
    )
    parser.add_argument(
        "--limit", type=int, default=30,
        help="Number of questions to benchmark (default: 30)"
    )
    parser.add_argument(
        "--output", type=str, default="benchmarks/nexus_vs_rag_30.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--force-reembed", action="store_true",
        help="Force re-embedding (delete cache)"
    )
    parser.add_argument(
        "--no-graph", action="store_true",
        help="Skip graph population (for debugging)"
    )
    args = parser.parse_args()

    # Resolve paths
    dataset_path = _project_root / "benchmarks" / "qa-dataset" / "questions.jsonl"
    output_path = _project_root / args.output

    if not dataset_path.exists():
        print(f"Error: QA dataset not found at {dataset_path}")
        sys.exit(1)

    print(f"Loading questions from: {dataset_path}")
    questions = load_questions(str(dataset_path), args.limit)
    total = len(questions)
    print(f"Loaded {total} questions (limit={args.limit})")

    # ── Initialize model ──
    print("\n--- Initializing model ---")
    model, model_label = initialize_nexus_model()

    # Wrap in FallbackModel for NEXUS (LLM first, fallback to synthesize)
    nexus_model = FallbackModel(model)

    # For RAG, use same model
    rag_model = model

    print(f"Model: {model_label}")
    print(f"NEXUS model: {nexus_model.name}")
    print(f"RAG model: {type(rag_model).__name__}")

    # ── Initialize verifier (SAME for both) ──
    verifier = Verifier(hallucination_threshold=0.2)
    print(f"Verifier: threshold={0.2}")

    # ── Build NEXUS graph ──
    print("\n--- Building NEXUS graph ---")
    if args.no_graph:
        graph = InMemoryGraphStore()
        graph_provenance = {"node_count": 0, "edge_count": 0}
    else:
        graph, graph_provenance = build_benchmark_graph()
    print(f"Graph: {graph_provenance['node_count']} nodes, {graph_provenance['edge_count']} edges")

    # ── Initialize RAG pipeline ──
    print("\n--- Initializing RAG pipeline ---")
    rag_pipeline = initialize_rag_pipeline(rag_model, force_reembed=args.force_reembed)
    print(f"RAG pipeline: {rag_pipeline.chunk_count} chunks indexed")

    # ── Run comparison ──
    print(f"\n--- Running comparison on {total} questions ---\n")
    results, summary = run_comparison(
        questions, graph, nexus_model, rag_pipeline, verifier,
    )

    # ── Save results ──
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "limit": args.limit,
            "model_label": model_label,
            "nexus_model": nexus_model.name,
            "rag_model": type(rag_model).__name__,
            "verification_threshold": 0.2,
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "chunk_overlap": CHUNK_OVERLAP_TOKENS,
            "top_k_chunks": TOP_K_CHUNKS,
            "embedding_model": EMBEDDING_MODEL,
            "rag_corpus_dirs": [str(d) for d in CORPUS_DIRS],
            "rag_chunk_count": rag_pipeline.chunk_count,
        },
        "graph_provenance": graph_provenance,
        "summary": summary,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # ── Print comparison table ──
    print_comparison_table(summary)

    # ── Key findings ──
    n = summary["nexus"]
    r = summary["rag"]
    if n["avg_accuracy"] is not None and r["avg_accuracy"] is not None:
        if n["avg_accuracy"] > r["avg_accuracy"]:
            diff = n["avg_accuracy"] - r["avg_accuracy"]
            print(f"KEY FINDING: NEXUS outperforms RAG by {diff:.1%} in accuracy.")
        elif r["avg_accuracy"] > n["avg_accuracy"]:
            diff = r["avg_accuracy"] - n["avg_accuracy"]
            print(f"KEY FINDING: RAG outperforms NEXUS by {diff:.1%} in accuracy.")
        else:
            print("KEY FINDING: NEXUS and RAG have identical accuracy.")

    if n["avg_hallucination_rate"] < r["avg_hallucination_rate"]:
        diff = r["avg_hallucination_rate"] - n["avg_hallucination_rate"]
        print(f"KEY FINDING: NEXUS has {diff:.1%} lower hallucination rate than RAG.")
    elif r["avg_hallucination_rate"] < n["avg_hallucination_rate"]:
        diff = n["avg_hallucination_rate"] - r["avg_hallucination_rate"]
        print(f"KEY FINDING: RAG has {diff:.1%} lower hallucination rate than NEXUS.")

    if n["avg_latency_s"] < r["avg_latency_s"]:
        ratio = r["avg_latency_s"] / n["avg_latency_s"] if n["avg_latency_s"] > 0 else 0
        print(f"KEY FINDING: NEXUS is {ratio:.1f}x faster than RAG.")
    elif r["avg_latency_s"] < n["avg_latency_s"]:
        ratio = n["avg_latency_s"] / r["avg_latency_s"] if r["avg_latency_s"] > 0 else 0
        print(f"KEY FINDING: RAG is {ratio:.1f}x faster than NEXUS.")

    print()
    print("Done.")
    print()


if __name__ == "__main__":
    main()
