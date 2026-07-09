"""
Semantic entity resolution via embedding similarity.

Uses all-MiniLM-L6-v2 (384-dim, CPU-friendly, ~80MB) to embed
node names + descriptions. At query time, finds top-K nodes by
cosine similarity to the question embedding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nexus.graph.store import InMemoryGraphStore


class NodeEmbeddingIndex:
    """Semantic entity resolution via embedding similarity.

    Uses all-MiniLM-L6-v2 (384-dim, CPU-friendly, ~80MB) to embed
    node names + descriptions. At query time, finds top-K nodes by
    cosine similarity to the question embedding.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._node_ids: list[str] = []
        self._embeddings: np.ndarray | None = None  # built by build_index()

    def build_index(self, graph: InMemoryGraphStore):
        """Index all graph nodes: embed name + description + key_finding."""
        texts = []
        for node_id in sorted(graph._nodes.keys()):
            node = graph.get_node(node_id)
            # Build rich text for embedding: name + type + description + key_finding
            name = node_id.replace("_", " ")
            desc = node.properties.get("description", "") if node.properties else ""
            kf = node.properties.get("key_finding", "") if node.properties else ""
            text = f"{name}. {desc} {kf}".strip()
            if not text:
                text = name
            self._node_ids.append(node_id)
            texts.append(text)

        self._embeddings = self._model.encode(texts, show_progress_bar=True)
        print(f"[embedding] Indexed {len(self._node_ids)} nodes")

    def query(self, question: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find top-K nodes by cosine similarity to question."""
        if self._embeddings is None:
            return []
        q_emb = self._model.encode([question])[0]
        # Normalize for cosine similarity
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        emb_norm = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8)
        scores = np.dot(emb_norm, q_norm)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self._node_ids[i], float(scores[i])) for i in top_indices]
