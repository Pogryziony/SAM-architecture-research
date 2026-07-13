"""
In-memory graph store for NEXUS (Phase 1-2 prototype).

Simple dict-based implementation with:
- Node storage by ID
- Edge storage (outgoing and incoming indexes)
- Type indexes for fast lookup
- Name index for fuzzy matching
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from . import Node, Edge, Path, PathStep


def _split_into_segments(node_id: str) -> list[str]:
    """Split a node ID into word-like segments on _, space, and camelCase boundaries."""
    # First split on underscore and space
    parts = re.split(r"[_ ]+", node_id)
    segments: list[str] = []
    for part in parts:
        if not part:
            continue
        sub = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
        sub = re.sub(r"([A-Za-z])(\d)", r"\1 \2", sub)
        sub = re.sub(r"(\d)([A-Za-z])", r"\1 \2", sub)
        segments.extend(sub.split())
    return [s.lower() for s in segments if s]


def _compute_trigrams(text: str) -> set[str]:
    """Compute all character trigrams from a normalized text string.
    Pads with $ prefix/suffix for boundary sensitivity."""
    padded = f"$${text}$$"
    return {padded[i:i + 3] for i in range(len(padded) - 2)} if len(padded) >= 3 else set()


class InMemoryGraphStore:
    """Dict-based graph store for prototyping."""

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges_out: dict[str, list[Edge]] = defaultdict(list)
        self._edges_in: dict[str, list[Edge]] = defaultdict(list)
        self._type_index: dict[str, list[str]] = defaultdict(list)
        self._name_index: dict[str, str] = {}  # normalized_name → node_id
        self._alias_index: dict[str, str] = {}  # normalized_alias → node_id
        self._property_index: dict[str, list[str]] = defaultdict(list)
        # token (normalized) → list of node_ids that have that token in properties

        # ── Parser optimization: inverted word index ──
        # Each word → set of node_ids whose normalized name contains that word.
        # Built once at populate time, used by spot_entities to prune fuzzy-match candidates.
        self._word_index: dict[str, set[str]] = defaultdict(set)

        # ── Parser optimization: precomputed property text ──
        # node_id → concatenated key_finding + description for keyword-boost scoring.
        # Avoids per-candidate get_node() + property lookups during ranking.
        self._property_text: dict[str, str] = {}

        # ── Parser optimization: cached name-index key list ──
        # Avoids list(self._name_index.keys()) allocation on every find_entity call.
        self._name_index_keys: list[str] = []
        self._alias_index_keys: list[str] = []

        # ── Parser optimization: combined word set for n-gram pre-filtering ──
        # Set of all words that appear in ANY node's name-index key segments.
        # Used to skip n-grams that have zero chance of matching any node.
        self._all_indexed_words: set[str] = set()

        # ── Parser optimization: trigram index for fast fuzzy matching ──
        # Each trigram (3-char substring) → set of (norm_name, node_id) tuples.
        # Replaces expensive get_close_matches with fast set-intersection scoring.
        self._trigram_index: dict[str, set[tuple[str, str]]] = defaultdict(set)
        # node_id → precomputed trigram set for Jaccard scoring
        self._node_trigrams: dict[str, set[str]] = {}

        # ── Parser optimization: node_id → normalized_name reverse map ──
        # For O(1) lookup during second-pass word-boundary matching.
        self._norm_name_by_id: dict[str, str] = {}

    # ── Node operations ──

    def add_node(self, node: Node) -> None:
        """Add or update a node. Updates do not duplicate type-index entries."""
        is_new = node.id not in self._nodes
        self._nodes[node.id] = node
        if is_new:
            self._type_index[node.type].append(node.id)
        self._name_index[self._normalize(node.id)] = node.id

        # Build node_id → normalized_name reverse mapping
        self._norm_name_by_id[node.id] = self._normalize(node.id)

        # ── Word index: index each word-segment of the node ID ──
        for segment in _split_into_segments(node.id):
            segment_lower = segment.lower()
            self._word_index[segment_lower].add(node.id)
            self._all_indexed_words.add(segment_lower)

        # Index aliases for human-friendly entity resolution
        if node.aliases:
            for alias in node.aliases:
                normalized_alias = self._normalize(alias)
                # Only index if not already taken (first-come, first-served)
                if normalized_alias not in self._alias_index:
                    self._alias_index[normalized_alias] = node.id
                # Also index alias words into word index
                for segment in _split_into_segments(normalized_alias):
                    segment_lower = segment.lower()
                    self._word_index[segment_lower].add(node.id)
                    self._all_indexed_words.add(segment_lower)

        # Index property values for keyword-based entity lookup
        for value in node.properties.values():
            if isinstance(value, str):
                for token in self._tokenize(value):
                    if node.id not in self._property_index[token]:
                        self._property_index[token].append(node.id)

        # ── Precompute property text for keyword-boost scoring ──
        text_parts: list[str] = []
        for prop_name in ("key_finding", "description"):
            val = node.properties.get(prop_name, "")
            if isinstance(val, str) and val:
                text_parts.append(val.lower())
        if text_parts:
            self._property_text[node.id] = " ".join(text_parts)

        # Invalidate cached name-index and alias-index key lists
        self._name_index_keys = []
        self._alias_index_keys = []

        # ── Build trigram index for fast fuzzy matching ──
        norm_name = self._normalize(node.id)
        trigrams = _compute_trigrams(norm_name)
        self._node_trigrams[node.id] = trigrams
        for tg in trigrams:
            self._trigram_index[tg].add((norm_name, node.id))

        # Also index alias trigrams
        if node.aliases:
            for alias in node.aliases:
                norm_alias = self._normalize(alias)
                alias_trigrams = _compute_trigrams(norm_alias)
                for tg in alias_trigrams:
                    self._trigram_index[tg].add((norm_alias, node.id))

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def nodes_of_type(self, node_type: str) -> list[Node]:
        return [self._nodes[nid] for nid in self._type_index.get(node_type, [])]

    # ── Edge operations ──

    def add_edge(self, edge: Edge) -> None:
        """Add a directed edge. Both source and target nodes must exist.
        Duplicate edges (same type, source, target) are silently ignored."""
        if edge.source not in self._nodes:
            raise KeyError(f"Source node '{edge.source}' not found")
        if edge.target not in self._nodes:
            raise KeyError(f"Target node '{edge.target}' not found")
        # Dedup: check if identical edge already exists
        for existing in self._edges_out.get(edge.source, []):
            if existing.type == edge.type and existing.target == edge.target:
                return  # Duplicate — skip
        self._edges_out[edge.source].append(edge)
        self._edges_in[edge.target].append(edge)

    def get_outgoing(self, node_id: str) -> list[Edge]:
        return self._edges_out.get(node_id, [])

    def get_incoming(self, node_id: str) -> list[Edge]:
        return self._edges_in.get(node_id, [])

    def get_edges(self, node_id: str, direction: str = "both") -> list[Edge]:
        """Get edges for a node. direction: 'out', 'in', or 'both'."""
        edges = []
        if direction in ("out", "both"):
            edges.extend(self.get_outgoing(node_id))
        if direction in ("in", "both"):
            edges.extend(self.get_incoming(node_id))
        return edges

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._edges_out.values())

    # ── Entity lookup ──

    def find_entity(self, name: str, cutoff: float = 0.8) -> Optional[str]:
        """Find a node by name or alias (exact/normalized, then alias, then fuzzy). Returns node_id or None."""
        normalized = self._normalize(name)

        # Exact match on node ID
        if normalized in self._name_index:
            return self._name_index[normalized]

        # Exact alias match
        if normalized in self._alias_index:
            return self._alias_index[normalized]

        # Fast trigram-based fuzzy match (replaces expensive get_close_matches)
        return self.find_entity_fast(name, cutoff=cutoff)

    def find_entity_fast(
        self, name: str, cutoff: float = 0.8, candidate_ids: set[str] | None = None,
    ) -> Optional[str]:
        """Fast entity lookup using trigram Jaccard similarity instead of SequenceMatcher.
        If candidate_ids is provided, only considers those node IDs."""
        normalized = self._normalize(name)

        # Exact match on node ID
        if normalized in self._name_index:
            nid = self._name_index[normalized]
            if candidate_ids is None or nid in candidate_ids:
                return nid
            return None

        # Exact alias match
        if normalized in self._alias_index:
            nid = self._alias_index[normalized]
            if candidate_ids is None or nid in candidate_ids:
                return nid
            return None

        # ── Trigram-based fuzzy scoring ──
        query_trigrams = _compute_trigrams(normalized)
        if not query_trigrams:
            return None

        # Collect candidates: all (norm_name, node_id) that share at least one trigram
        trigram_hits: dict[str, int] = {}  # node_id → shared trigram count
        for tg in sorted(query_trigrams):
            for norm_name, nid in sorted(self._trigram_index.get(tg, ())):
                if candidate_ids is not None and nid not in candidate_ids:
                    continue
                trigram_hits[nid] = trigram_hits.get(nid, 0) + 1

        if not trigram_hits:
            return None

        # Score using trigram containment: shared / min(|query|, |node|)
        # This doesn't penalize longer node names that simply have extra context.
        # Require at least 2 shared trigrams for short strings to avoid false positives.
        best_nid: Optional[str] = None
        best_score: float = 0.0
        for nid, shared in sorted(trigram_hits.items()):
            if shared < 2:
                continue
            node_tgs = self._node_trigrams.get(nid, set())
            if not node_tgs:
                continue
            # Containment: how much of the SMALLER set is covered by the intersection
            min_size = min(len(query_trigrams), len(node_tgs))
            score = shared / min_size if min_size > 0 else 0.0
            if score > best_score or (score == best_score and best_nid is not None and nid < best_nid):
                best_score = score
                best_nid = nid

        if best_nid is not None and best_score >= cutoff:
            return best_nid
        return None

    def find_entity_indexed(
        self, name: str, candidate_ids: set[str] | None = None, cutoff: float = 0.8,
    ) -> Optional[str]:
        """Like find_entity, but only matches against candidate_ids (from word index).
        If candidate_ids is None or empty, falls back to full find_entity."""
        normalized = self._normalize(name)

        # Exact match on node ID
        if normalized in self._name_index:
            nid = self._name_index[normalized]
            if candidate_ids is None or nid in candidate_ids:
                return nid
            return None

        # Exact alias match
        if normalized in self._alias_index:
            nid = self._alias_index[normalized]
            if candidate_ids is None or nid in candidate_ids:
                return nid
            return None

        # Trigram-based fuzzy match against candidates
        return self.find_entity_fast(name, cutoff=cutoff, candidate_ids=candidate_ids)

    def has_any_indexed_word(self, text: str) -> bool:
        """Return True if any word-segment of text exists in _all_indexed_words.
        Used to skip n-grams that have zero chance of matching any node."""
        for segment in _split_into_segments(text):
            if segment.lower() in self._all_indexed_words:
                return True
        return False

    def get_word_index_candidates(self, query_text: str) -> set[str]:
        """Return the set of node IDs whose name-index key shares at least one
        word-segment with query_text. Used to prune fuzzy-match candidates."""
        candidates: set[str] = set()
        for segment in _split_into_segments(query_text):
            segment_lower = segment.lower()
            if segment_lower in self._word_index:
                candidates.update(self._word_index[segment_lower])
        return candidates

    def get_property_text(self, node_id: str) -> str:
        """Return the precomputed property text for keyword-boost scoring."""
        return self._property_text.get(node_id, "")

    def find_entity_exact(self, name: str) -> Optional[str]:
        """Find a node by name or alias with exact matching only (no fuzzy fallback)."""
        normalized = self._normalize(name)
        if normalized in self._name_index:
            return self._name_index[normalized]
        if normalized in self._alias_index:
            return self._alias_index[normalized]
        return None

    def find_entities(self, names: list[str]) -> list[Optional[str]]:
        """Fuzzy-find multiple entities at once."""
        return [self.find_entity(name) for name in names]

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into normalized tokens for keyword indexing."""
        normalized = text.lower().replace("-", " ").replace("_", " ").replace(".", " ")
        tokens = re.findall(r'[a-z0-9]+', normalized)
        # Filter out very short tokens and common stop words
        return [t for t in tokens if len(t) >= 3]

    def find_entity_by_keywords(self, question: str, cutoff: float = 0.5) -> list[tuple[str, int]]:
        """
        Find entity IDs whose property tokens match keywords in the question.

        Uses token intersection: for each token in the question, find nodes
        that have that token in their property values. Returns (node_id, match_count)
        tuples ranked by number of matching tokens (highest first).
        """
        question_tokens = set(self._tokenize(question))
        if not question_tokens:
            return []

        # Count matches per node
        node_hits: dict[str, int] = {}
        for token in question_tokens:
            for nid in self._property_index.get(token, []):
                node_hits[nid] = node_hits.get(nid, 0) + 1

        # Filter: require at least 2 token matches for confidence
        hits = [(nid, count) for nid, count in node_hits.items() if count >= 2]
        hits.sort(key=lambda x: (-x[1], x[0]))
        return hits

    # ── Traversal ──

    def traverse(
        self,
        start_nodes: list[str],
        max_depth: int = 4,
        edge_types: Optional[set[str]] = None,
        direction: str = "both",
    ) -> list[Path]:
        """
        BFS traversal from start nodes with cycle protection.

        Args:
            start_nodes: Entry node IDs
            max_depth: Maximum path length in edges
            edge_types: Allowed edge types (None = all)
            direction: 'out', 'in', or 'both'

        Returns:
            List of Paths found during traversal. Each PathStep records
            whether the edge was traversed in reverse.
        """
        paths: list[Path] = []

        for start in start_nodes:
            if start not in self._nodes:
                continue
            # BFS queue: (current_node, steps_so_far, visited_set)
            queue: list[tuple[str, list[PathStep], set[str]]] = [
                (start, [], {start})
            ]

            while queue:
                current, steps, visited = queue.pop(0)

                if len(steps) >= max_depth:
                    if steps:
                        paths.append(Path(steps=list(steps)))
                    continue

                edges = self.get_edges(current, direction)
                expanded = False

                for edge in edges:
                    if edge_types and edge.type not in edge_types:
                        continue
                    # Determine next node and whether edge is reversed
                    if direction == "out":
                        # Only follow edges where current is the source
                        if edge.source != current:
                            continue
                        next_node = edge.target
                        reversed_flag = False
                    elif direction == "in":
                        # Only follow edges where current is the target
                        if edge.target != current:
                            continue
                        next_node = edge.source
                        reversed_flag = True
                    else:  # both
                        if edge.source == current:
                            next_node = edge.target
                            reversed_flag = False
                        elif edge.target == current:
                            next_node = edge.source
                            reversed_flag = True
                        else:
                            continue

                    # Cycle protection
                    if next_node in visited:
                        continue

                    step = PathStep(edge=edge, reversed=reversed_flag)
                    queue.append((next_node, steps + [step], visited | {next_node}))
                    expanded = True

                if not expanded and steps:
                    paths.append(Path(steps=list(steps)))

        return paths

    # ── Stats ──

    def stats(self) -> dict:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "node_types": {t: len(ids) for t, ids in self._type_index.items()},
        }

    def __repr__(self) -> str:
        s = self.stats()
        return f"InMemoryGraphStore(nodes={s['node_count']}, edges={s['edge_count']})"
