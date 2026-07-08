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
from difflib import get_close_matches
from typing import Optional

from . import Node, Edge, Path, PathStep


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

    # ── Node operations ──

    def add_node(self, node: Node) -> None:
        """Add or update a node. Updates do not duplicate type-index entries."""
        is_new = node.id not in self._nodes
        self._nodes[node.id] = node
        if is_new:
            self._type_index[node.type].append(node.id)
        self._name_index[self._normalize(node.id)] = node.id
        # Index aliases for human-friendly entity resolution
        if node.aliases:
            for alias in node.aliases:
                normalized_alias = self._normalize(alias)
                # Only index if not already taken (first-come, first-served)
                if normalized_alias not in self._alias_index:
                    self._alias_index[normalized_alias] = node.id
        # Index property values for keyword-based entity lookup
        for value in node.properties.values():
            if isinstance(value, str):
                for token in self._tokenize(value):
                    if node.id not in self._property_index[token]:
                        self._property_index[token].append(node.id)

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

        # Fuzzy match on node IDs
        matches = get_close_matches(normalized, list(self._name_index.keys()), n=1, cutoff=cutoff)
        if matches:
            return self._name_index[matches[0]]

        # Fuzzy match on aliases
        alias_matches = get_close_matches(normalized, list(self._alias_index.keys()), n=1, cutoff=cutoff)
        if alias_matches:
            return self._alias_index[alias_matches[0]]

        return None

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
        hits.sort(key=lambda x: x[1], reverse=True)
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
