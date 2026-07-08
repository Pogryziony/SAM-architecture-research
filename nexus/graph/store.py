"""
In-memory graph store for NEXUS (Phase 1-2 prototype).

Simple dict-based implementation with:
- Node storage by ID
- Edge storage (outgoing and incoming indexes)
- Type indexes for fast lookup
- Name index for fuzzy matching
"""

from __future__ import annotations

from collections import defaultdict
from difflib import get_close_matches
from typing import Optional

from . import Node, Edge, Path


class InMemoryGraphStore:
    """Dict-based graph store for prototyping."""

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges_out: dict[str, list[Edge]] = defaultdict(list)
        self._edges_in: dict[str, list[Edge]] = defaultdict(list)
        self._type_index: dict[str, list[str]] = defaultdict(list)
        self._name_index: dict[str, str] = {}  # normalized_name → node_id

    # ── Node operations ──

    def add_node(self, node: Node) -> None:
        """Add or update a node."""
        self._nodes[node.id] = node
        self._type_index[node.type].append(node.id)
        self._name_index[self._normalize(node.id)] = node.id

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
        """Add a directed edge. Both source and target nodes must exist."""
        if edge.source not in self._nodes:
            raise KeyError(f"Source node '{edge.source}' not found")
        if edge.target not in self._nodes:
            raise KeyError(f"Target node '{edge.target}' not found")
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
        """Fuzzy-find a node by name. Returns node_id or None."""
        # Exact match first
        normalized = self._normalize(name)
        if normalized in self._name_index:
            return self._name_index[normalized]

        # Fuzzy match
        matches = get_close_matches(normalized, list(self._name_index.keys()), n=1, cutoff=cutoff)
        if matches:
            return self._name_index[matches[0]]
        return None

    def find_entities(self, names: list[str]) -> list[Optional[str]]:
        """Fuzzy-find multiple entities at once."""
        return [self.find_entity(name) for name in names]

    @staticmethod
    def _normalize(name: str) -> str:
        return name.strip().lower().replace(" ", "_").replace("-", "_")

    # ── Traversal ──

    def traverse(
        self,
        start_nodes: list[str],
        max_depth: int = 4,
        edge_types: Optional[set[str]] = None,
        direction: str = "both",
    ) -> list[Path]:
        """
        BFS traversal from start nodes.

        Args:
            start_nodes: Entry node IDs
            max_depth: Maximum path length in edges
            edge_types: Allowed edge types (None = all)
            direction: 'out', 'in', or 'both'

        Returns:
            List of Paths found during traversal
        """
        paths: list[Path] = []

        for start in start_nodes:
            if start not in self._nodes:
                continue
            # BFS queue: (current_node, path_edges_so_far)
            queue: list[tuple[str, list[Edge]]] = [(start, [])]

            while queue:
                current, path_edges = queue.pop(0)

                if len(path_edges) >= max_depth:
                    if path_edges:
                        paths.append(Path(edges=list(path_edges)))
                    continue

                edges = self.get_edges(current, direction)
                expanded = False

                for edge in edges:
                    if edge_types and edge.type not in edge_types:
                        continue
                    # Determine next node based on direction relative to edge
                    if direction == "out" or (direction == "both" and edge.source == current):
                        next_node = edge.target
                    elif direction == "in" or (direction == "both" and edge.target == current):
                        next_node = edge.source
                    else:
                        continue
                    queue.append((next_node, path_edges + [edge]))
                    expanded = True

                if not expanded and path_edges:
                    paths.append(Path(edges=list(path_edges)))

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
