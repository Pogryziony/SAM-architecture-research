"""
Graph data structures for NEXUS.

The graph store is the central knowledge representation:
- Nodes: entities, concepts, documents, code, tests, bugs, decisions, etc.
- Edges: typed, directed relationships with confidence scores and sources.

Storage backends:
  - InMemoryGraphStore: dict-based, for prototyping (Phase 1-2)
  - KuzuGraphStore: embedded graph DB, for production (Phase 4+)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Node:
    """A node in the knowledge graph."""
    id: str
    type: str  # Entity, Concept, Document, CodeFile, Function, TestCase, Bug, Decision, Requirement, Experiment, Metric
    properties: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)  # Evidence pointers
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


@dataclass
class Edge:
    """A directed, typed edge between two nodes."""
    type: str  # depends_on, caused_by, validates, contradicts, implements, mentioned_in, derived_from, related_to, replaces, blocked_by
    source: str  # source node ID
    target: str  # target node ID
    confidence: float = 1.0  # [0.0, 1.0]
    evidence: str = ""  # Source reference
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class Path:
    """A path through the graph: sequence of edges from start to end."""
    edges: list[Edge]
    score: float = 0.0

    @property
    def nodes(self) -> list[str]:
        """List of node IDs in traversal order."""
        if not self.edges:
            return []
        ids = [self.edges[0].source]
        for edge in self.edges:
            ids.append(edge.target)
        return ids

    @property
    def length(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        if not self.edges:
            return "Path(empty)"
        parts = [self.edges[0].source]
        for edge in self.edges:
            parts.append(f"--[{edge.type}]-->")
            parts.append(edge.target)
        return f"Path(score={self.score:.3f}, {' '.join(parts)})"


# Edge type weights for traversal scoring (higher = preferred during traversal)
EDGE_TYPE_WEIGHTS: dict[str, float] = {
    "caused_by": 1.0,
    "blocked_by": 0.95,
    "depends_on": 0.85,
    "validates": 0.80,
    "contradicts": 0.75,
    "implements": 0.70,
    "derived_from": 0.60,
    "replaces": 0.55,
    "related_to": 0.30,
    "mentioned_in": 0.20,
}

# Valid node types
NODE_TYPES = frozenset({
    "Entity", "Concept", "Document", "CodeFile", "Function",
    "TestCase", "Bug", "Decision", "Requirement", "Experiment", "Metric",
})

# Valid edge types
EDGE_TYPES = frozenset(EDGE_TYPE_WEIGHTS.keys())
