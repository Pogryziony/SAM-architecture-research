"""
Dialogue state: temporary activation subgraph with recency decay.

Entities activated in prior turns get boosted resolution priority.
Activation decays exponentially with each turn. Pronouns and definite
references resolve to highest-activation type-compatible node.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexus.graph.store import InMemoryGraphStore


# ── Pronoun / definite reference patterns ──
# These trigger reference resolution against the dialogue state rather
# than fresh global entity spotting.
_PRONOUN_PATTERNS: list[tuple[str, str]] = [
    (r"\bit\b", "it"),           # "why did IT fail"
    (r"\bthis\b", "this"),       # "what does THIS prove"
    (r"\bthat\b", "that"),       # "was THAT decision justified"
    (r"\bthese\b", "these"),     # "how do THESE compare"
    (r"\bthose\b", "those"),     # "what about THOSE"
]

# Definite references: "the experiment", "the retriever", "the model", etc.
_DEFINITE_PATTERNS: list[tuple[str, str]] = [
    (r"\bthe experiment\b", "experiment"),
    (r"\bthe retriever\b", "retriever"),
    (r"\bthe model\b", "model"),
    (r"\bthe selector\b", "selector"),
    (r"\bthe gate\b", "gate"),
    (r"\bthe result\b", "result"),
    (r"\bthe architecture\b", "architecture"),
    (r"\bthe memory\b", "memory"),
    (r"\bthe baseline\b", "baseline"),
    (r"\bthe bottleneck\b", "bottleneck"),
    (r"\bthe verifier\b", "verifier"),
    (r"\bthe pipeline\b", "pipeline"),
    (r"\bthe dataset\b", "dataset"),
    (r"\bthe pivot\b", "pivot"),
    (r"\bthe validation\b", "validation"),
    (r"\bten model\b", "model"),          # Polish: "ten model"
    (r"\bta architektura\b", "architektura"),  # Polish: "ta architektura"
    (r"\bto rozwiązanie\b", "rozwiązanie"),    # Polish: "to rozwiązanie"
]

# Polish pronouns
_POLISH_PRONOUN_PATTERNS: list[tuple[str, str]] = [
    (r"\bto\b", "to"),           # "dlaczego TO zawiodło"
    (r"\bten\b", "ten"),         # "jak TEN model działa"
    (r"\bta\b", "ta"),           # "czy TA architektura"
    (r"\btego\b", "tego"),       # "wynik TEGO eksperymentu"
    (r"\btej\b", "tej"),         # "celem TEJ walidacji"
]


def _detect_pronouns(question: str) -> list[str]:
    """Detect pronoun/definite reference tokens in a question.
    
    Returns a list of reference token types (e.g., ['it', 'the experiment']).
    """
    lowered = question.lower()
    refs: list[str] = []
    
    for pattern, ref_type in _PRONOUN_PATTERNS:
        if re.search(pattern, lowered):
            refs.append(ref_type)
    
    for pattern, ref_type in _DEFINITE_PATTERNS:
        if re.search(pattern, lowered):
            refs.append(ref_type)
    
    for pattern, ref_type in _POLISH_PRONOUN_PATTERNS:
        if re.search(pattern, lowered):
            refs.append(ref_type)
    
    return refs


class DialogueState:
    """Temporary activation subgraph with recency decay.
    
    Entities activated in prior turns get boosted resolution priority.
    Activation decays exponentially with each turn. Pronouns and definite
    references resolve to the highest-activation type-compatible node.
    
    Attributes:
        decay: Multiplicative decay factor applied each turn to all activations.
        context_window: Maximum number of past turns considered for activation.
        _activation: Map from entity_id to activation strength [0.0, 1.0].
    """
    
    def __init__(self, decay: float = 0.7, context_window: int = 5):
        self.decay = decay
        self.context_window = context_window
        self._activation: dict[str, float] = {}  # entity_id -> activation
        self._turn_count: int = 0
    
    def update(self, entity_ids: list[str]) -> None:
        """Decay all activations, then boost newly-active entities.
        
        Called after each turn's entity resolution. Existing activations
        are multiplied by decay; entities resolved in this turn receive
        a +0.5 boost (capped at 1.0).
        
        Args:
            entity_ids: Entity IDs that were resolved in the current turn.
        """
        self._turn_count += 1
        
        # Decay all existing activations
        for eid in list(self._activation):
            self._activation[eid] *= self.decay
        
        # Boost newly-active entities
        for eid in entity_ids:
            self._activation[eid] = min(1.0, self._activation.get(eid, 0.0) + 0.5)
        
        # Remove dead activations (below noise floor)
        self._activation = {k: v for k, v in self._activation.items() if v > 0.01}
    
    def get_active_entities(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Return most-activated entities, sorted by activation descending.
        
        Args:
            top_k: Maximum number of entities to return.
            
        Returns:
            List of (entity_id, activation_strength) tuples.
        """
        return sorted(self._activation.items(), key=lambda x: -x[1])[:top_k]
    
    def get_activation(self, entity_id: str) -> float:
        """Get the current activation of a specific entity.
        
        Args:
            entity_id: The entity ID to look up.
            
        Returns:
            Activation strength [0.0, 1.0], or 0.0 if not in state.
        """
        return self._activation.get(entity_id, 0.0)
    
    def has_references(self, question: str) -> bool:
        """Check if the question contains pronoun or definite references.
        
        Args:
            question: The natural language question text.
            
        Returns:
            True if the question has references that should be resolved
            against the dialogue state rather than fresh global search.
        """
        return len(_detect_pronouns(question)) > 0
    
    def resolve_reference(self, pronoun_or_ref: str, graph) -> str | None:
        """Resolve pronoun/definite reference to highest-activation type-compatible node.
        
        Handles: "it", "this", "that", "the experiment", "the result",
        Polish: "to", "ten model", "ta architektura".
        
        For pronouns like "it"/"this"/"that", returns the single highest-activated
        entity regardless of type.
        
        For definite references like "the experiment", filters active entities
        by type compatibility (e.g., "the experiment" prefers Experiment nodes,
        "the retriever" prefers nodes with "retriev" in their name).
        
        Args:
            pronoun_or_ref: The reference token (e.g., "it", "the experiment").
            graph: The InMemoryGraphStore for type lookups.
            
        Returns:
            Resolved entity ID, or None if no compatible active entity found.
        """
        active = self.get_active_entities(top_k=10)
        if not active:
            return None
        
        lowered_ref = pronoun_or_ref.lower()
        
        # Definite references with type hints
        type_hints: dict[str, set[str]] = {
            "experiment": {"Experiment"},
            "retriever": {"Experiment", "Concept"},
            "model": {"Experiment", "Concept", "Decision"},
            "selector": {"Experiment", "Concept"},
            "gate": {"Experiment", "Concept"},
            "result": {"Experiment", "Metric", "Concept"},
            "architecture": {"Concept", "Decision"},
            "memory": {"Experiment", "Concept"},
            "baseline": {"Experiment"},
            "bottleneck": {"Concept", "Bug"},
            "verifier": {"Concept", "Decision"},
            "pipeline": {"Experiment", "Concept", "Decision"},
            "dataset": {"Experiment"},
            "pivot": {"Decision", "Concept"},
            "validation": {"Experiment"},
            # Polish hints
            "model": {"Experiment", "Concept"},
            "architektura": {"Concept", "Decision"},
            "rozwiązanie": {"Concept", "Decision"},
        }
        
        # Check if the reference has a type hint
        for hint_key, preferred_types in type_hints.items():
            if hint_key in lowered_ref:
                # Filter active entities by preferred types
                for eid, act in active:
                    node = graph.get_node(eid)
                    if node and node.type in preferred_types:
                        return eid
                # No type-compatible entity found — fall through to generic resolution
        
        # Generic pronoun resolution: return highest-activated entity
        return active[0][0] if active else None
    
    def get_active_entity_ids(self) -> set[str]:
        """Return set of all currently active entity IDs.
        
        Useful for checking whether an entity is in the dialogue context.
        """
        return set(self._activation.keys())
    
    def reset(self) -> None:
        """Clear all dialogue state — for starting a new dialogue."""
        self._activation.clear()
        self._turn_count = 0
    
    def __repr__(self) -> str:
        active_list = [f"{eid}:{act:.2f}" for eid, act in self.get_active_entities(5)]
        return f"DialogueState(turn={self._turn_count}, active=[{', '.join(active_list)}])"
