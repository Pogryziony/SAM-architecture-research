"""Bounded, versioned rule engine for Stage 4 (toy Datalog-like subset).

Rules are Horn-style implications over typed graph edges. Evaluation is
depth-bounded and records premises + rule IDs for every inferred fact.
Asserted (graph) facts stay distinct from inferred facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from nexus.graph.store import InMemoryGraphStore


@dataclass(frozen=True)
class Rule:
    """One versioned inference rule."""

    rule_id: str
    version: str
    # Body: list of (src_var, relation, dst_var)
    body: tuple[tuple[str, str, str], ...]
    # Head: (src_var, relation, dst_var)
    head: tuple[str, str, str]


@dataclass(frozen=True)
class InferredFact:
    source: str
    relation: str
    target: str
    rule_id: str
    rule_version: str
    premises: tuple[tuple[str, str, str], ...]
    depth: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "premises": [list(item) for item in self.premises],
            "depth": self.depth,
        }


@dataclass
class RuleEngineResult:
    inferred: list[InferredFact] = field(default_factory=list)
    activations: int = 0
    truncated: bool = False
    truncation_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "inferred": [fact.to_dict() for fact in self.inferred],
            "activations": self.activations,
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }


class RuleEngine:
    """Evaluate a small set of versioned rules with hard activation bounds."""

    def __init__(
        self,
        rules: Sequence[Rule],
        *,
        max_depth: int = 2,
        max_activations: int = 64,
    ):
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if max_activations < 1:
            raise ValueError("max_activations must be >= 1")
        self.rules = list(rules)
        self.max_depth = int(max_depth)
        self.max_activations = int(max_activations)

    def evaluate(self, graph: InMemoryGraphStore) -> RuleEngineResult:
        asserted = self._asserted_facts(graph)
        known = set(asserted)
        inferred: list[InferredFact] = []
        activations = 0
        frontier = list(asserted)
        depth = 0
        result = RuleEngineResult()

        while frontier and depth < self.max_depth:
            depth += 1
            next_frontier: list[tuple[str, str, str]] = []
            for rule in self.rules:
                for binding, premises in self._match_rule(rule, known):
                    head_src = binding[rule.head[0]]
                    head_rel = rule.head[1]
                    head_tgt = binding[rule.head[2]]
                    fact = (head_src, head_rel, head_tgt)
                    activations += 1
                    if activations > self.max_activations:
                        result.inferred = inferred
                        result.activations = activations - 1
                        result.truncated = True
                        result.truncation_reason = "max_activations"
                        return result
                    if fact in known:
                        continue
                    known.add(fact)
                    next_frontier.append(fact)
                    inferred.append(
                        InferredFact(
                            source=head_src,
                            relation=head_rel,
                            target=head_tgt,
                            rule_id=rule.rule_id,
                            rule_version=rule.version,
                            premises=tuple(premises),
                            depth=depth,
                        )
                    )
            frontier = next_frontier

        result.inferred = inferred
        result.activations = activations
        if frontier and depth >= self.max_depth:
            result.truncated = True
            result.truncation_reason = "max_depth"
        return result

    @staticmethod
    def _asserted_facts(graph: InMemoryGraphStore) -> list[tuple[str, str, str]]:
        facts: list[tuple[str, str, str]] = []
        for node_id in sorted(graph._nodes):
            for edge in graph.get_outgoing(node_id):
                facts.append((edge.source, edge.type, edge.target))
        return facts

    def _match_rule(
        self,
        rule: Rule,
        known: set[tuple[str, str, str]],
    ) -> Iterable[tuple[dict[str, str], list[tuple[str, str, str]]]]:
        """Yield variable bindings that satisfy the rule body against known facts."""
        if not rule.body:
            return

        def search(
            body_index: int,
            binding: dict[str, str],
            premises: list[tuple[str, str, str]],
        ) -> Iterable[tuple[dict[str, str], list[tuple[str, str, str]]]]:
            if body_index >= len(rule.body):
                yield dict(binding), list(premises)
                return
            src_var, relation, dst_var = rule.body[body_index]
            for source, rel, target in list(known):
                if rel != relation:
                    continue
                next_binding = dict(binding)
                if src_var in next_binding and next_binding[src_var] != source:
                    continue
                if dst_var in next_binding and next_binding[dst_var] != target:
                    continue
                next_binding[src_var] = source
                next_binding[dst_var] = target
                yield from search(
                    body_index + 1,
                    next_binding,
                    premises + [(source, rel, target)],
                )

        yield from search(0, {}, [])
