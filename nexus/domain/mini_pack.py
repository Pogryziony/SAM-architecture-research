"""Minimal second test domain — independent of SAM vocabulary.

Used to prove domain-pack loading without modifying NEXUS core.
"""

from __future__ import annotations

from typing import Any

from nexus.domain.pack import DomainPack, DomainPackMeta, populate_store
from nexus.graph import Edge, Node
from nexus.graph.store import InMemoryGraphStore


class MiniDomainPack(DomainPack):
    """Tiny geography/fact domain for interface tests."""

    @property
    def meta(self) -> DomainPackMeta:
        return DomainPackMeta(
            domain_id="mini",
            version="mini-v1",
            description="Minimal non-SAM test domain (cities and countries).",
            locales=("en",),
        )

    def build_graph(self) -> InMemoryGraphStore:
        nodes = [
            Node(
                id="City_Warsaw",
                type="Entity",
                aliases=["Warsaw", "Warszawa"],
                properties={"key_finding": "Warsaw is the capital of Poland."},
                sources=["mini://cities/warsaw"],
            ),
            Node(
                id="Country_Poland",
                type="Entity",
                aliases=["Poland", "Polska"],
                properties={"key_finding": "Poland is a country in Europe."},
                sources=["mini://countries/poland"],
            ),
            Node(
                id="City_Krakow",
                type="Entity",
                aliases=["Krakow", "Cracow", "Kraków"],
                properties={"key_finding": "Krakow is a major city in Poland."},
                sources=["mini://cities/krakow"],
            ),
        ]
        edges = [
            Edge(
                type="related_to",
                source="City_Warsaw",
                target="Country_Poland",
                confidence=1.0,
                evidence="mini://cities/warsaw",
            ),
            Edge(
                type="related_to",
                source="City_Krakow",
                target="Country_Poland",
                confidence=1.0,
                evidence="mini://cities/krakow",
            ),
        ]
        return populate_store(nodes, edges)

    def entity_aliases(self) -> dict[str, list[str]]:
        return {
            "City_Warsaw": ["Warsaw", "Warszawa"],
            "Country_Poland": ["Poland", "Polska"],
            "City_Krakow": ["Krakow", "Cracow", "Kraków"],
        }

    def evaluation_tasks(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "mini_q1",
                "question": "What is the capital of Poland?",
                "gold_answer": "Warsaw is the capital of Poland.",
                "gold_entities": ["City_Warsaw", "Country_Poland"],
                "question_type": "single_hop",
                "domain": "mini",
                "should_abstain": False,
            },
            {
                "id": "mini_q2",
                "question": "Is Krakow related to Poland?",
                "gold_answer": "Krakow is a major city in Poland.",
                "gold_entities": ["City_Krakow", "Country_Poland"],
                "question_type": "relation",
                "domain": "mini",
                "should_abstain": False,
            },
            {
                "id": "mini_q3",
                "question": "What is the population of Atlantis?",
                "gold_answer": "Insufficient evidence to answer.",
                "gold_entities": [],
                "question_type": "unanswerable",
                "domain": "mini",
                "should_abstain": True,
            },
        ]
