"""
NEXUS model interface — clean abstraction for plugging in any reasoning model.

Provides:
  - ModelInterface: abstract base with generate(prompt) -> str
  - DummyModel: rule-based synthesizer for end-to-end testing (no ML)
  - LocalModel: placeholder for future local model integration
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional


class ModelInterface(ABC):
    """Abstract interface for reasoning models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt string."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class DummyModel(ModelInterface):
    """
    Rule-based model for testing without real inference.

    Parses the evidence section of the prompt and synthesizes a
    coherent, plausible answer by extracting key facts. Produces
    deterministic output — no randomness.

    Designed to exercise the full pipeline end-to-end: a question
    with evidence should produce a citation-backed answer; a question
    without evidence should produce the "insufficient evidence" response.
    """

    def generate(self, prompt: str) -> str:
        """
        Synthesize an answer from the evidence in the prompt.

        Strategy:
        1. Extract all "Extracted facts:" bullet points
        2. If none found, return "Insufficient evidence to answer."
        3. Otherwise, synthesize a summary paragraph citing the facts
        """
        # Check for "no evidence" marker
        if "(No evidence found" in prompt:
            return "Insufficient evidence to answer."

        # Extract facts from the evidence section
        facts_section = _extract_section(prompt, "Extracted facts:", "Sources")
        if not facts_section:
            facts_section = _extract_section(prompt, "Extracted facts:", "ANSWER:")

        facts = []
        for line in facts_section.split("\n"):
            line = line.strip()
            # Match bullet-point facts: "  - ..."
            if line.startswith("- "):
                facts.append(line[2:].strip())

        if not facts:
            # Fallback: try to extract facts from path chains
            paths_section = _extract_section(prompt, "Knowledge graph paths:", "Extracted facts")
            if paths_section:
                # Try to extract relation info from path descriptions
                path_facts = _extract_facts_from_paths(paths_section)
                if path_facts:
                    facts = path_facts

        if not facts:
            return "Insufficient evidence to answer."

        # Extract the question
        question = _extract_line(prompt, "QUESTION:")
        # Build a summary answer
        answer_parts = ["Based on the evidence:"]

        # Group facts by confidence level for better readability
        high_conf_facts = []
        low_conf_facts = []
        for fact in facts:
            conf_match = re.search(r'confidence:\s*([\d.]+)', fact)
            if conf_match:
                conf = float(conf_match.group(1))
                (high_conf_facts if conf >= 0.7 else low_conf_facts).append(fact)
            else:
                high_conf_facts.append(fact)

        # Present high-confidence facts first
        for fact in high_conf_facts[:5]:
            # Clean up the fact for a natural-sounding answer
            clean = _clean_fact_for_answer(fact)
            answer_parts.append(f"- {clean}")

        if low_conf_facts:
            answer_parts.append("\nAdditional lower-confidence findings:")
            for fact in low_conf_facts[:3]:
                clean = _clean_fact_for_answer(fact)
                answer_parts.append(f"- {clean}")

        # Add source citation
        sources_section = _extract_section(prompt, "Sources", "ANSWER:")
        source_count = sources_section.count("[") if sources_section else 0
        if source_count > 0:
            answer_parts.append(f"\nSources: {source_count} document(s) referenced in the evidence pack.")

        return "\n".join(answer_parts)


class LocalModel(ModelInterface):
    """
    Placeholder for local model integration (e.g., llama.cpp, Ollama, vLLM).

    Usage (future):
        model = LocalModel(model_path="./models/phi-3-mini.Q4_K_M.gguf")
        model.load()   # lazily loads when needed
        answer = model.generate(prompt)

    Currently raises NotImplementedError — this is scaffolding for Phase 3+.
    """

    def __init__(self, model_path: str, **kwargs: Any):
        self._model_path = model_path
        self._model: Any = None
        self._loaded = False
        self._kwargs = kwargs

    def load(self) -> None:
        """Lazy-load the model (not implemented yet)."""
        raise NotImplementedError(
            "LocalModel is a placeholder for future implementation. "
            "Use DummyModel for testing."
        )

    def generate(self, prompt: str) -> str:
        """Generate a response (not implemented yet)."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        raise NotImplementedError(
            "LocalModel.generate() is a placeholder. Implement with your "
            "chosen inference backend (llama.cpp, Ollama, vLLM, etc.)."
        )

    @property
    def name(self) -> str:
        return f"LocalModel({self._model_path})"

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"LocalModel(path={self._model_path!r}, status={status})"


# ── Helper functions for prompt parsing ──


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """Extract text between two markers."""
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    start_idx += len(start_marker)
    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        return text[start_idx:].strip()
    return text[start_idx:end_idx].strip()


def _extract_line(text: str, marker: str) -> str:
    """Extract the content on the line after a marker."""
    idx = text.find(marker)
    if idx == -1:
        return ""
    line_start = idx + len(marker)
    line_end = text.find("\n", line_start)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip()


def _clean_fact_for_answer(fact: str) -> str:
    """Clean a fact string for inclusion in a natural-sounding answer."""
    # Remove confidence suffix
    fact = re.sub(r'\s*\(confidence:\s*[\d.]+\s*\)\s*$', '', fact).strip()
    # Capitalize first letter
    if fact:
        fact = fact[0].upper() + fact[1:]
    # Ensure it ends with a period
    if fact and fact[-1] not in ".!?":
        fact += "."
    return fact


def _extract_facts_from_paths(paths_text: str) -> list[str]:
    """Extract relation facts from path chain descriptions."""
    facts = []
    # Match arrow chains: X --[type](conf:N)--> Y
    pattern = r'(\S+)\s+--\[(\w+)\].*?-->\s+(\S+)'
    for m in re.finditer(pattern, paths_text):
        from_node = m.group(1)
        edge_type = m.group(2).replace("_", " ")
        to_node = m.group(3)
        facts.append(f"{from_node} {edge_type} {to_node}")
    return facts
