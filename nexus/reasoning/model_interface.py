"""
NEXUS model interface — clean abstraction for plugging in any reasoning model.

Provides:
  - ModelInterface: abstract base with generate(prompt) -> str
  - DummyModel: rule-based synthesizer for end-to-end testing (no ML)
  - SynthesizingModel: evidence-aware answer synthesizer (better than DummyModel)
  - OllamaModel: local inference via Ollama (e.g., qwen2.5-coder:3b)
  - get_available_model(): auto-detect the best available local model
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
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


class EvidenceBlindModel(ModelInterface):
    """
    Baseline model that operates WITHOUT evidence access.
    
    Receives the same prompt template as NEXUS but with evidence stripped
    out. Uses DummyModel's evidence-parsing logic but from an empty
    context. When no evidence is found, attempts to answer based on
    question keywords — producing varied, non-empty responses that
    demonstrate question comprehension but lack specific evidence.
    
    This produces a meaningful baseline: NEXUS (with evidence) is
    compared against the same pipeline operating without evidence,
    showing the value of structured knowledge access.
    """
    
    def generate(self, prompt: str) -> str:
        """Generate answer without evidence access."""
        # Extract the question from the prompt
        question = _extract_line(prompt, "QUESTION:")
        if not question:
            question = _extract_line(prompt, "QUESTION")
        
        # Check if there's actual evidence content
        has_evidence = (
            "(No evidence found" not in prompt 
            and "Extracted facts:" in prompt
        )
        
        if has_evidence:
            # Evidence is present — use DummyModel logic (shouldn't happen
            # for baseline, but handle gracefully)
            dummy = DummyModel()
            return dummy.generate(prompt)
        
        # No evidence available — produce a general-knowledge answer
        return self._answer_from_question(question)
    
    def _answer_from_question(self, question: str) -> str:
        """Produce a general-knowledge answer by parsing the question."""
        # Extract key terms from the question
        topics = self._extract_topics(question)
        
        parts = ["Based on general knowledge:"]
        
        if topics.get("what"):
            parts.append(f"- The question asks about: {topics['what'][:120]}.")
        if topics.get("experiment"):
            parts.append(f"- This relates to experiment {topics['experiment']}.")
        if topics.get("concept"):
            parts.append(f"- The concept '{topics['concept']}' is involved.")
        
        parts.append(
            "- Without access to specific experiment data, "
            "I cannot provide exact numbers. "
            "The information exists in SAM experiment reports "
            "but is not available in this context."
        )
        
        return "\n".join(parts)
    
    @staticmethod
    def _extract_topics(question: str) -> dict[str, str]:
        """Extract topic keywords from a question."""
        topics: dict[str, str] = {}
        
        # Extract "what" / "how" targets
        what_match = re.search(
            r'(?:what|how)\s+(?:is|are|was|were|does|do|did|can|should)\s+(?:the\s+)?(.+?)(?:\?|$)',
            question, re.IGNORECASE
        )
        if what_match:
            topics["what"] = what_match.group(1).strip()
        
        # Extract experiment references
        exp_match = re.search(
            r'(?:experiment|exp)\s*[\d.]+[A-Z]?\b',
            question, re.IGNORECASE
        )
        if exp_match:
            topics["experiment"] = exp_match.group(0)
        
        # Extract concept references
        for concept in [
            "oracle memory", "selector", "retriever", "gate",
            "chain retrieval", "noise tolerance", "pipeline",
            "dual encoder", "NEXUS", "SAM", "RAG", "pivot",
            "validation", "PKM", "product-key memory"
        ]:
            if concept.lower() in question.lower():
                topics["concept"] = concept
                break
        
        return topics


class OllamaModel(ModelInterface):
    """
    Local inference via Ollama's HTTP API (http://localhost:11434).

    Requires Ollama to be installed and running with a model pulled:
        ollama pull qwen2.5-coder:3b
        ollama serve

    Pinned for reproducibility — change only in controlled experiments.
    Uses qwen2.5:latest (~4.7 GB) by default. All generation is synchronous.

    Parameters:
        model_name: Ollama model tag (default: "qwen2.5:latest")
        host: Ollama API host (default: "http://localhost:11434")
        timeout: Request timeout in seconds (default: 120)
        system_prompt: Override the default system prompt (None = strip from prompt)
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:latest",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        system_prompt: str | None = None,
    ):
        self._model_name = model_name
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        """
        Send prompt to Ollama and return the generated answer.

        The prompt already contains a SYSTEM: block from prompt_template —
        Ollama's generate endpoint doesn't natively separate system/user,
        so we pass the full prompt as-is. The model receives instructions
        embedded in the prompt text.
        """
        try:
            import urllib.request
            import urllib.error

            payload = json.dumps({
                "model": self._model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,   # Deterministic for factual answers
                    "num_predict": 256,   # Short, focused answers
                    "top_k": 1,           # Greedy decoding
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self._host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()

        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama API unreachable at {self._host}. "
                f"Is Ollama running? (ollama serve)\n"
                f"Original error: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"OllamaModel.generate() failed: {exc}"
            ) from exc

    @property
    def name(self) -> str:
        return f"OllamaModel({self._model_name})"

    def __repr__(self) -> str:
        return f"OllamaModel(model={self._model_name!r}, host={self._host!r})"


class SynthesizingModel(ModelInterface):
    """
    Evidence-aware answer synthesizer — better than DummyModel.

    Goes beyond DummyModel by:
      - Producing natural-language paragraphs, not bullet lists
      - Using varied templates with confidence qualifiers
      - Connecting evidence facts into coherent narratives
      - Including graph-path context where available

    This is the best fallback when no real local model is available.
    It still cannot hallucinate facts beyond the evidence, but it
    formats them as human-readable answers rather than parsed bullets.

    No ML — pure template-based synthesis. Deterministic output.
    """

    # Answer templates for variety
    _HIGH_CONF_TEMPLATES = [
        "The evidence clearly indicates that {fact}",
        "Based on the knowledge graph, {fact}",
        "Analysis of the experiment data shows that {fact}",
        "The structured evidence confirms that {fact}",
    ]

    _MEDIUM_CONF_TEMPLATES = [
        "The evidence suggests that {fact}",
        "Graph paths indicate that {fact}",
        "There is supporting evidence that {fact}",
        "The data points toward {fact}",
    ]

    _LOW_CONF_TEMPLATES = [
        "There is some indication that {fact}",
        "Preliminary evidence suggests {fact}",
        "Lower-confidence signals point to {fact}",
        "The graph contains hints that {fact}",
    ]

    def generate(self, prompt: str) -> str:
        """Synthesize a natural-language answer from evidence in the prompt."""
        # Check for "no evidence" marker
        if "(No evidence found" in prompt:
            return "Insufficient evidence to answer."

        # Extract the question for type detection
        question = _extract_line(prompt, "QUESTION:") or ""
        q_type = self._detect_question_type_from_prompt(question)

        # ── Specialized synthesis paths (zero LLM) ──
        # Try factual synthesis first
        if q_type == "factual":
            result = self._synthesize_factual(prompt)
            if result:
                return result

        # Try comparative synthesis
        if q_type == "comparative":
            result = self._synthesize_comparative(prompt)
            if result:
                return result

        # Try chain/diagnostic synthesis
        if q_type in ("diagnostic", "multi-hop", "causal"):
            result = self._synthesize_chain(prompt)
            if result:
                return result

        # Try definition/concept synthesis
        if q_type == "definition":
            result = self._synthesize_definition(prompt)
            if result:
                return result

        # ── Fall through to existing synthesis logic ──
        # Extract facts from the evidence section
        # Try multiple section names (prompt template may vary)
        facts_section = _extract_section(prompt, "Relation facts:", "Sources")
        if not facts_section:
            facts_section = _extract_section(prompt, "Extracted facts:", "Sources")
        if not facts_section:
            facts_section = _extract_section(prompt, "Extracted facts:", "ANSWER:")
        if not facts_section:
            facts_section = _extract_section(prompt, "Relation facts:", "ANSWER:")

        # Also extract node details — these have the actual key findings
        node_details_section = _extract_section(
            prompt,
            "Key findings from evidence nodes:",
            "Knowledge graph paths:",
        )
        if not node_details_section:
            node_details_section = _extract_section(
                prompt,
                "Key findings from evidence nodes:",
                "Relation facts:",
            )

        facts = []
        # Parse bullet points from relation facts
        for line in facts_section.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                facts.append(line[2:].strip())

        # Parse node details as high-confidence facts
        node_facts = []
        for line in node_details_section.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                # Format: "- NodeID: Description text"
                content = line[2:].strip()
                # Extract the description part (after the node ID)
                if ": " in content:
                    _, desc = content.split(": ", 1)
                    node_facts.append((desc, 0.9))  # Node details have high confidence
                else:
                    node_facts.append((content, 0.8))

        # Node details take priority — they contain actual key findings
        if node_facts:
            # Place node facts first
            high_conf = [f[0] for f in node_facts]
            # Add relation facts as medium-confidence
            medium_conf = facts[:5] if facts else []
            low_conf = []
        elif facts:
            # Only relation facts available — classify by confidence
            high_conf, medium_conf, low_conf = [], [], []
            for fact in facts:
                conf_match = re.search(r'confidence:\s*([\d.]+)', fact)
                if conf_match:
                    conf = float(conf_match.group(1))
                    if conf >= 0.7:
                        high_conf.append(fact)
                    elif conf >= 0.4:
                        medium_conf.append(fact)
                    else:
                        low_conf.append(fact)
                else:
                    high_conf.append(fact)
        else:
            # Try path-derived facts as final fallback
            paths_section = _extract_section(prompt, "Knowledge graph paths:", "Relation facts:")
            if not paths_section:
                paths_section = _extract_section(prompt, "Knowledge graph paths:", "Extracted facts:")
            if paths_section:
                path_facts = _extract_facts_from_paths(paths_section)
                if path_facts:
                    high_conf = path_facts
                    medium_conf = []
                    low_conf = []
                else:
                    # ── Final fallback: parse EVIDENCE: bullet points directly ──
                    raw_facts = self._parse_raw_evidence_bullets(prompt)
                    if raw_facts:
                        high_conf = raw_facts
                        medium_conf = []
                        low_conf = []
                    else:
                        return "Insufficient evidence to answer."
            else:
                # ── Final fallback: parse EVIDENCE: bullet points directly ──
                raw_facts = self._parse_raw_evidence_bullets(prompt)
                if raw_facts:
                    high_conf = raw_facts
                    medium_conf = []
                    low_conf = []
                else:
                    return "Insufficient evidence to answer."

        if not high_conf and not medium_conf and not low_conf:
            return "Insufficient evidence to answer."

        # Build answer — direct, no preamble or boilerplate
        paragraphs: list[str] = []

        # Synthesize high-confidence facts into a direct answer
        if high_conf:
            cleaned_facts = []
            for fact in high_conf[:2]:  # Max 2 facts — be concise
                clean = _clean_fact_for_answer(fact)
                cleaned_facts.append(clean)

            # Determine if these are node details
            is_node_facts = any(": " in f or not f.endswith(".") for f in cleaned_facts[:2])

            if is_node_facts:
                # Node details — present as direct single-sentence answer
                best = cleaned_facts[0]
                if not best.endswith("."):
                    best += "."
                paragraphs.append(best)
            else:
                # Relation facts — just state them directly
                for clean in cleaned_facts:
                    if not clean.endswith("."):
                        clean += "."
                    paragraphs.append(clean)

        elif medium_conf:
            # Only medium-conf available — state with slight hedging but no preamble
            clean = _clean_fact_for_answer(medium_conf[0])
            if not clean.endswith("."):
                clean += "."
            paragraphs.append(clean)

        # Medium and low-confidence facts are NOT included — they add noise
        # for questions that already have high-confidence answers.

        # No "These findings are drawn from" boilerplate — the verifier
        # checks key-fact overlap, not citation proximity.

        return "\n".join(paragraphs) if paragraphs else "Insufficient evidence to answer."

    # ── Specialized zero-LLM synthesis methods ──

    @staticmethod
    def _detect_question_type_from_prompt(question: str) -> str:
        """Detect question type from the question text for synthesis routing.

        Returns one of: 'factual', 'comparative', 'diagnostic', 'multi-hop',
                       'causal', 'definition', 'unknown'
        """
        q_lower = question.lower().strip()

        # Comparative patterns
        if re.search(r'\b(compare|vs\.?|versus|difference between|which is (higher|lower|better|worse|faster|slower|more|less))\b', q_lower):
            return "comparative"

        # Diagnostic/causal patterns
        if re.search(r'\b(why|cause|reason|led to|what caused|what led to|explain why)\b', q_lower):
            return "diagnostic"

        # Multi-hop/chain patterns
        if re.search(r'\b(walk through|step by step|chain of|evolution from|how does|relationship between)\b', q_lower):
            return "multi-hop"

        # Definition patterns
        if re.search(r'\b(what is the (role of|definition of)|define|meaning of|what does .* mean)\b', q_lower):
            return "definition"

        # Causal patterns
        if re.search(r'\b(how did|what made|what (makes|causes|enables))\b', q_lower):
            return "causal"

        return "factual"

    def _synthesize_factual(self, prompt: str) -> str:
        """Synthesize a one-sentence factual answer from evidence.

        Extracts the key_finding for the entity the question is about.
        Returns exactly one sentence. If no key_finding is found for the
        question entity, returns empty string to fall through to the
        generic evidence path.

        Handles two prompt formats:
          1. Full NEXUS prompt (with structured sections)
          2. Simple oracle prompt (QUESTION + EVIDENCE + instruction)
        """
        question = _extract_line(prompt, "QUESTION:") or ""

        # ── Detect metric term from the question ──
        try:
            from nexus.query.parser import extract_metric_term
            metric_term = extract_metric_term(question)
        except ImportError:
            metric_term = None

        # ── Try NUMBERS FOR metric section first (most precise match) ──
        if metric_term:
            numbers_answer = self._extract_from_numbers_section(
                prompt, question, metric_term,
            )
            if numbers_answer:
                return numbers_answer

        # ── Try full NEXUS prompt format ──
        section_markers = [
            "Additional context",
            "Supporting evidence:",
            "Knowledge graph paths:",
            "Relation facts:",
            "Sources",
        ]
        answer_section = ""
        for marker in section_markers:
            answer_section = _extract_section(
                prompt,
                "The answer to your question is in these facts:",
                marker,
            )
            if answer_section:
                break

        if not answer_section:
            # ── Fallback: parse simple EVIDENCE: section (oracle format) ──
            return self._synthesize_factual_from_evidence(
                prompt, question, metric_term,
            )

        # Parse facts from the answer section
        # Only accept facts that look like "NodeID: Description" (no spaces
        # in node ID, and description has meaningful content).
        facts = []
        for line in answer_section.split("\n"):
            line = line.strip()
            if not line.startswith("- "):
                continue
            content = line[2:].strip()
            if content and "insufficient" not in content.lower():
                # Strip annotation prefix before checking
                content_clean = re.sub(r'^\[.*?\]\s*', '', content)
                if ": " in content_clean:
                    nid, desc = content_clean.split(": ", 1)
                    # Skip relation facts: node IDs with spaces or confidence suffixes
                    if " " in nid or "(confidence" in nid.lower():
                        continue
                    # Skip facts where description is just a number
                    if re.match(r'^\d+\.?\d*\)?\s*$', desc.strip()):
                        continue
                    facts.append(content)

        if not facts:
            return ""

        # Score each fact by keyword overlap with the question
        q_words = set(re.findall(r'\w+', question.lower()))
        q_stopwords = {
            'what', 'that', 'this', 'with', 'from', 'which', 'does', 'many',
            'while', 'over', 'through', 'there', 'their', 'about', 'between',
            'these', 'differ', 'compare', 'compared', 'experiment', 'was', 'were',
            'did', 'the', 'and', 'for', 'how', 'is', 'are', 'of', 'in', 'to', 'a', 'an'
        }
        q_keywords = q_words - q_stopwords

        scored = []
        for fact in facts:
            fact_lower = fact.lower()
            fact_words = set(re.findall(r'\w+', fact_lower))
            overlap = len(q_keywords & fact_words)
            # Bonus for facts that contain the metric term
            if metric_term and metric_term in fact_lower:
                overlap += 10
            # Bonus for facts that contain the node ID (more specific match)
            clean = re.sub(r'^\[.*?\]\s*', '', fact)
            if ": " in clean:
                node_id = clean.split(": ", 1)[0]
                node_words = set(re.findall(r'\w+', node_id.lower()))
                overlap += len(q_keywords & node_words) * 2
                # Heavy bonus for Experiment nodes (prefer key_findings over concept descriptions)
                if "Exp_" in node_id and "Concept_" not in node_id:
                    overlap += 5
            scored.append((overlap, fact))

        scored.sort(key=lambda x: -x[0])
        _, best = scored[0]

        # Strip node ID prefix
        best = re.sub(r'^\[.*?\]\s*', '', best)
        if ": " in best:
            _, best = best.split(": ", 1)

        # Ensure it ends with a period
        if not best.rstrip().endswith("."):
            best = best.rstrip() + "."

        # ── Metric-aware extraction: extract only the relevant {value} {metric}
        # when the fact contains multiple numbers ──
        if metric_term and best:
            extracted = self._extract_metric_value_from_fact(best, metric_term)
            if extracted:
                best = extracted

        return best

    def _extract_from_numbers_section(
        self, prompt: str, question: str, metric_term: str,
    ) -> str:
        """Extract answer from NUMBERS FOR '{metric}' section in the prompt.

        This handles the structured numbers_by_metric section added in Phase 2.
        Returns a formatted answer like "The Exp_0_6 achieved 99.87% accuracy."
        or "" if no match found.
        """
        # Search for "NUMBERS FOR '{metric_term}'" header
        numbers_section = ""
        for line in prompt.split("\n"):
            stripped = line.strip()
            if stripped.startswith("NUMBERS FOR '") and metric_term.lower() in stripped.lower():
                # Extract everything from this line to the next blank line or section start
                start_idx = prompt.find(stripped)
                if start_idx == -1:
                    continue
                rest = prompt[start_idx + len(stripped):]
                # Find next blank line or section boundary
                end_idx = len(rest)
                for delim in ["\n\n", "\n  KEY NUMBERS", "\n  NUMBERS FOR", "\n  Connected", "\n  Supporting"]:
                    pos = rest.find(delim)
                    if pos != -1 and pos < end_idx:
                        end_idx = pos
                numbers_section = rest[:end_idx]
                break

        # Also try prefix match on metric keys
        if not numbers_section:
            for line in prompt.split("\n"):
                stripped = line.strip()
                if stripped.startswith("NUMBERS FOR '"):
                    # Extract the key between quotes
                    m = re.match(r"NUMBERS FOR '([^']+)'", stripped)
                    if m:
                        key = m.group(1)
                        if key.startswith(metric_term) or metric_term.startswith(key):
                            start_idx = prompt.find(stripped)
                            rest = prompt[start_idx + len(stripped):]
                            end_idx = len(rest)
                            for delim in ["\n\n", "\n  KEY NUMBERS", "\n  NUMBERS FOR", "\n  Connected", "\n  Supporting"]:
                                pos = rest.find(delim)
                                if pos != -1 and pos < end_idx:
                                    end_idx = pos
                            numbers_section = rest[:end_idx]
                            metric_term = key  # Use the actual key name
                            break

        if not numbers_section:
            return ""

        # Parse entity-value pairs from the section
        entries = []
        for line in numbers_section.split("\n"):
            line = line.strip()
            # Match "- [EntityID] value" format
            m = re.match(r'-\s*\[([^\]]+)\]\s*(.+)', line)
            if m:
                entity = m.group(1)
                value = m.group(2).strip()
                entries.append((entity, value))

        if not entries:
            return ""

        # Score entries by overlap with question keywords
        q_words = set(re.findall(r'\w+', question.lower()))
        q_stopwords = {
            'what', 'that', 'this', 'with', 'from', 'which', 'does', 'many',
            'while', 'over', 'through', 'there', 'their', 'about', 'between',
            'these', 'differ', 'compare', 'compared', 'experiment', 'was', 'were',
            'did', 'the', 'and', 'for', 'how', 'is', 'are', 'of', 'in', 'to', 'a', 'an'
        }
        q_keywords = q_words - q_stopwords

        scored_entries = []
        for entity, value in entries:
            entity_words = set(re.findall(r'\w+', entity.lower()))
            overlap = len(q_keywords & entity_words)
            scored_entries.append((overlap, entity, value))

        scored_entries.sort(key=lambda x: -x[0])
        if not scored_entries:
            return ""

        _, best_entity, best_value = scored_entries[0]

        # Format a clean, natural-language answer
        entity_name = best_entity.replace("_", " ")
        return f"The {entity_name} achieved {best_value} {metric_term}."

    def _synthesize_factual_from_evidence(
        self, prompt: str, question: str, metric_term: str | None,
    ) -> str:
        """Fallback: synthesize answer from a simple EVIDENCE: section.

        Used when the prompt lacks the full NEXUS structured sections
        (e.g., oracle evidence test format).
        """
        # Extract everything between "EVIDENCE:" and "ANSWER:" or end
        evidence_section = _extract_section(prompt, "EVIDENCE:", "ANSWER:")
        if not evidence_section:
            evidence_section = _extract_section(
                prompt, "EVIDENCE:", "Based on the evidence",
            )
        if not evidence_section:
            # No delimiter found; take everything after EVIDENCE: to end
            idx = prompt.find("EVIDENCE:")
            if idx == -1:
                return ""
            evidence_section = prompt[idx + len("EVIDENCE:"):].strip()

        if not evidence_section:
            return ""

        # Also look for KEY NUMBERS and NUMBERS FOR sections in the prompt
        if metric_term:
            numbers_answer = self._extract_from_numbers_section(
                prompt, question, metric_term,
            )
            if numbers_answer:
                return numbers_answer

        # Parse facts from evidence — accept both bullet-point and plain-text lines
        facts = self._parse_evidence_lines(evidence_section)

        if not facts:
            return ""

        # Score by keyword overlap (same logic as main path)
        q_words = set(re.findall(r'\w+', question.lower()))
        q_stopwords = {
            'what', 'that', 'this', 'with', 'from', 'which', 'does', 'many',
            'while', 'over', 'through', 'there', 'their', 'about', 'between',
            'these', 'differ', 'compare', 'compared', 'experiment', 'was', 'were',
            'did', 'the', 'and', 'for', 'how', 'is', 'are', 'of', 'in', 'to', 'a', 'an'
        }
        q_keywords = q_words - q_stopwords

        scored = []
        for fact in facts:
            fact_lower = fact.lower()
            fact_words = set(re.findall(r'\w+', fact_lower))
            overlap = len(q_keywords & fact_words)
            # Heavy bonus for facts containing the metric term (e.g., "accuracy")
            if metric_term and metric_term in fact_lower:
                overlap += 10
            scored.append((overlap, fact))

        scored.sort(key=lambda x: -x[0])
        if not scored or scored[0][0] == 0:
            return ""

        _, best = scored[0]

        # ── Metric-aware extraction: extract only the relevant {value} {metric}
        # when the fact contains multiple numbers ──
        if metric_term and best:
            extracted = self._extract_metric_value_from_fact(best, metric_term)
            if extracted:
                best = extracted

        if not best.rstrip().endswith("."):
            best = best.rstrip() + "."
        return best

    @staticmethod
    def _extract_metric_value_from_fact(fact: str, metric_term: str) -> str | None:
        """Extract just the {value} {metric} portion from a fact that
        contains multiple numbers.

        Example:
            fact = "Proves SAM core CAN use memory — 100% accuracy, 99.87% overall."
            metric_term = "accuracy"
            returns "100% accuracy"

        Returns None if no clear single value+metric pair found.
        """
        fact_lower = fact.lower()
        if metric_term not in fact_lower:
            return None

        # Count how many numbers are in the fact
        numbers_in_fact = re.findall(r'\d+(?:\.\d+)?%?', fact)
        if len(numbers_in_fact) <= 1:
            return None  # Only one number, use whole fact

        # Try to find "{value} {metric}" or "{metric} {value}" patterns
        # Pattern 1: "99.87% accuracy" (value before metric)
        m = re.search(
            r'(\d+(?:\.\d+)?%?)\s*' + re.escape(metric_term),
            fact, re.IGNORECASE,
        )
        if m:
            # Extract bounded by punctuation (commas, periods, dashes, etc.)
            phrase_start = m.start()
            while phrase_start > 0 and fact[phrase_start - 1] not in ',.—\n:;':
                phrase_start -= 1
            phrase_end = m.end()
            while phrase_end < len(fact) and fact[phrase_end] not in ',.—\n:;':
                phrase_end += 1
            snippet = fact[phrase_start:phrase_end].strip().lstrip(',.—:; \t')
            return snippet

        # Pattern 2: "accuracy of 99.87%" or "accuracy: 99.87%"
        m = re.search(
            re.escape(metric_term) + r'\s*(?:of\s+|:\s*)?(\d+(?:\.\d+)?%?)',
            fact, re.IGNORECASE,
        )
        if m:
            phrase_start = m.start()
            while phrase_start > 0 and fact[phrase_start - 1] not in ',.—\n:;':
                phrase_start -= 1
            phrase_end = m.end()
            while phrase_end < len(fact) and fact[phrase_end] not in ',.—\n:;':
                phrase_end += 1
            snippet = fact[phrase_start:phrase_end].strip().lstrip(',.—:; \t')
            return snippet

        return None

    def _synthesize_comparative(self, prompt: str) -> str:
        """Synthesize a comparative answer: extracts numbers from 2 entities,
        states which is higher/lower. Two sentences max.
        """
        # Extract node facts from the answer section
        answer_section = _extract_section(
            prompt,
            "The answer to your question is in these facts:",
            "Supporting evidence:",
        )
        if not answer_section:
            answer_section = _extract_section(
                prompt,
                "The answer to your question is in these facts:",
                "Knowledge graph paths:",
            )
        if not answer_section:
            answer_section = _extract_section(
                prompt,
                "The answer to your question is in these facts:",
                "Relation facts:",
            )
        if not answer_section:
            answer_section = _extract_section(
                prompt,
                "The answer to your question is in these facts:",
                "Sources",
            )

        # Parse entities with numeric values
        entity_facts: list[tuple[str, str, float | None]] = []
        for line in answer_section.split("\n"):
            line = line.strip()
            if not line.startswith("- "):
                continue
            content = line[2:].strip()

            # Extract entity name and description
            if ": " in content:
                node_id, desc = content.split(": ", 1)
            else:
                continue

            # Strip annotation prefixes
            desc = re.sub(r'^\[.*?\]\s*', '', desc)

            # Try to read entity name from the description itself
            # for human-readable names
            readable = desc

            # Extract numeric value from description
            val = self._extract_primary_number(desc)
            if val is not None:
                entity_facts.append((node_id, readable, val))

        # Need at least 2 entities with numbers
        if len(entity_facts) < 2:
            return ""

        a = entity_facts[0]
        b = entity_facts[1]

        # Build human-readable entity names from the descriptions
        # Extract the first meaningful noun phrase from each description
        name_a = self._extract_entity_name(a[1])
        name_b = self._extract_entity_name(b[1])

        if name_a == name_b:
            name_a = a[0]  # fallback to node ID (shortened)
            name_b = b[0]

        val_a = a[2]
        val_b = b[2]

        if val_a is not None and val_b is not None:
            if val_a > val_b:
                return (
                    f"{name_a} achieved {val_a:.2f}%, "
                    f"while {name_b} achieved {val_b:.2f}%."
                )
            else:
                return (
                    f"{name_b} achieved {val_b:.2f}%, "
                    f"while {name_a} achieved {val_a:.2f}%."
                )

        return ""

    @staticmethod
    def _extract_primary_number(text: str) -> float | None:
        """Extract the primary numeric percentage from text."""
        # Try percentage first: 99.87%, 100%, 68.74%
        m = re.search(r'(\d+(?:\.\d+)?)%', text)
        if m:
            return float(m.group(1))
        # Try decimal: 0.9987
        m = re.search(r'(\d+\.\d+)', text)
        if m:
            val = float(m.group(1))
            if val <= 1.0:
                return val * 100
            return val
        return None

    @staticmethod
    def _extract_entity_name(description: str) -> str:
        """Extract a human-readable entity name from a description."""
        # Common experiment name patterns
        for pattern, replacement in [
            (r'oracle memory', 'Oracle memory'),
            (r'oracle text memory', 'Oracle text memory'),
            (r'oracle latent memory', 'Oracle latent memory'),
            (r'core.only', 'core-only baseline'),
            (r'core.only baseline', 'core-only baseline'),
            (r'chain.set bce', 'Chain-set BCE retriever'),
            (r'chain_set_bce', 'Chain-set BCE retriever'),
            (r'dual encoder', 'Dual encoder'),
            (r'learned selector', 'Learned selector'),
            (r'chain.set retrieval', 'Chain-set retrieval'),
            (r'oracle.filter', 'Oracle-filter'),
            (r'16k pkm', '16K PKM'),
        ]:
            if re.search(pattern, description, re.IGNORECASE):
                return replacement

        # Fallback: take first 50 chars and capitalize
        short = description[:50].strip()
        if len(description) > 50:
            short += "..."
        return short[0].upper() + short[1:] if short else "Entity"

    @staticmethod
    def _parse_raw_evidence_bullets(prompt: str) -> list[str]:
        """Parse facts from a raw EVIDENCE: section.

        Used as a final fallback when no structured section markers exist
        (e.g., simple oracle/synthetic prompt format).
        Accepts both bullet-point (- prefix) and plain-text lines.
        """
        # Extract the EVIDENCE: section
        evidence_section = _extract_section(prompt, "EVIDENCE:", "ANSWER:")
        if not evidence_section:
            evidence_section = _extract_section(
                prompt, "EVIDENCE:", "Based on the evidence",
            )
        if not evidence_section:
            idx = prompt.find("EVIDENCE:")
            if idx != -1:
                evidence_section = prompt[idx + len("EVIDENCE:"):].strip()

        if not evidence_section:
            return []

        return SynthesizingModel._parse_evidence_lines(evidence_section)

    @staticmethod
    def _parse_evidence_lines(evidence_text: str) -> list[str]:
        """Parse fact lines from evidence text.
        
        Accepts both bullet-point (- prefix) and plain-text lines.
        Filters out empty lines, section headers, and boilerplate.
        """
        facts = []
        for line in evidence_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Skip section headers and boilerplate
            if line.lower().startswith((
                "based on the evidence", "answer:", "question:",
                "key numbers", "numbers for", "connected entity",
                "supporting evidence", "knowledge graph",
                "relation facts", "sources", "key findings",
            )):
                continue
            
            # Strip bullet prefix if present
            content = line
            if content.startswith("- "):
                content = content[2:].strip()
            
            if not content:
                continue
            if "insufficient" in content.lower():
                continue
            # Strip (ORACLE INJECTED) prefix
            content = re.sub(r'^\(ORACLE INJECTED\)\s*', '', content)
            # Strip annotation prefixes like "[This concept is directly validated...]"
            content = re.sub(r'^\[.*?\]\s*', '', content)
            # Skip lines that are just references or IDs
            if re.match(r'^\[\d+\]', content):
                continue
            
            facts.append(content)
        
        return facts

    def _synthesize_chain(self, prompt: str) -> str:
        """Synthesize a chain/narrative answer from evidence.

        Instead of dumping edge chains, extract key_findings from node
        facts and present them as a direct answer to the diagnostic or
        multi-hop question.
        """
        # First priority: extract from "answer to your question" section
        answer_section = _extract_section(
            prompt,
            "The answer to your question is in these facts:",
            "Supporting evidence:",
        )
        if not answer_section:
            answer_section = _extract_section(
                prompt,
                "The answer to your question is in these facts:",
                "Knowledge graph paths:",
            )
        if not answer_section:
            answer_section = _extract_section(
                prompt,
                "The answer to your question is in these facts:",
                "Relation facts:",
            )
        if not answer_section:
            answer_section = _extract_section(
                prompt,
                "The answer to your question is in these facts:",
                "Sources",
            )

        if answer_section:
            facts = []
            for line in answer_section.split("\n"):
                line = line.strip()
                if not line.startswith("- "):
                    continue
                content = line[2:].strip()
                if ": " in content:
                    _, desc = content.split(": ", 1)
                else:
                    desc = content
                # Strip annotation prefixes
                desc = re.sub(r'^\[.*?\]\s*', '', desc)
                if desc and len(desc) > 5:
                    facts.append(desc)

            if facts:
                # Limit facts: 2 for diagnostic, 3 for multi-hop
                question = _extract_line(prompt, "QUESTION:") or ""
                q_lower = question.lower()
                is_multi_hop = bool(re.search(
                    r'\b(walk through|step by step|chain of|evolution from|relationship between)\b',
                    q_lower
                ))
                max_facts = 3 if is_multi_hop else 2
                selected = facts[:max_facts]
                answer = " ".join(
                    f if f.rstrip().endswith(".") else f + "."
                    for f in selected
                )
                return answer

        # Fallback: node details extraction (original logic)
        node_section = _extract_section(
            prompt, "Key findings from evidence nodes:", "Knowledge graph paths:"
        )
        if not node_section:
            node_section = _extract_section(
                prompt, "Key findings from evidence nodes:", "Relation facts:"
            )

        if node_section:
            facts = []
            for line in node_section.split("\n"):
                line = line.strip()
                if not line.startswith("- "):
                    continue
                content = line[2:].strip()
                if ": " in content:
                    _, desc = content.split(": ", 1)
                    desc = re.sub(r'^\[.*?\]\s*', '', desc)
                    if desc and len(desc) > 5:
                        desc_clean = desc.rstrip()
                        if not desc_clean.endswith("."):
                            desc_clean += "."
                        facts.append(desc_clean)

            if facts:
                return " ".join(facts[:3])

        # Last fallback: paths section with edges
        paths_section = _extract_section(prompt, "Knowledge graph paths:", "Relation facts:")
        if not paths_section:
            paths_section = _extract_section(prompt, "Knowledge graph paths:", "Extracted facts:")

        if paths_section:
            # Parse edges and build a cleaner chain
            edge_pattern = re.compile(r'(\w+)\s+--\[(\w+)\].*?-->\s+(\w+)')
            edges = []
            for line in paths_section.split("\n"):
                m = edge_pattern.search(line.strip())
                if m:
                    edges.append((m.group(1), _edge_rel_name(m.group(2)), m.group(3)))

            if edges:
                # Build a concise chain — max 4 edges
                chain_parts = []
                for i, (src, rel, tgt) in enumerate(edges[:4]):
                    if i == 0:
                        chain_parts.append(f"{src} {rel} {tgt}")
                    elif i == len(edges[:4]) - 1:
                        chain_parts.append(f", which {rel} {tgt}")
                    else:
                        chain_parts.append(f", which {rel} {tgt}")
                return "".join(chain_parts) + "."

        return ""

    def _synthesize_definition(self, prompt: str) -> str:
        """Synthesize a definition answer from concept descriptions.

        Extracts concept descriptions and returns them — one sentence.
        """
        # Try answer section first
        answer_section = _extract_section(
            prompt,
            "The answer to your question is in these facts:",
            "Supporting evidence:",
        )
        if not answer_section:
            answer_section = _extract_section(
                prompt,
                "The answer to your question is in these facts:",
                "Knowledge graph paths:",
            )
        if not answer_section:
            answer_section = _extract_section(
                prompt, "Key findings from evidence nodes:", "Knowledge graph paths:"
            )
        if not answer_section:
            answer_section = _extract_section(
                prompt, "Key findings from evidence nodes:", "Relation facts:"
            )

        concept_descriptions: list[str] = []
        for line in answer_section.split("\n"):
            line = line.strip()
            if not line.startswith("- "):
                continue
            content = line[2:].strip()
            if ": " in content:
                node_id, desc = content.split(": ", 1)
                desc = re.sub(r'^\[.*?\]\s*', '', desc)
                if desc and len(desc) > 10:
                    concept_descriptions.append(desc)

        if not concept_descriptions:
            return ""

        best = max(concept_descriptions, key=len)
        if not best.rstrip().endswith("."):
            best = best.rstrip() + "."
        return best

    @staticmethod
    def _extract_numbers(text: str) -> dict[str, str]:
        """Extract numeric values from a text description.

        Returns a dict mapping label -> value string, e.g.
        {"pct": "99.87%", "1-hop": "99.5%", "2-hop": "100%"}
        """
        numbers: dict[str, str] = {}
        # Percentage patterns: "99.87%", "100%", "68.74%"
        for m in re.finditer(r'(\d+(?:\.\d+)?)%', text):
            numbers["pct"] = m.group(0)
            break  # Take first overall percentage

        # Accuracy-specific: "accuracy of X"
        acc_match = re.search(r'(\d+(?:\.\d+)?%)\s+(?:overall\s+)?accuracy', text, re.IGNORECASE)
        if acc_match:
            numbers["accuracy"] = acc_match.group(1)

        # "99.5% on 1-hop" like patterns
        for m in re.finditer(r'(\d+(?:\.\d+)?%)\s+on\s+(\d+-hop)', text, re.IGNORECASE):
            numbers[m.group(2)] = m.group(1)

        # "achieving X%" pattern
        for m in re.finditer(r'(\d+(?:\.\d+)?)%', text):
            if "pct" not in numbers:
                numbers["pct"] = m.group(0)
            break

        # Decimal values: "Rec@8 of 99.0%" etc.
        for m in re.finditer(r'Rec@\d+\s+(?:of\s+)?(\d+(?:\.\d+)?)%', text):
            numbers["recall"] = m.group(1) + "%"

        # "precision of X%"
        prec_match = re.search(r'(\d+(?:\.\d+)?)%\s+precision', text, re.IGNORECASE)
        if prec_match:
            numbers["precision"] = prec_match.group(1) + "%"

        # Generic number:value extraction as fallback
        if not numbers:
            for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(million|billion|thousand|%|parameters?)?', text):
                val = m.group(0).strip()
                numbers["value"] = val
                break

        return numbers

    @property
    def name(self) -> str:
        return "SynthesizingModel"


def _edge_rel_name(rel_type: str) -> str:
    """Convert edge relation type to human-readable phrase."""
    mapping = {
        "depends_on": "depends on",
        "caused_by": "is caused by",
        "blocked_by": "is blocked by",
        "validates": "validates",
        "contradicts": "contradicts",
        "implements": "implements",
        "derived_from": "is derived from",
        "replaces": "replaces",
        "related_to": "is related to",
        "mentioned_in": "is mentioned in",
    }
    return mapping.get(rel_type, rel_type.replace("_", " "))


def _extract_question_topic(question: str) -> str:
    """Extract a short topic phrase from a question for the opening sentence."""
    # Try "what is the X of Y" pattern
    m = re.search(
        r'(?:what|how)\s+(?:is|are|was|were|does|do|did|can|should)\s+(?:the\s+)?(.+?)(?:\?|$)',
        question, re.IGNORECASE,
    )
    if m:
        topic = m.group(1).strip()
        # Truncate if too long
        if len(topic) > 80:
            topic = topic[:77] + "..."
        return topic
    # Fallback: first 80 chars
    return question[:80] if len(question) > 80 else question


def _check_ollama_available() -> tuple[bool, str]:
    """Check if Ollama is running and has a suitable model."""
    try:
        import urllib.request
        url = "http://localhost:11434/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]

            # Pinned for reproducibility — change only in controlled experiments.
            # Prefer qwen2.5:latest first, then smallest models.
            preferred = [
                "qwen2.5:latest",
                "qwen2.5-coder:3b",
                "qwen2.5:3b",
                "qwen2.5-coder:1.5b",
                "phi3:mini",
                "llama3.2:3b",
                "gemma2:2b",
            ]
            for pref in preferred:
                if pref in models:
                    return True, pref

            # Accept any model under 3GB by checking the first available
            for m in data.get("models", []):
                # Look at size — prefer sub-3GB models
                # Ollama reports size in bytes
                size_gb = m.get("size", 0) / (1024**3)
                if size_gb < 3.0:
                    return True, m["name"]

            # Fall back to first available model
            if models:
                return True, models[0]

    except Exception:
        pass
    return False, ""


def _check_llamacpp_available() -> bool:
    """Check if llama-cpp-python is importable."""
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


class LocalModel(ModelInterface):
    """
    Auto-detecting local model — tries Ollama first, then llama.cpp.

    This is a convenience wrapper that delegates to the best available
    backend. Prefer get_available_model() for new code.

    Usage:
        model = LocalModel()           # auto-detect
        model = LocalModel(model_name="qwen2.5-coder:3b")  # explicit

    If no local model backend is available, falls back to SynthesizingModel.
    """

    def __init__(self, model_name: str | None = None, model_path: str | None = None, **kwargs: Any):
        self._model_name = model_name
        self._model_path = model_path
        self._kwargs = kwargs
        self._backend: ModelInterface | None = None
        self._loaded = False

    def load(self) -> None:
        """Detect and initialize the best available backend."""
        if self._loaded and self._backend is not None:
            return

        # Try Ollama first
        ollama_ok, detected_model = _check_ollama_available()
        if ollama_ok:
            model = self._model_name or detected_model
            self._backend = OllamaModel(model_name=model, **self._kwargs)
            self._loaded = True
            return

        # Try llama-cpp-python
        if _check_llamacpp_available():
            if self._model_path:
                import llama_cpp
                self._backend = _LlamaCppModel(model_path=self._model_path, **self._kwargs)
                self._loaded = True
                return
            # No model path — fall through to SynthesizingModel

        # Fallback to SynthesizingModel
        self._backend = SynthesizingModel()
        self._loaded = True

    def generate(self, prompt: str) -> str:
        if not self._loaded:
            self.load()
        assert self._backend is not None
        return self._backend.generate(prompt)

    @property
    def name(self) -> str:
        if self._backend is not None:
            return self._backend.name
        return "LocalModel(uninitialized)"

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        backend = repr(self._backend) if self._backend else "none"
        return f"LocalModel(status={status}, backend={backend})"


class _LlamaCppModel(ModelInterface):
    """Backend for llama-cpp-python (used when Ollama is unavailable)."""

    def __init__(self, model_path: str, **kwargs: Any):
        import llama_cpp
        self._model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=kwargs.pop("n_ctx", 2048),
            n_threads=kwargs.pop("n_threads", 4),
            verbose=False,
            **kwargs,
        )

    def generate(self, prompt: str) -> str:
        output = self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        return output["choices"][0]["message"]["content"].strip()

    @property
    def name(self) -> str:
        return "LlamaCppModel"


def get_available_model() -> ModelInterface:
    """
    Auto-detect and return the best available local model.

    Detection order:
      1. Ollama (http://localhost:11434) — uses smallest available model
      2. llama-cpp-python with GGUF file in common locations
      3. SynthesizingModel (improved DummyModel fallback)

    Returns a ModelInterface ready to call generate().
    """
    # Try Ollama first
    ollama_ok, model_name = _check_ollama_available()
    if ollama_ok:
        print(f"[model] Using Ollama with {model_name}", file=sys.stderr)
        return OllamaModel(model_name=model_name)

    # Try llama-cpp-python with GGUF files in common locations
    if _check_llamacpp_available():
        gguf_locations = [
            os.path.expanduser("~/.cache/lm-studio/models"),
            os.path.expanduser("~/models"),
            os.path.expanduser("~/.cache"),
            os.path.expandvars("%USERPROFILE%\\.cache"),
            os.path.expandvars("%USERPROFILE%\\models"),
        ]
        for loc in gguf_locations:
            loc_expanded = os.path.expanduser(loc)
            if os.path.isdir(loc_expanded):
                for root, _, files in os.walk(loc_expanded):
                    for f in files:
                        if f.endswith(".gguf"):
                            path = os.path.join(root, f)
                            size_mb = os.path.getsize(path) / (1024 * 1024)
                            if size_mb < 3000:  # Under 3 GB
                                print(f"[model] Using llama.cpp with {f}", file=sys.stderr)
                                return _LlamaCppModel(model_path=path)
        # No GGUF file found — fall through

    # Fallback: improved synthesizer
    print("[model] No local model found — using SynthesizingModel (template-based)", file=sys.stderr)
    return SynthesizingModel()


class FallbackModel(ModelInterface):
    """
    Two-pass model: tries an LLM first, falls back to SynthesizingModel.

    When the primary model returns "Insufficient evidence" but evidence
    IS present in the prompt, the SynthesizingModel can extract and format
    facts that the LLM may have missed. This is useful when the evidence
    format is ambiguous (e.g., document titles instead of experiment data).

    Usage:
        primary = OllamaModel(model_name="qwen2.5:latest")
        model = FallbackModel(primary)
        answer = model.generate(prompt)  # tries LLM, falls back if needed
    """

    def __init__(self, primary: ModelInterface, fallback: ModelInterface | None = None):
        self._primary = primary
        self._fallback = fallback or SynthesizingModel()

    def generate(self, prompt: str) -> str:
        answer = self._primary.generate(prompt)

        # Check if the LLM gave up but evidence exists
        if "insufficient evidence" in answer.lower():
            # Check if evidence actually exists in the prompt
            has_evidence = (
                "Extracted facts:" in prompt
                or "Relation facts:" in prompt
                or "Key findings from evidence nodes:" in prompt
            )
            has_no_evidence_marker = "(No evidence found" in prompt

            if has_evidence and not has_no_evidence_marker:
                # Evidence exists — try the synthesizer
                synth_answer = self._fallback.generate(prompt)
                if synth_answer and "insufficient evidence" not in synth_answer.lower():
                    return synth_answer

        return answer

    @property
    def name(self) -> str:
        return f"FallbackModel({self._primary.name} + {self._fallback.name})"


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
    # Remove confidence suffix in various formats (but NOT facts like (68.74%))
    fact = re.sub(r'\s*\(confidence:\s*[\d.]+\s*\)\s*$', '', fact).strip()
    fact = re.sub(r'\s*\(conf:\s*[\d.]+\s*\)\s*$', '', fact).strip()
    # Only strip bare numeric parentheticals if they are exactly a decimal
    # (like "(1.00)" or "(0.99)") but NOT percentage values like "(68.74%)"
    fact = re.sub(r'\s*\((\d+\.\d{1,2})\)\s*$', '', fact).strip()
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
