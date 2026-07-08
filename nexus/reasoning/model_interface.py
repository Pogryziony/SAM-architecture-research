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

    Uses the smallest available model by default (qwen2.5-coder:3b ~1.9 GB).
    All generation is synchronous — call generate() on each question.

    Parameters:
        model_name: Ollama model tag (default: "qwen2.5-coder:3b")
        host: Ollama API host (default: "http://localhost:11434")
        timeout: Request timeout in seconds (default: 120)
        system_prompt: Override the default system prompt (None = strip from prompt)
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:3b",
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
                    return "Insufficient evidence to answer."
            else:
                return "Insufficient evidence to answer."

        if not high_conf and not medium_conf and not low_conf:
            return "Insufficient evidence to answer."

        # Build answer as natural language paragraph
        paragraphs: list[str] = []

        # Opening sentence — connect to question
        question = _extract_line(prompt, "QUESTION:")
        if question:
            topic = _extract_question_topic(question)
            if topic:
                paragraphs.append(f'Regarding "{topic}", the evidence reveals the following:')
            else:
                paragraphs.append("Based on the available evidence:")
        else:
            paragraphs.append("Based on the available evidence:")

        # Synthesize high-confidence facts into paragraph form
        if high_conf:
            cleaned_facts = []
            for fact in high_conf[:6]:
                clean = _clean_fact_for_answer(fact)
                cleaned_facts.append(clean)

            # Determine if these are node details (no confidence suffixes, often contain ":")
            # or relation facts (with confidence suffixes stripped)
            is_node_facts = any(": " in f for f in cleaned_facts[:2])

            if is_node_facts:
                # Node details — present as direct findings
                for clean in cleaned_facts:
                    paragraphs.append(clean)
            else:
                # Relation facts — use varied templates
                import random
                rng = random.Random(hash(cleaned_facts[0]) if cleaned_facts else 42)

                if len(cleaned_facts) == 1:
                    template = rng.choice(self._HIGH_CONF_TEMPLATES)
                    paragraphs.append(template.format(fact=cleaned_facts[0]))
                else:
                    sentences = []
                    for i, clean in enumerate(cleaned_facts):
                        if i == 0:
                            sentences.append(f"The evidence clearly indicates that {clean}")
                        elif i == len(cleaned_facts) - 1:
                            sentences.append(f"Furthermore, {clean[0].lower() + clean[1:] if clean[0].isupper() else clean}")
                        else:
                            sentences.append(f"Additionally, {clean[0].lower() + clean[1:] if clean[0].isupper() else clean}")
                    paragraphs.append(" ".join(sentences))

        # Medium-confidence facts — qualifying language
        if medium_conf:
            med_facts = []
            for fact in medium_conf[:3]:
                clean = _clean_fact_for_answer(fact)
                med_facts.append(clean)

            import random
            rng = random.Random(hash("\n".join(med_facts)))

            if len(med_facts) == 1:
                template = rng.choice(self._MEDIUM_CONF_TEMPLATES)
                paragraphs.append(template.format(fact=med_facts[0]))
            else:
                sentences = [f"Additional evidence suggests the following:"]
                for clean in med_facts:
                    sentences.append(f"  - {clean}")
                paragraphs.append("\n".join(sentences))

        # Low-confidence facts — hedging language
        if low_conf:
            low_facts = []
            for fact in low_conf[:3]:
                clean = _clean_fact_for_answer(fact)
                low_facts.append(clean)

            if len(low_facts) == 1:
                paragraphs.append(f"There is some weak indication that {low_facts[0]}")
            else:
                sentences = ["Several lower-confidence signals were also observed:"]
                for clean in low_facts:
                    sentences.append(f"  - {clean}")
                paragraphs.append("\n".join(sentences))

        # Source citation footer
        sources_section = _extract_section(prompt, "Sources", "ANSWER:")
        if sources_section:
            source_count = sources_section.count("[")
            if source_count > 0:
                paragraphs.append(
                    f"\nThese findings are drawn from {source_count} source "
                    f"document(s) in the knowledge graph."
                )

        return "\n\n".join(paragraphs)

    @property
    def name(self) -> str:
        return "SynthesizingModel"


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

            # Prefer smallest models first
            preferred = [
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
        primary = OllamaModel(model_name="qwen2.5-coder:3b")
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
                if "insufficient evidence" not in synth_answer.lower():
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
    # Remove confidence suffix in various formats
    fact = re.sub(r'\s*\(confidence:\s*[\d.]+\s*\)\s*$', '', fact).strip()
    fact = re.sub(r'\s*\(conf:\s*[\d.]+\s*\)\s*$', '', fact).strip()
    fact = re.sub(r'\s*\([\d.]+\s*\)\s*$', '', fact).strip()
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
