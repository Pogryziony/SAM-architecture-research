"""
NEXUS — Non-Parametric Execution and Understanding System.

Graph-first knowledge store with traversal-based reasoning.
The LLM is a language interface, not the knowledge store.

Public entry points: ``nexus.api`` (``ask``, CLI) and
``nexus.pipeline.config.ProductionNEXUSConfig``. Prefer
``ProductionNEXUSConfig.grounded()`` for production QA; the library Realizer
default remains ``synth`` for Stage 2 compatibility.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
