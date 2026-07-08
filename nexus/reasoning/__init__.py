"""
NEXUS reasoning module.

Builds evidence packs from graph paths, formats prompts, generates answers,
and verifies output against evidence.
"""

from nexus.reasoning.evidence_builder import (
    build_evidence,
    build_evidence_pack,
)
from nexus.reasoning.prompt_template import build_prompt
from nexus.reasoning.model_interface import (
    ModelInterface,
    DummyModel,
    LocalModel,
)
from nexus.reasoning.verifier import (
    VerificationResult,
    Verifier,
    extract_claims,
    extract_factual_claims,
    _is_factual_claim,
)
from nexus.reasoning.answer import answer_question

__all__ = [
    "build_evidence",
    "build_evidence_pack",
    "build_prompt",
    "ModelInterface",
    "DummyModel",
    "LocalModel",
    "VerificationResult",
    "Verifier",
    "extract_claims",
    "extract_factual_claims",
    "_is_factual_claim",
    "answer_question",
]
