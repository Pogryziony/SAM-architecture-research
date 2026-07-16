"""CPU-first evidence-to-answer model components for NEXUS."""

from .tokenizer import ByteTokenizer
from .grounded import GroundedRealization, realize_grounded
from .pointer_copy import PointerCopyResult, realize_pointer_copy

__all__ = [
    "ByteTokenizer", "GroundedRealization", "PointerCopyResult",
    "realize_grounded", "realize_pointer_copy",
]
