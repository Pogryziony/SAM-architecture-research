"""CPU-first evidence-to-answer model components for NEXUS."""

from .tokenizer import ByteTokenizer
from .grounded import GroundedRealization, realize_grounded
from .pointer_copy import PointerCopyResult, realize_pointer_copy
from .answer_plan import compile_answer_plan, validate_answer_plan
from .plan_serializer import serialize_answer_plan, serialize_answer_plan_for_model
from .subword_tokenizer import TrainOnlySubwordTokenizer
from .edit_script import compute_edit_target, apply_edit_target, edit_accuracy, tokenize
from .copy_edit_transducer import build_copy_edit_transducer

__all__ = [
    "ByteTokenizer", "GroundedRealization", "PointerCopyResult",
    "realize_grounded", "realize_pointer_copy",
    "TrainOnlySubwordTokenizer", "compile_answer_plan", "serialize_answer_plan",
    "serialize_answer_plan_for_model",
    "validate_answer_plan",
    "compute_edit_target", "apply_edit_target", "edit_accuracy", "tokenize",
    "build_copy_edit_transducer",
]
