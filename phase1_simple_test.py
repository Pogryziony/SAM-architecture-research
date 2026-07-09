"""
Phase 1 Simplified: Evidence Impact Measurement

Since the oracle test showed evidence_recall = 0%, the real issue is that
_extract_key_facts() isn't matching facts in the evidence format.

Instead, we'll manually verify one question, inspect its evidence, and test
if a **manually crafted, obviously good evidence pack** improves accuracy.

This tells us: "Is evidence quality the bottleneck?" (yes/no)
"""

import json
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from benchmarks.run_benchmark import load_questions, compute_key_fact_score
from nexus.reasoning.model_interface import get_available_model, SynthesizingModel

# Load first question
dataset_path = Path(__file__).parent / "benchmarks" / "qa-dataset" / "questions.jsonl"
questions = load_questions(str(dataset_path), 1)
q = questions[0]

qtext = q["question"]
gt = q["answer"]

print("="*80)
print("PHASE 1: EVIDENCE IMPACT TEST")
print("="*80)
print()
print(f"Question: {qtext}")
print(f"Ground truth: {gt}")
print()

# Test 1: LLM with minimal evidence
model = get_available_model()
print("Test 1: Minimal evidence")
minimal_prompt = f"QUESTION: {qtext}\n\nEVIDENCE:\n(No evidence available)\n\nANSWER:"
minimal_answer = model.generate(minimal_prompt)
minimal_score = compute_key_fact_score(minimal_answer, gt)
print(f"Answer: {minimal_answer}")
print(f"Score: {minimal_score.get('fuzzy_accuracy', 0):.2%}")
print()

# Test 2: LLM with hand-crafted obvious evidence
print("Test 2: Hand-crafted OBVIOUS evidence containing the answer")
obvious_prompt = f"QUESTION: {qtext}\n\nEVIDENCE:\n{gt}\n\nANSWER:"
obvious_answer = model.generate(obvious_prompt)
obvious_score = compute_key_fact_score(obvious_answer, gt)
improvement = (obvious_score.get('fuzzy_accuracy', 0) - minimal_score.get('fuzzy_accuracy', 0))
print(f"Answer: {obvious_answer}")
print(f"Score: {obvious_score.get('fuzzy_accuracy', 0):.2%}")
print(f"Improvement: +{improvement:.2%}")
print()

if improvement > 0.1:
    print("CONCLUSION: Evidence IS valuable. Evidence quality is the bottleneck.")
    print("Decision: PROCEED to Phase 2 (Structured Metric Ingestion)")
else:
    print("CONCLUSION: Evidence has NO IMPACT. Generators cannot use evidence effectively.")
    print("Decision: PIVOT to generator fine-tuning (out of scope)")

print("="*80)
