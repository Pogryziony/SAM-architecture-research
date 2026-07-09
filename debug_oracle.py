import json

with open('benchmarks/results/oracle_evidence_test.json') as f:
    data = json.load(f)

# Look at first 3 questions
for i, q in enumerate(data['questions'][:3], 1):
    print(f"\n=== Question {i} ===")
    print(f"Q: {q['question'][:70]}")
    print(f"GT: {q['ground_truth'][:100]}")
    print(f"Evidence recall: {q.get('evidence_recall', 'N/A')}")
    print(f"Baseline accuracy: {q.get('baseline_accuracy')}")
    print(f"Baseline answer: {q.get('baseline_answer', 'N/A')[:150]}")
