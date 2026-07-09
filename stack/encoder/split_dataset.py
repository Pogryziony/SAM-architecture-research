"""Split the 750 questions into train/val/test by deterministic ID sort."""
import json
import os

lines = open(
    r"C:\Users\Pogry\Projects\SAM-architecture-research\benchmarks\qa-dataset\questions.jsonl",
    encoding="utf-8",
).readlines()
questions = [json.loads(l) for l in lines]

# Sort deterministically by ID
questions.sort(key=lambda d: d["id"])

# Map question_type to intent and category
intent_map = {
    "factual": "factual_lookup",
    "comparative": "comparison",
    "multi-hop": "multi_hop",
    "diagnostic": "diagnostic",
}

for q in questions:
    q["intent"] = intent_map[q["question_type"]]
    q["category"] = q["question_type"]

total = len(questions)
n_train = int(total * 0.50)
n_val = int(total * 0.20)
n_test = total - n_train - n_val

train = questions[:n_train]
val = questions[n_train : n_train + n_val]
test = questions[n_train + n_val :]

outdir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(outdir, exist_ok=True)

for name, data in [("train", train), ("val", val), ("test", test)]:
    path = os.path.join(outdir, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{name}: {len(data)} questions -> {path}")

print(f"\nSplit: train={n_train}, val={n_val}, test={n_test}")
print(f"First train ID: {train[0]['id']}")
print(f"Last train ID: {train[-1]['id']}")
print(f"First test ID: {test[0]['id']}")
print(f"Last test ID: {test[-1]['id']}")
