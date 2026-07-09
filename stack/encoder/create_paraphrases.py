"""Create paraphrase_30.jsonl — 30 test questions rewritten with different wording, min 5 Polish."""
import json

test_path = r"C:\Users\Pogry\Projects\SAM-architecture-research\stack\encoder\data\test.jsonl"
test = [json.loads(l) for l in open(test_path, encoding="utf-8")]

# Map test question IDs to original questions for easy lookup
test_by_id = {q["id"]: q for q in test}

paraphrases = [
    # ── Factual questions (English) ──
    {
        "original_id": "q532",
        "original_question": "What is relation extraction?",
        "paraphrase": "Can you explain what relation extraction means?",
        "language": "en",
    },
    {
        "original_id": "q534",
        "original_question": "What is KuzuDB?",
        "paraphrase": "Tell me about KuzuDB — what is it?",
        "language": "en",
    },
    {
        "original_id": "q535",
        "original_question": "What is the oracle gap?",
        "paraphrase": "I'd like to understand the concept of the oracle gap — what does it refer to?",
        "language": "en",
    },
    {
        "original_id": "q536",
        "original_question": "What is a hard negative?",
        "paraphrase": "Define a hard negative for me.",
        "language": "en",
    },
    {
        "original_id": "q589",
        "original_question": "How many nodes are in the current NEXUS graph?",
        "paraphrase": "What's the total node count in the NEXUS graph right now?",
        "language": "en",
    },
    {
        "original_id": "q590",
        "original_question": "How many edges are in the current NEXUS graph?",
        "paraphrase": "How many connections exist between nodes in the NEXUS graph?",
        "language": "en",
    },
    # ── Diagnostic questions (English) ──
    {
        "original_id": "q538",
        "original_question": "What is the significance of the oracle memory experiment achieving 99.87% accuracy?",
        "paraphrase": "Why does it matter that oracle memory hit 99.87% accuracy?",
        "language": "en",
    },
    {
        "original_id": "q539",
        "original_question": "What is the significance of the chain-set BCE retriever achieving 100% all_required@32?",
        "paraphrase": "What makes the 100% all_required@32 result from the chain-set BCE retriever so important?",
        "language": "en",
    },
    {
        "original_id": "q540",
        "original_question": "What is the significance of the selector achieving only 50% precision?",
        "paraphrase": "The selector only managed 50% precision — what does that tell us?",
        "language": "en",
    },
    {
        "original_id": "q541",
        "original_question": "What is the significance of the noise tolerance experiment showing 91.6% at +8 distractors?",
        "paraphrase": "How should we interpret the noise tolerance result of 91.6% with 8 distractors?",
        "language": "en",
    },
    {
        "original_id": "q544",
        "original_question": "What is the significance of the pivot from SAM to NEXUS?",
        "paraphrase": "Why was the decision to move from SAM to NEXUS so meaningful?",
        "language": "en",
    },
    {
        "original_id": "q546",
        "original_question": "What is the significance of the NEXUS verifier being rule-based rather than LLM-based?",
        "paraphrase": "What difference does it make that NEXUS uses rules instead of an LLM for verification?",
        "language": "en",
    },
    {
        "original_id": "q549",
        "original_question": "What is the significance of the 3-hop collapse at +16 distractors?",
        "paraphrase": "Why is it important that 3-hop reasoning breaks down completely with 16 distractors?",
        "language": "en",
    },
    # ── Comparative questions (English) ──
    {
        "original_id": "q550",
        "original_question": "Compare the 3-hop accuracy across different SAM configurations: core_only (22.00%) vs oracle_memory (100%) vs controlled noise +8 (79.33%) vs +16 (39.00%). What does this tell us?",
        "paraphrase": "How do the 3-hop accuracy numbers compare — 22% core, 100% oracle, 79% at +8 noise, and 39% at +16? What's the takeaway?",
        "language": "en",
    },
    {
        "original_id": "q551",
        "original_question": "Compare the overall accuracy across all memory modes at experiment 0.6 across different SAM configurations: core_only=68.74%, random_memory=68.74%, retrieved_memory=68.74%, oracle_memory=99.87%, oracle_text=100%. What does this tell us?",
        "paraphrase": "Looking at the experiment 0.6 overall accuracy — core, random, and retrieved all at 68.74%, oracle at 99.87%, text at 100%. What does this pattern reveal?",
        "language": "en",
    },
    {
        "original_id": "q554",
        "original_question": "Compare the 3-hop accuracy under noise at +1, +2, +4, +8, +16 distractors across different SAM configurations: +1: 99.50%, +2: 98.17%, +4: 95.00%, +8: 79.33%, +16: 39.00%. What does this tell us?",
        "paraphrase": "Walk me through the 3-hop accuracy as noise increases — 99.5%, 98.2%, 95%, 79.3%, then 39% at 16 distractors. What does this trend mean?",
        "language": "en",
    },
    {
        "original_id": "q560",
        "original_question": "What SAM concepts map directly to NEXUS concepts?",
        "paraphrase": "Which ideas from SAM carry over directly to NEXUS without change?",
        "language": "en",
    },
    {
        "original_id": "q561",
        "original_question": "What SAM concepts have NO equivalent in NEXUS?",
        "paraphrase": "Are there SAM concepts that NEXUS simply doesn't use or replicate?",
        "language": "en",
    },
    {
        "original_id": "q562",
        "original_question": "What SAM experimental findings directly informed NEXUS design?",
        "paraphrase": "Which specific SAM experiment results shaped how NEXUS was built?",
        "language": "en",
    },
    # ── Diagnostic (English, more) ──
    {
        "original_id": "q584",
        "original_question": "What if SAM had been tested on real-world data from the start?",
        "paraphrase": "Suppose SAM was evaluated on real-world benchmarks from day one — how would things change?",
        "language": "en",
    },
    {
        "original_id": "q588",
        "original_question": "What if entity extraction turns out to be the bottleneck for NEXUS?",
        "paraphrase": "Imagine entity extraction becomes the limiting factor for NEXUS performance — what then?",
        "language": "en",
    },
    # ── Polish phrasings (at least 5) ──
    {
        "original_id": "q532",
        "original_question": "What is relation extraction?",
        "paraphrase": "Czym jest ekstrakcja relacji?",
        "language": "pl",
    },
    {
        "original_id": "q535",
        "original_question": "What is the oracle gap?",
        "paraphrase": "Co to jest luka oracle?",
        "language": "pl",
    },
    {
        "original_id": "q538",
        "original_question": "What is the significance of the oracle memory experiment achieving 99.87% accuracy?",
        "paraphrase": "Jakie znaczenie ma to, że eksperyment z pamięcią oracle osiągnął dokładność 99.87%?",
        "language": "pl",
    },
    {
        "original_id": "q544",
        "original_question": "What is the significance of the pivot from SAM to NEXUS?",
        "paraphrase": "Dlaczego przejście z SAM do NEXUS było tak istotne?",
        "language": "pl",
    },
    {
        "original_id": "q549",
        "original_question": "What is the significance of the 3-hop collapse at +16 distractors?",
        "paraphrase": "Jakie znaczenie ma załamanie się rozumowania 3-skokowego przy 16 dystraktorach?",
        "language": "pl",
    },
    {
        "original_id": "q550",
        "original_question": "Compare the 3-hop accuracy across different SAM configurations: core_only (22.00%) vs oracle_memory (100%) vs controlled noise +8 (79.33%) vs +16 (39.00%). What does this tell us?",
        "paraphrase": "Porównaj dokładność 3-skokową w różnych konfiguracjach SAM: core_only 22%, oracle 100%, szum kontrolowany +8 79.33%, +16 39%. Co z tego wynika?",
        "language": "pl",
    },
    {
        "original_id": "q589",
        "original_question": "How many nodes are in the current NEXUS graph?",
        "paraphrase": "Ile węzłów ma obecnie graf NEXUS?",
        "language": "pl",
    },
    # ── More English to reach 30 ──
    {
        "original_id": "q537",
        "original_question": "What is a controlled distractor?",
        "paraphrase": "Explain the concept of a controlled distractor — what is it exactly?",
        "language": "en",
    },
    {
        "original_id": "q555",
        "original_question": "How would you add a new experiment result to the NEXUS graph?",
        "paraphrase": "What's the procedure for inserting a new experiment result into the NEXUS graph?",
        "language": "en",
    },
]

print(f"Created {len(paraphrases)} paraphrases")
pl_count = sum(1 for p in paraphrases if p["language"] == "pl")
print(f"Polish: {pl_count}, English: {len(paraphrases) - pl_count}")

# Validate all originals exist in test set
for p in paraphrases:
    if p["original_id"] not in test_by_id:
        print(f"WARNING: {p['original_id']} not in test set!")
    else:
        orig = test_by_id[p["original_id"]]
        # Copy over GT data from original
        p["gt_entities"] = orig["entities"]
        p["question_type"] = orig["question_type"]
        p["category"] = orig["category"]
        p["intent"] = orig["intent"]
        p["difficulty"] = orig["difficulty"]

out_path = r"C:\Users\Pogry\Projects\SAM-architecture-research\benchmarks\qa-dataset\paraphrase_30.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for p in paraphrases:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Written to {out_path}")
