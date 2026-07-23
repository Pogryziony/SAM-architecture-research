# Hidden question schema

JSONL, one object per line:

```json
{
  "id": "ext-001",
  "question": "...",
  "question_type": "factual|comparison|multi_hop|temporal|unanswerable|...",
  "gold_answer": "...",
  "should_abstain": false,
  "gold_entities": ["optional"],
  "relevant_doc_ids": ["optional"],
  "domain": "external-domain-id"
}
```

Gold fields are evaluator-private until adjudication.
