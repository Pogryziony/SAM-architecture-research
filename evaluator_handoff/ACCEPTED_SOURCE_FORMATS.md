# Accepted source formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt`, `.md` | UTF-8 |
| JSON document | `.json` | Must include `id`, `text` |
| JSONL corpus | `.jsonl` | One object per line with `id`, `text` |
| HTML (stripped) | `.html` | Evaluator provides stripper hash |

Binary PDFs require an evaluator-declared extraction tool identity (name+version+hash).
