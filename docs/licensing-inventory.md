# Licensing and governance inventory

**Status:** Inventory only — not legal advice.  
**Questions requiring legal review are marked REVIEW.**

## Repository license

| Item | Evidence |
|------|----------|
| License file | `LICENSE` |
| Terms | Proprietary / all rights reserved (copyright 2025–2026 Pogryziony) |
| Open source? | No — explicitly “Not open source.” |

**REVIEW:** redistribution of prepared datasets, model weights, and evaluation
artifacts derived from third-party corpora under this proprietary wrapper.

## Third-party / prepared data (known touchpoints)

| Asset | Location / reference | Notes | Legal question |
|-------|----------------------|-------|----------------|
| PoQuAD / realizer corpus v2 | `docs/realizer-corpus-v2.md` | Pinned public sources claimed in docs | **REVIEW** share-alike / attribution for redistributed prepared splits |
| Oracle QA (`oracle_v1`) | `benchmarks/qa-dataset/oracle_v1.jsonl` | Curated internal questions over project docs | Internal use OK under repo license; **REVIEW** if publishing externally |
| Experiment markdown corpus | `sam-lm/experiments/`, docs | Project-authored | Covered by repo license |
| Mini domain pack | `nexus/domain/mini_pack.py` | Synthetic tiny graph | Repo license |

## Model weights

| Asset | Location | Notes | Legal question |
|-------|----------|-------|----------------|
| ER3 checkpoint | `models/encoder/entity_ranker_v3_20260711T081545Z/` | Trained in-repo; manifest SHA checked | **REVIEW** if redistributing weights alone |
| Comparison-plan pilot | `models/realizer/abstractive_v1_plan_v3/` | Pilot checkpoint | Same |
| Optional sentence-transformers | runtime optional dep | Third-party model cards apply | **REVIEW** weight caching/redistribution |
| External LLM APIs | via env credentials | Outputs may be retained in artifacts | **REVIEW** API ToS + retention |

## Attribution obligations

- Document any public dataset used in prepared corpora with source URL, license, and SHA.
- Keep `docs/realizer-corpus-v2.md` and dataset manifests as the attribution ledger.

## Governance roadmap hooks

Issue-ready items live in [`REMAINING_WORK_BACKLOG.md`](REMAINING_WORK_BACKLOG.md)
under “Governance / licensing.”
