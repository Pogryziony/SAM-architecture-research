# Experiment Index

Quick-reference index of all experiment reports in the repository.

Reports are in `sam-lm/experiments/`. Each experiment builds on previous
findings and tests a specific hypothesis.

| Experiment | Report file | Question | Key result |
|-----------|------------|----------|-----------|
| **0 (Diagnosis)** | `diagnosis_report.md` | Is the pipeline working? | Found and fixed 3 critical bugs. Retrieval is bottleneck. |
| **0.2** | `experiment_0_2_report.md` | Can compact PKM retrieval work? | 16K PKM: 25.8% Rec@8. Oracle text: 100% — core CAN use memory. |
| **0.3** | `experiment_0_3_report.md` | Is PKM candidate generation the bottleneck? | Candidate gen: SOLVED (100%). Ranking generalizes poorly (29% val). |
| **0.5** | `experiment_0_5_report.md` | Can retrieval work with better data? | Yes: dense dataset with 21.8 ex/slot → 99.0% Rec@8. Gate 1 PASSED. |
| **0.6** | `experiment_0_6_final_report.md` | Full validation — does SAM work end-to-end? | Oracle memory: 99.87%. Retrieved memory: = core_only (68.74%). Query projection mismatch. |
| **0.7** | `experiment_0_7_report.md` | Can external-text-query fix the mismatch? | External text query bypasses hidden-state projection. Tested topK sweep. |
| **0.8** | `experiment_0_8_report.md` | Aggregation and selection variants? | Tested weighted, threshold, softmax-mass, score-gap selection. |
| **0.9** | `experiment_0_9_report.md` | Oracle-filter and multi-query? | Oracle filter achieves 79.95%. Multi-query implemented but not yet effective. |
| **0.10** | `experiment_0_10_report.md` | Required-set retrieval — where are slots missing? | all_required@64 = 27%. Dual encoder misses intermediate chain slots. Not a ranking problem. |
| **0.11** | `experiment_0_11_report.md` | Can chain-aware retrieval solve the multi-hop bottleneck? | Yes! Chain-set BCE: all_required@32 = 100%. But SAM still = core_only. |
| **0.12** | `experiment_0_12_report.md` | Can we select the right slots from chain retrieval? | Oracle-filter: 100%. Learned selector: recall 96.6%, precision 50%. Selector is bottleneck. |
| **0.13A** | `experiment_0_13A_noisy_memory_report.md` | How much memory noise can SAM tolerate? | Tolerates +8 random distractors (91.6%). 3-hop collapses at +16 (39%). Gate is NOT the bottleneck. |
| **0.13B** | *(planned)* | Are realistic retrieval distractors harder than random? | Testing in progress. |

## Reading order for understanding the research arc

1. **Diagnosis report (0)** — establishes the pipeline and initial problems
2. **Experiment 0.5** — the dataset fix that made retrieval possible
3. **Experiment 0.6** — the critical validation that oracle memory works but retrieval doesn't
4. **Experiment 0.10** — discovering that multi-hop required slots are missing from retrieval
5. **Experiment 0.11** — solving the multi-hop retrieval bottleneck with chain-set BCE
6. **Experiment 0.12** — discovering that slot selection (not retrieval) is the new bottleneck
7. **Experiment 0.13A** — proving SAM tolerates controlled memory noise, shifting the diagnosis to selector quality

---

*Last updated: 2026-06-18*
