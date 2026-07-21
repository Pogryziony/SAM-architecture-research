# Branch history archive

**Cleanup date:** 2026-07-21  
**Policy:** Merged remote topic branches are deleted after their tip SHA and PR
identity are recorded here. Historical negative evidence stays in
`benchmarks/results/`, `STAGE*_NEGATIVE.md`, and related docs — branch deletion
does not erase that evidence.

GitHub PRs remain the durable record of discussion and CI. Tip SHAs below let
you recover a deleted branch tip with:

```bash
git fetch origin <tip_sha>
git switch -c recover/<name> <tip_sha>
```

---

## Recent branches (this cleanup request)

| Branch | Tip SHA | PR | Merged (UTC) | Notes |
|--------|---------|----|--------------|-------|
| `feature/answer-plan-transducer-and-comparison-improvements` | `f54f8b69fe024bcbba352e95d8a0b2e77c39fff2` | [#32](https://github.com/Pogryziony/SAM-architecture-research/pull/32) | 2026-07-18 | Copy/edit transducer + comparison Realizer improvements; was the open working branch before stack-v1 execution |
| `feat/execute-next-nexus-plan` | `113735f7966ac04a04d06088b4220993b83e684b` | [#33](https://github.com/Pogryziony/SAM-architecture-research/pull/33) | 2026-07-21 | Stack-v1 freeze docs, architecture registry, public API/CLI, traversal budgets, canonical graph hash; remote deleted on merge |

---

## Remote branches deleted in the 2026-07-21 cleanup

All were already merged into `master` (`git branch -r --merged origin/master`).

| Branch | Tip SHA | PR | Title / outcome |
|--------|---------|----|-----------------|
| `feature/answer-plan-transducer-and-comparison-improvements` | `f54f8b69fe024bcbba352e95d8a0b2e77c39fff2` | [#32](https://github.com/Pogryziony/SAM-architecture-research/pull/32) | Non-AR copy/edit transducer + comparison improvements |
| `feature/realizer-v2-pilot-rejected` | `ea5a303d0affa8c253cef12aeb75ad9f7654c6ee` | [#25](https://github.com/Pogryziony/SAM-architecture-research/pull/25) | **NEGATIVE** — neural v2 checkpoint rejected (0% grounding) |
| `feature/realizer-pilot-fail` | `80217619b0ec7615a2643e1e6790770b367613c0` | [#23](https://github.com/Pogryziony/SAM-architecture-research/pull/23) | **NEGATIVE** — Realizer v1 mode collapse |
| `feature/phase4-training-readiness` | `3e318fcf6f44f33864fc5c2929cd834657da8115` | [#22](https://github.com/Pogryziony/SAM-architecture-research/pull/22) | Phase 4 training readiness |
| `fix/realizer-answer-quality` | `d761a27cadd6669adf216e19721b1e0ec1c5fbff` | [#24](https://github.com/Pogryziony/SAM-architecture-research/pull/24) | Realizer quality recovery |
| `fix/nexus-pilot-integrity` | `1a94eedec02f0a829a07a58182afaa0de9d6bbde` | [#21](https://github.com/Pogryziony/SAM-architecture-research/pull/21) | Pilot integrity / training gates |
| `fix/t1-explicit-weights` | `97a537de600dd5d4c30e1f9cb796cc6adabd6067` | (superseded by [#15](https://github.com/Pogryziony/SAM-architecture-research/pull/15)) | T1 weights; tip merged via later fix chain |
| `agent/realizer-answer-plan-v1` | `01861032d43d33da90502a04b7607378fb230226` | [#31](https://github.com/Pogryziony/SAM-architecture-research/pull/31) | AnswerPlan fail-closed gates |
| `agent/realizer-corpus-v2` | `0b084702189cfa9b979e62e34957f7c134cc2c56` | [#30](https://github.com/Pogryziony/SAM-architecture-research/pull/30) | PL/EN Realizer corpus v2 |
| `agent/integrate-abstractive-realizer-v3` | `4b71ea67800874041c1773db9cb782b33a7903ed` | [#29](https://github.com/Pogryziony/SAM-architecture-research/pull/29) | Comparison-plan runtime integration |
| `agent/accept-constrained-realizer-pilot` | `6f910a293370b9b96544f6d9148e6e743cbb7ba2` | [#28](https://github.com/Pogryziony/SAM-architecture-research/pull/28) | Comparison-plan pilot accepted |
| `agent/prepare-abstractive-realizer-pilot` | `91a6b53f81b1e6a2bd47a513b80aae2dfc618079` | [#27](https://github.com/Pogryziony/SAM-architecture-research/pull/27) | Bounded abstractive pilot prep |
| `agent/pointer-copy-realizer-v3` | `d06ac0b12a97a0573da9750e2d6dfabe8ff47a10` | [#26](https://github.com/Pogryziony/SAM-architecture-research/pull/26) | Pointer/Copy v3 accepted |
| `agent/nexus-unique-train-data-relevance` | `6fe2f350bd35b7f764ea8cec2ef2f91bcb3942ea` | [#11](https://github.com/Pogryziony/SAM-architecture-research/pull/11) | Unique Realizer train data / relevance |
| `agent/nexus-auditability-foundation` | `fcb480803d31f390ca5f05d3cb820675acf078ae` | [#9](https://github.com/Pogryziony/SAM-architecture-research/pull/9) | Reasoning audit + oracle evaluation |
| `research-docs` | `ba5a11140c8c6433eb8796654bd2adfc421acb7f` | (merged tip; early docs) | Early research/docs tip already on master ancestry |

### Already pruned before this cleanup (merged; removed by `git fetch --prune`)

These remotes were gone after prune on 2026-07-21; PR numbers retained for history:

| Branch | PR |
|--------|----|
| `feat/execute-next-nexus-plan` | [#33](https://github.com/Pogryziony/SAM-architecture-research/pull/33) |
| `feature/training-presets` | [#20](https://github.com/Pogryziony/SAM-architecture-research/pull/20) |
| `fix/candidate-pool-size` | [#19](https://github.com/Pogryziony/SAM-architecture-research/pull/19) |
| `fix/nexus-final-cleanup` | [#18](https://github.com/Pogryziony/SAM-architecture-research/pull/18) |
| `fix/nexus-stage3-deps` | [#17](https://github.com/Pogryziony/SAM-architecture-research/pull/17) |
| `fix/nexus-pilot-readiness` | [#16](https://github.com/Pogryziony/SAM-architecture-research/pull/16) |
| `fix/t1-explicit-weights-v2` | [#15](https://github.com/Pogryziony/SAM-architecture-research/pull/15) |
| `fix/t1-flaky-test` | [#14](https://github.com/Pogryziony/SAM-architecture-research/pull/14) |
| `fix/nexus-audit-phases-1-3` | [#13](https://github.com/Pogryziony/SAM-architecture-research/pull/13) |
| `fix/stage2-quality-regression` | [#12](https://github.com/Pogryziony/SAM-architecture-research/pull/12) |
| `fix/critical-runner-bugs` | [#8](https://github.com/Pogryziony/SAM-architecture-research/pull/8) |
| `fix/entity-ranker-v3-reproducibility` | [#2](https://github.com/Pogryziony/SAM-architecture-research/pull/2) |
| `stage-d-infrastructure` | [#7](https://github.com/Pogryziony/SAM-architecture-research/pull/7) |
| `stage-c-evidence-quality` | [#6](https://github.com/Pogryziony/SAM-architecture-research/pull/6) |
| `stage-b-baseline` | [#5](https://github.com/Pogryziony/SAM-architecture-research/pull/5) |
| `stage-a-canonical-pipeline` | [#4](https://github.com/Pogryziony/SAM-architecture-research/pull/4) |

---

## Branches retained

| Branch | Tip SHA | Reason |
|--------|---------|--------|
| `master` | (moving) | Default branch |
| `research/nexus-architecture-docs` | `89ec92cfb15084710727be61c0995573e1da7ac7` | **Not merged** into `master` as of cleanup (`git branch -r --no-merged`). Retained until explicitly merged or retired. Contains early NEXUS research transition docs tip. |

---

## Negative / rejected work (branch names are historical markers)

Deleting these branches does **not** retract the scientific outcomes. See:

- `STAGE1B_NEGATIVE.md`, `STAGE1C_NEGATIVE.md`, `STAGE2_NEGATIVE.md`
- `PROTOCOL_VIOLATIONS.md`
- `training/REJECTED_ARCHITECTURES.json`
- `docs/stack-v1-freeze.md`
- Pilot reports under `benchmarks/results/realizer/`

| Outcome | Branch (deleted) | PR |
|---------|------------------|----|
| Realizer v1 mode collapse | `feature/realizer-pilot-fail` | #23 |
| Realizer v2 0% grounding | `feature/realizer-v2-pilot-rejected` | #25 |
