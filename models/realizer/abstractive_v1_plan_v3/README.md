# NEXUS Realizer comparison-plan pilot checkpoint

This is the accepted three-epoch CPU pilot checkpoint for the constrained
comparison-plan Realizer. It is not evidence that the neural model performs
comparison reasoning. NEXUS computes and verifies `SAME` or `DIFFERENT` from
immutable evidence; the model must follow that plan, and constrained decoding
prevents malformed control output. Exact sources, subjects and values are
materialized outside the model.

The full 356-record validation split passes at 100% materialized exact match,
100% relation-plan adherence for both classes, 100% slot preservation and 0%
hallucination. A plan contradicting its evidence values fails closed before
model inference.

Load only after verifying `manifest.json`, the configured dataset and the
SHA-256 of `model.pt`. The pilot authorizes integration testing and makes the
repository ready for an explicitly requested full-training run. No such full
run was launched.
