# ADR-0001: Kuzu persistence scope for Phase 3

**Status:** Accepted  
**Date:** 2026-07-22

## Context

Phase 3 asks whether authoritative persistent storage (Kuzu) is an explicit
product requirement before investing in full backend parity.

## Decision

**Authoritative Kuzu persistence is not an explicit Phase 3 product requirement.**

- Keep Kuzu experimental.
- Preserve the unsupported claim that Kuzu is a production backend.
- Do not spend Phase 3 on a large backend rewrite.
- Revisit when product scope explicitly requires durable multi-process storage.

## Consequences

- Kuzu parity workstream marked `NOT_RUN` / deferred.
- In-memory graph remains the supported evaluation backend for Phase 3 evidence.
