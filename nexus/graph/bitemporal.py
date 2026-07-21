"""Stage 6 bi-temporal helpers (partial).

Supports optional valid_from/valid_to and observed_at/retracted_at on fact
dicts or edge properties. Look-ahead queries that use knowledge observed
after the as_known_at cutoff are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


@dataclass(frozen=True)
class BiTemporalStamp:
    valid_from: str = ""
    valid_to: str = ""
    observed_at: str = ""
    retracted_at: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "BiTemporalStamp":
        payload = data or {}
        return cls(
            valid_from=str(payload.get("valid_from", "") or ""),
            valid_to=str(payload.get("valid_to", "") or ""),
            observed_at=str(payload.get("observed_at", "") or ""),
            retracted_at=str(payload.get("retracted_at", "") or ""),
        )


def is_valid_at(stamp: BiTemporalStamp, as_valid_at: str) -> bool:
    """Return True when the fact's validity window covers as_valid_at."""
    point = _parse_ts(as_valid_at)
    if point is None:
        return True
    start = _parse_ts(stamp.valid_from)
    end = _parse_ts(stamp.valid_to)
    if start is not None and point < start:
        return False
    if end is not None and point >= end:
        return False
    return True


def is_known_at(stamp: BiTemporalStamp, as_known_at: str) -> bool:
    """Return True when the fact was observed by as_known_at and not retracted."""
    point = _parse_ts(as_known_at)
    if point is None:
        return True
    observed = _parse_ts(stamp.observed_at)
    if observed is not None and observed > point:
        return False
    retracted = _parse_ts(stamp.retracted_at)
    if retracted is not None and retracted <= point:
        return False
    return True


def filter_facts_bitemporal(
    facts: list[Mapping[str, Any]],
    *,
    as_valid_at: str = "",
    as_known_at: str = "",
) -> list[Mapping[str, Any]]:
    """Filter facts by valid-time and known-time cutoffs."""
    kept: list[Mapping[str, Any]] = []
    for fact in facts:
        stamp = BiTemporalStamp.from_mapping(fact)
        if as_valid_at and not is_valid_at(stamp, as_valid_at):
            continue
        if as_known_at and not is_known_at(stamp, as_known_at):
            continue
        kept.append(fact)
    return kept


def assert_no_lookahead(
    facts: list[Mapping[str, Any]],
    *,
    as_known_at: str,
) -> list[str]:
    """Return errors for facts observed after the known-at cutoff."""
    errors: list[str] = []
    cutoff = _parse_ts(as_known_at)
    if cutoff is None:
        return errors
    for fact in facts:
        stamp = BiTemporalStamp.from_mapping(fact)
        observed = _parse_ts(stamp.observed_at)
        if observed is not None and observed > cutoff:
            errors.append(
                f"look-ahead fact {fact.get('source')}->{fact.get('target')} "
                f"observed_at={stamp.observed_at} after as_known_at={as_known_at}"
            )
    return errors


# Stable epoch for production ingest when source documents lack dates.
# Must remain constant so canonical graph content hashes stay reproducible.
DEFAULT_INGEST_EPOCH = "2026-07-08T00:00:00+00:00"


def stable_ingest_stamp(
    *,
    observed_at: str = "",
    valid_from: str = "",
    valid_to: str = "",
    retracted_at: str = "",
    fallback_observed_at: str = DEFAULT_INGEST_EPOCH,
) -> dict[str, str]:
    """Return bi-temporal Edge kwargs with deterministic fallback stamps.

    Prefer explicit provenance dates when available; otherwise use the fixed
    ingest epoch (never ``datetime.now``).
    """
    observed = str(observed_at or "").strip() or fallback_observed_at
    start = str(valid_from or "").strip() or observed
    return {
        "observed_at": observed,
        "valid_from": start,
        "valid_to": str(valid_to or "").strip(),
        "retracted_at": str(retracted_at or "").strip(),
    }


def edge_to_fact(edge: Any) -> dict[str, Any]:
    """Project an Edge (or edge-like object) into a bi-temporal fact dict."""
    if hasattr(edge, "bitemporal_stamp"):
        return dict(edge.bitemporal_stamp())
    return {
        "source": str(getattr(edge, "source", "")),
        "relation": str(getattr(edge, "type", getattr(edge, "relation", ""))),
        "target": str(getattr(edge, "target", "")),
        "valid_from": str(getattr(edge, "valid_from", "") or ""),
        "valid_to": str(getattr(edge, "valid_to", "") or ""),
        "observed_at": str(getattr(edge, "observed_at", "") or ""),
        "retracted_at": str(getattr(edge, "retracted_at", "") or ""),
    }


def filter_edges_bitemporal(
    edges: list[Any],
    *,
    as_valid_at: str = "",
    as_known_at: str = "",
) -> list[Any]:
    """Keep edges whose bi-temporal stamps pass the cutoffs."""
    kept: list[Any] = []
    for edge in edges:
        fact = edge_to_fact(edge)
        stamp = BiTemporalStamp.from_mapping(fact)
        if as_valid_at and not is_valid_at(stamp, as_valid_at):
            continue
        if as_known_at and not is_known_at(stamp, as_known_at):
            continue
        kept.append(edge)
    return kept
