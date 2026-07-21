"""Structured provenance records with compatibility adapters.

Free-form source strings remain accepted. New ingestion should prefer
``SourceRecord`` identities so unconditional answers can require coverage.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any


_LOCATOR_RE = re.compile(
    r"^(?P<path>[^:#]+)(?::(?P<line>\d+)(?:-(?P<end>\d+))?)?(?:#(?P<fragment>.+))?$"
)


@dataclass(frozen=True)
class SourceRecord:
    """Stable source identity attached to nodes/edges."""

    source_id: str
    locator: str = ""
    content_hash: str = ""
    observed_at: str = ""
    extraction_method: str = "legacy_freeform"
    reliability: float = 0.5
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_id_for(locator: str, content: str = "") -> str:
    """Deterministic source ID from locator + optional content."""
    payload = f"{locator.strip()}|{content}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"src_{digest}"


def parse_freeform_source(raw: str) -> SourceRecord:
    """Adapt a legacy free-form source string into a ``SourceRecord``."""
    text = (raw or "").strip()
    if not text:
        return SourceRecord(
            source_id=source_id_for(""),
            locator="",
            extraction_method="empty",
            reliability=0.0,
            raw="",
        )
    match = _LOCATOR_RE.match(text)
    locator = match.group("path") if match else text
    return SourceRecord(
        source_id=source_id_for(text),
        locator=locator,
        extraction_method="legacy_freeform",
        reliability=0.5,
        raw=text,
    )


def normalize_sources(sources: list[str] | list[SourceRecord] | None) -> list[SourceRecord]:
    """Normalize mixed free-form / structured sources."""
    if not sources:
        return []
    out: list[SourceRecord] = []
    seen: set[str] = set()
    for item in sources:
        record = item if isinstance(item, SourceRecord) else parse_freeform_source(str(item))
        if record.source_id in seen:
            continue
        seen.add(record.source_id)
        out.append(record)
    return out


def provenance_coverage(records: list[SourceRecord]) -> float:
    """Fraction of records that have a non-empty locator."""
    if not records:
        return 0.0
    present = sum(1 for record in records if record.locator)
    return present / len(records)


__all__ = [
    "SourceRecord",
    "normalize_sources",
    "parse_freeform_source",
    "provenance_coverage",
    "source_id_for",
]
