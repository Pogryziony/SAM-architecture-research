"""
NEXUS query understanding module.

Parses natural language questions into entities, intent, and constraints.
"""

from nexus.query.parser import (
    ParsedQuery,
    parse_question,
    detect_intent,
    spot_entities,
    find_entities_by_substring,
)

__all__ = [
    "ParsedQuery",
    "parse_question",
    "detect_intent",
    "spot_entities",
    "find_entities_by_substring",
]
