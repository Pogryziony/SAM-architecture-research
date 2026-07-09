"""
Dialogue state module — temporary activation subgraph with recency decay.

Stage 3 of the SAM+NEXUS stack. Entities activated in prior turns get
boosted resolution priority. Activation decays exponentially with each turn.
"""

from stack.dialogue.state import DialogueState

__all__ = ["DialogueState"]
