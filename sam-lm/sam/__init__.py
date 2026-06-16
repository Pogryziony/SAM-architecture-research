"""SAM — Sparse Associative Memory Language Model (research POC).

This package contains the smallest falsifiable experiment for the SAM thesis:
decouple knowledge (sparse product-key memory) from computation (small dense core),
and test whether that helps multi-hop reasoning over external facts versus a
same-size dense Transformer.

Nothing here is optimized for production. See docs/ for the research framing.
"""

__version__ = "0.0.1"
