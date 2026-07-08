"""
Cost model for NEXUS benchmarking.

Provides pricing for frontier API models and local inference,
plus utility functions to estimate cost per question and per 1K questions.
"""

from __future__ import annotations

# ── Pricing per 1M tokens (in USD) ──

FRONTIER_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini":    {"input": 0.15, "output": 0.60},
    "gpt-4o":         {"input": 2.50, "output": 10.00},
    "gpt-4-turbo":    {"input": 10.00, "output": 30.00},
    "claude-haiku":   {"input": 0.25, "output": 1.25},
    "claude-sonnet":  {"input": 3.00, "output": 15.00},
    "claude-opus":    {"input": 15.00, "output": 75.00},
    "gemini-flash":   {"input": 0.075, "output": 0.30},
    "gemini-pro":     {"input": 1.25, "output": 5.00},
}

# Marginal cost for local inference (electricity is negligible at this scale)
LOCAL_COST: float = 0.0


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model_backend: str,
    local: bool = False,
) -> float:
    """
    Estimate USD cost for a single inference call.

    Args:
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens
        model_backend: Model name (e.g., "gpt-4o-mini", "claude-haiku")
        local: If True, cost is $0.00 regardless of backend

    Returns:
        Estimated cost in USD
    """
    if local:
        return LOCAL_COST

    pricing = FRONTIER_PRICING.get(model_backend)
    if pricing is None:
        # Unknown model — return 0.0 rather than guessing
        return 0.0

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def estimate_cost_per_1k(
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
    model_backend: str,
    local: bool = False,
) -> float:
    """
    Estimate USD cost for 1,000 inference calls.

    Args:
        avg_prompt_tokens: Average prompt tokens per call
        avg_completion_tokens: Average completion tokens per call
        model_backend: Model name
        local: If True, cost is $0.00

    Returns:
        Estimated cost per 1K questions in USD
    """
    if local:
        return LOCAL_COST

    pricing = FRONTIER_PRICING.get(model_backend)
    if pricing is None:
        return 0.0

    total_input_tokens = avg_prompt_tokens * 1000
    total_output_tokens = avg_completion_tokens * 1000
    input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 4)


def format_cost_comparison(
    system_name: str,
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
    local: bool = False,
) -> list[str]:
    """
    Build a human-readable cost comparison table for a system.

    Returns a list of formatted strings.
    """
    lines: list[str] = []
    models_to_compare = ["gpt-4o-mini", "claude-haiku", "gemini-flash"]

    for model in models_to_compare:
        cost_1k = estimate_cost_per_1k(
            avg_prompt_tokens, avg_completion_tokens, model, local=local,
        )
        cost_1 = estimate_cost(
            int(avg_prompt_tokens), int(avg_completion_tokens), model, local=local,
        )

        if local:
            label = f"  {system_name} + local (any model):  $0.00"
        else:
            label = f"  {system_name} + {model}:  ${cost_1k:.2f}/1K (${cost_1:.4f}/q)"
        lines.append(label)

    return lines
