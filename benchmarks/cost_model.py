"""
Cost model for NEXUS benchmarking — local-only pricing.

The target is $0.01/1M tokens. Frontier APIs are dead at this price point;
everything runs locally. Cost is computed from electricity consumption:
CPU TDP (watts) × time × electricity rate.

All throughput numbers are read EXCLUSIVELY from the newest benchmark results
JSON. No hardcoded throughput values exist anywhere in this file.

Provides:
  - LocalCostModel: dataclass for local inference cost estimation,
    with classmethods to auto-load from throughput results
  - BlendedRouterCost: computes effective cost for NEXUS router (80% synth = $0)
  - Utility functions for cost comparison and formatting
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# ── Frontier pricing retained for historical reference only ──
# These are NOT used in the local-only cost model.
# Each is roughly 15-7500× more expensive than the $0.01/1M target.
FRONTIER_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini":    {"input": 0.15, "output": 0.60},
    "gpt-4o":         {"input": 2.50, "output": 10.00},
    "claude-haiku":   {"input": 0.25, "output": 1.25},
    "claude-sonnet":  {"input": 3.00, "output": 15.00},
    "gemini-flash":   {"input": 0.075, "output": 0.30},
}

# Marginal cost for local inference when electricity is not modeled
LOCAL_COST: float = 0.0

# Default path for throughput results
_DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"

# ── Throughput results loader ──

def _find_newest_throughput_json(results_dir: Path | None = None) -> Path | None:
    """
    Find the newest throughput_<timestamp>.json in the results directory.

    Returns the Path, or None if no results found.
    """
    if results_dir is None:
        results_dir = _DEFAULT_RESULTS_DIR
    if not results_dir.exists():
        return None

    candidates: list[Path] = []
    for f in results_dir.iterdir():
        if f.is_file() and f.name.startswith("throughput_") and f.name.endswith(".json"):
            candidates.append(f)

    if not candidates:
        return None

    # Sort by modification time — newest last
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _find_newest_zero_weight_json(results_dir: Path | None = None) -> Path | None:
    """Find the newest zero_weight_<timestamp>.json."""
    if results_dir is None:
        results_dir = _DEFAULT_RESULTS_DIR
    if not results_dir.exists():
        return None

    candidates: list[Path] = []
    for f in results_dir.iterdir():
        if f.is_file() and f.name.startswith("zero_weight_") and f.name.endswith(".json"):
            candidates.append(f)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


# ── Local Cost Model ──

@dataclass
class LocalCostModel:
    """
    Cost model for local CPU inference based on electricity consumption.

    The cost of running locally is just the electricity used by the CPU
    during inference. No API markup, no per-token pricing — just watts × time.

    Attributes:
        tokens_per_second: Measured throughput (loaded from benchmark results)
        ram_mb: Peak RAM usage in MB (loaded from benchmark results)
        watts_at_load: CPU power draw during inference (default: 65W)
        electricity_cost_per_kwh: Local electricity rate (default: $0.15/kWh)
        source_file: Path to the throughput JSON this was loaded from
    """
    tokens_per_second: float
    ram_mb: float = 0.0
    watts_at_load: float = 65.0
    electricity_cost_per_kwh: float = 0.15
    source_file: str = ""

    @classmethod
    def from_latest_throughput(
        cls,
        results_dir: Path | str | None = None,
        metric: str = "p50_tps",
    ) -> "LocalCostModel | None":
        """
        Auto-load parameters from the newest throughput results JSON.

        Args:
            results_dir: Directory containing throughput_*.json files.
                         Defaults to benchmarks/results/.
            metric: Which throughput metric to use — 'p50_tps' (default)
                    or 'p95_tps' or 'mean_tps'.

        Returns:
            LocalCostModel instance or None if no results found.
        """
        if results_dir is None:
            results_dir = _DEFAULT_RESULTS_DIR
        elif isinstance(results_dir, str):
            results_dir = Path(results_dir)

        newest = _find_newest_throughput_json(results_dir)
        if newest is None:
            return None

        with open(newest, "r", encoding="utf-8") as f:
            data = json.load(f)

        tps = data.get(metric, 0.0)
        if tps is None or tps <= 0:
            # Try fallback metrics
            for fallback in ["p50_tps", "mean_tps", "p95_tps"]:
                tps = data.get(fallback, 0.0)
                if tps and tps > 0:
                    break

        # Get RAM — could be a number or a string like "unavailable: ..."
        ram_raw = data.get("ram_mb", 0.0)
        if isinstance(ram_raw, (int, float)):
            ram_mb = float(ram_raw)
        else:
            ram_mb = 0.0

        return cls(
            tokens_per_second=float(tps),
            ram_mb=ram_mb,
            source_file=str(newest),
        )

    @classmethod
    def from_file(cls, filepath: Path | str, metric: str = "p50_tps") -> "LocalCostModel":
        """
        Load parameters from a specific throughput results JSON.

        Args:
            filepath: Path to the throughput results JSON.
            metric: Which throughput metric to use.

        Returns:
            LocalCostModel instance.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tps = data.get(metric, 0.0)
        if tps is None or tps <= 0:
            for fallback in ["p50_tps", "mean_tps", "p95_tps"]:
                tps = data.get(fallback, 0.0)
                if tps and tps > 0:
                    break

        ram_raw = data.get("ram_mb", 0.0)
        if isinstance(ram_raw, (int, float)):
            ram_mb = float(ram_raw)
        else:
            ram_mb = 0.0

        return cls(
            tokens_per_second=float(tps),
            ram_mb=ram_mb,
            source_file=str(filepath),
        )

    def cost_per_1m_tokens(self) -> float:
        """
        Compute electricity cost to generate 1M tokens.

        Formula: kWh = (watts / 1000) × (1e6 / tps / 3600)
                 cost = kWh × $/kWh

        Returns:
            Cost in USD for 1 million tokens.
        """
        if self.tokens_per_second <= 0:
            return float("inf")
        seconds = 1_000_000 / self.tokens_per_second
        kwh = (self.watts_at_load / 1000) * (seconds / 3600)
        return kwh * self.electricity_cost_per_kwh

    def cost_for_tokens(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost for a specific prompt + completion pair."""
        total = prompt_tokens + completion_tokens
        return self.cost_per_1m_tokens() * (total / 1_000_000)

    def queries_per_hour(self, avg_tokens_per_query: int) -> float:
        """
        Max queries per hour given average tokens per query.

        Args:
            avg_tokens_per_query: Average total tokens (prompt + completion)

        Returns:
            Maximum queries per hour (assuming sequential, non-batched).
        """
        if self.tokens_per_second <= 0 or avg_tokens_per_query <= 0:
            return 0.0
        tokens_per_hour = self.tokens_per_second * 3600
        return tokens_per_hour / avg_tokens_per_query

    def tps_needed_for_target(self, target_cost: float = 0.01) -> float:
        """
        Compute tokens/sec needed to hit a per-1M-token cost target.

        Derivation:
            target = (watts/1000) × (1e6/tps/3600) × electricity
            tps = watts × 1e6 × electricity / (1000 × 3600 × target)

        Args:
            target_cost: Target cost per 1M tokens in USD (default: $0.01)

        Returns:
            Required tokens/second to meet the target.
        """
        return (self.watts_at_load * 1_000_000 * self.electricity_cost_per_kwh) / (
            1000 * 3600 * target_cost
        )

    def meets_target(self, target_cost: float = 0.01) -> bool:
        """Check if current throughput meets the cost target."""
        return self.cost_per_1m_tokens() <= target_cost

    def target_gap(self, target_cost: float = 0.01) -> float:
        """
        How many × more throughput is needed to hit the target.

        Returns 1.0 if already at target, >1.0 if below target.
        """
        if self.meets_target(target_cost):
            return 1.0
        return self.tps_needed_for_target(target_cost) / self.tokens_per_second

    def summary(self) -> str:
        """Human-readable cost summary."""
        cost = self.cost_per_1m_tokens()
        needed = self.tps_needed_for_target(0.01)
        qph_500 = self.queries_per_hour(500)

        lines = [
            f"LocalCostModel(tps={self.tokens_per_second:.1f}, "
            f"ram={self.ram_mb:.0f}MB, "
            f"watts={self.watts_at_load:.0f}W, elec=${self.electricity_cost_per_kwh:.2f}/kWh)",
        ]
        if self.source_file:
            lines.append(f"  Source: {self.source_file}")
        lines += [
            f"  Cost per 1M tokens: ${cost:.4f}",
            f"  Queries/hour (500 tok/q): {qph_500:.0f}",
            f"  To hit $0.01/1M: need {needed:.0f} tok/s "
            f"({needed/self.tokens_per_second:.1f}x current)",
        ]
        if self.meets_target(0.01):
            lines.append("  [PASS] MEETS $0.01/1M target")
        else:
            lines.append(f"  [FAIL] ${cost - 0.01:.4f} above target")
        return "\n".join(lines)


@dataclass
class BlendedRouterCost:
    """
    Cost model for the NEXUS router's blended pricing.

    The router sends synth_ratio (e.g., 80%) of queries to the template
    synthesizer ($0 cost, CPU-only), and the rest to the LLM.

    This dramatically reduces effective cost since template synthesis
    has negligible electricity cost (CPU overhead is ~10-50ms per query).

    Attributes:
        llm_cost_model: LocalCostModel for the LLM backend
        synth_ratio: Fraction of queries routed to synthesizer (default: 0.8)
    """
    llm_cost_model: LocalCostModel
    synth_ratio: float = 0.8

    @classmethod
    def from_latest_throughput(
        cls,
        results_dir: Path | str | None = None,
        synth_ratio: float = 0.8,
        metric: str = "p50_tps",
    ) -> "BlendedRouterCost | None":
        """
        Auto-load from the newest throughput results JSON.

        Args:
            results_dir: Directory containing throughput_*.json files.
            synth_ratio: Fraction of queries routed to synthesizer.
            metric: Which throughput metric to use.

        Returns:
            BlendedRouterCost instance or None if no results found.
        """
        llm_model = LocalCostModel.from_latest_throughput(
            results_dir=results_dir, metric=metric,
        )
        if llm_model is None:
            return None
        return cls(llm_cost_model=llm_model, synth_ratio=synth_ratio)

    def effective_cost_per_1m_tokens(self) -> float:
        """
        Blended cost per 1M user-facing tokens.

        Only (1 - synth_ratio) of tokens go through the LLM.
        The rest are template-synthesized at ~$0 cost.
        """
        llm_fraction = 1 - self.synth_ratio
        return llm_fraction * self.llm_cost_model.cost_per_1m_tokens()

    def effective_queries_per_hour(self, avg_tokens_per_query: int) -> float:
        """
        Max queries/hour with blended routing.

        Template synthesis is much faster than LLM inference, so the
        bottleneck is the LLM portion. Effective throughput = 1 / blended_time.
        """
        llm_fraction = 1 - self.synth_ratio
        if self.llm_cost_model.tokens_per_second <= 0:
            return 0.0

        # Template synthesis time ≈ 0.05s (50ms CPU overhead)
        synth_time_s = 0.05
        # LLM time per query
        llm_time_s = avg_tokens_per_query / self.llm_cost_model.tokens_per_second

        # Blended average time per query
        blended_time_s = (
            self.synth_ratio * synth_time_s + llm_fraction * llm_time_s
        )

        return 3600 / blended_time_s if blended_time_s > 0 else float("inf")

    def meets_target(self, target_cost: float = 0.01) -> bool:
        """Check if blended cost meets the target."""
        return self.effective_cost_per_1m_tokens() <= target_cost

    def min_synth_ratio_for_target(self, target_cost: float = 0.01) -> float:
        """
        Minimum synthesizer ratio needed to hit the cost target.

        blended = (1 - synth) × llm_cost ≤ target
        synth ≥ 1 - target/llm_cost
        """
        llm_cost = self.llm_cost_model.cost_per_1m_tokens()
        if llm_cost <= 0:
            return 0.0
        min_ratio = 1 - (target_cost / llm_cost)
        return max(0.0, min(min_ratio, 1.0))

    def summary(self, avg_tokens_per_query: int = 500) -> str:
        """Human-readable summary of blended cost."""
        eff_cost = self.effective_cost_per_1m_tokens()
        qph = self.effective_queries_per_hour(avg_tokens_per_query)
        min_synth = self.min_synth_ratio_for_target(0.01)
        llm_cost = self.llm_cost_model.cost_per_1m_tokens()

        lines = [
            f"BlendedRouterCost(synth={self.synth_ratio:.0%}, llm_tps={self.llm_cost_model.tokens_per_second:.1f})",
        ]
        if self.llm_cost_model.source_file:
            lines.append(f"  Source: {self.llm_cost_model.source_file}")
        lines += [
            f"  Raw LLM cost per 1M tokens:  ${llm_cost:.4f}",
            f"  Blended cost per 1M tokens:  ${eff_cost:.4f} "
            f"({self.synth_ratio:.0%} synth = $0)",
            f"  Effective queries/hour:      {qph:.0f}",
        ]
        if self.meets_target(0.01):
            lines.append("  [PASS] MEETS $0.01/1M target")
        elif min_synth <= 1.0:
            lines.append(f"  [FAIL] Above target -- need {min_synth:.0%} synth ratio to hit $0.01")
        else:
            lines.append(f"  [FAIL] Cannot hit target -- LLM alone costs ${llm_cost:.4f}")
        return "\n".join(lines)


# ── Utility functions ──

def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    cost_model: LocalCostModel | None = None,
    model_backend: str = "",
    local: bool = True,
) -> float:
    """
    Estimate USD cost for a single inference call.

    Uses LocalCostModel if provided, otherwise falls back to frontier pricing
    for backward compatibility (deprecated path).

    Args:
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens
        cost_model: LocalCostModel instance (preferred)
        model_backend: Model name for frontier pricing (deprecated)
        local: Must be True for the local-only paradigm

    Returns:
        Estimated cost in USD
    """
    if cost_model is not None:
        return round(cost_model.cost_for_tokens(prompt_tokens, completion_tokens), 6)

    # Backward compatibility: fall back to frontier pricing if no LocalCostModel
    if local:
        return LOCAL_COST

    pricing = FRONTIER_PRICING.get(model_backend)
    if pricing is None:
        return 0.0

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def estimate_cost_per_1k(
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
    cost_model: LocalCostModel | None = None,
    model_backend: str = "",
    local: bool = True,
) -> float:
    """
    Estimate USD cost for 1,000 inference calls.

    Uses LocalCostModel if provided.

    Args:
        avg_prompt_tokens: Average prompt tokens per call
        avg_completion_tokens: Average completion tokens per call
        cost_model: LocalCostModel instance (preferred)
        model_backend: Model name for frontier pricing (deprecated)
        local: Must be True for the local-only paradigm

    Returns:
        Estimated cost per 1K questions in USD
    """
    if cost_model is not None:
        total_tokens = (avg_prompt_tokens + avg_completion_tokens) * 1000
        return round(cost_model.cost_per_1m_tokens() * (total_tokens / 1_000_000), 4)

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
    cost_model: LocalCostModel | None = None,
    local: bool = True,
) -> list[str]:
    """
    Build a human-readable cost comparison table for a system.

    With LocalCostModel: shows local electricity cost vs frontier API costs.
    Without: shows frontier API costs only (deprecated path).

    Returns a list of formatted strings.
    """
    lines: list[str] = []
    avg_total_tokens = avg_prompt_tokens + avg_completion_tokens

    if cost_model is not None:
        # Local cost comparison
        local_cost_1k = estimate_cost_per_1k(
            avg_prompt_tokens, avg_completion_tokens, cost_model=cost_model, local=True,
        )
        cost_1m = cost_model.cost_per_1m_tokens()
        qph = cost_model.queries_per_hour(int(avg_total_tokens))

        lines.append(f"  {system_name} (local, electricity):  "
                     f"${local_cost_1k:.4f}/1K | "
                     f"${cost_1m:.4f}/1M tokens | "
                     f"{qph:.0f} queries/hour")
    else:
        lines.append(f"  {system_name} + local (any model):  $0.00")

    # Show frontier costs for comparison (what you'd pay if using APIs)
    lines.append(f"  {'-'*60}")
    lines.append(f"  Comparison -- what frontier APIs would cost:")
    models_to_compare = ["gpt-4o-mini", "claude-haiku", "gemini-flash"]
    for model in models_to_compare:
        cost_1k = estimate_cost_per_1k(
            avg_prompt_tokens, avg_completion_tokens, model_backend=model, local=False,
        )
        lines.append(f"    {system_name} + {model}:  ${cost_1k:.2f}/1K")

    return lines


def format_router_cost_comparison(
    system_name: str,
    avg_prompt_tokens: float,
    avg_completion_tokens: float,
    blended: BlendedRouterCost,
) -> list[str]:
    """
    Build a cost comparison table for the NEXUS router configuration.

    Shows effective blended cost with template synthesis at $0.
    """
    lines: list[str] = []
    eff_cost = blended.effective_cost_per_1m_tokens()
    llm_cost = blended.llm_cost_model.cost_per_1m_tokens()
    qph = blended.effective_queries_per_hour(int(avg_prompt_tokens + avg_completion_tokens))

    lines.append(f"  {system_name}:")
    lines.append(f"    Raw LLM cost:       ${llm_cost:.4f}/1M tokens")
    lines.append(f"    Blended cost:       ${eff_cost:.4f}/1M tokens "
                 f"({blended.synth_ratio:.0%} synth = $0)")
    lines.append(f"    Effective Q/hour:   {qph:.0f}")
    lines.append(f"    Savings vs raw LLM: ${llm_cost - eff_cost:.4f}/1M "
                 f"({(1 - eff_cost/llm_cost)*100:.0f}%)" if llm_cost > 0 else "")

    return lines


def make_cost_model(tokens_per_second: float, **kwargs) -> LocalCostModel:
    """Convenience factory for LocalCostModel (explicit tps)."""
    return LocalCostModel(tokens_per_second=tokens_per_second, **kwargs)


def make_blended_router(
    tokens_per_second: float,
    synth_ratio: float = 0.8,
    **kwargs,
) -> BlendedRouterCost:
    """Convenience factory for BlendedRouterCost (explicit tps)."""
    llm_model = LocalCostModel(tokens_per_second=tokens_per_second, **kwargs)
    return BlendedRouterCost(llm_cost_model=llm_model, synth_ratio=synth_ratio)


def load_cost_model_from_results(
    results_dir: Path | str | None = None,
    metric: str = "p50_tps",
) -> LocalCostModel | None:
    """
    Preferred entry point: auto-load cost model from the newest throughput JSON.

    Prints the source file path so users know exactly which benchmark
    produced the numbers.
    """
    model = LocalCostModel.from_latest_throughput(results_dir=results_dir, metric=metric)
    if model is not None:
        print(f"[cost_model] Loaded from: {model.source_file}")
        print(f"[cost_model] p50={model.tokens_per_second:.1f} tok/s, "
              f"ram={model.ram_mb:.0f}MB, "
              f"${model.cost_per_1m_tokens():.4f}/1M tokens (raw LLM)")
    else:
        print("[cost_model] WARNING: No throughput results found. Run "
              "`python benchmarks/throughput_bench.py` first.")
    return model


def load_router_cost_from_results(
    results_dir: Path | str | None = None,
    synth_ratio: float = 0.8,
    metric: str = "p50_tps",
) -> BlendedRouterCost | None:
    """
    Preferred entry point: auto-load router cost from newest throughput JSON.
    """
    router = BlendedRouterCost.from_latest_throughput(
        results_dir=results_dir, synth_ratio=synth_ratio, metric=metric,
    )
    if router is not None:
        print(f"[cost_model] Router loaded from: {router.llm_cost_model.source_file}")
        print(f"[cost_model] Raw LLM: ${router.llm_cost_model.cost_per_1m_tokens():.4f}/1M, "
              f"Blended: ${router.effective_cost_per_1m_tokens():.4f}/1M")
    else:
        print("[cost_model] WARNING: No throughput results found.")
    return router
