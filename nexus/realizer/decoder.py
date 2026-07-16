"""Configurable autoregressive decoder for NEXUS Realizer byte-level Transformer.

Supports greedy, temperature, top-k, top-p (nucleus), beam search,
repetition penalty, no-repeat n-gram blocking, and length control.

All strategies are deterministic when a seed is supplied.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import torch

from nexus.realizer.tokenizer import ByteTokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DecoderConfig:
    """Immutable decoding configuration.

    All fields are configuration-driven.  Supply ``seed`` for deterministic
    sampling (temperature / top-k / top-p).
    """

    strategy: str = "greedy"  # greedy | beam | sample
    temperature: float = 1.0  # > 0; 1.0 = no change, < 1 = sharper, > 1 = flatter
    top_k: int = 0  # 0 = disabled
    top_p: float = 0.0  # 0.0 = disabled (nucleus sampling)
    beam_width: int = 1  # 1 = greedy
    length_penalty: float = 0.0  # beam-search length penalty (positive = favour longer)
    repetition_penalty: float = 1.0  # > 1.0 penalises repeated tokens
    no_repeat_ngram_size: int = 0  # 0 = disabled; 2 = bigram, 3 = trigram, etc.
    min_length: int = 1  # minimum generated tokens (excluding BOS)
    max_length: int = 256
    eos_token_id: int = ByteTokenizer.EOS
    pad_token_id: int = ByteTokenizer.PAD
    bos_token_id: int = ByteTokenizer.BOS
    seed: int | None = None  # None = non-deterministic sampling

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.min_length < 0:
            raise ValueError("min_length must be >= 0")
        if self.max_length < 1:
            raise ValueError("max_length must be >= 1")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")


# ═══════════════════════════════════════════════════════════════════════════
# Decoding strategies
# ═══════════════════════════════════════════════════════════════════════════


def _apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: list[int], penalty: float,
) -> torch.Tensor:
    """Penalise already-generated tokens by dividing their logits by *penalty*."""
    if penalty <= 1.0:
        return logits
    for tid in set(generated_ids):
        if logits[tid] > 0:
            logits[tid] = logits[tid] / penalty
        else:
            logits[tid] = logits[tid] * penalty
    return logits


def _block_repeated_ngrams(
    logits: torch.Tensor, generated_ids: list[int], ngram_size: int,
) -> torch.Tensor:
    """Set logits to -inf for tokens that would create a repeated n-gram."""
    if ngram_size < 2 or len(generated_ids) < ngram_size - 1:
        return logits
    prefix = tuple(generated_ids[-(ngram_size - 1):])
    # Find which next token would complete an n-gram already present
    full_history = generated_ids + [0]  # placeholder
    for token in range(logits.shape[0]):
        full_history[-1] = token
        # Check if last ngram_size tokens appear earlier
        candidate = tuple(full_history[-ngram_size:])
        for i in range(len(full_history) - ngram_size):
            if tuple(full_history[i:i + ngram_size]) == candidate:
                logits[token] = float("-inf")
                break
    return logits


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top k."""
    if k <= 0 or k >= logits.shape[0]:
        return logits
    top_k_values, _ = torch.topk(logits, k)
    threshold = top_k_values[-1]
    logits[logits < threshold] = float("-inf")
    return logits


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens with cumulative prob >= p."""
    if p <= 0.0 or p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift to keep at least one token
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(
        0, sorted_indices, sorted_indices_to_remove,
    )
    logits[indices_to_remove] = float("-inf")
    return logits


def _sample_token(logits: torch.Tensor, rng: random.Random | None) -> int:
    """Sample a token from logits. If rng is None, use torch.multinomial."""
    logits = logits.clone()
    # Apply temperature
    temperature = 1.0  # controlled externally
    probs = torch.softmax(logits, dim=-1)
    if rng is not None:
        # Deterministic sampling via random.Random
        cumsum = 0.0
        r = rng.random()
        for token in range(probs.shape[0]):
            cumsum += probs[token].item()
            if r < cumsum:
                return token
        return probs.argmax().item()
    else:
        return torch.multinomial(probs, 1).item()


# ═══════════════════════════════════════════════════════════════════════════
# Greedy / sampling decoder
# ═══════════════════════════════════════════════════════════════════════════


def _greedy_or_sample_decode(
    model: Any,
    source: torch.Tensor,
    config: DecoderConfig,
    rng: random.Random | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Autoregressive decode one sequence (greedy or sampling)."""
    generated: list[int] = []
    target = torch.tensor([[config.bos_token_id]], dtype=torch.long)
    eos_reached = False
    token_count = 0
    entropies: list[float] = []

    with torch.no_grad():
        while len(generated) < config.max_length:
            logits = model(source, target)
            next_logits = logits[0, -1, :].clone()

            # Track entropy before modifications
            probs = torch.softmax(next_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropies.append(entropy)

            # Enforce min_length: block EOS
            if len(generated) < config.min_length:
                next_logits[config.eos_token_id] = float("-inf")

            # Apply repetition penalty
            next_logits = _apply_repetition_penalty(
                next_logits, generated, config.repetition_penalty,
            )

            # Apply no-repeat n-gram
            next_logits = _block_repeated_ngrams(
                next_logits, generated, config.no_repeat_ngram_size,
            )

            # Apply temperature
            if config.temperature != 1.0:
                next_logits = next_logits / config.temperature

            # Apply top-k / top-p
            next_logits = _top_k_filter(next_logits.clone(), config.top_k)
            next_logits = _top_p_filter(next_logits.clone(), config.top_p)

            if config.strategy == "greedy":
                next_token = next_logits.argmax().item()
            else:
                # Sampling
                next_token = _sample_token(next_logits, rng)

            if next_token == config.eos_token_id:
                eos_reached = True
                break
            if next_token == config.pad_token_id:
                break

            generated.append(next_token)
            target = torch.cat(
                [target, torch.tensor([[next_token]], dtype=torch.long)], dim=1,
            )
            token_count += 1

    diagnostics = {
        "token_count": token_count,
        "eos_reached": eos_reached,
        "entropy_mean": sum(entropies) / len(entropies) if entropies else 0.0,
        "entropy_final": entropies[-1] if entropies else 0.0,
    }
    return generated, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# Beam search decoder
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(order=True)
class _BeamHypothesis:
    score: float = field(compare=True)
    token_ids: list[int] = field(compare=False)
    target_tensor: torch.Tensor = field(compare=False)
    eos_reached: bool = field(compare=False)


def _beam_decode(
    model: Any,
    source: torch.Tensor,
    config: DecoderConfig,
    rng: random.Random | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Beam-search decode one sequence."""
    beam_width = config.beam_width
    bos = torch.tensor([[config.bos_token_id]], dtype=torch.long)
    beams = [_BeamHypothesis(0.0, [], bos, False)]

    with torch.no_grad():
        for step in range(config.max_length):
            new_beams: list[_BeamHypothesis] = []
            for beam in beams:
                if beam.eos_reached:
                    new_beams.append(beam)
                    continue

                logits = model(source, beam.target_tensor)
                next_logits = logits[0, -1, :].clone()

                # Enforce min_length
                if step < config.min_length:
                    next_logits[config.eos_token_id] = float("-inf")

                # Repetition penalty
                next_logits = _apply_repetition_penalty(
                    next_logits, beam.token_ids, config.repetition_penalty,
                )

                # No-repeat n-gram
                next_logits = _block_repeated_ngrams(
                    next_logits, beam.token_ids, config.no_repeat_ngram_size,
                )

                # Temperature
                if config.temperature != 1.0:
                    next_logits = next_logits / config.temperature

                log_probs = torch.log_softmax(next_logits, dim=-1)
                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_width * 2)

                for lp, tok in zip(topk_log_probs, topk_tokens):
                    token = tok.item()
                    lp_val = lp.item()
                    # Length penalty
                    new_score = beam.score + lp_val
                    if config.length_penalty != 0.0:
                        curr_len = len(beam.token_ids) + 1
                        new_score = new_score / (curr_len ** config.length_penalty)

                    new_ids = beam.token_ids + [token]
                    new_target = torch.cat(
                        [beam.target_tensor, torch.tensor([[token]], dtype=torch.long)], dim=1,
                    )
                    eos = token == config.eos_token_id
                    new_beams.append(_BeamHypothesis(new_score, new_ids, new_target, eos))

            # Prune to beam_width
            new_beams.sort(key=lambda b: b.score, reverse=True)
            beams = new_beams[:beam_width]

            # All beams done?
            if all(b.eos_reached for b in beams):
                break

    # Pick best completed beam, or best overall
    completed = [b for b in beams if b.eos_reached]
    best = max(completed if completed else beams, key=lambda b: b.score)

    return best.token_ids, {
        "token_count": len(best.token_ids),
        "eos_reached": best.eos_reached,
        "beam_score": best.score,
        "completed_beams": len(completed),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def decode(
    model: Any,
    source_ids: list[int],
    config: DecoderConfig | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Generate token IDs from source using the configured decoder.

    Args:
        model: Trained NEXUS Realizer model with ``forward(source, target)``.
        source_ids: Encoded source token IDs (including BOS / EOS).
        config: Decoding configuration (defaults to greedy).

    Returns:
        (generated_token_ids, diagnostics_dict)
    """
    if config is None:
        config = DecoderConfig()

    source = torch.tensor([source_ids], dtype=torch.long)
    model.eval()

    # Set up deterministic random state for sampling
    rng = random.Random(config.seed) if config.seed is not None and config.strategy == "sample" else None

    if config.strategy == "beam" and config.beam_width > 1:
        return _beam_decode(model, source, config, rng)

    # For beam_width=1, fall through to greedy
    if config.strategy == "beam" and config.beam_width <= 1:
        config = DecoderConfig(
            strategy="greedy",
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            min_length=config.min_length,
            max_length=config.max_length,
            seed=config.seed,
        )

    return _greedy_or_sample_decode(model, source, config, rng)


def decode_to_text(
    model: Any,
    source_ids: list[int],
    tokenizer: ByteTokenizer | None = None,
    config: DecoderConfig | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate text from source IDs.  Convenience wrapper around :func:`decode`."""
    if tokenizer is None:
        tokenizer = ByteTokenizer()
    token_ids, diagnostics = decode(model, source_ids, config)
    text = tokenizer.decode(token_ids)
    return text.strip(), diagnostics


def score_candidate_texts(
    model: Any,
    source_ids: list[int],
    candidate_texts: list[str],
    tokenizer: ByteTokenizer | None = None,
    max_length: int = 256,
) -> tuple[str, dict[str, Any]]:
    """Select one complete allowed output by mean teacher-forced NLL.

    This is constrained decoding, not label lookup: every candidate is scored
    only from the model, source and its own prefix.  It is intended for output
    contracts with a small finite language, where free byte generation can
    corrupt an otherwise valid identifier or control token.
    """
    if not candidate_texts:
        raise ValueError("candidate_texts must not be empty")
    if len(set(candidate_texts)) != len(candidate_texts):
        raise ValueError("candidate_texts must be unique")
    tokenizer = tokenizer or ByteTokenizer()
    device = next(model.parameters()).device
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    scored: list[tuple[float, str, int]] = []
    model.eval()
    with torch.no_grad():
        for text in candidate_texts:
            ids = tokenizer.encode(text, max_length)
            target = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(source, target[:, :-1])
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target[:, 1:].reshape(-1),
                reduction="none",
            )
            score = float(losses.mean())
            scored.append((score, text, len(ids) - 2))
    scored.sort(key=lambda item: (item[0], item[1]))
    best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else best[0]
    return best[1], {
        "strategy": "constrained_candidates",
        "candidate_mean_nll": {
            text: round(score, 8) for score, text, _ in scored
        },
        "score_margin": round(runner_up - best[0], 8),
        "token_count": best[2],
        "eos_reached": True,
    }


def compute_repetition_rates(token_ids: list[int]) -> dict[str, float]:
    """Compute n-gram repetition rates for generated token sequence."""
    result: dict[str, float] = {}
    for n in (2, 3, 4):
        if len(token_ids) < n:
            result[f"rep_{n}gram"] = 0.0
            continue
        ngrams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
        unique = len(set(ngrams))
        result[f"rep_{n}gram"] = 1.0 - (unique / len(ngrams))
    return result


__all__ = [
    "DecoderConfig",
    "decode",
    "decode_to_text",
    "score_candidate_texts",
    "compute_repetition_rates",
]
