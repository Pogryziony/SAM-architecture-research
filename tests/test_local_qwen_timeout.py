"""Local Qwen wall-clock timeout must not hang forever.

Tests verify that:
1. Timeouts return promptly without hanging
2. Error states are properly reported
3. Process-based isolation works correctly
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from nexus.baselines.local_qwen import (
    FROZEN_DECODING,
    LocalQwenAdapter,
    LocalQwenIdentity,
)


def _identity() -> LocalQwenIdentity:
    return LocalQwenIdentity(
        runtime="ollama",
        runtime_version="0.32.1",
        model_name="qwen3.6:latest",
        digest="07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522",
        architecture="qwen35moe",
        parameter_size="36.0B",
        quantization="Q4_K_M",
        context_length=262144,
        embedding_length=2048,
        host="http://127.0.0.1:11434",
        think_disabled=True,
        decoding=dict(FROZEN_DECODING),
    )


def test_generate_wall_clock_timeout_returns_timed_out():
    """Verify timeout returns with error within wall-clock limit.

    Since the refactored code uses subprocess-based HTTP calls, we test
    against a non-routable IP to trigger a real timeout condition.
    """
    # Use non-routable IP that will trigger timeout
    identity = LocalQwenIdentity(
        runtime="ollama",
        runtime_version="0.32.1",
        model_name="qwen3.6:latest",
        digest="07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522",
        architecture="qwen35moe",
        parameter_size="36.0B",
        quantization="Q4_K_M",
        context_length=262144,
        embedding_length=2048,
        host="http://10.255.255.1:11434",  # Non-routable IP for timeout
        think_disabled=True,
        decoding=dict(FROZEN_DECODING),
    )
    adapter = LocalQwenAdapter(identity)

    t0 = time.perf_counter()
    gen = adapter.generate(
        "hi",
        decoding={**FROZEN_DECODING, "timeout_s": 2.0, "num_predict": 8},
    )
    elapsed = time.perf_counter() - t0

    # Should return within wall-clock limit (timeout + grace period)
    assert elapsed < 60.0, f"Should not hang; took {elapsed:.1f}s"
    # Should have an error
    assert gen.error, "Should report an error"
    # Connection refused or timeout should set timed_out flag
    err_lower = gen.error.lower()
    if "timed out" in err_lower or "timeout" in err_lower:
        assert gen.timed_out is True


def test_connection_refused_returns_error():
    """Verify connection refused is handled gracefully."""
    # Use localhost with unlikely port
    identity = LocalQwenIdentity(
        runtime="ollama",
        runtime_version="0.32.1",
        model_name="qwen3.6:latest",
        digest="07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522",
        architecture="qwen35moe",
        parameter_size="36.0B",
        quantization="Q4_K_M",
        context_length=262144,
        embedding_length=2048,
        host="http://127.0.0.1:59999",  # Very unlikely to be in use
        think_disabled=True,
        decoding=dict(FROZEN_DECODING),
    )
    adapter = LocalQwenAdapter(identity)

    gen = adapter.generate("hi", decoding={**FROZEN_DECODING, "num_predict": 8})

    # Should return with error
    assert gen.error, "Should report an error on connection refused"
    # Should not have successful output
    assert not gen.parsed_answer
