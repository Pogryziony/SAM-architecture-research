"""Local Qwen wall-clock timeout must not hang forever."""

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
    adapter = LocalQwenAdapter(_identity())

    def _hang(*_a, **_k):
        time.sleep(60)

    t0 = time.perf_counter()
    with patch("nexus.baselines.local_qwen.urllib.request.urlopen", side_effect=_hang):
        gen = adapter.generate(
            "hi",
            decoding={**FROZEN_DECODING, "timeout_s": 1.0, "num_predict": 8},
        )
    elapsed = time.perf_counter() - t0
    assert gen.timed_out is True
    assert gen.error
    assert elapsed < 30.0
