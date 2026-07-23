"""Provider-neutral baseline execution contract."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class BaselineStatus(str, Enum):
    OK = "OK"
    NOT_RUN = "NOT_RUN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class BaselineRequest:
    """One evaluation request under a named baseline arm."""

    arm_id: str
    question_id: str
    question: str
    corpus_id: str = ""
    decoding: Mapping[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    max_context_tokens: int | None = None


@dataclass
class BaselineResult:
    """Terminal result for one baseline request."""

    arm_id: str
    question_id: str
    status: BaselineStatus
    answer: str = ""
    retrieved_documents: list[str] = field(default_factory=list)
    model_id: str = ""
    provider: str = ""
    prompt_fingerprint: str = ""
    decoding: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    token_cost: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    prerequisites: list[str] = field(default_factory=list)
    command: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class BaselineArm:
    """Declarative baseline arm metadata."""

    arm_id: str
    family: str
    description: str
    requires_env: tuple[str, ...] = ()
    requires_packages: tuple[str, ...] = ()
    is_placeholder: bool = False
    modern_rag: bool = False
    run: Callable[[BaselineRequest], BaselineResult] | None = None


def missing_prerequisites(arm: BaselineArm) -> list[str]:
    missing: list[str] = []
    for name in arm.requires_env:
        if not os.environ.get(name):
            missing.append(f"env:{name}")
    for pkg in arm.requires_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"package:{pkg}")
    return missing


def run_baseline_or_not_run(
    arm: BaselineArm,
    request: BaselineRequest,
    *,
    command: str = "",
) -> BaselineResult:
    """Execute an arm or emit an honest NOT_RUN record."""
    if arm.is_placeholder:
        return BaselineResult(
            arm_id=arm.arm_id,
            question_id=request.question_id,
            status=BaselineStatus.NOT_RUN,
            failure_reason=(
                "arm is a deterministic placeholder and must not be reported "
                "as a real LLM/RAG baseline"
            ),
            prerequisites=list(arm.requires_env) + list(arm.requires_packages),
            command=command,
            model_id="PLACEHOLDER",
            provider="local-deterministic",
        )
    missing = missing_prerequisites(arm)
    if missing:
        return BaselineResult(
            arm_id=arm.arm_id,
            question_id=request.question_id,
            status=BaselineStatus.NOT_RUN,
            failure_reason="missing_prerequisites",
            prerequisites=missing,
            command=command,
        )
    if arm.run is None:
        return BaselineResult(
            arm_id=arm.arm_id,
            question_id=request.question_id,
            status=BaselineStatus.NOT_RUN,
            failure_reason="runner_not_implemented",
            command=command,
        )
    try:
        return arm.run(request)
    except Exception as exc:  # noqa: BLE001 — baseline boundary
        return BaselineResult(
            arm_id=arm.arm_id,
            question_id=request.question_id,
            status=BaselineStatus.ERROR,
            failure_reason=str(exc),
            command=command,
        )


def batch_not_run(
    arm: BaselineArm,
    requests: Sequence[BaselineRequest],
    *,
    command: str = "",
) -> list[BaselineResult]:
    return [
        run_baseline_or_not_run(arm, req, command=command) for req in requests
    ]
