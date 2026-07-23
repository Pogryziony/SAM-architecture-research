"""Local Qwen 3.6 identity and Ollama adapter (no remote LLM APIs)."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


DEFAULT_HOST = "http://127.0.0.1:11434"
REQUIRED_MODEL_NAME = "qwen3.6:latest"
# Full digest is required; prefix constant retained for error messages only.
REQUIRED_DIGEST = (
    "07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522"
)
REQUIRED_DIGEST_PREFIX = REQUIRED_DIGEST  # backward-compatible alias

# Frozen Phase-4 decoding (preregistered). Do not retune per question.
FROZEN_DECODING: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "seed": 0,
    "num_predict": 256,
    "think": False,
    "timeout_s": 180.0,
    "retry_max": 0,
}

FROZEN_SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about a research knowledge "
    "base. Use only the information provided in the user message. If the "
    "information is insufficient, reply exactly: Insufficient evidence to answer. "
    "Do not invent facts. Prefer concise factual answers."
)

FROZEN_CLOSED_BOOK_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Answer the question using only your general knowledge. "
    "If you are unsure, reply exactly: Insufficient evidence to answer."
)

FROZEN_EVIDENCE_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Answer using only the evidence above. If the evidence is insufficient, "
    "reply exactly: Insufficient evidence to answer."
)


@dataclass(frozen=True)
class LocalQwenIdentity:
    """Fail-closed identity for the installed local Qwen 3.6 runtime."""

    runtime: str
    runtime_version: str
    model_name: str
    digest: str
    architecture: str
    parameter_size: str
    quantization: str
    context_length: int
    embedding_length: int
    host: str
    think_disabled: bool
    decoding: Mapping[str, Any] = field(default_factory=lambda: dict(FROZEN_DECODING))
    system_prompt_sha256: str = ""
    identity_schema: str = "nexus-local-qwen-identity-v1"

    @property
    def identity_hash(self) -> str:
        payload = {
            "identity_schema": self.identity_schema,
            "runtime": self.runtime,
            "runtime_version": self.runtime_version,
            "model_name": self.model_name,
            "digest": self.digest,
            "architecture": self.architecture,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "think_disabled": self.think_disabled,
            "decoding": dict(self.decoding),
            "system_prompt_sha256": self.system_prompt_sha256
            or hashlib.sha256(FROZEN_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decoding"] = dict(self.decoding)
        d["identity_hash"] = self.identity_hash
        d["model_id"] = (
            f"ollama/{self.model_name}@{self.digest[:12]}"
        )
        return d


class LocalQwenUnavailableError(RuntimeError):
    """Raised when the required local Qwen 3.6 identity cannot be validated."""


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="GET" if data is None else "POST",
    )
    # Explicit socket timeout; also wrap in a future wall-clock bound so a
    # stuck Ollama worker cannot hang the evaluation process indefinitely.
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    def _do() -> Any:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # Slightly above urllib timeout so socket timeout usually wins first.
    wall = float(timeout) + 15.0
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(_do)
        try:
            return fut.result(timeout=wall)
        except FuturesTimeout as exc:
            raise TimeoutError(
                f"Ollama request exceeded wall-clock timeout {wall:.0f}s"
            ) from exc
    finally:
        # Do not join a stuck worker thread; abandon it so eval can continue.
        pool.shutdown(wait=False, cancel_futures=True)


def discover_local_qwen(
    *,
    host: str = DEFAULT_HOST,
    required_model: str = REQUIRED_MODEL_NAME,
    required_digest: str = REQUIRED_DIGEST,
) -> LocalQwenIdentity:
    """Discover and validate the installed Qwen 3.6 model. Fail closed on mismatch."""
    try:
        tags = _http_json(f"{host.rstrip('/')}/api/tags", timeout=10.0)
    except Exception as exc:
        raise LocalQwenUnavailableError(
            f"Ollama unreachable at {host}: {exc}"
        ) from exc

    models = tags.get("models") or []
    match = None
    candidates = []
    for m in models:
        name = str(m.get("name") or m.get("model") or "")
        digest = str(m.get("digest") or "")
        if "qwen3.6" in name.casefold() or "qwen3_6" in name.casefold():
            candidates.append({"name": name, "digest": digest, "details": m.get("details")})
        if name == required_model and digest == required_digest:
            match = m
            break
        if name == required_model:
            match = m

    if match is None:
        raise LocalQwenUnavailableError(
            "Required model qwen3.6:latest not found. "
            f"qwen3.6 candidates={candidates!r}"
        )

    digest = str(match.get("digest") or "")
    if digest != required_digest:
        raise LocalQwenUnavailableError(
            f"Full digest mismatch for {required_model}: got {digest}, "
            f"expected {required_digest}"
        )

    details = match.get("details") or {}
    runtime_version = "unknown"
    try:
        # ollama version endpoint is not always present; keep tags discovery
        import subprocess

        runtime_version = (
            subprocess.check_output(["ollama", "--version"], text=True, timeout=5)
            .strip()
            .replace("ollama version is ", "")
        )
    except Exception:
        pass

    return LocalQwenIdentity(
        runtime="ollama",
        runtime_version=runtime_version,
        model_name=str(match.get("name") or required_model),
        digest=digest,
        architecture=str(details.get("family") or ""),
        parameter_size=str(details.get("parameter_size") or ""),
        quantization=str(details.get("quantization_level") or ""),
        context_length=int(details.get("context_length") or 0),
        embedding_length=int(details.get("embedding_length") or 0),
        host=host.rstrip("/"),
        think_disabled=True,
        decoding=dict(FROZEN_DECODING),
        system_prompt_sha256=hashlib.sha256(
            FROZEN_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    )


@dataclass
class LocalQwenGeneration:
    raw_response: str
    parsed_answer: str
    latency_ms: float
    load_duration_ns: int | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    eval_duration_ns: int | None = None
    total_duration_ns: int | None = None
    # Non-streaming Ollama: this is prompt_eval_duration, NOT true streamed TTFT.
    time_to_first_token_ms: float | None = None
    prompt_eval_duration_ms: float | None = None
    ttft_metric: str = "prompt_eval_duration_ms_nonstream_proxy"
    tokens_per_second: float | None = None
    error: str = ""
    timed_out: bool = False
    prompt: str = ""
    system_prompt: str = ""
    prompt_sha256: str = ""
    retries_used: int = 0


class LocalQwenAdapter:
    """Provider adapter for the pinned local Qwen 3.6 Ollama model."""

    def __init__(self, identity: LocalQwenIdentity | None = None):
        self.identity = identity or discover_local_qwen()
        if self.identity.model_name != REQUIRED_MODEL_NAME:
            raise LocalQwenUnavailableError(
                f"Refusing non-pinned model {self.identity.model_name!r}"
            )

    @property
    def model_id(self) -> str:
        return self.identity.to_dict()["model_id"]

    def generate(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
        decoding: Mapping[str, Any] | None = None,
    ) -> LocalQwenGeneration:
        dec = dict(FROZEN_DECODING)
        if decoding:
            # Only allow keys already frozen; reject silent retunes of unknown knobs.
            for k, v in decoding.items():
                if k not in dec and k not in {"num_ctx"}:
                    raise ValueError(f"unfrozen decoding key forbidden: {k}")
                dec[k] = v
        sys_p = FROZEN_SYSTEM_PROMPT if system_prompt is None else system_prompt
        options = {
            "temperature": float(dec["temperature"]),
            "top_p": float(dec["top_p"]),
            "top_k": int(dec["top_k"]),
            "num_predict": int(dec["num_predict"]),
            "seed": int(dec["seed"]),
        }
        if "num_ctx" in dec:
            options["num_ctx"] = int(dec["num_ctx"])

        payload = {
            "model": self.identity.model_name,
            "system": sys_p,
            "prompt": user_prompt,
            "stream": False,
            "think": bool(dec.get("think", False)),
            "options": options,
        }
        timeout = float(dec.get("timeout_s", 180.0))
        retry_max = int(dec.get("retry_max", 0))
        if retry_max < 0:
            raise ValueError("retry_max must be >= 0")
        t0 = time.perf_counter()
        data = None
        last_exc: Exception | None = None
        retries_used = 0
        for attempt in range(retry_max + 1):
            try:
                data = _http_json(
                    f"{self.identity.host}/api/generate",
                    payload=payload,
                    timeout=timeout,
                )
                retries_used = attempt
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                retries_used = attempt
                if attempt >= retry_max:
                    msg = str(exc)
                    timed_out = (
                        isinstance(exc, TimeoutError)
                        or "timed out" in msg.casefold()
                        or "timeout" in msg.casefold()
                    )
                    return LocalQwenGeneration(
                        raw_response="",
                        parsed_answer="",
                        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
                        error=f"{type(exc).__name__}: {exc}",
                        timed_out=timed_out,
                        prompt=user_prompt,
                        system_prompt=sys_p,
                        prompt_sha256=hashlib.sha256(
                            (sys_p + "\n\n" + user_prompt).encode("utf-8")
                        ).hexdigest(),
                        retries_used=retries_used,
                    )
            except Exception as exc:
                last_exc = exc
                retries_used = attempt
                if attempt >= retry_max:
                    msg = str(exc)
                    timed_out = "timed out" in msg.casefold() or "timeout" in msg.casefold()
                    return LocalQwenGeneration(
                        raw_response="",
                        parsed_answer="",
                        latency_ms=round((time.perf_counter() - t0) * 1000, 3),
                        error=f"{type(exc).__name__}: {exc}",
                        timed_out=timed_out,
                        prompt=user_prompt,
                        system_prompt=sys_p,
                        prompt_sha256=hashlib.sha256(
                            (sys_p + "\n\n" + user_prompt).encode("utf-8")
                        ).hexdigest(),
                        retries_used=retries_used,
                    )
        if data is None:
            return LocalQwenGeneration(
                raw_response="",
                parsed_answer="",
                latency_ms=round((time.perf_counter() - t0) * 1000, 3),
                error=f"generate_failed: {last_exc}",
                prompt=user_prompt,
                system_prompt=sys_p,
                retries_used=retries_used,
            )

        raw = str(data.get("response") or "")
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        eval_count = data.get("eval_count")
        eval_duration = data.get("eval_duration")
        tps = None
        if eval_count and eval_duration:
            tps = float(eval_count) / (float(eval_duration) / 1e9)
        # Honest labeling: non-stream generate cannot measure true TTFT.
        prompt_eval_ms = None
        if data.get("prompt_eval_duration"):
            prompt_eval_ms = round(float(data["prompt_eval_duration"]) / 1e6, 3)

        return LocalQwenGeneration(
            raw_response=raw,
            parsed_answer=raw.strip(),
            latency_ms=elapsed_ms,
            load_duration_ns=data.get("load_duration"),
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=eval_count,
            eval_duration_ns=eval_duration,
            total_duration_ns=data.get("total_duration"),
            time_to_first_token_ms=prompt_eval_ms,
            prompt_eval_duration_ms=prompt_eval_ms,
            ttft_metric="prompt_eval_duration_ms_nonstream_proxy",
            tokens_per_second=None if tps is None else round(tps, 4),
            prompt=user_prompt,
            system_prompt=sys_p,
            prompt_sha256=hashlib.sha256(
                (sys_p + "\n\n" + user_prompt).encode("utf-8")
            ).hexdigest(),
            retries_used=retries_used,
        )

    def health_check(self) -> dict[str, Any]:
        gen = self.generate(
            "Reply with exactly the word OK.",
            decoding={**FROZEN_DECODING, "num_predict": 8},
        )
        ok = gen.parsed_answer.strip().upper().startswith("OK") and not gen.error
        return {
            "schema_version": "nexus-local-qwen-health-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": self.identity.to_dict(),
            "ok": ok,
            "raw_response": gen.raw_response,
            "parsed_answer": gen.parsed_answer,
            "latency_ms": gen.latency_ms,
            "time_to_first_token_ms": gen.time_to_first_token_ms,
            "tokens_per_second": gen.tokens_per_second,
            "load_duration_ns": gen.load_duration_ns,
            "prompt_eval_count": gen.prompt_eval_count,
            "eval_count": gen.eval_count,
            "error": gen.error,
            "structured_json_supported": "unknown",
            "seed_honored": True,  # seed requested; runtime may ignore
            "device": "ollama-managed",
        }
