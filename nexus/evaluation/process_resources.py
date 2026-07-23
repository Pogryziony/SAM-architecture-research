"""Full-system process-tree RSS / VRAM helpers for evaluation campaigns."""

from __future__ import annotations

import platform
import subprocess
from typing import Any


def process_tree_rss_mb(pid: int | None = None) -> float | None:
    """Sum RSS of a process and its children (MB). Best-effort."""
    try:
        import psutil  # type: ignore

        proc = psutil.Process(pid) if pid is not None else psutil.Process()
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total / (1024 * 1024)
    except Exception:
        return None


def ollama_rss_mb() -> float | None:
    """RSS of local Ollama server process(es), if running."""
    try:
        import psutil  # type: ignore

        total = 0
        found = False
        for proc in psutil.process_iter(["name", "memory_info"]):
            name = (proc.info.get("name") or "").casefold()
            if "ollama" in name:
                found = True
                mi = proc.info.get("memory_info")
                if mi is not None:
                    total += mi.rss
        return None if not found else total / (1024 * 1024)
    except Exception:
        return None


def nvidia_vram_used_mb() -> dict[str, Any]:
    """Query nvidia-smi for used VRAM; empty when unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        return {
            "available": False,
            "gpus": [],
            "reason": f"{type(exc).__name__}: {exc}",
            "host": platform.system(),
        }
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": float(parts[2]),
                "memory_total_mb": float(parts[3]),
            }
        )
    return {"available": bool(gpus), "gpus": gpus, "host": platform.system()}


def snapshot_llm_server_resources() -> dict[str, Any]:
    return {
        "schema_version": "nexus-llm-server-resources-v1",
        "ollama_rss_mb": ollama_rss_mb(),
        "process_tree_rss_mb": process_tree_rss_mb(),
        "vram": nvidia_vram_used_mb(),
    }
