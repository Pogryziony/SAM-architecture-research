"""Lightweight logging + JSON metric persistence.

We avoid heavyweight experiment trackers on purpose. Every run writes:
  - a human-readable .log (console mirror)
  - metrics.jsonl (one JSON object per logged step)
  - metrics.json  (final summary, written by the trainer)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional


def get_logger(name: str = "sam", logfile: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class MetricLogger:
    """Append step metrics to metrics.jsonl and keep a running summary."""

    def __init__(self, run_dir: str, run_name: str = "run"):
        self.run_dir = run_dir
        self.run_name = run_name
        os.makedirs(run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(run_dir, "metrics.jsonl")
        self.summary_path = os.path.join(run_dir, "metrics.json")
        self.logger = get_logger(run_name, os.path.join(run_dir, "run.log"))
        self.start_time = time.time()
        # fresh file
        open(self.jsonl_path, "w").close()

    def log(self, step: int, metrics: Dict[str, Any], stdout: bool = True) -> None:
        record = {"step": int(step), "wall_s": round(time.time() - self.start_time, 2)}
        record.update(metrics)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if stdout:
            kv = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            self.logger.info("step %d | %s", step, kv)

    def save_summary(self, summary: Dict[str, Any]) -> str:
        summary = dict(summary)
        summary.setdefault("run_name", self.run_name)
        summary.setdefault("total_wall_s", round(time.time() - self.start_time, 2))
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self.logger.info("wrote summary -> %s", self.summary_path)
        return self.summary_path
