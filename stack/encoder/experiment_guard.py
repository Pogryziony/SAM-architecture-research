"""Lightweight provenance guards shared by optional-PyTorch experiments."""
from __future__ import annotations

import subprocess
from pathlib import Path


def check_worktree_clean(root: str | Path = ".") -> bool:
    """Reject tracked, staged, and untracked changes before gated runs."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=Path(root),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and not result.stdout.strip()
