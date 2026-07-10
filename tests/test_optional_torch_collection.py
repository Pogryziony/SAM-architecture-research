from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_collection_and_nexus_tests_work_without_torch():
    root = Path(__file__).resolve().parent.parent
    script = (
        "import builtins,sys; "
        "real=builtins.__import__; "
        "builtins.__import__=lambda name,*a,**k: (_ for _ in ()).throw(ImportError('torch unavailable')) "
        "if name == 'torch' else real(name,*a,**k); "
        "import pytest; from pathlib import Path; "
        "paths=[str(p) for p in Path('tests').glob('test_nexus_*.py')]; "
        "raise SystemExit(pytest.main(paths+['-q']))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    completed = subprocess.run([sys.executable, "-c", script], cwd=root, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
