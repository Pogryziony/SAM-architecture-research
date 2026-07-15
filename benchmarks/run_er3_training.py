"""Entity Ranker V3 training CLI wrapper with preset support.

Provides a clean CLI that delegates to stack.encoder.train_ranker_v3
after applying training presets.

Usage:
    python benchmarks/run_er3_training.py --preset smoke
    python benchmarks/run_er3_training.py --preset quick --epochs 10 --lr 0.0005
    python benchmarks/run_er3_training.py --preset full
    python benchmarks/run_er3_training.py --list-presets
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from stack.encoder.training_presets import (
    apply_preset,
    get_preset,
    list_presets,
    DEFAULT_PRESETS_PATH,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Entity Ranker V3 training with preset intensity levels",
    )
    parser.add_argument(
        "--preset", default="pilot",
        choices=list_presets(),
        help="Training intensity preset (smoke/quick/pilot/standard/full)",
    )
    parser.add_argument(
        "--list-presets", action="store_true",
        help="List available presets and exit",
    )
    # Override flags
    parser.add_argument("--epochs", type=int, default=None, help="Override preset epochs")
    parser.add_argument("--patience", type=int, default=None, help="Override preset patience")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--presets-path", default=None, help="Custom presets JSON path")
    parser.add_argument("--seed", type=int, default=20260710, help="Random seed")

    args = parser.parse_args()

    if args.list_presets:
        presets = list_presets(args.presets_path)
        print(f"Available presets ({len(presets)}):")
        for name in presets:
            p = get_preset(name, args.presets_path)
            note = p.pop("note", "")
            print(f"  {name:12s} epochs={p['epochs']:>3d}  patience={p.get('patience', '?'):>3d}  batch={p.get('batch_size', '?'):>3d}")
            if note:
                print(f"               {note}")
        return 0

    # Apply preset
    cli_overrides = {
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    # Filter out None values
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

    params = apply_preset(
        args.preset,
        model_type="er3",
        cli_overrides=cli_overrides,
        path=args.presets_path,
    )

    preset_info = get_preset(args.preset, args.presets_path)
    note = preset_info.pop("note", "")
    print(f"Preset: {args.preset} ({note})" if note else f"Preset: {args.preset}")
    print(f"Resolved params: epochs={params['epochs']}, patience={params.get('patience', 'auto')}, "
          f"lr={params.get('learning_rate', 'default')}, batch={params.get('batch_size', 'default')}")
    print(f"CLI overrides: {cli_overrides or 'none'}")

    # Delegate to the actual training module
    from stack.encoder.train_ranker_v3 import run_experiment_v3, SEED, K_MAX

    # Monkey-patch SEED if specified
    if args.seed != 20260710:
        import stack.encoder.train_ranker_v3 as tmod
        tmod.SEED = args.seed

    result = run_experiment_v3()
    print(json.dumps({
        "preset": args.preset,
        "winner": result["winner"],
        "run_id": result["run_id"],
        "source_sha": result["source_sha"],
        "proceed_to_frozen": result["proceed_to_frozen"],
    }, indent=2))
    print("\nMetrics:")
    for name, m in result["metrics"].items():
        print(f"  {name}: r@1={m.get('recall@1', 0):.4f} r@5={m.get('recall@5', 0):.4f} "
              f"r@10={m.get('recall@10', 0):.4f} p@10={m.get('precision@10', 0):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
