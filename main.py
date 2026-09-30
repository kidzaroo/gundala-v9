"""CLI entry point.

Example::

    python main.py --benign data/benign.json --tunnel data/tunnel.json \
        --config config.yaml --output outputs/experiment_01
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.config import (
    SUPPORTED_BALANCING,
    SUPPORTED_HISTORY_POLICIES,
    SUPPORTED_MODELS,
    SUPPORTED_SPLITS,
    apply_overrides,
    load_config,
    validate_config,
)
from src.experiment import run_experiments


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DNS tunneling detection: per-event baseline vs. sliding-window features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benign", help="Path to benign.json (label 0).")
    p.add_argument("--tunnel", help="Path to tunnel.json (label 1).")
    p.add_argument("--config", default=None, help="YAML config (defaults are built in).")
    p.add_argument("--output", help="Output directory (must be new/empty unless --overwrite).")
    p.add_argument("--input-format", choices=["auto", "jsonl", "json_array"], help="Override format detection.")
    p.add_argument("--split-strategy", choices=SUPPORTED_SPLITS, help="Default: stratified_random.")
    p.add_argument("--test-size", type=float, help="Default: 0.30.")
    p.add_argument("--random-seed", type=int, help="Default: 42.")
    p.add_argument("--split-manifest", help="Reuse an existing split_manifest.csv.")
    p.add_argument("--balancing", choices=SUPPORTED_BALANCING, help="Default: random_oversampling.")
    p.add_argument("--sampling-strategy", help="auto | minority | 'not majority' | float ratio (e.g. 0.5).")
    p.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, help="Classifiers to run.")
    p.add_argument("--windows", nargs="*", type=float,
                   help="Window sizes in seconds (e.g. 5 10 15 30 60). Give no values to run baseline only.")
    p.add_argument("--skip-baseline", action="store_true", help="Do not run the no-window baseline.")
    p.add_argument("--window-history-policy", choices=SUPPORTED_HISTORY_POLICIES,
                   help="split_isolated (main) | train_carryover (separate add-on, chronological only).")
    p.add_argument("--n-jobs", type=int, help="Workers/threads for windows and models (-1 = all cores).")
    p.add_argument("--synthetic", action="store_true",
                   help="Mark input as SYNTHETIC smoke-test data (results are NOT research results).")
    p.add_argument("--overwrite", action="store_true", help="Delete a non-empty output directory first.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    overrides = {
        "input.benign_path": args.benign,
        "input.tunnel_path": args.tunnel,
        "input.format": args.input_format,
        "output_dir": args.output,
        "split.strategy": args.split_strategy,
        "split.test_size": args.test_size,
        "split.manifest_path": args.split_manifest,
        "random_seed": args.random_seed,
        "balancing.method": args.balancing,
        "balancing.sampling_strategy": args.sampling_strategy,
        "models.enabled": args.models,
        "windows.sizes": None if args.windows is None else list(args.windows),
        "windows.history_policy": args.window_history_policy,
        "runtime.n_jobs": args.n_jobs,
        "run_baseline": False if args.skip_baseline else None,
        "data_origin": "synthetic_smoke_test" if args.synthetic else None,
    }
    apply_overrides(cfg, overrides)
    try:
        validate_config(cfg)
        results = run_experiments(cfg, overwrite=args.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    show = ["experiment_id", "f1", "precision", "recall", "false_positive_rate", "roc_auc",
            "average_precision", "training_seconds", "inference_ms_per_event"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print("\n" + results[show].to_string(index=False))
    print(f"\nArtefacts written to: {cfg['output_dir']}")
    if cfg["data_origin"] != "unspecified":
        print(f"NOTE: data_origin={cfg['data_origin']}. Synthetic/smoke-test numbers are NOT research results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
