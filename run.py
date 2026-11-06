#!/usr/bin/env python3
"""CLI entry point for GuardSLM Benchmark Suite."""

import argparse
import os
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from guardslm.data import load_ground_truth, load_test_cases, validate_cases_against_ground_truth
from guardslm.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="GuardSLM: Minimal, Reproducible Benchmarking Framework for Web Agent Authorization Guardrails."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock_guard",
        help="Model to benchmark ('mock_guard', 'policyguard', 'dynaguard'). Default: mock_guard."
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run benchmarks sequentially for all enabled models."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test cases evaluated (e.g., 20 or 100). Default: all."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models.yaml",
        help="Path to YAML configuration file. Default: configs/models.yaml."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom destination directory for experiment results."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic resumption from existing predictions.jsonl."
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate test cases and ground-truth pairing without running inference."
    )

    args = parser.parse_args()

    # Fallback config path if configs/ does not exist but config/ does
    config_path = args.config
    if not os.path.isfile(config_path) and os.path.isfile("config/models.yaml"):
        config_path = "config/models.yaml"

    if args.validate_data:
        print("[DATA VALIDATION] Inspecting test cases and isolated ground truth...")
        cases = load_test_cases()
        gt = load_ground_truth()
        validate_cases_against_ground_truth(cases, gt)
        print(f"  [SUCCESS] All {len(cases)} test cases are valid and match ground truth annotations 1-to-1.")
        sys.exit(0)

    models_to_run = []
    if args.all_models:
        models_to_run = ["mock_guard", "policyguard", "dynaguard"]
    else:
        models_to_run = [args.model]

    for m in models_to_run:
        print(f"\n{'='*70}\nStarting GuardSLM Benchmark Run: {m}\n{'='*70}")
        run_experiment(
            model_name=m,
            config_path=config_path,
            limit=args.limit,
            output_dir=args.output_dir,
            resume=not args.no_resume
        )


if __name__ == "__main__":
    main()
