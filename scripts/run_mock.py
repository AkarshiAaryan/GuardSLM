#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_test_cases
from src.models.mock_guard import MockGuard
from src.baselines.rule_baseline import RuleBaseline
from src.evaluation.runner import run_evaluation_for_model
from src.evaluation.metrics import calculate_overall_metrics
from src.evaluation.pair_metrics import calculate_pair_metrics
from src.evaluation.failure_analysis import analyze_failures
from src.evaluation.decision_gate import generate_decision_gate_report
from src.utils.io import create_run_directory, save_predictions_jsonl, save_predictions_csv, save_json
from src.utils.logging import setup_logger

logger = setup_logger("run_mock")

def main():
    parser = argparse.ArgumentParser(description="Run local CPU mock model evaluation.")
    parser.add_argument("--cases", type=str, default="data/dummy/dummy_cases.json", help="Path to test cases dataset.")
    parser.add_argument("--mode", type=str, default="perfect", choices=["perfect", "context_blind", "always_allow", "always_block"], help="Mock model prediction mode.")
    parser.add_argument("--baseline", action="store_true", help="Also run RuleBaseline.")
    args = parser.parse_args()

    logger.info(f"Loading cases from {args.cases}...")
    cases = load_test_cases(args.cases)

    run_dir = create_run_directory("data/results")
    logger.info(f"Created run output directory: {run_dir}")

    # Evaluate MockGuard
    mock_model = MockGuard(name="mock_guard", config={"mode": args.mode})
    results = run_evaluation_for_model(mock_model, cases)

    # Save predictions
    jsonl_path = os.path.join(run_dir, "predictions.jsonl")
    csv_path = os.path.join(run_dir, "predictions.csv")
    save_predictions_jsonl([r.to_dict() for r in results], jsonl_path)
    save_predictions_csv([r.to_dict() for r in results], csv_path)

    # Calculate metrics
    overall_m = calculate_overall_metrics(results)
    pair_m = calculate_pair_metrics(results)
    failures = analyze_failures(results)

    combined_metrics = {
        "overall": overall_m,
        "pairwise": pair_m,
        "failures": failures
    }
    metrics_path = os.path.join(run_dir, "metrics.json")
    save_json(combined_metrics, metrics_path)

    # Generate Decision Gate report
    report_path = os.path.join(run_dir, "report.md")
    generate_decision_gate_report(overall_m, pair_m, failures, report_path)

    logger.info(f"Mock run complete!")
    logger.info(f"Overall Accuracy: {overall_m['overall_accuracy']*100:.1f}%")
    logger.info(f"Pair Accuracy:    {pair_m['pair_accuracy']*100:.1f}%")
    logger.info(f"Context Flip Rate:{pair_m['context_flip_rate']*100:.1f}%")
    logger.info(f"Results saved to: {run_dir}")

    if args.baseline:
        logger.info("\n--- Running RuleBaseline ---")
        rule_model = RuleBaseline(name="rule_baseline")
        rule_results = run_evaluation_for_model(rule_model, cases)
        rule_run_dir = create_run_directory("data/results")
        save_predictions_jsonl([r.to_dict() for r in rule_results], os.path.join(rule_run_dir, "predictions.jsonl"))
        save_predictions_csv([r.to_dict() for r in rule_results], os.path.join(rule_run_dir, "predictions.csv"))
        rule_overall = calculate_overall_metrics(rule_results)
        rule_pair = calculate_pair_metrics(rule_results)
        rule_fail = analyze_failures(rule_results)
        save_json({"overall": rule_overall, "pairwise": rule_pair, "failures": rule_fail}, os.path.join(rule_run_dir, "metrics.json"))
        generate_decision_gate_report(rule_overall, rule_pair, rule_fail, os.path.join(rule_run_dir, "report.md"))
        logger.info(f"RuleBaseline Accuracy: {rule_overall['overall_accuracy']*100:.1f}% | Pair Accuracy: {rule_pair['pair_accuracy']*100:.1f}%")

if __name__ == "__main__":
    main()
