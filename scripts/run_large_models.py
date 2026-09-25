#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_test_cases
from src.models.large_model import LargeModelJudge
from src.evaluation.runner import run_evaluation_for_model
from src.evaluation.metrics import calculate_overall_metrics
from src.evaluation.pair_metrics import calculate_pair_metrics
from src.evaluation.failure_analysis import analyze_failures
from src.evaluation.decision_gate import generate_decision_gate_report
from src.utils.io import load_yaml, create_run_directory, save_predictions_jsonl, save_predictions_csv, save_json
from src.utils.logging import setup_logger

logger = setup_logger("run_large_models")

def main():
    parser = argparse.ArgumentParser(description="Run commercial LLM Judge evaluation.")
    parser.add_argument("--cases", type=str, default="data/dummy/dummy_cases.json", help="Path to test cases dataset.")
    parser.add_argument("--config", type=str, default="config/models.yaml", help="Path to models config YAML.")
    args = parser.parse_args()

    models_config = load_yaml(args.config).get("models", {}).get("large_model_judge", {})
    cases = load_test_cases(args.cases)

    judge = LargeModelJudge(name="large_model_judge", config=models_config)
    try:
        results = run_evaluation_for_model(judge, cases)
    except (NotImplementedError, ValueError) as e:
        logger.warning(f"LargeModelJudge skipped: {e}")
        sys.exit(0)

    run_dir = create_run_directory("data/results")
    save_predictions_jsonl([r.to_dict() for r in results], os.path.join(run_dir, "predictions.jsonl"))
    save_predictions_csv([r.to_dict() for r in results], os.path.join(run_dir, "predictions.csv"))

    overall_m = calculate_overall_metrics(results)
    pair_m = calculate_pair_metrics(results)
    failures = analyze_failures(results)

    save_json({"overall": overall_m, "pairwise": pair_m, "failures": failures}, os.path.join(run_dir, "metrics.json"))
    generate_decision_gate_report(overall_m, pair_m, failures, os.path.join(run_dir, "report.md"))

    logger.info(f"LargeModelJudge evaluation complete. Output: {run_dir}")

if __name__ == "__main__":
    main()
