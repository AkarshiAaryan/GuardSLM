#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_test_cases
from src.models.mock_guard import MockGuard
from src.models.webguard import WebGuardAdapter
from src.models.llama_guard import LlamaGuardAdapter
from src.models.qwen_guard import QwenGuardAdapter
from src.models.dynaguard import DynaGuardAdapter
from src.models.policyguard import PolicyGuardAdapter
from src.evaluation.runner import run_evaluation_for_model
from src.evaluation.metrics import calculate_overall_metrics
from src.evaluation.pair_metrics import calculate_pair_metrics
from src.evaluation.failure_analysis import analyze_failures
from src.evaluation.decision_gate import generate_decision_gate_report
from src.utils.io import load_yaml, create_run_directory, save_predictions_jsonl, save_predictions_csv, save_json
from src.utils.logging import setup_logger

logger = setup_logger("run_guards")

MODEL_MAP = {
    "mock_guard": MockGuard,
    "webguard": WebGuardAdapter,
    "llama_guard": LlamaGuardAdapter,
    "qwen_guard": QwenGuardAdapter,
    "dynaguard": DynaGuardAdapter,
    "policyguard": PolicyGuardAdapter
}

def main():
    parser = argparse.ArgumentParser(description="Run SLM safety guard evaluation.")
    parser.add_argument("--cases", type=str, default="data/dummy/dummy_cases.json", help="Path to test cases dataset.")
    parser.add_argument("--config", type=str, default="config/models.yaml", help="Path to models config YAML.")
    parser.add_argument("--models", nargs="+", default=["mock_guard"], help="List of model names to run.")
    args = parser.parse_args()

    models_config = load_yaml(args.config).get("models", {})
    cases = load_test_cases(args.cases)

    for m_name in args.models:
        if m_name not in MODEL_MAP:
            logger.error(f"Unknown model name '{m_name}'. Available: {list(MODEL_MAP.keys())}")
            continue

        m_cfg = models_config.get(m_name, {})
        model_cls = MODEL_MAP[m_name]
        model_instance = model_cls(name=m_name, config=m_cfg)

        logger.info(f"Preparing model '{m_name}'...")
        try:
            results = run_evaluation_for_model(model_instance, cases)
        except (NotImplementedError, RuntimeError, ValueError) as e:
            logger.warning(f"Skipping model '{m_name}': {e}")
            continue

        run_dir = create_run_directory("data/results")
        save_predictions_jsonl([r.to_dict() for r in results], os.path.join(run_dir, "predictions.jsonl"))
        save_predictions_csv([r.to_dict() for r in results], os.path.join(run_dir, "predictions.csv"))

        overall_m = calculate_overall_metrics(results)
        pair_m = calculate_pair_metrics(results)
        failures = analyze_failures(results)

        save_json({"overall": overall_m, "pairwise": pair_m, "failures": failures}, os.path.join(run_dir, "metrics.json"))
        generate_decision_gate_report(overall_m, pair_m, failures, os.path.join(run_dir, "report.md"))

        logger.info(f"Model '{m_name}' evaluation complete. Output: {run_dir}")

if __name__ == "__main__":
    main()
