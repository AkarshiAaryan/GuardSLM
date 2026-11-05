"""GuardSLM v3 Text-Only Benchmark Experiment Runner with Resumption and Duplicate Prevention."""

import argparse
import datetime
import json
import os
import sys
from typing import Dict, Any, List, Optional
from ..data.loader import BenchmarkLoader
from ..data.schema import ModelInputCase
from ..models.base import BenchmarkModel, BenchmarkPrediction
from ..models.mock_benchmark_guard import MockBenchmarkGuard
from ..parser.output_parser import ParsedBenchmarkOutput
from ..evaluation.evaluator import BenchmarkEvaluator


def build_model_instance(
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    oracle_golds: Optional[Dict[str, Any]] = None
) -> BenchmarkModel:
    """Instantiates a model adapter based on model name."""
    cfg = config or {}
    if model_name.startswith("mock") or model_name == "mock_guard":
        cfg["oracle_golds"] = oracle_golds or {}
        return MockBenchmarkGuard(name=model_name, config=cfg)
    elif "policyguard" in model_name:
        from ..models.policyguard_benchmark_guard import PolicyGuardBenchmarkGuard
        return PolicyGuardBenchmarkGuard(name=model_name, config=cfg)
    elif "dynaguard" in model_name:
        from ..models.dynaguard_benchmark_guard import DynaGuardBenchmarkGuard
        return DynaGuardBenchmarkGuard(name=model_name, config=cfg)
    elif "webguard" in model_name:
        from ..models.webguard_benchmark_guard import WebGuardBenchmarkGuard
        return WebGuardBenchmarkGuard(name=model_name, config=cfg)
    elif "qwen" in model_name or "hf" in model_name:
        from ..models.hf_benchmark_guard import HFBenchmarkGuard
        return HFBenchmarkGuard(name=model_name, config=cfg)
    else:
        # Fallback to mock
        cfg["oracle_golds"] = oracle_golds or {}
        return MockBenchmarkGuard(name=model_name, config=cfg)


def run_benchmark_experiment(
    dataset_path: str = "research_snapshots",
    model_name: str = "mock_guard",
    model_config: Optional[Dict[str, Any]] = None,
    strategy: str = "text_only",
    case_ids: Optional[List[str]] = None,
    limit: Optional[int] = 20,
    output_dir: Optional[str] = None,
    resume: bool = True
) -> Dict[str, Any]:
    """Runs the benchmark pipeline for text-only cases with checkpoint resumption."""
    is_dir = os.path.isdir(dataset_path)
    loader = BenchmarkLoader(dataset_path=dataset_path, validate_on_load=(not is_dir))
    cases = loader.get_model_inputs(case_ids=case_ids, limit=limit)
    golds = loader.get_gold_annotations()


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{timestamp}_{model_name}_{strategy}"
    base_out_dir = output_dir or os.path.join("data/benchmark_results", exp_id)
    preds_dir = os.path.join(base_out_dir, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    # Save configuration snapshot
    config_snapshot = {
        "experiment_id": exp_id,
        "dataset_path": os.path.abspath(dataset_path),
        "model_name": model_name,
        "model_config": model_config or {},
        "strategy": strategy,
        "limit": limit,
        "total_cases_selected": len(cases),
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(os.path.join(base_out_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)

    model = build_model_instance(model_name, model_config, oracle_golds=golds)
    model.load()

    predictions: List[BenchmarkPrediction] = []

    try:
        print(f"\n--- Running Model: {model_name} (Strategy: {strategy}) on {len(cases)} cases ---")
        for idx, case in enumerate(cases, 1):
            case_pred_file = os.path.join(preds_dir, f"{case.case_id}_{strategy}.json")

            # Resumption check: avoid duplicate execution
            if resume and os.path.isfile(case_pred_file) and os.path.getsize(case_pred_file) > 0:
                try:
                    with open(case_pred_file, "r", encoding="utf-8") as f:
                        saved_dict = json.load(f)
                    parsed_dict = saved_dict["parsed"]
                    parsed_obj = ParsedBenchmarkOutput(**parsed_dict)
                    pred = BenchmarkPrediction(
                        case_id=saved_dict["case_id"],
                        strategy=saved_dict["strategy"],
                        parsed=parsed_obj,
                        raw_response=saved_dict.get("raw_response", ""),
                        latency_ms=saved_dict.get("latency_ms", 0.0),
                        input_tokens=saved_dict.get("input_tokens"),
                        output_tokens=saved_dict.get("output_tokens"),
                        retries=saved_dict.get("retries", 0),
                        error=saved_dict.get("error"),
                        model_name=saved_dict.get("model_name", model_name)
                    )
                    predictions.append(pred)
                    continue
                except Exception:
                    pass  # If corrupt, recompute

            # Predict pure text
            pred = model.predict(case, strategy=strategy, screenshot_path=None)
            with open(case_pred_file, "w", encoding="utf-8") as f:
                json.dump(pred.to_dict(), f, indent=2)
            predictions.append(pred)

            if idx % 5 == 0 or idx == len(cases):
                print(f"  [{strategy}] Completed {idx}/{len(cases)} cases (latest: {case.case_id}, {pred.latency_ms:.0f}ms, verdict: {pred.parsed.verdict}).", flush=True)
    finally:
        model.unload()

    # Post-inference evaluation
    evaluator = BenchmarkEvaluator(loader)
    eval_res = evaluator.evaluate_predictions(predictions, strategy=strategy, model_name=model_name)
    report_file = os.path.join(base_out_dir, f"report_{strategy}.md")
    evaluator.generate_markdown_report(eval_res, report_file)
    print(f"\nReport generated: {report_file}")

    consolidated_results = {strategy: eval_res}
    with open(os.path.join(base_out_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(consolidated_results, f, indent=2)

    return consolidated_results


def main():
    parser = argparse.ArgumentParser(description="GuardSLM LMS Snapshot Benchmark Experiment Runner.")
    parser.add_argument("--dataset", type=str, default="research_snapshots")
    parser.add_argument("--model", type=str, default="mock_guard")
    parser.add_argument("--strategy", type=str, default="text_only", choices=["text_only"])
    parser.add_argument("--case-ids", type=str, default=None, help="Comma-separated case IDs to run.")
    parser.add_argument("--limit", type=int, default=20, help="Limit number of cases (default: 20).")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resumption.")
    args = parser.parse_args()

    cids = [c.strip() for c in args.case_ids.split(",")] if args.case_ids else None
    run_benchmark_experiment(
        dataset_path=args.dataset,
        model_name=args.model,
        strategy=args.strategy,
        case_ids=cids,
        limit=args.limit,
        output_dir=args.output_dir,
        resume=(not args.no_resume)
    )


if __name__ == "__main__":
    main()
