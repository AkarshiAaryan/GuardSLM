#!/usr/bin/env python3
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.result_schema import CaseResult
from src.evaluation.metrics import calculate_overall_metrics
from src.evaluation.pair_metrics import calculate_pair_metrics
from src.evaluation.failure_analysis import analyze_failures
from src.utils.io import save_json
from src.utils.logging import setup_logger

logger = setup_logger("calculate_metrics")

def main():
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics from prediction files.")
    parser.add_argument("--predictions", type=str, help="Path to predictions.jsonl file.")
    parser.add_argument("--results_dir", type=str, help="Path to run directory containing predictions.jsonl.")
    args = parser.parse_args()

    pred_file = args.predictions
    if not pred_file and args.results_dir:
        pred_file = os.path.join(args.results_dir, "predictions.jsonl")

    if not pred_file or not os.path.exists(pred_file):
        # Fallback: search for most recent run directory in data/results/
        results_root = "data/results"
        if os.path.exists(results_root):
            subdirs = [os.path.join(results_root, d) for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
            if subdirs:
                latest_dir = max(subdirs, key=os.path.getmtime)
                candidate = os.path.join(latest_dir, "predictions.jsonl")
                if os.path.exists(candidate):
                    pred_file = candidate

    if not pred_file or not os.path.exists(pred_file):
        logger.error("No valid predictions.jsonl file found to calculate metrics.")
        sys.exit(1)

    logger.info(f"Loading predictions from: {pred_file}")
    results = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                res = CaseResult(
                    case_id=d["case_id"],
                    pair_id=d["pair_id"],
                    template_id=d["template_id"],
                    model=d["model"],
                    gold_verdict=d["gold_verdict"],
                    predicted_verdict=d["predicted_verdict"],
                    correct=d["correct"],
                    raw_response=d.get("raw_response", ""),
                    latency_ms=d.get("latency_ms", 0.0),
                    parse_status=d.get("parse_status", "success"),
                    timestamp=d.get("timestamp", ""),
                    error=d.get("error"),
                    template_name=d.get("template_name", "")
                )
                results.append(res)

    overall_m = calculate_overall_metrics(results)
    pair_m = calculate_pair_metrics(results)
    failures = analyze_failures(results)

    print("\n" + "="*60)
    print("SLM SAFETY GUARD PILOT - EVALUATION METRICS SUMMARY")
    print("="*60)
    print(f"Model:                 {overall_m['model']}")
    print(f"Total Test Cases:      {overall_m['total_cases']}")
    print(f"Overall Accuracy:      {overall_m['overall_accuracy']*100:.1f}% ({overall_m['correct_cases']}/{overall_m['total_cases']})")
    print(f"Total Matched Pairs:   {pair_m['total_pairs']}")
    print(f"Pair Accuracy:         {pair_m['pair_accuracy']*100:.1f}% ({pair_m['pair_correct_count']}/{pair_m['total_pairs']})")
    print(f"Context Flip Rate:     {pair_m['context_flip_rate']*100:.1f}% ({pair_m['correct_context_flips']}/{pair_m['applicable_context_flip_pairs']})")
    print(f"Average Latency:       {overall_m['average_latency_ms']} ms")
    print("-" * 60)
    print("PER-TEMPLATE ACCURACY:")
    for tid, tinfo in overall_m["per_template_accuracy"].items():
        print(f"  [{tid}] {tinfo['template_name'][:35]:<35} | {tinfo['accuracy']*100:5.1f}% ({tinfo['correct']}/{tinfo['total']})")
    print("="*60 + "\n")

    run_dir = os.path.dirname(pred_file)
    save_json({"overall": overall_m, "pairwise": pair_m, "failures": failures}, os.path.join(run_dir, "metrics.json"))
    logger.info(f"Updated metrics saved to: {os.path.join(run_dir, 'metrics.json')}")

if __name__ == "__main__":
    main()
