#!/usr/bin/env python3
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.decision_gate import generate_decision_gate_report
from src.utils.io import load_json
from src.utils.logging import setup_logger

logger = setup_logger("generate_report")

def main():
    parser = argparse.ArgumentParser(description="Generate Decision Gate report from metrics.")
    parser.add_argument("--metrics", type=str, help="Path to metrics.json file.")
    parser.add_argument("--results_dir", type=str, help="Path to run directory containing metrics.json.")
    args = parser.parse_args()

    metrics_file = args.metrics
    if not metrics_file and args.results_dir:
        metrics_file = os.path.join(args.results_dir, "metrics.json")

    if not metrics_file or not os.path.exists(metrics_file):
        results_root = "data/results"
        if os.path.exists(results_root):
            subdirs = [os.path.join(results_root, d) for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
            if subdirs:
                latest_dir = max(subdirs, key=os.path.getmtime)
                candidate = os.path.join(latest_dir, "metrics.json")
                if os.path.exists(candidate):
                    metrics_file = candidate

    if not metrics_file or not os.path.exists(metrics_file):
        logger.error("No valid metrics.json file found to generate report.")
        sys.exit(1)

    logger.info(f"Loading metrics from: {metrics_file}")
    data = load_json(metrics_file)

    run_dir = os.path.dirname(metrics_file)
    output_report_path = os.path.join(run_dir, "report.md")

    overall_m = data.get("overall", {})
    pair_m = data.get("pairwise", {})
    failures = data.get("failures", {})

    report_content = generate_decision_gate_report(overall_m, pair_m, failures, output_report_path)

    # Also copy to reports/decision_gate_report.md
    global_report_path = os.path.join("reports", "decision_gate_report.md")
    os.makedirs("reports", exist_ok=True)
    with open(global_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Report generated successfully!")
    logger.info(f"  - Run report: {output_report_path}")
    logger.info(f"  - Latest report: {global_report_path}")

if __name__ == "__main__":
    main()
