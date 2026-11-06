"""Unified experiment runner orchestrating the 9-step GuardSLM benchmark pipeline."""

import csv
import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional

import yaml

from .data import load_ground_truth, load_test_cases
from .evaluation import evaluate_predictions, generate_markdown_report
from .models import get_model
from .parser import parse_response
from .prompts import build_prompt


def load_yaml_config(config_path: str = "configs/models.yaml") -> Dict[str, Any]:
    """Loads configuration file with fallback defaults."""
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_experiment(
    model_name: str = "mock_guard",
    config_path: str = "configs/models.yaml",
    limit: Optional[int] = None,
    output_dir: Optional[str] = None,
    resume: bool = True,
    config_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Executes the complete 9-step benchmarking pipeline.

    1. Load config
    2. Load sanitized test cases (slm_inputs)
    3. Load ground-truth annotations (evaluation_only)
    4. Initialize model adapter
    5. Prepare output directory and resume cache
    6. Load model
    7. Inference loop with streaming predictions
    8. Unload model
    9. Evaluate results and write reports
    """
    start_total_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Config loading
    cfg = load_yaml_config(config_path)
    if config_override:
        cfg.update(config_override)

    input_dir = cfg.get("data", {}).get("slm_inputs_dir", "research_data/slm_inputs")
    gt_path = cfg.get("data", {}).get("ground_truth_path", "research_data/evaluation_only/ground_truth.csv")
    model_cfg = cfg.get("models", {}).get(model_name, {})

    # 2. Load test cases
    print(f"\n[1/7] Loading test cases from '{input_dir}'...")
    all_cases = load_test_cases(input_dir=input_dir)
    if limit is not None and limit > 0:
        all_cases = all_cases[:limit]
    print(f"      Loaded {len(all_cases)} test case(s).")

    # 3. Load isolated ground-truth
    print(f"[2/7] Loading isolated ground truth from '{gt_path}'...")
    ground_truth = load_ground_truth(gt_path=gt_path)
    print(f"      Loaded {len(ground_truth)} gold ground-truth label(s).")

    # 4. Initialize model
    print(f"[3/7] Initializing model adapter '{model_name}'...")
    # Inject oracle golds into mock config if applicable for verification testing
    if "mock" in model_name.lower() and model_cfg.get("mode") in ("perfect", "oracle"):
        model_cfg["oracle_golds"] = ground_truth

    model = get_model(model_name=model_name, config=model_cfg)

    # 5. Output directory & resume setup
    if not output_dir:
        output_dir = os.path.join("results", f"experiment_{timestamp}_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "predictions.jsonl")
    existing_predictions: Dict[str, Dict[str, Any]] = {}

    if resume and os.path.isfile(jsonl_path):
        print(f"      Checking existing results for resumption in '{jsonl_path}'...")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        cid = record.get("case_id")
                        if cid and record.get("model_name") == model.name:
                            existing_predictions[cid] = record
                    except json.JSONDecodeError:
                        continue
        print(f"      Found {len(existing_predictions)} cached predictions.")

    # 6. Load model
    print(f"[4/7] Loading model '{model.name}' weights/resources...")
    model.load()

    # 7. Inference loop
    print(f"[5/7] Executing inference on {len(all_cases)} case(s)...")
    predictions: List[Dict[str, Any]] = []

    # Open predictions.jsonl in append mode for streaming output
    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
        for idx, case in enumerate(all_cases, start=1):
            if case.case_id in existing_predictions:
                cached = existing_predictions[case.case_id]
                predictions.append(cached)
                print(f"  [{idx}/{len(all_cases)}] {case.case_id}: CACHED (verdict: {cached.get('verdict')})")
                continue

            prompt = build_prompt(model_name=model.name, case=case)
            raw_response, latency_ms, error = model.predict(prompt=prompt)
            parsed = parse_response(model_name=model.name, raw_response=raw_response)

            record = {
                "case_id": case.case_id,
                "title": case.title,
                "model_name": model.name,
                "prompt": prompt,
                "raw_response": raw_response,
                "verdict": parsed["verdict"],
                "native_label": parsed["native_label"],
                "parse_status": parsed["parse_status"],
                "parsing_error": parsed["parsing_error"],
                "latency_ms": latency_ms,
                "error": error,
            }

            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()
            predictions.append(record)

            status_str = f"verdict: {parsed['verdict']}"
            if error:
                status_str += f", error: {error}"
            print(f"  [{idx}/{len(all_cases)}] {case.case_id} ({latency_ms:.1f}ms): {status_str}")

    # 8. Unload model
    print(f"[6/7] Unloading model '{model.name}'...")
    model.unload()

    # 9. Evaluate & generate reports
    print(f"[7/7] Evaluating predictions and generating reports...")
    eval_results = evaluate_predictions(predictions=predictions, ground_truth=ground_truth)

    run_metadata = {
        "timestamp": timestamp,
        "model_name": model.name,
        "total_cases_evaluated": len(predictions),
        "limit": limit,
        "output_dir": output_dir,
        "total_runtime_seconds": round(time.time() - start_total_time, 2),
        "model_config": model_cfg,
    }

    # Write evaluation.json
    eval_json_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # Write run_metadata.json
    metadata_json_path = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    # Write report.md
    report_md = generate_markdown_report(eval_results=eval_results, run_metadata=run_metadata)
    report_md_path = os.path.join(output_dir, "report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # Write predictions.csv
    csv_path = os.path.join(output_dir, "predictions.csv")
    csv_headers = [
        "case_id", "title", "model_name", "verdict", "ground_truth",
        "correct", "native_label", "parse_status", "latency_ms", "error"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for p in predictions:
            cid = p.get("case_id", "")
            gt_val = ground_truth.get(cid, "UNKNOWN")
            verdict = p.get("verdict", "UNKNOWN")
            is_correct = (verdict == gt_val)
            writer.writerow({
                "case_id": cid,
                "title": p.get("title", ""),
                "model_name": p.get("model_name", ""),
                "verdict": verdict,
                "ground_truth": gt_val,
                "correct": is_correct,
                "native_label": p.get("native_label", ""),
                "parse_status": p.get("parse_status", ""),
                "latency_ms": p.get("latency_ms", 0.0),
                "error": p.get("error", "")
            })

    print(f"\n[DONE] Experiment finished successfully!")
    print(f"       Results saved to: {output_dir}")
    print(f"       Accuracy: {eval_results.get('accuracy', 0.0)*100:.2f}% | Macro F1: {eval_results.get('macro_f1', 0.0):.4f}")

    return {
        "output_dir": output_dir,
        "evaluation": eval_results,
        "metadata": run_metadata,
        "predictions_count": len(predictions),
        "report_path": report_md_path,
        "jsonl_path": jsonl_path,
        "csv_path": csv_path
    }
