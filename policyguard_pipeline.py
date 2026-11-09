"""PolicyGuard-4B Minimal Inference Pipeline.

This pipeline executes inference using the official pretrained checkpoint:
Rakancorle1/PolicyGuard-4B on preformatted .txt input files.

Pipeline steps:
1. Load PolicyGuard-4B model and tokenizer.
2. Read 100 input files containing preformatted prompts.
3. Send each prompt to the model with greedy decoding.
4. Record model responses as-is without any normalization.
5. Save streaming records immediately to predictions.jsonl and generate predictions.csv and summary.json.
6. Evaluate against ground-truth answers if provided.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def natural_sort_key(s: str) -> List[object]:
    """Generates a key for natural numerical sorting (e.g., TC1, TC2, TC10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_model(
    model_id: str = "Rakancorle1/PolicyGuard-4B",
    device: Optional[str] = None
) -> Tuple[Any, Any]:
    """Loads the pretrained PolicyGuard-4B tokenizer and causal language model.

    Args:
        model_id: Hugging Face checkpoint name.
        device: Target device ("cuda", "cpu", or None for auto).

    Returns:
        Tuple of (tokenizer, model).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model '{model_id}'...")
    if device is not None:
        torch_dtype = torch.float16 if "cuda" in device else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype
        ).to(device)
    elif torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32
        )

    model.eval()
    selected_device = getattr(model, "device", device or "cpu")
    print(f"Model successfully loaded on device: {selected_device}")
    return tokenizer, model


def generate_response(
    prompt: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int = 4
) -> Tuple[str, float, Optional[str]]:
    """Generates greedy decoding response for a single preformatted prompt.

    Records the model's decoded text directly as-is without modification.

    Args:
        prompt: Complete, unmodified input prompt string.
        tokenizer: Pretrained AutoTokenizer.
        model: Pretrained AutoModelForCausalLM.
        max_new_tokens: Maximum tokens to generate (default 4).

    Returns:
        Tuple of (raw_generated_text, latency_ms, error_message_or_None).
    """
    import torch

    start_time = time.perf_counter()
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        target_device = getattr(model, "device", "cpu")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return raw_text, round(latency_ms, 2), None
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return "", round(latency_ms, 2), str(e)


def load_ground_truth(ground_truth_path: str) -> Dict[str, Dict[str, str]]:
    """Loads ground-truth answers from CSV and builds a multi-key lookup table."""
    if not os.path.isfile(ground_truth_path):
        raise FileNotFoundError(f"Ground-truth file not found: '{ground_truth_path}'")

    lookup: Dict[str, Dict[str, str]] = {}
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("case_id", "").strip()
            ifile = row.get("input_file", "").strip()
            exp_dec = row.get("expected_decision", "").strip()
            exp_lbl = row.get("expected_label", "").strip()

            entry = {
                "case_id": cid,
                "input_file": ifile,
                "expected_decision": exp_dec,
                "expected_label": exp_lbl,
            }

            if ifile:
                lookup[ifile] = entry
                lookup[Path(ifile).stem] = entry
            if cid:
                lookup[cid] = entry
                lookup[cid.lower()] = entry
                lookup[cid.upper()] = entry
    return lookup


def _format_markdown_report(eval_summary: Dict[str, Any]) -> str:
    """Generates a clean markdown evaluation report."""
    total = eval_summary.get("total_test_cases", 0)
    correct = eval_summary.get("correct_predictions", 0)
    incorrect = eval_summary.get("incorrect_predictions", 0)
    acc = eval_summary.get("accuracy_pct", 0.0)

    lines = [
        "# PolicyGuard-4B Evaluation Report",
        "",
        f"- **Ground Truth File:** `{eval_summary.get('ground_truth_file', '')}`",
        f"- **Total Test Cases Evaluated:** {total}",
        f"- **Correct Predictions:** {correct} / {total} ({acc}%)",
        f"- **Incorrect Predictions:** {incorrect} / {total}",
        "",
        "## Performance Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Total Test Cases** | {total} |",
        f"| **Correct** | {correct} |",
        f"| **Incorrect** | {incorrect} |",
        f"| **Accuracy** | **{acc}%** |",
        ""
    ]

    mismatches = eval_summary.get("mismatched_cases", [])
    if mismatches:
        lines.append(f"## Mismatched Test Cases ({len(mismatches)} Cases)")
        lines.append("")
        lines.append("| Case ID | Input File | Expected Label | Expected Decision | Model Output (As-Is) |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for m in mismatches:
            clean_raw = (m.get("model_output") or "").replace("\n", " ").replace("|", "\\|")
            lines.append(
                f"| `{m['case_id']}` | `{m['input_filename']}` | `{m['expected_label']}` | `{m['expected_decision']}` | `{clean_raw}` |"
            )
        lines.append("")
    else:
        lines.append("## Mismatched Test Cases")
        lines.append("")
        lines.append("None! Perfect 100% accuracy achieved on all evaluated test cases.")
        lines.append("")

    return "\n".join(lines)


def evaluate_predictions(
    predictions: Any,
    ground_truth_path: str = "policyguard_evaluation_ground_truth.csv",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluates prediction records directly against ground truth without normalizing outputs."""
    records: List[Dict[str, Any]] = []
    if isinstance(predictions, str):
        if predictions.endswith(".jsonl"):
            with open(predictions, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        elif predictions.endswith(".csv"):
            with open(predictions, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = list(reader)
        else:
            raise ValueError(f"Unsupported prediction file format: {predictions}")
    elif isinstance(predictions, list):
        records = predictions
    else:
        raise TypeError("predictions must be a list of records or a file path string.")

    gt_lookup = load_ground_truth(ground_truth_path)

    evaluated_cases = 0
    correct_count = 0
    incorrect_count = 0

    detailed_results = []
    mismatches = []
    correct_cases = []

    for r in records:
        fname = r.get("input_filename") or r.get("input_file") or ""
        cid = r.get("case_id") or ""
        model_out = (r.get("model_output") or r.get("raw_model_output") or "").strip()
        latency = r.get("inference_latency_ms", 0)

        # Match against ground truth
        gt_entry = gt_lookup.get(fname) or gt_lookup.get(cid) or gt_lookup.get(Path(fname).stem if fname else "")
        if not gt_entry and cid:
            short_id = cid.split("_")[0]
            gt_entry = gt_lookup.get(short_id)

        if not gt_entry:
            print(f"Warning: No ground truth entry found for case '{cid}' / '{fname}'. Skipping.")
            continue

        evaluated_cases += 1
        expected_label = gt_entry["expected_label"]
        expected_decision = gt_entry["expected_decision"]
        canonical_cid = gt_entry["case_id"]

        # Check correctness directly against model output as-is
        out_lower = model_out.lower()
        if expected_label == "no_violation":
            is_correct = "no_violation" in out_lower or "no violation" in out_lower
        elif expected_label == "violation":
            is_correct = "violation" in out_lower and "no_violation" not in out_lower and "no violation" not in out_lower
        else:
            is_correct = False

        if is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

        item = {
            "case_id": canonical_cid,
            "input_filename": fname or gt_entry["input_file"],
            "expected_decision": expected_decision,
            "expected_label": expected_label,
            "model_output": model_out,
            "is_correct": is_correct,
            "latency_ms": latency
        }
        detailed_results.append(item)

        if is_correct:
            correct_cases.append(item)
        else:
            mismatches.append(item)

    accuracy = (correct_count / evaluated_cases) if evaluated_cases > 0 else 0.0
    accuracy_pct = round(accuracy * 100.0, 2)

    eval_summary = {
        "ground_truth_file": ground_truth_path,
        "total_test_cases": evaluated_cases,
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "accuracy_pct": accuracy_pct,
        "mismatched_cases": mismatches,
        "correct_cases": correct_cases
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        eval_json_path = os.path.join(output_dir, "evaluation.json")
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)

        eval_csv_path = os.path.join(output_dir, "evaluation_results.csv")
        csv_cols = [
            "case_id",
            "input_filename",
            "expected_decision",
            "expected_label",
            "model_output",
            "is_correct",
            "latency_ms"
        ]
        with open(eval_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols)
            writer.writeheader()
            for item in detailed_results:
                writer.writerow({
                    "case_id": item["case_id"],
                    "input_filename": item["input_filename"],
                    "expected_decision": item["expected_decision"],
                    "expected_label": item["expected_label"],
                    "model_output": item["model_output"],
                    "is_correct": item["is_correct"],
                    "latency_ms": item["latency_ms"]
                })

        report_md_path = os.path.join(output_dir, "evaluation_report.md")
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(_format_markdown_report(eval_summary))

        eval_summary["evaluation_json_path"] = eval_json_path
        eval_summary["evaluation_csv_path"] = eval_csv_path
        eval_summary["evaluation_report_md_path"] = report_md_path

    return eval_summary


def run_batch(
    input_dir: str = "policyguard_prompts_100/policyguard_inputs",
    output_dir: Optional[str] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    limit: Optional[int] = None,
    resume: bool = True,
    model_id: str = "Rakancorle1/PolicyGuard-4B",
    max_new_tokens: int = 4,
    ground_truth_path: Optional[str] = "policyguard_evaluation_ground_truth.csv"
) -> Dict[str, Any]:
    """Runs sequential inference over .txt prompt files and records outputs as-is."""
    start_wall_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Validate input directory
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    if not txt_files:
        raise ValueError(f"No .txt input files found in directory '{input_dir}'.")

    # Sort files deterministically
    txt_files = sorted(txt_files, key=natural_sort_key)

    if limit is None and len(txt_files) != 100:
        raise ValueError(
            f"Expected exactly 100 .txt input files in '{input_dir}', but found {len(txt_files)}. "
            "A full experiment run requires exactly 100 inputs. For testing fewer inputs, pass limit=<N>."
        )

    if limit is not None and limit > 0:
        txt_files = txt_files[:limit]

    total_cases = len(txt_files)

    # 2. Setup output directory
    if not output_dir:
        output_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    jsonl_path = os.path.join(output_dir, "predictions.jsonl")
    csv_path = os.path.join(output_dir, "predictions.csv")
    summary_path = os.path.join(output_dir, "summary.json")

    # 3. Load resume cache if predictions.jsonl exists
    existing_cache: Dict[str, Dict[str, Any]] = {}
    if resume and os.path.isfile(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        fname = rec.get("input_filename")
                        if fname:
                            existing_cache[fname] = rec
                    except json.JSONDecodeError:
                        continue

    # 4. Ensure model and tokenizer are loaded
    if model is None or tokenizer is None:
        tokenizer, model = load_model(model_id=model_id)

    # 5. Sequential inference loop
    records: List[Dict[str, Any]] = []

    print(f"\nStarting inference on {total_cases} case(s)...")
    print(f"Output directory: {output_dir}\n")

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_file:
        for idx, fname in enumerate(txt_files, start=1):
            file_path = os.path.join(input_dir, fname)
            case_id = Path(fname).stem

            with open(file_path, "r", encoding="utf-8") as pf:
                prompt_text = pf.read()

            prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

            # Check resume cache
            if resume and fname in existing_cache:
                cached_rec = existing_cache[fname]
                if cached_rec.get("prompt_hash") == prompt_hash:
                    raw_out = cached_rec.get("model_output") or cached_rec.get("raw_model_output") or ""
                    cached_rec["model_output"] = raw_out
                    cached_rec["raw_model_output"] = raw_out
                    # Clean out legacy fields
                    cached_rec.pop("normalized_label", None)
                    cached_rec.pop("parse_status", None)
                    cached_rec.pop("decision", None)

                    records.append(cached_rec)
                    print(f"[{idx}/{total_cases}] {case_id}: CACHED ({repr(raw_out)})")
                    continue

            # Generate response directly as-is
            raw_output, latency_ms, error = generate_response(
                prompt=prompt_text,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max_new_tokens
            )

            record = {
                "case_id": case_id,
                "input_filename": fname,
                "prompt_hash": prompt_hash,
                "full_input_prompt": prompt_text,
                "model_output": raw_output,
                "raw_model_output": raw_output,
                "inference_latency_ms": latency_ms,
                "inference_error": error
            }

            # Write and flush immediately
            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()
            records.append(record)

            status_disp = repr(raw_output) if not error else f"ERROR: {error}"
            print(f"[{idx}/{total_cases}] {case_id}: {status_disp} | {latency_ms:.1f} ms")

    # 5.1 Ensure predictions.jsonl is synced
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for r in records:
            jf.write(json.dumps(r) + "\n")

    # 6. Consolidated CSV output
    csv_fields = [
        "case_id",
        "input_filename",
        "model_output",
        "inference_latency_ms",
        "inference_error"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "case_id": r.get("case_id", ""),
                "input_filename": r.get("input_filename", ""),
                "model_output": r.get("model_output") or r.get("raw_model_output", ""),
                "inference_latency_ms": r.get("inference_latency_ms", 0.0),
                "inference_error": r.get("inference_error", "") or ""
            })

    # 7. Summary JSON output
    latencies = [
        r["inference_latency_ms"]
        for r in records
        if r.get("inference_error") is None and isinstance(r.get("inference_latency_ms"), (int, float))
    ]
    successful_count = sum(1 for r in records if r.get("inference_error") is None)
    total_runtime = round(time.time() - start_wall_time, 2)

    summary = {
        "model_checkpoint": model_id,
        "execution_timestamp": timestamp,
        "number_of_inputs": len(records),
        "successfully_generated_count": successful_count,
        "average_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "median_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "total_runtime_seconds": total_runtime,
        "output_directory": output_dir,
        "predictions_jsonl_path": jsonl_path,
        "predictions_csv_path": csv_path,
        "summary_json_path": summary_path
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Inference Batch Completed Successfully")
    print("=" * 60)
    print(f"Total Inputs:           {len(records)}")
    print(f"Successful Inferences:  {successful_count}")
    print(f"Average Latency:        {summary['average_latency_ms']} ms")
    print(f"Total Runtime:          {total_runtime} s")
    print(f"Saved Files:\n  - {jsonl_path}\n  - {csv_path}\n  - {summary_path}\n")

    # 8. Automatic evaluation against ground truth if provided
    if ground_truth_path and os.path.isfile(ground_truth_path):
        eval_summary = evaluate_predictions(
            predictions=records,
            ground_truth_path=ground_truth_path,
            output_dir=output_dir
        )
        summary["evaluation"] = eval_summary

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("=" * 60)
        print("GROUND TRUTH EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total Evaluated:        {eval_summary['total_test_cases']}")
        print(f"Correct Test Cases:     {eval_summary['correct_predictions']} / {eval_summary['total_test_cases']} ({eval_summary['accuracy_pct']}%)")
        print(f"Incorrect Test Cases:   {eval_summary['incorrect_predictions']} / {eval_summary['total_test_cases']}")
        print(f"Evaluation Reports Saved:")
        print(f"  - {eval_summary['evaluation_json_path']}")
        print(f"  - {eval_summary['evaluation_csv_path']}")
        print(f"  - {eval_summary['evaluation_report_md_path']}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="PolicyGuard-4B Inference Pipeline for preformatted prompt test cases."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="policyguard_prompts_100/policyguard_inputs",
        help="Directory containing preformatted .txt input files (default: policyguard_prompts_100/policyguard_inputs)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save predictions.jsonl, predictions.csv, and summary.json."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Rakancorle1/PolicyGuard-4B",
        help="Hugging Face model checkpoint (default: Rakancorle1/PolicyGuard-4B)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of cases for smoke testing (e.g. 3)."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable skipping previously completed cases."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Maximum generation tokens (default: 4)."
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="policyguard_evaluation_ground_truth.csv",
        help="Path to ground truth CSV for evaluation (default: policyguard_evaluation_ground_truth.csv)."
    )
    parser.add_argument(
        "--evaluate-only",
        type=str,
        default=None,
        help="Path to an existing predictions.csv or predictions.jsonl to evaluate without running inference."
    )

    args = parser.parse_args()

    if args.evaluate_only:
        out_dir = args.output_dir or os.path.dirname(args.evaluate_only) or "results"
        print(f"Evaluating '{args.evaluate_only}' against '{args.ground_truth}'...")
        res = evaluate_predictions(
            predictions=args.evaluate_only,
            ground_truth_path=args.ground_truth,
            output_dir=out_dir
        )
        print(f"Evaluation complete. Accuracy: {res['accuracy_pct']}% ({res['correct_predictions']}/{res['total_test_cases']})")
        print(f"Report saved to: {res.get('evaluation_report_md_path')}")
        return

    run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        limit=args.limit,
        resume=not args.no_resume,
        max_new_tokens=args.max_new_tokens,
        ground_truth_path=args.ground_truth
    )


if __name__ == "__main__":
    main()
