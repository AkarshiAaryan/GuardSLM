"""PolicyGuard-4B Minimal Inference Pipeline.

This pipeline executes inference using the official pretrained checkpoint:
Rakancorle1/PolicyGuard-4B on preformatted .txt input files.

Pipeline steps:
1. Load PolicyGuard-4B model and tokenizer.
2. Read 100 input files containing preformatted prompts.
3. Send each prompt to the model with greedy decoding.
4. Record raw model responses, normalized decisions, and inference latencies.
5. Save streaming records immediately to predictions.jsonl and generate predictions.csv and summary.json.
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


def parse_output(raw_output: str) -> Tuple[str, str, Optional[str]]:
    """Strictly normalizes PolicyGuard outputs into canonical labels.

    Normalization mapping:
    - 'violation' -> 'violation'
    - 'no_violation' -> 'no_violation'
    - 'no violation' -> 'no_violation'
    - Anything else -> 'UNKNOWN'

    Returns:
        Tuple of (normalized_label, parse_status, parsing_error_or_None).
    """
    if not raw_output or not raw_output.strip():
        return "UNKNOWN", "failed", "Empty model output"

    clean = raw_output.strip().strip("'\"`.,;:! \n\r\t").lower()
    clean_norm = re.sub(r'[\s-]+', '_', clean)

    if clean_norm == "no_violation":
        return "no_violation", "success", None
    elif clean_norm == "violation":
        return "violation", "success", None
    else:
        return "UNKNOWN", "failed", f"Unrecognized model output: '{raw_output}'"


def run_batch(
    input_dir: str,
    output_dir: Optional[str] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    limit: Optional[int] = None,
    resume: bool = True,
    model_id: str = "Rakancorle1/PolicyGuard-4B",
    max_new_tokens: int = 4
) -> Dict[str, Any]:
    """Runs sequential inference over all .txt prompt files in input_dir.

    Args:
        input_dir: Directory containing .txt prompt files.
        output_dir: Directory where predictions and summaries will be written.
        model: Preloaded model instance (loads from model_id if None).
        tokenizer: Preloaded tokenizer instance (loads from model_id if None).
        limit: Optional cap on test cases for smoke testing (e.g., 3).
        resume: Skip already-completed cases matching prompt content hash.
        model_id: Model checkpoint identifier.
        max_new_tokens: Max new tokens for generation (default 4).

    Returns:
        Dictionary containing run summary and output file paths.
    """
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

    # Validate full experiment requirement (100 inputs) unless limit is specified
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
                    records.append(cached_rec)
                    print(f"[{idx}/{total_cases}] {case_id}: CACHED ({cached_rec.get('normalized_label')})")
                    continue

            # Generate and parse
            raw_output, latency_ms, error = generate_response(
                prompt=prompt_text,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max_new_tokens
            )

            if error:
                normalized_label = "UNKNOWN"
                parse_status = "failed"
                parsing_error = error
            else:
                normalized_label, parse_status, parsing_error = parse_output(raw_output)

            record = {
                "case_id": case_id,
                "input_filename": fname,
                "prompt_hash": prompt_hash,
                "full_input_prompt": prompt_text,
                "raw_model_output": raw_output,
                "normalized_label": normalized_label,
                "parse_status": parse_status,
                "parsing_error": parsing_error,
                "inference_latency_ms": latency_ms,
                "inference_error": error
            }

            # Write and flush immediately for crash resilience
            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()
            records.append(record)

            status_disp = normalized_label if not error else f"ERROR: {error}"
            print(f"[{idx}/{total_cases}] {case_id}: {status_disp} | {latency_ms:.1f} ms")

    # 6. Consolidated CSV output
    csv_fields = [
        "case_id",
        "input_filename",
        "raw_model_output",
        "normalized_label",
        "parse_status",
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
                "raw_model_output": r.get("raw_model_output", ""),
                "normalized_label": r.get("normalized_label", ""),
                "parse_status": r.get("parse_status", ""),
                "inference_latency_ms": r.get("inference_latency_ms", 0.0),
                "inference_error": r.get("inference_error", "") or ""
            })

    # 7. Summary JSON output
    latencies = [
        r["inference_latency_ms"]
        for r in records
        if r.get("inference_error") is None and isinstance(r.get("inference_latency_ms"), (int, float))
    ]

    count_violation = sum(1 for r in records if r.get("normalized_label") == "violation")
    count_no_violation = sum(1 for r in records if r.get("normalized_label") == "no_violation")
    count_unknown = sum(1 for r in records if r.get("normalized_label") == "UNKNOWN")
    successful_count = sum(1 for r in records if r.get("inference_error") is None)
    total_runtime = round(time.time() - start_wall_time, 2)

    summary = {
        "model_checkpoint": model_id,
        "execution_timestamp": timestamp,
        "number_of_inputs": len(records),
        "successfully_generated_count": successful_count,
        "count_violation": count_violation,
        "count_no_violation": count_no_violation,
        "count_UNKNOWN": count_unknown,
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
    print(f"Violations:             {count_violation}")
    print(f"No Violations:          {count_no_violation}")
    print(f"UNKNOWN / Errors:       {count_unknown}")
    print(f"Average Latency:        {summary['average_latency_ms']} ms")
    print(f"Total Runtime:          {total_runtime} s")
    print(f"Saved Files:\n  - {jsonl_path}\n  - {csv_path}\n  - {summary_path}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="PolicyGuard-4B Inference Pipeline for preformatted prompt test cases."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing preformatted .txt input files."
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

    args = parser.parse_args()

    run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        limit=args.limit,
        resume=not args.no_resume,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()
