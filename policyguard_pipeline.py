"""PolicyGuard-4B Minimal Inference Pipeline.

A simple, single-purpose inference pipeline:
1. Loads Rakancorle1/PolicyGuard-4B model and tokenizer.
2. Reads the 100 test case .txt files.
3. Sends each prompt to the model with greedy decoding.
4. Records raw model outputs mapped to case IDs directly into a CSV file.
"""

import argparse
import csv
import os
import re
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


def run_inference(
    input_dir: str = "policyguard_prompts_100/policyguard_inputs",
    output_csv: str = "policyguard_outputs.csv",
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    model_id: str = "Rakancorle1/PolicyGuard-4B",
    max_new_tokens: int = 4,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Runs sequential inference over all .txt prompt files and saves outputs to CSV.

    Args:
        input_dir: Directory containing .txt prompt files.
        output_csv: Path to output CSV file where results will be saved.
        model: Preloaded model instance (loads from model_id if None).
        tokenizer: Preloaded tokenizer instance (loads from model_id if None).
        model_id: Model checkpoint identifier.
        max_new_tokens: Max new tokens for greedy decoding (default 4).
        limit: Optional cap on test cases for smoke testing (e.g. 3).

    Returns:
        List of dictionaries with case_id, file_name, model_output, latency_ms.
    """
    import torch

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' not found.")

    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")], key=natural_sort_key)
    if not txt_files:
        raise ValueError(f"No .txt input files found in '{input_dir}'.")

    if limit is not None and limit > 0:
        txt_files = txt_files[:limit]

    total_cases = len(txt_files)

    if model is None or tokenizer is None:
        tokenizer, model = load_model(model_id=model_id)

    target_device = getattr(model, "device", "cpu")
    results: List[Dict[str, Any]] = []

    print(f"\nProcessing {total_cases} test cases from '{input_dir}'...")

    for idx, fname in enumerate(txt_files, start=1):
        file_path = os.path.join(input_dir, fname)
        case_id = fname.split("_")[0] if "_" in fname else Path(fname).stem

        with open(file_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()

        t0 = time.perf_counter()
        try:
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            error = None
        except Exception as e:
            raw_output = ""
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            error = str(e)

        record = {
            "case_id": case_id,
            "file_name": fname,
            "model_output": raw_output,
            "latency_ms": latency_ms,
            "error": error or ""
        }
        results.append(record)

        disp = repr(raw_output) if not error else f"ERROR: {error}"
        print(f"[{idx}/{total_cases}] {case_id} ({fname}): {disp} | {latency_ms} ms")

    # Ensure parent directory exists and save CSV
    parent_dir = os.path.dirname(output_csv)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fieldnames = ["case_id", "file_name", "model_output", "latency_ms", "error"]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("Inference Completed Successfully")
    print("=" * 60)
    print(f"Total Cases:     {len(results)}")
    print(f"Saved Output:    {output_csv}\n")

    return results


# Backward compatibility alias
run_batch = run_inference


def main():
    parser = argparse.ArgumentParser(
        description="PolicyGuard-4B Inference: Sends prompts to model and saves raw outputs to CSV."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="policyguard_prompts_100/policyguard_inputs",
        help="Directory containing preformatted .txt input files."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="policyguard_outputs.csv",
        help="Path where the output CSV will be saved."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Rakancorle1/PolicyGuard-4B",
        help="Hugging Face model checkpoint."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of cases for testing (e.g. 3)."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Maximum generation tokens (default: 4)."
    )

    args = parser.parse_args()

    run_inference(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        model_id=args.model_id,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()
