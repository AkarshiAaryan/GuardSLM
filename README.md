# PolicyGuard-4B Minimal Inference Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: PyTorch / HuggingFace](https://img.shields.io/badge/Framework-PyTorch%20%2F%20HuggingFace-orange.svg)](https://huggingface.co/)
[![Model: PolicyGuard-4B](https://img.shields.io/badge/Model-Rakancorle1%2FPolicyGuard--4B-purple.svg)](https://huggingface.co/Rakancorle1/PolicyGuard-4B)

A minimal, single-purpose inference pipeline and Google Colab execution harness for running the pretrained **PolicyGuard-4B** model on preformatted policy-compliance test cases.

---

## 1. What This Repository Does

This repository has one specific purpose:

1. Loads the pretrained `Rakancorle1/PolicyGuard-4B` causal language model.
2. Reads preformatted `.txt` input files containing complete PolicyGuard prompts.
3. Sends each prompt directly to the model using deterministic greedy decoding (`do_sample=False`, `max_new_tokens=4`).
4. Decodes and strictly normalizes responses into `violation`, `no_violation`, or `UNKNOWN`.
5. Records inference latency (ms) and captures generation errors.
6. Streams predictions immediately to `predictions.jsonl`, `predictions.csv`, and `summary.json`.

> **Note on Pipeline Scope:**  
> This pipeline **does not** generate prompts, parse policies or trajectories, construct synthetic datasets, evaluate predictions against ground truth, or benchmark against other models. It receives complete, model-ready prompt files as strings and collects raw and normalized model outputs.

---

## 2. PolicyGuard-4B Checkpoint

- **Hugging Face Model ID:** `Rakancorle1/PolicyGuard-4B`
- **Architecture:** Causal Language Model fine-tuned on policy-trajectory verification (based on Qwen3-4B-Instruct-2507).
- **Inference Mode:** Greedy token generation with temperature zero (`do_sample=False`, `max_new_tokens=4`).

---

## 3. Native Input and Output Interface

### Expected Input Format
PolicyGuard-4B is trained on prompts structured as:

```text
Policy: {policy}

Trajectory Actions:
{actions}

Output violation or no_violation.
```

The pipeline does not alter, wrap, or template the prompt. Each supplied `.txt` file is read as an unmodified UTF-8 string and sent directly to the tokenizer.

### Model Output Mapping

| Model Output (Decoded) | Normalized Label | Parse Status |
| :--- | :--- | :--- |
| `violation` | `violation` | `success` |
| `no_violation` | `no_violation` | `success` |
| `no violation` | `no_violation` | `success` |
| *Anything else / unparseable* | `UNKNOWN` | `failed` |

Surrounding whitespace and harmless punctuation are stripped before exact matching. Substring guessing is strictly prohibited, ensuring `no_violation` is never misclassified as `violation`.

---

## 4. Expected Input Directory Arrangement

Supply an input directory containing your preformatted `.txt` prompt files:

```text
PolicyGuardInputs/
├── TC001.txt
├── TC002.txt
├── TC003.txt
...
└── TC100.txt
```

- Each file must contain **one complete prompt** ready for PolicyGuard.
- For a full experiment run, exactly **100 `.txt` files** are expected.
- Files are sorted deterministically using natural numerical ordering (`TC1`, `TC2`, ..., `TC100`).

---

## 5. Google Colab Execution (`PolicyGuard_100_Test_Cases.ipynb`)

A complete Google Colab notebook is provided in the repository root:

`PolicyGuard_100_Test_Cases.ipynb`

### How to Run in Colab:
1. Open [Google Colab](https://colab.research.google.com/) and upload `PolicyGuard_100_Test_Cases.ipynb`.
2. Select a GPU runtime: **Runtime -> Change runtime type -> Hardware accelerator: T4 GPU** (or A100 / V100).
3. Mount Google Drive and set your folder paths:
   ```python
   INPUT_DIR = "/content/drive/MyDrive/PolicyGuardInputs"
   OUTPUT_DIR = "/content/drive/MyDrive/PolicyGuardOutputs"
   ```
4. Run the cells sequentially:
   - **Cell 1:** Environment setup and GPU availability verification.
   - **Cell 2:** Mount Google Drive and import `policyguard_pipeline`.
   - **Cell 3:** User configuration parameters.
   - **Cell 4:** Input file validation (checks for 100 `.txt` files).
   - **Cell 5:** Load PolicyGuard-4B model and tokenizer into GPU memory.
   - **Cell 6:** Smoke test (runs 3 cases to a test directory).
   - **Cell 7:** Full 100-case experiment execution with streaming progress.
   - **Cell 8:** Output collection, summary display, and CSV preview.
   - **Cell 9:** Safe cleanup and GPU memory release.

All predictions are written to Google Drive so they persist after the Colab session disconnects.

---

## 6. GPU Runtime Requirement

PolicyGuard-4B has 4 billion parameters.
- **Recommended GPU:** NVIDIA T4 (16GB VRAM, standard free Colab tier), A10G, or A100.
- **Precision:** FP16 on CUDA devices (`device_map="auto"`).
- **CPU Fallback:** FP32 on CPU (for testing and verification).

---

## 7. Output Files and Format

Inference results are written to the configured output directory:

### `predictions.jsonl`
Streaming line-by-line JSON record flushed immediately after each test case:
```json
{
  "case_id": "TC001",
  "input_filename": "TC001.txt",
  "prompt_hash": "a1b2c3d4...",
  "full_input_prompt": "Policy: ... Trajectory Actions: ...",
  "raw_model_output": "no_violation",
  "normalized_label": "no_violation",
  "parse_status": "success",
  "parsing_error": null,
  "inference_latency_ms": 24.5,
  "inference_error": null
}
```

### `predictions.csv`
Consolidated tabular record for analysis:
| case_id | input_filename | raw_model_output | normalized_label | parse_status | inference_latency_ms | inference_error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TC001 | TC001.txt | no_violation | no_violation | success | 24.5 | |
| TC002 | TC002.txt | violation | violation | success | 23.8 | |

### `summary.json`
Run operational metrics (no ground truth or accuracy calculations):
```json
{
  "model_checkpoint": "Rakancorle1/PolicyGuard-4B",
  "execution_timestamp": "20261109_120000",
  "number_of_inputs": 100,
  "successfully_generated_count": 100,
  "count_violation": 49,
  "count_no_violation": 51,
  "count_UNKNOWN": 0,
  "average_latency_ms": 25.1,
  "median_latency_ms": 24.8,
  "total_runtime_seconds": 31.4,
  "output_directory": "..."
}
```

---

## 8. Command-Line Usage

You can also run the pipeline locally via terminal:

### Installation
```bash
git clone https://github.com/AkarshiAaryan/GuardSLM.git
cd GuardSLM
pip install -r requirements.txt
```

### Smoke Test (3 Cases)
```bash
python policyguard_pipeline.py \
  --input-dir /path/to/PolicyGuardInputs \
  --output-dir smoke_test_output \
  --limit 3
```

### Full 100-Case Experiment
```bash
python policyguard_pipeline.py \
  --input-dir /path/to/PolicyGuardInputs \
  --output-dir results/full_experiment
```

### Resume Capability
If interrupted, rerunning with the same output directory automatically skips completed inputs whose prompt content hash matches the saved prediction. To re-run from scratch, add `--no-resume`.

---

## 9. Repository Structure

```text
GuardSLM/
│
├── policyguard_pipeline.py         # Reusable inference pipeline and CLI
├── PolicyGuard_100_Test_Cases.ipynb # Colab notebook for running 100 test cases
├── requirements.txt                # Minimal dependencies (torch, transformers, accelerate, safetensors)
├── README.md                       # Documentation and usage guide
├── .gitignore                      # Git ignore rules
└── LICENSE                         # MIT License
```

---

## 10. License
This project is licensed under the [MIT License](LICENSE).
