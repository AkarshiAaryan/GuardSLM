# GuardSLM: Context-Aware Safety & Authorization Benchmark for Autonomous Web Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: PyTorch / HuggingFace](https://img.shields.io/badge/Framework-PyTorch%20%2F%20HuggingFace-orange.svg)](https://huggingface.co/)

**GuardSLM** is a modular, lightweight benchmark framework designed to evaluate Small Language Model (SLM) safety guards and agent guardrails on **contextual authorization and safety test cases** within institutional Learning Management Systems (LMS).

---

## 📖 Table of Contents
1. [The Research Problem](#-the-research-problem)
2. [Evaluated Guard Models](#-evaluated-guard-models)
3. [End-to-End System Architecture](#-end-to-end-system-architecture)
4. [Dataset & Zero-Leakage Oracle Isolation](#-dataset--zero-leakage-oracle-isolation)
5. [Evaluation Metrics](#-evaluation-metrics)
6. [Quickstart — Running Locally (CPU Only)](#-quickstart--running-locally-cpu-only)
7. [Running in Google Colab (GPU)](#-running-in-google-colab-gpu)
8. [Repository Structure](#-repository-structure)
9. [License](#-license)

---

## 🎯 The Research Problem

Autonomous AI agents operating inside institutional management systems execute user requests by interacting with web interfaces (clicking buttons, editing assignments, viewing gradebooks, submitting assessments). In institutional software, security is **context-dependent**: the exact same operation (e.g., clicking "Edit submission" or opening grades) is allowed or prohibited depending strictly on:
* **Who** is acting (`Student`, `Teaching Assistant`, `Professor`, `Administrator`),
* **Which** policy applies (due dates, grading status, course enrolment),
* **Whose** resource is being accessed (`self` vs. `peer`).

Existing general-purpose safety guards often react blindly to surface keywords. `GuardSLM` rigorously benchmarks specialized SLMs to measure whether a guard model genuinely evaluates permission context, role hierarchy, and browser state.

---

## 🤖 Evaluated Guard Models

| Model | Parameters | Checkpoint | Adapter Class | Native Format & Logic |
| :--- | :---: | :--- | :--- | :--- |
| **PolicyGuard-4B** | 4.0B | `Rakancorle1/PolicyGuard-4B` | `PolicyGuardBenchmarkGuard` | Official plain-text trajectory evaluation format (`Policy: ... \n\nTrajectory Actions:\n... \n\nOutput violation or no_violation.` mapped to `BLOCK` / `ALLOW`) |
| **DynaGuard-1.7B** | 1.7B | `tomg-group-umd/DynaGuard-1.7B` | `DynaGuardBenchmarkGuard` | Official fast-inference dialogue compliance format (`[INST]` policy + dialogue `[/INST]\n<answer>` → `PASS` / `FAIL` mapped to `ALLOW` / `BLOCK`) |
| **MockGuard** | — | Deterministic CPU mock | `MockBenchmarkGuard` | Offline pipeline verification |

---

## 🏗️ End-to-End System Architecture

```text
       1. RESEARCH SNAPSHOTS (100 Authentic Playwright Test Cases)
       (research_snapshots_100/slm_inputs/TC*.md: User, Role, Policy, A11y Tree)
                                 │
                                 ▼
                   2. BENCHMARK LOADER & ISOLATION
                   (src/benchmark/data/loader.py)
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
       Public Model Inputs             Isolated Gold Oracle
      (SnapshotInputCase)             (PrivateGoldAnnotation)
     [Zero-Leakage Sanitized]         [Hidden from LLMs]
                  │                             │
                  ▼                             │
       3. SPECIALIZED MODEL ADAPTERS            │
          (PolicyGuard / DynaGuard)             │
                  │                             │
                  ▼                             │
       4. RESILIENT OUTPUT PARSER               │
     (Word-boundary regex & JSON repair)        │
                  │                             │
                  ▼                             │
       5. BENCHMARK PREDICTIONS                 │
     (Parsed verdicts: ALLOW / BLOCK)           │
                  │                             │
                  └──────────────┬──────────────┘
                                 ▼
                     6. BENCHMARK EVALUATOR
                 (src/benchmark/evaluation/evaluator.py)
                                 │
                                 ▼
                     7. COMPREHENSIVE REPORT
                (Overall Accuracy, Decision Macro F1,
                 False ALLOW Hazard, False BLOCK Usability)
```

---

## 🔒 Dataset & Zero-Leakage Oracle Isolation

1. **Public Model Inputs**:
   Located in `research_snapshots_100/slm_inputs/TC01_*.md` to `TC100_*.md`. Each case includes:
   - Authenticated role and user account
   - Target resource and requested operation
   - Formal institutional policy rules
   - Authentic Playwright accessibility tree of the target web page
   - Method `to_sanitized_dict()` guarantees zero exposure to ground truth decisions or institutional rationales.

2. **Evaluation Oracle**:
   Isolated in `research_snapshots_100/evaluation_only/ground_truth.csv` (100 private ground truth verdicts: 48 ALLOW, 52 BLOCK). Loaded strictly into `BenchmarkEvaluator` post-inference. (The 20-case pilot dataset in `research_snapshots/` remains fully supported).

---

## 📊 Evaluation Metrics

* **Overall Accuracy**: Fraction of cases where the model decision matches the ground-truth oracle.
* **Decision Macro F1**: Unweighted arithmetic average of per-class F1 across `ALLOW`, `BLOCK`, and `ASK`.
* **False-ALLOW Rate (Security Hazard)**:
  $$\text{False-ALLOW Rate} = \frac{\text{Mistakenly allowed unsafe actions}}{\text{Total truly unsafe actions (BLOCK + ASK)}}$$
* **False-BLOCK Rate (Usability Penalty)**:
  $$\text{False-BLOCK Rate} = \frac{\text{Mistakenly blocked safe actions}}{\text{Total truly safe actions (ALLOW)}}$$
* **Operational Performance**: Mean, median, and P95 latency in milliseconds.

---

## ⚡ Quickstart — Running Locally (CPU Only)

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python3 -m pytest tests/ -v
```

### 3. Run CPU Smoke Test
```bash
python3 -m src.benchmark.runner.experiment_runner \
  --model mock_guard \
  --dataset research_snapshots_100 \
  --limit 100 \
  --output-dir data/benchmark_results/smoke_test
```

---

## 🚀 Running in Google Colab (GPU)

Open `notebooks/colab_pilot.ipynb` or `notebooks/colab_guard_slm_100_pairs.ipynb` in Google Colab:

1. **Step 1**: Clones repository into `/content/GuardSLM`.
2. **Step 2**: Installs Hugging Face transformers, accelerate, and bitsandbytes.
3. **Step 3**: Detects GPU hardware.
4. **Step 4**: Loads and verifies the benchmark snapshot cases.
5. **Step 5**: Runs CPU sanity test with `MockBenchmarkGuard`.
6. **Step 6A**: Evaluates **PolicyGuard-4B**.
7. **Step 6B**: Evaluates **DynaGuard-1.7B**.
8. **Step 7**: Renders the consolidated comparative leaderboard.

---

## 📂 Repository Structure

```text
slm-guard-pilot/
├── config/
│   ├── config.yaml                     # Global benchmark paths and parameters
│   └── models.yaml                     # Model definitions and checkpoints
├── data/
│   └── GuardSLM_v3_Single_Master_Dataset.json  # Reference master dataset
├── notebooks/
│   ├── colab_guard_slm_100_pairs.ipynb # Google Colab multi-model evaluation notebook
│   └── colab_pilot.ipynb               # Google Colab pilot evaluation notebook
├── prompts/
│   ├── base_guard_prompt.txt           # Base gatekeeper prompt template
│   └── guardslm_100_authorization_prompt.txt # Institutional authorization prompt
├── research_snapshots_100/
│   ├── evaluation_only/
│   │   └── ground_truth.csv            # Physical isolation oracle (100 cases)
│   ├── manifest.csv                    # Metadata mapping for all 100 cases
│   ├── raw_snapshots/                  # Raw page and policy snapshots
│   ├── reports/                        # Coverage, validation, inventory reports
│   └── slm_inputs/                     # 100 public test cases with Playwright trees
├── research_snapshots/                 # (Legacy 20-case pilot dataset)
├── src/
│   └── benchmark/
│       ├── data/
│       │   ├── loader.py               # Ingestion & oracle isolation
│       │   ├── schema.py               # Public vs private data contracts
│       │   └── validator.py            # Static dataset integrity validator
│       ├── evaluation/
│       │   ├── evaluator.py            # Multi-model reports and leaderboard
│       │   └── metrics.py              # Confusion matrix, Macro F1, Hazard rate
│       ├── models/
│       │   ├── base.py                 # BenchmarkModel abstract base class
│       │   ├── dynaguard_benchmark_guard.py   # DynaGuard-1.7B adapter
│       │   ├── mock_benchmark_guard.py        # Deterministic test mock
│       │   └── policyguard_benchmark_guard.py # PolicyGuard-4B adapter
│       ├── parser/
│       │   └── output_parser.py        # JSON repair and verdict extraction
│       ├── prompts/
│       │   └── renderer.py             # Prompt formatting helpers
│       ├── runner/
│       │   ├── experiment_runner.py    # Primary benchmark CLI runner
│       │   └── robustness.py           # Policy perturbation tests
│       └── vision/
│           └── screenshot_resolver.py  # Page PNG extractor
│   └── utils/
│       ├── io.py                       # File serialization helpers
│       ├── logging.py                  # Structured console and file logger
│       └── reproducibility.py          # Environment and git metadata logger
├── tests/                              # Comprehensive pytest suite (28 tests)
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
