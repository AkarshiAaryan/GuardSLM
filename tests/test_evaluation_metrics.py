"""Test metric calculations: decision accuracy, policy reasoning, pair flip accuracy."""

from src.benchmark.evaluation.metrics import (
    calculate_decision_metrics,
    calculate_policy_reasoning_metrics,
    calculate_matched_pair_metrics
)
from src.benchmark.data.schema import PrivateGoldAnnotation, PairRegistryEntry, PolicyCheck
from src.benchmark.models.base import BenchmarkPrediction
from src.benchmark.parser.output_parser import ParsedBenchmarkOutput


def build_mock_gold(case_id: str, pair_id: str, verdict: str, violated: list) -> PrivateGoldAnnotation:
    return PrivateGoldAnnotation(
        case_id=case_id,
        pair_id=pair_id,
        verdict=verdict,
        applicable_policy_ids=["POL-01"],
        violated_policy_ids=violated,
        policy_checks=[
            PolicyCheck(policy_id="POL-01", status="violated" if "POL-01" in violated else "satisfied")
        ],
        evidence=[],
        rationale="Test gold",
        annotation_status="test",
        gold_provenance="test"
    )


def build_mock_pred(case_id: str, verdict: str, violated: list) -> BenchmarkPrediction:
    parsed = ParsedBenchmarkOutput(
        case_id=case_id,
        verdict=verdict,
        applicable_policy_ids=["POL-01"],
        violated_policy_ids=violated,
        policy_checks=[
            {"policy_id": "POL-01", "status": "violated" if "POL-01" in violated else "satisfied", "evidence_quotes": []}
        ],
        evidence=[],
        rationale="Test pred",
        parse_status="success"
    )
    return BenchmarkPrediction(
        case_id=case_id,
        strategy="text_only",
        parsed=parsed,
        raw_response="{}",
        latency_ms=10.0
    )


def test_decision_metrics_calculation():
    # 2 cases: one BLOCK (gold BLOCK, pred BLOCK), one ALLOW (gold ALLOW, pred ALLOW)
    golds = {
        "C1": build_mock_gold("C1", "P1", "BLOCK", ["POL-01"]),
        "C2": build_mock_gold("C2", "P1", "ALLOW", [])
    }
    preds = [
        build_mock_pred("C1", "BLOCK", ["POL-01"]),
        build_mock_pred("C2", "ALLOW", [])
    ]

    metrics = calculate_decision_metrics(preds, golds)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["false_allow_count"] == 0
    assert metrics["false_block_count"] == 0
    assert metrics["false_allow_rate"] == 0.0
    assert metrics["false_block_rate"] == 0.0


def test_false_allow_and_false_block_rates():
    # C1: gold BLOCK, pred ALLOW (False ALLOW!)
    # C2: gold ALLOW, pred BLOCK (False BLOCK!)
    golds = {
        "C1": build_mock_gold("C1", "P1", "BLOCK", ["POL-01"]),
        "C2": build_mock_gold("C2", "P1", "ALLOW", [])
    }
    preds = [
        build_mock_pred("C1", "ALLOW", []),
        build_mock_pred("C2", "BLOCK", ["POL-01"])
    ]

    metrics = calculate_decision_metrics(preds, golds)
    assert metrics["accuracy"] == 0.0
    assert metrics["false_allow_count"] == 1
    assert metrics["false_allow_denominator"] == 1
    assert metrics["false_allow_rate"] == 1.0
    assert metrics["false_block_count"] == 1
    assert metrics["false_block_denominator"] == 1
    assert metrics["false_block_rate"] == 1.0


def test_matched_pair_metrics_calculation():
    # Pair P1: authorization_flip (C1=BLOCK, C2=ALLOW)
    # Both predicted correctly -> flip accuracy = 1.0
    golds = {
        "P1-A": build_mock_gold("P1-A", "P1", "BLOCK", ["POL-01"]),
        "P1-B": build_mock_gold("P1-B", "P1", "ALLOW", [])
    }
    preds = [
        build_mock_pred("P1-A", "BLOCK", ["POL-01"]),
        build_mock_pred("P1-B", "ALLOW", [])
    ]
    pair_registry = {
        "P1": PairRegistryEntry(
            pair_id="P1",
            scenario_family="test",
            scenario_id="test",
            pair_type="authorization_flip",
            user_request="test",
            proposed_action={},
            intervention_variable="test",
            intervention_description="test",
            value_A=None,
            value_B=None,
            expected_verdict_A="BLOCK",
            expected_verdict_B="ALLOW",
            violated_policy_ids_A=["POL-01"],
            violated_policy_ids_B=[],
            source_screenshot_A={},
            source_screenshot_B={},
            visual_support_level="test",
            pair_quality_note="test",
            one_factor_design="test"
        )
    }

    metrics = calculate_matched_pair_metrics(preds, golds, pair_registry)
    assert metrics["global_pair_accuracy"] == 1.0
    assert metrics["authorization_flip_accuracy"] == 1.0
    assert metrics["authorization_flip_correct"] == 1


def test_generate_consolidated_summary_table(tmp_path):
    from src.benchmark.evaluation.evaluator import generate_consolidated_summary_table
    import json

    # Test non-existent dir
    res_empty = generate_consolidated_summary_table(str(tmp_path / "nonexistent"))
    assert "does not exist" in res_empty

    # Create mock runs for policyguard, dynaguard
    results_dir = tmp_path / "benchmark_results"
    for model_name, acc, fa_rate in [
        ("policyguard", 0.90, 0.05),
        ("dynaguard", 0.85, 0.00)
    ]:
        run_dir = results_dir / f"20261102_120000_{model_name}_text_only"
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_data = {
            "text_only": {
                "model_name": model_name,
                "strategy": "text_only",
                "total_cases_evaluated": 20,
                "decision_performance": {
                    "accuracy": acc,
                    "macro_f1": acc,
                    "false_allow_count": int(fa_rate * 20),
                    "false_allow_denominator": 20,
                    "false_allow_rate": fa_rate,
                    "false_block_count": 1,
                    "false_block_denominator": 20,
                    "false_block_rate": 0.05
                },
                "matched_pair_performance": {
                    "total_pairs": 10,
                    "total_correct_pairs": 8,
                    "global_pair_accuracy": 0.8,
                    "authorization_flip_accuracy": 0.8
                },
                "operational_performance": {
                    "latency_median_ms": 350.0
                }
            }
        }
        with open(run_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(eval_data, f)

    table_md = generate_consolidated_summary_table(str(results_dir))
    assert "Consolidated Multi-Model Leaderboard" in table_md
    assert "policyguard" in table_md
    assert "dynaguard" in table_md
    assert "90.0%" in table_md
    assert "85.0%" in table_md

