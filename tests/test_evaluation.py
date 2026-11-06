"""Test 5: Evaluation metrics, confusion matrix, and reporting."""

import pytest
from guardslm.evaluation import evaluate_predictions, generate_markdown_report


def test_evaluate_predictions_metrics():
    # Synthetic ground truth: 3 ALLOW, 3 BLOCK
    gt = {
        "TC01": "ALLOW",
        "TC02": "ALLOW",
        "TC03": "ALLOW",
        "TC04": "BLOCK",
        "TC05": "BLOCK",
        "TC06": "BLOCK",
    }

    # Synthetic predictions:
    # TC01: ALLOW (TP for ALLOW)
    # TC02: ALLOW (TP for ALLOW)
    # TC03: BLOCK (FN for ALLOW, FP for BLOCK) -> False BLOCK!
    # TC04: BLOCK (TP for BLOCK)
    # TC05: ALLOW (FN for BLOCK, FP for ALLOW) -> False ALLOW! (Security breach)
    # TC06: UNKNOWN (Unparseable)
    predictions = [
        {"case_id": "TC01", "verdict": "ALLOW", "latency_ms": 10.0, "parse_status": "success"},
        {"case_id": "TC02", "verdict": "ALLOW", "latency_ms": 12.0, "parse_status": "success"},
        {"case_id": "TC03", "verdict": "BLOCK", "latency_ms": 8.0, "parse_status": "success"},
        {"case_id": "TC04", "verdict": "BLOCK", "latency_ms": 15.0, "parse_status": "success"},
        {"case_id": "TC05", "verdict": "ALLOW", "latency_ms": 11.0, "parse_status": "success"},
        {"case_id": "TC06", "verdict": "UNKNOWN", "latency_ms": 20.0, "parse_status": "failed"},
    ]

    res = evaluate_predictions(predictions, gt)

    assert res["total_cases"] == 6
    # Correct: TC01 (ALLOW), TC02 (ALLOW), TC04 (BLOCK) -> 3 correct
    assert res["correct_predictions"] == 3
    assert res["accuracy"] == 0.50

    # ALLOW metrics: TP=2, FP=1, FN=1
    allow_p = 2 / 3
    allow_r = 2 / 3
    assert pytest.approx(res["per_class"]["ALLOW"]["precision"], 0.01) == allow_p
    assert pytest.approx(res["per_class"]["ALLOW"]["recall"], 0.01) == allow_r

    # BLOCK metrics: TP=1, FP=1, FN=2
    block_p = 1 / 2
    block_r = 1 / 3
    assert pytest.approx(res["per_class"]["BLOCK"]["precision"], 0.01) == block_p
    assert pytest.approx(res["per_class"]["BLOCK"]["recall"], 0.01) == block_r

    # Security metrics:
    # False ALLOW: 1 out of 3 gold BLOCK was allowed -> 1/3 = 0.3333
    assert pytest.approx(res["security_metrics"]["false_allow_rate"], 0.01) == 0.3333
    # False BLOCK: 1 out of 3 gold ALLOW was blocked -> 1/3 = 0.3333
    assert pytest.approx(res["security_metrics"]["false_block_rate"], 0.01) == 0.3333

    # Unparseable count
    assert res["security_metrics"]["unparseable_count"] == 1

    # Confusion matrix
    cm = res["confusion_matrix"]
    assert cm["ALLOW"]["ALLOW"] == 2
    assert cm["ALLOW"]["BLOCK"] == 1
    assert cm["BLOCK"]["BLOCK"] == 1
    assert cm["BLOCK"]["ALLOW"] == 1
    assert cm["BLOCK"]["UNKNOWN"] == 1

    # Latency stats
    assert res["latency_stats"]["min_ms"] == 8.0
    assert res["latency_stats"]["max_ms"] == 20.0
    assert res["latency_stats"]["mean_ms"] == 12.67


def test_generate_markdown_report():
    eval_results = {
        "total_cases": 100,
        "correct_predictions": 95,
        "accuracy": 0.95,
        "macro_f1": 0.9498,
        "per_class": {
            "ALLOW": {"gold_total": 48, "precision": 0.96, "recall": 0.94, "f1": 0.95},
            "BLOCK": {"gold_total": 52, "precision": 0.94, "recall": 0.96, "f1": 0.95}
        },
        "confusion_matrix": {
            "ALLOW": {"ALLOW": 45, "BLOCK": 3, "ASK": 0, "UNKNOWN": 0},
            "BLOCK": {"ALLOW": 2, "BLOCK": 50, "ASK": 0, "UNKNOWN": 0}
        },
        "security_metrics": {
            "false_allow_rate": 0.0385,
            "false_block_rate": 0.0625,
            "unparseable_count": 0
        },
        "latency_stats": {
            "mean_ms": 15.2,
            "median_ms": 14.8,
            "p95_ms": 22.1,
            "min_ms": 9.4,
            "max_ms": 28.3
        }
    }
    meta = {"model_name": "policyguard", "timestamp": "20261106_190000"}

    report = generate_markdown_report(eval_results, meta)
    assert "# GuardSLM Benchmark Report: policyguard" in report
    assert "95.00%" in report
    assert "False ALLOW Rate" in report
    assert "Confusion Matrix" in report
    assert "15.20 ms" in report
