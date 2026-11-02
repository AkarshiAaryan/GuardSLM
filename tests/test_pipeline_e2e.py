"""End-to-End Pipeline integration test on GuardSLM v3 Master Dataset with MockBenchmarkGuard."""

import json
import os
import shutil
import tempfile
from src.benchmark.runner.experiment_runner import run_benchmark_experiment


def test_pipeline_end_to_end_mock():
    temp_dir = tempfile.mkdtemp(prefix="guardslm_v3_test_")
    try:
        results = run_benchmark_experiment(
            dataset_path="data/GuardSLM_v3_Single_Master_Dataset.json",
            model_name="mock_guard",
            model_config={"mode": "perfect"},
            strategy="text_only",
            limit=20,
            output_dir=temp_dir,
            resume=True
        )

        assert "text_only" in results
        text_res = results["text_only"]
        assert text_res["total_cases_evaluated"] == 20
        assert text_res["decision_performance"]["accuracy"] == 1.0
        assert text_res["matched_pair_performance"]["total_pairs"] == 10
        assert text_res["matched_pair_performance"]["total_correct_pairs"] == 10

        # Check files created on disk
        preds_dir = os.path.join(temp_dir, "predictions")
        assert os.path.isdir(preds_dir)
        assert os.path.isfile(os.path.join(temp_dir, "report_text_only.md"))
        assert os.path.isfile(os.path.join(temp_dir, "evaluation_results.json"))
        assert os.path.isfile(os.path.join(temp_dir, "config_snapshot.json"))

        # Test Resumption: Re-running with resume=True should reuse predictions
        res_resumed = run_benchmark_experiment(
            dataset_path="data/GuardSLM_v3_Single_Master_Dataset.json",
            model_name="mock_guard",
            model_config={"mode": "perfect"},
            strategy="text_only",
            limit=20,
            output_dir=temp_dir,
            resume=True
        )
        assert res_resumed["text_only"]["total_cases_evaluated"] == 20
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
