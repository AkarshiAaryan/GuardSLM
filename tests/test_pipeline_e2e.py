"""End-to-End Pipeline integration test using MockBenchmarkGuard across Strategy A and B."""

import json
import os
import shutil
import tempfile
from src.benchmark.runner.experiment_runner import run_benchmark_experiment


def test_pipeline_end_to_end_mock():
    temp_dir = tempfile.mkdtemp(prefix="guardslm_test_")
    try:
        results = run_benchmark_experiment(
            dataset_path="data/GuardSLM_100_Pairs_v2.json",
            model_name="mock_guard",
            model_config={"mode": "perfect"},
            strategy="both",
            limit=4,
            output_dir=temp_dir,
            resume=True
        )

        assert "text_only" in results
        assert "text_plus_screenshot" in results

        text_res = results["text_only"]
        assert text_res["total_cases_evaluated"] == 4
        assert text_res["decision_performance"]["accuracy"] == 1.0

        # Check files created on disk
        preds_dir = os.path.join(temp_dir, "predictions")
        assert os.path.isdir(preds_dir)
        assert os.path.isfile(os.path.join(temp_dir, "report_text_only.md"))
        assert os.path.isfile(os.path.join(temp_dir, "report_text_plus_screenshot.md"))
        assert os.path.isfile(os.path.join(temp_dir, "evaluation_results.json"))
        assert os.path.isfile(os.path.join(temp_dir, "config_snapshot.json"))

        # Test Resumption: Re-running with resume=True should reuse predictions
        res_resumed = run_benchmark_experiment(
            dataset_path="data/GuardSLM_100_Pairs_v2.json",
            model_name="mock_guard",
            model_config={"mode": "perfect"},
            strategy="both",
            limit=4,
            output_dir=temp_dir,
            resume=True
        )
        assert res_resumed["text_only"]["total_cases_evaluated"] == 4
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
