"""Test 6: Resumption, incremental caching, and end-to-end runner orchestration."""

import os
import shutil
import tempfile
import pytest
from guardslm.runner import run_experiment


def test_runner_orchestration_and_resume():
    temp_dir = tempfile.mkdtemp(prefix="guardslm_test_resume_")

    try:
        # Phase 1: Run limit 3 cases
        res1 = run_experiment(
            model_name="mock_guard",
            limit=3,
            output_dir=temp_dir,
            resume=True
        )

        assert res1["predictions_count"] == 3
        jsonl_path = res1["jsonl_path"]
        assert os.path.isfile(jsonl_path)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 3

        # Phase 2: Resume run with limit 6 cases on the SAME output_dir
        res2 = run_experiment(
            model_name="mock_guard",
            limit=6,
            output_dir=temp_dir,
            resume=True
        )

        assert res2["predictions_count"] == 6

        # Check that predictions.jsonl now has exactly 6 records (cases 1-3 were cached, 4-6 added)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines2 = [l for l in f if l.strip()]
        assert len(lines2) == 6

        # Check all artifacts were created
        assert os.path.isfile(os.path.join(temp_dir, "predictions.csv"))
        assert os.path.isfile(os.path.join(temp_dir, "evaluation.json"))
        assert os.path.isfile(os.path.join(temp_dir, "report.md"))
        assert os.path.isfile(os.path.join(temp_dir, "run_metadata.json"))

        # Verify evaluation evaluates on all 6 cases
        assert res2["evaluation"]["total_cases"] == 6
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
