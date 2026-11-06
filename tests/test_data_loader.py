"""Test 1: Dataset loading, isolation, and schema integrity."""

import os
import pytest
from guardslm.data import (
    TestCase,
    load_ground_truth,
    load_test_cases,
    natural_sort_key,
    parse_markdown_case,
    validate_cases_against_ground_truth,
)


def test_natural_sort_key():
    cases = ["TC10", "TC2", "TC1", "TC100", "TC20"]
    sorted_cases = sorted(cases, key=natural_sort_key)
    assert sorted_cases == ["TC1", "TC2", "TC10", "TC20", "TC100"]


def test_load_all_100_test_cases():
    cases = load_test_cases("research_data/slm_inputs")
    assert len(cases) == 100, f"Expected 100 cases, found {len(cases)}"

    # Deterministic natural ordering check
    case_ids = [c.case_id for c in cases]
    expected_first = ["TC01", "TC02", "TC03", "TC04", "TC05"]
    expected_last = ["TC96", "TC97", "TC98", "TC99", "TC100"]
    assert case_ids[:5] == expected_first
    assert case_ids[-5:] == expected_last

    # Uniqueness check
    assert len(set(case_ids)) == 100


def test_test_case_fields_and_zero_gold_leakage():
    cases = load_test_cases("research_data/slm_inputs")
    for case in cases:
        assert isinstance(case, TestCase)
        assert case.case_id.startswith("TC")
        assert len(case.title) > 0, f"{case.case_id} has empty title"
        assert len(case.user_request) > 0 or len(case.operation) > 0, f"{case.case_id} missing task/operation"
        assert len(case.policy_title) > 0 or len(case.policy_text) > 0, f"{case.case_id} missing policy"

        # Strict isolation check: TestCase dataclass must NOT have gold label attributes
        assert not hasattr(case, "expected_decision")
        assert not hasattr(case, "verdict")
        assert not hasattr(case, "ground_truth")


def test_load_isolated_ground_truth():
    gt = load_ground_truth("research_data/evaluation_only/ground_truth.csv")
    assert len(gt) == 100, f"Expected 100 GT rows, found {len(gt)}"

    # Exact gold label distribution check (48 ALLOW, 52 BLOCK)
    allow_count = sum(1 for v in gt.values() if v == "ALLOW")
    block_count = sum(1 for v in gt.values() if v == "BLOCK")
    assert allow_count == 48, f"Expected 48 ALLOW, got {allow_count}"
    assert block_count == 52, f"Expected 52 BLOCK, got {block_count}"
    assert allow_count + block_count == 100


def test_validation_against_ground_truth():
    cases = load_test_cases("research_data/slm_inputs")
    gt = load_ground_truth("research_data/evaluation_only/ground_truth.csv")
    # Must validate cleanly without exception
    validate_cases_against_ground_truth(cases, gt)
