"""Test 3: Model output parsing, label mapping, and error resilience."""

import pytest
from guardslm.parser import (
    parse_dynaguard_response,
    parse_policyguard_response,
    parse_response,
    parse_standard_response,
)


def test_parse_policyguard_labels():
    # ALLOW mappings
    res1 = parse_policyguard_response("no_violation")
    assert res1["verdict"] == "ALLOW"
    assert res1["native_label"] == "no_violation"
    assert res1["parse_status"] == "success"

    res2 = parse_policyguard_response("  No-Violation. \n")
    assert res2["verdict"] == "ALLOW"
    assert res2["parse_status"] == "success"

    # BLOCK mappings
    res3 = parse_policyguard_response("violation")
    assert res3["verdict"] == "BLOCK"
    assert res3["native_label"] == "violation"
    assert res3["parse_status"] == "success"

    res4 = parse_policyguard_response("  VIOLATION! ")
    assert res4["verdict"] == "BLOCK"
    assert res4["parse_status"] == "success"

    # Unrecognized / Non-native rejection
    res5 = parse_policyguard_response("ALLOW")
    assert res5["verdict"] == "UNKNOWN"
    assert res5["parse_status"] == "failed"

    res6 = parse_policyguard_response("PASS")
    assert res6["verdict"] == "UNKNOWN"
    assert res6["parse_status"] == "failed"

    res7 = parse_policyguard_response("I think this is a violation of the course rule.")
    assert res7["verdict"] == "UNKNOWN"
    assert res7["parse_status"] == "failed"


def test_parse_dynaguard_labels():
    # ALLOW mappings
    res1 = parse_dynaguard_response("PASS")
    assert res1["verdict"] == "ALLOW"
    assert res1["native_label"] == "PASS"
    assert res1["parse_status"] == "success"

    res2 = parse_dynaguard_response("<answer>\nPASS\n</answer>")
    assert res2["verdict"] == "ALLOW"
    assert res2["parse_status"] == "success"

    res3 = parse_dynaguard_response("PASS</s>")
    assert res3["verdict"] == "ALLOW"
    assert res3["parse_status"] == "success"

    # BLOCK mappings
    res4 = parse_dynaguard_response("FAIL")
    assert res4["verdict"] == "BLOCK"
    assert res4["native_label"] == "FAIL"
    assert res4["parse_status"] == "success"

    res5 = parse_dynaguard_response("<answer>\nFAIL\n</answer>")
    assert res5["verdict"] == "BLOCK"
    assert res5["parse_status"] == "success"

    # Unrecognized rejection
    res6 = parse_dynaguard_response("ALLOW")
    assert res6["verdict"] == "UNKNOWN"
    assert res6["parse_status"] == "failed"

    res7 = parse_dynaguard_response("violation")
    assert res7["verdict"] == "UNKNOWN"
    assert res7["parse_status"] == "failed"


def test_parse_standard_labels():
    # Standard verdicts
    assert parse_standard_response("ALLOW")["verdict"] == "ALLOW"
    assert parse_standard_response("BLOCK")["verdict"] == "BLOCK"
    assert parse_standard_response("ASK")["verdict"] == "ASK"

    # Native keywords inside text
    assert parse_standard_response("no_violation")["verdict"] == "ALLOW"
    assert parse_standard_response("violation")["verdict"] == "BLOCK"
    assert parse_standard_response("PASS")["verdict"] == "ALLOW"
    assert parse_standard_response("FAIL")["verdict"] == "BLOCK"

    # Empty / Malformed
    assert parse_standard_response("")["verdict"] == "UNKNOWN"
    assert parse_standard_response("   ")["parse_status"] == "failed"
    assert parse_standard_response("Unrelated explanation sentence.")["verdict"] == "UNKNOWN"


def test_parse_response_dispatch():
    assert parse_response("policyguard", "violation")["verdict"] == "BLOCK"
    assert parse_response("dynaguard", "PASS")["verdict"] == "ALLOW"
    assert parse_response("mock_guard", "ALLOW")["verdict"] == "ALLOW"
