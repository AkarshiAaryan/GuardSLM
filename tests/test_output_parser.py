"""Test strict output parser, contract enforcement, and schema validation."""

import json
from src.benchmark.parser.output_parser import (
    parse_model_output,
    build_repair_prompt,
    ParsedBenchmarkOutput
)


SAMPLE_VALID_POLICY_IDS = {
    f"POL-TEST-{i:02d}" for i in range(1, 37)
}


def test_valid_json_output_parsing():
    policy_ids = sorted(list(SAMPLE_VALID_POLICY_IDS))
    valid_payload = {
        "case_id": "GSLM-P001-A",
        "verdict": "BLOCK",
        "applicable_policy_ids": [policy_ids[0]],
        "violated_policy_ids": [policy_ids[0]],
        "policy_checks": [
            {
                "policy_id": pid,
                "status": "violated" if pid == policy_ids[0] else "not_applicable",
                "evidence_quotes": ["Evidence quote"] if pid == policy_ids[0] else []
            }
            for pid in policy_ids
        ],
        "evidence": [
            {
                "policy_id": policy_ids[0],
                "quote": "Evidence quote",
                "source_field_path": "actor_context.authenticated_role"
            }
        ],
        "rationale": "Action violates rule 1."
    }

    raw_text = f"```json\n{json.dumps(valid_payload, indent=2)}\n```"
    parsed = parse_model_output(raw_text, "GSLM-P001-A", SAMPLE_VALID_POLICY_IDS)

    assert parsed.parse_status == "success"
    assert parsed.verdict == "BLOCK"
    assert parsed.case_id == "GSLM-P001-A"
    assert len(parsed.violated_policy_ids) == 1
    assert len(parsed.policy_checks) == 36
    assert len(parsed.validation_errors) == 0


def test_invalid_verdict_rejection():
    raw_text = json.dumps({
        "case_id": "GSLM-P001-A",
        "verdict": "PERMITTED",  # Not in ALLOW/BLOCK/ASK
        "applicable_policy_ids": [],
        "violated_policy_ids": [],
        "policy_checks": [],
        "evidence": [],
        "rationale": "Invalid enum test"
    })
    parsed = parse_model_output(raw_text, "GSLM-P001-A", SAMPLE_VALID_POLICY_IDS)

    assert parsed.verdict == "UNKNOWN"
    assert parsed.parse_status == "failed"
    assert any("Invalid verdict" in err for err in parsed.validation_errors)


def test_missing_policy_checks_cardinality():
    policy_ids = sorted(list(SAMPLE_VALID_POLICY_IDS))
    # Provide only 1 policy check instead of 36
    raw_text = json.dumps({
        "case_id": "GSLM-P001-A",
        "verdict": "ALLOW",
        "applicable_policy_ids": [],
        "violated_policy_ids": [],
        "policy_checks": [
            {"policy_id": policy_ids[0], "status": "satisfied", "evidence_quotes": []}
        ],
        "evidence": [],
        "rationale": "Only one check provided."
    })
    parsed = parse_model_output(raw_text, "GSLM-P001-A", SAMPLE_VALID_POLICY_IDS)

    assert any("missing 35 policy clauses" in err for err in parsed.validation_errors)


def test_repair_prompt_generation():
    repair_prompt = build_repair_prompt(
        "broken response",
        ["Invalid verdict 'MAYBE'", "policy_checks missing 35 clauses"]
    )
    assert "Your previous response failed JSON schema validation" in repair_prompt
    assert "Invalid verdict 'MAYBE'" in repair_prompt


def test_truncated_json_recovery():
    # Simulates model running out of tokens while generating policy checks
    truncated_raw = """```json
{
  "case_id": "GSLM-P004-A",
  "verdict": "BLOCK",
  "applicable_policy_ids": ["POL-TEST-01"],
  "violated_policy_ids": ["POL-TEST-01"],
  "rationale": "Actor lacks authorization.",
  "policy_checks": [
    {
      "policy_id": "POL-TEST-01",
      "status": "violated",
      "evidence_quotes": ["quote
"""
    parsed = parse_model_output(truncated_raw, "GSLM-P004-A", SAMPLE_VALID_POLICY_IDS)
    assert parsed.verdict == "BLOCK"
    assert parsed.case_id == "GSLM-P004-A"
    assert "POL-TEST-01" in parsed.violated_policy_ids
    assert len(parsed.policy_checks) == 36  # Automatically filled to 36!

