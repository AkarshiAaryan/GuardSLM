"""GuardSLM Strict Model Output Parser with Schema Validation and Controlled Repair."""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Set, Optional, Tuple


VALID_VERDICTS = {"ALLOW", "BLOCK", "ASK"}
VALID_CHECK_STATUSES = {"not_applicable", "satisfied", "violated", "insufficient_evidence"}


@dataclass
class ParsedBenchmarkOutput:
    case_id: str
    verdict: str  # ALLOW, BLOCK, ASK, UNKNOWN
    applicable_policy_ids: List[str]
    violated_policy_ids: List[str]
    policy_checks: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    rationale: str
    parse_status: str  # success, repaired, failed
    validation_errors: List[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_json_block(text: str) -> Optional[str]:
    """Extracts JSON substring from markdown code blocks or curly braces."""
    text_clean = text.strip()

    # Case 1: Markdown code block ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_clean, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Case 2: Outermost curly braces
    first_brace = text_clean.find("{")
    last_brace = text_clean.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text_clean[first_brace:last_brace + 1].strip()

    return None


def parse_model_output(
    raw_text: str,
    expected_case_id: str,
    valid_policy_ids: Set[str]
) -> ParsedBenchmarkOutput:
    """Strictly parses and validates raw model text against the model_output_contract."""
    errors: List[str] = []

    json_str = extract_json_block(raw_text)
    if not json_str:
        return ParsedBenchmarkOutput(
            case_id=expected_case_id,
            verdict="UNKNOWN",
            applicable_policy_ids=[],
            violated_policy_ids=[],
            policy_checks=[],
            evidence=[],
            rationale="",
            parse_status="failed",
            validation_errors=["No JSON object found in model response."],
            raw_response=raw_text
        )

    try:
        data = json.loads(json_str)
    except Exception as e:
        return ParsedBenchmarkOutput(
            case_id=expected_case_id,
            verdict="UNKNOWN",
            applicable_policy_ids=[],
            violated_policy_ids=[],
            policy_checks=[],
            evidence=[],
            rationale="",
            parse_status="failed",
            validation_errors=[f"JSON syntax error: {e}"],
            raw_response=raw_text
        )

    if not isinstance(data, dict):
        return ParsedBenchmarkOutput(
            case_id=expected_case_id,
            verdict="UNKNOWN",
            applicable_policy_ids=[],
            violated_policy_ids=[],
            policy_checks=[],
            evidence=[],
            rationale="",
            parse_status="failed",
            validation_errors=["Parsed JSON is not an object."],
            raw_response=raw_text
        )

    # 1. Verdict validation
    raw_verdict = str(data.get("verdict", "")).strip().upper()
    if raw_verdict in VALID_VERDICTS:
        verdict = raw_verdict
    else:
        verdict = "UNKNOWN"
        errors.append(f"Invalid verdict '{raw_verdict}', expected one of {VALID_VERDICTS}")

    # 2. Case ID validation
    case_id = str(data.get("case_id", expected_case_id)).strip()
    if case_id != expected_case_id:
        errors.append(f"Output case_id '{case_id}' does not match expected '{expected_case_id}'")

    # 3. Applicable & Violated Policy IDs
    raw_app = data.get("applicable_policy_ids", [])
    applicable_policy_ids = [str(p) for p in raw_app] if isinstance(raw_app, list) else []
    for pid in applicable_policy_ids:
        if pid not in valid_policy_ids:
            errors.append(f"Applicable policy ID '{pid}' not in canonical policy book.")

    raw_viol = data.get("violated_policy_ids", [])
    violated_policy_ids = [str(p) for p in raw_viol] if isinstance(raw_viol, list) else []
    for pid in violated_policy_ids:
        if pid not in valid_policy_ids:
            errors.append(f"Violated policy ID '{pid}' not in canonical policy book.")

    # 4. Policy Checks Cardinality & Status
    raw_checks = data.get("policy_checks", [])
    policy_checks = []
    seen_check_ids = set()
    if isinstance(raw_checks, list):
        for c in raw_checks:
            if not isinstance(c, dict):
                continue
            pid = str(c.get("policy_id", "")).strip()
            status = str(c.get("status", "")).strip().lower()
            if pid not in valid_policy_ids:
                errors.append(f"Policy check references invalid policy_id '{pid}'")
            if status not in VALID_CHECK_STATUSES:
                errors.append(f"Policy check '{pid}' has invalid status '{status}'")
            seen_check_ids.add(pid)
            policy_checks.append({
                "policy_id": pid,
                "status": status,
                "evidence_quotes": c.get("evidence_quotes", [])
            })

        missing_checks = valid_policy_ids - seen_check_ids
        if missing_checks:
            errors.append(f"Policy checks missing {len(missing_checks)} policy clauses: {list(missing_checks)[:5]}")
    else:
        errors.append("policy_checks field is missing or not a list.")

    # 5. Evidence & Rationale
    raw_ev = data.get("evidence", [])
    evidence = raw_ev if isinstance(raw_ev, list) else []
    rationale = str(data.get("rationale", "")).strip()

    parse_status = "success" if not errors else ("failed" if verdict == "UNKNOWN" else "repaired")

    return ParsedBenchmarkOutput(
        case_id=case_id,
        verdict=verdict,
        applicable_policy_ids=applicable_policy_ids,
        violated_policy_ids=violated_policy_ids,
        policy_checks=policy_checks,
        evidence=evidence,
        rationale=rationale,
        parse_status=parse_status,
        validation_errors=errors,
        raw_response=raw_text
    )


def build_repair_prompt(raw_text: str, errors: List[str]) -> str:
    """Generates a controlled schema repair prompt that does not leak answers."""
    error_summary = "\n".join(f"- {err}" for err in errors[:5])
    return (
        f"Your previous response failed JSON schema validation for the following reason(s):\n"
        f"{error_summary}\n\n"
        f"Please re-output your authorization evaluation strictly as a single well-formed JSON object "
        f"matching the required output schema, with all 36 policy_checks included.\n"
        f"Output JSON only:"
    )
