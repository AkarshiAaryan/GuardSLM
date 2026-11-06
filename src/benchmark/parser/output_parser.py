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
    verdict: str  # ALLOW, BLOCK, ASK, UNKNOWN (serves as benchmark prediction)
    applicable_policy_ids: List[str]
    violated_policy_ids: List[str]
    policy_checks: List[Dict[str, Any]]
    evidence: List[Any]
    rationale: str
    parse_status: str  # success, repaired, failed (serves as parsing_status)
    validation_errors: List[str] = field(default_factory=list)
    raw_response: str = ""  # (serves as raw_output)
    native_label: Optional[str] = None
    parsing_error: Optional[str] = None
    raw_output: Optional[str] = None
    prediction: Optional[str] = None
    parsing_status: Optional[str] = None

    def __post_init__(self):
        if self.raw_output is None:
            self.raw_output = self.raw_response
        if not self.raw_response and self.raw_output:
            self.raw_response = self.raw_output
        if self.prediction is None:
            self.prediction = self.verdict
        if not self.verdict and self.prediction:
            self.verdict = self.prediction
        if self.parsing_status is None:
            self.parsing_status = self.parse_status
        if not self.parse_status and self.parsing_status:
            self.parse_status = self.parsing_status

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParsedBenchmarkOutput":
        fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in fields}
        return cls(**filtered)

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
    data = None
    if json_str:
        try:
            data = json.loads(json_str)
        except Exception:
            # Resilient JSON repair for truncated model outputs
            repaired = json_str.strip()
            repaired = re.sub(r',\s*"[^"]*":?\s*"?$', '', repaired)
            repaired = re.sub(r',\s*$', '', repaired)
            unescaped_quotes = len(re.findall(r'(?<!\\)"', repaired))
            if unescaped_quotes % 2 != 0:
                repaired += '"'
            open_sq = repaired.count("[") - repaired.count("]")
            if open_sq > 0:
                repaired += "]" * open_sq
            open_cur = repaired.count("{") - repaired.count("}")
            if open_cur > 0:
                repaired += "}" * open_cur
            try:
                data = json.loads(repaired)
            except Exception:
                pass

    # Regex extraction fallback if JSON parsing could not be completed
    if not isinstance(data, dict):
        data = {}
        v_match = re.search(r'"verdict"\s*:\s*"(ALLOW|BLOCK|ASK)"', raw_text, re.IGNORECASE)
        if not v_match:
            v_match = re.search(r'\b(ALLOW|BLOCK|ASK)\b', raw_text)
        if v_match:
            data["verdict"] = v_match.group(1).upper()

        c_match = re.search(r'"case_id"\s*:\s*"([^"]+)"', raw_text)
        if c_match:
            data["case_id"] = c_match.group(1)
        rat_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', raw_text)
        if rat_match:
            data["rationale"] = rat_match.group(1)
        app_match = re.search(r'"applicable_policy_ids"\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
        if app_match:
            data["applicable_policy_ids"] = re.findall(r'"([^"]+)"', app_match.group(1))
        viol_match = re.search(r'"violated_policy_ids"\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
        if viol_match:
            data["violated_policy_ids"] = re.findall(r'"([^"]+)"', viol_match.group(1))

    if not data or not data.get("verdict"):
        return ParsedBenchmarkOutput(
            case_id=expected_case_id,
            verdict="UNKNOWN",
            applicable_policy_ids=[],
            violated_policy_ids=[],
            policy_checks=[],
            evidence=[],
            rationale="",
            parse_status="failed",
            validation_errors=["Failed to extract valid verdict or JSON structure from model response."],
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

    # Auto-fill unmentioned policy clauses as not_applicable to ensure complete 36-policy coverage
    missing_checks = valid_policy_ids - seen_check_ids
    if missing_checks:
        errors.append(f"Policy checks missing {len(missing_checks)} policy clauses: {list(missing_checks)[:5]}")
        for pid in missing_checks:
            policy_checks.append({
                "policy_id": pid,
                "status": "not_applicable",
                "evidence_quotes": []
            })

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
