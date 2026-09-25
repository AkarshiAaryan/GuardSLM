from typing import List, Dict, Any
from .result_schema import CaseResult

def analyze_failures(results: List[CaseResult]) -> Dict[str, Any]:
    """Analyzes prediction failures and categorizes error types."""
    failures: List[Dict[str, Any]] = []

    error_type_counts = {
        "false_allow": 0,
        "false_block": 0,
        "false_ask": 0,
        "parse_failure": 0,
        "model_error": 0
    }

    for r in results:
        if r.correct:
            continue

        error_type = "unknown_error"
        if r.error:
            error_type = "model_error"
        elif r.parse_status == "failed" or r.predicted_verdict == "UNKNOWN":
            error_type = "parse_failure"
        elif r.gold_verdict in ["BLOCK", "ASK"] and r.predicted_verdict == "ALLOW":
            error_type = "false_allow"
        elif r.gold_verdict == "ALLOW" and r.predicted_verdict == "BLOCK":
            error_type = "false_block"
        elif r.predicted_verdict == "ASK":
            error_type = "false_ask"

        error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

        failures.append({
            "case_id": r.case_id,
            "pair_id": r.pair_id,
            "template_id": r.template_id,
            "gold_verdict": r.gold_verdict,
            "predicted_verdict": r.predicted_verdict,
            "error_type": error_type,
            "raw_response": r.raw_response,
            "error_detail": r.error
        })

    return {
        "total_failures": len(failures),
        "error_type_counts": error_type_counts,
        "failure_cases": failures
    }
