"""Response parsing and verdict normalization for GuardSLM models."""

import re
from typing import Any, Dict, Optional


def parse_policyguard_response(raw_response: str) -> Dict[str, Any]:
    """Parses PolicyGuard binary classification outputs into standardized verdicts.

    Native labels:
    - 'no_violation' / 'non_violation' -> ALLOW
    - 'violation' -> BLOCK
    All other outputs produce UNKNOWN with parse_status='failed'.
    """
    clean_text = raw_response.strip().strip("'\"`.,;:!").strip()
    clean_norm = re.sub(r'[\s-]+', '_', clean_text.lower())

    if clean_norm in ("no_violation", "non_violation"):
        return {
            "verdict": "ALLOW",
            "native_label": "no_violation",
            "parse_status": "success",
            "parsing_error": None
        }
    elif clean_norm == "violation":
        return {
            "verdict": "BLOCK",
            "native_label": "violation",
            "parse_status": "success",
            "parsing_error": None
        }

    return {
        "verdict": "UNKNOWN",
        "native_label": None,
        "parse_status": "failed",
        "parsing_error": f"Unrecognized PolicyGuard label in '{raw_response}'"
    }


def parse_dynaguard_response(raw_response: str) -> Dict[str, Any]:
    """Parses DynaGuard compliance output into standardized verdicts.

    Native labels:
    - 'PASS' -> ALLOW
    - 'FAIL' -> BLOCK
    All other outputs produce UNKNOWN with parse_status='failed'.
    """
    # Strip XML tags, special tokens, punctuation, and whitespace
    cleaned = re.sub(r'</?(?:answer|s|eos)>|<\|[^>]+?\|>', '', raw_response, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("'\"`.,;:!\n\r ").strip().upper()

    if cleaned == "PASS":
        return {
            "verdict": "ALLOW",
            "native_label": "PASS",
            "parse_status": "success",
            "parsing_error": None
        }
    elif cleaned == "FAIL":
        return {
            "verdict": "BLOCK",
            "native_label": "FAIL",
            "parse_status": "success",
            "parsing_error": None
        }

    return {
        "verdict": "UNKNOWN",
        "native_label": None,
        "parse_status": "failed",
        "parsing_error": f"Unrecognized DynaGuard label in '{raw_response}'"
    }


def parse_standard_response(raw_response: str) -> Dict[str, Any]:
    """Parses standard/generic/mock model output (ALLOW, BLOCK, ASK, or native tokens)."""
    if not raw_response or not raw_response.strip():
        return {
            "verdict": "UNKNOWN",
            "native_label": None,
            "parse_status": "failed",
            "parsing_error": "Empty model response"
        }

    clean = raw_response.strip().strip("'\"`.,;:! \n\r").upper()

    # Direct match on standard verdicts
    if clean in ("ALLOW", "BLOCK", "ASK"):
        return {
            "verdict": clean,
            "native_label": clean,
            "parse_status": "success",
            "parsing_error": None
        }

    # Check for PolicyGuard native keywords
    lower_norm = re.sub(r'[\s-]+', '_', raw_response.strip().lower())
    if "no_violation" in lower_norm:
        return {
            "verdict": "ALLOW",
            "native_label": "no_violation",
            "parse_status": "success",
            "parsing_error": None
        }
    elif "violation" in lower_norm:
        return {
            "verdict": "BLOCK",
            "native_label": "violation",
            "parse_status": "success",
            "parsing_error": None
        }

    # Check for DynaGuard native keywords
    dyna_cleaned = re.sub(r'</?(?:answer|s|eos)>|<\|[^>]+?\|>', '', raw_response, flags=re.IGNORECASE).strip().upper()
    if dyna_cleaned == "PASS":
        return {
            "verdict": "ALLOW",
            "native_label": "PASS",
            "parse_status": "success",
            "parsing_error": None
        }
    elif dyna_cleaned == "FAIL":
        return {
            "verdict": "BLOCK",
            "native_label": "FAIL",
            "parse_status": "success",
            "parsing_error": None
        }

    # Exact token boundary regex for ALLOW / BLOCK / ASK
    tokens = re.findall(r'\b(ALLOW|BLOCK|ASK)\b', clean)
    if len(tokens) == 1:
        return {
            "verdict": tokens[0],
            "native_label": tokens[0],
            "parse_status": "success",
            "parsing_error": None
        }

    return {
        "verdict": "UNKNOWN",
        "native_label": None,
        "parse_status": "failed",
        "parsing_error": f"Unable to parse verdict from response: '{raw_response}'"
    }


def parse_response(model_name: str, raw_response: str) -> Dict[str, Any]:
    """Dispatches raw response to model-specific parser and standardizes decision."""
    m_lower = model_name.lower()
    if "policyguard" in m_lower:
        return parse_policyguard_response(raw_response)
    elif "dynaguard" in m_lower:
        return parse_dynaguard_response(raw_response)
    else:
        return parse_standard_response(raw_response)
