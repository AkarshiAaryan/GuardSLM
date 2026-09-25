import json
import os
from typing import Dict, Any, Tuple, List

def validate_dataset_file(dataset_path: str, schema_path: str = "schemas/test_case_schema.json") -> Tuple[bool, List[str]]:
    """Validates a dataset JSON file against the JSON schema and structural integrity checks.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    if not os.path.exists(dataset_path):
        return False, [f"Dataset file does not exist: {dataset_path}"]
    if not os.path.exists(schema_path):
        return False, [f"Schema file does not exist: {schema_path}"]

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"Failed to parse JSON in {dataset_path}: {e}"]

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except Exception as e:
        return False, [f"Failed to parse schema JSON in {schema_path}: {e}"]

    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(schema)
        schema_errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
        for err in schema_errors:
            path_str = " -> ".join([str(p) for p in err.path])
            errors.append(f"Schema error at [{path_str}]: {err.message}")
    except ImportError:
        # Basic structural validation fallback if jsonschema package is not installed
        if isinstance(data, dict):
            if "cases" not in data:
                errors.append("Dataset dictionary missing 'cases' key.")
            cases = data.get("cases", [])
        elif isinstance(data, list):
            cases = data
        else:
            errors.append("Dataset root must be an object or a list.")
            cases = []

        for idx, case in enumerate(cases):
            for req_field in ["case_id", "pair_id", "template_id", "user", "request", "context", "proposed_action", "gold"]:
                if req_field not in case:
                    errors.append(f"Case #{idx} ({case.get('case_id', 'unknown')}) missing required field '{req_field}'.")

    # Additional semantic checks: pair completeness
    if isinstance(data, dict) and "cases" in data:
        cases = data["cases"]
    elif isinstance(data, list):
        cases = data
    else:
        cases = []

    pair_counts: Dict[str, List[str]] = {}
    for case in cases:
        pid = case.get("pair_id")
        cid = case.get("case_id")
        if pid and cid:
            pair_counts.setdefault(pid, []).append(cid)

    for pid, cids in pair_counts.items():
        if len(cids) != 2:
            errors.append(f"Pair '{pid}' does not have exactly 2 cases (found {len(cids)}: {cids}).")

    is_valid = len(errors) == 0
    return is_valid, errors
