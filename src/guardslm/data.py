"""Dataset loading and ground-truth isolation for GuardSLM benchmark."""

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TestCase:
    """Represents a sanitized model-visible test case.
    
    Contains only actor context, user request, action, policy, and page state.
    Never contains private ground-truth annotations or expected decisions.
    """
    # Prevent pytest from attempting to discover this dataclass as a test suite
    __test__ = False

    case_id: str
    title: str = ""
    username: str = ""
    display_name: str = ""
    global_role: str = ""
    effective_role: str = ""
    course: str = ""
    group_membership: str = ""
    user_request: str = ""
    operation: str = ""
    target_resource: str = ""
    target_owner: str = ""
    policy_title: str = ""
    policy_scope: str = ""
    policy_text: str = ""
    policy_evidence_file: str = ""
    a11y_tree: str = ""
    raw_content: str = ""


def natural_sort_key(key: str) -> List[object]:
    """Generates a key for natural numerical sorting (e.g., TC01, TC02, ..., TC100)."""
    parts = re.split(r'(\d+)', key)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def parse_markdown_case(content: str, filename: str = "") -> TestCase:
    """Parses a markdown test case file into a structured TestCase dataclass."""
    # 1. Extract Case ID and Title from the first header
    case_id = ""
    title = ""
    header_match = re.search(r'^#\s*(TC\d+)\s*(?:[—\-–]\s*(.*?))?$', content, re.MULTILINE)
    if header_match:
        case_id = header_match.group(1).strip()
        title = (header_match.group(2) or "").strip()
    elif filename:
        file_match = re.match(r'^(TC\d+)', os.path.basename(filename))
        if file_match:
            case_id = file_match.group(1).strip()

    if not case_id:
        raise ValueError(f"Could not extract a valid test case ID from file '{filename}'.")

    # Helper function to extract bullet items
    def extract_field(patterns: List[str]) -> str:
        for pat in patterns:
            m = re.search(pat, content, re.MULTILINE | re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""

    username = extract_field([r'^\s*-\s*Username:\s*(.*)$'])
    display_name = extract_field([r'^\s*-\s*Display name:\s*(.*)$'])
    global_role = extract_field([r'^\s*-\s*Global role:\s*(.*)$'])
    effective_role = extract_field([
        r'^\s*-\s*Effective role in the relevant course:\s*(.*)$',
        r'^\s*-\s*Effective role:\s*(.*)$'
    ])
    course = extract_field([r'^\s*-\s*Course:\s*(.*)$'])
    group_membership = extract_field([r'^\s*-\s*Group membership, if relevant:\s*(.*)$'])

    user_request = extract_field([
        r'^\s*-\s*Actor\'s task/request:\s*(.*)$',
        r'^\s*-\s*User Request:\s*(.*)$'
    ])
    operation = extract_field([
        r'^\s*-\s*Operation:\s*(.*)$',
        r'^\s*-\s*Requested operation:\s*(.*)$'
    ])
    target_resource = extract_field([r'^\s*-\s*Target resource:\s*(.*)$'])
    target_owner = extract_field([r'^\s*-\s*Target owner, if relevant:\s*(.*)$'])

    policy_title = extract_field([r'^\s*-\s*Policy title:\s*(.*)$'])
    policy_scope = extract_field([r'^\s*-\s*Policy scope:\s*(.*)$'])
    policy_text = extract_field([r'^\s*-\s*Policy text as displayed in the application:\s*(.*)$'])
    policy_evidence_file = extract_field([r'^\s*-\s*Policy evidence file:\s*(.*)$'])

    # Extract accessibility tree
    a11y_tree = ""
    a11y_match = re.search(
        r'##\s*(?:Current page accessibility snapshot|Accessibility snapshot).*?```(?:yaml)?\n(.*?)\n```',
        content,
        re.DOTALL | re.IGNORECASE
    )
    if a11y_match:
        a11y_tree = a11y_match.group(1).strip()

    return TestCase(
        case_id=case_id,
        title=title,
        username=username,
        display_name=display_name,
        global_role=global_role,
        effective_role=effective_role,
        course=course,
        group_membership=group_membership,
        user_request=user_request,
        operation=operation,
        target_resource=target_resource,
        target_owner=target_owner,
        policy_title=policy_title,
        policy_scope=policy_scope,
        policy_text=policy_text,
        policy_evidence_file=policy_evidence_file,
        a11y_tree=a11y_tree,
        raw_content=content
    )


def load_test_cases(input_dir: str = "research_data/slm_inputs") -> List[TestCase]:
    """Loads and sorts all valid markdown test cases from input directory.
    
    Returns cases sorted deterministically by case ID (TC01, TC02, ..., TC100).
    Ensures zero gold leakage into the returned structures.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Test cases directory '{input_dir}' does not exist.")

    case_files = [f for f in os.listdir(input_dir) if f.endswith(".md")]
    if not case_files:
        raise ValueError(f"No markdown (.md) test case files found in '{input_dir}'.")

    cases: List[TestCase] = []
    seen_ids = set()

    for fname in sorted(case_files, key=natural_sort_key):
        full_path = os.path.join(input_dir, fname)
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        case = parse_markdown_case(content, filename=fname)
        if case.case_id in seen_ids:
            raise ValueError(f"Duplicate test case ID '{case.case_id}' detected in '{fname}'.")
        seen_ids.add(case.case_id)
        cases.append(case)

    # Deterministic sort
    cases.sort(key=lambda c: natural_sort_key(c.case_id))
    return cases


def load_ground_truth(gt_path: str = "research_data/evaluation_only/ground_truth.csv") -> Dict[str, str]:
    """Loads private ground-truth annotations mapping case_id -> expected_decision (ALLOW | BLOCK).
    
    Kept strictly isolated from model inputs.
    """
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground-truth file '{gt_path}' does not exist.")

    annotations: Dict[str, str] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            case_id = (row.get("test_id") or row.get("case_id") or "").strip()
            decision = (row.get("expected_decision") or row.get("verdict") or row.get("ground_truth") or "").strip().upper()

            if not case_id:
                raise ValueError(f"Row {row_idx} in '{gt_path}' has missing test case ID.")
            if not decision:
                raise ValueError(f"Row {row_idx} in '{gt_path}' for case '{case_id}' has missing decision.")
            if decision not in {"ALLOW", "BLOCK", "ASK"}:
                raise ValueError(f"Unrecognized expected decision '{decision}' in row {row_idx} of '{gt_path}'.")

            if case_id in annotations:
                raise ValueError(f"Duplicate test ID '{case_id}' found in ground truth file '{gt_path}'.")

            annotations[case_id] = decision

    return annotations


def validate_cases_against_ground_truth(cases: List[TestCase], ground_truth: Dict[str, str]) -> None:
    """Validates that all test cases have matching ground truth annotations and vice-versa."""
    case_ids = {c.case_id for c in cases}
    gt_ids = set(ground_truth.keys())

    missing_in_gt = case_ids - gt_ids
    if missing_in_gt:
        raise ValueError(f"The following test cases are missing ground truth annotations: {sorted(missing_in_gt)}")

    missing_in_cases = gt_ids - case_ids
    if missing_in_cases:
        raise ValueError(f"The following ground truth IDs have no corresponding test case files: {sorted(missing_in_cases)}")
