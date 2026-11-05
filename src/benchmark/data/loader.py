"""GuardSLM v3 Master Dataset Loader with Strict Input/Gold Separation."""

import csv
import glob
import json
import os
import re
from typing import Dict, Any, List, Optional
from .schema import (
    ModelInputCase,
    SnapshotInputCase,
    PrivateGoldAnnotation,
    PairRegistryEntry,
    PolicyClause,
    PolicyCheck
)
from .validator import validate_benchmark_dataset


class BenchmarkLoader:
    """Canonical dataset loader strictly isolating model inputs from private gold annotations."""

    def __init__(
        self,
        dataset_path: str = "research_snapshots",
        validate_on_load: bool = True
    ):
        self.dataset_path = dataset_path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Canonical benchmark dataset not found: {dataset_path}")

        # Check if loading from research_snapshots directory
        if os.path.isdir(dataset_path):
            self._load_from_research_snapshots(dataset_path)
            return

        if validate_on_load:
            val_res = validate_benchmark_dataset(dataset_path)
            if not val_res.is_valid:
                critical_msgs = [i.message for i in val_res.issues if i.severity == "CRITICAL"]
                raise ValueError(f"Dataset failed structural validation: {critical_msgs}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            self._raw_data = json.load(f)

        self.dataset_name: str = self._raw_data.get("dataset_name", "")
        self.dataset_version: str = self._raw_data.get("dataset_version", "")
        self.model_output_contract: Dict[str, Any] = self._raw_data.get("model_output_contract", {})

        # Load cases (v3 single master dataset format)
        if "cases" in self._raw_data:
            cases_raw = self._raw_data["cases"]
            self._model_inputs: List[ModelInputCase] = [
                ModelInputCase.from_dict(c) for c in cases_raw
            ]
            self._private_gold: Dict[str, PrivateGoldAnnotation] = {
                c["case_id"]: PrivateGoldAnnotation.from_dict(c["private_gold"])
                for c in cases_raw
            }
            # Policy book proxy from first case policy_info
            first_policies = cases_raw[0].get("model_input", {}).get("policy_info", []) if cases_raw else []
            self._policy_book: List[PolicyClause] = [
                PolicyClause(policy_id=f"POL-{i+1:02d}", statement=s, name=f"Policy {i+1}")
                for i, s in enumerate(first_policies)
            ]
        else:
            # Legacy v2 fallback
            self._model_inputs = [
                ModelInputCase.from_dict(m) for m in self._raw_data.get("model_inputs", [])
            ]
            self._private_gold = {
                g["case_id"]: PrivateGoldAnnotation.from_dict(g)
                for g in self._raw_data.get("private_gold_annotations", [])
            }
            self._policy_book = [
                PolicyClause.from_dict(p) for p in self._raw_data.get("complete_policy_book", [])
            ]

        # Parse pair registry
        self._pair_registry: Dict[str, PairRegistryEntry] = {
            p["pair_id"]: PairRegistryEntry.from_dict(p)
            for p in self._raw_data.get("pair_registry", [])
        }

    def _load_from_research_snapshots(self, directory_path: str):
        """Loads benchmark test cases directly from research_snapshots markdown and ground truth CSV."""
        slm_inputs_dir = os.path.join(directory_path, "slm_inputs")
        ground_truth_file = os.path.join(directory_path, "evaluation_only", "ground_truth.csv")

        if not os.path.isdir(slm_inputs_dir):
            raise FileNotFoundError(f"Snapshot slm_inputs directory not found at {slm_inputs_dir}")
        if not os.path.isfile(ground_truth_file):
            raise FileNotFoundError(f"Snapshot ground_truth.csv not found at {ground_truth_file}")

        self.dataset_name = "GuardSLM LMS Research Snapshots Dataset"
        self.dataset_version = "1.0.0-snapshot-accompanied"
        self.model_output_contract = {
            "verdict": ["ALLOW", "BLOCK", "ASK"],
            "modalities": ["text_only", "accessibility_tree"]
        }

        # 1. Parse ground truth CSV
        gold_raw = {}
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gold_raw[row["case_id"].strip()] = row

        # 2. Parse all Markdown files in slm_inputs
        files = sorted(glob.glob(os.path.join(slm_inputs_dir, "TC*.md")))
        self._model_inputs = []
        self._private_gold = {}

        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            title_match = re.search(r"^#\s*(TC\d+)\s*—\s*(.*)$", text, re.MULTILINE)
            case_id = title_match.group(1).strip() if title_match else os.path.basename(path).split("_")[0]
            title = title_match.group(2).strip() if title_match else ""

            account = re.search(r"- Account:\s*(.*)", text)
            global_role = re.search(r"- Global account role:\s*(.*)", text)
            effective_role = re.search(r"- Effective role in this course:\s*(.*)", text)
            membership = re.search(r"- Course membership / relevant group membership:\s*(.*)", text)

            timestamp = re.search(r"- Capture timestamp:\s*(.*)", text)
            timezone = re.search(r"- Timezone used by the application:\s*(.*)", text)
            deadline_ts = re.search(r"- Relevant deadline/open/close/release timestamp, if any:\s*(.*)", text)

            resource = re.search(r"- Course/resource:\s*(.*)", text)
            operation = re.search(r"- Requested operation:\s*(.*)", text)
            target_owner = re.search(r"- Target owner/group, if relevant:\s*(.*)", text)

            policy_match = re.search(r"## Applicable policy context\s*\n(.*?)\n## Accessibility snapshot", text, re.DOTALL)
            policy_context = policy_match.group(1).strip() if policy_match else ""

            yaml_blocks = re.findall(r"```yaml\s*\n(.*?)\n```", text, re.DOTALL)
            target_a11y = yaml_blocks[0] if len(yaml_blocks) > 0 else ""
            policy_a11y = yaml_blocks[1] if len(yaml_blocks) > 1 else ""

            course_scope = "LMS"
            if resource:
                res_parts = resource.group(1).strip().split("/")
                course_scope = res_parts[0].strip()

            snapshot_case = SnapshotInputCase(
                case_id=case_id,
                title=title,
                account=account.group(1).strip() if account else "",
                global_role=global_role.group(1).strip() if global_role else "",
                effective_role=effective_role.group(1).strip() if effective_role else "",
                membership=membership.group(1).strip() if membership else "",
                timestamp=timestamp.group(1).strip() if timestamp else "",
                timezone=timezone.group(1).strip() if timezone else "UTC",
                deadline_ts=deadline_ts.group(1).strip() if deadline_ts else "None",
                course_or_scope=course_scope,
                target_resource=resource.group(1).strip() if resource else "",
                requested_operation=operation.group(1).strip() if operation else "",
                target_owner=target_owner.group(1).strip() if target_owner else "",
                policy_context=policy_context,
                target_a11y_tree=target_a11y,
                policy_a11y_tree=policy_a11y,
                pair_id="",
                variant=""
            )
            self._model_inputs.append(snapshot_case)

            # Build PrivateGoldAnnotation strictly isolated
            gold_row = gold_raw.get(case_id, {})
            verdict = gold_row.get("expected_decision", "ALLOW").strip().upper()
            rationale = gold_row.get("rationale", "")

            gold_anno = PrivateGoldAnnotation(
                case_id=case_id,
                pair_id="",
                verdict=verdict,
                applicable_policy_ids=["POL-SNAPSHOT"],
                violated_policy_ids=["POL-SNAPSHOT"] if verdict == "BLOCK" else [],
                policy_checks=[
                    PolicyCheck(
                        policy_id="POL-SNAPSHOT",
                        status="violated" if verdict == "BLOCK" else "satisfied",
                        evidence_quotes=[]
                    )
                ],
                evidence=[],
                rationale=rationale,
                annotation_status="canonical_snapshot",
                gold_provenance="research_snapshots/evaluation_only/ground_truth.csv"
            )
            self._private_gold[case_id] = gold_anno

        # Policy book
        self._policy_book = [
            PolicyClause(policy_id="POL-PRIVACY", statement="Institutional Grade & Submission Privacy Policy", name="Privacy"),
            PolicyClause(policy_id="POL-ROLES", statement="Course-Scoped Role Rule", name="Course Roles"),
            PolicyClause(policy_id="POL-DEADLINES", statement="Submission Deadlines & Extension Rules", name="Deadlines"),
            PolicyClause(policy_id="POL-ASSESSMENT", statement="Assessment Quota & Answer Key Policy", name="Assessments"),
            PolicyClause(policy_id="POL-GRADING", statement="Final Grade Publication & Locking Policy", name="Grade Locking"),
            PolicyClause(policy_id="POL-ADMIN", statement="Institutional Administration Governance Policy", name="Administration")
        ]

        # 3. Independent evaluation: pair registry is empty for snapshots
        self._pair_registry = {}


    def get_model_inputs(
        self,
        case_ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[ModelInputCase]:
        """Returns public model inputs strictly sanitized of any gold labels or expected verdicts."""
        cases = self._model_inputs
        if case_ids is not None:
            id_set = set(case_ids)
            cases = [c for c in cases if c.case_id in id_set]
        if limit is not None and limit > 0:
            cases = cases[:limit]
        return cases

    def get_gold_annotations(self) -> Dict[str, PrivateGoldAnnotation]:
        """Provides access to private gold annotations strictly for post-inference evaluation."""
        return self._private_gold

    def get_gold_for_case(self, case_id: str) -> Optional[PrivateGoldAnnotation]:
        """Fetches private gold annotation for a single case during post-inference evaluation."""
        return self._private_gold.get(case_id)

    def get_pair_registry(self) -> Dict[str, PairRegistryEntry]:
        """Returns the pair registry for stratification and matched-pair analysis."""
        return self._pair_registry

    def get_policy_book(self) -> List[PolicyClause]:
        """Returns the canonical policy clauses."""
        return self._policy_book

    def get_contract(self) -> Dict[str, Any]:
        """Returns the model output contract specification."""
        return self.model_output_contract
