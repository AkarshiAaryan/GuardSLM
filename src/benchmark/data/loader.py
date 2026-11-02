"""GuardSLM v3 Master Dataset Loader with Strict Input/Gold Separation."""

import json
import os
from typing import Dict, Any, List, Optional
from .schema import (
    ModelInputCase,
    PrivateGoldAnnotation,
    PairRegistryEntry,
    PolicyClause
)
from .validator import validate_benchmark_dataset


class BenchmarkLoader:
    """Canonical dataset loader strictly isolating model inputs from private gold annotations."""

    def __init__(
        self,
        dataset_path: str = "data/GuardSLM_v3_Single_Master_Dataset.json",
        validate_on_load: bool = True
    ):
        self.dataset_path = dataset_path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Canonical benchmark dataset not found: {dataset_path}")

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
