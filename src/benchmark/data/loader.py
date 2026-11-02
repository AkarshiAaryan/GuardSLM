"""GuardSLM 100-Pair Benchmark Loader with Strict Input/Gold Separation."""

import json
import os
from typing import Dict, Any, List, Optional
from .schema import (
    ModelInputCase,
    PrivateGoldAnnotation,
    PairRegistryEntry,
    PolicyClause
)
from ..data.validator import validate_benchmark_dataset


class BenchmarkLoader:
    """Canonical dataset loader strictly isolating model inputs from private gold annotations."""

    def __init__(self, dataset_path: str = "data/GuardSLM_100_Pairs_v2.json", validate_on_load: bool = True):
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

        # Parse policy clauses
        self._policy_book: List[PolicyClause] = [
            PolicyClause.from_dict(p) for p in self._raw_data.get("complete_policy_book", [])
        ]

        # Parse model inputs (public facing)
        self._model_inputs: List[ModelInputCase] = [
            ModelInputCase.from_dict(m) for m in self._raw_data.get("model_inputs", [])
        ]

        # Parse private gold annotations (isolated)
        self._private_gold: Dict[str, PrivateGoldAnnotation] = {
            g["case_id"]: PrivateGoldAnnotation.from_dict(g)
            for g in self._raw_data.get("private_gold_annotations", [])
        }

        # Parse pair registry
        self._pair_registry: Dict[str, PairRegistryEntry] = {
            p["pair_id"]: PairRegistryEntry.from_dict(p)
            for p in self._raw_data.get("pair_registry", [])
        }

    def get_model_inputs(self, case_ids: Optional[List[str]] = None, limit: Optional[int] = None) -> List[ModelInputCase]:
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
        """Returns the canonical 36-clause policy book."""
        return self._policy_book

    def get_source_pdf_manifest(self) -> List[Dict[str, Any]]:
        """Returns the source PDF manifest."""
        return self._raw_data.get("source_pdf_manifest", [])

    def get_contract(self) -> Dict[str, Any]:
        """Returns the model output contract specification."""
        return self.model_output_contract
