"""CPU Mock Benchmark Guard Model for testing GuardSLM 100-Pair Pipeline."""

import json
import time
from typing import Dict, Any, Optional
from .base import BenchmarkModel, BenchmarkPrediction
from ..data.schema import ModelInputCase
from ..parser.output_parser import parse_model_output


class MockBenchmarkGuard(BenchmarkModel):
    """Simulated guard model supporting various behavior modes for fast, deterministic pipeline testing."""

    def __init__(self, name: str = "mock_benchmark_guard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.mode = self.config.get("mode", "perfect")
        self.oracle_golds = self.config.get("oracle_golds", {})

    def load(self) -> None:
        pass

    def predict(
        self,
        case: Any,
        strategy: str = "text_only",
        screenshot_path: Optional[str] = None
    ) -> BenchmarkPrediction:
        start_time = time.time()
        policy_ids = [p.policy_id for p in case.policy_book]
        valid_policy_set = set(policy_ids)

        if self.mode == "perfect" and case.case_id in self.oracle_golds:
            gold = self.oracle_golds[case.case_id]
            verdict = gold.verdict
            applicable = gold.applicable_policy_ids
            violated = gold.violated_policy_ids
            policy_checks = [c.to_dict() for c in gold.policy_checks]
            evidence = gold.evidence
            rationale = gold.rationale
        elif self.mode == "always_block":
            verdict = "BLOCK"
            applicable = [policy_ids[0]] if policy_ids else []
            violated = [policy_ids[0]] if policy_ids else []
            policy_checks = [
                {
                    "policy_id": pid,
                    "status": "violated" if pid in violated else "not_applicable",
                    "evidence_quotes": ["Simulated block"] if pid in violated else []
                }
                for pid in policy_ids
            ]
            evidence = [{"policy_id": violated[0], "quote": "Simulated", "source_field_path": "mock"}] if violated else []
            rationale = "Mock always_block mode."
        elif self.mode == "always_allow":
            verdict = "ALLOW"
            applicable = [policy_ids[0]] if policy_ids else []
            violated = []
            policy_checks = [
                {
                    "policy_id": pid,
                    "status": "satisfied" if pid in applicable else "not_applicable",
                    "evidence_quotes": []
                }
                for pid in policy_ids
            ]
            evidence = []
            rationale = "Mock always_allow mode."
        elif self.mode == "malformed_json":
            raw_text = "This is not valid JSON at all!"
            latency_ms = (time.time() - start_time) * 1000.0
            parsed = parse_model_output(raw_text, case.case_id, valid_policy_set)
            return BenchmarkPrediction(
                case_id=case.case_id,
                strategy=strategy,
                parsed=parsed,
                raw_response=raw_text,
                latency_ms=round(latency_ms, 2),
                model_name=self.name,
                error="Mock malformed output"
            )
        else:
            verdict = "ALLOW"
            applicable = []
            violated = []
            policy_checks = [
                {"policy_id": pid, "status": "not_applicable", "evidence_quotes": []}
                for pid in policy_ids
            ]
            evidence = []
            rationale = f"Mock mode: {self.mode}"

        output_dict = {
            "case_id": case.case_id,
            "verdict": verdict,
            "applicable_policy_ids": applicable,
            "violated_policy_ids": violated,
            "policy_checks": policy_checks,
            "evidence": evidence,
            "rationale": rationale
        }
        raw_text = json.dumps(output_dict, indent=2)
        latency_ms = (time.time() - start_time) * 1000.0

        parsed = parse_model_output(raw_text, case.case_id, valid_policy_set)

        return BenchmarkPrediction(
            case_id=case.case_id,
            strategy=strategy,
            parsed=parsed,
            raw_response=raw_text,
            latency_ms=round(latency_ms, 2),
            input_tokens=1500,
            output_tokens=600,
            retries=0,
            model_name=self.name
        )

    def unload(self) -> None:
        pass
