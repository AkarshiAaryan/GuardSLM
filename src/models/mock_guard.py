import time
from typing import Optional, Dict, Any
from .base import GuardModel, GuardPrediction
from ..data.loader import TestCase

class MockGuard(GuardModel):
    """Lightweight CPU mock guard model for pipeline testing and local execution without GPU."""

    def __init__(self, name: str = "mock_guard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.mode = self.config.get("mode", "perfect")

    def load(self) -> None:
        """Mock load step."""
        pass

    def predict(self, case: TestCase) -> GuardPrediction:
        start_time = time.time()

        if self.mode == "perfect":
            verdict = case.gold.verdict
            raw_resp = f"VERDICT: {verdict}\nExplanation: Mock perfect prediction for case {case.case_id}."
        elif self.mode == "context_blind":
            # Always predicts ALLOW regardless of context
            verdict = "ALLOW"
            raw_resp = f"VERDICT: ALLOW\nExplanation: Context blind prediction."
        elif self.mode == "always_allow":
            verdict = "ALLOW"
            raw_resp = "VERDICT: ALLOW"
        elif self.mode == "always_block":
            verdict = "BLOCK"
            raw_resp = "VERDICT: BLOCK"
        else:
            verdict = case.gold.verdict
            raw_resp = f"VERDICT: {verdict}"

        latency_ms = (time.time() - start_time) * 1000.0

        return GuardPrediction(
            normalized_verdict=verdict,
            raw_response=raw_resp,
            latency_ms=round(latency_ms, 2),
            parse_status="success",
            reason=f"Mock verdict mode: {self.mode}",
            error=None
        )

    def unload(self) -> None:
        """Mock unload step."""
        pass
