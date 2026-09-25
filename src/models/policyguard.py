from typing import Optional, Dict, Any
from .base import GuardModel, GuardPrediction
from ..data.loader import TestCase

class PolicyGuardAdapter(GuardModel):
    """Adapter for PolicyGuard 4B model."""

    def __init__(self, name: str = "policyguard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.checkpoint = self.config.get("checkpoint")

    def load(self) -> None:
        if not self.config.get("enabled", False) or not self.checkpoint:
            raise NotImplementedError(
                f"PolicyGuard adapter is not enabled or missing checkpoint in config/models.yaml.\n"
                f"TODO: Configure verified Hugging Face checkpoint (e.g., 'PolicyGuard-4B') and enable model."
            )

    def predict(self, case: TestCase) -> GuardPrediction:
        raise NotImplementedError("PolicyGuard GPU inference pipeline requires configured checkpoint.")

    def unload(self) -> None:
        pass
