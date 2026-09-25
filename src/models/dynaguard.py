from typing import Optional, Dict, Any
from .base import GuardModel, GuardPrediction
from ..data.loader import TestCase

class DynaGuardAdapter(GuardModel):
    """Adapter for DynaGuard policy-following guard model."""

    def __init__(self, name: str = "dynaguard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.checkpoint = self.config.get("checkpoint")

    def load(self) -> None:
        if not self.config.get("enabled", False) or not self.checkpoint:
            raise NotImplementedError(
                f"DynaGuard adapter is not enabled or missing checkpoint in config/models.yaml.\n"
                f"TODO: Configure verified Hugging Face checkpoint (e.g., 'DynaGuard-8B') and enable model."
            )

    def predict(self, case: TestCase) -> GuardPrediction:
        raise NotImplementedError("DynaGuard GPU inference pipeline requires configured checkpoint.")

    def unload(self) -> None:
        pass
