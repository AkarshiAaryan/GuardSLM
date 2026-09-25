from typing import Optional, Dict, Any
from .base import GuardModel, GuardPrediction
from ..data.loader import TestCase

class LlamaGuardAdapter(GuardModel):
    """Adapter for Llama Guard safety model."""

    def __init__(self, name: str = "llama_guard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.checkpoint = self.config.get("checkpoint")

    def load(self) -> None:
        if not self.config.get("enabled", False) or not self.checkpoint:
            raise NotImplementedError(
                f"Llama Guard adapter is not enabled or missing checkpoint in config/models.yaml.\n"
                f"TODO: Configure verified Hugging Face checkpoint (e.g., 'meta-llama/Llama-Guard-3-8B') and enable model."
            )

    def predict(self, case: TestCase) -> GuardPrediction:
        raise NotImplementedError("Llama Guard GPU inference pipeline requires configured checkpoint.")

    def unload(self) -> None:
        pass
