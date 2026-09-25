from typing import Optional, Dict, Any
from .base import GuardModel, GuardPrediction
from ..data.loader import TestCase

class WebGuardAdapter(GuardModel):
    """Adapter for WebGuard consequence-based action safety classifier."""

    def __init__(self, name: str = "webguard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.checkpoint = self.config.get("checkpoint")
        self.pipeline = None

    def load(self) -> None:
        if not self.config.get("enabled", False) or not self.checkpoint:
            raise NotImplementedError(
                f"WebGuard adapter is not enabled or missing checkpoint in config/models.yaml.\n"
                f"TODO: Configure verified Hugging Face checkpoint (e.g., 'OSU-NLP/WebGuard-7B') and enable model."
            )
        try:
            import torch
            from transformers import pipeline
            self.pipeline = pipeline("text-classification", model=self.checkpoint, device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            raise RuntimeError(f"Failed to load WebGuard model '{self.checkpoint}': {e}")

    def predict(self, case: TestCase) -> GuardPrediction:
        if self.pipeline is None:
            raise RuntimeError("WebGuard model is not loaded. Call load() first.")
        # TODO: Implement WebGuard prompt formatting and token classification logic
        raise NotImplementedError("WebGuard GPU inference pipeline requires configured checkpoint.")

    def unload(self) -> None:
        self.pipeline = None
