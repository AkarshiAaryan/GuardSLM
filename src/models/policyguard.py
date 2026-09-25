import time
from typing import Optional, Dict, Any
from .base import GuardModel, GuardPrediction
from ..data.loader import TestCase
from ..prompts.renderer import render_prompt
from ..utils.logging import setup_logger

logger = setup_logger("PolicyGuardAdapter")

class PolicyGuardAdapter(GuardModel):
    """Adapter for PolicyGuard 4B policy compliance model."""

    def __init__(self, name: str = "policyguard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.checkpoint = self.config.get("checkpoint")
        self.prompt_template = self.config.get("prompt_template", "prompts/policy_guard_prompt.txt")
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if not self.config.get("enabled", False) or not self.checkpoint:
            raise NotImplementedError(
                f"PolicyGuard adapter is not enabled or missing checkpoint in config/models.yaml.\n"
                f"To run in Colab: set enabled: true and checkpoint: 'PolicyGuard-4B' (or custom checkpoint)."
            )

        logger.info(f"Loading PolicyGuard checkpoint: {self.checkpoint}...")
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed loading PolicyGuard model '{self.checkpoint}': {e}")
            raise RuntimeError(f"Model load error: {e}")

    def predict(self, case: TestCase) -> GuardPrediction:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")

        start_time = time.time()
        prompt_text = render_prompt(self.prompt_template, case)

        import torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128, temperature=0.0, do_sample=False)

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        latency_ms = (time.time() - start_time) * 1000.0

        verdict, parse_status = self.parse_verdict(raw_text)

        return GuardPrediction(
            normalized_verdict=verdict,
            raw_response=raw_text,
            latency_ms=round(latency_ms, 2),
            parse_status=parse_status,
            reason=raw_text.strip()
        )

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info(f"Unloaded model '{self.name}'.")
