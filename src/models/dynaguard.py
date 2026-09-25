import time
from typing import Optional, Dict, Any
from .base import GuardModel, GuardPrediction
from ..data.loader import TestCase
from ..prompts.renderer import render_prompt
from ..utils.logging import setup_logger

logger = setup_logger("DynaGuardAdapter")

class DynaGuardAdapter(GuardModel):
    """Adapter for DynaGuard open-source policy-following guard models."""

    def __init__(self, name: str = "dynaguard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.checkpoint = self.config.get("checkpoint", "DynaGuard/DynaGuard-8B")
        self.prompt_template = self.config.get("prompt_template", "prompts/base_guard_prompt.txt")
        self.load_in_4bit = self.config.get("load_in_4bit", True)
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if not self.config.get("enabled", False) or not self.checkpoint:
            raise NotImplementedError(
                f"DynaGuard adapter is disabled or missing checkpoint in config/models.yaml.\n"
                f"Set enabled: true and checkpoint: 'DynaGuard/DynaGuard-8B' (or custom checkpoint)."
            )

        logger.info(f"Loading DynaGuard checkpoint: {self.checkpoint}...")
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
            
            kwargs = {"trust_remote_code": True}
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"
                if self.load_in_4bit:
                    try:
                        from transformers import BitsAndBytesConfig
                        kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16
                        )
                    except Exception as q_err:
                        logger.warning(f"BitsAndBytes 4-bit quantization unavailable, falling back to float16: {q_err}")
                        kwargs["torch_dtype"] = torch.float16
                else:
                    kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, **kwargs)
            logger.info(f"Successfully loaded {self.checkpoint}!")
        except Exception as e:
            logger.error(f"Failed loading DynaGuard model '{self.checkpoint}': {e}")
            raise RuntimeError(f"Model load error: {e}")

    def predict(self, case: TestCase) -> GuardPrediction:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")

        start_time = time.time()
        prompt_text = render_prompt(self.prompt_template, case)

        import torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )


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
