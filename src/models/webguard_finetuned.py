import os
import time
from typing import Optional, Dict, Any, Tuple
from .base import GuardModel, GuardPrediction
from ..data.loader import TestCase
from ..training.webguard_prompt import format_webguard_messages, parse_webguard_output
from ..utils.logging import setup_logger

logger = setup_logger("WebGuardFineTunedAdapter")

class WebGuardFineTunedAdapter(GuardModel):
    """Adapter for Fine-Tuned WebGuard Guardrail model (Qwen + WebGuard LoRA)."""

    def __init__(self, name: str = "webguard_finetuned", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.checkpoint = self.config.get("checkpoint", "Qwen/Qwen2.5-3B-Instruct")
        self.adapter_path = self.config.get("adapter_path", "models/webguard_qwen_lora")
        self.load_in_4bit = self.config.get("load_in_4bit", False)
        # augmented: False -> Original unaugmented WebGuard input (baseline)
        # augmented: True  -> Context-augmented WebGuard input (with policy & state)
        self.augmented = self.config.get("augmented", False)
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if not self.config.get("enabled", True):
            raise NotImplementedError("WebGuard Fine-Tuned adapter is disabled.")

        logger.info(f"Loading WebGuard base model: {self.checkpoint}...")
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Load tokenizer
            tok_path = self.adapter_path if os.path.exists(os.path.join(self.adapter_path, "tokenizer_config.json")) else self.checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

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
                        kwargs["torch_dtype"] = torch.float16
                else:
                    kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float32

            base_model = AutoModelForCausalLM.from_pretrained(self.checkpoint, **kwargs)

            # Apply LoRA adapter if present
            if os.path.exists(self.adapter_path) and os.path.exists(os.path.join(self.adapter_path, "adapter_config.json")):
                try:
                    from peft import PeftModel
                    logger.info(f"Attaching WebGuard LoRA weights from {self.adapter_path}...")
                    self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
                    logger.info("Successfully loaded fine-tuned WebGuard LoRA adapter!")
                except Exception as peft_err:
                    logger.warning(f"Failed to load LoRA weights ({peft_err}); falling back to base model.")
                    self.model = base_model
            else:
                logger.info(f"No LoRA adapter found at {self.adapter_path}. Running base model with WebGuard prompt.")
                self.model = base_model

        except Exception as e:
            logger.error(f"Failed loading WebGuard model: {e}")
            raise RuntimeError(f"Model load error: {e}")

    def predict(self, case: TestCase) -> GuardPrediction:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() before predict().")

        start_time = time.time()
        
        # Extract case information into WebGuard format
        a11y_tree = getattr(case.context.page_context, "accessibility_tree", "") or "[page context]"
        proposed_action = f"{case.proposed_action.type} {case.proposed_action.element}".strip()
        page_title = getattr(case.context.page_context, "page_title", "Moodle")

        policy = case.context.policy if self.augmented else None
        moodle_state = case.context.moodle_state if self.augmented else None

        messages = format_webguard_messages(
            accessibility_tree=a11y_tree,
            proposed_action=proposed_action,
            description=case.request or "nan",
            tag_head="button",
            url=f"https://moodle.university.edu/{page_title}",
            policy=policy,
            moodle_state=moodle_state
        )

        import torch
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None) is not None:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = messages[0]["content"]

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        raw_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        latency_ms = (time.time() - start_time) * 1000.0

        risky_level, verdict, parse_status = parse_webguard_output(raw_text)

        return GuardPrediction(
            normalized_verdict=verdict,
            raw_response=raw_text,
            latency_ms=round(latency_ms, 2),
            parse_status=parse_status,
            reason=f"Stage 1 Native Risk: {risky_level} -> Stage 2 GuardSLM: {verdict}",
            native_output=risky_level
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
