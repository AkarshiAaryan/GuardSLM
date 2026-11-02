"""WebGuard Benchmark Model Adapter for GuardSLM Benchmark (Text-Only Accessibility Action Guard)."""

import json
import os
import re
import time
from typing import Dict, Any, Optional, Tuple
from .base import BenchmarkModel, BenchmarkPrediction
from ..data.schema import ModelInputCase
from ..parser.output_parser import ParsedBenchmarkOutput
from ...training.webguard_prompt import (
    WEBGARD_SYSTEM_PROMPT,
    format_webguard_user_content,
    format_webguard_messages,
    parse_webguard_output
)


class WebGuardBenchmarkGuard(BenchmarkModel):
    """Adapter for WebGuard-3B (Qwen2.5-3B + WebGuard a11y LoRA) using native accessibility tree format."""

    def __init__(self, name: str = "webguard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.checkpoint = self.config.get("checkpoint", "Qwen/Qwen2.5-3B-Instruct")
        self.adapter_path = self.config.get("adapter_path", "models/webguard_qwen_lora")
        self.load_in_4bit = self.config.get("load_in_4bit", False)
        self.augmented = self.config.get("augmented", True)
        self.hf_token = self.config.get("hf_token", None)
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if not self.config.get("enabled", True):
            raise NotImplementedError("WebGuard adapter is disabled in config/models.yaml.")

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tok_path = (
            self.adapter_path
            if os.path.exists(os.path.join(self.adapter_path, "tokenizer_config.json"))
            else self.checkpoint
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path,
            trust_remote_code=True,
            token=self.hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"trust_remote_code": True, "token": self.hf_token}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            if self.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                except Exception:
                    kwargs["torch_dtype"] = torch.float16
            else:
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(self.checkpoint, **kwargs)

        # Check for LoRA adapter
        if os.path.exists(self.adapter_path) and os.path.exists(os.path.join(self.adapter_path, "adapter_config.json")):
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            except Exception:
                self.model = base_model
        else:
            self.model = base_model

    def format_prompt(self, case: ModelInputCase) -> str:
        """Formats case into WebGuard's native accessibility tree prompt."""
        page_dict = case.page_context.get("page", {}) if isinstance(case.page_context, dict) else {}
        page_type = page_dict.get("page_type") or "page"
        course_ctx = page_dict.get("course_context") or ""
        controls = page_dict.get("visible_controls") or "none"
        a11y_tree = (
            f"RootWebArea '{page_type}'\n"
            f"  heading 'Course Context: {course_ctx}'\n"
            f"  main\n"
            f"    controls: {controls}"
        )

        operation = case.proposed_action.operation
        target_dict = case.proposed_action.target if isinstance(case.proposed_action.target, dict) else {}
        target_str = json.dumps(target_dict)
        proposed_action_str = f"{operation} {target_str}".strip()

        tag_head = target_dict.get("element_type", "button") if isinstance(target_dict, dict) else "button"
        desc = case.user_request or "nan"
        url = f"https://moodle.university.edu/{case.resource_context.get('course_id', 'course')}/{case.resource_context.get('resource_type', 'view')}"

        policy_str = None
        moodle_state = None
        if self.augmented:
            policy_lines = case.policy_info[:3] if case.policy_info else []
            policy_str = " | ".join(policy_lines) if policy_lines else None
            moodle_state = case.state_context

        content = format_webguard_user_content(
            accessibility_tree=a11y_tree,
            proposed_action=proposed_action_str,
            description=desc,
            tag_head=tag_head,
            url=url,
            policy=policy_str,
            moodle_state=moodle_state
        )
        return content

    def parse_native_output(self, raw_text: str) -> Tuple[str, str, str]:
        """Maps WebGuard output (SAFE / LOW -> ALLOW, HIGH -> BLOCK)."""
        return parse_webguard_output(raw_text)

    def predict(
        self,
        case: ModelInputCase,
        strategy: str = "text_only",
        screenshot_path: Optional[str] = None
    ) -> BenchmarkPrediction:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Model '{self.name}' not loaded. Call load() before predict().")

        if strategy != "text_only":
            raise ValueError(
                f"Model '{self.name}' is a text-only accessibility action guard. Strategy '{strategy}' is not supported."
            )

        start_time = time.time()
        user_content = self.format_prompt(case)
        messages = [{"role": "user", "content": user_content}]

        import torch
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None) is not None:
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted_prompt = user_content
        else:
            formatted_prompt = user_content

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
        raw_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        latency_ms = (time.time() - start_time) * 1000.0

        risky_level, mapped_verdict, parse_status = self.parse_native_output(raw_text)

        policy_checks = [
            {"policy_id": f"POL-{i+1:02d}", "status": "not_applicable", "evidence_quotes": []}
            for i in range(len(case.policy_info))
        ]

        parsed = ParsedBenchmarkOutput(
            case_id=case.case_id,
            verdict=mapped_verdict,
            applicable_policy_ids=[],
            violated_policy_ids=[],
            policy_checks=policy_checks,
            evidence=[],
            rationale=f"WebGuard native risk: {risky_level} -> Mapped verdict: {mapped_verdict}",
            parse_status=parse_status,
            validation_errors=[] if parse_status != "failed" else [f"Could not parse WebGuard risk level from '{raw_text}'"],
            raw_response=raw_text
        )

        return BenchmarkPrediction(
            case_id=case.case_id,
            strategy=strategy,
            parsed=parsed,
            raw_response=raw_text,
            latency_ms=round(latency_ms, 2),
            retries=0,
            model_name=self.name,
            error=None if parse_status != "failed" else f"Unrecognized WebGuard risk level in '{raw_text}'"
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
