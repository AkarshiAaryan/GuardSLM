"""Hugging Face Model Adapter for GuardSLM 100-Pair Benchmark (Text & Vision-Language)."""

import os
import time
from typing import Dict, Any, Optional, List, Set
from .base import BenchmarkModel, BenchmarkPrediction
from ..data.schema import ModelInputCase
from ..prompts.renderer import render_benchmark_prompt
from ..parser.output_parser import parse_model_output, build_repair_prompt


class HFBenchmarkGuard(BenchmarkModel):
    """Adapter for running open-source models via Hugging Face Transformers under Strategy A and B."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.checkpoint = self.config.get("checkpoint", "Qwen/Qwen2.5-1.5B-Instruct")
        self.is_vlm = self.config.get("is_vlm", False)
        self.load_in_4bit = self.config.get("load_in_4bit", False)
        self.max_new_tokens = self.config.get("max_new_tokens", 1024)
        self.temperature = self.config.get("temperature", 0.0)
        self.max_retries = self.config.get("max_retries", 1)

        self.tokenizer = None
        self.processor = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            else:
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32

        if self.is_vlm:
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
                if hasattr(self.processor, "tokenizer"):
                    self.tokenizer = self.processor.tokenizer
                self.model = AutoModelForVision2Seq.from_pretrained(self.checkpoint, **kwargs)
                return
            except Exception:
                # Fallback to causal LM if AutoModelForVision2Seq does not recognize class
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.checkpoint, **kwargs)
                    return
                except Exception:
                    pass

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, **kwargs)

    def _generate_raw_text(self, prompt_text: str, image_path: Optional[str] = None) -> str:
        import torch

        # Multimodal Vision-Language Model execution (supports both Strategy A and B)
        if self.is_vlm and self.processor is not None:
            if image_path:
                from PIL import Image
                image = Image.open(image_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(self.model.device)
            else:
                # Text-only Strategy A execution with VLM
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text_prompt], padding=True, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=(self.temperature > 0.0)
                )
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            return self.processor.decode(generated_ids, skip_special_tokens=True)

        # Pure text-only SLM generation (Strategy A only)
        if image_path:
            raise ValueError(
                f"Model '{self.name}' is a text-only SLM and does not support Strategy B (image input). "
                f"Use a Vision-Language Model (e.g. Qwen3-VL-2B) for Strategy B."
            )

        messages = [{"role": "user", "content": prompt_text}]
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted_prompt = prompt_text
        else:
            formatted_prompt = prompt_text

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0.0),
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def predict(
        self,
        case: ModelInputCase,
        strategy: str = "text_only",
        screenshot_path: Optional[str] = None
    ) -> BenchmarkPrediction:
        if self.model is None:
            raise RuntimeError(f"Model '{self.name}' not loaded. Call load() before predict().")

        start_time = time.time()
        valid_policy_set = {p.policy_id for p in case.policy_book}
        prompt_text = render_benchmark_prompt(case)

        actual_image = screenshot_path if (strategy == "text_plus_screenshot" and screenshot_path) else None

        retries = 0
        raw_text = self._generate_raw_text(prompt_text, actual_image)
        parsed = parse_model_output(raw_text, case.case_id, valid_policy_set)

        # Controlled repair retry if JSON parse completely failed
        while parsed.parse_status == "failed" and retries < self.max_retries:
            retries += 1
            repair_prompt = build_repair_prompt(raw_text, parsed.validation_errors)
            raw_text = self._generate_raw_text(repair_prompt, actual_image)
            parsed = parse_model_output(raw_text, case.case_id, valid_policy_set)
            if parsed.parse_status != "failed":
                parsed.parse_status = "repaired"

        latency_ms = (time.time() - start_time) * 1000.0

        return BenchmarkPrediction(
            case_id=case.case_id,
            strategy=strategy,
            parsed=parsed,
            raw_response=raw_text,
            latency_ms=round(latency_ms, 2),
            retries=retries,
            model_name=self.name,
            error=None if parsed.parse_status != "failed" else "; ".join(parsed.validation_errors)
        )

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        self.processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
