"""Model abstraction, HuggingFace adapters, and mock testing models for GuardSLM."""

import gc
import re
import time
from typing import Any, Dict, Optional, Tuple


class BaseModel:
    """Base interface for all GuardSLM inference models."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    def load(self) -> None:
        """Loads weights and prepares model for inference."""
        pass

    def predict(self, prompt: str) -> Tuple[str, float, Optional[str]]:
        """Executes inference on the input prompt.

        Returns:
            Tuple of (raw_response_text, latency_ms, error_message_or_None)
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Releases GPU/CPU memory and clears cached state."""
        pass


class PolicyGuardModel(BaseModel):
    """Adapter for Rakancorle1/PolicyGuard-4B.

    Paper specification:
    Policy-trajectory verification classifier producing native labels:
    - 'violation' -> BLOCK
    - 'no_violation' -> ALLOW
    """

    def __init__(self, name: str = "policyguard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.checkpoint = self.config.get("checkpoint", "Rakancorle1/PolicyGuard-4B")
        self.load_in_4bit = bool(self.config.get("load_in_4bit", False))
        self.hf_token = self.config.get("hf_token", None)
        self.max_new_tokens = int(self.config.get("max_new_tokens", 4))
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            token=self.hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {"trust_remote_code": True, "token": self.hf_token}
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

        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, **kwargs)

    def predict(self, prompt: str) -> Tuple[str, float, Optional[str]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Model '{self.name}' is not loaded. Call load() first.")

        start_time = time.perf_counter()
        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            raw_response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return raw_response, round(latency_ms, 2), None
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return "", round(latency_ms, 2), str(e)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()


class DynaGuardModel(BaseModel):
    """Adapter for tomg-group-umd/DynaGuard-1.7B.

    Paper specification:
    Policy + Dialogue compliance classifier producing native labels:
    - 'PASS' -> ALLOW
    - 'FAIL' -> BLOCK
    """

    def __init__(self, name: str = "dynaguard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.checkpoint = self.config.get("checkpoint", "tomg-group-umd/DynaGuard-1.7B")
        self.load_in_4bit = bool(self.config.get("load_in_4bit", False))
        self.hf_token = self.config.get("hf_token", None)
        self.max_new_tokens = int(self.config.get("max_new_tokens", 16))
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            token=self.hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {"trust_remote_code": True, "token": self.hf_token}
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

        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, **kwargs)

    def predict(self, prompt: str) -> Tuple[str, float, Optional[str]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Model '{self.name}' is not loaded. Call load() first.")

        start_time = time.perf_counter()
        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            raw_response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return raw_response, round(latency_ms, 2), None
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return "", round(latency_ms, 2), str(e)

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()


class MockModel(BaseModel):
    """Deterministic CPU mock model for fast, reproducible testing and CI pipelines."""

    def __init__(self, name: str = "mock_guard", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.mode = self.config.get("mode", "perfect")
        self.target_format = self.config.get("target_format", "standard")  # standard | policyguard | dynaguard
        self.oracle_golds = self.config.get("oracle_golds", {})
        self.simulated_latency_ms = float(self.config.get("simulated_latency_ms", 5.0))
        self.is_loaded = False

    def load(self) -> None:
        self.is_loaded = True

    def predict(self, prompt: str) -> Tuple[str, float, Optional[str]]:
        if not self.is_loaded:
            raise RuntimeError(f"Mock model '{self.name}' is not loaded. Call load() first.")

        start_time = time.perf_counter()
        time.sleep(self.simulated_latency_ms / 1000.0)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if self.mode == "error":
            return "", round(latency_ms, 2), "Simulated mock inference failure"

        if self.mode == "malformed":
            return "MALFORMED_OUTPUT_UNKNOWN_DECISION_TEXT", round(latency_ms, 2), None

        # Determine decision (ALLOW vs BLOCK)
        decision = "ALLOW"
        if self.mode == "always_block":
            decision = "BLOCK"
        elif self.mode == "always_allow":
            decision = "ALLOW"
        elif self.mode in ("perfect", "oracle"):
            # Extract case_id from prompt if present (e.g., 'TC01')
            m = re.search(r'\b(TC\d+)\b', prompt)
            case_id = m.group(1) if m else ""
            if case_id and case_id in self.oracle_golds:
                decision = self.oracle_golds[case_id]
            else:
                # Fallback heuristic based on content
                decision = "BLOCK" if any(w in prompt.lower() for w in ["violation", "unauthorized", "fail", "block"]) else "ALLOW"
        else:
            decision = "ALLOW"

        # Format output according to target format
        fmt = self.target_format.lower()
        if fmt == "policyguard":
            raw_text = "violation" if decision == "BLOCK" else "no_violation"
        elif fmt == "dynaguard":
            raw_text = f"<answer>\n{ 'FAIL' if decision == 'BLOCK' else 'PASS' }\n</answer>"
        else:
            raw_text = decision

        return raw_text, round(latency_ms, 2), None

    def unload(self) -> None:
        self.is_loaded = False


def get_model(model_name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
    """Factory to instantiate model adapter by name."""
    m_lower = model_name.lower()
    if "policyguard" in m_lower:
        return PolicyGuardModel(name=model_name, config=config)
    elif "dynaguard" in m_lower:
        return DynaGuardModel(name=model_name, config=config)
    elif "mock" in m_lower:
        return MockModel(name=model_name, config=config)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Supported models: 'policyguard', 'dynaguard', 'mock_guard'.")
