"""Abstract base class and prediction container for GuardSLM benchmark models."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from ..data.schema import ModelInputCase
from ..parser.output_parser import ParsedBenchmarkOutput


@dataclass
class BenchmarkPrediction:
    case_id: str
    strategy: str  # text_only or text_plus_screenshot
    parsed: ParsedBenchmarkOutput
    raw_response: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    retries: int = 0
    error: Optional[str] = None
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "strategy": self.strategy,
            "model_name": self.model_name,
            "parsed": self.parsed.to_dict(),
            "raw_response": self.raw_response,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "error": self.error
        }


class BenchmarkModel(ABC):
    """Abstract model interface for benchmark evaluation."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def load(self) -> None:
        """Loads weights, initializes pipeline, or prepares API client."""
        pass

    @abstractmethod
    def predict(
        self,
        case: ModelInputCase,
        strategy: str = "text_only",
        screenshot_path: Optional[str] = None
    ) -> BenchmarkPrediction:
        """Evaluates a single case under either Strategy A or Strategy B."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Releases memory, closes client, or frees accelerator resources."""
        pass
