import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from ..data.loader import TestCase

@dataclass
class GuardPrediction:
    normalized_verdict: str  # ALLOW, BLOCK, ASK, UNKNOWN
    raw_response: str
    latency_ms: float
    parse_status: str  # success, failed
    reason: str = ""
    error: Optional[str] = None

class GuardModel(ABC):
    name: str

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def load(self) -> None:
        """Loads model weights or initializes API client."""
        pass


    @abstractmethod
    def predict(self, case: TestCase) -> GuardPrediction:
        """Runs safety guard evaluation on a TestCase object."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Frees model resources / memory."""
        pass

    def parse_verdict(self, raw_text: str) -> tuple[str, str]:
        """Utility method to parse ALLOW, BLOCK, ASK from raw model outputs.

        Returns: (normalized_verdict, parse_status)
        """
        text_upper = raw_text.strip().upper()
        if "VERDICT:" in text_upper:
            parts = text_upper.split("VERDICT:")
            verdict_part = parts[1].strip().split()[0]
            if verdict_part in ["ALLOW", "BLOCK", "ASK"]:
                return verdict_part, "success"

        for word in ["ALLOW", "BLOCK", "ASK"]:
            if word in text_upper:
                return word, "success"

        return "UNKNOWN", "failed"
