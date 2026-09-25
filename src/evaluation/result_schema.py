from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class CaseResult:
    case_id: str
    pair_id: str
    template_id: str
    model: str
    gold_verdict: str
    predicted_verdict: str
    correct: bool
    raw_response: str
    latency_ms: float
    parse_status: str
    timestamp: str
    error: Optional[str] = None
    template_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        case_id: str,
        pair_id: str,
        template_id: str,
        model_name: str,
        gold_verdict: str,
        predicted_verdict: str,
        raw_response: str,
        latency_ms: float,
        parse_status: str,
        error: Optional[str] = None,
        template_name: str = ""
    ) -> "CaseResult":
        correct = (gold_verdict.upper() == predicted_verdict.upper()) and (parse_status == "success")
        return cls(
            case_id=case_id,
            pair_id=pair_id,
            template_id=template_id,
            model=model_name,
            gold_verdict=gold_verdict.upper(),
            predicted_verdict=predicted_verdict.upper(),
            correct=correct,
            raw_response=raw_response,
            latency_ms=latency_ms,
            parse_status=parse_status,
            timestamp=datetime.now().isoformat(),
            error=error,
            template_name=template_name
        )
