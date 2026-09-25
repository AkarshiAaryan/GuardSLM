from typing import List, Dict, Any
from .result_schema import CaseResult

def calculate_overall_metrics(results: List[CaseResult]) -> Dict[str, Any]:
    """Calculates overall accuracy, per-template accuracy, and latency metrics."""
    if not results:
        return {"total_cases": 0, "accuracy": 0.0}

    total_cases = len(results)
    correct_cases = sum(1 for r in results if r.correct)
    accuracy = round(correct_cases / total_cases, 4)

    # Per-template accuracy
    template_groups: Dict[str, List[CaseResult]] = {}
    for r in results:
        template_groups.setdefault(r.template_id, []).append(r)

    per_template_accuracy: Dict[str, Dict[str, Any]] = {}
    for tid, group in sorted(template_groups.items()):
        t_total = len(group)
        t_correct = sum(1 for r in group if r.correct)
        per_template_accuracy[tid] = {
            "template_name": group[0].template_name,
            "total": t_total,
            "correct": t_correct,
            "accuracy": round(t_correct / t_total, 4)
        }

    avg_latency_ms = round(sum(r.latency_ms for r in results) / total_cases, 2)

    return {
        "model": results[0].model if results else "unknown",
        "total_cases": total_cases,
        "correct_cases": correct_cases,
        "overall_accuracy": accuracy,
        "average_latency_ms": avg_latency_ms,
        "per_template_accuracy": per_template_accuracy
    }
