from typing import List, Dict, Any
from ..data.loader import TestCase
from ..models.base import GuardModel, GuardPrediction
from .result_schema import CaseResult
from ..utils.logging import setup_logger

logger = setup_logger("EvaluationRunner")

def run_evaluation_for_model(model: GuardModel, cases: List[TestCase]) -> List[CaseResult]:
    """Runs evaluation for a single model on a list of TestCases."""
    logger.info(f"Starting evaluation for model '{model.name}' on {len(cases)} cases...")
    results: List[CaseResult] = []

    try:
        model.load()
    except Exception as e:
        logger.error(f"Failed to load model '{model.name}': {e}")
        raise e

    try:
        for idx, case in enumerate(cases):
            try:
                pred: GuardPrediction = model.predict(case)
                res = CaseResult.create(
                    case_id=case.case_id,
                    pair_id=case.pair_id,
                    template_id=case.template_id,
                    model_name=model.name,
                    gold_verdict=case.gold.verdict,
                    predicted_verdict=pred.normalized_verdict,
                    raw_response=pred.raw_response,
                    latency_ms=pred.latency_ms,
                    parse_status=pred.parse_status,
                    error=pred.error,
                    template_name=case.template_name
                )
            except Exception as e:
                logger.error(f"Error predicting case {case.case_id} with model '{model.name}': {e}")
                res = CaseResult.create(
                    case_id=case.case_id,
                    pair_id=case.pair_id,
                    template_id=case.template_id,
                    model_name=model.name,
                    gold_verdict=case.gold.verdict,
                    predicted_verdict="UNKNOWN",
                    raw_response="",
                    latency_ms=0.0,
                    parse_status="failed",
                    error=str(e),
                    template_name=case.template_name
                )
            results.append(res)
    finally:
        model.unload()
        logger.info(f"Unloaded model '{model.name}'. Completed {len(results)} predictions.")

    return results
