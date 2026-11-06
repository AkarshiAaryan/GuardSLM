"""GuardSLM: Minimal Benchmarking Pipeline for Autonomous Web Agent Authorization Guardrails."""

__version__ = "1.0.0"

from .data import TestCase, load_ground_truth, load_test_cases
from .evaluation import evaluate_predictions, generate_markdown_report
from .models import BaseModel, DynaGuardModel, MockModel, PolicyGuardModel, get_model
from .parser import parse_response
from .prompts import build_dynaguard_prompt, build_policyguard_prompt, build_prompt
from .runner import run_experiment

__all__ = [
    "TestCase",
    "load_test_cases",
    "load_ground_truth",
    "build_prompt",
    "build_policyguard_prompt",
    "build_dynaguard_prompt",
    "BaseModel",
    "PolicyGuardModel",
    "DynaGuardModel",
    "MockModel",
    "get_model",
    "parse_response",
    "evaluate_predictions",
    "generate_markdown_report",
    "run_experiment",
]
