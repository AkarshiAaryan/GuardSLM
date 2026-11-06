"""Unit tests for specialized safety guards (PolicyGuard, DynaGuard) on GuardSLM v3."""

import pytest
from src.benchmark.data.loader import BenchmarkLoader
from src.benchmark.models.policyguard_benchmark_guard import PolicyGuardBenchmarkGuard
from src.benchmark.models.dynaguard_benchmark_guard import DynaGuardBenchmarkGuard
from src.benchmark.runner.experiment_runner import build_model_instance


@pytest.fixture(scope="module")
def sample_case():
    loader = BenchmarkLoader(dataset_path="data/GuardSLM_v3_Single_Master_Dataset.json")
    cases = loader.get_model_inputs(limit=1)
    assert len(cases) == 1
    return cases[0]


def test_policyguard_adapter_formatting_and_parsing(sample_case):
    guard = PolicyGuardBenchmarkGuard(name="policyguard", config={"enabled": True})
    prompt = guard.format_prompt(sample_case)

    # Check official PolicyGuard plain-text format: Policy + Trajectory Actions + Output violation or no_violation.
    assert prompt.startswith("Policy: ")
    assert "Trajectory Actions:\n" in prompt
    assert "Step 1: Authenticate as " in prompt
    assert prompt.endswith("Output violation or no_violation.")

    # Check strict native label mapping
    lbl, v, s = guard.parse_native_output("violation")
    assert lbl == "violation" and v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("no_violation")
    assert lbl == "no_violation" and v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("  NO-VIOLATION.  ")
    assert lbl == "no_violation" and v == "ALLOW" and s == "success"

    # Ensure no_violation is never misparsed as violation
    lbl, v, s = guard.parse_native_output("no_violation")
    assert lbl != "violation"

    # Reject arbitrary explanatory text and non-native labels
    lbl, v, s = guard.parse_native_output("This is a violation action.")
    assert v == "UNKNOWN" and s == "failed"

    lbl, v, s = guard.parse_native_output("ALLOW")
    assert v == "UNKNOWN" and s == "failed"

    lbl, v, s = guard.parse_native_output("unintelligible response")
    assert v == "UNKNOWN" and s == "failed"


def test_dynaguard_adapter_formatting_and_parsing(sample_case):
    guard = DynaGuardBenchmarkGuard(name="dynaguard", config={"enabled": True})
    prompt = guard.format_prompt(sample_case)

    # Check native DynaGuard [INST] Policy + Dialogue [/INST] <answer> format
    assert prompt.startswith("[INST]\nEvaluate the following dialogue for compliance with the given policy.")
    assert "## Policy\n" in prompt
    assert "## Dialogue\n" in prompt
    assert "User: As " in prompt
    assert "Agent: I am about to " in prompt
    assert prompt.endswith("[/INST]\n<answer>")

    # Check strict native PASS / FAIL label mapping
    lbl, v, s = guard.parse_native_output("PASS")
    assert lbl == "PASS" and v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("<answer> PASS")
    assert lbl == "PASS" and v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("PASS</s>")
    assert lbl == "PASS" and v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("PASS<|eot_id|>")
    assert lbl == "PASS" and v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("<answer>PASS</answer></s>")
    assert lbl == "PASS" and v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("FAIL")
    assert lbl == "FAIL" and v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("<answer> FAIL")
    assert lbl == "FAIL" and v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("FAIL</s>")
    assert lbl == "FAIL" and v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("<answer>FAIL</answer>")
    assert lbl == "FAIL" and v == "BLOCK" and s == "success"

    # Reject ALLOW, BLOCK, ASK and explanations as native labels
    assert guard.parse_native_output("ALLOW")[2] == "failed"
    assert guard.parse_native_output("BLOCK")[2] == "failed"
    assert guard.parse_native_output("ASK")[2] == "failed"
    assert guard.parse_native_output("This is a PASS.")[2] == "failed"
    assert guard.parse_native_output("unknown nonsense")[2] == "failed"


def test_runner_model_factory_routing():
    pg = build_model_instance("policyguard")
    assert isinstance(pg, PolicyGuardBenchmarkGuard)

    dg = build_model_instance("dynaguard")
    assert isinstance(dg, DynaGuardBenchmarkGuard)


def test_text_only_guards_reject_non_text_strategy(sample_case):
    for guard_cls, name in [
        (PolicyGuardBenchmarkGuard, "policyguard"),
        (DynaGuardBenchmarkGuard, "dynaguard")
    ]:
        guard = guard_cls(name=name, config={"enabled": True})
        guard.model = object()  # dummy loaded marker
        guard.tokenizer = object()
        with pytest.raises(ValueError) as excinfo:
            guard.predict(sample_case, strategy="text_plus_screenshot", screenshot_path="/fake/path.png")
        assert "not supported" in str(excinfo.value)


def test_shared_pipeline_output_contract(sample_case):
    from src.benchmark.models.base import BenchmarkPrediction
    from src.benchmark.parser.output_parser import ParsedBenchmarkOutput

    # Verify ParsedBenchmarkOutput preserves raw_output, native_label, prediction, parsing_status, parsing_error
    parsed_allow = ParsedBenchmarkOutput(
        case_id=sample_case.case_id,
        verdict="ALLOW",
        applicable_policy_ids=[],
        violated_policy_ids=[],
        policy_checks=[],
        evidence=[],
        rationale="Policy passed",
        parse_status="success",
        raw_response="PASS",
        native_label="PASS",
        parsing_error=None
    )

    pred = BenchmarkPrediction(
        case_id=sample_case.case_id,
        strategy="text_only",
        parsed=parsed_allow,
        raw_response="PASS",
        latency_ms=12.5,
        model_name="dynaguard"
    )

    # Check top-level properties
    assert pred.raw_output == "PASS"
    assert pred.native_label == "PASS"
    assert pred.prediction == "ALLOW"
    assert pred.parsing_status == "success"

    # Check dictionary serialization
    d = pred.to_dict()
    assert d["raw_output"] == "PASS"
    assert d["native_label"] == "PASS"
    assert d["prediction"] == "ALLOW"
    assert d["parsing_status"] == "success"

    # Verify failed parsing is visible to evaluator
    parsed_failed = ParsedBenchmarkOutput(
        case_id=sample_case.case_id,
        verdict="UNKNOWN",
        applicable_policy_ids=[],
        violated_policy_ids=[],
        policy_checks=[],
        evidence=[],
        rationale="Parse failed",
        parse_status="failed",
        raw_response="gibberish 123",
        native_label=None,
        parsing_error="Unrecognized output"
    )
    pred_failed = BenchmarkPrediction(
        case_id=sample_case.case_id,
        strategy="text_only",
        parsed=parsed_failed,
        raw_response="gibberish 123",
        latency_ms=10.0,
        model_name="dynaguard",
        error="Unrecognized output"
    )
    assert pred_failed.parsing_status == "failed"
    assert pred_failed.prediction == "UNKNOWN"
    assert pred_failed.error == "Unrecognized output"
