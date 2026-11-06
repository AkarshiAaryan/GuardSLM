"""Unit tests for specialized safety guards (PolicyGuard, DynaGuard, WebGuard) on GuardSLM v3."""

import pytest
from src.benchmark.data.loader import BenchmarkLoader
from src.benchmark.models.policyguard_benchmark_guard import PolicyGuardBenchmarkGuard
from src.benchmark.models.dynaguard_benchmark_guard import DynaGuardBenchmarkGuard
from src.benchmark.models.webguard_benchmark_guard import WebGuardBenchmarkGuard
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

    # Check native PolicyGuard plain-text format with clearly distinguished sections
    assert prompt.startswith("Policy:\n")
    assert "User Request:\n" in prompt
    assert "User Role:\n" in prompt
    assert "Current Website State:\n" in prompt
    assert "Trajectory History:\n" in prompt
    assert "Proposed Action (to evaluate):\n" in prompt
    assert "Output violation or no_violation." in prompt
    assert "Step 5: unknown" not in prompt
    assert "Step 1: Authenticate as" not in prompt

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

    lbl, v, s = guard.parse_native_output("FAIL")
    assert lbl == "FAIL" and v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("<answer> FAIL")
    assert lbl == "FAIL" and v == "BLOCK" and s == "success"

    # Reject ALLOW, BLOCK, ASK and explanations as native labels
    assert guard.parse_native_output("ALLOW")[2] == "failed"
    assert guard.parse_native_output("BLOCK")[2] == "failed"
    assert guard.parse_native_output("ASK")[2] == "failed"
    assert guard.parse_native_output("This is a PASS.")[2] == "failed"
    assert guard.parse_native_output("unknown nonsense")[2] == "failed"


def test_webguard_adapter_formatting_and_parsing(sample_case):
    guard = WebGuardBenchmarkGuard(name="webguard", config={"enabled": True, "augmented": True})
    prompt = guard.format_prompt(sample_case)

    # Check key components of WebGuard a11y tree risk format
    assert "Accessibility Tree of the webpage" in prompt
    assert "The accessibility tree of the web page is" in prompt
    assert "The proposed action is" in prompt
    assert '"risky_level"' in prompt

    # Check risk classification: SAFE, LOW, HIGH are parsed, but verdict is UNKNOWN (not equated to authorization)
    lbl, v, s = guard.parse_native_output('{"risky_level": "SAFE"}')
    assert lbl == "SAFE" and v == "UNKNOWN" and s == "success"

    lbl, v, s = guard.parse_native_output('{"risky_level": "LOW"}')
    assert lbl == "LOW" and v == "UNKNOWN" and s == "success"

    lbl, v, s = guard.parse_native_output('{"risky_level": "HIGH"}')
    assert lbl == "HIGH" and v == "UNKNOWN" and s == "success"

    # Clean token checks
    lbl, v, s = guard.parse_native_output("HIGH")
    assert lbl == "HIGH" and v == "UNKNOWN" and s == "success"

    # Invalid outputs produce parsing failures
    lbl, v, s = guard.parse_native_output("unknown text")
    assert v == "UNKNOWN" and s == "failed"

    lbl, v, s = guard.parse_native_output("ALLOW")
    assert v == "UNKNOWN" and s == "failed"

    lbl, v, s = guard.parse_native_output("BLOCK")
    assert v == "UNKNOWN" and s == "failed"


def test_webguard_ablation_native_vs_augmented(sample_case):
    """Verifies that WebGuard native and augmented produce genuinely distinct prompts."""
    native_guard = WebGuardBenchmarkGuard(name="webguard_native", config={"enabled": True, "augmented": False})
    augmented_guard = WebGuardBenchmarkGuard(name="webguard", config={"enabled": True, "augmented": True})

    native_prompt = native_guard.format_prompt(sample_case)
    augmented_prompt = augmented_guard.format_prompt(sample_case)

    # Native prompt must NOT contain institutional policy or platform state
    assert "The institutional policy governing this action is:" not in native_prompt
    assert "The institutional platform state is:" not in native_prompt

    # Augmented prompt MUST contain institutional policy
    assert "The institutional policy governing this action is:" in augmented_prompt


def test_webguard_fail_loud_on_missing_adapter():
    """Verifies that WebGuard raises FileNotFoundError if configured adapter does not exist."""
    guard = WebGuardBenchmarkGuard(
        name="webguard",
        config={"enabled": True, "adapter_path": "non_existent_lora_directory_12345"}
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        guard.load()
    assert "does not exist" in str(excinfo.value)


def test_runner_model_factory_routing():
    pg = build_model_instance("policyguard")
    assert isinstance(pg, PolicyGuardBenchmarkGuard)

    dg = build_model_instance("dynaguard")
    assert isinstance(dg, DynaGuardBenchmarkGuard)

    wg = build_model_instance("webguard")
    assert isinstance(wg, WebGuardBenchmarkGuard)


def test_text_only_guards_reject_non_text_strategy(sample_case):
    for guard_cls, name in [
        (PolicyGuardBenchmarkGuard, "policyguard"),
        (DynaGuardBenchmarkGuard, "dynaguard"),
        (WebGuardBenchmarkGuard, "webguard")
    ]:
        guard = guard_cls(name=name, config={"enabled": True})
        guard.model = object()  # dummy loaded marker
        guard.tokenizer = object()
        with pytest.raises(ValueError) as excinfo:
            guard.predict(sample_case, strategy="text_plus_screenshot", screenshot_path="/fake/path.png")
        assert "not supported" in str(excinfo.value)


def test_webguard_element_extraction_and_missing_handling():
    from src.benchmark.models.webguard_benchmark_guard import derive_element_metadata
    from src.benchmark.data.schema import SnapshotInputCase, ProposedAction

    # 1. Snapshot case with matching button element
    case_btn = SnapshotInputCase(
        case_id="TC_TEST_1",
        title="Test",
        account="user",
        global_role="student",
        effective_role="Student",
        membership="member",
        timestamp="2026-11-06T10:00:00",
        timezone="UTC",
        deadline_ts="2026-11-06T12:00:00",
        course_or_scope="CSE301",
        target_resource="CSE301 / Grades",
        requested_operation="Inspect breakdown",
        target_owner="Student User",
        policy_context="Policy text",
        target_a11y_tree='- button "Grades" [ref=1]:\n  - text: Grades'
    )
    tag, desc = derive_element_metadata(case_btn)
    assert tag == "button"
    assert desc == "Grades"

    # 2. Snapshot case with link element (derived tag 'a')
    case_link = SnapshotInputCase(
        case_id="TC_TEST_2",
        title="Test",
        account="user",
        global_role="student",
        effective_role="Student",
        membership="member",
        timestamp="2026-11-06T10:00:00",
        timezone="UTC",
        deadline_ts="2026-11-06T12:00:00",
        course_or_scope="CSE301",
        target_resource="CSE301 / Gradebook",
        requested_operation="Open gradebook",
        target_owner="Student User",
        policy_context="Policy text",
        target_a11y_tree='- link "Gradebook" [ref=2]:\n  - text: Gradebook'
    )
    tag, desc = derive_element_metadata(case_link)
    assert tag == "a"
    assert desc == "Gradebook"

    # 3. Missing/unidentifiable target handled without fabrication
    case_missing = SnapshotInputCase(
        case_id="TC_TEST_3",
        title="Test",
        account="user",
        global_role="student",
        effective_role="Student",
        membership="member",
        timestamp="2026-11-06T10:00:00",
        timezone="UTC",
        deadline_ts="2026-11-06T12:00:00",
        course_or_scope="CSE301",
        target_resource="NonExistentResource / NothingHere",
        requested_operation="Click something",
        target_owner="",
        policy_context="Policy text",
        target_a11y_tree='- generic:\n  - text: Welcome to Dashboard'
    )
    tag, desc = derive_element_metadata(case_missing)
    assert tag == "unavailable"
    assert desc == "unavailable"


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
