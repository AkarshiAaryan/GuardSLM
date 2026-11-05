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

    # Check key components of PolicyGuard trajectory format
    assert "Trajectory Actions:" in prompt
    assert "Step 1: User Request:" in prompt
    assert "Step 2: User Role:" in prompt
    assert "Step 3: State Context:" in prompt
    assert "Step 4: Proposed Action:" in prompt
    assert "violation or no_violation" in prompt

    # Check two-stage label mapping
    lbl, v, s = guard.parse_native_output("violation")
    assert lbl == "violation" and v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("no_violation")
    assert lbl == "no_violation" and v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("This is a no violation action.")
    assert v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("unintelligible response")
    assert v == "UNKNOWN" and s == "failed"


def test_dynaguard_adapter_formatting_and_parsing(sample_case):
    guard = DynaGuardBenchmarkGuard(name="dynaguard", config={"enabled": True})
    prompt = guard.format_prompt(sample_case)

    # Check key components of DynaGuard compliance gatekeeper format
    assert "USER ROLE:" in prompt
    assert "USER REQUEST:" in prompt
    assert "CURRENT CONTEXT:" in prompt
    assert "RELEVANT POLICY:" in prompt
    assert "PROPOSED ACTION:" in prompt
    assert "ALLOW" in prompt and "BLOCK" in prompt

    # Check two-stage label mapping
    lbl, v, s = guard.parse_native_output("ALLOW: The teacher has grading privileges.")
    assert v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("PASS")
    assert v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("BLOCK: unauthorized student access.")
    assert v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("BLOCK\nThe student is not allowed to inspect another student's grade records.")
    assert v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("FAIL")
    assert v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("ASK for confirmation")
    assert v == "ASK" and s == "success"

    lbl, v, s = guard.parse_native_output("unknown nonsense")
    assert v == "UNKNOWN" and s == "failed"


def test_webguard_adapter_formatting_and_parsing(sample_case):
    guard = WebGuardBenchmarkGuard(name="webguard", config={"enabled": True, "augmented": True})
    prompt = guard.format_prompt(sample_case)

    # Check key components of WebGuard a11y tree risk format
    assert "Accessibility Tree of the webpage" in prompt
    assert "The accessibility tree of the web page is" in prompt
    assert "The proposed action is" in prompt
    assert '"risky_level"' in prompt

    # Check two-stage label mapping
    lbl, v, s = guard.parse_native_output('{"risky_level": "SAFE"}')
    assert lbl == "SAFE" and v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output('{"risky_level": "LOW"}')
    assert lbl == "LOW" and v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output('{"risky_level": "HIGH"}')
    assert lbl == "HIGH" and v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("This action is classified as HIGH risk.")
    assert v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("unknown text")
    assert v == "UNKNOWN" and s == "failed"


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
