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

    # Check native PolicyGuard trajectory action format
    assert prompt.startswith("Policy:")
    assert "Trajectory Actions:" in prompt
    assert "Step 1: Authenticate as" in prompt
    assert "Step 2: Navigate to" in prompt
    assert "Step 3:" in prompt
    assert "Output violation or no_violation." in prompt
    assert "Step 5: unknown" not in prompt

    # Check native label mapping
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

    # Check native DynaGuard [INST] Policy + Dialogue [/INST] <answer> format
    assert prompt.startswith("[INST] Evaluate the following dialogue for compliance with the given policy.")
    assert "## Policy" in prompt
    assert "## Dialogue" in prompt
    assert "User: As a" in prompt
    assert "Agent: I am about to" in prompt
    assert prompt.endswith("[/INST] <answer>")

    # Check native PASS / FAIL label mapping
    lbl, v, s = guard.parse_native_output("PASS")
    assert lbl == "PASS" and v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("<answer> PASS")
    assert lbl == "PASS" and v == "ALLOW" and s == "success"

    lbl, v, s = guard.parse_native_output("FAIL")
    assert lbl == "FAIL" and v == "BLOCK" and s == "success"

    lbl, v, s = guard.parse_native_output("<answer> FAIL")
    assert lbl == "FAIL" and v == "BLOCK" and s == "success"

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
