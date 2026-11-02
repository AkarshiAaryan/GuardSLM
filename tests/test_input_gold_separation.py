"""Test strict separation of public model inputs from private gold annotations."""

from src.benchmark.data.loader import BenchmarkLoader


def test_strict_separation_of_inputs_and_golds():
    loader = BenchmarkLoader()
    cases = loader.get_model_inputs()
    assert len(cases) == 200

    prohibited_keys = {
        "verdict", "gold_verdict", "violated_policy_ids",
        "gold_provenance", "annotation_status", "expected_verdict"
    }

    for case in cases:
        sanitized = case.to_sanitized_dict()

        # Check top level
        for key in prohibited_keys:
            assert key not in sanitized, f"Prohibited gold key '{key}' found in model input {case.case_id}!"

        # Check that complete policy book is provided
        assert len(case.policy_book) == 36, f"Case {case.case_id} has incomplete policy book!"
        assert len(sanitized["policy_book"]) == 36

        # Check required public fields
        assert "task" in sanitized
        assert "actor_context" in sanitized
        assert "resource_context" in sanitized
        assert "state_context" in sanitized
        assert "page_observation" in sanitized
        assert "proposed_action" in sanitized
        assert "required_output" in sanitized

    # Verify that gold annotations are separately maintained
    golds = loader.get_gold_annotations()
    assert len(golds) == 200
    for case in cases:
        assert case.case_id in golds
        g = golds[case.case_id]
        assert g.verdict in {"ALLOW", "BLOCK", "ASK"}
        assert len(g.policy_checks) == 36
