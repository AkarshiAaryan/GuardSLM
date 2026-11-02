"""Test strict separation of public model inputs from private gold annotations on GuardSLM v3."""

from src.benchmark.data.loader import BenchmarkLoader


def test_strict_separation_of_inputs_and_golds():
    loader = BenchmarkLoader(dataset_path="data/GuardSLM_v3_Single_Master_Dataset.json")
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
            assert key not in sanitized.get("model_input", {}), f"Prohibited gold key '{key}' found in model_input of {case.case_id}!"

        # Check that complete policy info is provided (36 natural language strings)
        assert len(case.policy_info) == 36, f"Case {case.case_id} has incomplete policy info!"
        assert len(sanitized["model_input"]["policy_info"]) == 36

        # Check required public fields in model_input
        mi = sanitized["model_input"]
        assert "page_context" in mi
        assert "actor_info" in mi
        assert "policy_info" in mi
        assert "user_request" in mi["page_context"]
        assert "proposed_action" in mi["page_context"]

    # Verify that gold annotations are separately maintained
    golds = loader.get_gold_annotations()
    assert len(golds) == 200
    for case in cases:
        assert case.case_id in golds
        g = golds[case.case_id]
        assert g.verdict in {"ALLOW", "BLOCK", "ASK"}
        assert len(g.policy_checks) == 36
