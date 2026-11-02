"""Test dataset structural invariants on GuardSLM_v3_Single_Master_Dataset.json."""

from src.benchmark.data.validator import validate_benchmark_dataset


def test_canonical_dataset_validation():
    dataset_path = "data/GuardSLM_v3_Single_Master_Dataset.json"
    res = validate_benchmark_dataset(dataset_path)

    assert res.is_valid is True, f"Dataset failed validation: {[i.message for i in res.issues if i.severity == 'CRITICAL']}"
    assert res.stats["total_cases"] == 200
    assert res.stats["policy_clauses_per_case"] == 36
    assert res.stats["pairs_total"] == 100
    assert res.stats["pair_types"]["authorization_flip"] == 80
    assert res.stats["pair_types"]["invariant_control"] == 20
    assert res.stats["verdict_distribution"]["ALLOW"] == 97
    assert res.stats["verdict_distribution"]["BLOCK"] == 95
    assert res.stats["verdict_distribution"]["ASK"] == 8
