"""Test dataset structural invariants on GuardSLM_100_Pairs_v2.json."""

from src.benchmark.data.validator import validate_benchmark_dataset


def test_canonical_dataset_validation():
    dataset_path = "data/GuardSLM_100_Pairs_v2.json"
    res = validate_benchmark_dataset(dataset_path)

    assert res.is_valid is True, f"Dataset failed validation: {[i.message for i in res.issues if i.severity == 'CRITICAL']}"
    assert res.stats["policy_clauses"] == 36
    assert res.stats["source_pdfs"] == 6
    assert res.stats["model_inputs"] == 200
    assert res.stats["gold_annotations"] == 200
    assert res.stats["pairs_total"] == 100
    assert res.stats["pair_types"]["authorization_flip"] == 80
    assert res.stats["pair_types"]["invariant_control"] == 20
    assert res.stats["verdict_distribution"]["ALLOW"] == 97
    assert res.stats["verdict_distribution"]["BLOCK"] == 95
    assert res.stats["verdict_distribution"]["ASK"] == 8
