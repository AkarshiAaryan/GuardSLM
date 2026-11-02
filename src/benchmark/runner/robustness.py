"""GuardSLM Robustness Experiments: Policy-Order Shuffling & Clause Ablations."""

import random
from typing import List, Dict, Any, Optional
from ..data.schema import ModelInputCase, PolicyClause


def shuffle_policy_order(policies: List[PolicyClause], seed: int = 42) -> List[str]:
    """Returns a shuffled list of policy IDs for testing sensitivity to rule order."""
    rng = random.Random(seed)
    pids = [p.policy_id for p in policies]
    rng.shuffle(pids)
    return pids


def filter_irrelevant_clauses(
    policies: List[PolicyClause],
    relevant_policy_ids: List[str],
    distractor_count: int = 3,
    seed: int = 42
) -> List[PolicyClause]:
    """Ablation: Removes all irrelevant policy clauses except a controlled number of distractors."""
    relevant_set = set(relevant_policy_ids)
    rel_clauses = [p for p in policies if p.policy_id in relevant_set]
    other_clauses = [p for p in policies if p.policy_id not in relevant_set]

    rng = random.Random(seed)
    distractors = rng.sample(other_clauses, min(distractor_count, len(other_clauses)))
    combined = rel_clauses + distractors
    rng.shuffle(combined)
    return combined


def omit_violated_clauses(
    policies: List[PolicyClause],
    violated_policy_ids: List[str]
) -> List[PolicyClause]:
    """Ablation: Omits the rule responsible for a violation to test if the model hallucinates a violation."""
    viol_set = set(violated_policy_ids)
    return [p for p in policies if p.policy_id not in viol_set]
