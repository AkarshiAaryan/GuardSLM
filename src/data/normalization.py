from typing import Dict, Any
from .loader import TestCase, User, Context, PageContext, ProposedAction, Gold

def normalize_moodle_raw_case(raw_dict: Dict[str, Any]) -> TestCase:
    """Normalizes raw Moodle web service / log interceptor payload into standard TestCase object.

    This function isolates Moodle-specific dictionary fields from the core evaluation engine.
    When real Moodle data is extracted, update this single file rather than model adapters or metrics.
    """
    # Extract case metadata
    case_id = raw_dict.get("case_id", raw_dict.get("id", "UNKNOWN"))
    pair_id = raw_dict.get("pair_id", raw_dict.get("twin_id", "UNKNOWN"))
    template_id = raw_dict.get("template_id", "T1")
    template_name = raw_dict.get("template_name", "")

    # Extract user role
    raw_user = raw_dict.get("user", {})
    role = raw_user.get("role", raw_dict.get("role", "student"))

    # Extract request
    request = raw_dict.get("request", raw_dict.get("user_request", ""))

    # Extract context
    raw_context = raw_dict.get("context", {})
    page_context_raw = raw_context.get("page_context", {})
    page_title = page_context_raw.get("page_title", raw_dict.get("page_title", ""))
    accessibility_tree = page_context_raw.get("accessibility_tree", raw_dict.get("page_snapshot", ""))
    moodle_state = raw_context.get("moodle_state", raw_dict.get("db_facts", {}))
    policy = raw_context.get("policy", raw_dict.get("ai_policy", ""))

    # Extract proposed action
    raw_action = raw_dict.get("proposed_action", {})
    element = raw_action.get("element", raw_action.get("target", ""))
    tool = raw_action.get("tool", "browser_click")
    action_type = raw_action.get("type", "click")
    target = raw_action.get("target", element)
    ref = raw_action.get("ref", "")

    # Extract gold label
    raw_gold = raw_dict.get("gold", {})
    verdict = raw_gold.get("verdict", raw_dict.get("label", "ASK")).upper()
    reason = raw_gold.get("reason", raw_dict.get("gold_reason", ""))

    return TestCase(
        case_id=case_id,
        pair_id=pair_id,
        template_id=template_id,
        template_name=template_name,
        user=User(role=role),
        request=request,
        context=Context(
            page_context=PageContext(
                page_title=page_title,
                accessibility_tree=accessibility_tree
            ),
            moodle_state=moodle_state,
            policy=policy
        ),
        proposed_action=ProposedAction(
            element=element,
            tool=tool,
            type=action_type,
            target=target,
            ref=ref
        ),
        gold=Gold(verdict=verdict, reason=reason),
        metadata=raw_dict.get("metadata", {})
    )
