"""GuardSLM Benchmark Prompt Renderer for v3 Master Dataset."""

import json
import os
from typing import Dict, Any, List, Optional
from ..data.schema import ModelInputCase, PolicyClause


DEFAULT_TEMPLATE_PATH = "prompts/guardslm_100_authorization_prompt.txt"


def format_policy_book(policies: Any) -> str:
    """Formats the policy book rules into clear, structured text."""
    lines = []
    if not policies:
        return "Ensure all actions comply with institutional and course policies."

    # If list of plain strings (v3 format)
    if isinstance(policies[0], str):
        for idx, rule_str in enumerate(policies, 1):
            lines.append(f"Policy Rule #{idx}: {rule_str}")
        return "\n\n".join(lines).strip()

    # If list of PolicyClause objects
    for idx, p in enumerate(policies, 1):
        if hasattr(p, "statement") and p.statement:
            lines.append(f"Policy Rule #{idx} [{getattr(p, 'policy_id', f'POL-{idx:02d}')}]: {p.statement}")
        else:
            lines.append(f"Policy Rule #{idx}: {str(p)}")
    return "\n\n".join(lines).strip()


def render_benchmark_prompt(
    case: ModelInputCase,
    template_path: str = DEFAULT_TEMPLATE_PATH,
    policy_order: Optional[List[str]] = None,
    excluded_policy_ids: Optional[List[str]] = None
) -> str:
    """Renders the canonical authorization evaluation prompt for a given ModelInputCase."""
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Prompt template file not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template_str = f.read()

    task_obj = case.task
    user_req = task_obj.get("user_request", "")
    history = task_obj.get("conversation_history", [])
    task_summary = f"User Request: {user_req}"
    if history:
        task_summary += f"\nConversation History:\n{json.dumps(history, indent=2)}"

    actor = case.actor_context
    resource = case.resource_context
    obs = case.page_observation
    text_rep = obs.text_only_representation
    action = case.proposed_action

    fixture_facts = obs.fixture_state_facts
    fixture_facts_text = f"- Fixture Facts: {json.dumps(fixture_facts)}" if fixture_facts else ""

    policy_source = case.policy_info if case.policy_info else case.policy_book
    policy_formatted = format_policy_book(policy_source)

    replacements = {
        "case_id": case.case_id,
        "task_summary": task_summary,
        "actor_id": actor.get("actor_id", "N/A"),
        "authenticated_role": actor.get("authenticated_role", "N/A"),
        "role_verification_status": actor.get("role_verification_status", "N/A"),
        "enrolled_course_ids": json.dumps(actor.get("enrolled_course_ids", [])),
        "assigned_course_ids": json.dumps(actor.get("assigned_course_ids", [])),
        "preview_role_selected": actor.get("preview_role_selected") or "None",
        "resource_course_id": resource.get("course_id", "N/A"),
        "resource_id": resource.get("resource_id", "N/A"),
        "resource_type": resource.get("resource_type", "N/A"),
        "owner_actor_id": resource.get("owner_actor_id") or "None",
        "state_context_json": json.dumps(case.state_context, indent=2),
        "page_type": text_rep.page_type,
        "page_course_context": text_rep.course_context,
        "visible_controls": text_rep.visible_controls,
        "evidence_class": text_rep.evidence_class,
        "known_limitations": text_rep.known_limitations,
        "fixture_state_facts_text": fixture_facts_text,
        "action_operation": action.operation,
        "action_target_json": json.dumps(action.target),
        "policy_book_formatted": policy_formatted
    }

    prompt = template_str
    for key, val in replacements.items():
        prompt = prompt.replace(f"{{{key}}}", str(val))

    return prompt
