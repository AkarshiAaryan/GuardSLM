import json
import os
from typing import Dict, Any
from ..data.loader import TestCase

def render_prompt(template_path: str, case: TestCase) -> str:
    """Renders a prompt text template with variables from a TestCase object."""
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Prompt template file not found: {template_path}")

    with open(template_path, "r", encoding="utf-8") as f:
        template_str = f.read()

    context_summary = (
        f"Page Title: {case.context.page_context.page_title}\n"
        f"Accessibility Tree:\n{case.context.page_context.accessibility_tree}\n"
        f"Platform State: {json.dumps(case.context.moodle_state)}"
    )

    action_summary = (
        f"Tool: {case.proposed_action.tool}, Element: {case.proposed_action.element}"
    )

    replacements = {
        "user_role": case.user.get_role_str(),
        "request": case.request,
        "context": context_summary,
        "page_title": case.context.page_context.page_title,
        "accessibility_tree": case.context.page_context.accessibility_tree,
        "moodle_state": json.dumps(case.context.moodle_state),
        "policy": case.context.policy,
        "proposed_action": action_summary
    }

    prompt = template_str
    for key, val in replacements.items():
        prompt = prompt.replace(f"{{{key}}}", str(val))

    return prompt
