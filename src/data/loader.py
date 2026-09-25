import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

@dataclass
class User:
    role: Union[str, bool]

    def get_role_str(self) -> str:
        if isinstance(self.role, bool):
            return "true" if self.role else "false"
        return str(self.role)

@dataclass
class PageContext:
    page_title: str
    accessibility_tree: str

@dataclass
class Context:
    page_context: PageContext
    moodle_state: Dict[str, Any]
    policy: str

@dataclass
class ProposedAction:
    element: str
    tool: str = "browser_click"
    type: str = "click"
    target: str = ""
    ref: str = ""

@dataclass
class Gold:
    verdict: str  # ALLOW, BLOCK, ASK
    reason: str = ""

@dataclass
class TestCase:
    case_id: str
    pair_id: str
    template_id: str
    user: User
    request: str
    context: Context
    proposed_action: ProposedAction
    gold: Gold
    template_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestCase":
        return cls(
            case_id=d["case_id"],
            pair_id=d["pair_id"],
            template_id=d["template_id"],
            template_name=d.get("template_name", ""),
            user=User(role=d["user"]["role"]),
            request=d["request"],
            context=Context(
                page_context=PageContext(
                    page_title=d["context"]["page_context"].get("page_title", ""),
                    accessibility_tree=d["context"]["page_context"].get("accessibility_tree", "")
                ),
                moodle_state=d["context"].get("moodle_state", {}),
                policy=d["context"].get("policy", "")
            ),
            proposed_action=ProposedAction(
                element=d["proposed_action"].get("element", d["proposed_action"].get("target", "")),
                tool=d["proposed_action"].get("tool", "browser_click"),
                type=d["proposed_action"].get("type", "click"),
                target=d["proposed_action"].get("target", d["proposed_action"].get("element", "")),
                ref=d["proposed_action"].get("ref", "")
            ),
            gold=Gold(
                verdict=d["gold"]["verdict"].upper(),
                reason=d["gold"].get("reason", "")
            ),
            metadata=d.get("metadata", {})
        )


def load_test_cases(filepath: str) -> List[TestCase]:
    """Loads and parses test cases from a JSON dataset file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_cases = data.get("cases", data) if isinstance(data, dict) else data
    if not isinstance(raw_cases, list):
        raise ValueError("Invalid dataset format: expected a list of cases or a wrapper dict with a 'cases' key.")

    return [TestCase.from_dict(c) for c in raw_cases]
