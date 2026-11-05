"""GuardSLM v3 Benchmark Data Schema Definitions (Strict Separation of Public Inputs and Golds)."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union


@dataclass
class PolicyClause:
    policy_id: str
    statement: str
    name: str = ""
    category: str = ""
    rationale: str = ""
    moodle_capability_or_fixture_fact: str = ""
    enforcement_level: str = "mandatory"
    source_pdf_pages_cited: List[str] = field(default_factory=list)
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyClause":
        return cls(
            policy_id=d.get("policy_id", ""),
            name=d.get("name", d.get("title", "")),
            category=d.get("category", ""),
            statement=d.get("statement", d.get("policy_text", "")),
            rationale=d.get("rationale", ""),
            moodle_capability_or_fixture_fact=d.get("moodle_capability_or_fixture_fact", ""),
            enforcement_level=d.get("enforcement_level", "mandatory"),
            source_pdf_pages_cited=d.get("source_pdf_pages_cited", []),
            raw_dict=d
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.raw_dict:
            return self.raw_dict
        return asdict(self)


@dataclass
class ProposedAction:
    operation: str
    target: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProposedAction":
        return cls(
            operation=d.get("operation", "click"),
            target=d.get("target", {})
        )


@dataclass
class ScreenshotMapping:
    pdf_id: str = ""
    source_pdf_filename: str = ""
    page_number_1_indexed: int = 1
    page_id: str = ""
    page_type: str = ""
    course_context_from_inventory: str = ""
    visible_controls_from_inventory: str = ""
    data_pii_visible_from_inventory: str = ""
    privacy_redaction_required_before_external_model_call: bool = False
    evidence_class: str = ""
    inventory_caveat: str = ""
    mapping_relation: str = "primary_action_ui"
    screenshot_attachment_key: str = ""
    image_status: str = "embedded_in_original_pdf_not_extracted"
    external_image_run_instruction: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScreenshotMapping":
        return cls(
            pdf_id=d.get("pdf_id", ""),
            source_pdf_filename=d.get("source_pdf_filename", ""),
            page_number_1_indexed=int(d.get("page_number_1_indexed", 1)),
            page_id=d.get("page_id", ""),
            page_type=d.get("page_type", ""),
            course_context_from_inventory=d.get("course_context_from_inventory", ""),
            visible_controls_from_inventory=d.get("visible_controls_from_inventory", ""),
            data_pii_visible_from_inventory=d.get("data_pii_visible_from_inventory", ""),
            privacy_redaction_required_before_external_model_call=d.get("privacy_redaction_required_before_external_model_call", False),
            evidence_class=d.get("evidence_class", ""),
            inventory_caveat=d.get("inventory_caveat", ""),
            mapping_relation=d.get("mapping_relation", "primary_action_ui"),
            screenshot_attachment_key=d.get("screenshot_attachment_key", ""),
            image_status=d.get("image_status", "embedded_in_original_pdf_not_extracted"),
            external_image_run_instruction=d.get("external_image_run_instruction", "")
        )



@dataclass
class ModelInputCase:
    case_id: str
    page_context: Dict[str, Any]
    actor_info: Dict[str, Any]
    policy_info: List[str]  # 36 plain strings in v3
    pair_id: str = field(init=False)
    variant: str = field(init=False)

    def __post_init__(self):
        parts = self.case_id.rsplit("-", 1)
        self.pair_id = parts[0]
        self.variant = parts[1] if len(parts) > 1 else ""

    # Convenient accessors for model adapters and prompts
    @property
    def user_request(self) -> str:
        return self.page_context.get("user_request", "")

    @property
    def authenticated_role(self) -> str:
        return self.actor_info.get("authenticated_role", "unknown")

    @property
    def task(self) -> Dict[str, Any]:
        return {
            "user_request": self.user_request,
            "conversation_history": self.page_context.get("conversation_history", [])
        }

    @property
    def actor_context(self) -> Dict[str, Any]:
        return self.actor_info

    @property
    def resource_context(self) -> Dict[str, Any]:
        target = self.page_context.get("target", {})
        return target.get("resource", {}) if isinstance(target, dict) else {}

    @property
    def state_context(self) -> Dict[str, Any]:
        target = self.page_context.get("target", {})
        return target.get("state", {}) if isinstance(target, dict) else {}

    @property
    def proposed_action(self) -> ProposedAction:
        pa = self.page_context.get("proposed_action", {})
        if isinstance(pa, ProposedAction):
            return pa
        return ProposedAction.from_dict(pa if isinstance(pa, dict) else {})

    @property
    def page_observation(self) -> Any:
        page = self.page_context.get("page", {})
        return type("PageObservationProxy", (), {
            "text_only_representation": type("TextRepProxy", (), {
                "page_type": page.get("page_type", ""),
                "course_context": page.get("course_context", ""),
                "visible_controls": page.get("visible_controls", ""),
                "evidence_class": page.get("evidence_class", ""),
                "known_limitations": page.get("known_limitations", "")
            })(),
            "fixture_state_facts": self.state_context,
            "inferred_accessibility_tree": None,
            "original_pdf_screenshots": []
        })()

    @property
    def policy_book(self) -> List[PolicyClause]:
        return [
            PolicyClause(policy_id=f"POL-{i+1:02d}", statement=s, name=f"Policy {i+1}")
            for i, s in enumerate(self.policy_info)
        ]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelInputCase":
        cid = d.get("case_id", "")
        if "model_input" in d:
            mi = d["model_input"]
            return cls(
                case_id=cid,
                page_context=mi.get("page_context", {}),
                actor_info=mi.get("actor_info", {}),
                policy_info=mi.get("policy_info", [])
            )
        elif "page_context" in d and "actor_info" in d:
            return cls(
                case_id=cid,
                page_context=d.get("page_context", {}),
                actor_info=d.get("actor_info", {}),
                policy_info=d.get("policy_info", [])
            )
        else:
            page_ctx = {
                "user_request": d.get("task", {}).get("user_request", ""),
                "conversation_history": [],
                "page": d.get("page_observation", {}).get("text_only_representation", {}),
                "target": {
                    "resource": d.get("resource_context", {}),
                    "state": d.get("state_context", {})
                },
                "proposed_action": d.get("proposed_action", {})
            }
            policy_strings = [
                p.get("statement", "") if isinstance(p, dict) else str(p)
                for p in d.get("policy_book", [])
            ]
            return cls(
                case_id=cid,
                page_context=page_ctx,
                actor_info=d.get("actor_context", {}),
                policy_info=policy_strings
            )

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """Returns the public model input strictly free of any gold annotations."""
        return {
            "case_id": self.case_id,
            "model_input": {
                "page_context": self.page_context,
                "actor_info": self.actor_info,
                "policy_info": self.policy_info
            }
        }


@dataclass
class PolicyCheck:
    policy_id: str
    status: str  # not_applicable, satisfied, violated, insufficient_evidence
    evidence_quotes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyCheck":
        return cls(
            policy_id=d["policy_id"],
            status=d.get("status", "not_applicable"),
            evidence_quotes=d.get("evidence_quotes", [])
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrivateGoldAnnotation:
    case_id: str
    pair_id: str
    verdict: str  # ALLOW, BLOCK, ASK
    applicable_policy_ids: List[str]
    violated_policy_ids: List[str]
    policy_checks: List[PolicyCheck]
    evidence: List[Any]
    rationale: str
    annotation_status: str
    gold_provenance: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PrivateGoldAnnotation":
        checks = [PolicyCheck.from_dict(c) for c in d.get("policy_checks", [])]
        return cls(
            case_id=d["case_id"],
            pair_id=d["pair_id"],
            verdict=d["verdict"].upper(),
            applicable_policy_ids=d.get("applicable_policy_ids", []),
            violated_policy_ids=d.get("violated_policy_ids", []),
            policy_checks=checks,
            evidence=d.get("evidence", []),
            rationale=d.get("rationale", ""),
            annotation_status=d.get("annotation_status", ""),
            gold_provenance=d.get("gold_provenance", "")
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "pair_id": self.pair_id,
            "verdict": self.verdict,
            "applicable_policy_ids": self.applicable_policy_ids,
            "violated_policy_ids": self.violated_policy_ids,
            "policy_checks": [c.to_dict() for c in self.policy_checks],
            "evidence": self.evidence,
            "rationale": self.rationale,
            "annotation_status": self.annotation_status,
            "gold_provenance": self.gold_provenance
        }


@dataclass
class PairRegistryEntry:
    pair_id: str
    scenario_family: str
    scenario_id: str
    pair_type: str  # authorization_flip or invariant_control
    user_request: str
    proposed_action: Dict[str, Any]
    intervention_variable: str
    intervention_description: str
    value_A: Any
    value_B: Any
    expected_verdict_A: str
    expected_verdict_B: str
    violated_policy_ids_A: List[str] = field(default_factory=list)
    violated_policy_ids_B: List[str] = field(default_factory=list)
    source_screenshot_A: Dict[str, Any] = field(default_factory=dict)
    source_screenshot_B: Dict[str, Any] = field(default_factory=dict)
    visual_support_level: str = ""
    pair_quality_note: str = ""
    one_factor_design: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PairRegistryEntry":
        return cls(
            pair_id=d["pair_id"],
            scenario_family=d.get("scenario_family", ""),
            scenario_id=d.get("scenario_id", ""),
            pair_type=d.get("pair_type", "authorization_flip"),
            user_request=d.get("user_request", ""),
            proposed_action=d.get("proposed_action", {}),
            intervention_variable=d.get("intervention_variable", ""),
            intervention_description=d.get("intervention_description", ""),
            value_A=d.get("value_A"),
            value_B=d.get("value_B"),
            expected_verdict_A=d.get("expected_verdict_A", ""),
            expected_verdict_B=d.get("expected_verdict_B", ""),
            violated_policy_ids_A=d.get("violated_policy_ids_A", []),
            violated_policy_ids_B=d.get("violated_policy_ids_B", []),
            source_screenshot_A=d.get("source_screenshot_A", {}),
            source_screenshot_B=d.get("source_screenshot_B", {}),
            visual_support_level=d.get("visual_support_level", ""),
            pair_quality_note=d.get("pair_quality_note", ""),
            one_factor_design=d.get("one_factor_design", "")
        )


@dataclass
class SnapshotInputCase:
    """Benchmark test case accompanied by authentic Playwright accessibility tree snapshots."""
    case_id: str
    title: str
    account: str
    global_role: str
    effective_role: str
    membership: str
    timestamp: str
    timezone: str
    deadline_ts: str
    course_or_scope: str
    target_resource: str
    requested_operation: str
    target_owner: str
    policy_context: str
    target_a11y_tree: str
    policy_a11y_tree: str = ""
    pair_id: str = ""
    variant: str = ""
    is_snapshot: bool = True

    # Standardized properties compatible with ModelInputCase
    @property
    def user_request(self) -> str:
        return self.requested_operation

    @property
    def authenticated_role(self) -> str:
        return self.effective_role

    @property
    def policy_info(self) -> List[str]:
        lines = [p.strip() for p in self.policy_context.split("\n\n") if p.strip()]
        return lines if lines else [self.policy_context]

    @property
    def page_context(self) -> Dict[str, Any]:
        return {
            "user_request": self.requested_operation,
            "target": {
                "resource": {"course_id": self.course_or_scope, "resource": self.target_resource},
                "state": {"timestamp": self.timestamp, "deadline": self.deadline_ts}
            },
            "proposed_action": {
                "operation": self.requested_operation,
                "target": {"resource": self.target_resource, "owner": self.target_owner}
            },
            "page": {
                "course_context": self.course_or_scope,
                "page_type": "lms_page",
                "visible_controls": self.target_resource
            },
            "target_a11y_tree": self.target_a11y_tree,
            "policy_a11y_tree": self.policy_a11y_tree
        }

    @property
    def actor_info(self) -> Dict[str, Any]:
        return {
            "account": self.account,
            "global_role": self.global_role,
            "effective_role": self.effective_role,
            "membership": self.membership
        }

    @property
    def proposed_action(self) -> ProposedAction:
        return ProposedAction(
            operation=self.requested_operation,
            target={"resource": self.target_resource, "owner": self.target_owner}
        )

    @property
    def state_context(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "deadline": self.deadline_ts,
            "membership": self.membership
        }

    @property
    def resource_context(self) -> Dict[str, Any]:
        return {
            "course_id": self.course_or_scope,
            "resource": self.target_resource
        }

    @property
    def policy_book(self) -> List[PolicyClause]:
        return [
            PolicyClause(policy_id=f"POL-{i+1:02d}", statement=s, name=f"Policy {i+1}")
            for i, s in enumerate(self.policy_info)
        ]

    @property
    def page_observation(self) -> Any:
        return type("PageObservationProxy", (), {
            "text_only_representation": type("TextRepProxy", (), {
                "page_type": "lms_page",
                "course_context": self.course_or_scope,
                "visible_controls": self.target_resource,
                "evidence_class": "accessibility_tree",
                "known_limitations": ""
            })(),
            "fixture_state_facts": self.state_context,
            "inferred_accessibility_tree": self.target_a11y_tree,
            "original_pdf_screenshots": []
        })()

    def to_sanitized_dict(self) -> Dict[str, Any]:

        """Strict zero gold leakage representation of snapshot case."""
        return {
            "case_id": self.case_id,
            "title": self.title,
            "actor": {
                "account": self.account,
                "global_role": self.global_role,
                "effective_role": self.effective_role,
                "membership": self.membership
            },
            "decision_time": {
                "timestamp": self.timestamp,
                "timezone": self.timezone,
                "deadline": self.deadline_ts
            },
            "target_and_action": {
                "course_or_scope": self.course_or_scope,
                "target_resource": self.target_resource,
                "requested_operation": self.requested_operation,
                "target_owner": self.target_owner
            },
            "policy_context": self.policy_context,
            "target_a11y_tree": self.target_a11y_tree,
            "policy_a11y_tree": self.policy_a11y_tree
        }

