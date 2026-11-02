"""GuardSLM 100-Pair Benchmark Data Schema Definitions."""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union


@dataclass
class PolicyClause:
    policy_id: str
    name: str
    category: str
    statement: str
    rationale: str
    moodle_capability_or_fixture_fact: str
    enforcement_level: str
    source_pdf_pages_cited: List[str] = field(default_factory=list)
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyClause":
        return cls(
            policy_id=d["policy_id"],
            name=d.get("name", ""),
            category=d.get("category", ""),
            statement=d.get("statement", ""),
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
class ScreenshotMapping:
    pdf_id: str
    source_pdf_filename: str
    page_number_1_indexed: int
    page_id: str
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
            pdf_id=d["pdf_id"],
            source_pdf_filename=d["source_pdf_filename"],
            page_number_1_indexed=int(d["page_number_1_indexed"]),
            page_id=d["page_id"],
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
class TextOnlyRepresentation:
    page_type: str
    course_context: str
    visible_controls: str
    evidence_class: str
    known_limitations: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TextOnlyRepresentation":
        return cls(
            page_type=d.get("page_type", ""),
            course_context=d.get("course_context", ""),
            visible_controls=d.get("visible_controls", ""),
            evidence_class=d.get("evidence_class", ""),
            known_limitations=d.get("known_limitations", "")
        )


@dataclass
class PageObservation:
    original_pdf_screenshots: List[ScreenshotMapping]
    supporting_source_pages: List[Any]
    text_only_representation: TextOnlyRepresentation
    fixture_state_facts: Dict[str, Any]
    inferred_accessibility_tree: Optional[Any]
    accessibility_tree_status: str
    screenshot_condition_instruction: str
    text_only_condition_instruction: str
    modality_comparison_control: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PageObservation":
        screenshots = [ScreenshotMapping.from_dict(s) for s in d.get("original_pdf_screenshots", [])]
        text_rep = TextOnlyRepresentation.from_dict(d.get("text_only_representation", {}))
        return cls(
            original_pdf_screenshots=screenshots,
            supporting_source_pages=d.get("supporting_source_pages", []),
            text_only_representation=text_rep,
            fixture_state_facts=d.get("fixture_state_facts", {}),
            inferred_accessibility_tree=d.get("inferred_accessibility_tree"),
            accessibility_tree_status=d.get("accessibility_tree_status", ""),
            screenshot_condition_instruction=d.get("screenshot_condition_instruction", ""),
            text_only_condition_instruction=d.get("text_only_condition_instruction", ""),
            modality_comparison_control=d.get("modality_comparison_control", "")
        )


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
class ModelInputCase:
    case_id: str
    task: Dict[str, Any]
    actor_context: Dict[str, Any]
    resource_context: Dict[str, Any]
    state_context: Dict[str, Any]
    page_observation: PageObservation
    proposed_action: ProposedAction
    policy_book_version: str
    policy_book: List[PolicyClause]
    required_output: Dict[str, Any]
    pair_id: str = field(init=False)
    variant: str = field(init=False)

    def __post_init__(self):
        parts = self.case_id.rsplit("-", 1)
        self.pair_id = parts[0]
        self.variant = parts[1] if len(parts) > 1 else ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelInputCase":
        policies = [PolicyClause.from_dict(p) for p in d.get("policy_book", [])]
        page_obs = PageObservation.from_dict(d.get("page_observation", {}))
        action = ProposedAction.from_dict(d.get("proposed_action", {}))
        return cls(
            case_id=d["case_id"],
            task=d.get("task", {}),
            actor_context=d.get("actor_context", {}),
            resource_context=d.get("resource_context", {}),
            state_context=d.get("state_context", {}),
            page_observation=page_obs,
            proposed_action=action,
            policy_book_version=d.get("policy_book_version", ""),
            policy_book=policies,
            required_output=d.get("required_output", {})
        )

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """Returns the public model input strictly free of any gold annotations."""
        return {
            "case_id": self.case_id,
            "task": self.task,
            "actor_context": self.actor_context,
            "resource_context": self.resource_context,
            "state_context": self.state_context,
            "page_observation": {
                "text_only_representation": asdict(self.page_observation.text_only_representation),
                "fixture_state_facts": self.page_observation.fixture_state_facts,
                "screenshot_metadata": [
                    {
                        "pdf_id": s.pdf_id,
                        "source_pdf_filename": s.source_pdf_filename,
                        "page_number_1_indexed": s.page_number_1_indexed,
                        "page_id": s.page_id,
                        "page_type": s.page_type,
                        "evidence_class": s.evidence_class,
                        "screenshot_attachment_key": s.screenshot_attachment_key
                    } for s in self.page_observation.original_pdf_screenshots
                ]
            },
            "proposed_action": asdict(self.proposed_action),
            "policy_book_version": self.policy_book_version,
            "policy_book": [p.to_dict() for p in self.policy_book],
            "required_output": self.required_output
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
    violated_policy_ids_A: List[str]
    violated_policy_ids_B: List[str]
    source_screenshot_A: Dict[str, Any]
    source_screenshot_B: Dict[str, Any]
    visual_support_level: str
    pair_quality_note: str
    one_factor_design: str

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
