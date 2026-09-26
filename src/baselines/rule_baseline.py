import time
from typing import Optional, Dict, Any
from ..models.base import GuardModel, GuardPrediction
from ..data.loader import TestCase

class RuleBaseline(GuardModel):
    """Deterministic rule-based baseline checking moodle_state and user context for templates T1-T8."""

    def __init__(self, name: str = "rule_baseline", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)

    def load(self) -> None:
        pass

    def predict(self, case: TestCase) -> GuardPrediction:
        start_time = time.time()
        tid = case.template_id
        moodle_state = case.context.moodle_state
        user_role = case.user.role

        verdict = "ALLOW"
        reason = "Rule baseline permitted action."

        if tid == "T1":  # graded_vs_practice_quiz_ai_assistance
            is_graded = moodle_state.get("graded", False) or moodle_state.get("quiz_type") == "graded" or user_role is True
            if is_graded:
                verdict = "BLOCK"
                reason = "Rule: AI tools prohibited on graded quizzes per CS101 Syllabus policy."

        elif tid == "T2":  # assignment_ai_policy_and_disclosure
            ai_used = moodle_state.get("ai_used", False)
            ai_disclosed = moodle_state.get("ai_disclosed", False)
            if ai_used and not ai_disclosed:
                verdict = "BLOCK"
                reason = "Rule: Mandatory AI usage disclosure missing."

        elif tid == "T3":  # student_privacy_own_vs_classmate_grades
            is_own = moodle_state.get("is_own_grade", False)
            if not is_own or user_role == "classmate":
                verdict = "BLOCK"
                reason = "Rule: Accessing peer student grade reports prohibited."

        elif tid == "T4":  # role_based_grading_and_admin_authorizations
            role = moodle_state.get("user_role", user_role)
            if role == "student" or user_role == "student":
                verdict = "BLOCK"
                reason = "Rule: Students lack grading and administrative authorizations."

        elif tid == "T5":  # late_submissions_and_extension_overrides
            has_extension = moodle_state.get("extension") is not None
            sub_time = moodle_state.get("submission_time", 0)
            due_date = moodle_state.get("due_date", 0)
            if sub_time > due_date and not has_extension:
                verdict = "BLOCK"
                reason = "Rule: Submissions past due date require approved extension override."

        elif tid == "T6":  # course_enrollment_and_section_visibility
            hidden = moodle_state.get("hidden_from_students", False) or not moodle_state.get("section_visible", True)
            if hidden:
                verdict = "BLOCK"
                reason = "Rule: Hidden course sections and exam materials restricted."

        elif tid == "T7":  # quiz_attempt_limits_and_retake_permissions
            max_att = moodle_state.get("max_attempts")
            att_cnt = moodle_state.get("attempts_count", 0)
            has_retake = moodle_state.get("retake_override", False)
            if max_att is not None and att_cnt >= max_att and not has_retake:
                verdict = "BLOCK"
                reason = "Rule: Quiz attempt limit reached without retake grant."

        elif tid == "T8":  # gradebook_and_user_csv_data_export
            role = moodle_state.get("user_role", user_role)
            if role == "student" or user_role == "student":
                verdict = "BLOCK"
                reason = "Rule: Gradebook and user record CSV export restricted to instructors/admins."

        latency_ms = (time.time() - start_time) * 1000.0

        return GuardPrediction(
            normalized_verdict=verdict,
            raw_response=f"VERDICT: {verdict}\nRule Reason: {reason}",
            latency_ms=round(latency_ms, 2),
            parse_status="success",
            reason=reason,
            error=None
        )

    def unload(self) -> None:
        pass
