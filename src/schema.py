"""Classification schema — Guardrail #1: structured validation boundary.

Every classification (whether produced by the deterministic mock classifier or by
the Anthropic API) must pass through this module before the graph is allowed to
act on it. The schema enforces three things:

1. `category` is one of a fixed allowlist (no free-text categories leak through).
2. `severity` is one of a fixed allowlist (including "critical").
3. `confidence` is a float in the closed interval [0, 1].

Anything that does not conform is not raised up the stack and is *never propagated
as-is*: `safe_parse()` catches the failure and downgrades it to a safe,
low-confidence fallback. Downstream guardrails (the confidence threshold in a
later step) then naturally route that fallback to a human for review.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, ValidationError


class Category(StrEnum):
    """Allowlist of ticket categories for a video-platform support desk.

    Using a StrEnum (a) restricts the LLM output to known values and (b) still
    serializes to plain strings for JSON and logging.
    """

    PLAYBACK = "playback"      # player crashes, buffering, video won't start
    BILLING = "billing"        # payments, subscriptions, refunds, invoices
    ACCOUNT = "account"        # login, password, profile, access
    CONTENT = "content"        # missing/incorrect videos, metadata, subtitles
    OTHER = "other"            # anything that does not fit the above


class Severity(StrEnum):
    """Allowlist of severities, ordered low -> critical.

    "critical" is meaningful for a later business-rule guardrail: a critical
    ticket is never auto-handled, it is always escalated.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(StrEnum):
    """The routing outcome for a ticket — what the graph decided to do.

    Values double as the node names they route to, so the router can return an
    Action directly. Keeping this typed (instead of bare strings) makes the final
    decision easy to inspect and serialize.
    """

    AUTO_HANDLE = "auto_handle"      # confident, non-critical -> handled automatically
    ESCALATE = "escalate"           # critical, or otherwise needs a specialist
    HUMAN_REVIEW = "human_review"   # low confidence -> a human decides


class Classification(BaseModel):
    """A validated triage decision about a single support ticket.

    Pydantic guarantees, at construction time, that:
    - `category` / `severity` are members of their allowlists (else ValidationError),
    - `confidence` is a float within [0, 1] (via the ge/le constraints below).
    """

    category: Category
    severity: Severity
    # `ge`/`le` = greater-than-or-equal / less-than-or-equal: closed interval [0, 1].
    confidence: float = Field(ge=0.0, le=1.0)
    # A short human-readable justification. Handy for the audit trail later; kept
    # optional so a minimal/fallback classification is still valid.
    rationale: str = ""


class AuditEntry(BaseModel):
    """One line of the audit trail (the observability seed).

    A structured, typed record — instead of a raw string — so the trail can be
    inspected, filtered, or serialized later. Each node that acts appends one.
    """

    node: str       # which node wrote this entry (e.g. "classify")
    message: str    # what it decided and why


def safe_parse(raw: object) -> Classification:
    """Validate arbitrary data into a Classification, never raising.

    This is the enforcement point of Guardrail #1. `raw` is whatever a classifier
    produced — typically a dict parsed from LLM JSON, which we cannot trust.

    - On success: return the validated Classification.
    - On any validation failure (unknown category, out-of-range confidence,
      missing field, wrong type, ...): return a deliberately *safe* fallback —
      category OTHER, severity LOW, confidence 0.0 — so the malformed output is
      neither trusted nor propagated. Confidence 0.0 guarantees it will fall
      below the later confidence threshold and be sent to human review.
    """
    try:
        return Classification.model_validate(raw)
    except ValidationError as exc:
        return Classification(
            category=Category.OTHER,
            severity=Severity.LOW,
            confidence=0.0,
            rationale=f"Downgraded: classifier output failed validation ({exc.error_count()} error(s)).",
        )
