"""Classifiers — turn raw ticket text into a validated Classification.

Two interchangeable implementations share one contract:
`Callable[[str], Classification]`.

- `mock_classify`  : deterministic keyword matching, no network, no API key.
                     Builds a Classification directly (still schema-validated by
                     construction).
- `real_classify`  : (stubbed for now) will call the Anthropic API, then route the
                     JSON through `safe_parse` — implemented in a later step.

`get_classifier()` picks the implementation based on whether `ANTHROPIC_API_KEY`
is set, so the rest of the system depends only on the *contract*, never on how a
classification was produced.
"""

import os
from collections.abc import Callable

from src.schema import Category, Classification, Severity

# A classifier is any callable that maps ticket text to a validated Classification.
Classifier = Callable[[str], Classification]


# --- Keyword tables for the deterministic mock -----------------------------------
# Order matters only for readability; scoring below counts hits per category.
_CATEGORY_KEYWORDS: dict[Category, tuple[str, ...]] = {
    Category.PLAYBACK: ("player", "playback", "buffer", "buffering", "stream",
                        "video", "watch", "crash", "500"),
    Category.BILLING: ("payment", "pay", "charge", "charged", "refund",
                       "subscription", "invoice", "billing", "card"),
    Category.ACCOUNT: ("login", "log in", "sign in", "password", "account",
                       "locked", "2fa", "access"),
    Category.CONTENT: ("subtitle", "subtitles", "caption", "metadata",
                       "episode", "title", "missing", "wrong"),
}

# Words that signal high urgency. Presence of a CRITICAL word forces critical
# severity; that will later force escalation (Guardrail #3), never auto-handling.
_CRITICAL_KEYWORDS: tuple[str, ...] = (
    "500", "crash", "crashes", "outage", "down for everyone", "breach",
    "data loss", "cannot pay", "all users", "everyone",
)
_HIGH_KEYWORDS: tuple[str, ...] = ("urgent", "asap", "broken", "error", "immediately")


def _count_hits(text: str, keywords: tuple[str, ...]) -> int:
    """Count how many of `keywords` appear in the (lower-cased) text."""
    return sum(1 for kw in keywords if kw in text)


def mock_classify(ticket_text: str) -> Classification:
    """Deterministic, offline classifier used when no API key is present.

    The logic is intentionally simple and reproducible so the graph can run and
    be tested without any external dependency:

    - category  : the category with the most keyword hits (ties broken by table
                  order); OTHER if nothing matches.
    - severity  : CRITICAL if any critical keyword is present, else HIGH for high
                  keywords, else MEDIUM for a matched category, else LOW.
    - confidence: high (0.85) when we matched a category, low (0.30) when we fell
                  back to OTHER — low confidence deliberately routes vague tickets
                  to human review later.
    """
    text = ticket_text.lower()

    # 1) Pick the best-scoring category.
    scores = {cat: _count_hits(text, kws) for cat, kws in _CATEGORY_KEYWORDS.items()}
    best_category, best_score = max(scores.items(), key=lambda kv: kv[1])

    if best_score == 0:
        # Nothing matched: unknown ticket -> low confidence -> human review path.
        return Classification(
            category=Category.OTHER,
            severity=Severity.LOW,
            confidence=0.30,
            rationale="No known keywords matched; needs human triage.",
        )

    # 2) Derive severity from urgency signals.
    if _count_hits(text, _CRITICAL_KEYWORDS) > 0:
        severity = Severity.CRITICAL
    elif _count_hits(text, _HIGH_KEYWORDS) > 0:
        severity = Severity.HIGH
    else:
        severity = Severity.MEDIUM

    return Classification(
        category=best_category,
        severity=severity,
        confidence=0.85,
        rationale=f"Matched {best_score} '{best_category}' keyword(s); severity from urgency signals.",
    )


def real_classify(ticket_text: str) -> Classification:
    """Anthropic-backed classifier — implemented in a later step.

    Will prompt claude-haiku-4-5 for JSON, parse it, and run it through
    `safe_parse` (Guardrail #1) so malformed model output is downgraded rather
    than trusted.
    """
    raise NotImplementedError(
        "real_classify is stubbed; it will be implemented in the real-mode step."
    )


def active_mode() -> str:
    """Return the mode that get_classifier() would select ('real' or 'mock')."""
    return "real" if os.getenv("ANTHROPIC_API_KEY") else "mock"


def get_classifier() -> Classifier:
    """Auto-select the classifier based on the environment.

    Depends only on the ANTHROPIC_API_KEY env var so callers (the graph node)
    stay decoupled from the concrete implementation.
    """
    return real_classify if active_mode() == "real" else mock_classify
