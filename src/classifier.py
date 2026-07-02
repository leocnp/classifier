"""Classifiers — turn raw ticket text into a validated Classification.

Two interchangeable implementations share one contract:
`Callable[[str], Classification]`.

- `mock_classify`  : deterministic keyword matching, no network, no API key.
                     Builds a Classification directly (still schema-validated by
                     construction).
- `real_classify`  : calls the Anthropic API (Haiku), asks for JSON, then routes
                     that untrusted JSON through `safe_parse` (Guardrail #1).

`get_classifier()` picks the implementation based on whether `ANTHROPIC_API_KEY`
is set, so the rest of the system depends only on the *contract*, never on how a
classification was produced.
"""

import json
import os
from collections.abc import Callable

import anthropic

from src.knowledge import KBDocument
from src.schema import Category, Classification, Severity, safe_parse

# Haiku 4.5: fast and cheap, well suited to high-volume triage.
_MODEL = "claude-haiku-4-5-20251001"

# A classifier maps ticket text (+ optional retrieved KB context) to a
# validated Classification.
Classifier = Callable[[str, list[KBDocument]], Classification]


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


def mock_classify(
    ticket_text: str, context: list[KBDocument] | None = None
) -> Classification:
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

    `context` is the retrieved KB material (from the retrieve node). The mock keeps
    its decision keyword-driven for reproducibility, but records the retrieved
    hints in the rationale so you can *see* the grounding flow through — the real
    classifier actually feeds this context into the prompt.
    """
    text = ticket_text.lower()

    # 1) Pick the best-scoring category.
    scores = {cat: _count_hits(text, kws) for cat, kws in _CATEGORY_KEYWORDS.items()}
    best_category, best_score = max(scores.items(), key=lambda kv: kv[1])

    if best_score == 0:
        # Nothing matched: unknown ticket -> low confidence -> human review path.
        category, severity, confidence = Category.OTHER, Severity.LOW, 0.30
        rationale = "No known keywords matched; needs human triage."
    else:
        # 2) Derive severity from urgency signals.
        if _count_hits(text, _CRITICAL_KEYWORDS) > 0:
            severity = Severity.CRITICAL
        elif _count_hits(text, _HIGH_KEYWORDS) > 0:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM
        category, confidence = best_category, 0.85
        rationale = (
            f"Matched {best_score} '{best_category}' keyword(s); "
            "severity from urgency signals."
        )

    # Surface any retrieved grounding in the rationale.
    if context:
        hints = "; ".join(doc.hint for doc in context)
        rationale = f"{rationale} [KB: {hints}]"

    return Classification(
        category=category,
        severity=severity,
        confidence=confidence,
        rationale=rationale,
    )


def _build_system_prompt() -> str:
    """Build the classifier instructions, injecting the allowlists from the enums.

    Deriving the allowed values from the Enums (rather than hard-coding them)
    keeps the prompt in sync with the schema: add a Category and the model is told
    about it automatically.
    """
    categories = ", ".join(c.value for c in Category)
    severities = ", ".join(s.value for s in Severity)
    return (
        "You are a support-ticket triage classifier for a video platform.\n"
        "Classify the user's ticket and respond with ONLY a JSON object — no prose, "
        "no markdown fences. The object must have exactly these keys:\n"
        f'  "category": one of [{categories}]\n'
        f'  "severity": one of [{severities}]\n'
        '  "confidence": a number between 0 and 1 (your certainty)\n'
        '  "rationale": a short string explaining the decision\n'
        'Use "critical" severity only for outages, data loss, or platform-wide '
        "failures."
    )


def _format_context(context: list[KBDocument] | None) -> str:
    """Render retrieved KB entries as a grounding block for the prompt (RAG)."""
    if not context:
        return ""
    lines = "\n".join(f"- {doc.hint}" for doc in context)
    return (
        "\n\nRelevant knowledge-base guidance (use it to ground your decision):\n"
        f"{lines}"
    )


def real_classify(
    ticket_text: str, context: list[KBDocument] | None = None
) -> Classification:
    """Anthropic-backed classifier (Haiku).

    Prompts the model for JSON, parses it, and routes the result through
    `safe_parse` (Guardrail #1) so malformed model output is downgraded rather
    than trusted. Any failure — network error, refusal, invalid JSON — is caught
    and turned into the same safe low-confidence fallback, so a real-mode problem
    routes a ticket to human review instead of crashing the graph.

    `context` (retrieved KB material) is appended to the system prompt so the
    model's decision is grounded — this is the "augmented" in RAG.
    """
    # anthropic.Anthropic() reads ANTHROPIC_API_KEY from the environment.
    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=512,
            system=_build_system_prompt() + _format_context(context),
            messages=[{"role": "user", "content": ticket_text}],
        )
        if response.stop_reason == "refusal":
            raise ValueError("model refused to classify the ticket")
        # The response is a list of content blocks; take the first text block.
        text = next(block.text for block in response.content if block.type == "text")
        data = json.loads(text)
    except Exception as exc:  # API error, refusal, or invalid JSON
        return Classification(
            category=Category.OTHER,
            severity=Severity.LOW,
            confidence=0.0,
            rationale=f"Real-mode fallback ({type(exc).__name__}); routed to human review.",
        )

    # Untrusted model JSON crosses Guardrail #1 here.
    return safe_parse(data)


def active_mode() -> str:
    """Return the mode that get_classifier() would select ('real' or 'mock')."""
    return "real" if os.getenv("ANTHROPIC_API_KEY") else "mock"


def get_classifier() -> Classifier:
    """Auto-select the classifier based on the environment.

    Depends only on the ANTHROPIC_API_KEY env var so callers (the graph node)
    stay decoupled from the concrete implementation.
    """
    return real_classify if active_mode() == "real" else mock_classify
