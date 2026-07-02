"""Graph — the orchestration layer (StateGraph, typed state, nodes, edges).

Topology after this step:

    START -> classify -> (router) -> { auto_handle | escalate | human_review } -> END

The router is where two guardrails live:
- Guardrail #2 (confidence): confidence < THRESHOLD -> human_review.
- Guardrail #3 (business rule): a `critical` ticket is never auto-handled -> escalate.

Each destination node records a structured audit entry and sets the final `action`.
`human_review` is a real human-in-the-loop pause: it calls `interrupt()` to suspend
the graph and is resumed with `Command(resume=<decision>)`. This requires compiling
with a checkpointer (MemorySaver), and every invoke must pass a `thread_id`.
"""

from operator import add
from typing import Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.classifier import active_mode, get_classifier
from src.schema import Action, AuditEntry, Classification, Severity

# Guardrail #2 threshold: below this confidence, defer to a human.
CONFIDENCE_THRESHOLD = 0.6


class TriageState(BaseModel):
    """Shared state passed between nodes, as a validated Pydantic model.

    Channels:
    - ticket_text    : the raw support ticket (provided at invoke time).
    - classification : the validated triage decision (set by classify_node).
    - action         : the final routing outcome (set by the destination node).
    - audit_log      : the audit trail. `Annotated[..., add]` attaches an additive
                       reducer, so writes *append* (each node contributes without
                       clobbering the others).
    """

    ticket_text: str
    classification: Classification | None = None
    action: Action | None = None
    audit_log: Annotated[list[AuditEntry], add] = Field(default_factory=list)


def classify_node(state: TriageState) -> dict:
    """Node: classify the ticket and record one structured audit entry."""
    classifier = get_classifier()
    result = classifier(state.ticket_text)
    entry = AuditEntry(
        node="classify",
        message=(
            f"mode={active_mode()} -> {result.category}/{result.severity} "
            f"confidence={result.confidence:.2f}"
        ),
    )
    return {"classification": result, "audit_log": [entry]}


def route(state: TriageState) -> Action:
    """Router: decide the destination from the classification (Guardrails #2 & #3).

    This is a pure decision function — it reads state and returns *where to go*,
    it does not mutate anything. Its return value is matched against the path_map
    passed to `add_conditional_edges`.

    Precedence matters:
    1. Guardrail #3 first — `critical` always escalates, no matter how confident.
       (A hard business rule outranks the confidence check.)
    2. Guardrail #2 next — low confidence defers to a human.
    3. Otherwise the model is confident and the ticket is not critical: auto-handle.
    """
    classification = state.classification
    if classification.severity == Severity.CRITICAL:
        return Action.ESCALATE
    if classification.confidence < CONFIDENCE_THRESHOLD:
        return Action.HUMAN_REVIEW
    return Action.AUTO_HANDLE


def auto_handle_node(state: TriageState) -> dict:
    """Terminal node: the ticket is confidently, safely automatable."""
    entry = AuditEntry(
        node="auto_handle",
        message=f"Auto-handled: {state.classification.category} at "
                f"{state.classification.confidence:.2f} confidence.",
    )
    return {"action": Action.AUTO_HANDLE, "audit_log": [entry]}


def escalate_node(state: TriageState) -> dict:
    """Terminal node: hand off to a specialist (critical or otherwise not automatable)."""
    entry = AuditEntry(
        node="escalate",
        message=f"Escalated: severity={state.classification.severity} "
                f"(Guardrail #3 blocks auto-handling of critical tickets).",
    )
    return {"action": Action.ESCALATE, "audit_log": [entry]}


def human_review_node(state: TriageState) -> dict:
    """Terminal node: pause for a human because confidence is low (HITL).

    `interrupt(payload)` suspends the graph mid-node and surfaces `payload` to the
    caller. Execution stops here; the compiled graph must have a checkpointer so
    the paused state can be saved and later resumed on the same thread_id.

    When the caller resumes with `Command(resume=<value>)`, THIS SAME NODE re-runs
    from the top and `interrupt(...)` returns `<value>` (it does not raise). So we
    receive the human's decision as the return value and act on it.

    Contract with the human: resume with an Action string — "auto_handle" or
    "escalate". Anything unrecognized defaults to the safe choice, escalate.
    """
    classification = state.classification

    # Suspend and surface the proposed classification for a human to judge.
    human_decision = interrupt(
        {
            "reason": f"confidence {classification.confidence:.2f} < {CONFIDENCE_THRESHOLD}",
            "proposed_classification": classification.model_dump(mode="json"),
            "instructions": "Resume with an Action: 'auto_handle' or 'escalate'.",
        }
    )

    # Coerce the human's decision into a typed Action; escalate is the safe fallback.
    try:
        action = Action(human_decision)
    except ValueError:
        action = Action.ESCALATE

    entry = AuditEntry(
        node="human_review",
        message=(
            f"Human reviewed low-confidence ticket "
            f"({classification.confidence:.2f}) and chose: {action}."
        ),
    )
    return {"action": action, "audit_log": [entry]}


def build_graph() -> CompiledStateGraph:
    """Assemble and compile the graph with conditional routing.

        START
          |
          v
       classify
          |
       (route)          <- Guardrails #2 & #3 decide the branch
       /  |  \\
      v   v   v
 auto_   esca-  human_
 handle  late   review
      \\   |   /
       v  v  v
         END

    `add_conditional_edges(source, path, path_map)`:
    - `source`   = "classify": after it runs, evaluate the router.
    - `path`     = `route`: the decision function returning an Action.
    - `path_map` = maps each possible Action to the node name to go to. Here the
      Action values equal the node names, but the map makes the wiring explicit.
    """
    workflow = StateGraph(TriageState)

    workflow.add_node("classify", classify_node)
    workflow.add_node("auto_handle", auto_handle_node)
    workflow.add_node("escalate", escalate_node)
    workflow.add_node("human_review", human_review_node)

    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route,
        {
            Action.AUTO_HANDLE: "auto_handle",
            Action.ESCALATE: "escalate",
            Action.HUMAN_REVIEW: "human_review",
        },
    )
    # Every destination is terminal.
    workflow.add_edge("auto_handle", END)
    workflow.add_edge("escalate", END)
    workflow.add_edge("human_review", END)

    # A checkpointer is REQUIRED for interrupt()/resume: it persists the paused
    # state per thread_id so a resumed invoke can continue where it stopped.
    # MemorySaver keeps checkpoints in memory (fine for a demo; swap for a durable
    # backend in production).
    return workflow.compile(checkpointer=MemorySaver())
