"""Graph — the orchestration layer (StateGraph, typed state, nodes, edges).

This step builds the *minimal* graph: START -> classify -> END. It introduces the
three orchestration primitives everything else builds on:

1. A typed shared **state** (`TriageState`, a Pydantic model) that flows through
   every node — attribute access (`state.classification`) and validation included.
2. A **node** (`classify_node`) — a function `state -> update`.
3. **Edges** wiring the node between the START and END sentinels.

The `audit_log` field carries the observability trail. It uses an *additive
reducer* (`operator.add`), so when a node returns `{"audit_log": [...]}`, LangGraph
appends to the existing list instead of overwriting it.
"""

from operator import add
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.classifier import active_mode, get_classifier
from src.schema import AuditEntry, Classification


class TriageState(BaseModel):
    """Shared state passed between nodes, as a validated Pydantic model.

    Channels:
    - ticket_text    : the raw support ticket (provided at invoke time).
    - classification : the validated triage decision (set by classify_node).
    - audit_log      : the audit trail. `Annotated[..., add]` attaches an additive
                       reducer, so writes *append* (each node contributes without
                       clobbering the others). `default_factory=list` gives every
                       run a fresh empty log to append onto.

    Defaults on `classification`/`audit_log` let a node update just the channels it
    touches; LangGraph constructs/merges the model for us.
    """

    ticket_text: str
    classification: Classification | None = None
    audit_log: Annotated[list[AuditEntry], add] = Field(default_factory=list)


def classify_node(state: TriageState) -> dict:
    """Node: classify the ticket and record one structured audit entry.

    Returns a *partial update* — a dict naming only the channels it changes. This
    is LangGraph's contract: the graph merges each key into the state through that
    channel's reducer (`classification` overwrites; `audit_log` appends). Reads use
    attribute access thanks to the Pydantic state model.
    """
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


def build_graph():
    """Assemble and compile the minimal graph: START -> classify -> END.

    - `StateGraph(TriageState)` binds the graph to our Pydantic state schema.
    - `add_node` registers the function under a name.
    - `add_edge(START, "classify")` sets the entry point; `add_edge("classify",
      END)` sets the exit.
    - `compile()` validates the topology and returns a runnable graph.
    """
    builder = StateGraph(TriageState)
    builder.add_node("classify", classify_node)
    builder.add_edge(START, "classify")
    builder.add_edge("classify", END)
    return builder.compile()
