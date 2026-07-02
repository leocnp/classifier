"""Demo runner — exercises the whole triage graph end to end.

Run it with:

    uv run python -m src.run

It:
1. Prints the active mode (mock vs real) and the graph topology (Mermaid).
2. Feeds three tickets that hit all three routes: escalate, auto_handle, and the
   human-in-the-loop pause.
3. For the HITL ticket, shows the payload surfaced to the human and *simulates* a
   human decision via `Command(resume=...)`.
4. Prints the final action and the audit trail for each ticket.
"""

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from src.classifier import active_mode
from src.graph import build_graph

# Each tuple: (thread_id, ticket_text, simulated_human_decision_if_interrupted).
# The decision is only used if the graph pauses for human review.
TICKETS: list[tuple[str, str, str]] = [
    ("ticket-1", "The player crashes with a 500 error for everyone", "escalate"),
    ("ticket-2", "I want a refund on my subscription payment", "auto_handle"),
    ("ticket-3", "Hi, something feels off with the thing", "auto_handle"),
]


def run_ticket(
    graph: CompiledStateGraph, thread_id: str, ticket_text: str, human_decision: str
) -> dict:
    """Run one ticket, transparently handling a possible human-in-the-loop pause.

    The checkpointer requires a thread_id; we use one per ticket so their paused
    states never collide. If the first invoke returns an `__interrupt__`, we show
    the surfaced payload and resume on the SAME thread_id with the simulated
    human decision.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"ticket_text": ticket_text}, config=config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("  ⏸  PAUSED for human review — payload surfaced to the human:")
        print(f"       reason: {payload['reason']}")
        print(f"       proposed: {payload['proposed_classification']}")
        print(f"  👤 Simulating human decision: resume with '{human_decision}'")
        # Resume: interrupt() inside the node now returns `human_decision`.
        result = graph.invoke(Command(resume=human_decision), config=config)

    return result


def main() -> None:
    graph = build_graph()

    print("=" * 70)
    print(f"Support-ticket triage agent — mode: {active_mode().upper()}")
    print("=" * 70)

    print("\nGraph topology (Mermaid):\n")
    print(graph.get_graph().draw_mermaid())

    for thread_id, ticket_text, human_decision in TICKETS:
        print("-" * 70)
        print(f"{thread_id}: {ticket_text!r}")
        result = run_ticket(graph, thread_id, ticket_text, human_decision)

        print(f"  ➡  final action: {result['action']}")
        print("  📋 audit trail:")
        for entry in result["audit_log"]:
            print(f"       [{entry.node}] {entry.message}")

    print("-" * 70)


if __name__ == "__main__":
    main()
