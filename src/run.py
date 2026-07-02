"""Demo runner — exercises the whole triage graph end to end.

Run the three-ticket demo:

    uv run python -m src.run

Or classify your own ticket:

    uv run python -m src.run "the player buffers endlessly on 4K"

Mode (mock vs real) is auto-selected: real mode uses the Anthropic API when
ANTHROPIC_API_KEY is set (loaded from a local .env file), otherwise the
deterministic mock classifier runs.
"""

import sys

from dotenv import load_dotenv
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


def _print_result(result: dict) -> None:
    """Print the final action and audit trail of one graph run."""
    print(f"  ➡  final action: {result['action']}")
    print("  📋 audit trail:")
    for entry in result["audit_log"]:
        print(f"       [{entry.node}] {entry.message}")


def main() -> None:
    # Load ANTHROPIC_API_KEY (and anything else) from a local .env before we read
    # the environment to decide mock vs real mode.
    load_dotenv()

    graph = build_graph()

    print("=" * 70)
    print(f"Support-ticket triage agent — mode: {active_mode().upper()}")
    print("=" * 70)

    # A ticket passed on the command line runs just that one; otherwise the demo.
    custom_ticket = " ".join(sys.argv[1:]).strip()
    if custom_ticket:
        # For a custom ticket we auto-approve if it pauses for human review, so a
        # single command runs end to end. Change "auto_handle" to "escalate" to
        # simulate the other human decision.
        print("-" * 70)
        print(f"custom ticket: {custom_ticket!r}")
        result = run_ticket(graph, "custom", custom_ticket, "auto_handle")
        _print_result(result)
        print("-" * 70)
        return

    print("\nGraph topology (Mermaid):\n")
    print(graph.get_graph().draw_mermaid())

    for thread_id, ticket_text, human_decision in TICKETS:
        print("-" * 70)
        print(f"{thread_id}: {ticket_text!r}")
        result = run_ticket(graph, thread_id, ticket_text, human_decision)
        _print_result(result)

    print("-" * 70)


if __name__ == "__main__":
    main()
