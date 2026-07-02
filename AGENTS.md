# AGENTS.md

Guidance for any AI agent (and humans) working in this repo.

## What this project is

A **support-ticket triage agent** for a video platform, built as a reference
implementation of production-grade AI agent orchestration patterns with LangGraph:
typed state, conditional routing, layered guardrails, and human-in-the-loop. The
codebase is intentionally small and heavily commented so each pattern is easy to
read, reason about, and reuse.

The graph:

```
START → classify → conditional routing → {auto_handle | escalate | human_review} → END
```

### Concepts it must demonstrate

- **Chains / orchestration:** a `StateGraph` with a typed shared state, nodes, and
  conditional edges.
- **Guardrails at 3 levels:**
  1. Validate the LLM output against a Pydantic schema (allowlist of
     categories/severities, `confidence ∈ [0,1]`). A non-conforming output is
     **downgraded to low confidence**, never propagated as-is.
  2. Confidence threshold: `< 0.6 → human_review`.
  3. Business rule: a `critical` ticket is **never auto-handled** — always escalated.
- **Human-in-the-loop:** a `human_review` node using `interrupt()` to pause the
  graph (requires a checkpointer), surface the proposed classification, and resume
  via `Command(resume=<decision>)`.
- **Observability seed:** an audit field in the state (a `notes` list with an
  additive reducer) so each node records why it acted as it did.

### Two run modes (auto-selected on `ANTHROPIC_API_KEY`)

- **Mock mode** (no API key): a deterministic keyword-based classifier so the graph
  runs immediately and reproducibly.
- **Real mode:** classification via the Anthropic API using model
  `claude-haiku-4-5-20251001` (Haiku = fast/cheap, well suited to high-volume
  triage). Ask for JSON, parse it, then validate with the **same** Pydantic schema.

## Contribution conventions

Development follows a deliberate, incremental style. Follow these rules:

- **One change = one concept.** Land a single self-contained unit of work, then
  document *what* changed and *why* before moving on.
- **No big-bang commits.** Avoid introducing the whole system at once; build it up
  in small, reviewable increments.
- Agree on the approach before implementing; don't skip ahead to later work.
- All code comments and documentation in **English**.
- Prefer verifying against official documentation over assumptions.

## Development workflow

All changes go through pull requests — nothing is committed directly to `main`.

1. **Branch** off `main`: `git switch -c <type>/<short-description>`
   (e.g. `feat/human-review-node`, `chore/scaffold-project`).
2. **Commit** using [Conventional Commits](https://www.conventionalcommits.org):
   `<type>(<optional-scope>): <description>`. Common types: `feat`, `fix`,
   `chore`, `docs`, `refactor`, `test`. Keep commits small and single-purpose.
3. **Push** the branch to `origin` and **open a pull request into `main`**.
4. Merge via the PR once reviewed; delete the branch afterwards.

## Tech + gotchas

- Package manager: **uv** (see the uv quick-reference at the bottom of this file).
- Python: pinned via `.python-version` (currently 3.14.x). If a dependency won't
  resolve on 3.14, pin a supported interpreter with `uv python pin <version>`.
- LangGraph target: **1.2.x**. **Verify the actually-installed API before writing
  code** — imports and signatures change between versions. Expected API:
  - `from langgraph.graph import StateGraph, START, END`
  - `from langgraph.checkpoint.memory import MemorySaver`
  - `from langgraph.types import interrupt, Command`
  - conditional edges: `add_conditional_edges(source, path_fn, path_map)`
  - compile with `compile(checkpointer=MemorySaver())`
  - an interrupted `invoke` returns a state containing an `"__interrupt__"` key.
- Before writing any Anthropic API code, consult the current Claude API reference
  (model IDs / params change) rather than relying on memory.

## Layout & deliverables

Application code lives under `src/`. Modules are imported with the `src.` prefix
(e.g. `from src.schema import ...`); the repository root is already on `sys.path`
when running via uv from the project root, so no `PYTHONPATH` tweaking is needed:

```
uv run python -c "from src.classifier import mock_classify"
```

Planned modules, built incrementally: `src/schema.py`, `src/classifier.py`,
`src/graph.py`, `src/run.py` (+ `pyproject.toml` / lockfile). Bonus (later): a
`retrieve` node before `classify` = RAG.

## uv quick-reference

- `uv add <pkg>` / `uv remove <pkg>` — manage dependencies (updates
  `pyproject.toml` + `uv.lock` and installs into `.venv`).
- `uv run <cmd>` — run a command inside the project venv (auto-syncs first).
- `uv sync` — make `.venv` match the lockfile (e.g. after `git pull`).
- `uv lock` — re-resolve and rewrite `uv.lock`.

`uv.lock` is committed for reproducible installs; the `.venv` is not.
