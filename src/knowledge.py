"""Knowledge base + retrieval — the "R" in RAG.

This is a deliberately tiny, in-memory, deterministic stand-in for a real retrieval
stack. The point of the bonus step is to show that **retrieval is just a graph
node**: given the ticket text, fetch relevant reference material and hand it to the
classifier so its decision is *grounded* rather than guessed.

Where a production system differs (all inside `retrieve`, same node in the graph):
- embed the query and the docs, do vector (semantic) search instead of keyword
  overlap — see the marker in `retrieve`;
- add a rerank stage (a cross-encoder) after the first-pass retrieval to reorder
  candidates by precision;
- source docs from a real store (KB articles, resolved tickets) with chunking.
"""

from pydantic import BaseModel


class KBDocument(BaseModel):
    """One reference entry the classifier can be grounded on.

    - `keywords`: what the entry is about (used here for naive matching; a real
      system would match on embeddings, not these).
    - `hint`: short guidance handed to the classifier as retrieved context.
    """

    id: str
    text: str
    keywords: tuple[str, ...]
    hint: str


# A handful of curated entries for a video-platform support desk.
KNOWLEDGE_BASE: list[KBDocument] = [
    KBDocument(
        id="kb-cdn-500",
        text="Player HTTP 500 errors are almost always CDN-side outages affecting "
        "many users at once.",
        keywords=("500", "player", "crash", "outage", "everyone"),
        hint="Player 500s usually indicate a platform-wide CDN incident -> critical, escalate.",
    ),
    KBDocument(
        id="kb-refund",
        text="Refunds requested within 14 days of purchase are processed "
        "automatically by the billing bot.",
        keywords=("refund", "subscription", "payment", "billing", "charge"),
        hint="Refund/billing requests within policy are safely auto-handleable.",
    ),
    KBDocument(
        id="kb-account-lock",
        text="Locked accounts are resolved by sending an automated password-reset link.",
        keywords=("login", "password", "account", "locked", "2fa", "sign in"),
        hint="Account lockouts are low severity; a self-serve reset link resolves them.",
    ),
    KBDocument(
        id="kb-subtitles",
        text="Missing or wrong subtitles are content-pipeline issues fixed by a "
        "re-ingest job.",
        keywords=("subtitle", "subtitles", "caption", "metadata"),
        hint="Subtitle/caption gaps are medium-severity content issues, auto-handleable.",
    ),
]


def retrieve(query: str, k: int = 2) -> list[KBDocument]:
    """Return the top-`k` knowledge-base entries relevant to `query`.

    Deterministic keyword-overlap scoring keeps the demo reproducible and offline.

    >>> Production swap-in (same signature, same node): embed `query`, run vector
    >>> similarity over embedded docs for recall, then rerank the candidates with a
    >>> cross-encoder for precision. The rest of the graph does not change.
    """
    q = query.lower()
    scored = [
        (doc, sum(1 for kw in doc.keywords if kw in q)) for doc in KNOWLEDGE_BASE
    ]
    # Keep only docs that matched at least one keyword, best score first.
    ranked = sorted(
        (pair for pair in scored if pair[1] > 0),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [doc for doc, _score in ranked[:k]]
