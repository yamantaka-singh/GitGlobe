"""Embed a search query into the corpus's own vector space.

The bug this exists to prevent: the corpus and the query must be embedded by the
same model, at the same width, in the same *sub-space*. Get any of the three
wrong and the failure is not an error — it is plausible-looking results that are
quietly meaningless, or a dimension rejection that gets swallowed and degrades
to substring matching. Both have happened in this codebase.

Matching the corpus means matching on four axes, all four of them below:

    model         llama-nemotron-embed-1b-v2   a different model is a different space
    dimensions    2048                         the collection's declared size
    normalisation L2                           cosine is defined on unit vectors
    input type    query                        see below

**Input type is the subtle one.** This model is asymmetric: it embeds documents
and queries into deliberately different regions. The same sentence encoded as
`passage` and as `query` has cosine 0.67 — they are genuinely different vectors.
`pipeline/src/gitglobe/embed/nvidia.py` wrote the corpus as `passage`, so this
must send `query`. Sending `passage` here would still return a 2048-d vector and
still pass every width check, and search would just be subtly worse forever.

This replaced a Vertex `gemini-embedding-001` implementation. That path is gone
rather than kept behind a flag: the GCP billing account is closed, so it could
only fail, and a second embedder that cannot run is a thing to maintain and get
confused by. The globe's *layout* still comes from those gemini vectors in
`gitglobe_repos` — untouched, and not what this module talks to. Retrieval has
its own collection, embedded for retrieval, which is what the spec wanted.

`assert_matches_collection()` is the guard against this module and the pipeline
drifting: it checks the query width against what Qdrant actually holds, which is
the only place a mismatch can do harm.
"""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

log = logging.getLogger(__name__)

NIM_URL = "https://integrate.api.nvidia.com/v1/embeddings"

MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DIMENSIONS = 2048

#: `query`, not `passage`. The corpus went in as `passage`; see the module
#: docstring — this is the axis that fails silently rather than loudly.
INPUT_TYPE = "query"

#: The retrieval collection. `gitglobe_repos` holds the gemini vectors the globe
#: is laid out from and must not be searched with these.
COLLECTION = "gitglobe_nv"

#: A search query is never long; the clamp stops a pasted README from becoming
#: one enormous request.
MAX_QUERY_CHARS = 2_000


class EmbeddingUnavailable(RuntimeError):
    """The embedding service could not be reached, or is not configured.

    Raised rather than returned as None so the caller has to decide what to do.
    The whole point of this module is that a silently-empty dense arm is
    indistinguishable from a corpus with no match.
    """


def l2_normalise(vector: np.ndarray) -> np.ndarray:
    """Unit-length, safe on a zero vector.

    A zero row would divide by zero and poison the vector with NaN, which Qdrant
    reports as an opaque failure rather than as bad input.
    """
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


class QueryEmbedder:
    """Async client for one query at a time."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "").strip()
        self._client = None

    async def start(self) -> None:
        import httpx

        if not self.api_key:
            raise EmbeddingUnavailable(
                "No NVIDIA_API_KEY. Get one free at https://build.nvidia.com"
            )
        self._client = httpx.AsyncClient(timeout=20.0)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def embed(self, query: str) -> list[float]:
        """One query to one unit vector in the corpus's space."""
        if not self._client:
            raise EmbeddingUnavailable("QueryEmbedder.start() was never called")

        text = (query or "").strip()[:MAX_QUERY_CHARS]
        if not text:
            raise EmbeddingUnavailable("empty query")

        body = {
            "input": [text],
            "model": MODEL,
            "input_type": INPUT_TYPE,
            # Mandatory: without it the API 400s on over-long input rather than
            # truncating. A query is short, but the clamp above is characters
            # and this is tokens, so the guard stays.
            "truncate": "END",
        }
        try:
            response = await self._client.post(
                NIM_URL,
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        except Exception as exc:
            raise EmbeddingUnavailable(f"NIM unreachable: {exc}") from exc

        if response.status_code != 200:
            raise EmbeddingUnavailable(
                f"NIM returned {response.status_code}: {response.text[:200]}"
            )
        return vector_from(response.json())


def vector_from(payload: dict) -> list[float]:
    """Pull the vector out of a 200 body and enforce the corpus contract.

    A 200 does not guarantee the shape, and a wrong-width vector is precisely
    the failure that used to degrade to substring matching without a word in the
    logs. Checking the width here turns it into a message that names the problem.
    """
    try:
        values = payload["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EmbeddingUnavailable(f"unexpected NIM response shape: {exc}") from exc

    vector = np.asarray(values, dtype=np.float32)
    if vector.shape[0] != DIMENSIONS:
        raise EmbeddingUnavailable(
            f"query vector is {vector.shape[0]}-d, corpus is {DIMENSIONS}-d"
        )
    return l2_normalise(vector).tolist()


async def assert_matches_collection(qdrant, collection: str = COLLECTION) -> None:
    """Warn loudly if the collection is not the width this module produces.

    The pipeline owns the corpus contract and this module restates it, so the
    two can drift. Qdrant is the arbiter — it holds what was actually written —
    and this is the only place where a disagreement causes harm.
    """
    try:
        info = await qdrant.get_collection(collection)
        size = info.config.params.vectors.size
    except Exception as exc:
        log.warning("Could not verify collection width for %s: %s", collection, exc)
        return
    if size != DIMENSIONS:
        log.error(
            "VECTOR WIDTH MISMATCH: %s holds %d-d vectors, queries are %d-d. "
            "Dense search cannot work until these agree.",
            collection, size, DIMENSIONS,
        )
