"""Vertex AI text embeddings.

Three decisions here shape everything downstream.

**One text per request.** `gemini-embedding-001` takes a single instance per
call, so 100,000 repositories is 100,000 HTTP round-trips, not 1,000 batches of
100. Throughput is therefore entirely a function of concurrency, and the client
is built around that: a bounded worker pool, per-request retry, and a write-back
every `CHECKPOINT_EVERY` rows so an interrupted run keeps what it paid for.
Above a few hundred thousand rows, use Vertex **batch prediction** instead —
it is asynchronous, roughly half the price, and immune to per-minute quota.
`estimate()` prints the crossover.

**768 dimensions, not 3072.** The model is Matryoshka-trained, so a 768-prefix
is a real embedding rather than a lossy compression — Google measures 0.26%
quality loss for a quarter of the storage. That matters because UMAP holds the
whole matrix in RAM: 1M x 3072 float32 is 12 GB and will not fit on a laptop,
while 1M x 768 is 3 GB and will.

**Always L2-normalise.** `gemini-embedding-001` returns truncated vectors
*un-normalised* — only `gemini-embedding-2` normalises them for you. Skip this
and every cosine distance in UMAP is quietly computed against vectors of
differing length, which does not raise, does not look wrong, and produces a map
whose clusters are partly an artifact of text length. Normalising is a no-op on
an already-normalised vector, so doing it unconditionally is free insurance and
keeps the code correct across both models.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIM = 768
DEFAULT_LOCATION = "us-central1"

#: Model limit is 2,048 tokens. READMEs are prose and code, so ~3.5 chars per
#: token is a safe floor; `autoTruncate` is the backstop, not the plan.
MAX_INPUT_CHARS = 6_500

#: What the embedding is FOR. `CLUSTERING` optimises for grouping similar texts
#: together, which is exactly what the globe is — a map where distance means
#: relatedness. `RETRIEVAL_DOCUMENT` would optimise for matching short queries
#: against long documents, a different and asymmetric objective.
TASK_TYPE = "CLUSTERING"

DEFAULT_CONCURRENCY = 32
CHECKPOINT_EVERY = 2_000

#: Published rate for gemini-embedding-001, USD per million input tokens.
#: Only used to print an estimate before spending money; verify against
#: https://cloud.google.com/vertex-ai/generative-ai/pricing before a large run.
USD_PER_MILLION_TOKENS = 0.15


@dataclass
class EmbedConfig:
    project: str
    location: str = DEFAULT_LOCATION
    model: str = DEFAULT_MODEL
    dimensions: int = DEFAULT_DIM
    concurrency: int = DEFAULT_CONCURRENCY
    max_retries: int = 5
    timeout_s: float = 60.0

    @property
    def endpoint(self) -> str:
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project}"
            f"/locations/{self.location}/publishers/google/models/{self.model}:predict"
        )


@dataclass
class EmbedStats:
    requested: int = 0
    succeeded: int = 0
    failed: int = 0
    retries: int = 0
    truncated: int = 0
    billable_tokens: int = 0
    failures: dict = field(default_factory=dict)

    def summary(self) -> str:
        cost = self.billable_tokens / 1e6 * USD_PER_MILLION_TOKENS
        return (
            f"{self.succeeded}/{self.requested} embedded, {self.failed} failed, "
            f"{self.retries} retries, {self.truncated} truncated, "
            f"~{self.billable_tokens:,} tokens (~${cost:.2f})"
        )


def prepare_text(text: str) -> tuple[str, bool]:
    """Clamp to the model's input window. Returns (text, was_truncated).

    Cutting at a word boundary rather than mid-token avoids feeding the model a
    fragment, and the last few hundred characters of a long README are almost
    always reference material the cleaner did not catch.
    """
    text = (text or "").strip()
    if len(text) <= MAX_INPUT_CHARS:
        return text, False
    cut = text[:MAX_INPUT_CHARS]
    space = cut.rfind(" ")
    return (cut[:space] if space > MAX_INPUT_CHARS * 0.9 else cut), True


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, safe on zero rows.

    A zero row would divide by zero and poison the whole matrix with NaN, which
    UMAP reports as an opaque failure thousands of rows later.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def estimate(n_rows: int, mean_chars: float) -> dict:
    """Cost and wall-clock estimate. Print this before spending credits."""
    tokens = n_rows * mean_chars / 3.5
    online_hours = n_rows / (DEFAULT_CONCURRENCY * 8) / 3600  # ~8 req/s per worker
    return {
        "rows": n_rows,
        "est_tokens": int(tokens),
        "est_usd_online": round(tokens / 1e6 * USD_PER_MILLION_TOKENS, 2),
        "est_hours_online": round(online_hours, 2),
        "recommend_batch_prediction": n_rows > 300_000,
    }


class VertexEmbedder:
    """Async embedding client over the Vertex predict endpoint.

    Talks HTTP directly rather than through `google-cloud-aiplatform`. The SDK
    pulls in a large dependency tree to wrap one POST, and its synchronous
    client would serialise the very requests that need to run concurrently.
    """

    def __init__(self, config: EmbedConfig):
        self.config = config
        self.stats = EmbedStats()
        self._client = None
        self._credentials = None

    async def __aenter__(self) -> "VertexEmbedder":
        import os
        import httpx
        from google.oauth2.credentials import Credentials
        from google.auth import default as google_auth_default

        token = os.environ.get("GOOGLE_ACCESS_TOKEN")
        if token:
            self._credentials = Credentials(token=token)
            project = os.environ.get("GCP_PROJECT", "gitglobe")
        else:
            self._credentials, project = google_auth_default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        if not self.config.project:
            self.config.project = project or ""
        if not self.config.project:
            raise RuntimeError(
                "No GCP project. Set GCP_PROJECT, or run `gcloud config set project <id>`."
            )
        self._client = httpx.AsyncClient(timeout=self.config.timeout_s)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()

    def _token(self) -> str:
        from google.auth.transport.requests import Request

        # Access tokens last an hour; a 100k run outlives several. Refreshing on
        # demand is what keeps hour two from failing with 401s.
        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return self._credentials.token

    async def embed_one(self, text: str) -> np.ndarray | None:
        """One text to one unit vector. None if it failed after every retry."""
        import httpx

        prepared, truncated = prepare_text(text)
        if not prepared:
            return None
        if truncated:
            self.stats.truncated += 1

        body = {
            "instances": [{"content": prepared, "task_type": TASK_TYPE}],
            "parameters": {
                "outputDimensionality": self.config.dimensions,
                "autoTruncate": True,
            },
        }

        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.post(
                    self.config.endpoint,
                    json=body,
                    headers={"Authorization": f"Bearer {self._token()}"},
                )
                if response.status_code == 200:
                    payload = response.json()["predictions"][0]["embeddings"]
                    self.stats.billable_tokens += int(
                        payload.get("statistics", {}).get("token_count", 0)
                    )
                    return np.asarray(payload["values"], dtype=np.float32)

                # 4xx other than 429 will fail identically forever — retrying
                # a malformed request just wastes the quota the good rows need.
                if response.status_code != 429 and 400 <= response.status_code < 500:
                    self._record_failure(f"http_{response.status_code}")
                    log.warning("Permanent %s: %s", response.status_code, response.text[:200])
                    return None
                reason = f"http_{response.status_code}"
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                reason = type(exc).__name__

            self.stats.retries += 1
            # Full jitter would allow a near-zero sleep, which under a 429 means
            # hammering a service that just asked you to stop.
            delay = min(2**attempt, 32) * (0.5 + 0.5 * random.random())
            log.debug("Retry %d after %s (sleeping %.1fs)", attempt + 1, reason, delay)
            await asyncio.sleep(delay)

        self._record_failure("exhausted_retries")
        return None

    def _record_failure(self, reason: str) -> None:
        self.stats.failed += 1
        self.stats.failures[reason] = self.stats.failures.get(reason, 0) + 1

    async def embed_many(
        self,
        items: list[tuple[int, str]],
        *,
        on_batch=None,
    ) -> dict[int, np.ndarray]:
        """Embed `(repo_id, text)` pairs concurrently.

        `on_batch(dict)` is awaited every CHECKPOINT_EVERY successes so the
        caller can persist. Holding a million vectors in memory until the end
        means a crash at 90% costs the whole run.
        """
        semaphore = asyncio.Semaphore(self.config.concurrency)
        results: dict[int, np.ndarray] = {}
        pending: dict[int, np.ndarray] = {}
        lock = asyncio.Lock()

        async def worker(repo_id: int, text: str) -> None:
            async with semaphore:
                vector = await self.embed_one(text)
            if vector is None:
                return
            async with lock:
                self.stats.succeeded += 1
                results[repo_id] = vector
                pending[repo_id] = vector
                if on_batch and len(pending) >= CHECKPOINT_EVERY:
                    flush, pending_ref = dict(pending), pending
                    pending_ref.clear()
                    await on_batch(flush)

        self.stats.requested += len(items)
        await asyncio.gather(*(worker(rid, txt) for rid, txt in items))

        if on_batch and pending:
            await on_batch(dict(pending))
        return results


def pack(vector: np.ndarray) -> bytes:
    """Unit-normalised float32 bytes, as stored in `repo.embedding`."""
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v = v / norm
    return v.astype(np.float32).tobytes()


def unpack(blob: bytes, dim: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    if len(vector) != dim:
        raise ValueError(f"embedding is {len(vector)} floats, expected {dim}")
    return vector


def unpack_matrix(blobs: list[bytes], dim: int) -> np.ndarray:
    """Many blobs to one (n, dim) matrix, in one allocation.

    `np.vstack` on a million small arrays spends most of its time in Python.
    Joining the bytes first and reshaping once is a single memcpy.
    """
    if not blobs:
        return np.zeros((0, dim), dtype=np.float32)
    expected = dim * 4
    for i, blob in enumerate(blobs):
        if len(blob) != expected:
            raise ValueError(f"row {i}: {len(blob)} bytes, expected {expected} for dim {dim}")
    return np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(blobs), dim)
