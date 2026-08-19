"""Embed the corpus with NVIDIA NIM, into a second Qdrant collection.

This exists to remove the last paid dependency from search. The globe's layout
comes from `gemini-embedding-001` vectors that are already computed, already
projected, and must not move — so nothing here touches `repo.embedding` or the
`gitglobe_repos` collection. Retrieval gets its own collection instead, which is
also the design the retrieval spec wanted all along: query-to-document matching
against vectors embedded *for retrieval* rather than for clustering.

Four measured facts shape this file.

**Batches of 128.** The endpoint accepts a list, and cost is per request: 128
texts in one call ran 21 ms/doc against 677 ms/doc one at a time. That is what
makes 180,000 rows a 35-minute job on a free tier instead of a two-day one.

**`truncate: "END"` is mandatory.** Without it the API returns HTTP 400 —
"Input length 4032 exceeds maximum allowed token size" — rather than truncating.
One long README in a batch of 128 would fail all 128.

**`input_type` is real, not decoration.** The same sentence encoded as `query`
and as `passage` has cosine 0.67, so the two are genuinely different spaces. The
corpus goes in as `passage`; the API must query with `query`. Getting this
backwards is the same class of bug as the 768-vs-1024 mismatch that made search
silently impossible before — plausible results, quietly wrong.

**The model was chosen on context window, not dimension.**
`nv-embedqa-e5-v5` is smaller and faster (1024-d, 13 ms/doc) and sees only **512
tokens** — a quarter of what gemini saw, which is roughly a README's first
paragraph. `llama-nemotron-embed-1b-v2` accepts 4,032+, double gemini's 2,048,
for 2048-d at 21 ms/doc. Storage is 369 MB at int8 for 180k rows, inside
Qdrant's free 1 GB, so the window is worth the extra bytes.

Qdrant is written over its REST API with `httpx` rather than through
`qdrant-client`. The pipeline does not depend on that package and this is the
only place it would be needed, so a dependency for four HTTP calls is not worth
it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

NIM_URL = "https://integrate.api.nvidia.com/v1/embeddings"

DEFAULT_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_DIM = 2048
COLLECTION = "gitglobe_nv"

#: Texts per request. Measured: 1 -> 677 ms/doc, 32 -> 31, 128 -> 21.
BATCH = 128

#: In-flight requests. The free tier allows 40 requests/minute, and one batch is
#: one request, so this is deliberately low — the limiter below is what actually
#: paces the run, and burst concurrency only buys 429s.
DEFAULT_CONCURRENCY = 4

#: Requests per minute the free tier allows. `teach` ignored this with its
#: default concurrency of 60 and collected 2,856 rejections, so it is enforced
#: here rather than discovered.
RPM = 40


@dataclass
class EmbedStats:
    requested: int = 0
    embedded: int = 0
    failed: int = 0
    upserted: int = 0
    failures: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.embedded:,}/{self.requested:,} embedded, "
            f"{self.upserted:,} upserted, {self.failed:,} failed"
            + (f" {self.failures}" if self.failures else "")
        )


class _Limiter:
    """Simple requests-per-minute gate.

    A token bucket would be more elegant and this is a background job whose only
    requirement is "do not exceed 40 in any minute". Spacing requests evenly is
    the boring version that cannot burst by construction.
    """

    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(rpm, 1)
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if delay:
            await asyncio.sleep(delay)


async def embed_batch(client, api_key: str, texts: list[str], *,
                      model: str, input_type: str, limiter: _Limiter) -> list[list[float]] | None:
    """One request, many texts. None if it failed."""
    await limiter.wait()
    try:
        response = await client.post(
            NIM_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "input": texts,
                "model": model,
                "input_type": input_type,
                # Without this the whole batch 400s on the first long README.
                "truncate": "END",
            },
        )
    except Exception as exc:
        log.warning("batch of %d: %s", len(texts), exc)
        return None

    if response.status_code != 200:
        log.warning("batch of %d: HTTP %s %s", len(texts), response.status_code,
                    response.text[:160])
        return None

    payload = response.json().get("data") or []
    if len(payload) != len(texts):
        # Order matters: results are matched to repo ids positionally, so a
        # short or reordered response would attach vectors to the wrong repos.
        log.warning("batch of %d returned %d vectors — discarding", len(texts), len(payload))
        return None
    ordered = sorted(payload, key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in ordered]


async def ensure_collection(client, qdrant_url: str, *, dim: int, collection: str) -> None:
    """Create the collection if absent. Never deletes — this is additive.

    `seed_qdrant.py` recreates its collection on every run, which is fine there
    and would be destructive here: the gemini vectors behind the globe live one
    collection over, and a corpus re-embed is not a reason to risk them.
    """
    existing = await client.get(f"{qdrant_url}/collections/{collection}")
    if existing.status_code == 200:
        return
    body = {
        "vectors": {"size": dim, "distance": "Cosine"},
        # Same int8 quantisation the existing collection uses: 2048 floats is
        # 8 KB per repo raw, 2 KB quantised, which is the difference between
        # fitting a free 1 GB cluster and not.
        "quantization_config": {
            "scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}
        },
        "on_disk_payload": True,
    }
    created = await client.put(f"{qdrant_url}/collections/{collection}", json=body)
    created.raise_for_status()
    for field_name, schema in (("language", "keyword"), ("domain", "integer"),
                               ("stars", "integer")):
        await client.put(
            f"{qdrant_url}/collections/{collection}/index",
            json={"field_name": field_name, "field_schema": schema},
        )
    # Full-text payload indexes so this collection can also serve the lexical
    # arm — that is what lets the read path drop Postgres entirely.
    for field_name in ("full_name", "description"):
        await client.put(
            f"{qdrant_url}/collections/{collection}/index",
            json={
                "field_name": field_name,
                "field_schema": {"type": "text", "tokenizer": "word",
                                 "lowercase": True, "min_token_len": 2},
            },
        )
    log.info("Created collection %s (%d-d, int8)", collection, dim)


async def existing_ids(client, qdrant_url: str, collection: str) -> set[int]:
    """Point ids already embedded, so a re-run resumes instead of repeating."""
    ids: set[int] = set()
    offset = None
    while True:
        body = {"limit": 10_000, "with_payload": False, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        response = await client.post(
            f"{qdrant_url}/collections/{collection}/points/scroll", json=body
        )
        if response.status_code != 200:
            return ids
        result = response.json().get("result") or {}
        ids.update(p["id"] for p in result.get("points", []))
        offset = result.get("next_page_offset")
        if offset is None:
            return ids


SELECT_SQL = """
    SELECT id, full_name, description, language, domain, stars, embedding_input
    FROM repo
    WHERE embedding_input IS NOT NULL AND length(embedding_input) > 0
    ORDER BY stars DESC
"""


async def run(db, api_key: str, *, qdrant_url: str, model: str = DEFAULT_MODEL,
              dim: int = DEFAULT_DIM, collection: str = COLLECTION,
              limit: int = 0, concurrency: int = DEFAULT_CONCURRENCY) -> EmbedStats:
    """Embed every repository with cleaned text, into `collection`.

    Ordered by stars so an interrupted run has still covered what matters most,
    and resumable because ids already present in Qdrant are skipped.
    """
    import httpx

    stats = EmbedStats()
    async with httpx.AsyncClient(timeout=120.0) as client:
        await ensure_collection(client, qdrant_url, dim=dim, collection=collection)
        done = await existing_ids(client, qdrant_url, collection)
        log.info("%s already holds %s vectors", collection, f"{len(done):,}")

        async with db.pool.acquire() as conn:
            rows = [dict(r) for r in await conn.fetch(SELECT_SQL)]
        rows = [r for r in rows if r["id"] not in done]
        if limit:
            rows = rows[:limit]
        if not rows:
            return stats

        stats.requested = len(rows)
        batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
        limiter = _Limiter(RPM)
        semaphore = asyncio.Semaphore(concurrency)
        log.info("%s rows in %s batches (~%.0f min at %d rpm)",
                 f"{len(rows):,}", f"{len(batches):,}", len(batches) / RPM, RPM)

        async def one(batch: list[dict]) -> None:
            # Contained: `gather` cancels every sibling on the first exception,
            # and vectors already upserted are the run's only durable progress.
            try:
                async with semaphore:
                    vectors = await embed_batch(
                        client, api_key, [r["embedding_input"] for r in batch],
                        model=model, input_type="passage", limiter=limiter,
                    )
                if vectors is None:
                    stats.failed += len(batch)
                    stats.failures["batch_failed"] = stats.failures.get("batch_failed", 0) + 1
                    return
                stats.embedded += len(vectors)
                points = [
                    {
                        "id": r["id"],
                        "vector": v,
                        # The payload carries everything /search and /repo read
                        # back, so this collection alone can serve the read path.
                        "payload": {
                            "full_name": r["full_name"],
                            "description": r["description"] or "",
                            "language": r["language"] or "",
                            "domain": r["domain"],
                            "stars": r["stars"] or 0,
                        },
                    }
                    for r, v in zip(batch, vectors)
                ]
                upserted = await client.put(
                    f"{qdrant_url}/collections/{collection}/points?wait=false",
                    json={"points": points},
                )
                if upserted.status_code >= 400:
                    stats.failed += len(batch)
                    log.warning("upsert failed: %s %s", upserted.status_code, upserted.text[:160])
                    return
                stats.upserted += len(points)
                if stats.upserted % (BATCH * 20) < BATCH:
                    log.info("progress: %s upserted", f"{stats.upserted:,}")
            except Exception as exc:
                stats.failed += len(batch)
                stats.failures[type(exc).__name__] = stats.failures.get(type(exc).__name__, 0) + 1
                log.warning("batch failed: %s", exc)

        await asyncio.gather(*(one(b) for b in batches))
    return stats
