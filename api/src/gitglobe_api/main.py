import os
import json
import hashlib
import math
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# ----------------- Configuration -----------------
from dotenv import load_dotenv
load_dotenv()

# Uvicorn's `--log-level` only configures *its own* loggers, so without this the
# root logger stays at WARNING and this module's INFO lines vanish — including
# "Dense search enabled", which is the one line that says whether search is
# actually working. Being unable to see that cost real debugging time.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s:     %(name)s - %(message)s",
)

from fastapi import FastAPI, Query, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from redis.asyncio import Redis

from .embed import (
    COLLECTION,
    INPUT_TYPE,
    MODEL,
    EmbeddingUnavailable,
    QueryEmbedder,
    assert_matches_collection,
)
from .summarize import MODEL as SUMMARY_MODEL, SummaryGenerator, SummaryUnavailable

log = logging.getLogger(__name__)

ENV = os.getenv("ENV", "development")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    if ENV == "production":
        raise ValueError("DATABASE_URL is required in production")
    DATABASE_URL = "postgresql://gitglobe:gitglobe@localhost:5433/gitglobe"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# Trailing slashes and spaces stripped: the browser's `Origin` header never has
# a trailing slash, so "https://app.vercel.app/" silently matches nothing and
# every request fails CORS while the server logs look completely healthy.
CORS_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# ----------------- State -----------------
class AppState:
    db_pool: asyncpg.Pool = None
    qdrant: AsyncQdrantClient = None
    redis: Redis = None
    embedder: Optional[QueryEmbedder] = None
    summarizer: Optional[SummaryGenerator] = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    state.db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    if QDRANT_API_KEY:
        state.qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        state.qdrant = AsyncQdrantClient(url=QDRANT_URL)
    state.redis = Redis.from_url(REDIS_URL, decode_responses=True)

    # The dense arm is optional, and the API must serve lexical-only without it
    # — but it says so at startup rather than letting every search degrade in
    # silence. `assert_matches_collection` is the guard against the query and
    # the corpus drifting to different widths, which is the original bug.
    embedder = QueryEmbedder()
    try:
        await embedder.start()
        await assert_matches_collection(state.qdrant, COLLECTION)
        state.embedder = embedder
        log.info("Dense search enabled: %s / %s / %s", MODEL, INPUT_TYPE, COLLECTION)
    except EmbeddingUnavailable as e:
        log.error("DENSE SEARCH DISABLED — search is lexical-only. Reason: %s", e)
        await embedder.close()

    # Same key, same optionality as the dense arm: no NVIDIA_API_KEY means no
    # on-demand summaries, and `get_repo` falls back to the GitHub description
    # exactly as it already does for a repo the batch teacher never reached.
    summarizer = SummaryGenerator()
    try:
        await summarizer.start()
        state.summarizer = summarizer
        log.info("On-demand summaries enabled: %s", SUMMARY_MODEL)
    except SummaryUnavailable as e:
        log.info("On-demand summaries disabled: %s", e)
        await summarizer.close()

    yield

    # Shutdown
    if state.embedder:
        await state.embedder.close()
    if state.summarizer:
        await state.summarizer.close()
    await state.db_pool.close()
    await state.qdrant.close()
    await state.redis.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def cache_control(request, call_next):
    """Say what is cacheable. Every response used to leave it unstated.

    A response with no `Cache-Control` has no explicit freshness lifetime, and
    RFC 9111 lets a browser or any intermediate proxy apply its own heuristic —
    so a reply could be reused for an unspecified period with nothing here
    asking for it. The failure that produces is nasty: once a client has stored
    an error or an empty result, it keeps serving it, the page looks broken,
    and only a private window (a fresh cache) appears to fix it.

    Successful reads are cheap and change rarely, so they get a short shared
    lifetime with `stale-while-revalidate` to keep navigation snappy. Anything
    that is not a success gets `no-store`, so a bad minute can never be
    remembered as a bad hour.
    """
    response = await call_next(request)
    if "cache-control" not in response.headers:
        if 200 <= response.status_code < 300:
            response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=300"
        else:
            response.headers["Cache-Control"] = "no-store"
    return response

# ----------------- Models -----------------
class RepoMetadata(BaseModel):
    id: int
    full_name: str
    description: Optional[str]
    language: Optional[str]
    domain: Optional[int]
    stars: int
    #: One sentence saying what the software does, written by the teacher from
    #: the cleaned README. Nullable is load-bearing: it is how the client tells
    #: "no summary exists" from "the summary is empty", and it drives the
    #: fallback to GitHub's own description. Never a placeholder string.
    summary: Optional[str] = None
    #: 0-100. The student's prediction (source=1) covers the whole corpus; a
    #: teacher row (source=0) overrides it where one exists. Never from stars —
    #: `assert_no_popularity_features` keeps the student blind to them, so this
    #: is a read of the README and nothing else.
    onboarding_ease: Optional[float] = None
    learning_value: Optional[float] = None
    license: Optional[str] = None
    pushed_at: Optional[datetime] = None
    is_archived: bool = False

class SearchResult(BaseModel):
    repo: RepoMetadata
    score: float

class GraphEdge(BaseModel):
    source: str
    target: str

class GraphResponse(BaseModel):
    nodes: List[str]
    edges: List[GraphEdge]

# ----------------- Helpers -----------------

async def get_query_embedding(query: str) -> List[float]:
    """Query vector in the corpus's own space. Raises EmbeddingUnavailable.

    Cached for a week because the same handful of queries dominate traffic and
    an embedding of a fixed string never changes.
    """
    if not state.embedder:
        raise EmbeddingUnavailable("query embedder not configured")

    cache_key = f"embed:{MODEL}:{INPUT_TYPE}:{hashlib.sha256(query.encode()).hexdigest()}"
    cached = await state.redis.get(cache_key)
    if cached:
        return json.loads(cached)

    embedding = await state.embedder.embed(query)
    await state.redis.setex(cache_key, 7 * 24 * 3600, json.dumps(embedding))
    return embedding


#: OR rather than AND, which is worth stating because AND is the obvious choice
#: and it measured worse. `websearch_to_tsquery` ANDs its terms, so
#: "lightweight c++ web servers" demands all four and 7 of 30 eval queries
#: returned nothing. OR-ing and letting `ts_rank` sort scored 0.229 against
#: 0.169, and never returned an empty page. A repo matching three terms still
#: outranks one matching a single term, so breadth costs nothing at the top.
#: The slash has to go before `to_tsvector` sees it. Postgres classifies
#: `ohmyzsh/ohmyzsh` as a single `file` token — the lexeme is the whole
#: `'ohmyzsh/ohmyzsh'`, not the word `ohmyzsh` — so searching a repository by
#: its own name matched nothing at all. Replacing `/` with a space splits owner
#: and repo into ordinary words. Substring matching used to cover this by
#: accident, so it is a regression the eval set cannot see: all 30 of its
#: queries are descriptive, and none looks a repository up by name.
#:
#: This expression must stay character-for-character identical to the one in
#: `006_search_fts.sql`, or the GIN index stops matching and every search
#: becomes a sequential scan of 87k rows.
#: Unweighted on purpose, and that was measured rather than assumed.
#:
#: Weighting the name `A` and the description `B` is the textbook move and it
#: made things worse: descriptive recall@10 fell from 0.221 to 0.086, because
#: for "time series database" it promotes anything *named* `*-database` over the
#: repositories actually described as one. It bought exactly one extra name
#: lookup out of six. Name lookup is handled by `NAME_SQL` below instead, which
#: costs the descriptive arm nothing.
#:
#: Written once and interpolated into both the WHERE and the ORDER BY: Postgres
#: matches the GIN index by comparing the *expression*, so both copies and the
#: one in `006_search_fts.sql` must stay identical or the index is silently
#: ignored and every search scans 87k rows.
TSVECTOR_EXPR = (
    "to_tsvector('english', replace(coalesce(r.full_name,''), '/', ' ') "
    "|| ' ' || coalesce(r.description,''))"
)

LEXICAL_SQL = f"""
    SELECT r.id, r.full_name, r.description, r.language, r.domain, r.stars
    FROM repo r, to_tsquery('english', $1) AS q
    WHERE {TSVECTOR_EXPR} @@ q
    ORDER BY ts_rank({TSVECTOR_EXPR}, q) DESC, r.stars DESC
    LIMIT $2
"""

#: Exact repository-name lookup — the third arm.
#:
#: `ts_rank` is bad at this in every configuration tried: searching "react",
#: "vue" or "linux" found the canonical repository in 2 of 6 cases and ranked it
#: first in 1. That is not a tuning problem, it is the wrong tool — ranking by
#: term frequency cannot express "this repository is literally called that", and
#: a repo whose owner *and* name both repeat the word will always outscore it.
#:
#: Substring `ILIKE` used to cover this by accident, so removing it was a
#: regression that the eval set is structurally unable to see: all 30 of its
#: queries describe what software does and none looks one up by name.
#:
#: Anchored equality, not a prefix or a wildcard: this arm exists to be right
#: about the exact name, and RRF merges it with the arms that handle everything
#: fuzzier. A multi-word query matches nothing here and the arm simply sits out.
NAME_SQL = """
    SELECT r.id, r.full_name, r.description, r.language, r.domain, r.stars
    FROM repo r
    WHERE lower(split_part(r.full_name, '/', 2)) = lower($1)
       OR lower(r.full_name) = lower($1)
    ORDER BY r.stars DESC
    LIMIT $2
"""


def to_or_tsquery(q: str) -> str:
    """User text to a safe OR'd tsquery.

    Everything non-alphanumeric is dropped rather than escaped. `to_tsquery`
    has its own operator syntax (`&`, `|`, `!`, `:*`, parentheses) and a raw
    user string reaching it is both a syntax error waiting to happen and an
    injection surface into the query parser. Stripping to words removes the
    entire class; "c++" becoming "c" is an acceptable loss for that.
    """
    words = "".join(c if c.isalnum() or c.isspace() else " " for c in q).split()
    return " | ".join(words)


def _as_hits(rows) -> List[Dict[str, Any]]:
    """Postgres rows in the shape the fusion expects from every arm."""
    return [
        {
            "id": r["id"],
            "payload": {
                "full_name": r["full_name"],
                "description": r["description"],
                "language": r["language"],
                "domain": r["domain"],
                "stars": r["stars"],
            },
        }
        for r in rows
    ]


async def lexical_search(conn, q: str, limit: int) -> List[Dict[str, Any]]:
    """Postgres full-text arm, shaped like the dense arm for fusion."""
    tsquery = to_or_tsquery(q)
    if not tsquery:
        return []
    return _as_hits(await conn.fetch(LEXICAL_SQL, tsquery, limit))


async def name_search(conn, q: str, limit: int) -> List[Dict[str, Any]]:
    """Exact repository-name arm. Empty for anything that is not one name."""
    term = q.strip()
    if not term or " " in term:
        return []
    return _as_hits(await conn.fetch(NAME_SQL, term, limit))


#: Alive, licensed, not archived, and scored onboarding-friendly. The last one
#: is why this needed `gitglobe learn` first — `onboarding_ease` didn't exist
#: on 98% of the corpus before the student ran.
APPROACHABLE_PREDICATE = """
    NOT r.is_archived
    AND r.license IS NOT NULL
    AND r.pushed_at > now() - interval '2 years'
    AND COALESCE(t.onboarding_ease, st.onboarding_ease) >= 50
"""


async def filter_approachable(conn, fused: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the fused hits that look worth a newcomer's time.

    Runs once, after fusion, against whichever ids any arm returned — not a
    predicate baked into each arm's own query. The dense arm's Qdrant payload
    carries no license or activity data (re-embedding 186k vectors to add it
    is its own multi-hour job, not justified for a filter), so a single
    Postgres round-trip against the fused ids is the only place all three arms
    can be filtered the same way.

    ponytail: can return fewer than `limit` results when the filter is strict
    relative to the query — no backfill from a second round of candidates.
    Widen the arms' fetch size in `search_repos` if that measurably bites;
    not worth a second round-trip until it does.
    """
    ids = [item["id"] for item in fused]
    if not ids:
        return fused
    rows = await conn.fetch(
        f"""
        SELECT r.id
        FROM repo r
        LEFT JOIN LATERAL (
            SELECT onboarding_ease FROM repo_score
            WHERE repo_id = r.id AND source IN (0, 2)
            ORDER BY source ASC LIMIT 1
        ) t ON true
        LEFT JOIN repo_score st ON st.repo_id = r.id AND st.source = 1
        WHERE r.id = ANY($1) AND {APPROACHABLE_PREDICATE}
        """,
        ids,
    )
    keep = {r["id"] for r in rows}
    return [item for item in fused if item["id"] in keep]


#: Dense candidates fetched before re-ranking. Cosine alone is a *relevance*
#: order, not a *result* order, so the top 10 by cosine are not the 10 to show;
#: `rerank_by_authority` needs a pool deeper than the page to reorder.
DENSE_CANDIDATES = 100

#: Weight of the star prior against cosine similarity.
#:
#: Measured on the 30-query eval set, sweeping alpha over a fixed candidate pool:
#:
#:     0.00 -> 0.242    0.10 -> 0.472
#:     0.02 -> 0.330    0.15 -> 0.464
#:     0.05 -> 0.399    0.40 -> 0.472
#:
#: 0.10 is the knee, and the curve is flat from there to 0.40 — a plateau rather
#: than a spike, so this is not balanced on a knife edge.
STAR_PRIOR_WEIGHT = 0.10


def rerank_by_authority(hits: List[Any]) -> List[Any]:
    """Reorder dense hits by cosine *and* how established the repository is.

    Pure cosine is popularity-blind, and that was the whole reason dense recall
    was weak: for `json parser c` it returned four genuine C JSON parsers that
    nobody uses, while `rapidjson` sat at rank 19 and `nlohmann/json` at 25.
    Every expected repository was present and correctly understood — just buried
    under topically perfect obscurities. The arm was not failing at meaning, it
    was failing at ranking.

    `log10` because stars are Pareto-distributed: raw counts would let one
    400,000-star repository dominate every query, whereas the log makes the
    prior a tie-breaker among things already judged relevant.

    Note this is deliberately *not* the teacher's rubric, which asserts
    popularity never enters a quality score (`assert_no_popularity`). Judging
    whether a project is good and ordering search results are different jobs:
    a repository is not better for being popular, but someone typing two generic
    words is usually looking for the one everybody uses.
    """
    return sorted(
        hits,
        key=lambda h: h.score + STAR_PRIOR_WEIGHT * math.log10(1 + (h.payload.get("stars") or 0)),
        reverse=True,
    )


def reciprocal_rank_fusion(*arms, k=60, weights=None) -> List[Dict[str, Any]]:
    """Merge two ranked lists by rank alone.

    RRF was chosen because it needs no score normalisation between a cosine
    similarity and a `ts_rank` — two scales with no common unit. It does need
    two inputs, and it used to be called with `lexical_results=[]`, which made
    the fusion decorative: a one-input RRF is just the input, reordered by a
    monotonic function of its own rank.

    Variadic because there are three arms now — dense, exact-name, full-text —
    and passing them as one concatenated list would double-count any repository
    two arms both found, which is a scoring rule nobody chose. As separate arms
    that repository still scores twice, but because two arms genuinely ranked
    it, which is the whole point of the fusion.

    Each arm may hold Qdrant ScoredPoints or the dicts the Postgres arms return,
    so both shapes are accepted rather than forcing one to impersonate the other.
    """
    scores: Dict[Any, float] = {}
    repos: Dict[Any, Any] = {}

    for index, results in enumerate(arms):
        weight = weights[index] if weights else 1.0
        for rank, hit in enumerate(results):
            hit_id = hit["id"] if isinstance(hit, dict) else hit.id
            payload = hit["payload"] if isinstance(hit, dict) else hit.payload
            scores[hit_id] = scores.get(hit_id, 0.0) + weight / (k + rank + 1)
            # Prefer whichever arm saw it first; both carry the same columns.
            repos.setdefault(hit_id, payload)

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [{"id": rid, "payload": repos[rid], "score": scores[rid]} for rid in sorted_ids]

# ----------------- Endpoints -----------------

@app.get("/search", response_model=List[SearchResult])
async def search_repos(
    response: Response,
    q: str = Query(..., min_length=2),
    limit: int = Query(50, ge=1, le=100),
    approachable: bool = Query(False, description="Alive, licensed, not archived, onboarding-friendly"),
):
    """Hybrid search: Vertex+Qdrant dense, Postgres full-text lexical, fused by RRF.

    Which arms actually ran is reported in `X-Search-Path`. That header is the
    difference between "search feels bad" being a diagnosable fact and being a
    matter of opinion — a degraded dense arm and a genuinely sparse corpus
    produce identical-looking results, and previously the only way to tell them
    apart was to read the server logs.
    """
    cache_key = f"search:v2:{limit}:{approachable}:{hashlib.sha256(q.encode()).hexdigest()}"
    cached = await state.redis.get(cache_key)
    if cached:
        payload = json.loads(cached)
        response.headers["X-Search-Path"] = payload.get("path", "cached")
        return payload["results"]

    dense_hits: List[Any] = []
    path = []

    # Filtering shrinks the fused list, so ask each arm for more than `limit`
    # up front when it's active — otherwise a strict filter against a
    # `limit`-sized pool routinely returns far fewer than `limit` results.
    fetch_limit = max(limit, DENSE_CANDIDATES) if approachable else limit

    if state.embedder:
        try:
            query_vector = await get_query_embedding(q)
            # `search`, not `query_points`, and the client is pinned to <1.10 to
            # match the 1.9 server. The Universal Query API landed in Qdrant
            # 1.10: a 1.19 client only offers `query_points`, which the 1.9
            # server 404s, while `search` exists only on the older client. There
            # is no single call that spans both, so the versions have to agree.
            #
            # This sat unnoticed because the dense arm could never start to
            # reach Qdrant at all. To upgrade, move the server image and the
            # client together and re-run api/tests/eval_search.py.
            candidates = await state.qdrant.search(
                collection_name=COLLECTION,
                query_vector=query_vector,
                limit=max(limit, DENSE_CANDIDATES),
            )
            dense_hits = rerank_by_authority(candidates)[:fetch_limit]
            path.append("dense")
        except EmbeddingUnavailable as e:
            # Loud on purpose. This used to `print` and continue, so a permanent
            # misconfiguration was indistinguishable from a corpus with no match
            # — which is exactly how a 1024-d query against a 768-d collection
            # went unnoticed while every search quietly became a substring match.
            log.error("Dense arm unavailable, serving lexical only: %s", e)
        except Exception as e:
            log.exception("Dense arm failed unexpectedly: %s", e)

    async with state.db_pool.acquire() as conn:
        lexical_hits = await lexical_search(conn, q, fetch_limit)
        name_hits = await name_search(conn, q, fetch_limit)
    if lexical_hits:
        path.append("lexical")
    if name_hits:
        path.append("name")

    # Arms are not equals, and treating them as equals measurably lost results.
    #
    # The name arm is boosted because typing `linux` is not a request to be
    # ranked against semantically similar repositories, it names one — yet
    # unweighted, the name arm's `torvalds/linux` and the dense arm's
    # `buraksecer/linux-101` both scored 1/61 and the tie broke on argument
    # order, so the wrong one won (name lookups 5/6 -> 6/6 at weight 2.0).
    #
    # The lexical arm is halved because at equal weight it *displaced* better
    # dense hits: recall@10 measured 0.417 at (1,1,1) against 0.472 once dense
    # outweighed it. Sweeping the third weight to 0.0 also gives 0.472 — on this
    # eval set lexical adds nothing at all. It is kept, at a weight where it
    # cannot do harm, because it is the only arm left when the dense arm is
    # down, and because all 30 eval queries are descriptive prose, which is
    # precisely the shape lexical is worst at and the eval cannot see.
    fused = reciprocal_rank_fusion(
        dense_hits, name_hits, lexical_hits, k=60, weights=(2.0, 2.0, 1.0)
    )
    if approachable and fused:
        async with state.db_pool.acquire() as conn:
            fused = await filter_approachable(conn, fused)
    fused = fused[:limit]
    final_results = [
        SearchResult(
            repo=RepoMetadata(
                id=item["id"],
                full_name=item["payload"]["full_name"],
                description=item["payload"].get("description"),
                language=item["payload"].get("language"),
                domain=item["payload"].get("domain"),
                stars=item["payload"].get("stars") or 0,
            ),
            score=item["score"],
        )
        for item in fused
    ]

    served = "+".join(path) or "none"
    response.headers["X-Search-Path"] = served
    await state.redis.setex(
        cache_key,
        3600,
        json.dumps({"path": served, "results": [r.model_dump() for r in final_results]}),
    )
    return final_results

#: `source = 0` is the teacher — an LLM that actually read the README. The
#: student (`source = 1`) predicts the six numeric scores for the whole corpus
#: but cannot write prose, so it never has a summary to offer.
#:
#: **The gate is `low_signal`, not `insufficient_evidence`.** The design spec
#: asked for the latter, and measurement says it is the wrong signal. `rubric.py`
#: defines it per-*dimension* — "if the README is too thin to judge a dimension,
#: use 50 and add insufficient_evidence" — so it fires whenever a README cannot
#: support a numeric quality score, which a curated list or a book collection
#: never can. It says nothing about whether the thing can be described. Over the
#: first 200 rated repositories it fired on 124, and their summaries were
#: indistinguishable in quality from the unflagged ones: `build-your-own-x` was
#: flagged and `sindresorhus/awesome` was not, and both got an accurate sentence.
#: Honouring it would suppress 62% of good summaries to prevent nothing.
#:
#: `repo.low_signal` is the cleaner's own verdict that a README carries too
#: little text to be worth anything, which is exactly the failure the spec was
#: reaching for: a fluent, confident, wrong description inferred from a title and
#: a badge row, published under a real person's repository name.
#:
#: `NULLIF` on the trimmed summary keeps the nullable contract honest — an empty
#: string must read as "no summary", not as a summary that happens to be blank.
#:
#: `SOURCE_TEACHER_DISPLAY` (2) covers two callers that never train the student:
#: the coverage-fill batch job (`gitglobe teach --source-tag 2 ...`, top-star
#: repos the original stratified sample missed) and `maybe_generate_summary`
#: below (one repo, on demand, when someone opens it). Sharing one value is
#: deliberate — nothing downstream needs to tell them apart, only `learn`'s
#: `source = 0` filter needs them kept out of training, and one shared bucket is
#: less to get wrong than two.
#:
#: `onboarding_ease`/`learning_value` prefer the teacher's read where one
#: exists and fall back to the student, which is the only one of the two with
#: 100% coverage — `gitglobe learn` is what makes that COALESCE non-trivial.
REPO_SELECT = """
    SELECT r.id, r.full_name, r.description, r.language, r.domain, r.stars,
           r.license, r.pushed_at, r.is_archived, r.clean_text, r.low_signal,
           CASE WHEN r.low_signal THEN NULL ELSE NULLIF(trim(t.summary), '') END
               AS summary,
           COALESCE(t.onboarding_ease, st.onboarding_ease) AS onboarding_ease,
           COALESCE(t.learning_value, st.learning_value) AS learning_value
    FROM repo r
    -- `source IN (0, 2)` cannot be a plain join condition: repo_score's key is
    -- (repo_id, source), so a repo with both a source=0 and a source=2 row
    -- would join twice and silently duplicate the result. LATERAL + LIMIT 1
    -- picks one, preferring 0 (the careful original sample) over 2 (batch or
    -- on-demand fill) when both exist.
    LEFT JOIN LATERAL (
        SELECT summary, onboarding_ease, learning_value
        FROM repo_score
        WHERE repo_id = r.id AND source IN (0, 2)
        ORDER BY source ASC
        LIMIT 1
    ) t ON true
    LEFT JOIN repo_score st ON st.repo_id = r.id AND st.source = 1
    WHERE {predicate}
"""

#: See the comment on `REPO_SELECT` above.
SOURCE_TEACHER_DISPLAY = 2


async def maybe_generate_summary(conn, row) -> Optional[str]:
    """One repo, one generation attempt, cached forever after.

    Only reached when `REPO_SELECT` already found no summary. Skips the same
    repos the display query already refuses to show a summary for — no
    `clean_text` or `low_signal` — for the same reason: a fluent sentence
    inferred from a title and a badge row is worse than none, published under
    a real person's repository name.

    Single-flight per repo via a Redis `NX` lock, not a distributed-lock
    library — the first concurrent request generates, the rest get nothing
    *this* time rather than duplicating the NIM call, and the next click finds
    the cached row. Bounded latency, not retried: a slow or failed call costs
    this one request, once; add a retry queue only if that measurably isn't
    enough under real traffic.
    """
    if not state.summarizer or not row["clean_text"] or row["low_signal"]:
        return None

    lock_key = f"summarizing:{row['id']}"
    acquired = await state.redis.set(lock_key, "1", nx=True, ex=60)
    if not acquired:
        return None

    try:
        summary = await state.summarizer.generate(
            row["full_name"], row["description"], row["license"], row["clean_text"]
        )
    except SummaryUnavailable as e:
        log.info("On-demand summary skipped for %s: %s", row["full_name"], e)
        return None

    await conn.execute(
        """
        INSERT INTO repo_score (repo_id, source, summary, model)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (repo_id, source) DO NOTHING
        """,
        row["id"], SOURCE_TEACHER_DISPLAY, summary, SUMMARY_MODEL,
    )
    return summary


@app.get("/repo/{repo_id}", response_model=Optional[RepoMetadata])
async def get_repo(repo_id: int, name: Optional[str] = Query(None)):
    async with state.db_pool.acquire() as conn:
        row = None

        # The frontend passes `repo_id` as the rank ordinal (1-87227), but our DB
        # uses a SERIAL id with gaps. We must look up by `full_name` to get the
        # correct row, because the ordinal does not match the DB id.
        if name:
            row = await conn.fetchrow(REPO_SELECT.format(predicate="r.full_name = $1"), name)

        # Fallback to ID ONLY if name wasn't provided (for legacy compatibility)
        if not row and not name:
            row = await conn.fetchrow(REPO_SELECT.format(predicate="r.id = $1"), repo_id)

        if row:
            metadata = RepoMetadata(**row)
            if metadata.summary is None:
                generated = await maybe_generate_summary(conn, row)
                if generated:
                    metadata.summary = generated
            return metadata
    
    # Final fallback: fetch from GitHub public API (60 req/hr unauthenticated).
    # Cache results in Redis for 24h so repeated clicks don't burn rate limits.
    if name:
        cache_key = f"gh:{name}"
        cached = await state.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            return RepoMetadata(**data) if data else None
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                gh_res = await client.get(
                    f"https://api.github.com/repos/{name}",
                    headers={"Accept": "application/vnd.github.v3+json"},
                    timeout=5.0,
                )
            if gh_res.status_code == 200:
                gh = gh_res.json()
                meta = {
                    "id": repo_id,
                    "full_name": gh.get("full_name", name),
                    "description": gh.get("description"),
                    "language": gh.get("language"),
                    "domain": None,
                    "stars": gh.get("stargazers_count", 0),
                }
                await state.redis.setex(cache_key, 24 * 3600, json.dumps(meta))
                return RepoMetadata(**meta)
            else:
                # Cache the miss too so we don't hammer GitHub
                await state.redis.setex(cache_key, 24 * 3600, json.dumps(None))
        except Exception as e:
            print(f"GitHub fallback failed for {name}: {e}")
    
    return None

@app.get("/graph/{repo_id}", response_model=GraphResponse)
async def get_graph(repo_id: int, name: Optional[str] = Query(None), depth: int = Query(1), kind: str = Query("dependency")):
    async with state.db_pool.acquire() as conn:
        true_db_id = repo_id
        if name:
            row = await conn.fetchrow("SELECT id FROM repo WHERE full_name = $1", name)
            if row:
                true_db_id = row["id"]
                
        # 0 = depends_on, 1 = similar_to, 2 = used_with
        edge_kinds = [0, 2] if kind == "dependency" else [0, 1, 2]
        
        query = """
            SELECT src_repo.full_name AS src_name, dst_repo.full_name AS dst_name
            FROM edge
            JOIN repo AS src_repo ON edge.src = src_repo.id
            JOIN repo AS dst_repo ON edge.dst = dst_repo.id
            WHERE (edge.src = $1 OR edge.dst = $1)
              AND edge.kind = ANY($2::int[])
            LIMIT 500
        """
        
        rows = await conn.fetch(query, true_db_id, edge_kinds)
        
        nodes = set()
        edges = []
        for r in rows:
            nodes.add(r["src_name"])
            nodes.add(r["dst_name"])
            edges.append(GraphEdge(source=r["src_name"], target=r["dst_name"]))
            
        if not nodes:
            r = await conn.fetchrow("SELECT full_name FROM repo WHERE id = $1", true_db_id)
            if r:
                nodes.add(r["full_name"])
                
        return GraphResponse(nodes=list(nodes), edges=edges)
