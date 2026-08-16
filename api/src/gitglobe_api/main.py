import os
import json
import hashlib
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# ----------------- Configuration -----------------
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpg
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import voyageai
from redis.asyncio import Redis

ENV = os.getenv("ENV", "development")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    if ENV == "production":
        raise ValueError("DATABASE_URL is required in production")
    DATABASE_URL = "postgresql://gitglobe:gitglobe@localhost:5433/gitglobe"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
# Trailing slashes and spaces stripped: the browser's `Origin` header never has
# a trailing slash, so "https://app.vercel.app/" silently matches nothing and
# every request fails CORS while the server logs look completely healthy.
CORS_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# ----------------- State -----------------
class AppState:
    db_pool: asyncpg.Pool = None
    qdrant: AsyncQdrantClient = None
    redis: Redis = None
    voyage_client = None

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
    if VOYAGE_API_KEY:
        state.voyage_client = voyageai.AsyncClient(api_key=VOYAGE_API_KEY)
    
    yield
    
    # Shutdown
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
    if not state.voyage_client:
        raise HTTPException(status_code=500, detail="Voyage API Key not set.")
    
    cache_key = f"embed:{hashlib.sha256(query.encode()).hexdigest()}"
    cached = await state.redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Voyage AI expects a list of queries
    result = await state.voyage_client.embed([query], model="voyage-2")
    embedding = result.embeddings[0]
    
    # Cache for 7 days
    await state.redis.setex(cache_key, 7 * 24 * 3600, json.dumps(embedding))
    return embedding

def reciprocal_rank_fusion(dense_results, lexical_results, k=60) -> List[Dict[str, Any]]:
    # dense_results and lexical_results are lists of Qdrant ScoredPoint
    scores = {}
    repos = {}
    
    for rank, hit in enumerate(dense_results):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
        repos[hit.id] = hit.payload
        
    for rank, hit in enumerate(lexical_results):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
        repos[hit.id] = hit.payload
        
    # Sort by RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    return [{"id": rid, "payload": repos[rid], "score": scores[rid]} for rid in sorted_ids]

# ----------------- Endpoints -----------------

@app.get("/search", response_model=List[SearchResult])
async def search_repos(q: str = Query(..., min_length=2)):
    cache_key = f"search:{hashlib.sha256(q.encode()).hexdigest()}"
    cached = await state.redis.get(cache_key)
    if cached:
        return json.loads(cached)
        
    final_results = []
    
    if state.voyage_client:
        try:
            # 1. Embed query
            query_vector = await get_query_embedding(q)
            
            # 2. Qdrant Dense Search
            if hasattr(state.qdrant, "query_points"):
                res = await state.qdrant.query_points(
                    collection_name="gitglobe_repos",
                    query=query_vector,
                    limit=50
                )
                dense_hits = res.points
            else:
                dense_hits = await state.qdrant.search(
                    collection_name="gitglobe_repos",
                    query_vector=query_vector,
                    limit=50
                )
            
            rrf_results = reciprocal_rank_fusion(dense_hits, [], k=60)
            top_50 = rrf_results[:50]
            
            if top_50:
                documents = [item["payload"]["description"] or item["payload"]["full_name"] for item in top_50]
                rerank_result = await state.voyage_client.rerank(q, documents, model="rerank-2", top_k=50)
                
                for r in rerank_result.results:
                    orig_item = top_50[r.index]
                    final_results.append(SearchResult(
                        repo=RepoMetadata(
                            id=orig_item["id"],
                            full_name=orig_item["payload"]["full_name"],
                            description=orig_item["payload"]["description"],
                            language=orig_item["payload"]["language"],
                            domain=orig_item["payload"]["domain"],
                            stars=orig_item["payload"]["stars"]
                        ),
                        score=r.relevance_score
                    ))
        except Exception as e:
            print(f"Voyage search failed: {e}")
            final_results = []

    if not final_results:
        # Fallback to Postgres ILIKE
        async with state.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, full_name, description, language, domain, stars
                FROM repo
                WHERE full_name ILIKE $1 OR description ILIKE $1
                ORDER BY stars DESC
                LIMIT 50
            """, f"%{q}%")
            
            for i, r in enumerate(rows):
                final_results.append(SearchResult(
                    repo=RepoMetadata(
                        id=r["id"],
                        full_name=r["full_name"],
                        description=r["description"],
                        language=r["language"],
                        domain=r["domain"],
                        stars=r["stars"]
                    ),
                    score=1.0 / (i + 1)
                ))
        
    # Cache result (1 hour)
    await state.redis.setex(cache_key, 3600, json.dumps([r.model_dump() for r in final_results]))
    
    return final_results

@app.get("/repo/{repo_id}", response_model=Optional[RepoMetadata])
async def get_repo(repo_id: int, name: Optional[str] = Query(None)):
    async with state.db_pool.acquire() as conn:
        row = None
        
        # The frontend passes `repo_id` as the rank ordinal (1-87227), but our DB
        # uses a SERIAL id with gaps. We must look up by `full_name` to get the
        # correct row, because the ordinal does not match the DB id.
        if name:
            row = await conn.fetchrow("""
                SELECT id, full_name, description, language, domain, stars
                FROM repo
                WHERE full_name = $1
            """, name)
            
        # Fallback to ID ONLY if name wasn't provided (for legacy compatibility)
        if not row and not name:
            row = await conn.fetchrow("""
                SELECT id, full_name, description, language, domain, stars
                FROM repo
                WHERE id = $1
            """, repo_id)
        
        if row:
            return RepoMetadata(**row)
    
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
