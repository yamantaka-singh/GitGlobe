# Exercises the cache_control middleware in isolation — no DB, no Qdrant.
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

app = FastAPI()

@app.middleware("http")
async def cache_control(request, call_next):
    response = await call_next(request)
    if "cache-control" not in response.headers:
        if 200 <= response.status_code < 300:
            response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=300"
        else:
            response.headers["Cache-Control"] = "no-store"
    return response

@app.get("/ok")
async def ok(): return {"ok": True}

@app.get("/boom")
async def boom(): raise HTTPException(status_code=500, detail="nope")

@app.get("/missing")
async def missing(): raise HTTPException(status_code=404, detail="nope")

c = TestClient(app, raise_server_exceptions=False)

r = c.get("/ok")
assert r.status_code == 200
assert r.headers["cache-control"] == "public, max-age=60, stale-while-revalidate=300", r.headers
print("200 ->", r.headers["cache-control"])

for path, code in (("/boom", 500), ("/missing", 404)):
    r = c.get(path)
    assert r.status_code == code
    assert r.headers["cache-control"] == "no-store", (path, r.headers)
    print(f"{code} ->", r.headers["cache-control"])

print("\nPASS: successes are briefly cacheable, failures are never stored")
