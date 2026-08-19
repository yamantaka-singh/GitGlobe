"""The teacher: an LLM reads READMEs and rates a few thousand repositories.

This is the only part of the brain that costs money per repository, and it runs
on roughly 4,000 of them rather than a million. Everything else predicts what
this would have said.

**The teacher never sees popularity.** `assert_no_popularity` runs on every
prompt before it is sent, and a leak raises rather than warns. If stars reached
the prompt, every label would be a laundered star count, the student would fit
it beautifully, and the whole exercise would have produced an expensive way to
recompute a number we already had. That check is the load-bearing line in this
file.

Resumability matters more here than elsewhere because the work is paid for. A
run that dies at row 3,000 must not re-rate those 3,000, so results are written
in batches and the pending list is computed from what is already stored.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field

from .rubric import (
    DIMENSION_KEYS,
    SYSTEM_PROMPT,
    assert_no_popularity,
    build_teacher_prompt,
    parse_teacher_response,
)

log = logging.getLogger(__name__)

#: Verified available on Vertex. Flash rather than Pro: the task is reading one
#: README against a fixed rubric, which is not where reasoning depth pays, and
#: the cost difference across thousands of calls is the whole budget.
DEFAULT_TEACHER_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"

#: NVIDIA's OpenAI-compatible endpoint. Free with a build.nvidia.com key.
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

#: Chosen for surviving sustained load, not for being the strongest model.
#:
#: `nemotron-3-ultra-550b-a55b` answers a real teacher prompt in ~2s and rates
#: better, but its free-tier quota collapses under a real run: 45s of sustained
#: load measured **10 successes against 10,909 rejections**, and afterwards it
#: returned 429 to every single request until the quota recovered. A model that
#: is 99.9% rejected is not faster, it is unusable.
#:
#: This one measured 95 successes against 29 rejections under the same load, and
#: parsed 3 of 3 real prompts. It is genuinely slower per call — ~65s against
#: ~2s on a full README at max_tokens 4096 — but it finishes, which ultra does
#: not.
#:
#: The cost is score drift: re-rating repositories ultra had already scored gave
#: a mean absolute difference of ~21 points on the 0-100 dimensions, though the
#: summaries stayed comparable. `repo_score.model` records which model produced
#: each row, so a future `learn` run can filter to one model rather than train
#: on a mixture. Set `--model` to override.
NIM_DEFAULT_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

DEFAULT_CONCURRENCY = 8

#: Workers for NIM. Sized as rate x latency: 40 requests/minute against ~90s
#: per call needs ~60 in flight to keep the limiter, rather than idle workers,
#: as the binding constraint. Measured from a real run that managed 5/min on 8.
NIM_CONCURRENCY = 60
CHECKPOINT_EVERY = 200


class RateLimiter:
    """Requests per minute, enforced across concurrent workers.

    A free tier's limit is a hard one: exceed it and you get 429s that cost more
    time than the pacing would have. Per-worker sleeps do not work — N workers
    each sleeping 1.5s still issue N requests at once — so the limiter holds a
    single shared clock and hands out slots in order.
    """

    def __init__(self, per_minute: float):
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait:
            await asyncio.sleep(wait)

#: Rough Flash pricing, USD per million tokens. Only used for the estimate
#: printed before a run; verify against the pricing page before a large one.
USD_IN_PER_MILLION = 0.30
USD_OUT_PER_MILLION = 2.50

#: Enough for six integers, a sentence, and a short flag list.
MAX_OUTPUT_TOKENS = 2048


@dataclass
class TeacherConfig:
    """Which model rates, and how fast.

    `provider` is "vertex" or "nim". Everything that decides label *quality* —
    the rubric, the prompt, the popularity guard, the parser — is shared. Only
    the HTTP shape differs, which is why this is a config flag rather than two
    parallel implementations that would drift.

    Being able to run both is worth more than either alone: two independent
    teachers disagreeing on a dimension means the rubric is ambiguous there, and
    that is a defect in the rubric, not in the models.
    """

    project: str = ""
    provider: str = "nim"
    location: str = DEFAULT_LOCATION
    model: str = ""
    concurrency: int = DEFAULT_CONCURRENCY
    max_retries: int = 5
    timeout_s: float = 180.0
    #: 0 disables pacing. NVIDIA's free tier is 40; Vertex is far higher.
    requests_per_minute: float = 0.0

    def __post_init__(self) -> None:
        if not self.model:
            self.model = NIM_DEFAULT_MODEL if self.provider == "nim" else DEFAULT_TEACHER_MODEL
        if self.provider == "nim":
            if not self.requests_per_minute:
                # Per key, not per run. One shared limiter paced at
                # `40 x len(keys)` combined with round-robin gives each key its
                # own 40/min, which is the whole reason a pool helps.
                import os as _os

                pool = [k for k in _os.environ.get("NVIDIA_API_KEYS", "").split(",") if k.strip()]
                self.requests_per_minute = 40.0 * max(len(pool), 1)
            # This used to clamp to 8, reasoning that at 40 RPM the limiter set
            # throughput anyway and extra workers would just wait. Measured, that
            # is backwards: a 550B reasoning call takes ~90s, so 8 workers give
            # ~5 requests a minute — an eighth of the limit, with the limiter
            # never once binding. Concurrency was the constraint the whole time.
            #
            # Workers needed to saturate a rate limit is rate x latency:
            # 40/min x 90s = 60. The limiter still enforces the ceiling, so
            # over-provisioning here cannot exceed the quota; it only stops
            # workers idling. Lower it if NIM starts returning 503s faster than
            # the retry budget absorbs them.
            if self.concurrency == DEFAULT_CONCURRENCY:
                self.concurrency = NIM_CONCURRENCY

    @property
    def endpoint(self) -> str:
        if self.provider == "nim":
            return NIM_BASE_URL
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project}"
            f"/locations/{self.location}/publishers/google/models/{self.model}:generateContent"
        )


@dataclass
class TeacherStats:
    requested: int = 0
    scored: int = 0
    unparseable: int = 0
    failed: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    flags: dict = field(default_factory=dict)
    failures: dict = field(default_factory=dict)

    def cost(self) -> float:
        return (
            self.input_tokens / 1e6 * USD_IN_PER_MILLION
            + self.output_tokens / 1e6 * USD_OUT_PER_MILLION
        )

    def summary(self) -> str:
        return (
            f"{self.scored}/{self.requested} rated, {self.unparseable} unparseable, "
            f"{self.failed} failed, {self.retries} retries, "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens "
            f"(~${self.cost():.2f})"
        )


def estimate(n_rows: int, mean_readme_chars: float, config=None) -> dict:
    """Cost and duration before spending them. Print this and look at it.

    Both numbers used to ignore the provider and both were wrong for NIM.

    **Duration.** The old formula was `n / (DEFAULT_CONCURRENCY * 1.5) / 60`,
    which asserts 24 rows a second and ignores the rate limiter entirely. It
    quoted 2.8 minutes for a 4,000-row run that actually takes hours, and a
    number that far off invites killing a healthy run for looking hung. Pace is
    set by `requests_per_minute`, so that is what this divides by now — still
    a floor rather than a promise, because a large reasoning model can be slower
    than the limiter and then latency binds instead.

    **Cost.** The rates below are Vertex Flash. NIM's free tier bills nothing,
    and quoting $3.19 for it is not a harmless overestimate — it is a reason not
    to run something that is free.
    """
    input_tokens = n_rows * (mean_readme_chars / 3.5 + 900)
    output_tokens = n_rows * 120
    provider = getattr(config, "provider", "vertex")
    billed = provider != "nim"
    usd = (
        input_tokens / 1e6 * USD_IN_PER_MILLION
        + output_tokens / 1e6 * USD_OUT_PER_MILLION
    ) if billed else 0.0

    rpm = getattr(config, "requests_per_minute", 0.0) or 0.0
    concurrency = getattr(config, "concurrency", DEFAULT_CONCURRENCY)
    minutes = n_rows / rpm if rpm > 0 else n_rows / max(concurrency, 1) / 60
    return {
        "rows": n_rows,
        "est_input_tokens": int(input_tokens),
        "est_output_tokens": int(output_tokens),
        "est_usd": round(usd, 2),
        "billed": billed,
        "est_minutes": round(minutes, 1),
        "rate_limit_rpm": rpm,
    }


class Teacher:
    """Async Vertex `generateContent` client for the rating pass."""

    def __init__(self, config: TeacherConfig):
        self.config = config
        self.stats = TeacherStats()
        self.limiter = RateLimiter(config.requests_per_minute)
        self._client = None
        self._credentials = None
        self._api_key = None
        self._api_keys: list[str] = []
        self._key_index = 0

    async def __aenter__(self) -> "Teacher":
        import os

        import httpx

        if self.config.provider == "nim":
            # Plural wins. The free tier's request limit is per key and it binds
            # hard — one key measured 103 HTTP 429s against 13 successes — so a
            # pool is the only thing that moves throughput. `requests_per_minute`
            # is scaled by the pool size in TeacherConfig, and keys rotate per
            # request, so each key still sees its own per-key rate.
            multi = os.environ.get("NVIDIA_API_KEYS", "").strip()
            if multi:
                self._api_keys = [k.strip() for k in multi.split(",") if k.strip()]
            else:
                single = os.environ.get("NVIDIA_API_KEY", "").strip()
                self._api_keys = [single] if single else []
            if not self._api_keys:
                raise RuntimeError(
                    "No NVIDIA_API_KEY. Get one free at https://build.nvidia.com "
                    "and export NVIDIA_API_KEY=nvapi-... "
                    "(or NVIDIA_API_KEYS=key1,key2,... for a pool)."
                )
            self._api_key = self._api_keys[0]
        else:
            from google.auth import default as google_auth_default
            from google.oauth2.credentials import Credentials

            token = os.environ.get("GOOGLE_ACCESS_TOKEN")
            if token:
                self._credentials = Credentials(token=token)
                project = os.environ.get("GCP_PROJECT", "")
            else:
                self._credentials, project = google_auth_default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            if not self.config.project:
                self.config.project = project or ""
            if not self.config.project:
                raise RuntimeError("No GCP project. Set GCP_PROJECT.")

        self._client = httpx.AsyncClient(timeout=self.config.timeout_s)
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client:
            await self._client.aclose()

    def _headers(self) -> dict:
        if self.config.provider == "nim":
            # Round-robin per request. Not thread-safe by design: this runs on
            # one event loop, and an occasional uneven split across keys costs
            # nothing, whereas a lock on every request buys nothing back.
            key = self._api_keys[self._key_index % len(self._api_keys)]
            self._key_index += 1
            return {"Authorization": f"Bearer {key}"}
        from google.auth.transport.requests import Request

        # Access tokens last an hour; a 4,000-row run at 40 RPM outlives two.
        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return {"Authorization": f"Bearer {self._credentials.token}"}

    def _request_body(self, prompt: str) -> dict:
        if self.config.provider == "nim":
            return {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                # Temperature 0 so the same README always gets the same scores.
                # A sampled teacher writes noise straight into the labels.
                "temperature": 0.0,
                # Reasoning models spend tokens thinking before answering, so
                # the budget has to cover both or the JSON is cut off mid-object
                # and the row is lost to a parse failure.
                "max_tokens": 4096,
                # Thinking off. The rubric asks for six integers and one
                # sentence about a README that is already in the prompt — there
                # is no multi-step problem for a chain of thought to help with,
                # and leaving it on cost 10.7s and 23.1s per repository at 523
                # and 1,168 output tokens, against 1.4s and 2.3s for 77 and 84
                # with it off, for *identical* scores on the repos checked.
                #
                # This does NOT fix the rate limiting, which was the reason it
                # was tried: a paced run still measured 103 HTTP 429s against 13
                # successes. The 429s are a per-model quota on this endpoint,
                # not a consequence of holding connections open, and no
                # concurrency or pacing setting works around them.
                #
                # Smaller models are not a substitute either. `lightning-30b-a3b`
                # spent the entire 4,096-token budget thinking and returned no
                # JSON at all on 2 of 3 repositories, and `super-120b` answered
                # fast but scored the canonical awesome list 75 where this model
                # says 100.
                "chat_template_kwargs": {"thinking": False},
            }
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
                "responseMimeType": "application/json",
            },
        }

    def _read_response(self, payload: dict) -> str | None:
        """Pull the text out of whichever response shape came back."""
        if self.config.provider == "nim":
            usage = payload.get("usage", {})
            self.stats.input_tokens += int(usage.get("prompt_tokens", 0))
            self.stats.output_tokens += int(usage.get("completion_tokens", 0))
            choices = payload.get("choices") or []
            if not choices:
                return None
            message = choices[0].get("message") or {}
            # Some NIM builds put the chain of thought in a separate field and
            # the answer in `content`; others inline both. Preferring `content`
            # and letting the parser strip <think> covers each.
            return message.get("content") or message.get("reasoning_content") or ""

        usage = payload.get("usageMetadata", {})
        self.stats.input_tokens += int(usage.get("promptTokenCount", 0))
        self.stats.output_tokens += int(usage.get("candidatesTokenCount", 0))
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts") or [{}]
        return parts[0].get("text", "")

    async def rate_one(self, repo: dict) -> dict | None:
        """Rate one repository. None if it could not be rated."""
        import httpx

        prompt = build_teacher_prompt(repo)
        # Raises, not warns. A popularity leak invalidates every label produced
        # after it, and a warning in a log nobody reads during a paid run is
        # indistinguishable from no check at all.
        assert_no_popularity(prompt)

        body = self._request_body(prompt)

        for attempt in range(self.config.max_retries):
            await self.limiter.acquire()
            try:
                response = await self._client.post(
                    self.config.endpoint, json=body, headers=self._headers()
                )
                if response.status_code == 200:
                    text = self._read_response(response.json())
                    if text is None:
                        # Usually a safety block. Costs this row, not the run.
                        self.stats.unparseable += 1
                        return None
                    scores = parse_teacher_response(text)
                    if scores is None:
                        self.stats.unparseable += 1
                        log.debug("Unparseable response: %s", text[:300])
                        return None
                    for flag in scores.get("flags", []):
                        self.stats.flags[flag] = self.stats.flags.get(flag, 0) + 1
                    return scores

                if response.status_code != 429 and 400 <= response.status_code < 500:
                    self._record_failure(f"http_{response.status_code}")
                    log.warning("Permanent %s: %s", response.status_code, response.text[:200])
                    return None
                reason = f"http_{response.status_code}"
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                reason = type(exc).__name__

            self.stats.retries += 1
            # Half-jittered, never near zero: full jitter under a 429 means
            # hammering a service that just asked you to slow down.
            await asyncio.sleep(min(2**attempt, 32) * (0.5 + 0.5 * random.random()))
            log.debug("Retry %d after %s", attempt + 1, reason)

        self._record_failure("exhausted_retries")
        return None

    def _record_failure(self, reason: str) -> None:
        self.stats.failed += 1
        self.stats.failures[reason] = self.stats.failures.get(reason, 0) + 1

    async def rate_many(self, repos: list[dict], *, on_batch=None) -> dict[int, dict]:
        """Rate a list of repositories concurrently, checkpointing as it goes."""
        semaphore = asyncio.Semaphore(self.config.concurrency)
        results: dict[int, dict] = {}
        pending: dict[int, dict] = {}
        lock = asyncio.Lock()

        async def worker(repo: dict) -> None:
            assert repo.get("id") is not None, "Repository payload missing ID"
            try:
                async with semaphore:
                    scores = await self.rate_one(repo)
            except Exception as exc:
                # `rate_one` calls `assert_no_popularity`, which RAISES by
                # design. Uncontained, one repository would cancel every other
                # in-flight worker and discard the unpersisted batch — an hour
                # of a 40 RPM run, lost to a single README.
                self._record_failure(f"worker_{type(exc).__name__}")
                log.warning("Worker failed for %s: %s", repo.get("full_name"), exc)
                return
            if scores is None:
                return
            async with lock:
                self.stats.scored += 1
                results[repo["id"]] = scores
                pending[repo["id"]] = scores
                if on_batch and len(pending) >= CHECKPOINT_EVERY:
                    flush = dict(pending)
                    pending.clear()
                    await on_batch(flush)

        self.stats.requested += len(repos)
        await asyncio.gather(*(worker(r) for r in repos), return_exceptions=True)
        if on_batch and pending:
            await on_batch(dict(pending))
        return results


def label_matrix(scores: dict[int, dict], repo_ids) -> tuple:
    """Teacher scores as an (n, 6) matrix aligned with `repo_ids`.

    Returns `(matrix, mask)`. The mask marks rows that were actually rated —
    unrated rows are NOT filled with a default. Imputing 50 would teach the
    student that "we could not read this" means "average", which is a fact about
    the pipeline rather than the repository.
    """
    import numpy as np

    n = len(repo_ids)
    matrix = np.zeros((n, len(DIMENSION_KEYS)), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    for i, rid in enumerate(repo_ids):
        entry = scores.get(int(rid))
        if entry is None:
            continue
        matrix[i] = [entry[k] for k in DIMENSION_KEYS]
        mask[i] = True
    return matrix, mask
