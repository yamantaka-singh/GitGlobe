"""Configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Searched upward from this file, so the CLI works from any directory. Running
#: `gitglobe` from `web/` is common enough that requiring a specific cwd would
#: be a daily annoyance.
_ENV_FILENAME = ".env"


def load_dotenv(start: Path | None = None) -> dict:
    """Read `.env` into the environment. Real env vars always win.

    Written by hand rather than adding `python-dotenv`: it is fifteen lines,
    and a dependency that exists to parse `KEY=value` is not worth the install.

    **Existing environment variables are never overwritten.** A value exported
    in the shell, or injected by CI, must beat a stale line in a file someone
    forgot about — the alternative is a secret that changes depending on which
    directory you ran from.
    """
    here = (start or Path(__file__).resolve()).parent
    loaded: dict = {}
    for directory in [here, *here.parents]:
        candidate = directory / _ENV_FILENAME
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded[key] = value
        break
    return loaded


def _tokens() -> list[str]:
    """Accept one token or many.

    GITHUB_TOKENS (comma-separated) wins; GITHUB_TOKEN is the single-token
    convenience. More tokens means proportionally faster ingest, because the
    5,000 points/hour limit is per token.
    """
    multi = os.getenv("GITHUB_TOKENS", "")
    if multi.strip():
        return [t.strip() for t in multi.split(",") if t.strip()]
    single = os.getenv("GITHUB_TOKEN", "").strip()
    return [single] if single else []


@dataclass
class Settings:
    database_url: str
    github_tokens: list[str] = field(default_factory=list)
    gcs_bucket: str = ""
    gcp_project: str = ""
    nvidia_api_key: str = ""
    teacher_provider: str = "nim"
    teacher_rpm: float = 0.0

    @classmethod
    def from_env(cls, *, require_github: bool = True) -> "Settings":
        """Read configuration.

        `require_github=False` for the Phase 2 stages. Projection and clustering
        run entirely on local CPU against rows already in Postgres, and failing
        them for a missing ingest credential would be an obstacle with no
        purpose — the sort of thing that makes people set a junk token and
        forget the check exists.
        """
        load_dotenv()
        settings = cls(
            database_url=os.getenv(
                "DATABASE_URL", "postgresql://gitglobe:gitglobe@localhost:5433/gitglobe"
            ),
            github_tokens=_tokens(),
            gcs_bucket=os.getenv("GCS_BUCKET", ""),
            gcp_project=os.getenv("GCP_PROJECT", ""),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", "").strip(),
            teacher_provider=os.getenv("TEACHER_PROVIDER", "nim").strip().lower(),
            teacher_rpm=float(os.getenv("TEACHER_RPM", "0") or 0),
        )
        if require_github and not settings.github_tokens:
            raise RuntimeError(
                "No GitHub token. Set GITHUB_TOKEN (or GITHUB_TOKENS for a pool).\n"
                "Create one at https://github.com/settings/tokens — `public_repo` is enough."
            )
        return settings
