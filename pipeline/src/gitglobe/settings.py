"""Configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    gcp_project: str = ""

    @classmethod
    def from_env(cls, *, require_github: bool = True) -> "Settings":
        """Read configuration.

        `require_github=False` for the Phase 2 stages. Projection and clustering
        run entirely on local CPU against rows already in Postgres, and failing
        them for a missing ingest credential would be an obstacle with no
        purpose — the sort of thing that makes people set a junk token and
        forget the check exists.
        """
        settings = cls(
            database_url=os.getenv(
                "DATABASE_URL", "postgresql://gitglobe:gitglobe@localhost:5433/gitglobe"
            ),
            github_tokens=_tokens(),
            gcp_project=os.getenv("GCP_PROJECT", ""),
        )
        if require_github and not settings.github_tokens:
            raise RuntimeError(
                "No GitHub token. Set GITHUB_TOKEN (or GITHUB_TOKENS for a pool).\n"
                "Create one at https://github.com/settings/tokens — `public_repo` is enough."
            )
        return settings
