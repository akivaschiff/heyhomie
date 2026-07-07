"""Central configuration. All env-driven knobs in one place."""

import os
from dataclasses import dataclass, field
from pathlib import Path

PI_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PI_DIR.parent

CLAUDE_MODEL = os.environ.get("HOMIE_MODEL", "claude-haiku-4-5-20251001")
CLAUDE_MAX_TOKENS = int(os.environ.get("HOMIE_MAX_TOKENS", "1500"))

# Rolling conversation window (~15 min). Older turns fall away; the model
# re-asks or re-fetches if it needs them. Distinct from the voice re-wake
# "listening window", which lives in the voice channel.
CONTEXT_WINDOW_SECONDS = int(os.environ.get("HOMIE_CONTEXT_WINDOW", str(15 * 60)))

# Default location for Shabbat times (Hebcal geonameid). 281184 = Jerusalem.
SHABBAT_GEONAME_ID = os.environ.get("HOMIE_SHABBAT_GEONAME", "281184")


@dataclass
class Config:
    store_backend: str = os.environ.get("HOMIE_STORE", "local")
    model: str = CLAUDE_MODEL
    max_tokens: int = CLAUDE_MAX_TOKENS
    context_window_seconds: int = CONTEXT_WINDOW_SECONDS
    shabbat_geoname_id: str = SHABBAT_GEONAME_ID
    extra: dict = field(default_factory=dict)
