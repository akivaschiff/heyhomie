"""Tool layer — the durable abstraction. Channels are executors; tools are the
contract the model orchestrates over.

A Tool is a name + Anthropic input schema + handler. The handler receives the
parsed input and a ToolContext (shared store, the active channel's surface,
scheduler/clock, web search) and returns a short text result the model reads back.
"""

from dataclasses import dataclass
from typing import Callable

from homie.channels.base import Channel
from homie.clock import Clock, Scheduler
from homie.config import Config
from homie.store import Store


@dataclass
class ToolContext:
    store: Store
    channel: Channel
    scheduler: Scheduler
    clock: Clock
    config: Config
    web_search: Callable[[str], list[dict]] = None
    fetch_url: Callable[[str], str] = None
    # (source_url, raw_page_text) -> {"name", "ingredients": [...], "steps": [...]}
    recipe_extractor: Callable[[str, str], dict] = None
    # geoname_id -> {"location": str, "rows": [(label, value), ...]}
    shabbat_times: Callable[[str], dict] = None
    # SmartHomeClient (services/smarthome.py) — lights/blinds/AC via the home server
    smarthome: object = None
    # CronStore (services/cron.py) — persistent scheduled home actions
    cron: object = None
    # system speaker volume with .get() -> percent and .set(percent) (services/volume.py)
    volume: object = None
    # ShufersalCart (services/shufersal.py) — resolve a term to a SKU and sync it to
    # the live supermarket cart. None when no cookie jar (tests, Mac harness) — the
    # list still works without it, preserving the portability guarantee.
    shufersal: object = None
    # deliver a rendered payload to another channel out of band (e.g. "send to Telegram")
    push: Callable[[str, object], bool] = None
    # mutable scratchpad shared across a session (e.g. the loaded recipe)
    session: dict = None


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict, ToolContext], str]

    def anthropic_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
