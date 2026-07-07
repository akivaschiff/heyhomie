"""Live-Claude eval: runs the real brain (real Anthropic calls, real tools) over
every flow in spec §6 against a FakeChannel, asserting the observable outcome.
External services (web search, page fetch, recipe extraction, Shabbat) are faked
so the only network dependency is Anthropic. Skips if ANTHROPIC_API_KEY is unset.

  cd pi && .venv/bin/python -m pytest tests/test_flows_live.py -q
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from homie.brain import Brain  # noqa: E402
from homie.channels.base import Surface  # noqa: E402
from homie.channels.fake import FakeChannel  # noqa: E402
from homie.clock import FakeClock, FakeScheduler  # noqa: E402
from homie.config import Config  # noqa: E402
from homie.store import LIST, MEMORY, RECIPES, LocalFileStore  # noqa: E402
from homie.tools import all_tools  # noqa: E402
from homie.tools.base import ToolContext  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)

COOKIE_RECIPE = {
    "name": "Chocolate Chip Cookies",
    "ingredients": ["250g flour", "2 eggs", "200g butter", "150g sugar"],
    "steps": ["Cream butter and sugar", "Add eggs and flour", "Bake 12 min at 180C"],
}


def make_brain(tmp_path, surface=None, store=None):
    from anthropic import Anthropic

    clock = FakeClock(datetime(2026, 6, 24, 16, 0, 0))
    scheduler = FakeScheduler(clock)
    store = store or LocalFileStore(tmp_path)
    channel = FakeChannel(surface or Surface(has_screen=True, has_speaker=True))
    ctx = ToolContext(
        store=store,
        channel=channel,
        scheduler=scheduler,
        clock=clock,
        config=Config(),
        web_search=lambda q: [{"url": "https://x/padthai", "title": "Authentic Pad Thai"}],
        fetch_url=lambda url: "raw recipe page",
        recipe_extractor=lambda url, raw: dict(COOKIE_RECIPE),
        shabbat_times=lambda g: {
            "location": "Jerusalem",
            "rows": [("Candle lighting", "7:12pm"), ("Havdalah", "8:25pm")],
        },
        session={},
    )
    ctx.pushed = []
    ctx.push = lambda name, rendered: (ctx.pushed.append((name, rendered)) or True)
    brain = Brain(Anthropic(), all_tools(), ctx, Config(), clock)
    return brain, channel, store, scheduler


def test_flow1_easy_add(tmp_path):
    brain, _, store, _ = make_brain(tmp_path)
    brain.handle("we finished the garbage bags")
    assert any("garbage" in i.lower() for i in store.lines(LIST))


def test_flow2_add_with_disambiguation(tmp_path):
    brain, _, store, _ = make_brain(tmp_path)
    brain.handle("I need more apples")
    # model should ask which kind before committing a bare "apples"
    brain.handle("green ones")
    items = " ".join(store.lines(LIST)).lower()
    assert "green" in items and "apple" in items


def test_flow3_show_and_send_list(tmp_path):
    brain, channel, store, _ = make_brain(tmp_path)
    store.set_lines(LIST, ["milk", "eggs"])
    brain.handle("show me the list")
    assert channel.shown
    brain.handle("send it to telegram")
    assert any(name == "telegram" for name, _ in brain.ctx.pushed)


def test_flow4_concurrent_timers(tmp_path):
    brain, channel, _, scheduler = make_brain(tmp_path)
    brain.handle("set a timer for the cake in the oven for 30 minutes, and another to flip pancakes in 30 seconds")
    scheduler.advance(30)
    assert any("pancake" in a.lower() for a in channel.announced)
    scheduler.advance(30 * 60)
    assert any("cake" in a.lower() for a in channel.announced)


def test_flow5_clock_reminder(tmp_path):
    brain, channel, _, scheduler = make_brain(tmp_path)
    brain.handle("set a reminder for 17:20 to prepare the dough")
    scheduler.advance(90 * 60)
    assert any("dough" in a.lower() for a in channel.announced)


def test_flow6_curated_recipe(tmp_path):
    brain, channel, store, _ = make_brain(tmp_path)
    store.write(RECIPES, "Chocolate Chip Cookies | https://example.com/cookies\n")
    brain.handle("show me the chocolate chip cookies from my recipes")
    assert channel.shown
    assert any("cookie" in (r.title or "").lower() for r in channel.shown)


def test_flow7_search_recipe(tmp_path):
    brain, channel, _, _ = make_brain(tmp_path)
    brain.handle("find me a recipe for pad thai")
    assert channel.shown


def test_flow8_handsfree_followup_from_context(tmp_path):
    brain, channel, store, _ = make_brain(tmp_path)
    store.write(RECIPES, "Chocolate Chip Cookies | https://example.com/cookies\n")
    brain.handle("load the chocolate chip cookies recipe")
    reply = brain.handle("how much flour does it need?")
    assert "250" in reply or "flour" in reply.lower()


def test_flow9_scale_and_redisplay(tmp_path):
    brain, channel, store, _ = make_brain(tmp_path)
    store.write(RECIPES, "Chocolate Chip Cookies | https://example.com/cookies\n")
    brain.handle("load the chocolate chip cookies recipe")
    before = len(channel.shown)
    brain.handle("make it for 6 people and show it again")
    assert len(channel.shown) > before


def test_flow10_memory_roundtrip(tmp_path):
    brain, _, store, _ = make_brain(tmp_path)
    brain.handle("remember that the spare key is in the shed")
    assert store.lines(MEMORY)
    reply = brain.handle("where is the spare key?")
    assert "shed" in reply.lower()


def test_flow11_cross_channel_list(tmp_path):
    shared = LocalFileStore(tmp_path)
    voice_brain, _, _, _ = make_brain(tmp_path, surface=Surface(has_speaker=True), store=shared)
    tg_brain, tg_channel, _, _ = make_brain(
        tmp_path, surface=Surface(is_chat=True), store=shared
    )
    voice_brain.handle("add olive oil to the shopping list")
    tg_brain.handle("what's on the shopping list?")
    seen = " ".join(tg_channel.chats + tg_channel.delivered).lower()
    assert "olive oil" in seen


def test_flow12_shabbat(tmp_path):
    brain, channel, _, _ = make_brain(tmp_path)
    brain.handle("show me the shabbat times")
    assert channel.shown
    assert any("shabbat" in (r.title or "").lower() for r in channel.shown)
