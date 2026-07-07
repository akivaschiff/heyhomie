"""Deterministic tool tests — no network. Direct handler calls against a local
store, fake channel, and a manually-advanced clock. Asserts observable state and
the surface interactions each tool is supposed to produce."""

import json

from homie.store import LIST, MEMORY, RECIPES
from homie.tools.memory import _query as memory_query
from homie.tools.memory import _save as memory_save
from homie.tools.recipes import _load as recipe_load
from homie.tools.reminders import _set as reminder_set
from homie.tools.shabbat import _show as shabbat_show
from homie.tools.shopping_list import _add, _flat, _parse, _remove, _show
from homie.tools.timers import _cancel as timer_cancel
from homie.tools.timers import _list as timer_list
from homie.tools.timers import _set as timer_set


# --- list ---

def _items(ctx):
    return _flat(_parse(ctx.store.read(LIST)))[0]


def test_list_add_appears_in_store(ctx):
    _add({"item": "green apples", "category": "Fruits"}, ctx)
    assert _items(ctx) == ["green apples"]
    assert ctx.store.read(LIST) == "# Fruits\ngreen apples\n"


def test_list_add_groups_by_section_order(ctx):
    _add({"item": "milk", "category": "Dairy"}, ctx)
    _add({"item": "bananas", "category": "Fruits"}, ctx)
    assert _items(ctx) == ["bananas", "milk"]  # Fruits sorts before Dairy


def test_list_remove_last(ctx):
    _add({"item": "milk", "category": "Dairy"}, ctx)
    _add({"item": "bread", "category": "Everything else"}, ctx)
    out = json.loads(_remove({"item": "last"}, ctx))
    assert out["removed"] == "bread"
    assert _items(ctx) == ["milk"]


def test_list_remove_by_match_and_correct(ctx):
    _add({"item": "red apples", "category": "Fruits"}, ctx)
    _remove({"item": "red"}, ctx)
    _add({"item": "green apples", "category": "Fruits"}, ctx)
    assert _items(ctx) == ["green apples"]


def test_list_show_renders_to_screen(ctx, channel):
    _add({"item": "eggs", "category": "Dairy"}, ctx)
    _show({}, ctx)
    assert channel.shown, "list should render on the screen surface"
    assert "eggs" in channel.shown[-1].structured["items"]


def test_list_remove_word_boundary(ctx):
    _add({"item": "shredded wheat", "category": "Everything else"}, ctx)
    _add({"item": "red apples", "category": "Fruits"}, ctx)
    out = json.loads(_remove({"item": "red"}, ctx))
    assert out["removed"] == "red apples"
    assert _items(ctx) == ["shredded wheat"]


def test_list_show_target_telegram_uses_push(ctx, channel):
    sent = {}
    ctx.push = lambda name, rendered: sent.setdefault("to", name) or True
    _add({"item": "eggs", "category": "Dairy"}, ctx)
    out = json.loads(_show({"target": "telegram"}, ctx))
    assert out["sent_to"] == "telegram"
    assert not channel.shown  # delivered elsewhere, not on this screen


# --- timers ---

def test_concurrent_named_timers_fire_independently(ctx, channel, scheduler):
    timer_set({"name": "flip pancakes", "seconds": 30}, ctx)
    timer_set({"name": "cake in the oven", "seconds": 1800}, ctx)
    assert json.loads(timer_list({}, ctx))["count"] == 2

    scheduler.advance(30)
    assert any("flip pancakes" in a for a in channel.announced)
    assert not any("cake in the oven" in a for a in channel.announced)

    scheduler.advance(1800)
    assert any("cake in the oven" in a for a in channel.announced)


def test_timer_cancel(ctx, channel, scheduler):
    timer_set({"name": "x", "seconds": 10}, ctx)
    timer_cancel({"name": "x"}, ctx)
    scheduler.advance(60)
    assert not channel.announced


# --- reminders ---

def test_reminder_at_clock_time_fires_on_speaker(ctx, channel, scheduler):
    # clock starts 16:00; 17:20 is 80 minutes away
    out = json.loads(reminder_set({"text": "prepare dough", "at": "17:20"}, ctx))
    assert out["in_seconds"] == 80 * 60
    scheduler.advance(80 * 60)
    assert any("prepare dough" in a for a in channel.announced)


def test_reminder_duration(ctx, channel, scheduler):
    reminder_set({"text": "check oven", "seconds": 120}, ctx)
    scheduler.advance(120)
    assert any("check oven" in a for a in channel.announced)


def test_reminder_past_clock_time_rolls_to_next_day(ctx):
    # clock starts 16:00; 08:00 has passed -> should be ~16h away, not negative
    out = json.loads(reminder_set({"text": "morning meds", "at": "08:00"}, ctx))
    assert out["in_seconds"] == 16 * 3600


# --- memory ---

def test_memory_roundtrip(ctx):
    memory_save({"fact": "the spare key is in the shed"}, ctx)
    out = json.loads(memory_query({"query": "spare key"}, ctx))
    assert any("shed" in f for f in out["facts"])


# --- recipes ---

def test_recipe_load_curated_renders_and_enters_context(ctx, channel):
    ctx.store.write(RECIPES, "Best Cookies | https://example.com/cookies\n")
    ctx.fetch_url = lambda url: "raw page"
    ctx.recipe_extractor = lambda url, raw: {
        "name": "Best Cookies",
        "ingredients": ["200g flour", "2 eggs"],
        "steps": ["mix", "bake"],
    }
    out = json.loads(recipe_load({"query": "cookies"}, ctx))
    assert out["loaded"] == "Best Cookies"
    assert out["source"] == "curated"
    assert channel.shown, "recipe should render on screen"
    assert ctx.session["recipe"]["name"] == "Best Cookies"  # available for follow-ups


def test_recipe_load_search_when_no_curated(ctx, channel):
    ctx.web_search = lambda q: [{"url": "https://x/y", "title": "Pad Thai"}]
    ctx.fetch_url = lambda url: "raw"
    ctx.recipe_extractor = lambda url, raw: {
        "name": "Pad Thai",
        "ingredients": ["rice noodles", "tofu"],
        "steps": ["soak", "fry"],
    }
    out = json.loads(recipe_load({"query": "pad thai", "prefer_curated": False}, ctx))
    assert out["source"] == "search"
    assert out["loaded"] == "Pad Thai"


def test_recipe_load_skips_empty_candidate(ctx):
    ctx.web_search = lambda q: [
        {"url": "https://js-only/page", "title": "Bad"},
        {"url": "https://good/page", "title": "Good Soup"},
    ]
    ctx.fetch_url = lambda url: "raw"
    ctx.recipe_extractor = lambda url, raw: (
        {"name": "", "ingredients": [], "steps": []}
        if "js-only" in url
        else {"name": "Good Soup", "ingredients": ["water"], "steps": ["boil"]}
    )
    out = json.loads(recipe_load({"query": "soup", "prefer_curated": False}, ctx))
    assert out["loaded"] == "Good Soup"


# --- shabbat ---

def test_shabbat_mode_renders(ctx, channel):
    ctx.shabbat_times = lambda geoname: {
        "location": "Jerusalem",
        "rows": [("Candle lighting", "7:12pm"), ("Havdalah", "8:25pm")],
    }
    shabbat_show({}, ctx)
    assert channel.shown
    assert "Jerusalem" in channel.shown[-1].title
