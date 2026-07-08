"""Persistent reminders: survive a 'restart' (fresh scheduler/session, same store),
fire recent missed ones on rearm, drop stale ones, and support list/cancel."""

import json
from datetime import datetime

from homie.channels.base import Surface
from homie.channels.fake import FakeChannel
from homie.clock import FakeClock, FakeScheduler
from homie.config import Config
from homie.store import REMINDERS
from homie.tools.base import ToolContext
from homie.tools.reminders import _cancel, _list, _set, rearm


def make_ctx(store, clock):
    return ToolContext(
        store=store,
        channel=FakeChannel(Surface(has_speaker=True)),
        scheduler=FakeScheduler(clock),
        clock=clock,
        config=Config(),
        session={},
    )


def test_fire_removes_line(ctx, channel, scheduler):
    _set({"text": "check oven", "seconds": 120}, ctx)
    assert ctx.store.lines(REMINDERS)
    scheduler.advance(120)
    assert any("check oven" in a for a in channel.announced)
    assert ctx.store.lines(REMINDERS) == []


def test_reminder_survives_restart(store):
    clock1 = FakeClock(datetime(2026, 7, 8, 10, 0, 0))
    ctx1 = make_ctx(store, clock1)
    _set({"text": "prepare dough", "seconds": 600}, ctx1)

    # "restart": same store, fresh clock/scheduler/session, 2 min later
    clock2 = FakeClock(datetime(2026, 7, 8, 10, 2, 0))
    ctx2 = make_ctx(store, clock2)
    rearm(ctx2)
    ctx2.scheduler.advance(8 * 60)
    assert any("prepare dough" in a for a in ctx2.channel.announced)
    assert store.lines(REMINDERS) == []


def test_recently_missed_fires_on_rearm(store):
    ctx1 = make_ctx(store, FakeClock(datetime(2026, 7, 8, 10, 0, 0)))
    _set({"text": "take out cake", "seconds": 60}, ctx1)

    # comes back 10 minutes late — within grace, should announce immediately
    ctx2 = make_ctx(store, FakeClock(datetime(2026, 7, 8, 10, 11, 0)))
    rearm(ctx2)
    assert any("take out cake" in a for a in ctx2.channel.announced)
    assert store.lines(REMINDERS) == []


def test_stale_missed_is_dropped_silently(store):
    ctx1 = make_ctx(store, FakeClock(datetime(2026, 7, 8, 10, 0, 0)))
    _set({"text": "morning meds", "seconds": 60}, ctx1)

    # comes back 5 hours late — stale, drop without announcing
    ctx2 = make_ctx(store, FakeClock(datetime(2026, 7, 8, 15, 0, 0)))
    rearm(ctx2)
    assert ctx2.channel.announced == []
    assert store.lines(REMINDERS) == []


def test_list_and_cancel_by_text(ctx, channel, scheduler):
    _set({"text": "prepare dough", "seconds": 600}, ctx)
    _set({"text": "call mom", "seconds": 1200}, ctx)
    listed = json.loads(_list({}, ctx))["reminders"]
    assert [r["text"] for r in listed] == ["prepare dough", "call mom"]
    assert listed[0]["minutes_away"] == 10

    out = json.loads(_cancel({"which": "dough"}, ctx))
    assert out["cancelled"] == "prepare dough"
    scheduler.advance(600)
    assert not any("dough" in a for a in channel.announced)
    scheduler.advance(600)
    assert any("call mom" in a for a in channel.announced)


def test_cancel_ambiguous_returns_candidates(ctx):
    _set({"text": "water the plants", "seconds": 600}, ctx)
    _set({"text": "water for pasta", "seconds": 900}, ctx)
    out = json.loads(_cancel({"which": "water"}, ctx))
    assert "ambiguous" in out and len(out["candidates"]) == 2
