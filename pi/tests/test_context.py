"""Conversation window: turn-atomic, time-bounded, tool-pairing-safe."""

from datetime import datetime

from homie.clock import FakeClock
from homie.context import ConversationContext


def _advance(clock, seconds):
    clock._set_mono(clock.monotonic() + seconds)


def test_old_turns_drop_recent_kept():
    clock = FakeClock(datetime(2026, 6, 24, 16, 0, 0))
    ctx = ConversationContext(window_seconds=15 * 60, clock=clock)

    ctx.start_turn("first")
    _advance(clock, 20 * 60)  # 20 min later, past the 15-min window
    ctx.start_turn("second")

    texts = [m["content"] for m in ctx.get() if m["role"] == "user"]
    assert "first" not in texts
    assert "second" in texts


def test_turn_atomic_preserves_tool_pairing():
    clock = FakeClock(datetime(2026, 6, 24, 16, 0, 0))
    ctx = ConversationContext(window_seconds=15 * 60, clock=clock)

    # an old turn that contained a tool_use/tool_result pair
    ctx.start_turn("set a timer")
    ctx.add("assistant", [{"type": "tool_use", "id": "t1", "name": "timer_set", "input": {}}])
    ctx.add("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}])
    ctx.add("assistant", [{"type": "text", "text": "done"}])

    _advance(clock, 20 * 60)
    ctx.start_turn("what's on the list")

    msgs = ctx.get()
    # the whole old turn is gone — no orphaned tool_result, no dangling tool_use
    assert all(
        not (isinstance(m["content"], list) and m["content"][0].get("type") == "tool_result")
        for m in msgs
    )
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "what's on the list"


def test_within_window_everything_retained():
    clock = FakeClock(datetime(2026, 6, 24, 16, 0, 0))
    ctx = ConversationContext(window_seconds=15 * 60, clock=clock)
    ctx.start_turn("a")
    _advance(clock, 5 * 60)
    ctx.start_turn("b")
    _advance(clock, 5 * 60)
    ctx.start_turn("c")
    texts = [m["content"] for m in ctx.get() if m["role"] == "user"]
    assert texts == ["a", "b", "c"]
