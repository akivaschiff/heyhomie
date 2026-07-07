"""Deterministic brain tool-loop tests with a scripted fake Anthropic client.
Covers the message-pairing invariants the real API enforces (no orphaned
tool_result, no consecutive user messages, valid history after errors) without
any network."""

from datetime import datetime

import pytest

from homie.brain import Brain
from homie.channels.base import Surface
from homie.channels.fake import FakeChannel
from homie.clock import FakeClock, FakeScheduler
from homie.config import Config
from homie.store import LocalFileStore
from homie.tools.base import Tool, ToolContext


class _Block:
    def __init__(self, d):
        self._d = d
        self.type = d["type"]

    def __getattr__(self, key):
        try:
            return self._d[key]
        except KeyError:
            raise AttributeError(key)

    def model_dump(self):
        return dict(self._d)


class _Usage:
    input_tokens = 10
    output_tokens = 5


class _Resp:
    def __init__(self, blocks, stop_reason):
        self.content = [_Block(b) for b in blocks]
        self.stop_reason = stop_reason
        self.usage = _Usage()


def _text(t):
    return {"type": "text", "text": t}


def _tool(id, name="echo", inp=None):
    return {"type": "tool_use", "id": id, "name": name, "input": inp or {}}


class FakeClient:
    """Replays scripted responses; optionally raises on the Nth call."""

    def __init__(self, responses, raise_on=None):
        self.responses = responses
        self.raise_on = raise_on
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        if self.raise_on and self.calls == self.raise_on:
            raise RuntimeError("simulated API failure")
        return self.responses[self.calls - 1]


def make_brain(tmp_path, responses, raise_on=None):
    clock = FakeClock(datetime(2026, 6, 24, 16, 0, 0))
    calls = []
    echo = Tool(
        name="echo",
        description="echo",
        input_schema={"type": "object", "properties": {}},
        handler=lambda args, ctx: calls.append(args) or '{"ok": true}',
    )
    channel = FakeChannel(Surface(has_screen=True, has_speaker=True))
    ctx = ToolContext(
        store=LocalFileStore(tmp_path),
        channel=channel,
        scheduler=FakeScheduler(clock),
        clock=clock,
        config=Config(),
        session={},
    )
    brain = Brain(FakeClient(responses, raise_on), [echo], ctx, Config(), clock)
    return brain, channel, calls


def roles(brain):
    return [m["role"] for m in brain.conversation.get()]


def no_consecutive_user(brain):
    r = roles(brain)
    return all(not (r[i] == r[i + 1] == "user") for i in range(len(r) - 1))


def test_text_only_turn(tmp_path):
    brain, channel, _ = make_brain(tmp_path, [_Resp([_text("hi there")], "end_turn")])
    assert brain.handle("hello") == "hi there"
    assert channel.delivered == ["hi there"]
    assert roles(brain)[-1] == "assistant"


def test_tool_then_text(tmp_path):
    brain, channel, calls = make_brain(
        tmp_path,
        [_Resp([_tool("t1")], "tool_use"), _Resp([_text("done")], "end_turn")],
    )
    assert brain.handle("do it") == "done"
    assert len(calls) == 1
    assert roles(brain) == ["user", "assistant", "user", "assistant"]
    assert no_consecutive_user(brain)


def test_tool_use_stop_reason_with_no_tool_blocks_is_final(tmp_path):
    # stop_reason lies; there are no actual tool_use blocks -> must finalize, not 400
    brain, channel, _ = make_brain(tmp_path, [_Resp([_text("never mind")], "tool_use")])
    assert brain.handle("hmm") == "never mind"
    assert brain.client.calls == 1
    assert no_consecutive_user(brain)


def test_two_turns_no_consecutive_user(tmp_path):
    brain, _, _ = make_brain(
        tmp_path, [_Resp([_text("a")], "end_turn"), _Resp([_text("b")], "end_turn")]
    )
    brain.handle("one")
    brain.handle("two")
    assert no_consecutive_user(brain)


def test_max_iterations_fallback_ends_on_assistant(tmp_path):
    brain, channel, _ = make_brain(tmp_path, [_Resp([_tool(f"t{i}")], "tool_use") for i in range(20)])
    out = brain.handle("loop forever")
    assert "stuck" in out.lower()
    assert roles(brain)[-1] == "assistant"
    assert no_consecutive_user(brain)


def test_exception_recovers_and_next_turn_is_valid(tmp_path):
    brain, channel, _ = make_brain(
        tmp_path,
        [_Resp([_text("won't reach")], "end_turn"), _Resp([_text("recovered")], "end_turn")],
        raise_on=1,
    )
    first = brain.handle("trigger error")
    assert "wrong" in first.lower()
    assert roles(brain)[-1] == "assistant"
    second = brain.handle("try again")
    assert second == "recovered"
    assert no_consecutive_user(brain)
