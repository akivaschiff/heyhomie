"""Render-capability routing (spec §3, load-bearing): a Rendered payload must land
on exactly one surface per channel profile, never double-emitted."""

from homie.channels.base import Rendered, Surface
from homie.channels.fake import FakeChannel


def _payload():
    return Rendered(title="List", speech="say it", html="<h1>x</h1>", text="text form")


def test_screen_plus_speaker_shows_and_speaks_once():
    ch = FakeChannel(Surface(has_screen=True, has_speaker=True))
    ch.render(_payload())
    assert len(ch.shown) == 1
    assert ch.spoken == ["say it"]
    assert ch.chats == []


def test_chat_only_sends_text_no_speech_no_screen():
    ch = FakeChannel(Surface(is_chat=True))
    ch.render(_payload())
    assert ch.chats == ["text form"]
    assert ch.shown == []
    assert ch.spoken == []


def test_speaker_only_speaks_exactly_once():
    ch = FakeChannel(Surface(has_speaker=True))
    ch.render(_payload())
    assert ch.spoken == ["say it"]
    assert ch.shown == []
    assert ch.chats == []
    assert ch.delivered == []  # not also delivered as a turn reply


def test_headless_falls_back_to_deliver():
    ch = FakeChannel(Surface())
    ch.render(_payload())
    assert ch.delivered == ["say it"]
    assert ch.shown == [] and ch.chats == [] and ch.spoken == []
