"""Voice UX: filler rotation and the markdown-to-speech guard."""

from datetime import datetime

from homie.brain import build_system_prompt
from homie.channels.base import Surface
from homie.channels.fake import FakeChannel
from homie.channels.voice import _strip_markdown
from homie.clock import FakeClock
from homie.services.fillers import FillerBank


class _NoSynthVoice:
    tts_model = "test"

    def synthesize(self, text):
        return b"mp3"


def _bank(tmp_path):
    return FillerBank(_NoSynthVoice(), cache_dir=tmp_path, phrases=["a", "b", "c", "d"])


def test_filler_bag_uses_every_phrase_before_repeating(tmp_path):
    bank = _bank(tmp_path)
    first_cycle = {bank.next_phrase() for _ in range(4)}
    assert first_cycle == {"a", "b", "c", "d"}


def test_filler_no_immediate_repeat_across_cycles(tmp_path):
    bank = _bank(tmp_path)
    seq = [bank.next_phrase() for _ in range(20)]
    assert all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))


def test_filler_synth_cached_once(tmp_path):
    calls = []

    class CountingVoice(_NoSynthVoice):
        def synthesize(self, text):
            calls.append(text)
            return b"mp3"

    bank = FillerBank(CountingVoice(), cache_dir=tmp_path, phrases=["hello"])
    bank._ensure("hello")
    bank._ensure("hello")
    assert calls == ["hello"]


def test_strip_markdown_flattens_bold_and_bullets():
    text = "**Shopping & Lists:** stuff\n\n- milk\n- eggs\n`code` and _italics_ and # heading"
    out = _strip_markdown(text)
    assert "*" not in out and "`" not in out and "#" not in out and "_" not in out
    assert "Shopping & Lists" in out
    assert "milk" in out and "eggs" in out


def test_speaker_prompt_forbids_markdown():
    channel = FakeChannel(Surface(has_screen=True, has_speaker=True))
    prompt = build_system_prompt(channel, FakeClock(datetime(2026, 7, 7, 12, 0)))
    assert "spoken aloud" in prompt
    assert "asterisks" in prompt


def test_chat_prompt_has_no_tts_rules():
    channel = FakeChannel(Surface(is_chat=True))
    prompt = build_system_prompt(channel, FakeClock(datetime(2026, 7, 7, 12, 0)))
    assert "asterisks" not in prompt
