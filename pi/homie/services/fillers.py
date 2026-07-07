"""Instant acknowledgment phrases for the voice channel.

The gap between hearing the user and speaking the answer (STT + Claude + TTS) is
a few seconds of dead air; a short spoken acknowledgment starts immediately while
the brain works. Phrases rotate through a shuffle bag so they don't repeat until
the whole bank has been used. Audio is synthesized once and cached on disk, so
playback after the first use costs nothing and starts instantly.
"""

import hashlib
import random
import subprocess
import threading
from pathlib import Path

FILLER_PHRASES = [
    "Let me look into that.",
    "Checking.",
    "Hang on, I'm on it.",
    "One sec.",
    "On it.",
    "Give me a moment.",
    "Let me check.",
    "Just a second.",
    "Looking into it.",
    "Let me see.",
    "Sure, one moment.",
    "Right, checking now.",
    "Okay, let me look.",
    "Hold on.",
    "Let me find out.",
    "Working on it.",
    "Just a moment.",
    "Okay, on it.",
    "Let me have a look.",
    "Checking that for you.",
    "One moment.",
    "Sure, checking.",
    "Right away.",
    "Let me dig that up.",
]


class FillerBank:
    def __init__(self, voice, cache_dir=None, phrases=None):
        self.voice = voice
        self.phrases = list(phrases or FILLER_PHRASES)
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "homie" / "fillers")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._bag: list[str] = []
        self._last: str = None

    def next_phrase(self) -> str:
        if not self._bag:
            self._bag = list(self.phrases)
            random.shuffle(self._bag)
            if len(self._bag) > 1 and self._bag[-1] == self._last:
                self._bag[0], self._bag[-1] = self._bag[-1], self._bag[0]
        self._last = self._bag.pop()
        return self._last

    def _cached_path(self, phrase: str) -> Path:
        key = hashlib.sha1(f"{self.voice.tts_model}:{phrase}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{key}.mp3"

    def _ensure(self, phrase: str) -> Path:
        path = self._cached_path(phrase)
        if not path.exists():
            path.write_bytes(self.voice.synthesize(phrase))
        return path

    def play(self):
        """Start playing the next filler without blocking. Returns the process
        (wait on it before speaking the real answer), or None on any failure —
        a filler must never break the turn."""
        try:
            from homie.services.voice import mp3_player_cmd

            path = self._ensure(self.next_phrase())
            return subprocess.Popen(
                mp3_player_cmd() + [str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            return None

    def prewarm(self):
        """Synthesize the whole bank into the cache in the background."""

        def work():
            for phrase in self.phrases:
                try:
                    self._ensure(phrase)
                except Exception:
                    return

        threading.Thread(target=work, daemon=True).start()
