"""Deepgram voice I/O — STT and TTS. Deepgram owns the speech layer; this module
is the thin REST binding the voice channel drives. Playback returns the subprocess
so the caller can kill it for barge-in."""

import os
import subprocess
import tempfile

import requests

STT_URL = "https://api.deepgram.com/v1/listen"
TTS_URL = "https://api.deepgram.com/v1/speak"


class DeepgramVoice:
    def __init__(self, stt_model="nova-2", tts_model="aura-2-thalia-en"):
        self.api_key = os.environ["DEEPGRAM_API_KEY"]
        self.stt_model = stt_model
        self.tts_model = tts_model

    def transcribe(self, wav_bytes: bytes) -> str:
        resp = requests.post(
            STT_URL,
            params={"model": self.stt_model, "smart_format": "true", "language": "en"},
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"},
            data=wav_bytes,
            timeout=30,
        )
        resp.raise_for_status()
        alts = resp.json()["results"]["channels"][0]["alternatives"]
        return alts[0]["transcript"].strip() if alts else ""

    def synthesize(self, text: str) -> bytes:
        resp = requests.post(
            TTS_URL,
            params={"model": self.tts_model},
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"},
            json={"text": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.content

    def speak(self, text: str) -> None:
        """Synthesize and play, blocking until done."""
        proc = self.play(text)
        if proc:
            proc.wait()

    def play(self, text: str):
        """Synthesize and start playback; return the process (for barge-in)."""
        if not text:
            return None
        audio = self.synthesize(text)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio)
            path = f.name
        return subprocess.Popen(
            mp3_player_cmd() + [path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


def _is_macos() -> bool:
    return os.uname().sysname == "Darwin"


def alsa_speaker_device() -> str:
    """The ALSA output for the kitchen speaker (the Anker), not the ALSA default
    (which on the Pi is the silent headphone jack). Override: HOMIE_SPEAKER_DEVICE."""
    override = os.environ.get("HOMIE_SPEAKER_DEVICE")
    if override:
        return override
    try:
        for line in open("/proc/asound/cards"):
            if "S330" in line or "Anker" in line:
                card = line.strip().split(" ")[0]
                return f"plughw:{card},0"
    except Exception:
        pass
    return "default"


def mp3_player_cmd() -> list:
    if _is_macos():
        return ["afplay"]
    return ["mpg123", "-q", "-a", alsa_speaker_device()]
