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

    def listening_open(self) -> None:
        """Rising two-tone: the mic is now open and it's the user's turn to speak.
        Fires both after the wake word and when a follow-up window opens."""
        _play_chime(_chime_path("open", _OPEN_TONES))

    def listening_close(self) -> None:
        """Falling two-tone: the mic has closed. Fires when the turn ends without a
        follow-up, or a follow-up window times out in silence."""
        _play_chime(_chime_path("close", _CLOSE_TONES))


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


_OPEN_TONES = ((660.0, 0.09), (880.0, 0.11))  # rising: "your turn to speak"
_CLOSE_TONES = ((880.0, 0.09), (660.0, 0.11))  # falling: "mic closed"


def _chime_path(name: str, tones) -> str:
    """A short two-tone blip synthesized once with the stdlib (no numpy) and cached
    in /tmp, keyed by name and volume."""
    import math
    import wave

    volume = int(os.environ.get("HOMIE_CHIME_VOLUME", "3000"))
    path = f"/tmp/homie_{name}_chime_{volume}.wav"
    if not os.path.exists(path):
        rate = 16000
        samples = []
        for freq, dur in tones:
            n = int(rate * dur)
            for i in range(n):
                fade = min(1.0, i / (rate * 0.01), (n - i) / (rate * 0.02))
                samples.append(int(volume * fade * math.sin(2 * math.pi * freq * i / rate)))
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples))
    return path


def _play_chime(path: str) -> None:
    if _is_macos():
        player = ["afplay"]
    else:
        player = ["aplay", "-q", "-D", alsa_speaker_device()]
    subprocess.run(player + [path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
