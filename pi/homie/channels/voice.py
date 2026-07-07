"""Voice channel — Deepgram STT/TTS over the shared core. Two entry modes:

  run()      Pi kitchen: wake word (Porcupine) -> record a turn -> STT -> brain
             -> TTS on the kitchen speaker. A short listening window keeps
             follow-ups going without re-waking.
  run_ptt()  Mac dev: push-to-talk (no Porcupine, no wake-word model). Press Enter
             -> record a turn -> same pipeline. This is the spec's Mac hotkey mode.

Both share the record -> STT -> brain -> TTS path; only turn-triggering differs.
The screen surface is the tablet on the Pi (HTML written to a kiosk file) and the
browser on a Mac. Heavy device deps (pvporcupine, pvrecorder) are imported inside
the run methods so the module imports cleanly anywhere — keeping the core portable.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

from homie.audio.processing import pcm_to_wav
from homie.channels.base import Channel, Rendered, Surface
from homie.services.fillers import FillerBank
from homie.services.voice import DeepgramVoice

MIC_NAME_HINTS = ("anker", "s330", "usb")
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.2
MIN_RECORDING = 1.0
MAX_RECORDING = 30
LISTENING_WINDOW = 8.0  # seconds to keep listening for a follow-up without re-waking
FOLLOWUP_SPEECH_FRAMES = 3  # ~100ms of sustained sound to open a follow-up (not a click)


class VoiceChannel(Channel):
    name = "voice"

    def __init__(self, tablet_html_path: str = None):
        self.surface = Surface(has_screen=True, has_speaker=True, is_chat=False)
        self.voice = DeepgramVoice()
        self.tablet_path = Path(
            tablet_html_path or os.environ.get("HOMIE_TABLET_HTML", "/tmp/homie_screen.html")
        )
        self.recorder = None
        self.porcupine = None
        self.frame_length = 512  # set from Porcupine in wake mode; fixed in PTT mode
        self.silence_threshold = SILENCE_THRESHOLD  # recalibrated from ambient at startup
        self._current_playback = None
        self._filler_proc = None
        self.fillers = FillerBank(self.voice)
        self.fillers.prewarm()

    # --- output surfaces ---
    def deliver(self, text: str) -> None:
        self.say(text)

    def say(self, text: str) -> None:
        if not text:
            return
        text = _strip_markdown(text)
        if self._filler_proc is not None:  # let the acknowledgment finish first
            try:
                self._filler_proc.wait()
            except Exception:
                pass
            self._filler_proc = None
        self._current_playback = self.voice.play(text)
        if self._current_playback:
            self._current_playback.wait()
            self._current_playback = None

    def announce(self, text: str) -> None:
        print(f"📢 announce: {text}")
        self.voice.speak(_strip_markdown(text))

    def show_screen(self, rendered: Rendered) -> None:
        content = rendered.html or rendered.chat_text()
        if _is_macos():  # dev: pop it in the browser like the Mac harness does
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
                    f.write(content)
                    path = f.name
                subprocess.run(["open", path], check=False)
            except Exception as exc:
                print(f"screen render failed: {exc}")
            return
        try:
            self.tablet_path.write_text(content)
        except Exception as exc:
            print(f"tablet render failed: {exc}")

    def send_chat(self, text: str) -> None:
        self.say(text)

    # --- input loop ---
    def run(self, brain) -> None:
        import pvporcupine
        from pvrecorder import PvRecorder

        access_key = os.environ["PORCUPINE_ACCESS_KEY"]
        keyword_path = os.environ.get(
            "WAKE_WORD_PATH", str(Path(__file__).resolve().parent.parent.parent / "hey-homie.ppn")
        )
        self.porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path])
        self.frame_length = self.porcupine.frame_length
        device_index = _select_mic(PvRecorder)
        print(f"Using mic device index {device_index}")
        self.recorder = PvRecorder(device_index=device_index, frame_length=self.frame_length)
        self.recorder.start()
        self._calibrate()
        print("Listening for wake word…")

        try:
            while True:
                pcm = self.recorder.read()
                if self.porcupine.process(pcm) >= 0:
                    print("🎤 wake")
                    self._converse(brain)
                    print("Listening for wake word…")
        except KeyboardInterrupt:
            pass
        finally:
            self.recorder.delete()
            self.porcupine.delete()

    def run_ptt(self, brain) -> None:
        """Push-to-talk loop for the Mac (no wake word). Press Enter to open a
        conversation; it then behaves exactly like after the wake word on the Pi —
        answers, and keeps listening for follow-ups on its own for a short window
        before falling back to the Enter prompt. Same _converse() path as wake mode."""
        from pvrecorder import PvRecorder

        self.frame_length = 512
        device_index = _select_mic(PvRecorder)
        print(f"Using mic device index {device_index}")
        self.recorder = PvRecorder(device_index=device_index, frame_length=self.frame_length)
        self.recorder.start()
        self._calibrate()
        self.recorder.stop()
        print("Push-to-talk ready. Press Enter, ask, then pause. It keeps listening for")
        print("follow-ups on its own. Ctrl+C to quit.")

        try:
            while True:
                try:
                    input()
                except EOFError:
                    break
                print("🎤 listening… speak, then pause")
                self.recorder.start()
                self._converse(brain)
                self.recorder.stop()
                print("\nPress Enter to talk again.")
        except KeyboardInterrupt:
            pass
        finally:
            self.recorder.delete()

    def _converse(self, brain) -> None:
        """Handle one wake: a turn, then a listening window for follow-ups."""
        prefill = []
        while True:
            wav = self._record_turn(prefill)
            prefill = []
            if not wav:
                return
            transcript = self.voice.transcribe(wav)
            if not transcript:
                return
            print(f"   heard: {transcript}")
            self._filler_proc = self.fillers.play()  # instant "on it" while the brain works
            brain.handle(transcript)
            self._flush_recorder()  # drop audio buffered during TTS so we don't hear ourselves
            prefill = self._await_followup()
            if prefill is None:
                return

    def _record_turn(self, prefill: list = None) -> bytes:
        frames = list(prefill or [])  # carry the frame that re-opened the turn
        fps = SAMPLE_RATE // self.frame_length
        silence_frames = 0
        silence_limit = int(SILENCE_DURATION * fps)
        min_frames = int(MIN_RECORDING * fps)
        peak = 0
        second_peaks = []
        current_second_peak = 0
        for i in range(int(MAX_RECORDING * fps)):
            pcm = self.recorder.read()
            frames.extend(pcm)
            amplitude = max((abs(s) for s in pcm), default=0)
            peak = max(peak, amplitude)
            current_second_peak = max(current_second_peak, amplitude)
            if (i + 1) % fps == 0:
                second_peaks.append(current_second_peak)
                current_second_peak = 0
            silence_frames = silence_frames + 1 if amplitude < self.silence_threshold else 0
            if i >= min_frames and silence_frames >= silence_limit:
                break
        print(
            f"   (recorded {len(frames) / SAMPLE_RATE:.1f}s, peak {peak}, "
            f"threshold {self.silence_threshold}, per-second {second_peaks})"
        )
        if len(frames) < SAMPLE_RATE // 2:
            return b""
        return pcm_to_wav(frames, SAMPLE_RATE)

    def _calibrate(self) -> None:
        """Measure the room's ambient noise and set the silence threshold above it.
        A fixed threshold can sit below a laptop's noise floor, in which case
        'silence' is never detected and recordings run to the max. Override with
        HOMIE_SILENCE_THRESHOLD if the auto value misbehaves."""
        override = os.environ.get("HOMIE_SILENCE_THRESHOLD")
        if override:
            self.silence_threshold = int(override)
            print(f"   silence threshold (env): {self.silence_threshold}")
            return
        fps = SAMPLE_RATE // self.frame_length
        peaks = []
        for _ in range(int(0.5 * fps)):
            pcm = self.recorder.read()
            peaks.append(max((abs(s) for s in pcm), default=0))
        ambient = sorted(peaks)[len(peaks) // 2]
        # clamp: the S330's AGC makes ambient samples swing wildly; close speech
        # peaks >10k, so anything above ~2500 would start rejecting real commands
        self.silence_threshold = min(2500, max(SILENCE_THRESHOLD, int(ambient * 2.5)))
        print(f"   mic calibrated: ambient peak ~{ambient}, silence threshold {self.silence_threshold}")

    def _flush_recorder(self) -> None:
        """Clear frames buffered while TTS was playing, so the follow-up window
        doesn't trigger on the assistant's own voice echoing into the mic."""
        try:
            self.recorder.stop()
            self.recorder.start()
        except Exception:
            pass

    def _await_followup(self):
        """Listen for the start of speech within the window. Requires a short burst
        of sustained sound (not a single click/pop) to open a follow-up turn.
        Returns the buffered speech frames (so the first word isn't clipped), or
        None if the window closes in silence."""
        fps = SAMPLE_RATE // self.frame_length
        print(f"   🎧 listening for follow-up ({int(LISTENING_WINDOW)}s)…")
        consecutive = 0
        buffered = []
        for _ in range(int(LISTENING_WINDOW * fps)):
            pcm = self.recorder.read()
            if max((abs(s) for s in pcm), default=0) >= self.silence_threshold:
                consecutive += 1
                buffered.extend(pcm)
                if consecutive >= FOLLOWUP_SPEECH_FRAMES:
                    return buffered
            else:
                consecutive = 0
                buffered = []
        print("   window closed")
        return None


def _is_macos() -> bool:
    return os.uname().sysname == "Darwin"


def _strip_markdown(text: str) -> str:
    """Last line of defense before TTS: markdown that slips through the prompt gets
    read out literally ('star star'), so flatten it to plain speech."""
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"\n{2,}", ". ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _select_mic(pv_recorder_cls) -> int:
    """Pick the mic device. HOMIE_MIC_DEVICE overrides; otherwise match a USB/Anker
    device by name; otherwise fall back to PvRecorder's default (-1)."""
    override = os.environ.get("HOMIE_MIC_DEVICE")
    if override is not None:
        return int(override)
    try:
        devices = pv_recorder_cls.get_available_devices()
    except Exception:
        return -1
    for idx, name in enumerate(devices):
        if any(hint in name.lower() for hint in MIC_NAME_HINTS):
            return idx
    return -1
