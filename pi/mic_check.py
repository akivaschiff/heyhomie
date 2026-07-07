"""Record a few seconds from the mic, play it back, and report what STT heard.

    sudo systemctl stop homie      # the mic is exclusive
    venv/bin/python mic_check.py
    sudo systemctl start homie
"""

import subprocess
import time

from dotenv import load_dotenv

from homie.audio.processing import pcm_to_wav
from homie.channels.voice import _select_mic

load_dotenv(".env")

SECONDS = 5
WAV_PATH = "/tmp/mic_check.wav"


def anker_card() -> str:
    for line in open("/proc/asound/cards"):
        if "S330" in line:
            return line.split("[")[0].strip().split(" ")[0]
    return "0"


def main():
    import subprocess as sp

    from pvrecorder import PvRecorder

    if sp.run(["systemctl", "is-active", "--quiet", "homie"]).returncode == 0:
        print("✋ the homie service is holding the mic. Run first:")
        print("   sudo systemctl stop homie")
        return

    devices = PvRecorder.get_available_devices()
    print("audio capture devices:")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d}")
    if not any("s330" in d.lower() or "anker" in d.lower() for d in devices):
        print("✋ the Anker S330 is NOT in the device list — it's unplugged,")
        print("   on a bad USB port, or not powered. Fix that first.")
        return

    idx = _select_mic(PvRecorder)
    rec = PvRecorder(device_index=idx, frame_length=512)
    print(f"using mic device index {idx} — recording {SECONDS}s, SPEAK NOW…")
    rec.start()
    frames = []
    for i in range(31 * SECONDS):
        frames.extend(rec.read())
        if (i + 1) % 31 == 0:
            print(f"  …{(i + 1) // 31}s")
    rec.stop()
    rec.delete()

    peak = max(abs(s) for s in frames)
    open(WAV_PATH, "wb").write(pcm_to_wav(frames, 16000))
    print(f"saved {WAV_PATH} (peak {peak})")

    print("playing back through the Anker…")
    card = anker_card()
    subprocess.run(["aplay", "-q", "-D", f"plughw:{card},0", WAV_PATH])

    try:
        from homie.services.voice import DeepgramVoice

        text = DeepgramVoice().transcribe(open(WAV_PATH, "rb").read())
        print(f"STT heard: {text or '(nothing)'}")
    except Exception as exc:
        print(f"(STT skipped: {exc})")

    print("\nIf the playback is your voice, loud and clear -> mic is fine.")
    print("If it's silence, buzzing or beeps -> the Anker mic is muted/broken;")
    print("tap its mic-mute touch button (red LED = muted) and rerun this.")


if __name__ == "__main__":
    main()
