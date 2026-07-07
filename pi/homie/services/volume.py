"""Speaker volume control. Playback only — never the capture control, which on
the Anker is a separate hardware knob and must stay at full mic sensitivity."""

import os
import re
import subprocess


class AlsaVolume:
    """Anker playback volume via its dedicated Playback Volume control."""

    def __init__(self):
        self._card = None
        self._numid = None
        self._max = 127

    def _resolve(self):
        if self._numid is not None:
            return
        for line in open("/proc/asound/cards"):
            if "S330" in line or "Anker" in line:
                self._card = line.strip().split(" ")[0]
                break
        if self._card is None:
            raise RuntimeError("Anker speaker not found")
        out = subprocess.run(
            ["amixer", "-c", self._card, "contents"], capture_output=True, text=True
        ).stdout
        block = re.search(
            r"numid=(\d+)[^\n]*Playback Volume'\n[^\n]*max=(\d+)", out
        )
        if not block:
            raise RuntimeError("no Playback Volume control on the Anker")
        self._numid = block.group(1)
        self._max = int(block.group(2))

    def get(self) -> int:
        self._resolve()
        out = subprocess.run(
            ["amixer", "-c", self._card, "cget", f"numid={self._numid}"],
            capture_output=True, text=True,
        ).stdout
        match = re.search(r": values=(\d+)", out)
        raw = int(match.group(1)) if match else 0
        return round(raw * 100 / self._max)

    def set(self, percent: int) -> int:
        self._resolve()
        percent = max(0, min(100, percent))
        raw = round(percent * self._max / 100)
        subprocess.run(
            ["amixer", "-c", self._card, "cset", f"numid={self._numid}", str(raw)],
            capture_output=True,
        )
        return percent


class MacVolume:
    def get(self) -> int:
        out = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True, text=True,
        ).stdout.strip()
        return int(out or 0)

    def set(self, percent: int) -> int:
        percent = max(0, min(100, percent))
        subprocess.run(["osascript", "-e", f"set volume output volume {percent}"])
        return percent


def system_volume():
    return MacVolume() if os.uname().sysname == "Darwin" else AlsaVolume()
