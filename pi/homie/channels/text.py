"""Text / Mac harness channel.

The portability surface: the whole assistant drivable from typed input with no
audio in the stack. Device-bound output resolves to the Mac (browser + `say`) or
a printed no-op. Audio is opt-in (`speak=True`); the default loop is silent.
"""

import os
import subprocess
import tempfile

from homie.channels.base import Channel, Rendered, Surface


class TextChannel(Channel):
    name = "text"

    def __init__(self, speak: bool = False, open_browser: bool = True):
        self.speak = speak and _is_macos()
        self.open_browser = open_browser
        self.surface = Surface(has_screen=True, has_speaker=self.speak, is_chat=False)

    def deliver(self, text: str) -> None:
        if text:
            print(f"Homie: {text}")

    def show_screen(self, rendered: Rendered) -> None:
        print(f"\n--- {rendered.title} (shown on screen) ---")
        print(rendered.chat_text())
        print("--- end ---\n")
        if self.open_browser and rendered.html:
            self._open_html(rendered.html)

    def send_chat(self, text: str) -> None:
        print(text)

    def say(self, text: str) -> None:
        if self.speak and text:
            try:
                subprocess.run(["say", "-v", "Samantha", text], check=False)
            except Exception:
                pass

    def announce(self, text: str) -> None:
        # A timer/reminder is an alert — make noise on the Mac even when the
        # conversational TTS flag is off; device-bound output resolves to this Mac.
        print(f"\n🔊 [kitchen speaker] {text}\n")
        if _is_macos():
            try:
                subprocess.run(
                    ["afplay", "/System/Library/Sounds/Glass.aiff"], check=False
                )
                subprocess.run(["say", "-v", "Samantha", text], check=False)
            except Exception:
                pass

    def run(self, brain, once: bool = False) -> None:
        if once:
            import sys

            for line in sys.stdin:
                line = line.strip()
                if line:
                    brain.handle(line)
            return
        print("Homie (text harness). Ctrl+C / Ctrl+D to exit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if user_input:
                brain.handle(user_input)

    def _open_html(self, html: str) -> None:
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".html", delete=False, encoding="utf-8"
            ) as f:
                f.write(html)
                path = f.name
            subprocess.run(["open", path], check=False)
        except Exception:
            pass


def _is_macos() -> bool:
    return os.uname().sysname == "Darwin"
