"""FakeChannel — records every surface interaction so tests can assert what the
assistant actually delivered, showed, sent, or announced. Capabilities are
configurable to simulate any device profile."""

from homie.channels.base import Channel, Rendered, Surface


class FakeChannel(Channel):
    name = "fake"

    def __init__(self, surface: Surface = None):
        self.surface = surface or Surface(has_screen=True, has_speaker=True)
        self.delivered: list[str] = []
        self.shown: list[Rendered] = []
        self.chats: list[str] = []
        self.spoken: list[str] = []
        self.announced: list[str] = []

    def deliver(self, text: str) -> None:
        self.delivered.append(text)

    def show_screen(self, rendered: Rendered) -> None:
        self.shown.append(rendered)

    def send_chat(self, text: str) -> None:
        self.chats.append(text)

    def say(self, text: str) -> None:
        self.spoken.append(text)

    def announce(self, text: str) -> None:
        self.announced.append(text)
