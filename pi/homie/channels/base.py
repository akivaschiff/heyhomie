"""Channel + render-capability model.

Device-bound output is a per-channel capability, not a fixed assumption. A tool
that wants to "show" something asks the channel what surface it has and renders
to it; it never assumes a tablet exists.

  Pi       -> tablet screen + speaker
  Mac/text -> own screen (browser/file) + speaker (say) or printed no-op
  Telegram -> inline chat
  Headless -> none

A `Rendered` payload carries both a rich form (html / structured) and a plain
`speech` / `text` fallback, so the same tool output lands correctly on any surface.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Surface:
    has_screen: bool = False
    has_speaker: bool = False
    is_chat: bool = False


@dataclass
class Rendered:
    title: str
    speech: str                 # spoken / short form for speaker + chat fallback
    html: str = ""              # screen form (tablet / Mac browser)
    text: str = ""              # plain-text form (chat / no-screen fallback)
    structured: dict = field(default_factory=dict)

    def chat_text(self) -> str:
        return self.text or self.speech


class Channel:
    """Base executor. Subclasses bind to a device; the core stays unaware of which."""

    name = "base"
    surface = Surface()

    # --- conversational turn output (the brain's final reply) ---
    def deliver(self, text: str) -> None:
        """Deliver the assistant's spoken/written reply for a turn."""
        raise NotImplementedError

    # --- tool-driven rendering ---
    def render(self, rendered: Rendered) -> None:
        """Show a rich payload on the best available surface for this channel.
        Exactly one surface consumes it, so nothing is emitted twice."""
        if self.surface.has_screen:
            self.show_screen(rendered)
            if self.surface.has_speaker:
                self.say(rendered.speech)
        elif self.surface.is_chat:
            self.send_chat(rendered.chat_text())
        elif self.surface.has_speaker:
            self.say(rendered.speech)
        else:
            self.deliver(rendered.speech)

    def show_screen(self, rendered: Rendered) -> None:
        raise NotImplementedError

    def send_chat(self, text: str) -> None:
        raise NotImplementedError

    def say(self, text: str) -> None:
        """Speaker output within a turn."""
        self.deliver(text)

    # --- out-of-band speaker (timers / reminders fire here) ---
    def announce(self, text: str) -> None:
        """Kitchen-speaker output outside a conversation turn."""
        self.say(text)
