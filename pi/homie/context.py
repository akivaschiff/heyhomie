"""Rolling conversation context — a simple time window, no summarization.

We keep the last ~15 minutes of conversation and let anything older fall away;
if something old is needed again, the model just asks or re-fetches it via a tool.
Interactions are ad-hoc, so a short window is plenty.

Pruning is turn-atomic: a "turn" is a user message plus every assistant/tool
message it produced. We only ever drop whole turns, so a tool_use is never
separated from its tool_result (which the API rejects) and the retained history
always starts on a real user message.
"""

from datetime import timedelta

from homie.clock import Clock


class ConversationContext:
    def __init__(self, window_seconds: int, clock: Clock = None):
        self.window = timedelta(seconds=window_seconds)
        self.clock = clock or Clock()
        self.turns: list[dict] = []  # each: {"started": datetime, "messages": [...]}

    def start_turn(self, user_text: str) -> None:
        """Begin a new user turn, dropping turns older than the window."""
        now = self.clock.now()
        self.turns = [t for t in self.turns if now - t["started"] <= self.window]
        self.turns.append({"started": now, "messages": [{"role": "user", "content": user_text}]})

    def add(self, role: str, content) -> None:
        """Append an assistant/tool message to the current turn."""
        if not self.turns:
            self.start_turn("")
        self.turns[-1]["messages"].append({"role": role, "content": content})

    def get(self) -> list[dict]:
        return [m for turn in self.turns for m in turn["messages"]]
