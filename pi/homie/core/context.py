"""Conversation context management with automatic timeout."""

from datetime import datetime, timedelta


class ConversationContext:
    """Manages conversation history with automatic timeout."""

    def __init__(self, timeout_seconds: int = 60):
        self.messages = []
        self.last_interaction = None
        self.timeout = timedelta(seconds=timeout_seconds)

    def add_message(self, role: str, content: str):
        now = datetime.now()
        if self.last_interaction and now - self.last_interaction > self.timeout:
            print("Context timeout - starting fresh")
            self.messages = []
        self.messages.append({"role": role, "content": content})
        self.last_interaction = now

    def get_messages(self):
        if self.last_interaction and datetime.now() - self.last_interaction > self.timeout:
            self.messages = []
        return self.messages
