"""Telegram channel (remote, when away). A chat surface over the shared core:
inbound messages drive the same brain + tools + state as voice. The brain runs
synchronously in a worker thread; replies are marshalled back onto PTB's event
loop. Renders (list, recipe, Shabbat) arrive inline as their text form."""

import asyncio
import os

from homie.channels.base import Channel, Rendered, Surface


class TelegramChannel(Channel):
    name = "telegram"

    def __init__(self):
        self.surface = Surface(has_screen=False, has_speaker=False, is_chat=True)
        self._loop = None
        self._bot = None
        self._active_chat_id = None

    def deliver(self, text: str) -> None:
        self.send_chat(text)

    def send_chat(self, text: str) -> None:
        if not text or self._active_chat_id is None:
            return
        future = asyncio.run_coroutine_threadsafe(
            self._bot.send_message(chat_id=self._active_chat_id, text=text), self._loop
        )
        future.result()

    def say(self, text: str) -> None:
        self.send_chat(text)

    def announce(self, text: str) -> None:
        self.send_chat(text)

    def show_screen(self, rendered: Rendered) -> None:  # no screen; render falls to chat
        self.send_chat(rendered.chat_text())

    def run(self, brain) -> None:
        from telegram.ext import Application, MessageHandler, filters

        token = os.environ["TELEGRAM_BOT_TOKEN"]
        app = Application.builder().token(token).build()
        self._bot = app.bot

        async def on_message(update, context):
            self._loop = asyncio.get_running_loop()
            self._active_chat_id = update.effective_chat.id
            await asyncio.to_thread(brain.handle, update.message.text)

        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
        print("Telegram channel running…")
        app.run_polling()
