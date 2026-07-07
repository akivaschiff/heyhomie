"""Entry point. Builds the shared core (brain + tools + store) and binds it to a
channel executor. The core is identical across channels; only the executor changes.

  python -m homie.app --channel text
  python -m homie.app --channel text --once   (read stdin, one turn per line)
  python -m homie.app --channel voice
  python -m homie.app --channel telegram
"""

import argparse
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from homie.brain import Brain
from homie.clock import Clock, Scheduler
from homie.config import Config
from homie.services.shabbat import fetch_shabbat_times
from homie.services.cron import CronStore
from homie.services.smarthome import SmartHomeClient
from homie.services.web import fetch_url, make_recipe_extractor, tavily_search
from homie.store import get_store
from homie.tools import all_tools
from homie.tools.base import ToolContext
from homie.tracing import build_tracer


def build_brain(channel, config: Config = None):
    config = config or Config()
    anthropic = Anthropic()
    clock = Clock()
    scheduler = Scheduler()
    store = get_store(config)

    ctx = ToolContext(
        store=store,
        channel=channel,
        scheduler=scheduler,
        clock=clock,
        config=config,
        web_search=tavily_search if os.environ.get("TAVILY_API_KEY") else None,
        fetch_url=fetch_url,
        recipe_extractor=make_recipe_extractor(anthropic, config.model),
        shabbat_times=fetch_shabbat_times,
        smarthome=SmartHomeClient(),
        cron=CronStore(),
        push=_make_push(channel),
        session={},
    )
    return Brain(anthropic, all_tools(), ctx, config, clock, tracer=build_tracer())


def _make_push(current_channel):
    """Deliver a rendered payload to another channel (only Telegram for now)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("HOMIE_TELEGRAM_CHAT_ID")

    def push(channel_name: str, rendered) -> bool:
        if channel_name == "telegram" and token and chat_id:
            import requests

            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": rendered.chat_text()},
                timeout=20,
            )
            return True
        return False

    return push


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="text", choices=["text", "voice", "telegram"])
    parser.add_argument("--once", action="store_true", help="text: one turn per stdin line")
    parser.add_argument("--speak", action="store_true", help="text: enable Mac TTS")
    parser.add_argument("--ptt", action="store_true", help="voice: push-to-talk (no wake word)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    if args.channel == "text":
        from homie.channels.text import TextChannel

        channel = TextChannel(speak=args.speak)
        brain = build_brain(channel)
        channel.run(brain, once=args.once)
    elif args.channel == "voice":
        from homie.channels.voice import VoiceChannel

        channel = VoiceChannel()
        brain = build_brain(channel)
        if args.ptt:
            channel.run_ptt(brain)
        else:
            channel.run(brain)
    elif args.channel == "telegram":
        from homie.channels.telegram import TelegramChannel

        channel = TelegramChannel()
        brain = build_brain(channel)
        channel.run(brain)


if __name__ == "__main__":
    main()
