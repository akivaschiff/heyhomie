"""Reminders at a clock time or after a duration. Delivered to the kitchen speaker."""

import json
from datetime import timedelta

from homie.tools.base import Tool, ToolContext


def _seconds_until_clock(ctx: ToolContext, hhmm: str) -> int:
    hour, minute = (int(p) for p in hhmm.strip().split(":"))
    now = ctx.clock.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return int((target - now).total_seconds())


def _set(args: dict, ctx: ToolContext) -> str:
    text = args["text"].strip()
    if args.get("at"):
        seconds = _seconds_until_clock(ctx, args["at"])
        when = args["at"]
    elif args.get("seconds") is not None:
        seconds = int(args["seconds"])
        when = f"in {seconds}s"
    else:
        return json.dumps({"error": "provide either 'at' (HH:MM) or 'seconds'"})

    def fire():
        ctx.channel.announce(f"Reminder: {text}.")

    ctx.scheduler.schedule(seconds, fire)
    return json.dumps({"reminder": text, "fires": when, "in_seconds": seconds})


TOOLS = [
    Tool(
        name="reminder_set",
        description=(
            "Set a SPOKEN reminder that announces on the kitchen speaker, either at a "
            "clock time (24h 'HH:MM', e.g. '17:20') or after a duration in seconds. "
            "Example: 'reminder for 17:20, prepare dough'. This only speaks — to "
            "actually control a device (AC, light, blind) at a time, use schedule_set; "
            "never substitute a spoken reminder for a device action."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "What to remind about."},
                "at": {"type": "string", "description": "Clock time 'HH:MM' (24h)."},
                "seconds": {"type": "integer", "description": "Delay in seconds."},
            },
            "required": ["text"],
        },
        handler=_set,
    ),
]
