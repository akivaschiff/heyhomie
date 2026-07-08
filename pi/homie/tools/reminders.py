"""Reminders at a clock time or after a duration, announced on the kitchen speaker.

Persistent: each reminder is one `id|fire-epoch|text` line in the store's
reminders doc. The line is written on set and removed on fire, so cleanup is
inherent. start_watch() (called at app startup) re-arms pending reminders after
a restart — announcing anything that came due while the service was down, unless
it's stale — and rescans periodically so reminders set from another channel's
process still fire here on the speaker."""

import json
import uuid
from datetime import timedelta

from homie.store import REMINDERS
from homie.tools.base import Tool, ToolContext

LATE_GRACE_SECONDS = 30 * 60
RESCAN_SECONDS = 60


def _entries(ctx: ToolContext) -> list:
    out = []
    for line in ctx.store.lines(REMINDERS):
        parts = line.split("|", 2)
        if len(parts) == 3:
            try:
                out.append({"id": parts[0], "at": float(parts[1]), "text": parts[2]})
            except ValueError:
                continue
    return out


def _write(ctx: ToolContext, entries: list) -> None:
    ctx.store.set_lines(REMINDERS, [f"{e['id']}|{e['at']}|{e['text']}" for e in entries])


def _jobs(ctx: ToolContext) -> dict:
    return ctx.session.setdefault("reminder_jobs", {})


def _fire(ctx: ToolContext, reminder_id: str) -> None:
    entries = _entries(ctx)
    entry = next((e for e in entries if e["id"] == reminder_id), None)
    _jobs(ctx).pop(reminder_id, None)
    if entry is None:  # cancelled (possibly from another channel) since arming
        return
    _write(ctx, [e for e in entries if e["id"] != reminder_id])
    ctx.channel.announce(f"Reminder: {entry['text']}.")


def _arm(ctx: ToolContext, entry: dict) -> None:
    delay = entry["at"] - ctx.clock.now().timestamp()
    job_id = ctx.scheduler.schedule(max(0.0, delay), lambda: _fire(ctx, entry["id"]))
    _jobs(ctx)[entry["id"]] = job_id


def rearm(ctx: ToolContext) -> None:
    """Arm any stored reminders this process doesn't know about; fire recent
    overdue ones (missed while down), silently drop stale ones."""
    now = ctx.clock.now().timestamp()
    keep = []
    for entry in _entries(ctx):
        if entry["id"] in _jobs(ctx):
            keep.append(entry)
        elif entry["at"] > now:
            keep.append(entry)
            _arm(ctx, entry)
        elif now - entry["at"] <= LATE_GRACE_SECONDS:
            ctx.channel.announce(f"Reminder: {entry['text']}.")
        # else: stale — drop the line
    _write(ctx, keep)


def start_watch(ctx: ToolContext) -> None:
    """Re-arm now and keep rescanning, so cross-process reminders fire here."""
    if ctx.session.get("reminder_watch"):
        return
    ctx.session["reminder_watch"] = True

    def tick():
        rearm(ctx)
        ctx.scheduler.schedule(RESCAN_SECONDS, tick)

    tick()


def _seconds_until_clock(ctx: ToolContext, hhmm: str) -> int:
    hour, minute = (int(p) for p in hhmm.strip().split(":"))
    now = ctx.clock.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return int((target - now).total_seconds())


def _set(args: dict, ctx: ToolContext) -> str:
    text = args["text"].strip().replace("|", "/")
    if args.get("at"):
        seconds = _seconds_until_clock(ctx, args["at"])
        when = args["at"]
    elif args.get("seconds") is not None:
        seconds = int(args["seconds"])
        when = f"in {seconds}s"
    else:
        return json.dumps({"error": "provide either 'at' (HH:MM) or 'seconds'"})

    entry = {
        "id": uuid.uuid4().hex[:6],
        "at": ctx.clock.now().timestamp() + seconds,
        "text": text,
    }
    _write(ctx, _entries(ctx) + [entry])
    _arm(ctx, entry)
    return json.dumps({"reminder": text, "id": entry["id"], "fires": when, "in_seconds": seconds})


def _list(args: dict, ctx: ToolContext) -> str:
    now = ctx.clock.now().timestamp()
    return json.dumps({
        "reminders": [
            {"id": e["id"], "text": e["text"], "minutes_away": max(0, round((e["at"] - now) / 60))}
            for e in sorted(_entries(ctx), key=lambda e: e["at"])
        ]
    })


def _cancel(args: dict, ctx: ToolContext) -> str:
    target = args["which"].strip().lower()
    entries = _entries(ctx)
    matched = [e for e in entries if e["id"] == target or target in e["text"].lower()]
    if not matched:
        return json.dumps({"error": f"no reminder matching '{target}'",
                           "reminders": [e["text"] for e in entries]})
    if len(matched) > 1:
        return json.dumps({"ambiguous": target, "candidates": [
            {"id": e["id"], "text": e["text"]} for e in matched]})
    entry = matched[0]
    _write(ctx, [e for e in entries if e["id"] != entry["id"]])
    job_id = _jobs(ctx).pop(entry["id"], None)
    if job_id:
        ctx.scheduler.cancel(job_id)
    return json.dumps({"cancelled": entry["text"]})


TOOLS = [
    Tool(
        name="reminder_set",
        description=(
            "Set a SPOKEN reminder that announces on the kitchen speaker, either at a "
            "clock time (24h 'HH:MM', e.g. '17:20') or after a duration in seconds. "
            "Persistent — survives restarts. Example: 'reminder for 17:20, prepare "
            "dough'. This only speaks — to actually control a device (AC, light, "
            "blind) at a time, use schedule_set; never substitute a spoken reminder "
            "for a device action."
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
    Tool(
        name="reminder_list",
        description="List pending reminders with how many minutes away each is.",
        input_schema={"type": "object", "properties": {}},
        handler=_list,
    ),
    Tool(
        name="reminder_cancel",
        description=(
            "Cancel a pending reminder by its id or by a word from its text "
            "(e.g. 'dough')."
        ),
        input_schema={
            "type": "object",
            "properties": {"which": {"type": "string"}},
            "required": ["which"],
        },
        handler=_cancel,
    ),
]
