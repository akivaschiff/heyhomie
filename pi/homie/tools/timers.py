"""Concurrent named timers. Each fires independently on the kitchen speaker."""

import json

from homie.tools.base import Tool, ToolContext


def _jobs(ctx: ToolContext) -> dict:
    return ctx.session.setdefault("timers", {})


def _set(args: dict, ctx: ToolContext) -> str:
    name = args["name"].strip()
    seconds = int(args["seconds"])
    jobs = _jobs(ctx)

    def fire():
        jobs.pop(name, None)
        ctx.channel.announce(f"Timer: {name}.")

    job_id = ctx.scheduler.schedule(seconds, fire)
    jobs[name] = {
        "job_id": job_id,
        "seconds": seconds,
        "due_at": ctx.clock.monotonic() + seconds,
    }
    return json.dumps({"set": name, "seconds": seconds})


def _list(args: dict, ctx: ToolContext) -> str:
    now = ctx.clock.monotonic()
    timers = [
        {"name": name, "seconds_remaining": max(0, int(j["due_at"] - now))}
        for name, j in _jobs(ctx).items()
    ]
    return json.dumps({"timers": timers, "count": len(timers)})


def _cancel(args: dict, ctx: ToolContext) -> str:
    name = args["name"].strip()
    job = _jobs(ctx).pop(name, None)
    if not job:
        return json.dumps({"error": f"no timer named '{name}'"})
    ctx.scheduler.cancel(job["job_id"])
    return json.dumps({"cancelled": name})


TOOLS = [
    Tool(
        name="timer_set",
        description=(
            "Set a named kitchen timer that rings on the speaker after a duration. "
            "Multiple timers run concurrently, each with its own name "
            "(e.g. 'cake in the oven' 1800s, 'flip pancakes' 30s)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name announced when it fires."},
                "seconds": {"type": "integer", "description": "Duration in seconds."},
            },
            "required": ["name", "seconds"],
        },
        handler=_set,
    ),
    Tool(
        name="timer_list",
        description="List active timers with remaining time.",
        input_schema={"type": "object", "properties": {}},
        handler=_list,
    ),
    Tool(
        name="timer_cancel",
        description="Cancel an active timer by name.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        handler=_cancel,
    ),
]
