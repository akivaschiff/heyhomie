"""Scheduled home actions — cron-backed, fully persistent.

schedule_set resolves the target devices NOW (via the smarthome server), embeds
the concrete curl commands in a crontab entry, and from then on plain cron fires
them — homie and Claude are not needed at fire time. Recurrence is cron-native:
once (date-guarded, auto-pruned), daily, weekdays (Sun-Thu), weekends (Fri-Sat).
"""

import json

from homie.services.cron import build_cron_line, build_curl
from homie.tools.base import Tool, ToolContext
from homie.tools.smart_home import _inventory, _resolve

RECURS = ("once", "daily", "weekdays", "weekends")


def _payloads(device: dict, action: str, args: dict) -> tuple:
    """(system, payload_json) for one device — mirrors home_control's actuation."""
    system = device["addr"][0]
    if system == "higoal":
        _, panel_id, idx = device["addr"]
        if device["kind"] == "blind":
            idx = idx if action == "open" else idx + 1
            return system, json.dumps({"device": panel_id, "idx": idx, "on": True})
        return system, json.dumps({"device": panel_id, "idx": idx, "on": action == "on"})
    _, unit_id = device["addr"]
    payload = {"id": unit_id, "power": action == "on"}
    for key in ("temp", "mode", "fan"):
        if args.get(key) is not None and action == "on":
            payload[key] = args[key]
    return system, json.dumps(payload)


def _today(ctx: ToolContext) -> str:
    return ctx.clock.now().strftime("%Y-%m-%d")


def _set(args: dict, ctx: ToolContext) -> str:
    if ctx.cron is None or ctx.smarthome is None:
        return json.dumps({"error": "scheduling not configured"})
    recur = args.get("recur", "once")
    if recur not in RECURS:
        return json.dumps({"error": f"recur must be one of {RECURS}"})
    date = args.get("date", "")
    if recur == "once" and not date:
        return json.dumps({"error": "one-time schedules need a date (YYYY-MM-DD)"})

    kind, target, action = args["kind"], args["target"], args["action"]
    matched = _resolve(_inventory(ctx), kind, target)
    if not matched:
        return json.dumps({"error": f"no {kind} matching '{target}'"})

    commands = [
        build_curl(ctx.smarthome.base_url, *_payloads(d, action, args)) for d in matched
    ]
    entries = ctx.cron.prune_stale(_today(ctx))
    from homie.services.cron import Entry

    description = args.get(
        "description", f"{action} {kind} {target} at {args['time']} ({recur})"
    )
    entry = Entry(
        id=ctx.cron.next_id(),
        recur=recur,
        date=date,
        description=description,
        cron_line=build_cron_line(args["time"], recur, date, commands),
    )
    entries.append(entry)
    ctx.cron.save_entries(entries)
    return json.dumps({
        "scheduled": entry.id,
        "description": description,
        "devices": [d["name"] for d in matched],
    })


def _list(args: dict, ctx: ToolContext) -> str:
    if ctx.cron is None:
        return json.dumps({"error": "scheduling not configured"})
    entries = ctx.cron.prune_stale(_today(ctx))
    return json.dumps({
        "schedules": [
            {"id": e.id, "recur": e.recur, "date": e.date or None, "description": e.description}
            for e in entries
        ]
    })


def _cancel(args: dict, ctx: ToolContext) -> str:
    if ctx.cron is None:
        return json.dumps({"error": "scheduling not configured"})
    entries = ctx.cron.prune_stale(_today(ctx))
    keep = [e for e in entries if e.id != args["id"]]
    if len(keep) == len(entries):
        return json.dumps({"error": f"no schedule '{args['id']}'"})
    ctx.cron.save_entries(keep)
    return json.dumps({"cancelled": args["id"]})


TOOLS = [
    Tool(
        name="schedule_set",
        description=(
            "Schedule a smart-home action (same kind/target/action semantics as "
            "home_control) at a clock time, persistently — it fires via cron even if "
            "the assistant is down. recur: 'once' (needs date YYYY-MM-DD — compute it "
            "from the current date, e.g. 'tonight'), 'daily', 'weekdays' (Sun-Thu) or "
            "'weekends' (Fri-Sat). For 'on at 19:00 and off at 8:00' make two calls."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "time": {"type": "string", "description": "24h clock time 'HH:MM'."},
                "recur": {"type": "string", "enum": list(RECURS)},
                "date": {"type": "string", "description": "YYYY-MM-DD, required when recur=once."},
                "kind": {"type": "string", "enum": ["light", "blind", "ac"]},
                "target": {"type": "string", "description": "Device name, 'all', or AC zone."},
                "action": {"type": "string", "enum": ["on", "off", "open", "close"]},
                "temp": {"type": "integer"},
                "mode": {"type": "string"},
                "fan": {"type": "string"},
                "description": {"type": "string", "description": "Short human label."},
            },
            "required": ["time", "recur", "kind", "target", "action"],
        },
        handler=_set,
    ),
    Tool(
        name="schedule_list",
        description="List the scheduled home actions.",
        input_schema={"type": "object", "properties": {}},
        handler=_list,
    ),
    Tool(
        name="schedule_cancel",
        description="Cancel a scheduled home action by its id (see schedule_list).",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
        handler=_cancel,
    ),
]
