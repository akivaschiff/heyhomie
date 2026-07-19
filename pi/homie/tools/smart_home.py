"""Smart-home control — lights, blinds and ACs via the smarthome server.

House topology (stable): lights and blinds (Higoal) are on the main floor —
kitchen, living room, dining room; names are often Hebrew. The central AC
(Electra, "קומת קרקע") covers the main floor / downstairs. Four split ACs
(Midea) are upstairs: Girls' Room, Boys' Room, Parent Bedroom, The Office.

The model resolves what the user meant (exact names come from home_status);
this tool resolves names to devices and fans out group actions in one call.
"""

import json
import re

from homie.tools.base import Tool, ToolContext

_CHANNEL_NOISE = re.compile(r"channel \d+$")


def _inventory(ctx: ToolContext) -> tuple:
    # Each backend is a separate cloud/LAN system; a blip on one must not hide the
    # others. Collect what we can and report which systems were unreachable so the
    # caller can tell "no such device" apart from "that system is temporarily down".
    devices, down = [], []
    try:
        for panel in ctx.smarthome.get("higoal"):
            for ent in panel.get("entities", []):
                name = (ent.get("name") or "").strip()
                if not name or _CHANNEL_NOISE.search(name):
                    continue
                kind = "blind" if ent["type"] == "shutter" else "light"
                devices.append({
                    "kind": kind,
                    "name": name,
                    "on": ent.get("on"),
                    "addr": ("higoal", panel["id"], ent["idx"]),
                })
    except Exception:
        down.append("higoal")
    try:
        for unit in ctx.smarthome.get("midea"):
            devices.append({
                "kind": "ac",
                "zone": "upstairs",
                "name": unit["name"],
                "on": unit.get("power"),
                "temp": unit.get("target"),
                "room_temp": unit.get("indoor"),
                "addr": ("midea", unit["id"]),
            })
    except Exception:
        down.append("midea")
    try:
        for unit in ctx.smarthome.get("electra"):
            devices.append({
                "kind": "ac",
                "zone": "main",
                "name": unit["name"],
                "on": unit.get("on"),
                "temp": unit.get("target"),
                "addr": ("electra", unit["id"]),
            })
    except Exception:
        down.append("electra")
    return devices, down


def _matches(target: str, name: str) -> bool:
    t, n = target.lower().strip(), name.lower()
    return t in n or n in t or any(w in n for w in t.split() if len(w) > 2)


def _resolve(devices: list, kind: str, target: str) -> list:
    pool = [d for d in devices if d["kind"] == kind]
    t = target.lower().strip()
    if t == "all":
        return pool
    if kind == "ac" and t in ("upstairs", "bedrooms"):
        return [d for d in pool if d.get("zone") == "upstairs"]
    if kind == "ac" and t in ("main", "downstairs", "central"):
        return [d for d in pool if d.get("zone") == "main"]
    exact = [d for d in pool if d["name"].lower() == t]
    if exact:
        return exact
    return [d for d in pool if _matches(t, d["name"])]


def _actuate(ctx: ToolContext, device: dict, action: str, args: dict) -> dict:
    system = device["addr"][0]
    if system == "higoal":
        _, panel_id, idx = device["addr"]
        if device["kind"] == "blind":
            # per API contract: open uses idx, the paired close relay is idx + 1
            idx = idx if action == "open" else idx + 1
            ctx.smarthome.set("higoal", {"device": panel_id, "idx": idx, "on": True})
        else:
            ctx.smarthome.set("higoal", {"device": panel_id, "idx": idx, "on": action == "on"})
    else:
        _, unit_id = device["addr"]
        payload = {"id": unit_id, "power": action == "on"}
        for key in ("temp", "mode", "fan"):
            if args.get(key) is not None and action == "on":
                payload[key] = args[key]
        resp = ctx.smarthome.set(system, payload)
        if isinstance(resp, dict) and resp.get("ok") is False:
            raise RuntimeError(f"{device['name']} did not reach {action} (still {'on' if resp.get('on') else 'off'})")
    return {"name": device["name"], "did": action}


_SYSTEMS_FOR = {"light": ("higoal",), "blind": ("higoal",), "ac": ("midea", "electra")}


_RETRY_WINDOW_S = 30
_RETRY_GAP_S = 5
_VERB = {"on": "turn on", "off": "turn off", "open": "open", "close": "close"}
_GERUND = {"on": "turning on", "off": "turning off", "open": "opening", "close": "closing"}


def _label(device: dict) -> str:
    if device["kind"] == "ac":
        return "the main air conditioner" if device.get("zone") == "main" else f"the {device['name']}"
    return device["name"]


def _join(labels: list) -> str:
    if len(labels) <= 1:
        return labels[0] if labels else ""
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _drive(ctx: ToolContext, pending: list, action: str, args: dict, deadline: float) -> None:
    # Optimistic execution: the turn already acknowledged, so actuate off the request
    # thread, keep retrying transient failures until the deadline, and only speak up
    # if the devices never reach the requested state.
    failed = []
    for device in pending:
        try:
            _actuate(ctx, device, action, args)
        except Exception as exc:
            print(f"⚠️  home_control: could not {action} {device['name']}: {exc}", flush=True)
            failed.append(device)
    if not failed:
        return
    if ctx.clock.monotonic() < deadline:
        ctx.scheduler.schedule(_RETRY_GAP_S, lambda: _drive(ctx, failed, action, args, deadline))
    else:
        ctx.channel.announce(f"Sorry, I couldn't {_VERB[action]} {_join([_label(d) for d in failed])}.")


def _status(args: dict, ctx: ToolContext) -> str:
    if ctx.smarthome is None:
        return json.dumps({"error": "smart home server not configured"})
    devices, down = _inventory(ctx)
    kind = args.get("kind")
    if kind:
        devices = [d for d in devices if d["kind"] == kind]
    out = {"devices": [{k: v for k, v in d.items() if k != "addr"} for d in devices]}
    if down:
        out["unavailable"] = down
    return json.dumps(out)


def _control(args: dict, ctx: ToolContext) -> str:
    if ctx.smarthome is None:
        return json.dumps({"error": "smart home server not configured"})
    kind, target, action = args["kind"], args["target"], args["action"]
    valid = {"light": ("on", "off"), "ac": ("on", "off"), "blind": ("open", "close")}
    if action not in valid.get(kind, ()):
        return json.dumps({"error": f"action '{action}' invalid for {kind}; use {valid.get(kind)}"})

    devices, down = _inventory(ctx)
    matched = _resolve(devices, kind, target)
    if not matched:
        relevant_down = [s for s in _SYSTEMS_FOR.get(kind, ()) if s in down]
        if relevant_down:
            return json.dumps({"error": f"the {kind} system is temporarily unreachable; try again",
                               "unavailable": relevant_down})
        names = [d["name"] for d in devices if d["kind"] == kind]
        return json.dumps({"error": f"no {kind} matching '{target}'", "available": names})
    if len(matched) > 3 and target.lower() not in ("all", "upstairs", "bedrooms", "main", "downstairs", "central"):
        return json.dumps({"ambiguous": target, "candidates": [d["name"] for d in matched]})

    deadline = ctx.clock.monotonic() + _RETRY_WINDOW_S
    ctx.scheduler.schedule(0, lambda: _drive(ctx, list(matched), action, args, deadline))
    return json.dumps({
        "started": action,
        "devices": [_label(d) for d in matched],
        "note": ("Running in the background — acknowledge in the present tense "
                 f"('{_GERUND[action]} …'), never say it's already done. The user is "
                 "told automatically only if it ultimately fails, so don't check state."),
    })


_TOPOLOGY = (
    "House layout: lights and blinds are on the main floor (kitchen, living room, "
    "dining room), names are often Hebrew (מטבח=kitchen, סלון=living room, "
    "פינת אוכל=dining, מאור=light, תריס=blind, קטן=small, גדול=big, "
    "חדר משחקים=game room). Match the user's English to these Hebrew names "
    "directly (e.g. 'small kitchen light' = מאור מטבח קטן). The 'main'/'downstairs' AC is the "
    "central Electra unit; the four 'upstairs' split ACs are Girls' Room, Boys' Room, "
    "Parent Bedroom and The Office."
)

TOOLS = [
    Tool(
        name="home_status",
        description=(
            "List smart-home devices and their current state (lights, blinds, ACs "
            "with temperatures). Use to check state, or to see exact device names "
            "before controlling something ambiguous. " + _TOPOLOGY
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["light", "blind", "ac"],
                         "description": "Filter by device kind (omit for everything)."},
            },
        },
        handler=_status,
    ),
    Tool(
        name="home_control",
        description=(
            "Control lights, blinds and air conditioners. target is a device name "
            "(exact name from home_status is safest), or 'all' for every device of "
            "that kind, or for ACs a zone: 'upstairs' (the 4 bedroom/office splits) "
            "or 'main' (the downstairs central unit). Group targets fan out in one "
            "call. Returns immediately: the action runs in the background with retries, "
            "and the user is notified only if it ultimately fails — so acknowledge in "
            "the present tense and never claim it is already done. " + _TOPOLOGY
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["light", "blind", "ac"]},
                "target": {"type": "string",
                           "description": "Device name, 'all', or AC zone ('upstairs'/'main')."},
                "action": {"type": "string", "enum": ["on", "off", "open", "close"],
                           "description": "on/off for lights and ACs; open/close for blinds."},
                "temp": {"type": "integer", "description": "AC target temperature (with action=on)."},
                "mode": {"type": "string", "description": "AC mode, e.g. COOL/HEAT (with action=on)."},
                "fan": {"type": "string", "description": "AC fan speed (with action=on)."},
            },
            "required": ["kind", "target", "action"],
        },
        handler=_control,
    ),
]
