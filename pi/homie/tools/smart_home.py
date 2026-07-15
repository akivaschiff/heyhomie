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


def _inventory(ctx: ToolContext) -> list:
    devices = []
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
    for unit in ctx.smarthome.get("electra"):
        devices.append({
            "kind": "ac",
            "zone": "main",
            "name": unit["name"],
            "on": unit.get("on"),
            "temp": unit.get("target"),
            "addr": ("electra", unit["id"]),
        })
    return devices


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


def _status(args: dict, ctx: ToolContext) -> str:
    if ctx.smarthome is None:
        return json.dumps({"error": "smart home server not configured"})
    devices = _inventory(ctx)
    kind = args.get("kind")
    if kind:
        devices = [d for d in devices if d["kind"] == kind]
    return json.dumps([{k: v for k, v in d.items() if k != "addr"} for d in devices])


def _control(args: dict, ctx: ToolContext) -> str:
    if ctx.smarthome is None:
        return json.dumps({"error": "smart home server not configured"})
    kind, target, action = args["kind"], args["target"], args["action"]
    valid = {"light": ("on", "off"), "ac": ("on", "off"), "blind": ("open", "close")}
    if action not in valid.get(kind, ()):
        return json.dumps({"error": f"action '{action}' invalid for {kind}; use {valid.get(kind)}"})

    devices = _inventory(ctx)
    matched = _resolve(devices, kind, target)
    if not matched:
        names = [d["name"] for d in devices if d["kind"] == kind]
        return json.dumps({"error": f"no {kind} matching '{target}'", "available": names})
    if len(matched) > 3 and target.lower() not in ("all", "upstairs", "bedrooms", "main", "downstairs", "central"):
        return json.dumps({"ambiguous": target, "candidates": [d["name"] for d in matched]})

    done, failed = [], []
    for device in matched:
        try:
            done.append(_actuate(ctx, device, action, args))
        except Exception as exc:
            failed.append({"name": device["name"], "error": str(exc)})
    result = {"done": done}
    if failed:
        result["failed"] = failed
    return json.dumps(result)


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
            "call. " + _TOPOLOGY
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
