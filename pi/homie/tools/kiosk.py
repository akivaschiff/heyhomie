"""Kiosk view control — kiosk_show. The kitchen tablet lives on a polling shell
page (/kiosk on the home server) that swaps to whichever view this tool selects.

The active view is a home-local, transient UI signal, so it is written to a plain
local file the home server reads directly — deliberately NOT through the
Drive-synced Store, which is for durable cross-channel state and would add sync
latency to something that must feel instant."""

import json

from homie.config import PI_DIR
from homie.tools.base import Tool, ToolContext

KIOSK_FILE = PI_DIR / ".homie-state" / "kiosk.txt"

VIEWS = {
    "dashboard": "the smart-home controls",
    "list": "the shopping list",
}


def _show(args: dict, ctx: ToolContext) -> str:
    view = (args.get("view") or "").strip().lower()
    if view not in VIEWS:
        return json.dumps({"error": f"unknown view '{view}'", "views": list(VIEWS)})
    KIOSK_FILE.parent.mkdir(parents=True, exist_ok=True)
    KIOSK_FILE.write_text(view + "\n")
    return json.dumps({"showing": view, "surface": VIEWS[view]})


TOOLS = [
    Tool(
        name="kiosk_show",
        description=(
            "Switch which page the kitchen tablet (kiosk) is displaying. "
            "'dashboard' shows the smart-home controls (lights, blinds, ACs); "
            "'list' shows the shopping list. Use for intents like 'show me the "
            "shopping list', 'show the lights', 'back to the dashboard'. The tablet "
            "updates within about a second. Home-local: has no visible effect from "
            "remote channels."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "enum": list(VIEWS),
                    "description": "Which view to show on the tablet.",
                }
            },
            "required": ["view"],
        },
        handler=_show,
    ),
]
