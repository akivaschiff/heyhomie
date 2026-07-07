"""volume_set — kitchen speaker volume by voice: quieter/louder or a percentage."""

import json

from homie.tools.base import Tool, ToolContext

STEP = 15


def _set(args: dict, ctx: ToolContext) -> str:
    if ctx.volume is None:
        return json.dumps({"error": "volume control unavailable on this device"})
    try:
        current = ctx.volume.get()
        if args.get("level") is not None:
            new = ctx.volume.set(int(args["level"]))
        elif args.get("direction") == "quieter":
            new = ctx.volume.set(current - STEP)
        elif args.get("direction") == "louder":
            new = ctx.volume.set(current + STEP)
        else:
            return json.dumps({"volume": current})
        return json.dumps({"volume": new, "was": current})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


TOOLS = [
    Tool(
        name="volume_set",
        description=(
            "Adjust the kitchen speaker volume (mic sensitivity is unaffected). "
            "Pass direction 'quieter'/'louder' for a step change, or level 0-100 "
            "for an absolute percentage ('set volume to 40 percent'). With no "
            "arguments, reports the current volume."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["quieter", "louder"]},
                "level": {"type": "integer", "description": "Absolute volume 0-100."},
            },
        },
        handler=_set,
    ),
]
