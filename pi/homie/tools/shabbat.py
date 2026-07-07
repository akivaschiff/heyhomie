"""shabbat_mode — display Shabbat times on the tablet. Display only, no gating."""

import json

from homie.channels.base import Rendered
from homie.render import shabbat_page
from homie.tools.base import Tool, ToolContext


def _show(args: dict, ctx: ToolContext) -> str:
    if not ctx.shabbat_times:
        return json.dumps({"error": "Shabbat times unavailable"})
    data = ctx.shabbat_times(ctx.config.shabbat_geoname_id)
    location = data.get("location", "")
    rows = data.get("rows", [])
    speech = "Shabbat times for " + location + ": " + "; ".join(f"{l} {v}" for l, v in rows)
    rendered = Rendered(
        title=f"Shabbat — {location}",
        speech=speech,
        html=shabbat_page(location, rows),
        text="Shabbat — " + location + "\n" + "\n".join(f"{l}: {v}" for l, v in rows),
        structured=data,
    )
    ctx.channel.render(rendered)
    return json.dumps({"shown": True, "location": location, "rows": rows})


TOOLS = [
    Tool(
        name="shabbat_mode",
        description="Display the upcoming Shabbat times (candle-lighting, havdalah) on the screen.",
        input_schema={"type": "object", "properties": {}},
        handler=_show,
    ),
]
