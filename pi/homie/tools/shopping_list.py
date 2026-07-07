"""Supermarket list — list_add, list_remove, list_show. One shared list."""

import json

from homie.channels.base import Rendered
from homie.render import list_page
from homie.store import LIST
from homie.tools.base import Tool, ToolContext

LIST_TITLE = "Shopping List"


def _match_index(items: list[str], target: str):
    """Find an item to remove, scanning newest-first. Prefer an exact match, then a
    whole-word match (so 'red' removes 'red apples' but not 'shredded wheat'),
    then a substring as a last resort."""
    lowered = [i.lower() for i in items]
    target_words = target.split()

    for i in range(len(items) - 1, -1, -1):
        if lowered[i] == target:
            return i
    for i in range(len(items) - 1, -1, -1):
        words = lowered[i].split()
        if all(w in words for w in target_words):
            return i
    for i in range(len(items) - 1, -1, -1):
        if target in lowered[i]:
            return i
    return None


def _add(args: dict, ctx: ToolContext) -> str:
    item = args["item"].strip()
    ctx.store.append_line(LIST, item)
    return json.dumps({"added": item, "list": ctx.store.lines(LIST)})


def _remove(args: dict, ctx: ToolContext) -> str:
    items = ctx.store.lines(LIST)
    if not items:
        return json.dumps({"error": "list is empty", "list": []})

    target = args.get("item", "").strip().lower()
    removed = None
    if target in ("", "last", "last one", "the last one", "last item"):
        removed = items.pop()
    else:
        idx = _match_index(items, target)
        if idx is not None:
            removed = items.pop(idx)
    if removed is None:
        return json.dumps({"error": f"no item matching '{target}'", "list": items})

    ctx.store.set_lines(LIST, items)
    return json.dumps({"removed": removed, "list": items})


def _show(args: dict, ctx: ToolContext) -> str:
    items = ctx.store.lines(LIST)
    speech = "The list is empty." if not items else "Here's the list: " + ", ".join(items) + "."
    rendered = Rendered(
        title=LIST_TITLE,
        speech=speech,
        html=list_page(LIST_TITLE, items),
        text=LIST_TITLE + ":\n" + ("\n".join(f"• {i}" for i in items) if items else "(empty)"),
        structured={"items": items},
    )

    target = (args.get("target") or "").strip().lower()
    if target and target != ctx.channel.name and ctx.push:
        ok = ctx.push(target, rendered)
        if ok:
            return json.dumps({"sent_to": target, "items": items})
    ctx.channel.render(rendered)
    return json.dumps({"shown": True, "items": items})


TOOLS = [
    Tool(
        name="list_add",
        description=(
            "Add an item to the shared supermarket list immediately, no confirmation. "
            "Use for intents like 'we finished the garbage bags' or 'I need more apples'. "
            "If the item is ambiguous (e.g. 'apples'), ask one clarifying question first, "
            "then call this with the specific item (e.g. 'green apples')."
        ),
        input_schema={
            "type": "object",
            "properties": {"item": {"type": "string", "description": "The item to add."}},
            "required": ["item"],
        },
        handler=_add,
    ),
    Tool(
        name="list_remove",
        description=(
            "Remove or correct an item on the shared supermarket list. Pass the item text "
            "to remove a matching item, or 'last' to remove the most recently added one. "
            "To correct ('not red, green'), remove then add."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "item": {
                    "type": "string",
                    "description": "Item text to remove, or 'last' for the most recent.",
                }
            },
            "required": ["item"],
        },
        handler=_remove,
    ),
    Tool(
        name="list_show",
        description=(
            "Present the current supermarket list on this channel's screen (or read it "
            "aloud / send inline if there is no screen). Set target to another channel "
            "name (e.g. 'telegram') to deliver it there instead."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Optional channel to send to, e.g. 'telegram'.",
                }
            },
        },
        handler=_show,
    ),
]
