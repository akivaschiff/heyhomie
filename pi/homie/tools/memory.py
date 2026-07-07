"""Ambient free-form fact store. Channel-agnostic, shared across channels."""

import json

from homie.store import MEMORY
from homie.tools.base import Tool, ToolContext


def _save(args: dict, ctx: ToolContext) -> str:
    fact = args["fact"].strip()
    ctx.store.append_line(MEMORY, fact)
    return json.dumps({"saved": fact})


def _query(args: dict, ctx: ToolContext) -> str:
    facts = ctx.store.lines(MEMORY)
    query = (args.get("query") or "").strip().lower()
    if query:
        terms = [t for t in query.split() if len(t) > 2]
        matches = [f for f in facts if any(t in f.lower() for t in terms)]
    else:
        matches = facts
    return json.dumps({"facts": matches})


TOOLS = [
    Tool(
        name="memory_save",
        description=(
            "Store a free-form fact for later recall, e.g. 'the spare key is in the shed'. "
            "Retrievable later from any channel."
        ),
        input_schema={
            "type": "object",
            "properties": {"fact": {"type": "string", "description": "The fact to store."}},
            "required": ["fact"],
        },
        handler=_save,
    ),
    Tool(
        name="memory_query",
        description=(
            "Recall stored facts. Pass a query (e.g. 'spare key') to narrow; returns "
            "matching facts to answer from. Use for 'where's the spare key?'."
        ),
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to recall."}},
        },
        handler=_query,
    ),
]
