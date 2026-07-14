"""Supermarket list — list_add, list_remove, list_show. One shared list,
organised into fixed sections. The stored document is human-inspectable:

    # Fruits
    green apples
    # Dairy
    buttermilk

The model picks the section as it adds (native classification, not a hardcoded
map); the tool owns placement and ordering."""

import json

from homie.channels.base import Rendered
from homie.render import grouped_list_page
from homie.store import LIST
from homie.tools.base import Tool, ToolContext

LIST_TITLE = "Shopping List"

CATEGORIES = [
    "Fruits",
    "Vegetables",
    "Dairy",
    "Meat & fish",
    "Everything else",
    "General supplies",
]
DEFAULT_CATEGORY = "Everything else"


def _parse(text: str) -> dict:
    """Sectioned text -> {category: [items]}, in CATEGORIES order. Lines before any
    header (or under an unknown one) fall into the default section, so a legacy flat
    list.txt migrates cleanly."""
    groups = {c: [] for c in CATEGORIES}
    current = DEFAULT_CATEGORY
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# "):
            name = line[2:].strip()
            current = name if name in groups else DEFAULT_CATEGORY
        else:
            groups[current].append(line)
    return groups


def _serialize(groups: dict) -> str:
    out = []
    for c in CATEGORIES:
        if groups[c]:
            out.append(f"# {c}")
            out.extend(groups[c])
    return ("\n".join(out) + "\n") if out else ""


def _sections(groups: dict) -> list:
    return [(c, groups[c]) for c in CATEGORIES if groups[c]]


def _nonempty(groups: dict) -> dict:
    return {c: groups[c] for c in CATEGORIES if groups[c]}


def _flat(groups: dict):
    """Items and their (category, index) locations, in section order."""
    items, locs = [], []
    for c in CATEGORIES:
        for i, it in enumerate(groups[c]):
            items.append(it)
            locs.append((c, i))
    return items, locs


def _match_index(items: list, target: str):
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
    category = args.get("category", DEFAULT_CATEGORY)
    if category not in CATEGORIES:
        category = DEFAULT_CATEGORY

    cart = _sync_to_cart(args, item, ctx)
    display = item
    if cart and cart.get("ok") and cart.get("name"):
        display = f"{item} — {cart['name']}"

    groups = _parse(ctx.store.read(LIST))
    groups[category].append(display)
    ctx.store.write(LIST, _serialize(groups))

    result = {"added": item, "category": category, "list": _nonempty(groups)}
    if cart is not None:
        result["cart"] = cart
    return json.dumps(result)


def _sync_to_cart(args: dict, item: str, ctx: ToolContext):
    """Project the add onto the live Shufersal cart. Best-effort: if there's no cart
    seam (tests, Mac harness) or it fails, the list add still stands."""
    shufersal = getattr(ctx, "shufersal", None)
    if shufersal is None:
        return None
    query = (args.get("search_query") or item).strip()
    quantity = args.get("quantity") or 1
    try:
        return shufersal.add_item(query, quantity)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _remove(args: dict, ctx: ToolContext) -> str:
    groups = _parse(ctx.store.read(LIST))
    items, locs = _flat(groups)
    if not items:
        return json.dumps({"error": "list is empty", "list": {}})

    target = args.get("item", "").strip().lower()
    if target in ("", "last", "last one", "the last one", "last item"):
        idx = len(items) - 1
    else:
        idx = _match_index(items, target)
    if idx is None:
        return json.dumps({"error": f"no item matching '{target}'", "list": _nonempty(groups)})

    cat, pos = locs[idx]
    removed = groups[cat].pop(pos)
    ctx.store.write(LIST, _serialize(groups))
    return json.dumps({"removed": removed, "category": cat, "list": _nonempty(groups)})


def _show(args: dict, ctx: ToolContext) -> str:
    groups = _parse(ctx.store.read(LIST))
    sections = _sections(groups)
    items, _ = _flat(groups)

    if not sections:
        speech = "The list is empty."
        text = LIST_TITLE + ":\n(empty)"
    else:
        speech = "Here's the list. " + " ".join(
            f"{cat}: {', '.join(its)}." for cat, its in sections
        )
        text = LIST_TITLE + ":\n" + "\n".join(
            cat + ":\n" + "\n".join(f"• {i}" for i in its) for cat, its in sections
        )

    rendered = Rendered(
        title=LIST_TITLE,
        speech=speech,
        html=grouped_list_page(LIST_TITLE, sections),
        text=text,
        structured={"items": items, "sections": [{"category": c, "items": its} for c, its in sections]},
    )

    target = (args.get("target") or "").strip().lower()
    if target and target != ctx.channel.name and ctx.push:
        ok = ctx.push(target, rendered)
        if ok:
            return json.dumps({"sent_to": target, "items": items})
    ctx.channel.render(rendered)
    return json.dumps({"shown": True, "items": items})


_CATEGORY_GUIDE = (
    "Fruits; Vegetables; Dairy (milk, cheese, yoghurt, eggs, butter); "
    "'Meat & fish' (poultry, beef, lamb, fish, seafood); "
    "'Everything else' for food that fits none of the above (bread, pasta, snacks, spices); "
    "'General supplies' for non-food household items (foil, bags, cleaning, toiletries)."
)


TOOLS = [
    Tool(
        name="list_add",
        description=(
            "Add an item to the shared supermarket list immediately, no confirmation. "
            "This also drops the item into the real Shufersal online cart (one unit / "
            "package by default). Use for intents like 'we finished the garbage bags' or "
            "'I need more apples'. If the item is ambiguous (e.g. 'apples'), ask one "
            "clarifying question first, then call this with the specific item. If the "
            "user did not say HOW MANY, ask 'how many should I add?' and wait for the "
            "answer before calling this. Always classify the item into one section. "
            "Sections: " + _CATEGORY_GUIDE + " "
            "The result's 'cart' field reports the supermarket sync: if it is absent or "
            "its 'ok' is false, the item is on the list but NOT in the cart — say so "
            "plainly (e.g. 'added to your list, but I couldn't add it to the cart'); "
            "never claim it reached the cart unless 'ok' is true."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "The item to add, in the user's words."},
                "category": {
                    "type": "string",
                    "enum": CATEGORIES,
                    "description": "Which section the item belongs in.",
                },
                "search_query": {
                    "type": "string",
                    "description": (
                        "The item as a Hebrew supermarket search term, since the store's "
                        "catalog is Hebrew (e.g. item 'green apples' -> 'תפוחים ירוקים', "
                        "'milk' -> 'חלב', 'garbage bags' -> 'שקיות אשפה'). Keep it short."
                    ),
                },
                "quantity": {
                    "type": "number",
                    "description": (
                        "How many units/packages/kg to put in the cart. If the user did "
                        "NOT say how many, ask them ('how many should I add?') and wait "
                        "for the answer before calling this — do not silently default. "
                        "Only once you know the amount, pass it here ('six cottage "
                        "cheeses' -> 6, 'two kilos of apples' -> 2)."
                    ),
                },
            },
            "required": ["item", "category", "search_query"],
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
