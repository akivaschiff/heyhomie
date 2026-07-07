"""recipe_load — resolve a recipe (curated link or web search), process it, show it,
and leave it in conversation context for follow-up questions (steps, scaling,
substitutions, units) handled by the model over the retained recipe."""

import json

from homie.channels.base import Rendered
from homie.render import recipe_page
from homie.store import RECIPES
from homie.tools.base import Tool, ToolContext


def _curated(ctx: ToolContext) -> list[tuple[str, str]]:
    """recipes.txt holds 'name | url' per line."""
    pairs = []
    for line in ctx.store.lines(RECIPES):
        if "|" in line:
            name, url = line.split("|", 1)
            pairs.append((name.strip(), url.strip()))
    return pairs


_STOPWORDS = {"the", "for", "from", "with", "and", "recipe", "make", "show", "load"}


def _match_curated(query: str, ctx: ToolContext):
    terms = [t for t in query.lower().split() if len(t) > 2 and t not in _STOPWORDS]
    for name, url in _curated(ctx):
        hay = name.lower()
        if any(t in hay for t in terms):
            return name, url
    return None


def _candidates(args: dict, ctx: ToolContext):
    """Ordered (url, name, source) candidates to try until one extracts content."""
    query = args["query"].strip()
    prefer_curated = args.get("prefer_curated", True)
    out = []

    curated = _match_curated(query, ctx)
    if curated and prefer_curated:
        out.append((curated[1], curated[0], "curated"))

    if ctx.web_search:
        for r in ctx.web_search(f"recipe {query}"):
            if r.get("url"):
                out.append((r["url"], r.get("title", query), "search"))

    if curated and not prefer_curated:
        out.append((curated[1], curated[0], "curated"))
    return out


def _extract(url: str, fallback_name: str, ctx: ToolContext) -> dict:
    raw = ctx.fetch_url(url) if ctx.fetch_url else ""
    if ctx.recipe_extractor:
        recipe = ctx.recipe_extractor(url, raw)
    else:
        recipe = {"name": fallback_name, "ingredients": [], "steps": []}
    recipe.setdefault("name", fallback_name)
    if not recipe.get("name"):
        recipe["name"] = fallback_name
    return recipe


def _load(args: dict, ctx: ToolContext) -> str:
    candidates = _candidates(args, ctx)
    if not candidates:
        return json.dumps({"error": f"could not resolve a recipe for '{args['query']}'"})

    recipe = None
    source_kind = None
    for url, fallback_name, kind in candidates:
        try:
            extracted = _extract(url, fallback_name, ctx)
        except Exception:
            continue
        if extracted.get("ingredients") or extracted.get("steps"):
            recipe, source_kind = extracted, kind
            recipe["source"] = url
            break

    if recipe is None:
        return json.dumps(
            {"error": f"found pages for '{args['query']}' but couldn't read a recipe off them"}
        )

    if ctx.session is not None:
        ctx.session["recipe"] = recipe

    name = recipe["name"]
    source_url = recipe.get("source", "")
    ingredients = recipe.get("ingredients", [])
    steps = recipe.get("steps", [])
    rendered = Rendered(
        title=name,
        speech=f"Loaded {name} — {len(ingredients)} ingredients, {len(steps)} steps. Ask me anything about it.",
        html=recipe_page(name, ingredients, steps, source_url),
        text=f"{name}\nIngredients:\n"
        + "\n".join(f"• {i}" for i in ingredients)
        + "\nSteps:\n"
        + "\n".join(f"{n}. {s}" for n, s in enumerate(steps, 1)),
        structured=recipe,
    )
    ctx.channel.render(rendered)
    return json.dumps(
        {"loaded": name, "source": source_kind, "ingredients": ingredients, "steps": steps}
    )


TOOLS = [
    Tool(
        name="recipe_load",
        description=(
            "Resolve, process and display a recipe, then keep it available for follow-up "
            "questions. Two sources: a curated link ('chicken from <source>') matched "
            "against the saved recipe list, or a web search ('find me a recipe for X'). "
            "Set prefer_curated=false to force a web search. After loading, answer "
            "follow-ups (next step, how much flour, scale to 6, substitutions, metric "
            "units) yourself from the loaded recipe; re-call recipe_load only to re-render "
            "(e.g. after scaling)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Dish or 'dish from source'."},
                "prefer_curated": {
                    "type": "boolean",
                    "description": "Try curated links before web search (default true).",
                },
            },
            "required": ["query"],
        },
        handler=_load,
    ),
]
