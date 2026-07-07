"""HTML renderers for screen surfaces (tablet / Mac browser). Self-contained pages,
no external assets, kitchen-legible at a glance."""

import html

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0;
    padding: 6vw; background: #fafaf7; color: #1a1a1a; }}
  h1 {{ font-size: 7vw; margin: 0 0 4vw; }}
  h2 {{ font-size: 4vw; margin: 6vw 0 1vw; color: #777;
    text-transform: uppercase; letter-spacing: 0.08em; }}
  ul {{ list-style: none; padding: 0; margin: 0; }}
  li {{ font-size: 5.5vw; padding: 3vw 0; border-bottom: 1px solid #ddd; }}
  .meta {{ color: #777; font-size: 4vw; }}
  ol {{ font-size: 5vw; line-height: 1.6; padding-left: 6vw; }}
  .ingredient {{ font-size: 5vw; padding: 2vw 0; border-bottom: 1px solid #eee; }}
  a {{ color: #2a6; }}
</style></head><body>{body}</body></html>"""


def page(title: str, body: str) -> str:
    return _PAGE.format(title=html.escape(title), body=body)


def list_page(title: str, items: list[str]) -> str:
    if items:
        rows = "".join(f"<li>{html.escape(i)}</li>" for i in items)
        body = f"<h1>{html.escape(title)}</h1><ul>{rows}</ul>"
    else:
        body = f"<h1>{html.escape(title)}</h1><p class='meta'>empty</p>"
    return page(title, body)


def grouped_list_page(title: str, sections: list[tuple[str, list[str]]]) -> str:
    if not sections:
        body = f"<h1>{html.escape(title)}</h1><p class='meta'>empty</p>"
        return page(title, body)
    blocks = "".join(
        f"<h2>{html.escape(cat)}</h2><ul>"
        + "".join(f"<li>{html.escape(i)}</li>" for i in items)
        + "</ul>"
        for cat, items in sections
    )
    return page(title, f"<h1>{html.escape(title)}</h1>{blocks}")


def recipe_page(name: str, ingredients: list[str], steps: list[str], source: str = "") -> str:
    ing = "".join(f"<div class='ingredient'>{html.escape(i)}</div>" for i in ingredients)
    st = "".join(f"<li>{html.escape(s)}</li>" for s in steps)
    src = f"<p class='meta'><a href='{html.escape(source)}'>source</a></p>" if source else ""
    body = (
        f"<h1>{html.escape(name)}</h1>{src}"
        f"<h2>Ingredients</h2>{ing}"
        f"<h2>Steps</h2><ol>{st}</ol>"
    )
    return page(name, body)


def shabbat_page(location: str, rows: list[tuple[str, str]]) -> str:
    items = "".join(
        f"<li>{html.escape(label)} <span class='meta'>{html.escape(value)}</span></li>"
        for label, value in rows
    )
    body = f"<h1>Shabbat — {html.escape(location)}</h1><ul>{items}</ul>"
    return page(f"Shabbat — {location}", body)
