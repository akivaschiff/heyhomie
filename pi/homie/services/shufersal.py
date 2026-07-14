"""Shufersal online-supermarket connector: search, cart, and order history.

Cookie-authenticated against www.shufersal.co.il. Reuses a Playwright
storageState cookie jar (captured out-of-band) — no login here. Injected via
ToolContext like the other services; pure requests + stdlib, no classes needed.
"""

from __future__ import annotations  # Pi runs Python 3.9; keep `X | None` annotations lazy

import json
import os
import re
from dataclasses import dataclass, field
from html import unescape

import requests

BASE = "https://www.shufersal.co.il/online/he"
HOST = "www.shufersal.co.il"
HOST_URL = "https://www.shufersal.co.il"

DEFAULT_COOKIE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "secrets",
    "shufersal_cookies.json",
)
STORAGE_STATE_ENV = "HOMIE_SHUFERSAL_COOKIES"

_NEEDED_COOKIE_NAMES = {
    "JSESSIONID",
    "XSRF-TOKEN",
    "miglog-cart",
    "acceleratorSecureGUID",
    "miglogstorefrontRememberMe",
    "accjwt",
    "AWSALB",
    "AWSALBCORS",
    "f5avraaaaaaaaaaaaaaaa_session_",
}


@dataclass
class Product:
    code: str
    sku: str
    name: str
    summary: str
    brand: str
    manufacturer: str
    price: float
    price_formatted: str
    selling_method: str
    in_stock: bool
    image_url: str
    unit_description: str
    min_qty: float
    max_qty: float


@dataclass
class CartLine:
    product_code: str
    name: str
    qty: float
    selling_method: str


@dataclass
class Cart:
    line_count: int
    total_units: float
    lines: list[CartLine] = field(default_factory=list)


def _load_storage_state(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cookies" in data:
        return data["cookies"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized cookie file shape: {path}")


def _shufersal_cookies(raw: list[dict]) -> list[dict]:
    return [c for c in raw if "shufersal.co.il" in c.get("domain", "")]


def extract_runtime_cookies(
    storage_state_path: str, out_path: str = DEFAULT_COOKIE_PATH
) -> str:
    """Distill a Playwright storageState into a compact name->value JSON scoped to
    www.shufersal.co.il, so the runtime never reaches into the capture dir."""
    raw = _shufersal_cookies(_load_storage_state(storage_state_path))
    compact = {
        c["name"]: c["value"]
        for c in raw
        if c["name"] in _NEEDED_COOKIE_NAMES or c["name"].startswith("TS")
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
    return out_path


def _cookie_map(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cookies" in data:
        return {c["name"]: c["value"] for c in _shufersal_cookies(data["cookies"])}
    if isinstance(data, list):
        return {c["name"]: c["value"] for c in _shufersal_cookies(data)}
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unrecognized cookie file shape: {path}")


USER_ENV = "HOMIE_SHUFERSAL_USER"
PASS_ENV = "HOMIE_SHUFERSAL_PASSWORD"


def _base_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Homie kitchen assistant)",
            "x-requested-with": "XMLHttpRequest",
            "Accept": "application/json, text/html;q=0.9,*/*;q=0.8",
        }
    )
    return session


def login(
    session: requests.Session | None = None,
    username: str | None = None,
    password: str | None = None,
) -> requests.Session:
    """Real form login. Captured cookies only ever yield an anonymous cart — the site
    keeps cart writes in a pre-login cart and offers to merge them on browser login.
    Logging in properly makes the session's cart the account cart, so adds land where
    the user sees them (and it sidesteps cookie expiry: just log in again)."""
    session = session or _base_session()
    username = username or os.environ.get(USER_ENV)
    password = password or os.environ.get(PASS_ENV)
    if not (username and password):
        raise RuntimeError(f"Shufersal credentials missing ({USER_ENV} / {PASS_ENV})")
    session.get(f"{BASE}/login", timeout=20)
    resp = session.post(
        f"{BASE}/j_spring_security_check",
        data={
            "fail_url": "/login/?error=true",
            "j_username": username,
            "j_password": password,
            "remember-me": "true",
            "CSRFToken": _prime_csrf(session),
        },
        # A browser FORM post, not XHR: the XHR/JSON headers route to a handler that
        # 405s. Send form/html headers (and drop x-requested-with) for this one call.
        headers={
            "x-requested-with": None,
            "Accept": "text/html,application/xhtml+xml",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": HOST_URL,
            "Referer": f"{BASE}/login",
        },
        timeout=25,
        allow_redirects=False,
    )
    if resp.status_code != 302 or "error=true" in resp.headers.get("Location", ""):
        raise RuntimeError(f"Shufersal login failed (status {resp.status_code})")
    # Bind this session to the persistent account cart. Without this, the session gets
    # its own transient empty cart, so writes never reach the cart the user sees.
    session.get(f"{BASE}/cart/load", params={"restoreCart": "true"}, timeout=20)
    return session


def make_session(cookie_path: str | None = None) -> requests.Session:
    """Prefer a real login (credentials in env); fall back to the captured cookie jar
    for read-only use when no credentials are set. Cart writes require login."""
    if os.environ.get(USER_ENV) and os.environ.get(PASS_ENV):
        return login()
    path = cookie_path or os.environ.get(STORAGE_STATE_ENV) or DEFAULT_COOKIE_PATH
    session = _base_session()
    for name, value in _cookie_map(path).items():
        if name == "miglog-cart":
            continue
        session.cookies.set(name, value, domain=HOST)
    return session


def _prime_csrf(session: requests.Session) -> str | None:
    """The XSRF-TOKEN rotates on every response, scoped to /online, and cart writes are
    checked against the current one — a stale token 302-redirects the write to the
    homepage. Rotations also leave duplicate cookies, so picking the wrong one fails
    every other add. Clear them, fetch exactly one fresh token, and use that."""
    for c in list(session.cookies):
        if c.name == "XSRF-TOKEN":
            try:
                session.cookies.clear(name=c.name, domain=c.domain, path=c.path)
            except KeyError:
                pass
    session.get(f"{BASE}/cart/load", timeout=20)
    scoped = [c.value for c in session.cookies if c.name == "XSRF-TOKEN" and c.path == "/online"]
    if scoped:
        return scoped[-1]
    any_tok = [c.value for c in session.cookies if c.name == "XSRF-TOKEN"]
    return any_tok[-1] if any_tok else None


def _first_image(images: list[dict]) -> str:
    if not images:
        return ""
    for want in ("medium", "product", "thumbnail", "small"):
        for img in images:
            if img.get("format") == want and img.get("url"):
                return img["url"]
    return images[0].get("url", "")


def _to_product(raw: dict) -> Product:
    price = raw.get("price") or {}
    selling = raw.get("sellingMethod") or {}
    stock = ((raw.get("stock") or {}).get("stockLevelStatus") or {}).get("code")
    brand = raw.get("brandName") or (raw.get("brand") or {}).get("name") or ""
    return Product(
        code=raw.get("code", ""),
        sku=raw.get("sku", ""),
        name=raw.get("name", ""),
        summary=raw.get("summary", ""),
        brand=brand,
        manufacturer=raw.get("manufacturer", ""),
        price=float(price.get("value") or 0.0),
        price_formatted=price.get("formattedValue", ""),
        selling_method=selling.get("code", "BY_UNIT"),
        in_stock=(stock == "inStock"),
        image_url=_first_image(raw.get("images") or []),
        unit_description=raw.get("unitDescription", ""),
        min_qty=float(raw.get("minOrderQuantity") or raw.get("minOrderWeight") or 0),
        max_qty=float(raw.get("maxOrderQuantity") or raw.get("maxOrderWeight") or 0),
    )


def search(term: str, limit: int = 10, session: requests.Session | None = None) -> list[Product]:
    session = session or make_session()
    resp = session.get(
        f"{BASE}/search/results",
        params={"q": f"{term}:relevance", "limit": limit},
        timeout=25,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [_to_product(r) for r in results[:limit]]


def add_to_cart(
    product_code: str,
    qty,
    selling_method: str = "BY_UNIT",
    comment: str = "",
    session: requests.Session | None = None,
) -> dict:
    """qty is the ABSOLUTE target quantity for this product in the cart, not a
    delta: add(code, 5) then add(code, 2) leaves 2 units, not 7. Requires the
    CSRFToken header (set on the session) or the server 302-redirects to home."""
    session = session or make_session()
    qty_str = str(qty)
    body = {
        "productCodePost": product_code,
        "productCode": product_code,
        "sellingMethod": selling_method,
        "qty": qty_str,
        "frontQuantity": qty_str,
        "comment": comment,
        "affiliateCode": "",
    }
    headers = {"content-type": "application/json"}
    csrf = _prime_csrf(session)
    if csrf:
        headers["CSRFToken"] = csrf
    resp = session.post(
        f"{BASE}/cart/add",
        params={
            "cartContext[openFrom]": "SEARCH",
            "cartContext[recommendationType]": "REGULAR",
        },
        json=body,
        headers=headers,
        timeout=25,
        allow_redirects=False,
    )
    ok = resp.status_code == 200
    out = {"ok": ok, "status": resp.status_code, "product_code": product_code, "qty": qty_str}
    ctype = resp.headers.get("content-type", "")
    if "json" in ctype:
        try:
            out["response"] = resp.json()
        except ValueError:
            out["response_text"] = resp.text[:300]
    else:
        out["response_text"] = resp.text[:300]
    return out


_ARTICLE_RE = re.compile(
    r'<article[^>]*class="miglog-prod[^"]*miglog-sellingmethod-(?P<method>by_unit|by_weight|by_package)'
    r'[^>]*data-product-code="(?P<code>[^"]+)"'
    r'[^>]*data-entry-qty="(?P<qty>[^"]+)"',
    re.DOTALL,
)
_NAME_RE = re.compile(r"aria-label=['\"]הסר מוצר&nbsp;([^'\"]+)['\"]")


def _parse_cart_html(html: str) -> Cart:
    names = [unescape(n).strip() for n in _NAME_RE.findall(html)]
    lines: list[CartLine] = []
    for i, m in enumerate(_ARTICLE_RE.finditer(html)):
        try:
            qty = float(m.group("qty"))
        except ValueError:
            qty = 0.0
        lines.append(
            CartLine(
                product_code=m.group("code"),
                name=names[i] if i < len(names) else "",
                qty=qty,
                selling_method=m.group("method").upper(),
            )
        )
    return Cart(
        line_count=len(lines),
        total_units=sum(line.qty for line in lines),
        lines=lines,
    )


def get_cart(session: requests.Session | None = None) -> Cart:
    session = session or make_session()
    resp = session.get(f"{BASE}/cart/load", params={"restoreCart": "true"}, timeout=25)
    resp.raise_for_status()
    return _parse_cart_html(resp.text)


def get_orders(session: requests.Session | None = None) -> list[dict]:
    session = session or make_session()
    resp = session.get(f"{BASE}/my-account/orders", timeout=25)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("activeOrders") or []) + (data.get("closedOrders") or [])


def get_order(order_code: str, session: requests.Session | None = None) -> dict:
    session = session or make_session()
    resp = session.get(f"{BASE}/my-account/orders/{order_code}", timeout=25)
    resp.raise_for_status()
    return resp.json()


def order_line_items(order: dict) -> list[dict]:
    entries = order.get("entries") or []
    items: list[dict] = []
    for e in entries:
        product = e.get("product") or {}
        selling = product.get("sellingMethod") or e.get("sellingMethod") or {}
        method = selling.get("code") if isinstance(selling, dict) else selling
        raw_qty = e.get("quantity")
        qty = raw_qty / 1000 if isinstance(raw_qty, (int, float)) else raw_qty
        items.append(
            {
                "product_code": product.get("code") or e.get("productCode"),
                "name": product.get("name") or e.get("name"),
                "qty": qty,
                "selling_method": method,
            }
        )
    return items


@dataclass
class HistoryItem:
    code: str
    name: str
    selling_method: str


@dataclass
class Resolution:
    code: str
    name: str
    selling_method: str
    price: str
    source: str  # "history" | "search"


def build_history_index(
    session: requests.Session | None = None, max_orders: int = 4
) -> list[HistoryItem]:
    """Products from recent orders, most-recent-first, deduped by code. This is the
    resolution prior: a term you've bought before resolves to that exact SKU. Order
    entries are not inline in the list response, so each order is fetched; capped to
    the most recent few to keep first-add latency low (staples repeat weekly)."""
    session = session or make_session()
    orders = get_orders(session)
    orders.sort(key=lambda o: o.get("created") or 0, reverse=True)
    index: list[HistoryItem] = []
    seen: set[str] = set()
    for order in orders[:max_orders]:
        code = order.get("code") or order.get("orderCode")
        if not code:
            continue
        try:
            detail = get_order(code, session)
        except requests.RequestException:
            continue
        for it in order_line_items(detail):
            pc = it["product_code"]
            if not pc or pc in seen:
                continue
            seen.add(pc)
            index.append(HistoryItem(pc, it["name"] or "", it["selling_method"] or "BY_UNIT"))
    return index


def _match_history(index: list[HistoryItem], query: str) -> HistoryItem | None:
    tokens = [t for t in query.lower().split() if len(t) >= 2]
    if not tokens:
        return None
    best, best_frac = None, 0.0
    for item in index:  # already most-recent-first; strict > keeps the most recent on ties
        words = item.name.lower().split()
        matched = sum(
            1
            for t in tokens
            if any(w == t or (len(t) >= 3 and w.startswith(t)) for w in words)
        )
        frac = matched / len(tokens)
        # Require most query words to land — otherwise 'תפוחי אדמה' (potato) matches
        # 'תפוחים' (apples) on one shared prefix. A loose substring is not enough.
        if frac >= 0.6 and frac > best_frac:
            best, best_frac = item, frac
    return best


class ShufersalCart:
    """The projection seam: resolve a free-text (Hebrew) grocery term to a real SKU —
    history first, catalog search second — and drop it in the live cart. Session and
    history index are built lazily and cached for the process lifetime."""

    def __init__(
        self,
        session_factory=make_session,
        max_history_orders: int = 3,
        search_fallback: bool = False,
    ):
        self._factory = session_factory
        self._max = max_history_orders
        self._search_fallback = search_fallback
        self._session: requests.Session | None = None
        self._history: list[HistoryItem] | None = None

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = self._factory()
        return self._session

    def history(self) -> list[HistoryItem]:
        if self._history is None:
            self._history = build_history_index(self.session, self._max)
        return self._history

    def resolve(self, query: str) -> Resolution | None:
        hit = _match_history(self.history(), query)
        if hit:
            return Resolution(hit.code, hit.name, hit.selling_method, "", "history")
        if not self._search_fallback:
            return None
        results = search(query, session=self.session)
        ranked = [p for p in results if p.in_stock] or results
        if not ranked:
            return None
        p = ranked[0]
        return Resolution(p.code, p.name or p.summary, p.selling_method, p.price_formatted, "search")

    def add_item(self, query: str, qty=1) -> dict:
        res = self.resolve(query)
        if res is None:
            return {"ok": False, "reason": "not_found", "query": query}
        out = add_to_cart(res.code, qty, res.selling_method, session=self.session)
        if not out.get("ok"):
            # The login session expires on a long-lived service; a failed write is
            # usually a dead session. Re-login once and retry before giving up.
            self._session = None
            out = add_to_cart(res.code, qty, res.selling_method, session=self.session)
        return {
            "ok": bool(out.get("ok")),
            "status": out.get("status"),
            "source": res.source,
            "code": res.code,
            "name": res.name,
            "price": res.price,
            "selling_method": res.selling_method,
            "qty": qty,
        }
