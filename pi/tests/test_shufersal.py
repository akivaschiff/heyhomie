"""History matcher + verified-add behavior. No network: pure functions and
injected fakes only."""

import homie.services.shufersal as sh
from homie.services.shufersal import HistoryItem, ShufersalCart, _match_history

HISTORY = [
    HistoryItem("P_SAUCE", 'רסק תפו"ע ללא תוספת סוכר', "BY_UNIT"),
    HistoryItem("P_MILK", "חלב בקרטון 3% שומן 1 ל'", "BY_UNIT"),
    HistoryItem("P_EGGS", "ביצים ארוזות L", "BY_UNIT"),
    HistoryItem("P_APPLES", "מארז תפוחים סמיט", "BY_PACKAGE"),
    HistoryItem("P_YELLOW", "גבינה צהובה פרוסה 22%", "BY_UNIT"),
]


def test_qualifier_word_does_not_resolve():
    assert _match_history(HISTORY, "סוכר") is None


def test_head_noun_resolves():
    assert _match_history(HISTORY, "חלב").code == "P_MILK"
    assert _match_history(HISTORY, "ביצים").code == "P_EGGS"


def test_packaging_word_is_skipped_to_head():
    assert _match_history(HISTORY, "תפוחים").code == "P_APPLES"


def test_partial_match_still_rejected():
    assert _match_history(HISTORY, "גבינת קוטג") is None


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeSession:
    def __init__(self, cart_codes):
        self.cart_codes = cart_codes

    def get(self, url, **kwargs):
        articles = "".join(
            f'<article class="miglog-prod miglog-sellingmethod-by_unit" '
            f'data-product-code="{c}" data-entry-qty="1">' for c in self.cart_codes
        )
        return FakeResponse(articles)


def _cart_with(monkeypatch, add_results, session_carts):
    """ShufersalCart whose add_to_cart and sessions are scripted per attempt."""
    sessions = [FakeSession(codes) for codes in session_carts]
    cart = ShufersalCart(session_factory=lambda: sessions.pop(0))
    cart._history = HISTORY
    calls = []

    def fake_add(code, qty, method, session=None):
        calls.append(session)
        return add_results.pop(0)

    monkeypatch.setattr(sh, "add_to_cart", fake_add)
    return cart, calls


def test_add_verified_in_cart(monkeypatch):
    cart, calls = _cart_with(
        monkeypatch,
        add_results=[{"ok": True, "status": 200}],
        session_carts=[["P_MILK"]],
    )
    out = cart.add_item("חלב", 1)
    assert out["ok"] is True
    assert len(calls) == 1


def test_add_200_but_missing_from_cart_relogs_and_retries(monkeypatch):
    cart, calls = _cart_with(
        monkeypatch,
        add_results=[{"ok": True, "status": 200}, {"ok": True, "status": 200}],
        session_carts=[[], ["P_MILK"]],
    )
    out = cart.add_item("חלב", 1)
    assert out["ok"] is True
    assert len(calls) == 2
    assert calls[0] is not calls[1]


def test_add_fails_honestly_when_never_lands(monkeypatch):
    cart, calls = _cart_with(
        monkeypatch,
        add_results=[
            {"ok": False, "status": 0, "reason": "session_expired"},
            {"ok": False, "status": 0, "reason": "session_expired"},
        ],
        session_carts=[[], []],
    )
    out = cart.add_item("חלב", 1)
    assert out["ok"] is False
    assert out["reason"] == "session_expired"
    assert len(calls) == 2
