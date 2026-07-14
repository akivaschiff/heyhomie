"""Live smoke test for the Shufersal connector. Proves the full loop against
the real site: search -> orders -> order detail -> add-to-cart -> read-back.

Only ever ADDS to cart (additive, reversible). Never checks out.

Run:  pi/.venv/bin/python -m homie.services.shufersal_smoke
Requires a valid cookie jar (env HOMIE_SHUFERSAL_COOKIES or secrets/shufersal_cookies.json).
"""

from homie.services import shufersal as sh


def main() -> None:
    session = sh.make_session()

    print("=== a. search('חלב') top 3 ===")
    products = sh.search("חלב", limit=10, session=session)
    for p in products[:3]:
        print(f"  {p.code}  {p.name!r}  {p.price_formatted}  {p.selling_method}  in_stock={p.in_stock}")
    milk = next((p for p in products if p.selling_method == "BY_UNIT" and p.in_stock), products[0])

    print("\n=== b. get_orders() ===")
    orders = sh.get_orders(session=session)
    print(f"  {len(orders)} past orders")
    recent = orders[0]["code"] if orders else None
    print(f"  most recent order code: {recent}")

    if recent:
        print("\n=== c. get_order(most recent) line items ===")
        order = sh.get_order(recent, session=session)
        items = sh.order_line_items(order)
        print(f"  {len(items)} line items; first 5:")
        for it in items[:5]:
            print(f"    {it['product_code']}  {it['name']!r}  qty={it['qty']}  {it['selling_method']}")

    print("\n=== d/e. cart before -> add milk -> cart after ===")
    before = sh.get_cart(session=session)
    print(f"  before: {before.line_count} lines, {before.total_units} units")
    print(f"  adding {milk.code} {milk.name!r} qty=1")
    result = sh.add_to_cart(milk.code, qty=1, selling_method=milk.selling_method, session=session)
    print(f"  add_to_cart -> {result}")
    after = sh.get_cart(session=session)
    print(f"  after:  {after.line_count} lines, {after.total_units} units")
    for line in after.lines:
        print(f"    {line.product_code}  {line.name!r}  qty={line.qty}  {line.selling_method}")


if __name__ == "__main__":
    main()
