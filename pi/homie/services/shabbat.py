"""Shabbat times via the Hebcal JSON API. Display-only data."""

import requests


def fetch_shabbat_times(geoname_id: str) -> dict:
    resp = requests.get(
        "https://www.hebcal.com/shabbat",
        params={"cfg": "json", "geonameid": geoname_id, "b": "18", "M": "on"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    location = data.get("location", {}).get("title", "")
    rows = []
    for item in data.get("items", []):
        category = item.get("category")
        if category == "candles":
            rows.append(("Candle lighting", _time(item)))
        elif category == "havdalah":
            rows.append(("Havdalah", _time(item)))
        elif category == "parashat":
            rows.append(("Parasha", item.get("title", "").replace("Parashat ", "")))
    return {"location": location, "rows": rows}


def _time(item: dict) -> str:
    title = item.get("title", "")
    # "Candle lighting: 7:12pm" -> "7:12pm"
    return title.split(": ", 1)[1] if ": " in title else title
