import os
import sys
import json
import html
import asyncio
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request, Response, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "pi")))

from homie.services.cron import CronStore, Entry, build_cron_line, build_curl

SCHEDULE_BASE_URL = os.environ.get("HOMIE_SMARTHOME_URL", "http://localhost:8787").rstrip("/")
SCHEDULE_RECURS = ("once", "daily", "weekdays", "weekends")
_cron = CronStore()

LIST_FILE = os.environ.get(
    "HOMIE_LIST_FILE",
    os.path.normpath(os.path.join(HERE, "..", "pi", ".homie-state", "list.txt")),
)

KIOSK_FILE = os.environ.get(
    "HOMIE_KIOSK_FILE",
    os.path.normpath(os.path.join(HERE, "..", "pi", ".homie-state", "kiosk.txt")),
)

import higoal_cli
import electra_cli
import midea_cli

app = Flask(__name__)

# ---------- Higoal (persistent socket) ----------
_higoal_lock = threading.Lock()
_higoal = {"manager": None}


def higoal_manager():
    with _higoal_lock:
        if _higoal["manager"] is None:
            _higoal["manager"] = higoal_cli.connect(settle=4.0)
        return _higoal["manager"]


def higoal_entity(device_id, idx):
    m = higoal_manager()
    for dev in m.device_map.values():
        if getattr(dev, "id", None) == device_id and hasattr(dev, "entities"):
            for e in dev.entities:
                if e.id == int(idx):
                    return e
    return None


# ---------- Electra ----------
electra_cli._load_env()


# ---------- routes ----------
@app.get("/")
def index():
    with open(os.path.join(HERE, "dashboard.html"), encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html")


@app.get("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(os.path.join(HERE, "assets"), filename)


LIST_SECTIONS = [
    "Fruits", "Vegetables", "Dairy", "Meat & fish", "Everything else", "General supplies",
]


@app.get("/list")
def shopping_list():
    try:
        with open(LIST_FILE, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""
    return Response(_list_html(_parse_sections(text)), mimetype="text/html")


# ---------- kiosk view switching ----------
# The tablet lives on /kiosk, an iframe shell that polls /api/kiosk/view once a
# second and swaps to the selected page. Voice (kiosk_show tool) and the dashboard
# both drive it by writing KIOSK_FILE. VIEWS maps a view token -> the page URL.
KIOSK_VIEWS = {"dashboard": "/", "list": "/list"}
DEFAULT_VIEW = "dashboard"


def _read_view():
    try:
        with open(KIOSK_FILE, encoding="utf-8") as f:
            v = f.read().strip()
    except FileNotFoundError:
        v = ""
    return v if v in KIOSK_VIEWS else DEFAULT_VIEW


@app.get("/api/kiosk/view")
def api_kiosk_view():
    return jsonify({"view": _read_view()})


@app.post("/api/kiosk/view")
def api_kiosk_view_set():
    body = request.get_json(force=True)
    view = str(body.get("view", "")).strip().lower()
    if view not in KIOSK_VIEWS:
        return jsonify({"error": "unknown view", "views": list(KIOSK_VIEWS)}), 400
    os.makedirs(os.path.dirname(KIOSK_FILE), exist_ok=True)
    with open(KIOSK_FILE, "w", encoding="utf-8") as f:
        f.write(view + "\n")
    return jsonify({"ok": True, "view": view})


_KIOSK_SHELL = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    "<title>Homie Kiosk</title><style>"
    "html,body{margin:0;height:100%;background:#191919;overflow:hidden;}"
    "iframe{border:0;width:100%;height:100%;display:block;}"
    "</style></head><body><iframe id='v'></iframe><script>"
    "const VIEWS={dashboard:'/',list:'/list'};let cur=null;"
    "async function tick(){try{"
    "const r=await fetch('/api/kiosk/view',{cache:'no-store'});"
    "const d=await r.json();"
    "if(d.view!==cur&&VIEWS[d.view]){cur=d.view;"
    "document.getElementById('v').src=VIEWS[d.view];}"
    "}catch(e){}}"
    "tick();setInterval(tick,1000);"
    "</script></body></html>"
)


@app.get("/kiosk")
def kiosk():
    return Response(_KIOSK_SHELL, mimetype="text/html")


def _parse_sections(text):
    groups = {c: [] for c in LIST_SECTIONS}
    current = "Everything else"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# "):
            name = line[2:].strip()
            current = name if name in groups else "Everything else"
        else:
            groups[current].append(line)
    return [(c, groups[c]) for c in LIST_SECTIONS if groups[c]]


def _list_html(sections):
    if sections:
        total = sum(len(items) for _, items in sections)
        head = f"<p class='meta'>{total} item{'s' if total != 1 else ''}</p>"
        body = head + "".join(
            f"<h2>{html.escape(cat)}</h2><ul>"
            + "".join(f"<li>{html.escape(i)}</li>" for i in items)
            + "</ul>"
            for cat, items in sections
        )
    else:
        body = "<p class='empty'>Nothing on the list yet.</p>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<meta http-equiv='refresh' content='30'>"
        "<title>Shopping List</title><style>"
        ":root{color-scheme:light dark;"
        "--bg:#ffffff;--fg:#37352f;--muted:#9b9a97;--line:#ecebe9;--bullet:#d3d1cb;}"
        "@media (prefers-color-scheme:dark){:root{"
        "--bg:#191919;--fg:#e9e9e7;--muted:#8f8f8c;--line:#2a2a2a;--bullet:#4a4a48;}}"
        "*{box-sizing:border-box;}"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;"
        "margin:0;background:var(--bg);color:var(--fg);"
        "-webkit-font-smoothing:antialiased;line-height:1.5;}"
        "main{max-width:640px;margin:0 auto;padding:56px 24px 80px;}"
        "h1{font-size:2rem;font-weight:700;letter-spacing:-0.02em;margin:0 0 2px;}"
        "h2{font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;"
        "color:var(--muted);margin:28px 0 2px;}"
        ".meta{color:var(--muted);font-size:0.9rem;margin:0 0 8px;}"
        "ul{list-style:none;padding:0;margin:0;}"
        "li{font-size:1.05rem;padding:11px 0 11px 26px;position:relative;"
        "border-bottom:1px solid var(--line);}"
        "li:before{content:'';position:absolute;left:6px;top:1.35em;width:6px;height:6px;"
        "border-radius:50%;background:var(--bullet);}"
        ".empty{color:var(--muted);font-size:1.05rem;margin:20px 0 0;}"
        "</style></head><body><main><h1>Shopping List</h1>"
        f"{body}</main></body></html>"
    )


@app.get("/api/higoal")
def api_higoal():
    m = higoal_manager()
    return jsonify(higoal_cli.snapshot(m))


@app.post("/api/higoal/set")
def api_higoal_set():
    body = request.get_json(force=True)
    e = higoal_entity(body["device"], body["idx"])
    if e is None:
        return jsonify({"error": "not found"}), 404
    e.turn_on() if body["on"] else e.turn_off()
    return jsonify({"ok": True})


# ---------- schedules (shared crontab block with the LLM's schedule tools) ----------
def _today():
    return datetime.now().strftime("%Y-%m-%d")


@app.get("/api/schedules")
def api_schedules_list():
    entries = _cron.prune_stale(_today())
    return jsonify([
        {"id": e.id, "recur": e.recur, "date": e.date or None, "description": e.description}
        for e in entries
    ])


@app.post("/api/schedules")
def api_schedules_create():
    body = request.get_json(force=True)
    recur = body.get("recur", "once")
    if recur not in SCHEDULE_RECURS:
        return jsonify({"error": f"recur must be one of {SCHEDULE_RECURS}"}), 400
    date = body.get("date") or ""
    if recur == "once" and not date:
        return jsonify({"error": "one-time schedules need a date (YYYY-MM-DD)"}), 400
    commands = body.get("commands") or []
    if not commands:
        return jsonify({"error": "no commands"}), 400

    curls = [build_curl(SCHEDULE_BASE_URL, c["system"], json.dumps(c["payload"])) for c in commands]
    entries = _cron.prune_stale(_today())
    entry = Entry(
        id=_cron.next_id(),
        recur=recur,
        date=date,
        description=body.get("description", ""),
        cron_line=build_cron_line(body["time"], recur, date, curls),
    )
    entries.append(entry)
    _cron.save_entries(entries)
    return jsonify({"id": entry.id, "description": entry.description})


@app.delete("/api/schedules/<sid>")
def api_schedules_delete(sid):
    entries = _cron.prune_stale(_today())
    keep = [e for e in entries if e.id != sid]
    if len(keep) == len(entries):
        return jsonify({"error": f"no schedule '{sid}'"}), 404
    _cron.save_entries(keep)
    return jsonify({"deleted": sid})


@app.get("/api/electra")
def api_electra():
    devs = electra_cli.devices()
    with ThreadPoolExecutor(max_workers=len(devs) or 1) as ex:
        states = list(ex.map(_electra_state_safe, devs))
    return jsonify(states)


def _electra_state_safe(d):
    try:
        return electra_cli.read_state(d)
    except Exception as e:
        return {"id": d["id"], "name": d["name"], "on": None, "error": str(e)}


@app.post("/api/electra/set")
def api_electra_set():
    body = request.get_json(force=True)
    electra_cli.set_state(
        body["id"],
        power=body.get("power"),
        mode=body.get("mode"),
        temp=body.get("temp"),
        fan=body.get("fan"),
    )
    return jsonify({"ok": True})


@app.get("/api/midea")
def api_midea():
    devices = midea_cli.load_devices()
    results = asyncio.run(_midea_states(devices))
    return jsonify(results)


async def _midea_states(devices):
    out = await asyncio.gather(*[midea_cli.state(d) for d in devices], return_exceptions=True)
    return [r if not isinstance(r, Exception) else {"name": d["name"], "online": False, "error": str(r)}
            for d, r in zip(devices, out)]


@app.post("/api/midea/set")
def api_midea_set():
    body = request.get_json(force=True)
    devices = {d["name"]: d for d in midea_cli.load_devices()}
    d = devices[body["id"]]
    asyncio.run(midea_cli.control(d, power=body.get("power"), mode=body.get("mode"), temp=body.get("temp"), fan=body.get("fan")))
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8787, threaded=True)
