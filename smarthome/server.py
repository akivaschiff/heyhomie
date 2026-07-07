import os
import sys
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, request, Response

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

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
