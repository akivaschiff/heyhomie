# Home Control — HTTP API contract (STABLE)

The Flask server (`server.py`) serves the dashboard at `/` and exposes these JSON endpoints.
The frontend (`dashboard.html`) is a single self-contained file rendered against this contract.
**Do not change endpoint paths or payload field names** — the backend and homie tools depend on them.

Base: `http://<host>:8787`

## Lights & Blinds — Higoal

`GET /api/higoal` → array of devices (wall panels):
```json
[{ "device": "פאנל מטבח", "id": "MCKBTZ", "model": "2R",
   "entities": [
     { "idx": 2, "name": "top light", "type": "switch", "online": true, "on": true },
     { "idx": 0, "name": "תריס מטבח", "type": "shutter", "online": true, "on": false, "state": "closed" }
   ]}]
```
- `type` ∈ `"switch"` | `"dimmer/light"` | `"shutter"`.
- Switches/dimmers → render a light toggle (use `on`).
- Shutters → render Open/Close. The "open" button uses `idx`; the paired "close" button uses `idx+1`.
- Shutter entities also carry `state` ∈ `"open"` | `"closed"` | `null`, derived from the latched relay status bytes (persists across the device's own reports); `on` and the raw `percentage` byte are unreliable for shutters. `null` before the blind has been actuated since connect. Only present on `"shutter"` entities.
- Entities whose `name` is empty or matches `channel <n>` are unconfigured — hide them.
- `on`/`online` may be `null` before first state arrives.

`POST /api/higoal/set` → `{ "device": "<id>", "idx": <int>, "on": <bool> }`
- Cloud device: state confirmation lags a few seconds. Use **optimistic UI** — flip instantly, reconcile with a delayed refetch (~3s).

## Central A/C — Electra

`GET /api/electra` → array:
```json
[{ "id": 236084, "name": "קומת קרקע", "kind": "מיני מרכזי",
   "on": true, "mode": "COOL", "fan": "MED", "target": 21, "current": 20,
   "modes": ["STBY","COOL","FAN","DRY","HEAT","AUTO"],
   "fans": ["AUTO","LOW","MED","HIGH"], "min": 16, "max": 30 }]
```
- `on: null` + `error` field means that unit failed to read.
- Also cloud — state lags a few seconds → optimistic UI.

`POST /api/electra/set` → any subset of:
`{ "id": <int>, "power": <bool>, "mode": "<mode>", "temp": <int>, "fan": "<fan>" }`
- `power:false` (or `mode:"STBY"`) turns off. `power:true` with no mode defaults to COOL.

## Split A/C — Midea (local LAN, fast)

`GET /api/midea` → array:
```json
[{ "name": "The Office", "id": "net_ac_4DE0", "ip": "192.168.68.61",
   "online": true, "power": false, "mode": "COOL",
   "modes": ["AUTO","COOL","DRY","HEAT","FAN_ONLY","SMART_DRY"],
   "fan": "MEDIUM", "fans": ["AUTO","MAX","HIGH","MEDIUM","LOW","SILENT"],
   "target": 23.0, "indoor": 25.5, "min": 16, "max": 30 }]
```
- `name` = friendly room name; `id` = stable key to send commands with.
- `indoor` = live measured room temp; `target` = setpoint.

`POST /api/midea/set` → any subset of:
`{ "id": "<net_ac key>", "power": <bool>, "mode": "<mode>", "temp": <number>, "fan": "<fan>" }`
- Local + fast; state reflects almost immediately.

## Notes
- All POST bodies are JSON. All endpoints return `{ "ok": true }` on success.
- Hebrew names are common and must render correctly (RTL-aware where shown).
- Sections to present: **Lights**, **Blinds**, **Central A/C**, **Split A/C**.
