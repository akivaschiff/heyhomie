# Home Control

Direct control of the house — no vendor apps. A small Flask server serves a tablet
dashboard and talks to three systems through thin driver modules.

| System | Devices | Transport | Driver |
|---|---|---|---|
| **Higoal** | lights, blinds, boiler | vendor cloud (HTTPS `:8143` + TCP socket `:17670`) | `higoal_cli.py` + vendored `higoal_client/` |
| **Electra** | central A/C (`קומת קרקע`) | vendor cloud (REST, phone + SMS OTP) | `electra_cli.py` (`electrasmart`) |
| **Midea** | 4 split A/Cs by room | **local LAN** (port 6444, no cloud) | `midea_cli.py` (`msmart-ng`) |

Only Midea is local; Higoal and Electra are cloud round-trips (a few seconds of latency).

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cp .env.example .env          # fill in Higoal creds + Electra phone
./.venv/bin/python electra_cli.py request        # sends SMS OTP
./.venv/bin/python electra_cli.py token <otp>    # saves Electra imei+token to .env
./.venv/bin/python midea_cli.py discover         # scans LAN -> midea_devices.json
```

Room names for the Midea units live in `midea_devices.json` under `"room"`. `discover`
preserves them across re-scans. `.env`, `.electra_imei`, and `midea_devices.json` hold
secrets and are git-ignored.

## Run

```bash
./.venv/bin/python server.py     # http://<host>:8787
```

The HTTP API the dashboard uses is documented in `API_CONTRACT.md`.

## CLI (per system, without the web UI)

```bash
./.venv/bin/python higoal_cli.py list | off-lights
./.venv/bin/python electra_cli.py state | on [id] | off [id]
./.venv/bin/python midea_cli.py  list | on <room> | off <room> | discover
```

## Quick know-how (for future simple changes)

File map:
- `server.py` — Flask app: serves the dashboard at `/` and the 6 JSON endpoints. Port at the bottom (`8787`).
- `dashboard.html` — the entire UI, one self-contained file (inline CSS+JS, no dependencies). Server reads it **fresh per request**, so just edit and refresh the browser — no restart.
- `higoal_cli.py` / `electra_cli.py` / `midea_cli.py` — the device drivers. Each is also runnable standalone (see CLI section). Each exposes a "read state" + "set" surface that `server.py` calls.
- `higoal_client/` — vendored Higoal protocol library (don't edit unless upstream changes).
- `.env` — secrets (git-ignored). `midea_devices.json` — Midea room names + local keys (git-ignored).
- `API_CONTRACT.md` — the HTTP shapes the dashboard depends on. Keep it current if you change an endpoint.

Common tweaks:
- **Rename a Midea room** → edit its `"room"` in `midea_devices.json`, refresh.
- **Rename a Higoal light/blind** → names come from the Higoal app; rename there, then refresh (mind the single-connection caveat below).
- **Change the web port** → `server.py`, `app.run(..., port=8787)`.
- **Restart the server** → `pkill -f server.py; ./.venv/bin/python server.py &`.
- **Add a control to a device** → extend the driver's set-function + the matching card in `dashboard.html`, and update `API_CONTRACT.md`.

Latency reality: Higoal & Electra are **cloud** (a few seconds to confirm; the UI is optimistic). Midea is **local** (instant).

## Moving to the Pi

The Pi becomes the always-on host so the tablet has a permanent URL. Notes for whoever does the port:

**Hard requirements**
- **Python 3.11+** is fine (Raspberry Pi OS Bookworm's default 3.11 works — the one PEP-695 line that needed 3.12 has been patched out of `higoal_client/manager.py`). Verify: `python3 --version`.
- The Pi **must be on the same LAN/subnet as the Midea ACs** (`192.168.68.x`). Midea control is local (UDP discovery + TCP `:6444`); it will NOT work from another network. Higoal & Electra are cloud, so they only need internet.

**Steps**
1. Get the code onto the Pi (git clone the repo, or rsync `smarthome/`). Secrets are **not** in git.
2. `python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt`
3. Bring secrets over:
   - `.env` — copy from the Mac as-is. The Higoal creds and the Electra `imei`/`token` are host-independent and long-lived. (Or recreate from `.env.example` and re-run `electra_cli.py request` + `token`.)
   - `midea_devices.json` — copy the Mac's file over first (to keep the room names), then run `./.venv/bin/python midea_cli.py discover` **on the Pi** to refresh IPs/keys for the Pi's view of the LAN. `discover` preserves the `room` fields.
4. Smoke test: `./.venv/bin/python server.py`, then browse `http://<pi-ip>:8787`.
5. Run on boot with systemd (template `pi/homie.service` exists in this repo). Minimal unit:
   ```ini
   [Unit]
   Description=Home Control dashboard
   After=network-online.target
   Wants=network-online.target
   [Service]
   WorkingDirectory=/home/pi/homie/smarthome
   ExecStart=/home/pi/homie/smarthome/.venv/bin/python server.py
   Restart=on-failure
   User=pi
   [Install]
   WantedBy=multi-user.target
   ```
   `sudo systemctl enable --now home-control` after placing it in `/etc/systemd/system/`.

**Gotchas**
- **Higoal allows only ONE active connection per account**, and connecting logs the phone app out. Do **not** run the Mac server and the Pi server at the same time on the same Higoal account. For an always-on Pi, create a dedicated **second Higoal account** and share the home to it.
- Give the Pi a **static IP / DHCP reservation** so the tablet's kiosk Start URL never breaks.
- Open inbound **TCP 8787** if the Pi has a firewall.
- The HA `docker-compose.yml` is **not** needed on the Pi — the standalone server is the path.

## Notes
- `higoal_client/` is vendored from https://github.com/Minitour/ha-higoal (MIT); `manager.py` has a one-line local patch for Python 3.11 compatibility.
- `docker-compose.yml` runs Home Assistant as an optional alternative gateway; the
  standalone server above is the primary path (and the base for homie tool integration).
