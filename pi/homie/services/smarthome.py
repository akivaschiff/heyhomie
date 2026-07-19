"""HTTP client for the smarthome server (smarthome/server.py, port 8787).

Always through the server, never the vendor CLIs directly: the server owns the
single allowed Higoal connection, and its host runs the Python the vendor libs
need — homie only speaks the stable contract in smarthome/API_CONTRACT.md.
"""

import os
import time

import requests

DEFAULT_URL = "http://localhost:8787"


class SmartHomeClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or os.environ.get("HOMIE_SMARTHOME_URL", DEFAULT_URL)).rstrip("/")

    def get(self, system: str) -> list:
        # The Electra cloud read (token + device list) blips intermittently; one
        # retry turns a transient 500/timeout into a success instead of failing
        # the caller's whole request.
        last = None
        for attempt in range(2):
            try:
                resp = requests.get(f"{self.base_url}/api/{system}", timeout=15)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last = exc
                if attempt == 0:
                    time.sleep(1)
        raise last

    def set(self, system: str, payload: dict) -> dict:
        # Electra verifies + retries server-side (a few cloud round-trips), so the
        # set call can take longer than a read; give it room before timing out.
        resp = requests.post(f"{self.base_url}/api/{system}/set", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
