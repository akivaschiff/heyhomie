import os
import sys
import json
import time
import asyncio

from msmart.device import AirConditioner as AC
from msmart.discover import Discover
from msmart.lan import AuthenticationError, ProtocolError

HERE = os.path.dirname(__file__)
DEVICES = os.path.join(HERE, "midea_devices.json")

_ip_by_id = {}
_ip_by_id_at = 0.0
_IP_TTL = 20.0


def load_devices():
    with open(DEVICES) as f:
        return json.load(f)


async def discover_and_save():
    """Scan the LAN for Midea ACs, preserving existing room names by device name."""
    rooms = {}
    if os.path.exists(DEVICES):
        rooms = {d["name"]: d.get("room") for d in load_devices()}
    found = await Discover.discover()
    devices = []
    for dev in found:
        entry = {"name": dev.name, "ip": dev.ip, "id": dev.id,
                 "token": dev.token, "key": dev.key}
        if rooms.get(dev.name):
            entry["room"] = rooms[dev.name]
        devices.append(entry)
    devices.sort(key=lambda d: d["ip"])
    with open(DEVICES, "w") as f:
        json.dump(devices, f, indent=2)
    return devices


def _persist_ips(ip_by_id):
    if not os.path.exists(DEVICES):
        return
    devices = load_devices()
    changed = False
    for x in devices:
        ip = ip_by_id.get(int(x["id"]))
        if ip and x.get("ip") != ip:
            x["ip"] = ip
            changed = True
    if changed:
        with open(DEVICES, "w") as f:
            json.dump(devices, f, indent=2, ensure_ascii=False)


async def _resolve_ips():
    global _ip_by_id, _ip_by_id_at
    found = await Discover.discover(auto_connect=False, timeout=4)
    _ip_by_id = {int(x.id): x.ip for x in found}
    _ip_by_id_at = time.monotonic()
    _persist_ips(_ip_by_id)
    return _ip_by_id


async def _open(ip, d):
    ac = AC(ip=ip, device_id=int(d["id"]), port=6444)
    await ac.authenticate(d["token"], d["key"])
    await ac.refresh()
    return ac


async def connect(d):
    try:
        return await _open(d["ip"], d)
    except (AuthenticationError, ProtocolError, OSError):
        did = int(d["id"])
        ip = _ip_by_id.get(did) if (time.monotonic() - _ip_by_id_at) < _IP_TTL else None
        if not ip or ip == d["ip"]:
            ip = (await _resolve_ips()).get(did)
        if not ip or ip == d["ip"]:
            raise
        d["ip"] = ip
        return await _open(ip, d)


async def state(d):
    ac = await connect(d)
    return {
        "name": d.get("room") or d["name"],
        "id": d["name"],
        "ip": d["ip"],
        "online": ac.online,
        "power": ac.power_state,
        "mode": ac.operational_mode.name if ac.operational_mode else None,
        "modes": [m.name for m in ac.supported_operation_modes] if ac.supported_operation_modes else [m.name for m in AC.OperationalMode],
        "fan": ac.fan_speed.name if hasattr(ac.fan_speed, "name") else ac.fan_speed,
        "fans": [f.name for f in ac.supported_fan_speeds] if ac.supported_fan_speeds else [f.name for f in AC.FanSpeed],
        "target": ac.target_temperature,
        "indoor": ac.indoor_temperature,
        "min": ac.min_target_temperature,
        "max": ac.max_target_temperature,
    }


async def control(d, power=None, mode=None, temp=None, fan=None):
    ac = await connect(d)
    turn_on = power is None and (mode is not None or temp is not None or fan is not None)
    if power is not None:
        ac.power_state = power
    elif turn_on:
        ac.power_state = True
    if mode is not None:
        ac.operational_mode = AC.OperationalMode[mode]
    if temp is not None:
        ac.target_temperature = float(temp)
    if fan is not None:
        ac.fan_speed = AC.FanSpeed[fan]
    await ac.apply()
    return d["name"]


async def set_power(d, on, mode=None, temp=None, fan=None):
    return await control(d, power=on, mode=mode, temp=temp, fan=fan)


def _match(devices, target):
    if target is None:
        return devices
    return [d for d in devices if target.lower() in d["name"].lower() or str(d["id"]) == target or d["ip"] == target]


async def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "discover":
        devices = await discover_and_save()
        print(f"Saved {len(devices)} device(s) to midea_devices.json:")
        for d in devices:
            print(f"  {d['ip']:15} {d.get('room') or d['name']}")
        return

    devices = load_devices()

    if cmd == "list":
        results = await asyncio.gather(*[state(d) for d in devices], return_exceptions=True)
        for d, r in zip(devices, results):
            if isinstance(r, Exception):
                print(f"{d['name']}: ERROR {r}")
            else:
                print(json.dumps(r, ensure_ascii=False))

    elif cmd in ("on", "off"):
        target = sys.argv[2] if len(sys.argv) > 2 else None
        on = cmd == "on"
        for d in _match(devices, target):
            name = await set_power(d, on)
            print(f"{cmd} -> {name}")

    else:
        print("usage: midea_cli.py list | on [name/id/ip] | off [name/id/ip]")


if __name__ == "__main__":
    asyncio.run(main())
