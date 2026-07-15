import os
import sys
import json
import time

from electrasmart.client import send_otp_request, get_otp_token, get_devices, AC

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(HERE, ".env")
IMEI_TMP = os.path.join(HERE, ".electra_imei")

MODES = ["STBY", "COOL", "FAN", "DRY", "HEAT", "AUTO"]
FAN_SPEEDS = ["AUTO", "LOW", "MED", "HIGH"]
TEMP_MIN, TEMP_MAX = 16, 30


def _load_env():
    with open(ENV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k] = v


def _set_env(key, value):
    lines, found = [], False
    with open(ENV) as f:
        for line in f:
            if line.startswith(f"{key}="):
                lines.append(f"{key}={value}\n")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}\n")
    with open(ENV, "w") as f:
        f.writelines(lines)


def local_phone():
    p = os.environ["ELECTRA_PHONE"]
    return "0" + p[3:] if p.startswith("972") else p


def creds():
    return os.environ["ELECTRA_IMEI"], os.environ["ELECTRA_TOKEN"]


def devices():
    imei, token = creds()
    return get_devices(imei, token)


def read_state(d):
    imei, token = creds()
    ac = AC(imei, token, d["id"])
    ac.renew_sid()
    ac.update_status()
    s = ac.status
    return {
        "id": d["id"],
        "name": d["name"],
        "kind": d.get("manufactor"),
        "on": s.is_on,
        "mode": s.ac_mode,
        "fan": s.fan_speed,
        "target": s.spt,
        "current": s.current_temp,
        "modes": MODES,
        "fans": FAN_SPEEDS,
        "min": TEMP_MIN,
        "max": TEMP_MAX,
    }


def set_state(ac_id, power=None, mode=None, temp=None, fan=None, verify=True):
    imei, token = creds()
    ac = AC(imei, token, ac_id)
    ac.renew_sid()

    turning_off = power is False or mode == "STBY"
    if turning_off:
        want_on = False
    elif power is True or (mode is not None and mode != "STBY"):
        want_on = True
    else:
        want_on = None

    def apply():
        if turning_off:
            ac.turn_off()
            return
        kwargs = {}
        if mode is not None:
            kwargs["ac_mode"] = mode
        elif power is True:
            kwargs["ac_mode"] = "COOL"
        if fan is not None:
            kwargs["fan_speed"] = fan
        if temp is not None:
            kwargs["temperature"] = int(temp)
        if kwargs:
            ac.modify_oper(**kwargs)

    apply()

    # Electra's cloud silently drops commands: it returns 200 while the unit never
    # actuates. Confirm the unit reached the requested power state from telemetry,
    # and re-send once if it didn't, so callers get the truth instead of a blind ok.
    if not verify or want_on is None:
        return {"ok": True, "verified": False, "on": want_on}
    for attempt in range(2):
        for _ in range(3):
            time.sleep(1.2)
            ac.update_status()
            if ac.status.is_on == want_on:
                return {"ok": True, "verified": True, "on": want_on}
        if attempt == 0:
            apply()
    return {"ok": False, "verified": True, "on": ac.status.is_on}


def main():
    _load_env()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "request":
        imei = send_otp_request(local_phone())
        with open(IMEI_TMP, "w") as f:
            f.write(imei)
        print(f"OTP SMS requested for {local_phone()}. imei={imei}")

    elif cmd == "token":
        otp = sys.argv[2]
        with open(IMEI_TMP) as f:
            imei = f.read().strip()
        imei, token = get_otp_token(imei, local_phone(), otp)
        _set_env("ELECTRA_IMEI", imei)
        _set_env("ELECTRA_TOKEN", token)
        print("Authenticated. imei+token saved to .env")

    elif cmd == "devices":
        print(json.dumps([{"id": d["id"], "name": d["name"]} for d in devices()], ensure_ascii=False, indent=2))

    elif cmd == "state":
        for d in devices():
            print(json.dumps(read_state(d), ensure_ascii=False))

    elif cmd in ("on", "off"):
        target = sys.argv[2] if len(sys.argv) > 2 else None
        for d in devices():
            if target and str(d["id"]) != str(target):
                continue
            set_state(d["id"], power=(cmd == "on"))
            print(f"{cmd} -> {d['name']} ({d['id']})")

    else:
        print("usage: electra_cli.py request | token <otp> | devices | state | on [id] | off [id]")


if __name__ == "__main__":
    main()
