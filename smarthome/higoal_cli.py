import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from higoal_client.manager import Manager, EntityListener
from higoal_client.device import TYPE_SWITCH, TYPE_DIMMER, TYPE_SHUTTER

TYPE_NAME = {TYPE_SWITCH: "switch", TYPE_DIMMER: "dimmer/light", TYPE_SHUTTER: "shutter"}


def _load_env():
    env = os.path.join(os.path.dirname(__file__), ".env")
    with open(env) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


class Collector(EntityListener):
    def __init__(self):
        self.changed = 0

    def on_entity_changed(self, entity):
        self.changed += 1

    def on_device_added(self, device):
        pass

    def on_device_removed(self, device):
        pass


def connect(settle=4.0):
    _load_env()
    listener = Collector()
    m = Manager(
        username=os.environ["HIGOAL_USERNAME"],
        password=os.environ["HIGOAL_PASSWORD"],
        entity_listener=listener,
    )
    m.get_devices()
    m.refresh()
    time.sleep(settle)
    return m


def snapshot(m):
    out = []
    for dev in m.device_map.values():
        if not hasattr(dev, "entities"):
            continue
        d = {"device": dev.name, "id": dev.id, "model": dev.model_name, "entities": []}
        for e in dev.entities:
            resp = e.response
            entity = {
                "idx": e.id,
                "name": e.display_name,
                "type": TYPE_NAME.get(e.type, e.type),
                "online": e.is_online() if resp else None,
                "on": e.is_turned_on() if resp else None,
            }
            if e.type == TYPE_SHUTTER:
                entity["percentage"] = e.percentage() if resp else None
            d["entities"].append(entity)
        out.append(d)
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    m = connect()
    snap = snapshot(m)

    if cmd == "list":
        print(json.dumps(snap, ensure_ascii=False, indent=2))

    elif cmd == "off-lights":
        keywords = ("light", "מאור", "אור")
        n = 0
        for dev in m.device_map.values():
            if not hasattr(dev, "entities"):
                continue
            for e in dev.entities:
                if e.type not in (TYPE_SWITCH, TYPE_DIMMER):
                    continue
                if any(k in (e.name or "").lower() for k in keywords):
                    print(f"turn_off -> {e.name} (was on={e.is_turned_on() if e.response else '?'})")
                    e.turn_off()
                    n += 1
                    time.sleep(0.3)
        print(f"sent turn_off to {n} light(s)")
        time.sleep(3)

    m.mq.stop()


if __name__ == "__main__":
    main()
