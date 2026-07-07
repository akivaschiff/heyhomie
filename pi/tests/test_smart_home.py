"""smart_home tool against a fake server client — resolution, fan-out, and the
blind open/close idx contract, with no hardware touched."""

import json

import pytest

from homie.tools.smart_home import _control, _status


class FakeSmartHome:
    base_url = "http://fakehome:8787"

    def __init__(self):
        self.calls = []
        self.data = {
            "higoal": [
                {"device": "PANEL1", "id": "PANEL1", "entities": [
                    {"idx": 2, "name": "תריס מטבח", "type": "shutter", "on": False},
                    {"idx": 3, "name": "פאנל מטבח channel 4", "type": "shutter", "on": None},
                    {"idx": 5, "name": "מאור מטבח", "type": "switch", "on": True},
                    {"idx": 3, "name": "מאור מטבח קטן", "type": "switch", "on": False},
                ]},
                {"device": "PANEL2", "id": "PANEL2", "entities": [
                    {"idx": 0, "name": "תריס חדר משחקים", "type": "shutter", "on": False},
                    {"idx": 2, "name": "top light", "type": "switch", "on": False},
                ]},
            ],
            "midea": [
                {"name": "Girls' Room", "id": "net_ac_BE06", "power": False, "target": 23.0, "indoor": 27.0},
                {"name": "Boys' Room", "id": "net_ac_4F0A", "power": False, "target": 24.0, "indoor": 26.0},
                {"name": "Parent Bedroom", "id": "net_ac_0654", "power": True, "target": 22.0, "indoor": 25.0},
                {"name": "The Office", "id": "net_ac_4DE0", "power": False, "target": 23.0, "indoor": 26.5},
            ],
            "electra": [
                {"name": "קומת קרקע", "id": 236084, "on": True, "target": 21},
            ],
        }

    def get(self, system):
        return self.data[system]

    def set(self, system, payload):
        self.calls.append((system, payload))
        return {"ok": True}


@pytest.fixture
def home(ctx):
    fake = FakeSmartHome()
    ctx.smarthome = fake
    return fake


def test_status_filters_channel_noise_and_kinds(ctx, home):
    out = json.loads(_status({"kind": "blind"}, ctx))
    names = [d["name"] for d in out]
    assert "תריס מטבח" in names and "תריס חדר משחקים" in names
    assert all("channel" not in n for n in names)


def test_close_all_blinds_uses_idx_plus_one(ctx, home):
    out = json.loads(_control({"kind": "blind", "target": "all", "action": "close"}, ctx))
    assert len(out["done"]) == 2
    assert ("higoal", {"device": "PANEL1", "idx": 3, "on": True}) in home.calls
    assert ("higoal", {"device": "PANEL2", "idx": 1, "on": True}) in home.calls


def test_open_blind_uses_idx(ctx, home):
    _control({"kind": "blind", "target": "תריס מטבח", "action": "open"}, ctx)
    assert home.calls == [("higoal", {"device": "PANEL1", "idx": 2, "on": True})]


def test_exact_name_beats_substring(ctx, home):
    out = json.loads(_control({"kind": "light", "target": "מאור מטבח", "action": "off"}, ctx))
    assert [d["name"] for d in out["done"]] == ["מאור מטבח"]
    assert home.calls == [("higoal", {"device": "PANEL1", "idx": 5, "on": False})]


def test_upstairs_acs_fan_out_to_all_midea(ctx, home):
    out = json.loads(_control({"kind": "ac", "target": "upstairs", "action": "off"}, ctx))
    assert len(out["done"]) == 4
    assert all(sys == "midea" and p["power"] is False for sys, p in home.calls)


def test_main_ac_is_electra_with_temp(ctx, home):
    _control({"kind": "ac", "target": "main", "action": "on", "temp": 22}, ctx)
    assert home.calls == [("electra", {"id": 236084, "power": True, "temp": 22})]


def test_girls_ac_by_name(ctx, home):
    _control({"kind": "ac", "target": "girls", "action": "on"}, ctx)
    assert home.calls == [("midea", {"id": "net_ac_BE06", "power": True})]


def test_unknown_target_returns_available_names(ctx, home):
    out = json.loads(_control({"kind": "ac", "target": "garage", "action": "on"}, ctx))
    assert "error" in out and "Girls' Room" in out["available"]
    assert home.calls == []


def test_invalid_action_for_kind(ctx, home):
    out = json.loads(_control({"kind": "blind", "target": "all", "action": "on"}, ctx))
    assert "error" in out
    assert home.calls == []
