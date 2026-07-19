"""smart_home tool against a fake server client — resolution, fan-out, and the
blind open/close idx contract, with no hardware touched."""

import json

import pytest

from homie.tools.smart_home import _control, _status


def _drive(ctx, args, seconds=40):
    """home_control is optimistic — actuation runs on the scheduler. Advance the
    fake clock so the background job (and any retries) fire, then return the result."""
    out = json.loads(_control(args, ctx))
    ctx.scheduler.advance(seconds)
    return out


class FakeSmartHome:
    base_url = "http://fakehome:8787"

    def __init__(self, fail=None, heal_after=0):
        self.fail = fail or set()
        self.heal_after = heal_after
        self.attempts = 0
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
        self.attempts += 1
        # fail = systems that error; heal_after>0 means they recover after N attempts
        # (a transient blip), heal_after==0 means they stay broken (permanent failure).
        failing = system in self.fail and (self.heal_after == 0 or self.attempts <= self.heal_after)
        if failing:
            raise ConnectionError(f"{system} unreachable")
        self.calls.append((system, payload))
        return {"ok": True}


@pytest.fixture
def home(ctx):
    fake = FakeSmartHome()
    ctx.smarthome = fake
    return fake


def test_status_filters_channel_noise_and_kinds(ctx, home):
    out = json.loads(_status({"kind": "blind"}, ctx))["devices"]
    names = [d["name"] for d in out]
    assert "תריס מטבח" in names and "תריס חדר משחקים" in names
    assert all("channel" not in n for n in names)


def test_close_all_blinds_uses_idx_plus_one(ctx, home):
    _drive(ctx, {"kind": "blind", "target": "all", "action": "close"})
    assert ("higoal", {"device": "PANEL1", "idx": 3, "on": True}) in home.calls
    assert ("higoal", {"device": "PANEL2", "idx": 1, "on": True}) in home.calls


def test_open_blind_uses_idx(ctx, home):
    _drive(ctx, {"kind": "blind", "target": "תריס מטבח", "action": "open"})
    assert home.calls == [("higoal", {"device": "PANEL1", "idx": 2, "on": True})]


def test_exact_name_beats_substring(ctx, home):
    _drive(ctx, {"kind": "light", "target": "מאור מטבח", "action": "off"})
    assert home.calls == [("higoal", {"device": "PANEL1", "idx": 5, "on": False})]


def test_upstairs_acs_fan_out_to_all_midea(ctx, home):
    _drive(ctx, {"kind": "ac", "target": "upstairs", "action": "off"})
    assert len(home.calls) == 4
    assert all(sys == "midea" and p["power"] is False for sys, p in home.calls)


def test_main_ac_is_electra_with_temp(ctx, home):
    _drive(ctx, {"kind": "ac", "target": "main", "action": "on", "temp": 22})
    assert home.calls == [("electra", {"id": 236084, "power": True, "temp": 22})]


def test_girls_ac_by_name(ctx, home):
    _drive(ctx, {"kind": "ac", "target": "girls", "action": "on"})
    assert home.calls == [("midea", {"id": "net_ac_BE06", "power": True})]


def test_unknown_target_returns_available_names(ctx, home):
    out = json.loads(_control({"kind": "ac", "target": "garage", "action": "on"}, ctx))
    assert "error" in out and "Girls' Room" in out["available"]
    assert home.calls == []


def test_invalid_action_for_kind(ctx, home):
    out = json.loads(_control({"kind": "blind", "target": "all", "action": "on"}, ctx))
    assert "error" in out
    assert home.calls == []


def test_control_acknowledges_before_actuating(ctx, home):
    out = json.loads(_control({"kind": "ac", "target": "main", "action": "on"}, ctx))
    assert out["started"] == "on" and out["devices"] == ["the main air conditioner"]
    assert home.calls == []  # nothing actuated on the request thread yet
    ctx.scheduler.advance(1)
    assert home.calls == [("electra", {"id": 236084, "power": True})]
    assert ctx.channel.announced == []  # success stays silent


def test_transient_failure_recovers_silently(ctx):
    ctx.smarthome = FakeSmartHome(fail={"electra"}, heal_after=1)
    _drive(ctx, {"kind": "ac", "target": "main", "action": "off"})
    assert ("electra", {"id": 236084, "power": False}) in ctx.smarthome.calls
    assert ctx.channel.announced == []  # recovered within the window, no complaint


def test_permanent_failure_announces_after_window(ctx):
    ctx.smarthome = FakeSmartHome(fail={"electra"})
    _drive(ctx, {"kind": "ac", "target": "main", "action": "off"})
    assert ctx.smarthome.calls == []  # never succeeded
    assert ctx.smarthome.attempts >= 2  # retried across the window
    assert len(ctx.channel.announced) == 1
    assert "couldn't turn off the main air conditioner" in ctx.channel.announced[0]


def test_offline_room_ac_failure_label_has_no_double_article(ctx):
    ctx.smarthome = FakeSmartHome(fail={"midea"})
    _drive(ctx, {"kind": "ac", "target": "The Office", "action": "on"})
    assert ctx.channel.announced == ["Sorry, I couldn't turn on The Office."]
