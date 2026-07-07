"""Schedule tools against FakeCronStore + FakeSmartHome — cron line correctness,
prune, cancel. Never touches the real crontab."""

import json

import pytest

from homie.services.cron import BLOCK_START, FakeCronStore, build_cron_line
from homie.tools.schedule import _cancel, _list, _set
from tests.test_smart_home import FakeSmartHome


@pytest.fixture
def home_ctx(ctx):
    ctx.smarthome = FakeSmartHome()
    ctx.cron = FakeCronStore("0 5 * * * /usr/bin/backup\n")
    return ctx


def test_daily_schedule_writes_cron_line(home_ctx):
    out = json.loads(_set({
        "time": "19:00", "recur": "daily",
        "kind": "ac", "target": "girls", "action": "on", "temp": 23,
    }, home_ctx))
    assert out["devices"] == ["Girls' Room"]
    content = home_ctx.cron.read_crontab()
    assert "0 5 * * * /usr/bin/backup" in content  # user's own lines untouched
    assert BLOCK_START in content
    assert "0 19 * * *" in content
    assert '"id": "net_ac_BE06"' in content and '"temp": 23' in content


def test_weekend_recurrence_uses_fri_sat(home_ctx):
    _set({"time": "09:30", "recur": "weekends",
          "kind": "blind", "target": "all", "action": "open"}, home_ctx)
    assert "30 9 * * 5,6" in home_ctx.cron.read_crontab()


def test_once_has_escaped_date_guard(home_ctx):
    _set({"time": "19:00", "recur": "once", "date": "2026-07-08",
          "kind": "ac", "target": "boys", "action": "on"}, home_ctx)
    content = home_ctx.cron.read_crontab()
    assert '[ "$(date +\\%F)" = "2026-07-08" ]' in content


def test_once_requires_date(home_ctx):
    out = json.loads(_set({"time": "19:00", "recur": "once",
                           "kind": "ac", "target": "boys", "action": "on"}, home_ctx))
    assert "error" in out


def test_stale_once_entries_pruned(home_ctx):
    # clock in fixtures is 2026-06-24; this entry is already in the past
    _set({"time": "10:00", "recur": "once", "date": "2026-06-20",
          "kind": "ac", "target": "boys", "action": "on"}, home_ctx)
    # writing it succeeds, but any subsequent touch prunes it
    out = json.loads(_list({}, home_ctx))
    assert out["schedules"] == []


def test_group_target_embeds_multiple_curls(home_ctx):
    _set({"time": "22:00", "recur": "daily",
          "kind": "ac", "target": "upstairs", "action": "off"}, home_ctx)
    line = [l for l in home_ctx.cron.read_crontab().splitlines() if "22" in l and "curl" in l][0]
    assert line.count("curl ") == 4


def test_list_and_cancel(home_ctx):
    _set({"time": "19:00", "recur": "daily",
          "kind": "ac", "target": "girls", "action": "on"}, home_ctx)
    _set({"time": "08:00", "recur": "daily",
          "kind": "ac", "target": "girls", "action": "off"}, home_ctx)
    listed = json.loads(_list({}, home_ctx))["schedules"]
    assert len(listed) == 2
    out = json.loads(_cancel({"id": listed[0]["id"]}, home_ctx))
    assert out["cancelled"] == listed[0]["id"]
    assert len(json.loads(_list({}, home_ctx))["schedules"]) == 1
    assert "0 5 * * * /usr/bin/backup" in home_ctx.cron.read_crontab()


def test_cron_line_minutes_hours_order():
    line = build_cron_line("07:05", "daily", "", ["echo hi"])
    assert line.startswith("5 7 * * *")
