"""volume_set against a fake system volume — steps, absolute levels, clamping."""

import json

import pytest

from homie.tools.volume import _set


class FakeVolume:
    def __init__(self, level=60):
        self.level = level

    def get(self):
        return self.level

    def set(self, percent):
        self.level = max(0, min(100, percent))
        return self.level


@pytest.fixture
def vol_ctx(ctx):
    ctx.volume = FakeVolume(60)
    return ctx


def test_quieter_steps_down(vol_ctx):
    out = json.loads(_set({"direction": "quieter"}, vol_ctx))
    assert out == {"volume": 45, "was": 60}


def test_louder_steps_up(vol_ctx):
    out = json.loads(_set({"direction": "louder"}, vol_ctx))
    assert out == {"volume": 75, "was": 60}


def test_absolute_level(vol_ctx):
    out = json.loads(_set({"level": 40}, vol_ctx))
    assert out["volume"] == 40


def test_clamped_at_bounds(vol_ctx):
    vol_ctx.volume.level = 95
    assert json.loads(_set({"direction": "louder"}, vol_ctx))["volume"] == 100
    assert json.loads(_set({"level": 250}, vol_ctx))["volume"] == 100


def test_no_args_reports_current(vol_ctx):
    assert json.loads(_set({}, vol_ctx)) == {"volume": 60}


def test_unavailable(ctx):
    assert "error" in json.loads(_set({"direction": "quieter"}, ctx))
