import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from homie.channels.base import Surface  # noqa: E402
from homie.channels.fake import FakeChannel  # noqa: E402
from homie.clock import FakeClock, FakeScheduler  # noqa: E402
from homie.config import Config  # noqa: E402
from homie.store import LocalFileStore  # noqa: E402
from homie.tools.base import ToolContext  # noqa: E402


@pytest.fixture
def clock():
    return FakeClock(datetime(2026, 6, 24, 16, 0, 0))


@pytest.fixture
def scheduler(clock):
    return FakeScheduler(clock)


@pytest.fixture
def store(tmp_path):
    return LocalFileStore(tmp_path)


@pytest.fixture
def channel():
    return FakeChannel(Surface(has_screen=True, has_speaker=True))


@pytest.fixture
def ctx(store, channel, scheduler, clock):
    return ToolContext(
        store=store,
        channel=channel,
        scheduler=scheduler,
        clock=clock,
        config=Config(),
        session={},
    )
