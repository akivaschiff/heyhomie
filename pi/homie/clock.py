"""Time + scheduling abstractions.

Injected everywhere a timer or clock time matters so tests stay deterministic:
the real implementation uses wall-clock + threading.Timer; the fake advances
manually and fires callbacks synchronously.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Callable

import pytz


class Clock:
    """Wall-clock time."""

    def now(self) -> datetime:
        return datetime.now().astimezone()

    def monotonic(self) -> float:
        return time.monotonic()


class Scheduler:
    """Schedules a callback to fire after N seconds. Real implementation."""

    def __init__(self):
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def schedule(self, seconds: float, callback: Callable[[], None]) -> str:
        with self._lock:
            self._counter += 1
            job_id = f"job-{self._counter}"

        def fire():
            with self._lock:
                self._timers.pop(job_id, None)
            callback()

        t = threading.Timer(max(0.0, seconds), fire)
        t.daemon = True
        with self._lock:
            self._timers[job_id] = t
        t.start()
        return job_id

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            t = self._timers.pop(job_id, None)
        if t is not None:
            t.cancel()
            return True
        return False

    def cancel_all(self):
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for t in timers:
            t.cancel()


class FakeClock(Clock):
    """Manually-advanced clock. Wall time tracks the monotonic offset."""

    def __init__(self, start: datetime):
        if start.tzinfo is None:
            start = pytz.UTC.localize(start)
        self._base_wall = start
        self._mono = 0.0

    def now(self) -> datetime:
        return self._base_wall + timedelta(seconds=self._mono)

    def monotonic(self) -> float:
        return self._mono

    def _set_mono(self, value: float):
        self._mono = value


class FakeScheduler(Scheduler):
    """Records scheduled jobs and fires them on manual advance. No threads."""

    def __init__(self, clock: FakeClock):
        self._clock = clock
        self._jobs: dict[str, tuple[float, Callable[[], None]]] = {}
        self._counter = 0

    def schedule(self, seconds: float, callback: Callable[[], None]) -> str:
        self._counter += 1
        job_id = f"fake-{self._counter}"
        self._jobs[job_id] = (self._clock.monotonic() + max(0.0, seconds), callback)
        return job_id

    def cancel(self, job_id: str) -> bool:
        return self._jobs.pop(job_id, None) is not None

    def cancel_all(self):
        self._jobs.clear()

    def advance(self, seconds: float):
        """Advance the fake clock, firing any due jobs in chronological order."""
        target = self._clock.monotonic() + seconds
        while True:
            due = sorted(
                (fire_at, jid)
                for jid, (fire_at, _) in self._jobs.items()
                if fire_at <= target
            )
            if not due:
                break
            fire_at, job_id = due[0]
            _, callback = self._jobs.pop(job_id)
            self._clock._set_mono(fire_at)
            callback()
        self._clock._set_mono(target)
