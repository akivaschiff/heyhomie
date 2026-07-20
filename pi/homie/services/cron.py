"""Managed crontab block for scheduled home actions.

Entries live between marker lines in the user's crontab, each as a metadata
comment plus the cron line itself:

    # --- homie schedules start ---
    # homie-schedule:<id>:<recur>:<date-or-empty>:<description>
    0 19 * * * curl -s -m 20 -X POST http://host:8787/api/midea/set ... >/dev/null 2>&1
    # --- homie schedules end ---

Firing is pure cron + curl — homie doesn't need to be running. One-time entries
are date-guarded and pruned automatically whenever the block is touched.
Weekdays/weekends follow the Israeli week: weekdays Sun-Thu (0-4), weekend Fri-Sat (5,6).
sun_fri is every day except Shabbat: Sun-Fri (0-5).
"""

import subprocess
from dataclasses import dataclass

BLOCK_START = "# --- homie schedules start ---"
BLOCK_END = "# --- homie schedules end ---"
META_PREFIX = "# homie-schedule:"

DOW = {"once": "*", "daily": "*", "weekdays": "0-4", "weekends": "5,6", "sun_fri": "0-5"}


@dataclass
class Entry:
    id: str
    recur: str
    date: str  # YYYY-MM-DD for once, "" otherwise
    description: str
    cron_line: str

    def meta_line(self) -> str:
        return f"{META_PREFIX}{self.id}:{self.recur}:{self.date}:{self.description}"


def build_cron_line(time_hhmm: str, recur: str, date: str, commands: list) -> str:
    hour, minute = time_hhmm.split(":")
    joined = "; ".join(commands)
    if recur == "once":
        # %% is special in crontab; \% passes a literal % to the shell's date call
        joined = f'[ "$(date +\\%F)" = "{date}" ] && {{ {joined}; }}'
    return f"{int(minute)} {int(hour)} * * {DOW[recur]} {joined}"


def build_curl(base_url: str, system: str, payload_json: str) -> str:
    return (
        f"curl -s -m 20 -X POST {base_url}/api/{system}/set "
        f"-H 'Content-Type: application/json' -d '{payload_json}' >/dev/null 2>&1"
    )


class CronStore:
    """Reads/writes the managed block in the real crontab."""

    def read_crontab(self) -> str:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else ""

    def write_crontab(self, content: str) -> None:
        subprocess.run(["crontab", "-"], input=content, text=True, check=True)

    # --- block manipulation (pure string work, shared with FakeCronStore) ---

    def entries(self) -> list:
        lines = self.read_crontab().splitlines()
        entries, meta = [], None
        inside = False
        for line in lines:
            if line.strip() == BLOCK_START:
                inside = True
            elif line.strip() == BLOCK_END:
                inside = False
            elif inside and line.startswith(META_PREFIX):
                meta = line[len(META_PREFIX):].split(":", 3)
            elif inside and meta is not None:
                entries.append(Entry(meta[0], meta[1], meta[2], meta[3], line))
                meta = None
        return entries

    def save_entries(self, entries: list) -> None:
        lines = self.read_crontab().splitlines()
        kept, inside = [], False
        for line in lines:
            if line.strip() == BLOCK_START:
                inside = True
            elif line.strip() == BLOCK_END:
                inside = False
            elif not inside:
                kept.append(line)
        while kept and not kept[-1].strip():
            kept.pop()
        if entries:
            kept.append("")
            kept.append(BLOCK_START)
            for e in entries:
                kept.append(e.meta_line())
                kept.append(e.cron_line)
            kept.append(BLOCK_END)
        self.write_crontab("\n".join(kept) + "\n")

    def prune_stale(self, today: str) -> list:
        """Drop one-time entries whose date has passed. Returns what remains."""
        entries = [e for e in self.entries() if not (e.recur == "once" and e.date < today)]
        self.save_entries(entries)
        return entries

    def next_id(self) -> str:
        existing = {e.id for e in self.entries()}
        n = 1
        while f"s{n}" in existing:
            n += 1
        return f"s{n}"


class FakeCronStore(CronStore):
    """In-memory crontab for tests."""

    def __init__(self, initial: str = ""):
        self._content = initial

    def read_crontab(self) -> str:
        return self._content

    def write_crontab(self, content: str) -> None:
        self._content = content
