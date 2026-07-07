"""Shared state store.

One store reachable from home and remote; the Pi is never the system of record.
State is three line-based text files — list.txt, recipes.txt, memory.txt — so it
stays human-inspectable and trivially syncable through Google Drive.

`Store` is the interface. `LocalFileStore` (a directory) backs tests and the Mac
harness. `DriveStore` (see drive.py) is the production binding.
"""

from abc import ABC, abstractmethod

LIST = "list"
RECIPES = "recipes"
MEMORY = "memory"

DOCS = (LIST, RECIPES, MEMORY)


class Store(ABC):
    """Reads/writes the raw text of a named document."""

    @abstractmethod
    def read(self, doc: str) -> str: ...

    @abstractmethod
    def write(self, doc: str, content: str) -> None: ...

    # --- line helpers, shared by every backend ---

    def lines(self, doc: str) -> list[str]:
        return [ln.strip() for ln in self.read(doc).splitlines() if ln.strip()]

    def append_line(self, doc: str, line: str) -> None:
        existing = self.lines(doc)
        existing.append(line.strip())
        self.write(doc, "\n".join(existing) + "\n")

    def set_lines(self, doc: str, lines: list[str]) -> None:
        cleaned = [ln.strip() for ln in lines if ln.strip()]
        self.write(doc, ("\n".join(cleaned) + "\n") if cleaned else "")


class LocalFileStore(Store):
    """Plain files in a directory. Used by tests and the Mac harness."""

    def __init__(self, directory):
        from pathlib import Path

        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, doc: str):
        if doc not in DOCS:
            raise ValueError(f"unknown doc: {doc}")
        return self.dir / f"{doc}.txt"

    def read(self, doc: str) -> str:
        p = self._path(doc)
        return p.read_text() if p.exists() else ""

    def write(self, doc: str, content: str) -> None:
        self._path(doc).write_text(content)


def get_store(config) -> Store:
    """Build the configured store. Defaults to a local store under pi/.homie-state."""
    backend = getattr(config, "store_backend", "local")
    if backend == "drive":
        from homie.store.drive import DriveStore

        return DriveStore.from_env()
    from homie.config import PI_DIR

    return LocalFileStore(PI_DIR / ".homie-state")
