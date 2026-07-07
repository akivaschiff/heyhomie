"""Inspect the live store: `python -m homie.store dump <list|recipes|memory>`."""

import sys

from homie.config import Config
from homie.store import DOCS, get_store


def main(argv):
    if len(argv) < 2 or argv[0] != "dump" or argv[1] not in DOCS:
        print("usage: python -m homie.store dump <list|recipes|memory>")
        return 1
    store = get_store(Config())
    content = store.read(argv[1])
    print(content if content.strip() else f"({argv[1]} is empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
