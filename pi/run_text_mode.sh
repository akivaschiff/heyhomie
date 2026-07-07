#!/bin/bash
# Run Homie's text/Mac harness — the whole assistant from typed input, no audio.
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating venv…"
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
fi

exec .venv/bin/python -m homie.app --channel text "$@"
