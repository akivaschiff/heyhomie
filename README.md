# Homie

A voice-first kitchen assistant with a remote text channel and a local test harness.
One shared core; channels are thin executors. See `CLAUDE.md` for the full spec and
`CONNECTORS.md` for how to observe it working end-to-end.

## Architecture

```
                       ┌──────────────────────────────┐
   voice (Pi) ───┐     │  Brain  (Claude tool-loop)   │     ┌─ Drive store
  telegram   ────┼────▶│  no audio, no STT/TTS        │────▶├─  list.txt
   text (Mac) ───┘     │  drivable from typed input   │     ├─  recipes.txt
                       └──────────────────────────────┘     └─  memory.txt
        executors              durable tool layer              shared state
```

- **Core** (`homie/brain.py`, `homie/tools/`) is channel-agnostic. The model
  orchestrates; there is no dialog state machine. Tools operate on a shared `Store`
  and the active channel's render surface.
- **Channels** (`homie/channels/`) are executors over the same core:
  - `text` — Mac harness, the hard portability surface (typed input, no audio).
  - `voice` — Pi kitchen; **Deepgram** owns STT/TTS, Porcupine wake word.
  - `telegram` — remote chat.
- **Store** is the system of record, reachable from home and remote — the Pi never
  owns state. `DriveStore` (Google Drive text files) in production, `LocalFileStore`
  for tests and the Mac harness.

## Tools

`list_add`, `list_remove`, `list_show`, `timer_set`, `reminder_set`, `recipe_load`
(curated links + web search), `memory_save`, `memory_query`, `shabbat_mode`. Recipe
follow-ups (next step, scaling, units, substitutions) are handled by the model over
the recipe retained in context.

## Run

```bash
cd pi
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in keys

.venv/bin/python -m homie.app --channel text       # Mac harness (REPL)
.venv/bin/python -m homie.app --channel voice      # Pi
.venv/bin/python -m homie.app --channel telegram   # remote
```

## Test

```bash
cd pi
.venv/bin/python -m pytest tests/test_tools.py -q        # deterministic, no network
.venv/bin/python -m pytest tests/test_flows_live.py -q   # real brain over every §6 flow
```

## Production store setup (Google Drive)

1. Enable the Drive API on the GCP project that owns the service account.
2. Share `list.txt`, `recipes.txt`, `memory.txt` (Editor) with the service-account email.
3. Set `HOMIE_STORE=drive` and the three `*_FILE_ID` vars in `.env`.

`recipes.txt` is the curated source: one `name | url` per line.
