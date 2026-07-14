# CONNECTORS — how to observe Homie working end-to-end

The evaluation loop. A change is done when it's been watched working through these, not when it was reasoned to work.

## 1. Text harness (primary observation point)

The whole assistant is drivable from typed input, no audio in the stack. This is both the Mac test channel and the portability guarantee.

```bash
cd pi
.venv/bin/python app.py --channel text          # interactive REPL
echo "we finished the garbage bags" | .venv/bin/python app.py --channel text --once
```

Everything device-bound resolves to the Mac or a printed no-op. If a flow can't be exercised this way, voice has leaked into the core.

## 2. Shared store (system of record)

State lives in Google Drive text files (reachable from home and remote; the Pi is never the system of record):

- `list.txt`    — supermarket list
- `recipes.txt` — curated recipe links (`name | url` per line)
- `memory.txt`  — ambient facts

Inspect what the assistant actually wrote:

```bash
cd pi
.venv/bin/python -m homie.store dump list      # print current list from the live store
.venv/bin/python -m homie.store dump memory
```

Tests run against `LocalFileStore` (a temp dir) so they need no network. Production uses `DriveStore` when `HOMIE_STORE=drive` and the service account can read the files. The 3 Drive files must be shared with `homie-calendar@yohome-482813.iam.gserviceaccount.com` (Editor).

## 3. Deterministic unit tests (no network)

Every tool exercised against a `LocalFileStore` + `FakeChannel` + `FakeScheduler` (manual clock). Asserts observable state and surface calls. Fast, hermetic.

```bash
cd pi && .venv/bin/python -m pytest tests/test_tools.py -q
```

## 4. Live-Claude eval (the real loop)

Runs the real brain (real Anthropic calls, real tools) over every flow in spec §6 against a `FakeChannel`, asserting the observable outcome (item on list, recipe in context, reminder scheduled on the kitchen speaker, fact recalled cross-channel). This is "watch it work through the real loop."

```bash
cd pi && .venv/bin/python -m pytest tests/test_flows_live.py -q   # needs ANTHROPIC_API_KEY
```

The `FakeChannel` records every `say` / `show` / `send` / `announce`, so an assertion can prove the list rendered on a screen, the recipe page was shown, the kitchen speaker fired, etc.

## 5. Channels (executors over the core)

| Channel  | Run                                              | Needs                          |
|----------|--------------------------------------------------|--------------------------------|
| text     | `app.py --channel text`                          | ANTHROPIC                      |
| voice    | `app.py --channel voice`                         | ANTHROPIC + DEEPGRAM + Porcupine (wake word) |
| voice/PTT| `app.py --channel voice --ptt`                   | ANTHROPIC + DEEPGRAM (no Porcupine) |
| telegram | `app.py --channel telegram`                      | ANTHROPIC + TELEGRAM_BOT_TOKEN |

**Voice on the Mac (push-to-talk).** The wake word is a Pi-compiled Porcupine model,
so on the Mac use `--ptt`: press Enter, speak, pause — the same Deepgram STT → brain
→ Deepgram TTS pipeline, no wake word. Screen renders open in the browser; the
speaker is Deepgram TTS via `afplay`. Needs mic permission for the terminal.

```bash
cd pi && .venv/bin/python -m homie.app --channel voice --ptt
```

All three feed the same brain + tools + store. Voice (Deepgram) and Telegram are thin; the core does not know which one is driving it.

## Smart home (lights / blinds / ACs)

The `home_status` / `home_control` tools call the smarthome Flask server
(`smarthome/server.py`, port 8787 — contract in `smarthome/API_CONTRACT.md`), never
the vendor CLIs (the server owns the single allowed Higoal connection).

**The server runs on the Pi** (systemd unit `smarthome`, enabled on boot).
Dashboard / kiosk URL for the tablet: **http://raspberrypi.local:8787**

```bash
curl -s http://raspberrypi.local:8787/api/midea | python3 -m json.tool   # live AC state
ssh akiva@raspberrypi journalctl -u smarthome -f                     # server logs
```

`HOMIE_SMARTHOME_URL` points homie at the server: `http://localhost:8787` on the
Pi, `http://raspberrypi.local:8787` from the Mac. Unit tests fake the client
(`tests/test_smart_home.py`); the live check is a status question + a reversible
light toggle through the real brain.

## Scheduled home actions (cron on the Pi)

`schedule_set/list/cancel` manage a tagged block in the Pi user's crontab; each
entry is a plain `curl` to the smarthome server, so it fires with homie down.

```bash
ssh akiva@raspberrypi crontab -l                              # see the block
ssh akiva@raspberrypi 'journalctl -u cron --since "1h ago"'   # did it fire?
```

## Shufersal cart (the supermarket projection)

The Pi list is the source of truth; every `list_add` also projects the item into the
real Shufersal online cart (`services/shufersal.py`). Resolution is history-first
(a term you've ordered before → that exact SKU), catalog-search second. The chosen
product is appended to the list line (`bananas — אשכול בננה`) so a wrong pick is
visible on the kiosk.

Auth is a real credential login (`HOMIE_SHUFERSAL_USER`/`HOMIE_SHUFERSAL_PASSWORD`
in `pi/.env`): FORM post to `j_spring_security_check`, then `GET
/cart/load?restoreCart=true` to bind the session to the account cart — that's the
same call the site's own JS fires on every page boot, so connector sessions and
browser sessions converge on one cart. No creds → the seam is `None` and adds
still hit the list only (portability preserved). Captured cookie jars are obsolete
(they only ever reach an anonymous cart).

Writes are verified visible in the user's real browser cart (2026-07-14). Gotcha
when checking: an already-open browser session keeps its own session cart (hard
reload keeps the JSESSIONID and doesn't re-restore) — verify with a fresh login
(incognito / app relaunch), or `.scratch/shufersal/browser_probe.js` which does a
scripted fresh-context login and prints the cart.

Session expiry is the trap on the long-running Pi service: past the server's idle
timeout the JSESSIONID dies and requests ride the remember-me cookie — adds still
return 200 but fork a "pre-identification" cart that hijacks the account cart
(newest-modified wins on restore) and triggers a merge popup in the user's
browser. `add_to_cart` therefore refuses to write unless
`/authentication/get-status-includes-otp` returns `true`, and `ShufersalCart`
only reports ok after reading the bound cart back and finding the item (re-login
+ one retry in between).

```bash
# watch the projection working, against the live cart (creds come from pi/.env):
cd pi && venv/bin/python - <<'PY'
from dotenv import load_dotenv; load_dotenv(".env")
from homie.services.shufersal import ShufersalCart, get_cart
cart = ShufersalCart()
print(cart.resolve("חלב"))          # history vs search + the SKU it picked
print(cart.add_item("תפוחים", 1))   # resolve + add one package
print(get_cart(cart.session))        # read the live cart back (HTML-parsed)
PY
```

Endpoints (reverse-engineered, full notes in `.scratch/shufersal/API_FINDINGS.md`):
`GET /search/results?q=<he>:relevance`, `POST /cart/add` (needs `CSRFToken` header,
qty is absolute not additive), `GET /cart/load` (HTML only), `GET /my-account/orders`.

## Pi debugging

```bash
ssh akiva@raspberrypi journalctl -u homie -f    # live: wake, heard, tools, ⚠️ turn errors
```

Every turn (including errors) is traced to Langfuse from the Pi as well —
https://cloud.langfuse.com, one `turn` trace per exchange.

Traces are queryable directly (keys in `pi/.env`), no UI needed:

```bash
cd pi && export $(grep -E '^LANGFUSE' .env | xargs)
curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  "https://cloud.langfuse.com/api/public/traces?limit=30"          # recent turns (input/output per trace)
curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  "https://cloud.langfuse.com/api/public/observations?traceId=<id>" # generations + tool spans inside a turn
```

Note the trace detail shows `turn` twice (trace header + root span) — v3 SDK
mirrors the root observation to the trace; it's one execution, not a duplicate.

## Keys

`pi/.env` (copy from `.env.example`): `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`, `TAVILY_API_KEY` (recipe search), `TELEGRAM_BOT_TOKEN`, `PORCUPINE_ACCESS_KEY`, `GOOGLE_SERVICE_ACCOUNT_PATH`, and the three `*_FILE_ID` for the Drive store.
