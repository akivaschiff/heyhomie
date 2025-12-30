# Homie

Voice-controlled home assistant for Raspberry Pi. Wake word: **"Yo Home"** (customizable)

## Architecture

```
┌─────────────────────────────────────┐
│           Raspberry Pi 4            │
│                                     │
│  - Porcupine (local wake word)      │
│  - OpenAI Whisper API (cloud STT)   │
│  - Claude API (cloud LLM)           │
│  - OpenAI TTS API (cloud)           │
│  - USB conference mic/speaker       │
└─────────────────────────────────────┘
```

## Features

- Wake word detection (Porcupine)
- Streaming responses with sentence-by-sentence TTS
- Parallel TTS pipeline (audio generates while Claude streams)
- Pleasant chime feedback sounds
- 60-second conversation context

## Cost Estimate

~$0.02 per interaction (STT + Claude + TTS). Heavy use < $1/day.

## Setup

### 1. System Dependencies

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv espeak alsa-utils mpg123
```

### 2. Python Setup

```bash
cd pi
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env  # Add your API keys
```

### 3. Picovoice Wake Word

1. Create account at https://console.picovoice.ai/
2. Get your Access Key
3. Train your wake word (e.g., "Yo Home") for Raspberry Pi
4. Download the `.ppn` file to `pi/` directory

### 4. API Keys

Get keys from:
- **Picovoice**: https://console.picovoice.ai/
- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/

### 5. Run

```bash
python main.py
```

## Configuration

All constants are at the top of `main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SILENCE_THRESHOLD` | 500 | Amplitude below this = silence |
| `SILENCE_DURATION` | 1.5s | Silence before stopping recording |
| `MIN_RECORDING_DURATION` | 3.0s | Minimum recording time |
| `CHIME_VOLUME` | 0.2 | Chime loudness (0.0-1.0) |
| `CONTEXT_TIMEOUT` | 60s | Conversation context timeout |
| `CLAUDE_MODEL` | haiku-4.5 | Claude model to use |
| `TTS_VOICE` | nova | OpenAI TTS voice |

## Run on Boot (Optional)

Create `/etc/systemd/system/homie.service`:

```ini
[Unit]
Description=Homie Voice Assistant
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/homie/pi
EnvironmentFile=/home/pi/homie/pi/.env
ExecStart=/home/pi/homie/pi/venv/bin/python main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable homie
sudo systemctl start homie
```

## MCP Architecture

### MCPs

| MCP | Covers |
|-----|--------|
| `home` | lights, AC, locks, security, routines |
| `media` | music, podcasts, TV |
| `productivity` | tasks, events, notes (multi-backend: calendar, Monday, etc.) |
| `comms` | messages, contacts |
| `shopping` | pantry, inventory, lists |

### Core Homie (not MCP)

| Feature | Covers |
|---------|--------|
| `timers` | short-term reminders, in-memory, Homie speaks when done |

Timers are kept in core because they need to fire and trigger Homie's speaker directly. MCPs are request/response and can't push to Homie when a timer completes.

### Reminder Routing

| Phrase | Destination |
|--------|-------------|
| "remind me" / "remind me in X" | Core timers → Homie speaks |
| "remind me on WhatsApp/Telegram" | `comms` MCP |
| "create calendar event" | `productivity` MCP |
| "add task" / "add to Monday" | `productivity` MCP |

### Implementation Status

- [ ] `home` MCP
- [ ] `media` MCP
- [ ] `productivity` MCP
- [ ] `comms` MCP
- [ ] `shopping` MCP
- [ ] Core timers module
