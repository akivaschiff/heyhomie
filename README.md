# Homie

Voice-controlled home assistant. Wake word: **"Hey Homie"**

## Architecture (v1 - Pi Only)

```
┌─────────────────────────────────────┐
│           Raspberry Pi 4            │
│                                     │
│  - Porcupine (local wake word)      │
│  - OpenAI Whisper API (cloud STT)   │
│  - Claude API + MCP (cloud)         │
│  - OpenAI TTS API (cloud)           │
│  - Anker conference mic/speaker     │
└─────────────────────────────────────┘
```

## Cost Estimate

~$0.02 per interaction (STT + Claude + TTS). Heavy use < $1/day.

## Setup

### 1. Picovoice Wake Word

1. Create account at https://console.picovoice.ai/
2. Get your Access Key from the main dashboard
3. Go to **Porcupine → Train Wake Word**
4. Enter "Hey Homie", select **Raspberry Pi** (Linux / Cortex-A72)
5. Train and download the `.ppn` file
6. Copy to `pi/` directory

### 2. API Keys

Get keys from:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

### 3. Pi Setup

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv mpg123 espeak alsa-utils

# Clone and setup
cd /home/pi
git clone <your-repo> homie
cd homie/pi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env  # Add your API keys

# Copy hey-homie.ppn to this directory

# Test run
python main.py
```

### 4. Audio Setup

Make sure your Anker mic/speaker is the default device:

```bash
# List audio devices
arecord -l
aplay -l

# Test recording
arecord -d 3 test.wav
aplay test.wav
```

If needed, set default device in `/etc/asound.conf` or `~/.asoundrc`.

### 5. Run on Boot (Optional)

Create `/etc/systemd/system/homie.service`:

```ini
[Unit]
Description=Homie Voice Assistant
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/homie/pi
Environment=PATH=/home/pi/homie/pi/venv/bin
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

## Usage

1. Say "Hey Homie"
2. Wait for beep
3. Speak your command
4. Wait for response

60-second context window - after that, conversation resets.

## MCP Servers (TODO)

- [ ] Google Sheets (pantry + shopping list)
- [ ] Gmail
- [ ] Web search
- [ ] Spotify (future)

## Future: Pi + Mac Split

For lower latency or cost savings, can split to:
- Pi: wake word + audio I/O
- Mac: local Whisper + local TTS + Claude API

See `mac/` directory (not currently active).
