"""
Homie - Voice-controlled home assistant
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from homie.assistant import Homie

load_dotenv()

# =============================================================================
# CONFIGURATION - All tweakable constants in one place
# =============================================================================

# --- Interaction Mode ---
INTERACTION_MODE = os.environ.get("INTERACTION_MODE", "audio")  # "audio" or "text"

# --- Environment Variables (from .env) ---
PORCUPINE_ACCESS_KEY = os.environ.get("PORCUPINE_ACCESS_KEY")
WAKE_WORD_PATH = os.environ.get("WAKE_WORD_PATH", "yo-home.ppn")
WAKE_PHRASE = os.environ.get("WAKE_PHRASE", "Yo Home")

# --- Audio Settings ---
SAMPLE_RATE = 16000                # Audio sample rate in Hz
SILENCE_THRESHOLD = 500            # Amplitude below this = silence
SILENCE_DURATION = 1.0             # Seconds of silence before stopping recording
MIN_RECORDING_DURATION = 3.0       # Minimum seconds to record before silence detection kicks in
MAX_RECORDING_DURATION = 30        # Maximum seconds to record
CLOSING_PHRASES = ["bye home", "thanks home", "that's all"]  # Phrases to end recording immediately

# --- Chime Settings ---
CHIME_VOLUME = 0.2                 # Volume of acknowledgement chimes (0.0 to 1.0)
ALERT_CHIME_VOLUME = 0.6           # Volume of alert/timer chimes (0.0 to 1.0)
CHIME_FADE_DURATION = 0.02         # Fade in/out duration in seconds
CHIME_FREQ_LOW = 523.25            # C5 note frequency
CHIME_FREQ_HIGH = 659.25           # E5 note frequency
CHIME_TONE1_DURATION = 0.1         # First tone duration
CHIME_TONE2_DURATION = 0.15        # Second tone duration

# --- Conversation Settings ---
CONTEXT_TIMEOUT = 60               # Seconds before conversation context resets

# --- Model Settings ---
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 1500
WHISPER_MODEL = "whisper-1"
WHISPER_LANGUAGE = "en"            # Change to "he" for Hebrew or None for auto-detect
TTS_MODEL = "tts-1"
TTS_VOICE = "nova"                 # Options: alloy, echo, fable, onyx, nova, shimmer

# --- MCP Settings ---
ENABLE_SYSTEM_MCP = os.environ.get("ENABLE_SYSTEM_MCP", "true").lower() == "true"
ENABLE_CALENDAR_MCP = os.environ.get("ENABLE_CALENDAR_MCP", "true").lower() == "true"
DEFAULT_CALENDAR_ID = os.environ.get("DEFAULT_CALENDAR_ID", "primary")
GOOGLE_SERVICE_ACCOUNT_PATH = os.environ.get(
    "GOOGLE_SERVICE_ACCOUNT_PATH",
    str(Path(__file__).parent.parent / "secrets" / "google-calendar.json")
)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    # Check required API keys
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        return

    # Audio mode requires additional setup
    if INTERACTION_MODE == "audio":
        if not PORCUPINE_ACCESS_KEY:
            print("Error: PORCUPINE_ACCESS_KEY not set")
            print("Get your key at https://console.picovoice.ai/")
            return

        if not Path(WAKE_WORD_PATH).exists():
            print(f"Error: Wake word file not found: {WAKE_WORD_PATH}")
            print("Train your wake word at https://console.picovoice.ai/")
            return

        if not os.environ.get("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set (required for Whisper STT)")
            return

    homie = Homie(
        mode=INTERACTION_MODE,
        sample_rate=SAMPLE_RATE,
        silence_threshold=SILENCE_THRESHOLD,
        silence_duration=SILENCE_DURATION,
        min_recording_duration=MIN_RECORDING_DURATION,
        max_recording_duration=MAX_RECORDING_DURATION,
        closing_phrases=CLOSING_PHRASES,
        chime_volume=CHIME_VOLUME,
        alert_chime_volume=ALERT_CHIME_VOLUME,
        chime_fade_duration=CHIME_FADE_DURATION,
        chime_freq_low=CHIME_FREQ_LOW,
        chime_freq_high=CHIME_FREQ_HIGH,
        chime_tone1_duration=CHIME_TONE1_DURATION,
        chime_tone2_duration=CHIME_TONE2_DURATION,
        context_timeout=CONTEXT_TIMEOUT,
        claude_model=CLAUDE_MODEL,
        claude_max_tokens=CLAUDE_MAX_TOKENS,
        whisper_model=WHISPER_MODEL,
        whisper_language=WHISPER_LANGUAGE,
        tts_model=TTS_MODEL,
        tts_voice=TTS_VOICE,
        enable_system_mcp=ENABLE_SYSTEM_MCP,
        enable_calendar_mcp=ENABLE_CALENDAR_MCP,
        default_calendar_id=DEFAULT_CALENDAR_ID,
        google_service_account_path=GOOGLE_SERVICE_ACCOUNT_PATH,
        porcupine_access_key=PORCUPINE_ACCESS_KEY,
        wake_word_path=WAKE_WORD_PATH,
        wake_phrase=WAKE_PHRASE
    )
    homie.start()


if __name__ == "__main__":
    main()
