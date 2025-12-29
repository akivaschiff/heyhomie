from __future__ import annotations

import io
import os
import re
import struct
import subprocess
import tempfile
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from openwakeword.model import Model
import pyaudio
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Config
WAKE_WORD = os.environ.get("WAKE_WORD", "hey_jarvis")  # hey_jarvis, alexa, hey_mycroft, etc.
WAKE_THRESHOLD = float(os.environ.get("WAKE_THRESHOLD", "0.5"))
PIPER_MODEL_PATH = os.environ.get("PIPER_MODEL_PATH", os.path.expanduser("~/piper-voices/en_US-lessac-medium.onnx"))
SAMPLE_RATE = 16000
FRAME_LENGTH = 1280  # 80ms chunks for OpenWakeWord
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5  # seconds
CONTEXT_TIMEOUT = 60    # seconds

SYSTEM_PROMPT = """You are Homie, a friendly home assistant. You help with:
- Managing shopping lists and pantry inventory (via Google Sheets)
- Reading and responding to emails (via Gmail)
- Managing chores and tasks
- Answering questions (via web search)

Keep responses concise and conversational - they will be spoken aloud.
Aim for 1-2 sentences when possible.

When asked to perform an action, confirm what you're about to do and wait for confirmation.
Example: "I'll add hummus to the shopping list. Should I do that?"

After user confirms with "yes", "yeah", "do it", "go ahead", etc., execute the action.
If they say "no", "cancel", "never mind", acknowledge and don't execute.
"""


class ConversationContext:
    def __init__(self, timeout_seconds: int = CONTEXT_TIMEOUT):
        self.messages = []
        self.last_interaction = None
        self.timeout = timedelta(seconds=timeout_seconds)

    def add_message(self, role: str, content: str):
        now = datetime.now()
        if self.last_interaction and now - self.last_interaction > self.timeout:
            print("Context timeout - starting fresh")
            self.messages = []
        self.messages.append({"role": role, "content": content})
        self.last_interaction = now

    def get_messages(self):
        if self.last_interaction and datetime.now() - self.last_interaction > self.timeout:
            self.messages = []
        return self.messages


class Homie:
    def __init__(self):
        self.oww_model = None
        self.audio = None
        self.stream = None
        self.openai = OpenAI()
        self.anthropic = Anthropic()
        self.context = ConversationContext()

    def start(self):
        print("Starting Homie...")

        # Initialize OpenWakeWord
        print(f"Loading wake word model: {WAKE_WORD}")
        self.oww_model = Model(wakeword_models=[WAKE_WORD])

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_LENGTH
        )

        print(f"Listening for '{WAKE_WORD.replace('_', ' ')}'...")

        try:
            while True:
                # Read audio chunk
                audio_data = self.stream.read(FRAME_LENGTH, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.int16)

                # Process with OpenWakeWord
                prediction = self.oww_model.predict(audio_array)

                # Check if wake word detected
                for model_name, score in prediction.items():
                    if score > WAKE_THRESHOLD:
                        print(f"\n🎤 Wake word detected! ({model_name}: {score:.2f})")
                        self.oww_model.reset()  # Reset to avoid re-triggering
                        self.play_listening_sound()
                        self.handle_command()

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()

    def handle_command(self):
        """Record speech, transcribe, process, respond."""
        audio = self.record_until_silence()
        if not audio:
            print("No speech detected")
            return

        print("📝 Transcribing...")
        transcript = self.transcribe(audio)
        if not transcript:
            print("Could not transcribe")
            return
        print(f"   You: {transcript}")

        print("🤖 Thinking + Speaking...")
        full_response = self.process_and_speak_streaming(transcript)
        print(f"   [Full response: {full_response}]")

    def record_until_silence(self) -> bytes | None:
        """Record audio until silence is detected."""
        frames = []
        silence_frames = 0
        frames_per_second = SAMPLE_RATE // FRAME_LENGTH
        silence_threshold_frames = int(SILENCE_DURATION * frames_per_second)
        max_duration_frames = 30 * frames_per_second

        print("   Recording...")

        for _ in range(max_duration_frames):
            audio_data = self.stream.read(FRAME_LENGTH, exception_on_overflow=False)
            frames.append(audio_data)

            # Check amplitude
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            amplitude = np.max(np.abs(audio_array))

            if amplitude < SILENCE_THRESHOLD:
                silence_frames += 1
            else:
                silence_frames = 0

            if silence_frames >= silence_threshold_frames and len(frames) > frames_per_second:
                break

        if len(frames) < frames_per_second // 2:
            return None

        # Convert to WAV bytes
        return self.frames_to_wav(frames)

    def frames_to_wav(self, frames: list) -> bytes:
        """Convert audio frames to WAV bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
        return buffer.getvalue()

    def transcribe(self, audio_bytes: bytes) -> str | None:
        """Transcribe audio using OpenAI Whisper API."""
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"

            response = self.openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
            return response.text.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def process_and_speak_streaming(self, user_message: str) -> str:
        """Stream Claude response and speak sentences as they complete."""
        self.context.add_message("user", user_message)
        full_response = ""
        buffer = ""
        sentence_endings = re.compile(r'([.!?])\s+')
        start_time = time.time()
        first_token_time = None

        try:
            with self.anthropic.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=self.context.get_messages()
            ) as stream:
                for text in stream.text_stream:
                    if first_token_time is None:
                        first_token_time = time.time()
                        print(f"   ⚡ First token: {(first_token_time - start_time)*1000:.0f}ms")

                    buffer += text
                    full_response += text
                    print(f"   📥 +\"{text}\"", end="", flush=True)

                    while True:
                        match = sentence_endings.search(buffer)
                        if not match:
                            break

                        end_pos = match.end()
                        sentence = buffer[:end_pos].strip()
                        buffer = buffer[end_pos:]

                        if sentence:
                            print(f"\n   🎯 Sentence complete: \"{sentence}\"")
                            self.speak_with_timing(sentence)

            if buffer.strip():
                print(f"\n   🎯 Final chunk: \"{buffer.strip()}\"")
                self.speak_with_timing(buffer.strip())

            self.context.add_message("assistant", full_response)
            print(f"   ✅ Total time: {(time.time() - start_time)*1000:.0f}ms")
            return full_response

        except Exception as e:
            print(f"Claude error: {e}")
            self.speak_local("Sorry, I couldn't process that.")
            return "Sorry, I couldn't process that."

    def speak_with_timing(self, text: str):
        """Convert text to speech - uses macOS say or Piper on Linux."""
        try:
            t0 = time.time()
            print(f"   📤 TTS: \"{text[:50]}...\"" if len(text) > 50 else f"   📤 TTS: \"{text}\"")

            if os.uname().sysname == "Darwin":
                subprocess.run(["say", "-v", "Samantha", text], check=True)
                t1 = time.time()
                print(f"   ⏹️  Done: {(t1-t0)*1000:.0f}ms")
            else:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name

                process = subprocess.run(
                    ["piper", "--model", PIPER_MODEL_PATH, "--output_file", temp_path],
                    input=text,
                    capture_output=True,
                    text=True
                )

                t1 = time.time()
                print(f"   🔊 Piper: {(t1-t0)*1000:.0f}ms")

                if process.returncode != 0:
                    print(f"   Piper error: {process.stderr}")
                    self.speak_local(text)
                    return

                t2 = time.time()
                subprocess.run(["aplay", "-q", temp_path], check=True)
                t3 = time.time()
                print(f"   ⏹️  Playback: {(t3-t2)*1000:.0f}ms")

                os.unlink(temp_path)

        except Exception as e:
            print(f"TTS error: {e}")
            self.speak_local(text)

    def speak_local(self, text: str):
        """Local TTS fallback."""
        try:
            subprocess.run(["espeak", text], check=True)
        except Exception as e:
            print(f"Local TTS error: {e}")

    def play_listening_sound(self):
        """Play a short sound to indicate listening started."""
        try:
            if os.uname().sysname == "Darwin":
                subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"], check=True)
            else:
                subprocess.run(
                    ["speaker-test", "-t", "sine", "-f", "880", "-l", "1", "-p", "0.1"],
                    capture_output=True,
                    timeout=0.3
                )
        except:
            pass

    def cleanup(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        return

    homie = Homie()
    homie.start()


if __name__ == "__main__":
    main()
