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
import pvporcupine
from pvrecorder import PvRecorder
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Config
PORCUPINE_ACCESS_KEY = os.environ.get("PORCUPINE_ACCESS_KEY")
WAKE_WORD_PATH = os.environ.get("WAKE_WORD_PATH", "hey-homie.ppn")
PIPER_MODEL_PATH = os.environ.get("PIPER_MODEL_PATH", os.path.expanduser("~/piper-voices/en_US-lessac-medium.onnx"))
SAMPLE_RATE = 16000
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
        # Reset if timed out
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
        self.porcupine = None
        self.recorder = None
        self.openai = OpenAI()
        self.anthropic = Anthropic()
        self.context = ConversationContext()

    def start(self):
        print("Starting Homie...")

        # Initialize Porcupine
        self.porcupine = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keyword_paths=[WAKE_WORD_PATH]
        )

        # Initialize recorder
        self.recorder = PvRecorder(
            device_index=-1,
            frame_length=self.porcupine.frame_length
        )

        print(f"Audio device: {self.recorder.selected_device}")
        print("Listening for 'Hey Homie'...")

        self.recorder.start()

        try:
            while True:
                pcm = self.recorder.read()
                keyword_index = self.porcupine.process(pcm)

                if keyword_index >= 0:
                    print("\n🎤 Wake word detected!")
                    self.play_listening_sound()
                    self.handle_command()

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()

    def handle_command(self):
        """Record speech, transcribe, process, respond."""
        # Record until silence
        audio = self.record_until_silence()
        if not audio:
            print("No speech detected")
            return

        # Transcribe
        print("📝 Transcribing...")
        transcript = self.transcribe(audio)
        if not transcript:
            print("Could not transcribe")
            return
        print(f"   You: {transcript}")

        # Process with Claude (streaming) and speak sentences as they arrive
        print("🤖 Thinking + Speaking...")
        full_response = self.process_and_speak_streaming(transcript)
        print(f"   [Full response: {full_response}]")

    def record_until_silence(self) -> bytes | None:
        """Record audio until silence is detected."""
        frames = []
        silence_frames = 0
        frames_per_second = SAMPLE_RATE // self.porcupine.frame_length
        silence_threshold_frames = int(SILENCE_DURATION * frames_per_second)
        max_duration_frames = 30 * frames_per_second  # Max 30 seconds

        print("   Recording...")

        for _ in range(max_duration_frames):
            pcm = self.recorder.read()
            frames.extend(pcm)

            # Check amplitude
            amplitude = max(abs(s) for s in pcm) if pcm else 0

            if amplitude < SILENCE_THRESHOLD:
                silence_frames += 1
            else:
                silence_frames = 0

            # Stop after enough silence (but need some audio first)
            if silence_frames >= silence_threshold_frames and len(frames) > SAMPLE_RATE:
                break

        if len(frames) < SAMPLE_RATE // 2:  # Less than 0.5s
            return None

        # Convert to WAV bytes
        return self.pcm_to_wav(frames)

    def pcm_to_wav(self, pcm: list) -> bytes:
        """Convert PCM samples to WAV bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(struct.pack(f'{len(pcm)}h', *pcm))
        return buffer.getvalue()

    def transcribe(self, audio_bytes: bytes) -> str | None:
        """Transcribe audio using OpenAI Whisper API."""
        try:
            # Create a file-like object
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"

            response = self.openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"  # Change to "he" for Hebrew or remove for auto-detect
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

                    # Check for complete sentences
                    while True:
                        match = sentence_endings.search(buffer)
                        if not match:
                            break
                        
                        # Extract sentence up to and including punctuation
                        end_pos = match.end()
                        sentence = buffer[:end_pos].strip()
                        buffer = buffer[end_pos:]

                        if sentence:
                            print(f"\n   🎯 Sentence complete: \"{sentence}\"")
                            self.speak_with_timing(sentence)

            # Speak any remaining text
            if buffer.strip():
                print(f"\n   🎯 Final chunk: \"{buffer.strip()}\"")
                self.speak_with_timing(buffer.strip())

            self.context.add_message("assistant", full_response)
            print(f"   ✅ Total time: {(time.time() - start_time)*1000:.0f}ms")
            return full_response

        except Exception as e:
            print(f"Claude error: {e}")
            self.speak("Sorry, I couldn't process that.")
            return "Sorry, I couldn't process that."

    def speak_with_timing(self, text: str):
        """Convert text to speech - uses macOS say or Piper on Linux."""
        try:
            t0 = time.time()
            print(f"   📤 TTS: \"{text[:50]}...\"" if len(text) > 50 else f"   📤 TTS: \"{text}\"")
            
            if os.uname().sysname == "Darwin":
                # macOS - use built-in say
                subprocess.run(["say", "-v", "Samantha", text], check=True)
                t1 = time.time()
                print(f"   ⏹️  Done: {(t1-t0)*1000:.0f}ms")
            else:
                # Linux (Pi) - use Piper
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

    def speak(self, text: str):
        """Convert text to speech and play using OpenAI TTS."""
        try:
            response = self.openai.audio.speech.create(
                model="tts-1",
                voice="nova",  # Options: alloy, echo, fable, onyx, nova, shimmer
                input=text,
                response_format="mp3"
            )

            # Save to temp file and play
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(response.content)
                temp_path = f.name

            # Play with afplay (macOS) or mpg123 (Linux)
            if os.uname().sysname == "Darwin":
                subprocess.run(["afplay", temp_path], check=True)
            else:
                subprocess.run(["mpg123", "-q", temp_path], check=True)
            os.unlink(temp_path)

        except Exception as e:
            print(f"TTS error: {e}")
            # Fallback to espeak
            self.speak_local(text)

    def speak_local(self, text: str):
        """Local TTS fallback."""
        try:
            subprocess.run(["espeak", text], check=True)
        except Exception as e:
            print(f"Local TTS error: {e}")

    def play_listening_sound(self):
        """Play a short sound to indicate listening started."""
        # Generate a short beep
        try:
            subprocess.run(
                ["speaker-test", "-t", "sine", "-f", "880", "-l", "1", "-p", "0.1"],
                capture_output=True,
                timeout=0.3
            )
        except:
            pass  # Not critical if this fails

    def cleanup(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder.delete()
        if self.porcupine:
            self.porcupine.delete()


def main():
    if not PORCUPINE_ACCESS_KEY:
        print("Error: PORCUPINE_ACCESS_KEY not set")
        print("Get your key at https://console.picovoice.ai/")
        return

    if not Path(WAKE_WORD_PATH).exists():
        print(f"Error: Wake word file not found: {WAKE_WORD_PATH}")
        print("Train 'Hey Homie' at https://console.picovoice.ai/")
        return

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
