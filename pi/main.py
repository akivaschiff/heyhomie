from __future__ import annotations

import io
import os
import re
import struct
import subprocess
import tempfile
import time
import wave
import threading
import queue
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
WAKE_WORD_PATH = os.environ.get("WAKE_WORD_PATH", "yo-home.ppn")
WAKE_PHRASE = os.environ.get("WAKE_PHRASE", "Yo Home")
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5  # seconds of silence to stop
MIN_RECORDING_DURATION = 3.0  # seconds before silence detection kicks in
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


def generate_tone(frequency: float, duration: float, sample_rate: int = 16000, fade: bool = True) -> bytes:
    """Generate a pleasant tone as WAV bytes."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    
    # Add fade in/out for a softer sound
    if fade:
        fade_samples = int(sample_rate * 0.02)  # 20ms fade
        tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
        tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Convert to 16-bit PCM (0.2 = quieter volume)
    tone = (tone * 32767 * 0.2).astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(tone.tobytes())
    return buffer.getvalue()


def generate_chime(rising: bool = True) -> bytes:
    """Generate a pleasant two-tone chime."""
    if rising:
        # Rising chime: "I'm listening"
        tone1 = generate_tone(523.25, 0.1)  # C5
        tone2 = generate_tone(659.25, 0.15)  # E5
    else:
        # Falling chime: "Got it, processing"
        tone1 = generate_tone(659.25, 0.1)  # E5
        tone2 = generate_tone(523.25, 0.15)  # C5
    
    # Combine tones
    buffer1 = io.BytesIO(tone1)
    buffer2 = io.BytesIO(tone2)
    
    with wave.open(buffer1, 'rb') as w1, wave.open(buffer2, 'rb') as w2:
        frames1 = w1.readframes(w1.getnframes())
        frames2 = w2.readframes(w2.getnframes())
    
    combined = io.BytesIO()
    with wave.open(combined, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(frames1 + frames2)
    
    return combined.getvalue()


class TTSPipeline:
    """Async TTS pipeline - generates and plays audio without blocking Claude stream."""
    
    def __init__(self, openai_client):
        self.openai = openai_client
        self.tts_queue = queue.Queue()      # Text waiting for TTS
        self.playback_queue = queue.Queue() # Audio waiting for playback
        self.tts_thread = None
        self.playback_thread = None
        self.running = False
    
    def start(self):
        """Start the TTS and playback worker threads."""
        self.running = True
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.tts_thread.start()
        self.playback_thread.start()
    
    def stop(self):
        """Stop the workers."""
        self.running = False
    
    def submit(self, text: str):
        """Submit text for TTS (non-blocking)."""
        self.tts_queue.put(text)
    
    def finish_and_wait(self):
        """Signal no more text coming, wait for all audio to finish playing."""
        self.tts_queue.put(None)  # Sentinel
        self.tts_queue.join()     # Wait for TTS to finish
        self.playback_queue.put(None)  # Sentinel
        self.playback_queue.join()     # Wait for playback to finish
    
    def _tts_worker(self):
        """Worker thread: pull text from queue, generate audio, push to playback."""
        while self.running:
            try:
                text = self.tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if text is None:  # Sentinel
                self.tts_queue.task_done()
                break
            
            try:
                t0 = time.time()
                print(f"   📤 TTS: \"{text[:50]}...\"" if len(text) > 50 else f"   📤 TTS: \"{text}\"")
                
                if os.uname().sysname == "Darwin":
                    # macOS - use say command, pass text directly to playback
                    self.playback_queue.put(("say", text))
                else:
                    # Linux - use OpenAI TTS
                    response = self.openai.audio.speech.create(
                        model="tts-1",
                        voice="nova",
                        input=text,
                        response_format="mp3"
                    )
                    t1 = time.time()
                    print(f"   📥 TTS response: {(t1-t0)*1000:.0f}ms")
                    self.playback_queue.put(("mp3", response.content))
            except Exception as e:
                print(f"   TTS error: {e}")
            finally:
                self.tts_queue.task_done()
    
    def _playback_worker(self):
        """Worker thread: pull audio from queue, play it."""
        while self.running:
            try:
                item = self.playback_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if item is None:  # Sentinel
                self.playback_queue.task_done()
                break
            
            try:
                audio_type, data = item
                t0 = time.time()
                
                if audio_type == "say":
                    subprocess.run(["say", "-v", "Samantha", data], check=True)
                elif audio_type == "mp3":
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        f.write(data)
                        temp_path = f.name
                    subprocess.run(["mpg123", "-q", temp_path], check=True)
                    os.unlink(temp_path)
                
                t1 = time.time()
                print(f"   ⏹️  Playback: {(t1-t0)*1000:.0f}ms")
            except Exception as e:
                print(f"   Playback error: {e}")
            finally:
                self.playback_queue.task_done()


class Homie:
    def __init__(self):
        self.porcupine = None
        self.recorder = None
        self.openai = OpenAI()
        self.anthropic = Anthropic()
        self.context = ConversationContext()
        self.tts_pipeline = TTSPipeline(self.openai)
        
        # Pre-generate chimes
        self.listening_chime = generate_chime(rising=True)
        self.processing_chime = generate_chime(rising=False)

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
        
        # Start TTS pipeline
        self.tts_pipeline.start()

        print(f"Audio device: {self.recorder.selected_device}")
        print(f"Listening for '{WAKE_PHRASE}'...")

        self.recorder.start()

        try:
            while True:
                pcm = self.recorder.read()
                keyword_index = self.porcupine.process(pcm)

                if keyword_index >= 0:
                    print(f"\n🎤 Wake word detected!")
                    self.play_chime(self.listening_chime)
                    try:
                        self.handle_command()
                    except Exception as e:
                        print(f"Error handling command: {e}")
                    print(f"\nListening for '{WAKE_PHRASE}'...")

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()

    def handle_command(self):
        """Record speech, transcribe, process, respond."""
        audio = self.record_until_silence()
        
        # Play "got it" chime after recording stops
        self.play_chime(self.processing_chime)
        
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
        frames_per_second = SAMPLE_RATE // self.porcupine.frame_length
        silence_threshold_frames = int(SILENCE_DURATION * frames_per_second)
        min_frames = int(MIN_RECORDING_DURATION * frames_per_second)
        max_duration_frames = 30 * frames_per_second

        print("   Recording...")

        frame_count = 0
        for _ in range(max_duration_frames):
            pcm = self.recorder.read()
            frames.extend(pcm)
            frame_count += 1

            amplitude = max(abs(s) for s in pcm) if pcm else 0

            if amplitude < SILENCE_THRESHOLD:
                silence_frames += 1
            else:
                silence_frames = 0

            # Only check for silence after minimum recording time
            if frame_count >= min_frames and silence_frames >= silence_threshold_frames:
                break

        if len(frames) < SAMPLE_RATE // 2:
            return None

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
                            self.tts_pipeline.submit(sentence)  # Non-blocking!

            if buffer.strip():
                print(f"\n   🎯 Final chunk: \"{buffer.strip()}\"")
                self.tts_pipeline.submit(buffer.strip())  # Non-blocking!

            # Wait for all audio to finish playing
            self.tts_pipeline.finish_and_wait()
            
            # Restart pipeline for next command
            self.tts_pipeline = TTSPipeline(self.openai)
            self.tts_pipeline.start()

            self.context.add_message("assistant", full_response)
            print(f"   ✅ Total time: {(time.time() - start_time)*1000:.0f}ms")
            return full_response

        except Exception as e:
            print(f"Claude error: {e}")
            self.speak("Sorry, I couldn't process that.")
            return "Sorry, I couldn't process that."

    def speak(self, text: str):
        """Convert text to speech using OpenAI TTS."""
        try:
            t0 = time.time()
            print(f"   📤 TTS: \"{text[:50]}...\"" if len(text) > 50 else f"   📤 TTS: \"{text}\"")

            if os.uname().sysname == "Darwin":
                # macOS - use built-in say for speed
                subprocess.run(["say", "-v", "Samantha", text], check=True)
            else:
                # Linux - use OpenAI TTS for better quality
                response = self.openai.audio.speech.create(
                    model="tts-1",
                    voice="nova",
                    input=text,
                    response_format="mp3"
                )

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(response.content)
                    temp_path = f.name

                subprocess.run(["mpg123", "-q", temp_path], check=True)
                os.unlink(temp_path)

            t1 = time.time()
            print(f"   ⏹️  Done: {(t1-t0)*1000:.0f}ms")

        except Exception as e:
            print(f"TTS error: {e}")

    def play_chime(self, chime_data: bytes):
        """Play a chime sound."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(chime_data)
                temp_path = f.name

            if os.uname().sysname == "Darwin":
                subprocess.run(["afplay", temp_path], check=True)
            else:
                subprocess.run(["aplay", "-q", temp_path], check=True)

            os.unlink(temp_path)
        except Exception as e:
            print(f"Chime error: {e}")

    def cleanup(self):
        if self.tts_pipeline:
            self.tts_pipeline.stop()
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
        print("Train your wake word at https://console.picovoice.ai/")
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
