"""
Homie - Voice-controlled home assistant
"""

from __future__ import annotations

import io
import json
import os
import queue
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

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

# Conditional imports for audio mode only
if INTERACTION_MODE == "audio":
    import pvporcupine
    from pvrecorder import PvRecorder

# --- Audio Settings ---
SAMPLE_RATE = 16000                # Audio sample rate in Hz
SILENCE_THRESHOLD = 500            # Amplitude below this = silence
SILENCE_DURATION = 1.5             # Seconds of silence before stopping recording
MIN_RECORDING_DURATION = 3.0       # Minimum seconds to record before silence detection kicks in
MAX_RECORDING_DURATION = 30        # Maximum seconds to record

# --- Chime Settings ---
CHIME_VOLUME = 0.2                 # Volume of acknowledgement chimes (0.0 to 1.0)
CHIME_FADE_DURATION = 0.02         # Fade in/out duration in seconds
CHIME_FREQ_LOW = 523.25            # C5 note frequency
CHIME_FREQ_HIGH = 659.25           # E5 note frequency
CHIME_TONE1_DURATION = 0.1         # First tone duration
CHIME_TONE2_DURATION = 0.15        # Second tone duration

# --- Conversation Settings ---
CONTEXT_TIMEOUT = 60               # Seconds before conversation context resets

# --- Model Settings ---
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 300
WHISPER_MODEL = "whisper-1"
WHISPER_LANGUAGE = "en"            # Change to "he" for Hebrew or None for auto-detect
TTS_MODEL = "tts-1"
TTS_VOICE = "nova"                 # Options: alloy, echo, fable, onyx, nova, shimmer

# --- MCP Settings ---
ENABLE_CALENDAR_MCP = os.environ.get("ENABLE_CALENDAR_MCP", "true").lower() == "true"
DEFAULT_CALENDAR_ID = os.environ.get("DEFAULT_CALENDAR_ID", "primary")
GOOGLE_SERVICE_ACCOUNT_PATH = os.environ.get(
    "GOOGLE_SERVICE_ACCOUNT_PATH",
    str(Path(__file__).parent.parent / "secrets" / "google-calendar.json")
)

# --- System Prompt ---
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


# =============================================================================
# HELPER CLASSES
# =============================================================================

class MCPClient:
    """Client for communicating with MCP servers via stdio."""

    def __init__(self, server_command: list[str], env: dict = None):
        """Initialize MCP client with server command.

        Args:
            server_command: Command to start the MCP server (e.g., ["node", "build/index.js"])
            env: Environment variables to pass to the server
        """
        self.server_command = server_command
        self.env = env or {}
        self.process = None
        self.tools = []
        self.message_id = 0

    def start(self):
        """Start the MCP server process."""
        env = os.environ.copy()
        env.update(self.env)

        try:
            self.process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1
            )

            # Initialize and list tools
            self._initialize()
            self._list_tools()
        except Exception:
            # Clean up if initialization fails
            if self.process:
                self.process.kill()
                self.process = None
            raise

    def _initialize(self):
        """Send initialize request to the MCP server."""
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "homie",
                    "version": "1.0.0"
                }
            }
        }
        self._send_request(init_request)
        response = self._read_response()
        if "error" in response:
            raise RuntimeError(f"MCP initialize failed: {response['error']}")

        # Send initialized notification
        initialized_notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        self._send_request(initialized_notif)

    def _list_tools(self):
        """Fetch available tools from the MCP server."""
        list_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {}
        }
        self._send_request(list_request)
        response = self._read_response()

        if "error" in response:
            raise RuntimeError(f"MCP tools/list failed: {response['error']}")

        self.tools = response.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as a dictionary
        """
        call_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        self._send_request(call_request)
        response = self._read_response()

        if "error" in response:
            return {"error": response["error"]}

        return response.get("result", {})

    def get_anthropic_tools(self) -> list[dict]:
        """Convert MCP tools to Anthropic tool format."""
        anthropic_tools = []
        for tool in self.tools:
            # Convert MCP tool schema to Anthropic format
            input_schema = tool.get("inputSchema", {"type": "object", "properties": {}})
            anthropic_tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": input_schema
            })
        return anthropic_tools

    def _next_id(self):
        """Generate next message ID."""
        self.message_id += 1
        return self.message_id

    def _send_request(self, request: dict):
        """Send a JSON-RPC request to the server."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server not started")

        message = json.dumps(request) + "\n"
        self.process.stdin.write(message)
        self.process.stdin.flush()

    def _read_response(self, timeout: float = 30.0) -> dict:
        """Read a JSON-RPC response from the server with timeout."""
        if not self.process or not self.process.stdout:
            raise RuntimeError("MCP server not started")

        import select
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)

        if not ready:
            raise TimeoutError(f"MCP server did not respond within {timeout}s")

        line = self.process.stdout.readline()
        if not line:
            if self.process.poll() is not None:
                raise RuntimeError(f"MCP server terminated (exit code: {self.process.returncode})")
            raise RuntimeError("MCP server closed connection")

        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from MCP: {line[:100]}") from e

    def stop(self):
        """Stop the MCP server process."""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.stdout.close()
                self.process.stderr.close()
            except Exception:
                pass  # Already closed

            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
            except Exception as e:
                print(f"Warning: Error stopping MCP server: {e}")
            finally:
                self.process = None


class ConversationContext:
    """Manages conversation history with automatic timeout."""
    
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


class TTSPipeline:
    """Async TTS pipeline - generates and plays audio without blocking Claude stream.
    In text mode, this just prints to stdout instead of generating audio."""

    def __init__(self, openai_client, mode="audio"):
        self.openai = openai_client
        self.mode = mode
        self.tts_queue = queue.Queue()
        self.playback_queue = queue.Queue()
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
        self.tts_queue.put(None)
        self.tts_queue.join()
        self.playback_queue.put(None)
        self.playback_queue.join()
    
    def _tts_worker(self):
        """Worker thread: pull text from queue, generate audio, push to playback."""
        while self.running:
            try:
                text = self.tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if text is None:
                self.tts_queue.task_done()
                break

            try:
                t0 = time.time()

                if self.mode == "text":
                    # In text mode, just print the response
                    print(f"Homie: {text}")
                    self.tts_queue.task_done()
                    continue

                print(f"   📤 TTS: \"{text[:50]}...\"" if len(text) > 50 else f"   📤 TTS: \"{text}\"")

                if os.uname().sysname == "Darwin":
                    self.playback_queue.put(("say", text))
                else:
                    response = self.openai.audio.speech.create(
                        model=TTS_MODEL,
                        voice=TTS_VOICE,
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

            if item is None:
                self.playback_queue.task_done()
                break

            # Skip playback in text mode
            if self.mode == "text":
                self.playback_queue.task_done()
                continue

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


# =============================================================================
# AUDIO UTILITIES
# =============================================================================

def generate_tone(frequency: float, duration: float) -> bytes:
    """Generate a pleasant tone as WAV bytes."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    
    # Add fade in/out for a softer sound
    fade_samples = int(SAMPLE_RATE * CHIME_FADE_DURATION)
    tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Convert to 16-bit PCM with configured volume
    tone = (tone * 32767 * CHIME_VOLUME).astype(np.int16)
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(tone.tobytes())
    return buffer.getvalue()


def generate_chime(rising: bool = True) -> bytes:
    """Generate a pleasant two-tone chime."""
    if rising:
        tone1 = generate_tone(CHIME_FREQ_LOW, CHIME_TONE1_DURATION)
        tone2 = generate_tone(CHIME_FREQ_HIGH, CHIME_TONE2_DURATION)
    else:
        tone1 = generate_tone(CHIME_FREQ_HIGH, CHIME_TONE1_DURATION)
        tone2 = generate_tone(CHIME_FREQ_LOW, CHIME_TONE2_DURATION)
    
    buffer1 = io.BytesIO(tone1)
    buffer2 = io.BytesIO(tone2)
    
    with wave.open(buffer1, 'rb') as w1, wave.open(buffer2, 'rb') as w2:
        frames1 = w1.readframes(w1.getnframes())
        frames2 = w2.readframes(w2.getnframes())
    
    combined = io.BytesIO()
    with wave.open(combined, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(frames1 + frames2)
    
    return combined.getvalue()


def pcm_to_wav(pcm: list) -> bytes:
    """Convert PCM samples to WAV bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f'{len(pcm)}h', *pcm))
    return buffer.getvalue()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class Homie:
    """Main voice assistant class. Supports both audio and text modes."""

    def __init__(self, mode="audio"):
        self.mode = mode
        self.porcupine = None
        self.recorder = None
        self.openai = OpenAI()
        self.anthropic = Anthropic()
        self.context = ConversationContext()
        self.tts_pipeline = TTSPipeline(self.openai, mode=mode)
        self.mcp_client = None

        # Pre-generate chimes (only used in audio mode)
        if mode == "audio":
            self.listening_chime = generate_chime(rising=True)
            self.processing_chime = generate_chime(rising=False)

        # Initialize MCP client if enabled
        if ENABLE_CALENDAR_MCP:
            self._init_calendar_mcp()

    def _init_calendar_mcp(self):
        """Initialize the calendar MCP server."""
        try:
            # Validate service account file exists
            creds_path = Path(GOOGLE_SERVICE_ACCOUNT_PATH)
            if not creds_path.exists():
                print(f"⚠️  Calendar MCP disabled: Credentials not found at {creds_path}")
                self.mcp_client = None
                return

            # Validate MCP build exists
            mcp_path = Path(__file__).parent.parent / "mcps" / "calendar"
            mcp_binary = mcp_path / "build" / "index.js"
            if not mcp_binary.exists():
                print(f"⚠️  Calendar MCP disabled: Build not found. Run: cd {mcp_path} && npm run build")
                self.mcp_client = None
                return

            server_command = ["node", str(mcp_binary)]
            env = {
                "GOOGLE_SERVICE_ACCOUNT_PATH": GOOGLE_SERVICE_ACCOUNT_PATH,
                "DEFAULT_CALENDAR_ID": DEFAULT_CALENDAR_ID
            }

            self.mcp_client = MCPClient(server_command, env)
            self.mcp_client.start()
            print(f"✅ Calendar MCP initialized with {len(self.mcp_client.tools)} tools")
            print(f"   Using calendar: {DEFAULT_CALENDAR_ID}")
        except Exception as e:
            print(f"⚠️  Failed to initialize Calendar MCP: {e}")
            self.mcp_client = None

    def start(self):
        """Start the assistant in either audio or text mode."""
        if self.mode == "audio":
            self._start_audio_mode()
        else:
            self._start_text_mode()

    def _start_audio_mode(self):
        """Start the voice assistant in audio mode."""
        print("Starting Homie in AUDIO mode...")

        self.porcupine = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keyword_paths=[WAKE_WORD_PATH]
        )

        self.recorder = PvRecorder(
            device_index=-1,
            frame_length=self.porcupine.frame_length
        )

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

    def _start_text_mode(self):
        """Start the assistant in text mode (stdin/stdout)."""
        print("Starting Homie in TEXT mode...")
        print("Type your messages and press Enter. Use Ctrl+C to exit.\n")

        self.tts_pipeline.start()

        try:
            while True:
                try:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue

                    print("🤖 Thinking...")
                    full_response = self.process_and_speak_streaming(user_input)
                    print()  # Add a blank line for readability

                except EOFError:
                    break

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()

    def handle_command(self):
        """Record speech, transcribe, process, respond (audio mode only)."""
        audio = self.record_until_silence()

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
        max_frames = int(MAX_RECORDING_DURATION * frames_per_second)

        print("   Recording...")

        frame_count = 0
        for _ in range(max_frames):
            pcm = self.recorder.read()
            frames.extend(pcm)
            frame_count += 1

            amplitude = max(abs(s) for s in pcm) if pcm else 0

            if amplitude < SILENCE_THRESHOLD:
                silence_frames += 1
            else:
                silence_frames = 0

            if frame_count >= min_frames and silence_frames >= silence_threshold_frames:
                break

        if len(frames) < SAMPLE_RATE // 2:
            return None

        return pcm_to_wav(frames)

    def transcribe(self, audio_bytes: bytes) -> str | None:
        """Transcribe audio using OpenAI Whisper API."""
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"

            response = self.openai.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                language=WHISPER_LANGUAGE
            )
            return response.text.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def process_and_speak_streaming(self, user_message: str) -> str:
        """Stream Claude response and speak sentences as they complete."""
        self.context.add_message("user", user_message)
        start_time = time.time()

        # Prepare tools if MCP is available
        tools = None
        if self.mcp_client:
            tools = self.mcp_client.get_anthropic_tools()

        try:
            # First API call to Claude
            response = self.anthropic.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=self.context.get_messages(),
                tools=tools if tools else None
            )

            # Handle tool use
            if response.stop_reason == "tool_use":
                return self._handle_tool_use(response, start_time)

            # Handle text response
            full_response = ""
            for block in response.content:
                if block.type == "text":
                    full_response += block.text

            # Speak the response
            self._speak_text(full_response)

            self.context.add_message("assistant", full_response)
            if self.mode == "audio":
                print(f"   ✅ Total time: {(time.time() - start_time)*1000:.0f}ms")
            return full_response

        except Exception as e:
            print(f"Claude error: {e}")
            if self.mode == "audio":
                self.speak_error("Sorry, I couldn't process that.")
            else:
                print("Homie: Sorry, I couldn't process that.")
            return "Sorry, I couldn't process that."

    def _handle_tool_use(self, response, start_time) -> str:
        """Handle tool use in Claude's response."""
        if not self.mcp_client:
            return "Sorry, I don't have access to those tools right now."

        tool_results = []
        text_parts = []

        # Process all content blocks
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                tool_id = block.id

                if self.mode == "audio":
                    print(f"   🔧 Calling tool: {tool_name}")
                else:
                    print(f"🔧 Calling tool: {tool_name} with {tool_input}")

                # Call the MCP tool
                result = self.mcp_client.call_tool(tool_name, tool_input)

                # Extract text content from MCP response
                content_text = ""
                if "content" in result:
                    for content_item in result["content"]:
                        if content_item.get("type") == "text":
                            content_text += content_item.get("text", "")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": content_text or json.dumps(result)
                })

        # Add assistant's response to context (with tool use blocks)
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

        self.context.messages.append({
            "role": "assistant",
            "content": assistant_content
        })

        # Add tool results to context
        self.context.messages.append({
            "role": "user",
            "content": tool_results
        })

        # Make follow-up call to get Claude's response with tool results
        follow_up = self.anthropic.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=self.context.messages,
            tools=self.mcp_client.get_anthropic_tools() if self.mcp_client else None
        )

        full_response = ""
        for block in follow_up.content:
            if block.type == "text":
                full_response += block.text

        # Speak the final response
        self._speak_text(full_response)

        # Update context with final response
        self.context.messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": full_response}]
        })

        if self.mode == "audio":
            print(f"   ✅ Total time: {(time.time() - start_time)*1000:.0f}ms")

        return full_response

    def _speak_text(self, text: str):
        """Speak text by splitting into sentences."""
        if not text:
            return

        # In text mode, just print the whole response
        if self.mode == "text":
            print(f"Homie: {text}")
            return

        # Audio mode: split into sentences for streaming TTS
        buffer = text
        sentence_endings = re.compile(r'([.!?])\s+')

        while True:
            match = sentence_endings.search(buffer)
            if not match:
                break

            end_pos = match.end()
            sentence = buffer[:end_pos].strip()
            buffer = buffer[end_pos:]

            if sentence:
                print(f"   🎯 Sentence: \"{sentence}\"")
                self.tts_pipeline.submit(sentence)

        if buffer.strip():
            print(f"   🎯 Final: \"{buffer.strip()}\"")
            self.tts_pipeline.submit(buffer.strip())

        self.tts_pipeline.finish_and_wait()

        # Restart pipeline for next command
        self.tts_pipeline = TTSPipeline(self.openai, mode=self.mode)
        self.tts_pipeline.start()

    def speak_error(self, text: str):
        """Speak an error message (blocking, used for error handling)."""
        try:
            if os.uname().sysname == "Darwin":
                subprocess.run(["say", "-v", "Samantha", text], check=True)
            else:
                response = self.openai.audio.speech.create(
                    model=TTS_MODEL,
                    voice=TTS_VOICE,
                    input=text,
                    response_format="mp3"
                )
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(response.content)
                    temp_path = f.name
                subprocess.run(["mpg123", "-q", temp_path], check=True)
                os.unlink(temp_path)
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
        """Clean up resources."""
        if self.tts_pipeline:
            self.tts_pipeline.stop()
        if self.recorder:
            self.recorder.stop()
            self.recorder.delete()
        if self.porcupine:
            self.porcupine.delete()
        if self.mcp_client:
            self.mcp_client.stop()


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

    homie = Homie(mode=INTERACTION_MODE)
    homie.start()


if __name__ == "__main__":
    main()
