"""Main Homie voice assistant application."""

import io
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytz
from anthropic import Anthropic
from openai import OpenAI

from homie.audio.chimes import generate_chime
from homie.audio.processing import pcm_to_wav
from homie.core.context import ConversationContext
from homie.core.mcp_client import MCPClient
from homie.core.timer import TimerManager
from homie.core.tts import TTSPipeline


def get_system_prompt() -> str:
    """Generate system prompt with current datetime."""
    # Get current time in local system timezone
    current_time = datetime.now().astimezone()
    formatted_time = current_time.strftime("%A, %B %d, %Y at %I:%M %p %Z")

    return f"""You are Homie, a friendly home assistant.

Current date and time: {formatted_time}

You help with:
- Setting timers and reminders (e.g., "remind me in 30 minutes to take out the cake")
- Managing shopping lists and pantry inventory (via Google Sheets)
- Reading and responding to emails (via Gmail)
- Managing chores and tasks
- Answering questions (via web search)
- Checking and managing calendar events

Keep responses concise and conversational - they will be spoken aloud.
Aim for 1-2 sentences when possible.

When setting timers, confirm the duration and message.
Example: "I've set a timer for 30 minutes to remind you to take out the cake."
"""


class Homie:
    """Main voice assistant class. Supports both audio and text modes."""

    def __init__(
        self,
        mode="audio",
        # Configuration parameters
        sample_rate=16000,
        silence_threshold=500,
        silence_duration=1.0,
        min_recording_duration=3.0,
        max_recording_duration=30,
        closing_phrases=None,
        chime_volume=0.2,
        alert_chime_volume=0.6,
        chime_fade_duration=0.02,
        chime_freq_low=523.25,
        chime_freq_high=659.25,
        chime_tone1_duration=0.1,
        chime_tone2_duration=0.15,
        context_timeout=60,
        claude_model="claude-haiku-4-5-20251001",
        claude_max_tokens=1500,
        whisper_model="whisper-1",
        whisper_language="en",
        tts_model="tts-1",
        tts_voice="nova",
        enable_system_mcp=True,
        enable_calendar_mcp=True,
        enable_shopping_mcp=True,
        default_calendar_id="primary",
        pantry_sheet_id="",
        google_service_account_path=None,
        porcupine_access_key=None,
        wake_word_path=None,
        wake_phrase="Hey Homie"
    ):
        self.mode = mode
        self.porcupine = None
        self.recorder = None

        # Store configuration
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_recording_duration = min_recording_duration
        self.max_recording_duration = max_recording_duration
        self.closing_phrases = closing_phrases or ["that's all", "done", "over"]
        self.claude_model = claude_model
        self.claude_max_tokens = claude_max_tokens
        self.whisper_model = whisper_model
        self.whisper_language = whisper_language
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.porcupine_access_key = porcupine_access_key
        self.wake_word_path = wake_word_path
        self.wake_phrase = wake_phrase

        # Initialize clients
        self.openai = OpenAI()
        self.anthropic = Anthropic()
        self.context = ConversationContext(timeout_seconds=context_timeout)
        self.tts_pipeline = TTSPipeline(self.openai, mode=mode, tts_model=tts_model, tts_voice=tts_voice)
        self.mcp_clients = []
        self.timer_manager = TimerManager(self._on_timer_fire)

        # Pre-generate chimes (only used in audio mode)
        if mode == "audio":
            self.listening_chime = generate_chime(
                rising=True,
                freq_low=chime_freq_low,
                freq_high=chime_freq_high,
                tone1_duration=chime_tone1_duration,
                tone2_duration=chime_tone2_duration,
                sample_rate=sample_rate,
                fade_duration=chime_fade_duration,
                volume=chime_volume
            )
            self.processing_chime = generate_chime(
                rising=False,
                freq_low=chime_freq_low,
                freq_high=chime_freq_high,
                tone1_duration=chime_tone1_duration,
                tone2_duration=chime_tone2_duration,
                sample_rate=sample_rate,
                fade_duration=chime_fade_duration,
                volume=chime_volume
            )
            self.alert_chime = generate_chime(
                rising=True,
                freq_low=chime_freq_low,
                freq_high=chime_freq_high,
                tone1_duration=chime_tone1_duration * 2,  # Longer, more drawn out
                tone2_duration=chime_tone2_duration * 2,  # Longer, more drawn out
                sample_rate=sample_rate,
                fade_duration=chime_fade_duration,
                volume=alert_chime_volume
            )

        # Initialize MCP clients
        if enable_system_mcp:
            self._init_system_mcp()
        if enable_calendar_mcp:
            self._init_calendar_mcp(default_calendar_id, google_service_account_path)
        if enable_shopping_mcp:
            self._init_shopping_mcp(pantry_sheet_id, google_service_account_path)

        print(f"✅ Timer tools initialized")

    def _on_timer_fire(self, message: str):
        """Called when a timer fires. Plays alert chime 3 times and speaks the reminder."""
        print(f"\n⏰ Timer fired: {message}")

        # Play alert chime 3 times
        if self.mode == "audio" and hasattr(self, 'alert_chime'):
            for i in range(3):
                self.play_chime(self.alert_chime)
                if i < 2:  # Don't wait after the last chime
                    time.sleep(0.3)

        # Speak the reminder
        reminder_text = f"Reminder: {message}"
        if self.mode == "audio":
            try:
                if os.uname().sysname == "Darwin":
                    subprocess.run(["say", "-v", "Samantha", reminder_text], check=True)
                else:
                    response = self.openai.audio.speech.create(
                        model=self.tts_model,
                        voice=self.tts_voice,
                        input=reminder_text,
                        response_format="mp3"
                    )
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        f.write(response.content)
                        temp_path = f.name
                    subprocess.run(["mpg123", "-q", temp_path], check=True)
                    os.unlink(temp_path)
            except Exception as e:
                print(f"Timer TTS error: {e}")
        else:
            print(f"Homie: {reminder_text}")

    def _get_all_tools(self) -> list[dict]:
        """Get all available tools (MCP + timer tools)."""
        tools = []
        for client in self.mcp_clients:
            tools.extend(client.get_anthropic_tools())
        tools.extend(TimerManager.TOOLS)
        return tools

    def _init_system_mcp(self):
        """Initialize the system MCP server (datetime, volume control)."""
        try:
            # Validate MCP build exists
            mcp_path = Path(__file__).parent.parent.parent / "mcps" / "system"
            mcp_binary = mcp_path / "build" / "index.js"
            if not mcp_binary.exists():
                print(f"⚠️  System MCP disabled: Build not found. Run: cd {mcp_path} && npm run build")
                return

            server_command = ["node", str(mcp_binary)]
            env = {}

            client = MCPClient(server_command, env)
            client.start()
            self.mcp_clients.append(client)
            print(f"✅ System MCP initialized with {len(client.tools)} tools")
        except Exception as e:
            print(f"⚠️  Failed to initialize System MCP: {e}")

    def _init_calendar_mcp(self, default_calendar_id, google_service_account_path):
        """Initialize the calendar MCP server."""
        try:
            # Validate service account file exists
            if not google_service_account_path:
                google_service_account_path = str(Path(__file__).parent.parent.parent / "secrets" / "google-calendar.json")

            creds_path = Path(google_service_account_path)
            if not creds_path.exists():
                print(f"⚠️  Calendar MCP disabled: Credentials not found at {creds_path}")
                return

            # Validate MCP build exists
            mcp_path = Path(__file__).parent.parent.parent / "mcps" / "calendar"
            mcp_binary = mcp_path / "build" / "index.js"
            if not mcp_binary.exists():
                print(f"⚠️  Calendar MCP disabled: Build not found. Run: cd {mcp_path} && npm run build")
                return

            server_command = ["node", str(mcp_binary)]
            env = {
                "GOOGLE_SERVICE_ACCOUNT_PATH": str(creds_path),
                "DEFAULT_CALENDAR_ID": default_calendar_id
            }

            client = MCPClient(server_command, env)
            client.start()
            self.mcp_clients.append(client)
            print(f"✅ Calendar MCP initialized with {len(client.tools)} tools")
            print(f"   Using calendar: {default_calendar_id}")
        except Exception as e:
            print(f"⚠️  Failed to initialize Calendar MCP: {e}")

    def _init_shopping_mcp(self, pantry_sheet_id, google_service_account_path):
        """Initialize the shopping/pantry MCP server."""
        try:
            # Validate pantry sheet ID
            if not pantry_sheet_id:
                print(f"⚠️  Shopping MCP disabled: PANTRY_SHEET_ID not set")
                return

            # Validate service account file exists
            if not google_service_account_path:
                google_service_account_path = str(Path(__file__).parent.parent.parent / "secrets" / "google-calendar.json")

            creds_path = Path(google_service_account_path)
            if not creds_path.exists():
                print(f"⚠️  Shopping MCP disabled: Credentials not found at {creds_path}")
                return

            # Validate MCP build exists
            mcp_path = Path(__file__).parent.parent.parent / "mcps" / "shopping"
            mcp_binary = mcp_path / "build" / "index.js"
            if not mcp_binary.exists():
                print(f"⚠️  Shopping MCP disabled: Build not found. Run: cd {mcp_path} && npm run build")
                return

            server_command = ["node", str(mcp_binary)]
            env = {
                "GOOGLE_SERVICE_ACCOUNT_PATH": str(creds_path),
                "PANTRY_SHEET_ID": pantry_sheet_id
            }

            client = MCPClient(server_command, env)
            client.start()
            self.mcp_clients.append(client)
            print(f"✅ Shopping MCP initialized with {len(client.tools)} tools")
            print(f"   Using sheet: {pantry_sheet_id}")
        except Exception as e:
            print(f"⚠️  Failed to initialize Shopping MCP: {e}")

    def _check_startup_errors(self):
        """Check for recent errors in systemd logs and play error chime if found."""
        try:
            # Get logs from the last boot
            result = subprocess.run(
                ["journalctl", "-u", "homie", "-b", "--priority=err", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                # Count error lines
                error_lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
                error_count = len(error_lines)

                if error_count > 0:
                    print(f"⚠️  Found {error_count} error(s) in startup logs - check journalctl")
                    # Play error chime pattern (3 descending tones) instead of TTS
                    if self.mode == "audio" and hasattr(self, 'processing_chime'):
                        try:
                            for _ in range(3):
                                self.play_chime(self.processing_chime)  # Descending = something's wrong
                                time.sleep(0.2)
                        except Exception as e:
                            print(f"Could not play error chime: {e}")
        except Exception as e:
            # Silently fail if we can't check logs
            pass

    def start(self):
        """Start the assistant in either audio or text mode."""
        if self.mode == "audio":
            self._start_audio_mode()
        else:
            self._start_text_mode()

    def _start_audio_mode(self):
        """Start the voice assistant in audio mode."""
        print("Starting Homie in AUDIO mode...")

        # Set volume to 80% on startup
        try:
            subprocess.run(["amixer", "sset", "Master", "80%"], check=True, capture_output=True)
            print("🔊 Volume set to 80%")
        except Exception as e:
            print(f"⚠️  Could not set volume: {e}")

        # Check for startup errors
        self._check_startup_errors()

        import pvporcupine
        from pvrecorder import PvRecorder

        self.porcupine = pvporcupine.create(
            access_key=self.porcupine_access_key,
            keyword_paths=[self.wake_word_path]
        )

        self.recorder = PvRecorder(
            device_index=-1,
            frame_length=self.porcupine.frame_length
        )

        self.tts_pipeline.start()

        print(f"Audio device: {self.recorder.selected_device}")
        print(f"Listening for '{self.wake_phrase}'...")

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
                    print(f"\nListening for '{self.wake_phrase}'...")

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

    def record_until_silence(self) -> Optional[bytes]:
        """Record audio until silence is detected or closing phrase is heard."""
        frames = []
        silence_frames = 0
        frames_per_second = self.sample_rate // self.porcupine.frame_length
        silence_threshold_frames = int(self.silence_duration * frames_per_second)
        min_frames = int(self.min_recording_duration * frames_per_second)
        max_frames = int(self.max_recording_duration * frames_per_second)

        # Check for closing phrases every 2 seconds
        check_interval_frames = int(2.0 * frames_per_second)
        last_check_frame = 0

        print("   Recording...")
        if self.closing_phrases:
            print(f"   (Say {', '.join(repr(p) for p in self.closing_phrases)} to finish)")

        frame_count = 0
        for _ in range(max_frames):
            pcm = self.recorder.read()
            frames.extend(pcm)
            frame_count += 1

            amplitude = max(abs(s) for s in pcm) if pcm else 0

            if amplitude < self.silence_threshold:
                silence_frames += 1
            else:
                silence_frames = 0

            # Check for closing phrases periodically
            if frame_count >= min_frames and frame_count - last_check_frame >= check_interval_frames:
                last_check_frame = frame_count
                # Quick transcription check
                audio_so_far = pcm_to_wav(frames, self.sample_rate)
                try:
                    audio_file = io.BytesIO(audio_so_far)
                    audio_file.name = "audio.wav"
                    response = self.openai.audio.transcriptions.create(
                        model=self.whisper_model,
                        file=audio_file,
                        language=self.whisper_language
                    )
                    transcript = response.text.strip().lower()

                    # Check if any closing phrase is in the transcript
                    for phrase in self.closing_phrases:
                        if phrase.lower() in transcript:
                            print(f"   Closing phrase '{phrase}' detected!")
                            return audio_so_far
                except Exception as e:
                    # Ignore transcription errors during recording
                    pass

            # Check for silence
            if frame_count >= min_frames and silence_frames >= silence_threshold_frames:
                break

        if len(frames) < self.sample_rate // 2:
            return None

        return pcm_to_wav(frames, self.sample_rate)

    def transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe audio using OpenAI Whisper API."""
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"

            response = self.openai.audio.transcriptions.create(
                model=self.whisper_model,
                file=audio_file,
                language=self.whisper_language
            )
            return response.text.strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def process_and_speak_streaming(self, user_message: str) -> str:
        """Stream Claude response and speak sentences as they complete."""
        self.context.add_message("user", user_message)
        start_time = time.time()

        # Prepare tools from all sources
        tools = self._get_all_tools()

        try:
            # First API call to Claude
            response = self.anthropic.messages.create(
                model=self.claude_model,
                max_tokens=self.claude_max_tokens,
                system=get_system_prompt(),
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
        tool_results = []
        text_parts = []
        timer_tool_names = TimerManager.get_tool_names()

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

                # Check if it's a timer tool
                if tool_name in timer_tool_names:
                    result = self.timer_manager.call_tool(tool_name, tool_input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": json.dumps(result)
                    })
                    continue

                # Otherwise, find the right MCP client that has this tool
                result = None
                for client in self.mcp_clients:
                    tool_names = [t["name"] for t in client.tools]
                    if tool_name in tool_names:
                        result = client.call_tool(tool_name, tool_input)
                        break

                if not result:
                    result = {"error": f"Tool {tool_name} not found"}

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
        tools = self._get_all_tools()

        follow_up = self.anthropic.messages.create(
            model=self.claude_model,
            max_tokens=self.claude_max_tokens,
            system=get_system_prompt(),
            messages=self.context.messages,
            tools=tools if tools else None
        )

        # If Claude wants to use more tools, recurse
        if follow_up.stop_reason == "tool_use":
            return self._handle_tool_use(follow_up, start_time)

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
        self.tts_pipeline = TTSPipeline(self.openai, mode=self.mode, tts_model=self.tts_model, tts_voice=self.tts_voice)
        self.tts_pipeline.start()

    def speak_error(self, text: str):
        """Speak an error message (blocking, used for error handling)."""
        try:
            if os.uname().sysname == "Darwin":
                subprocess.run(["say", "-v", "Samantha", text], check=True)
            else:
                response = self.openai.audio.speech.create(
                    model=self.tts_model,
                    voice=self.tts_voice,
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
        if self.timer_manager:
            self.timer_manager.cancel_all()
        if self.tts_pipeline:
            self.tts_pipeline.stop()
        if self.recorder:
            self.recorder.stop()
            self.recorder.delete()
        if self.porcupine:
            self.porcupine.delete()
        for client in self.mcp_clients:
            client.stop()
