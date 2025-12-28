import asyncio
import json
import base64
import struct
import time
import subprocess
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta

import websockets
from faster_whisper import WhisperModel
import numpy as np
from anthropic import Anthropic

# Config
HOST = "0.0.0.0"
PORT = 8765
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 500  # Amplitude threshold for silence detection
SILENCE_DURATION = 1.5   # Seconds of silence to end utterance
CONTEXT_TIMEOUT = 60     # Seconds before context resets

# Claude system prompt
SYSTEM_PROMPT = """You are Homie, a friendly home assistant. You help with:
- Managing shopping lists and pantry inventory (via Google Sheets)
- Reading and responding to emails (via Gmail)
- Managing chores and tasks
- Answering questions (via web search)

Keep responses concise and conversational - they will be spoken aloud.

When asked to perform an action, confirm what you're about to do and wait for confirmation before executing.
For example: "I'll add hummus to the shopping list. Should I do that?"

After the user confirms with "yes", "yeah", "do it", "go ahead", etc., execute the action.
If they say "no", "cancel", "never mind", etc., acknowledge and don't execute.
"""


class ConversationContext:
    def __init__(self, timeout_seconds: int = CONTEXT_TIMEOUT):
        self.messages = []
        self.last_interaction = None
        self.timeout = timedelta(seconds=timeout_seconds)
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.last_interaction = datetime.now()
    
    def get_messages(self):
        # Check if context expired
        if self.last_interaction and datetime.now() - self.last_interaction > self.timeout:
            self.messages = []
        return self.messages
    
    def clear(self):
        self.messages = []
        self.last_interaction = None


class HomieServer:
    def __init__(self):
        print("Loading Whisper model...")
        self.whisper = WhisperModel("base", device="cpu", compute_type="int8")
        print("Whisper loaded.")
        
        self.anthropic = Anthropic()
        self.context = ConversationContext()
        # TODO: Initialize MCP client here
        
    async def handle_connection(self, websocket):
        print(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    data = json.loads(message)
                    if data["type"] == "start_listening":
                        await self.handle_audio_stream(websocket)
                else:
                    # Raw audio data (shouldn't happen here, handled in handle_audio_stream)
                    pass
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
    
    async def handle_audio_stream(self, websocket):
        """Receive audio, transcribe, process with Claude, respond."""
        print("Listening for speech...")
        
        audio_buffer = []
        silence_frames = 0
        frames_per_second = SAMPLE_RATE // 512  # Approximate frames per second
        silence_frames_threshold = int(SILENCE_DURATION * frames_per_second)
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Decode audio frame
                    frame = struct.unpack(f'{len(message)//2}h', message)
                    audio_buffer.extend(frame)
                    
                    # Check for silence
                    amplitude = max(abs(s) for s in frame) if frame else 0
                    
                    if amplitude < SILENCE_THRESHOLD:
                        silence_frames += 1
                    else:
                        silence_frames = 0
                    
                    # End of utterance detected
                    if silence_frames >= silence_frames_threshold and len(audio_buffer) > SAMPLE_RATE:
                        print("End of utterance detected")
                        await websocket.send(json.dumps({"type": "utterance_complete"}))
                        break
                else:
                    # Control message
                    data = json.loads(message)
                    if data.get("type") == "cancel":
                        return
            
            if not audio_buffer:
                return
            
            # Transcribe
            print("Transcribing...")
            transcript = await self.transcribe(audio_buffer)
            print(f"Transcript: {transcript}")
            
            if not transcript.strip():
                return
            
            # Process with Claude
            print("Processing with Claude...")
            response = await self.process_with_claude(transcript)
            print(f"Response: {response}")
            
            # Convert to speech
            print("Generating speech...")
            audio_data = await self.text_to_speech(response)
            
            # Send audio back
            await websocket.send(json.dumps({
                "type": "audio_response",
                "audio": base64.b64encode(audio_data).decode()
            }))
            
        except Exception as e:
            print(f"Error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": str(e)
            }))
    
    async def transcribe(self, audio: list) -> str:
        """Transcribe audio using Whisper."""
        # Convert to numpy array
        audio_np = np.array(audio, dtype=np.float32) / 32768.0
        
        # Transcribe
        segments, info = self.whisper.transcribe(audio_np, beam_size=5)
        
        transcript = " ".join(segment.text for segment in segments)
        return transcript.strip()
    
    async def process_with_claude(self, user_message: str) -> str:
        """Send message to Claude and get response."""
        # Add user message to context
        self.context.add_message("user", user_message)
        
        # TODO: Add MCP tool use here
        response = self.anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=self.context.get_messages()
        )
        
        assistant_message = response.content[0].text
        self.context.add_message("assistant", assistant_message)
        
        return assistant_message
    
    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech using macOS say command."""
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            temp_aiff = f.name
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
        
        try:
            # Generate speech with macOS say
            subprocess.run([
                "say", "-v", "Samantha", "-o", temp_aiff, text
            ], check=True)
            
            # Convert to WAV (Pi needs WAV)
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_aiff, "-ar", "16000", "-ac", "1", temp_wav
            ], check=True, capture_output=True)
            
            with open(temp_wav, "rb") as f:
                return f.read()
        finally:
            os.unlink(temp_aiff)
            os.unlink(temp_wav)


async def main():
    server = HomieServer()
    
    print(f"Starting Homie server on ws://{HOST}:{PORT}")
    async with websockets.serve(server.handle_connection, HOST, PORT):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
