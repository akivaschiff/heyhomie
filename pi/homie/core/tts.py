"""Text-to-speech pipeline with async audio generation and playback."""

import os
import queue
import subprocess
import tempfile
import threading
import time


class TTSPipeline:
    """Async TTS pipeline - generates and plays audio without blocking Claude stream.
    In text mode, this just prints to stdout instead of generating audio."""

    def __init__(self, openai_client, mode="audio", tts_model="tts-1", tts_voice="nova"):
        self.openai = openai_client
        self.mode = mode
        self.tts_model = tts_model
        self.tts_voice = tts_voice
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
                        model=self.tts_model,
                        voice=self.tts_voice,
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
