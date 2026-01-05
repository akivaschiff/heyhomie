"""Audio processing utilities."""

import io
import struct
import wave


def pcm_to_wav(pcm: list, sample_rate: int = 16000) -> bytes:
    """Convert PCM samples to WAV bytes."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f'{len(pcm)}h', *pcm))
    return buffer.getvalue()
