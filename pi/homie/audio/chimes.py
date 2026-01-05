"""Audio chime generation for acknowledgment sounds."""

import io
import wave

import numpy as np


def generate_tone(
    frequency: float,
    duration: float,
    sample_rate: int = 16000,
    fade_duration: float = 0.02,
    volume: float = 0.2
) -> bytes:
    """Generate a pleasant tone as WAV bytes."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)

    # Add fade in/out for a softer sound
    fade_samples = int(sample_rate * fade_duration)
    tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
    tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    # Convert to 16-bit PCM with configured volume
    tone = (tone * 32767 * volume).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(tone.tobytes())
    return buffer.getvalue()


def generate_chime(
    rising: bool = True,
    freq_low: float = 523.25,
    freq_high: float = 659.25,
    tone1_duration: float = 0.1,
    tone2_duration: float = 0.15,
    sample_rate: int = 16000,
    fade_duration: float = 0.02,
    volume: float = 0.2
) -> bytes:
    """Generate a pleasant two-tone chime."""
    if rising:
        tone1 = generate_tone(freq_low, tone1_duration, sample_rate, fade_duration, volume)
        tone2 = generate_tone(freq_high, tone2_duration, sample_rate, fade_duration, volume)
    else:
        tone1 = generate_tone(freq_high, tone1_duration, sample_rate, fade_duration, volume)
        tone2 = generate_tone(freq_low, tone2_duration, sample_rate, fade_duration, volume)

    buffer1 = io.BytesIO(tone1)
    buffer2 = io.BytesIO(tone2)

    with wave.open(buffer1, 'rb') as w1, wave.open(buffer2, 'rb') as w2:
        frames1 = w1.readframes(w1.getnframes())
        frames2 = w2.readframes(w2.getnframes())

    combined = io.BytesIO()
    with wave.open(combined, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames1 + frames2)

    return combined.getvalue()
