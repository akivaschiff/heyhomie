"""Voice endpointing with a scripted fake recorder — no device, no network.
Covers the paths a live mic exercises: hesitation before speaking, speech then
silence, and never speaking at all."""

from homie.channels.voice import SAMPLE_RATE, VoiceChannel


class FakeRecorder:
    """Yields scripted per-frame peak levels as flat frames."""

    def __init__(self, levels):
        self.levels = list(levels)

    def read(self):
        level = self.levels.pop(0) if self.levels else 0
        return [level] * 512


def make_channel(levels, threshold=1000):
    ch = VoiceChannel.__new__(VoiceChannel)
    ch.frame_length = 512
    ch.silence_threshold = threshold
    ch.recorder = FakeRecorder(levels)
    return ch


FPS = SAMPLE_RATE // 512


def test_hesitation_then_speech_is_captured():
    # 2s quiet (user thinking), 2s speech, then silence
    levels = [100] * (2 * FPS) + [8000] * (2 * FPS) + [100] * (3 * FPS)
    wav = make_channel(levels)._record_turn()
    assert wav, "turn should capture speech that starts after a pause"


def test_no_speech_times_out_empty():
    levels = [100] * (10 * FPS)
    wav = make_channel(levels)._record_turn()
    assert wav == b""


def test_speech_then_silence_ends_promptly():
    ch = make_channel([8000] * (1 * FPS) + [100] * (20 * FPS))
    wav = ch._record_turn()
    assert wav
    # should stop ~1.2s after speech ends, not consume the whole script
    assert len(ch.recorder.levels) > 15 * FPS


def test_retune_threshold_tracks_noise_floor():
    ch = make_channel([])
    ch._retune_threshold([300] * 100)
    quiet = ch.silence_threshold
    ch._retune_threshold([3000] * 100)  # vacuum cleaner
    loud = ch.silence_threshold
    assert quiet < loud <= 9000
    assert quiet == 750
