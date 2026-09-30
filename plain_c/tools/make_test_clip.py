"""
Synthesize a short music-like stereo clip for the end-to-end parity check
(plain_c/separator vs the C++ build/separator). No audio is committed to the repo.

    python3 plain_c/tools/make_test_clip.py out.wav [seconds]

Chords panned left/right, a centred vibrato "voice", noise-burst drums, and a
1 s silent gap at 6 s so the noise gate has something to do. 44.1 kHz 16-bit.
Encode to mp3 with ffmpeg to exercise the decode path as well.
"""
import sys
import wave

import numpy as np


def main():
    dst = sys.argv[1] if len(sys.argv) > 1 else "clip.wav"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    sr = 44100
    n = int(secs * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    x = np.zeros((n, 2))

    for i, (f0, pan) in enumerate([(110, 0.3), (164.8, 0.7), (220, 0.4), (277.2, 0.6)]):  # chords
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t + i)
        tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6)) * env * 0.08
        x[:, 0] += tone * (1 - pan)
        x[:, 1] += tone * pan

    f = 330 + 8 * np.sin(2 * np.pi * 5.5 * t) + 40 * np.sign(np.sin(2 * np.pi * 0.5 * t))  # "voice"
    ph = 2 * np.pi * np.cumsum(f) / sr
    voice = sum(np.sin(k * ph) * np.exp(-(((k * 330 - 700) / 500) ** 2)) for k in range(1, 12)) * 0.15
    voice *= np.sin(2 * np.pi * 0.4 * t) > -0.3
    x += voice[:, None]

    for s0 in range(0, n, sr // 2):  # drums
        L = min(6000, n - s0)
        x[s0:s0 + L] += rng.standard_normal((L, 2)) * np.exp(-np.arange(L) / 800)[:, None] * 0.3

    x[int(6 * sr):int(7 * sr)] = 0  # silent gap
    x = np.clip(x / np.abs(x).max() * 0.8, -1, 1)
    with wave.open(dst, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    print(f"wrote {dst}: {secs} s")


if __name__ == "__main__":
    main()
