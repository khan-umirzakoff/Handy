"""Synthesise Shivir's feedback sounds (start, stop, lock).

Original sounds made from scratch (soft mallet tones), so we own them outright. The
target envelope comes from measuring Wispr Flow's default pack (research/pill-smoothness.md
section 5): 0.17-0.22 s long, energy at 250-450 Hz, RMS about -35 dBFS (lock -40),
peak about -16 dBFS (lock -21). We level by RMS, which is what the ear follows. No Flow
audio is used. The first pack (FM bells at 660-1320 Hz, -6 dBFS peak) read as thin and
chirpy next to it.
Run: python3 scripts/sounds/make_shivir_sounds.py src-tauri/resources
"""

import sys
import wave
from pathlib import Path

import numpy as np

SR = 48_000
A3, D4, A4, D3 = 220.0, 293.66, 440.0, 146.83


def mallet(freq, dur, decay, attack=0.012):
    """A soft wooden tap: the fundamental plus two quiet partials that die out first."""
    t = np.arange(int(SR * dur)) / SR
    tone = (
        np.sin(2 * np.pi * freq * t)
        + 0.22 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-t / (decay * 0.5))
        + 0.05 * np.sin(2 * np.pi * 3 * freq * t) * np.exp(-t / (decay * 0.3))
    )
    rise = np.where(t < attack, 0.5 - 0.5 * np.cos(np.pi * t / attack), 1.0)
    return tone * rise * np.exp(-t / decay)


def phrase(notes, total):
    out = np.zeros(int(SR * total))
    for freq, start, decay, gain in notes:
        tone = mallet(freq, total - start, decay) * gain
        i = int(SR * start)
        out[i : i + len(tone)] += tone[: len(out) - i]
    return out


def finish(x, rms_dbfs):
    # One-pole low-pass (~2.5 kHz) keeps it round; a 20 ms fade ends it cleanly.
    a = 1 - np.exp(-2 * np.pi * 2500 / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += a * (v - acc)
        y[i] = acc
    fade = int(SR * 0.02)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y / np.sqrt(np.mean(y**2)) * 10 ** (rms_dbfs / 20)


SOUNDS = {
    # Rising fourth, D4 to A4: "listening".
    "shivir_start": (phrase([(D4, 0.0, 0.032, 0.8), (A4, 0.055, 0.038, 1.0)], 0.20), -34.0),
    # Falling, A4 to D4, a touch lower and softer: "done".
    "shivir_stop": (phrase([(A4, 0.0, 0.032, 0.75), (D4, 0.06, 0.045, 1.0)], 0.22), -35.0),
    # Two low taps, D3 + A3 then A3: "locked, hands-free".
    "shivir_lock": (
        phrase([(D3, 0.0, 0.03, 0.7), (A3, 0.0, 0.03, 0.6), (A3, 0.07, 0.035, 0.8)], 0.18),
        -39.0,
    ),
}


def write(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())


if __name__ == "__main__":
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for name, (samples, rms) in SOUNDS.items():
        assert np.max(np.abs(samples)) > 0
        x = finish(samples, rms)
        peak = 20 * np.log10(np.max(np.abs(x)))
        assert peak < -6, f"{name} would clip"
        write(out_dir / f"{name}.wav", x)
        print(f"{name}.wav  {len(x) / SR:.2f}s  peak {peak:.1f} dBFS  rms {rms} dBFS")
