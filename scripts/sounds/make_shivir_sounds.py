"""Synthesise Shivir's feedback sounds (start, stop, lock).

Original sounds made from scratch (FM bell tones), so we own them outright.
Lengths follow Wispr Flow's measured clip lengths (start 0.15-0.48 s, stop
0.22-0.48 s, lock 0.17-0.95 s; research/ux-spec.md section 8); no Flow audio is
used. Run: python3 scripts/sounds/make_shivir_sounds.py src-tauri/resources
"""

import sys
import wave
from pathlib import Path

import numpy as np

SR = 48_000
PEAK_DBFS = -6.0
E5, B5, E6 = 659.25, 987.77, 1318.51


def bell(freq, dur, decay, index=1.6, ratio=2.0):
    """One soft FM bell: brightness fades faster than loudness, like glass."""
    t = np.arange(int(SR * dur)) / SR
    mod_index = index * np.exp(-t / (decay * 0.35))
    tone = np.sin(2 * np.pi * freq * t + mod_index * np.sin(2 * np.pi * freq * ratio * t))
    attack = np.minimum(1.0, t / 0.003)
    return tone * attack * np.exp(-t / decay)


def phrase(notes, total):
    out = np.zeros(int(SR * total))
    for freq, start, decay, gain in notes:
        tone = bell(freq, total - start, decay) * gain
        i = int(SR * start)
        out[i : i + len(tone)] += tone[: len(out) - i]
    return out


def finish(x):
    # One-pole low-pass (~6 kHz) to round off the FM edge.
    a = 1 - np.exp(-2 * np.pi * 6000 / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += a * (v - acc)
        y[i] = acc
    fade = int(SR * 0.02)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y / np.max(np.abs(y)) * 10 ** (PEAK_DBFS / 20)


SOUNDS = {
    # Rising fifth: "listening".
    "shivir_start": phrase([(E5, 0.0, 0.11, 0.8), (B5, 0.07, 0.13, 1.0)], 0.36),
    # Falling fifth, a little softer: "done".
    "shivir_stop": phrase([(B5, 0.0, 0.10, 0.85), (E5, 0.065, 0.12, 0.8)], 0.34),
    # Quick rising triad that settles: "locked, hands-free".
    "shivir_lock": phrase(
        [(E5, 0.0, 0.07, 0.6), (B5, 0.055, 0.07, 0.7), (E6, 0.11, 0.16, 0.85)], 0.46
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
    for name, samples in SOUNDS.items():
        assert np.max(np.abs(samples)) > 0
        x = finish(samples)
        assert abs(20 * np.log10(np.max(np.abs(x))) - PEAK_DBFS) < 0.1
        write(out_dir / f"{name}.wav", x)
        rms = 20 * np.log10(np.sqrt(np.mean(x**2)))
        print(f"{name}.wav  {len(x) / SR:.2f}s  peak {PEAK_DBFS} dBFS  rms {rms:.1f} dBFS")
