#!/usr/bin/env python3
"""
frequency_visually.py — Load a WAV file, perform frequency analysis,
and save visualizations (waveform, magnitude spectrum, spectrogram).

Usage examples:
  python frequency_visually.py                   # defaults to samples/sample08.wav
  python frequency_visually.py --path Workshop-samples/samples/sample08.wav
  python frequency_visually.py --outdir analysis_out
"""

import argparse
import os
import sys
import numpy as np
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import get_window, spectrogram


def find_wav_path(user_path: str) -> str:
    """Resolve WAV path, trying common fallbacks if the provided path doesn't exist."""
    candidates = [user_path]
    # Common alternate location in this repo
    if user_path.startswith("samples/"):
        candidates.append(os.path.join("Workshop-samples", user_path))
    # Explicit workshop path first if not already provided
    if user_path == "samples/sample08.wav":
        candidates.insert(0, os.path.join("Workshop-samples", "samples", "sample08.wav"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"WAV not found. Tried: {candidates}")


def to_mono_float(x: np.ndarray) -> np.ndarray:
    """Convert PCM array to mono float32 in [-1, 1]."""
    if x.dtype == np.int16:
        y = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        y = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        y = (x.astype(np.float32) - 128.0) / 128.0
    else:
        y = x.astype(np.float32)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    return y


def plot_waveform(y: np.ndarray, sr: int, out_path: str) -> None:
    t = np.arange(len(y)) / float(sr)
    plt.figure(figsize=(10, 3))
    plt.plot(t, y, linewidth=0.8, color="#2c7fb8")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_spectrum(y: np.ndarray, sr: int, out_path: str, n_fft: int = 16384) -> None:
    n = min(len(y), n_fft)
    # Window the first n samples to get a snapshot spectrum
    seg = y[:n]
    win = get_window("hann", n, fftbins=True).astype(np.float32)
    segw = seg * win
    spec = np.fft.rfft(segw, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    mag = 20.0 * np.log10(np.maximum(1e-10, np.abs(spec)))

    plt.figure(figsize=(10, 3))
    plt.semilogx(freqs, mag, color="#e6550d")
    plt.xlim(20, sr/2)
    plt.ylim(np.max(mag) - 100, np.max(mag) + 5)
    plt.xlabel("Frequency [Hz] (log)")
    plt.ylabel("Magnitude [dB]")
    plt.title("Magnitude Spectrum (Hann window)")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_spectrogram(y: np.ndarray, sr: int, out_path: str, n_fft: int = 2048, hop: int = 512) -> None:
    f, t, Sxx = spectrogram(y, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft-hop,
                            detrend=False, scaling="spectrum", mode="magnitude")
    Sxx_db = 20.0 * np.log10(np.maximum(1e-12, Sxx))
    vmin = np.max(Sxx_db) - 80.0

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma", vmin=vmin, vmax=np.max(Sxx_db))
    plt.ylim(0, sr/2)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title("Spectrogram [dB]")
    cbar = plt.colorbar()
    cbar.set_label("dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Frequency analysis and visualization for a WAV file.")
    ap.add_argument("--path", default="samples/sample08.wav", help="Path to WAV (default: samples/sample08.wav)")
    ap.add_argument("--outdir", default="analysis_output", help="Directory to save plots")
    args = ap.parse_args()

    try:
        wav_path = find_wav_path(args.path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    sr, x = wavfile.read(wav_path)
    y = to_mono_float(x)

    os.makedirs(args.outdir, exist_ok=True)
    wave_png = os.path.join(args.outdir, "waveform.png")
    spec_png = os.path.join(args.outdir, "spectrum.png")
    specgram_png = os.path.join(args.outdir, "spectrogram.png")

    plot_waveform(y, sr, wave_png)
    plot_spectrum(y, sr, spec_png)
    plot_spectrogram(y, sr, specgram_png)

    print("Saved:")
    print(wave_png)
    print(spec_png)
    print(specgram_png)


if __name__ == "__main__":
    main()

