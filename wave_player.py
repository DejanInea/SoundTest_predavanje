#!/usr/bin/env python3
"""
wave_player.py — Minimal streaming WAV player for Linux (PulseAudio via PyAudio).

Features:
- Streams 16‑bit PCM WAV without loading whole file into memory
- Device selection by index or name substring (defaults to PulseAudio)
- Optional simultaneous write‑through to a new WAV while playing

Usage:
  - Adjust the constants below (WAV_PATH, FRAMES_PER_BUFFER, DEVICE_INDEX, DEVICE_NAME, SAVE_OUT, LIST_DEVICES)
  - Run: python wave_player.py
"""

import os
import sys
import wave
import shutil

import pyaudio
from scipy.fft import dct, idct


class capture_stderr:
    """Temporarily silence C‑level stderr (ALSA/PortAudio chatter)."""
    def __enter__(self):
        import tempfile, os as _os
        self._old = _os.dup(2)
        self._tmp = tempfile.TemporaryFile()
        _os.dup2(self._tmp.fileno(), 2)
        return self
    def __exit__(self, exc_type, exc, tb):
        import os as _os
        _os.dup2(self._old, 2)
        _os.close(self._old)
        self._tmp.seek(0)
        self.data = self._tmp.read().decode(errors="ignore")
        self._tmp.close()


# -------------------------- Configuration constants --------------------------
# Path to WAV to play (default points to workshop sample)
WAV_PATH = os.path.join("Workshop-samples", "samples", "sample08.wav")

# Target sample rate (Hz) — enforce 44.1 kHz playback files
TARGET_RATE = 44100

# Frames per buffer (stream chunk size)
FRAMES_PER_BUFFER = 1024  # zahtevano: branje v blokih po 1024 vzorcev (frames)

# Device selection (choose one or keep both as None)
# If DEVICE_INDEX is not None, it takes precedence.
DEVICE_INDEX: int | None = None
DEVICE_NAME: str | None = None   # e.g., "pulse"; None keeps default PulseAudio preference

# Optional: write-through to a new WAV file while playing (None to disable)
SAVE_OUT: str | None = None  # e.g., "out.wav"

# Set True to print device list and exit
LIST_DEVICES = False

# Progress-bar visuals and spectral entropy option
BAR_WIDTH = 28
PROGRESS_BAR_WIDTH = 36  # overall playback progress bar width
ENTROPY_TRACK_WIDTH = 40  # width for entropy timeline bars
ENTROPY_CHARSET = " .:-=+*#%@"  # low->high density symbols
ENABLE_SPECTRAL_ENTROPY = True  # requires NumPy


def list_output_devices(pa: pyaudio.PyAudio):
    lines = []
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d.get("maxOutputChannels", 0) > 0:
            api = pa.get_host_api_info_by_index(d["hostApi"])["name"]
            lines.append(f"[{i}] {d['name']}  (host API: {api})")
    return lines


def select_output_device(pa: pyaudio.PyAudio, prefer_name_substr: str = "pulse",
                         explicit_index: int | None = None,
                         explicit_name: str | None = None) -> int:
    # index beats everything
    if explicit_index is not None:
        info = pa.get_device_info_by_index(int(explicit_index))
        if info.get("maxOutputChannels", 0) > 0:
            return int(explicit_index)
        raise RuntimeError(f"Device index {explicit_index} is not an output device.")

    def find_by_name(substr: str):
        s = substr.lower()
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d.get("maxOutputChannels", 0) > 0 and s in d.get("name", "").lower():
                return i
        return None

    if explicit_name:
        idx = find_by_name(explicit_name)
        if idx is not None:
            return idx
    if prefer_name_substr:
        idx = find_by_name(prefer_name_substr)
        if idx is not None:
            return idx
    try:
        return pa.get_default_output_device_info().get("index")
    except Exception:
        pass
    raise RuntimeError("No usable PyAudio output device found.")


def stream_wav(path: str, frames_per_buffer: int = 2048,
               device_index: int | None = None,
               device_name: str | None = None,
               save_out: str | None = None) -> str:
    """Stream a WAV file to PulseAudio using PyAudio. Returns empty string on success, otherwise error text."""
    if not os.path.isfile(path):
        return f"File not found: {path}"

    with wave.open(path, "rb") as wf, capture_stderr() as cap:
        p = pyaudio.PyAudio()
        try:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()  # bytes per sample
            rate = wf.getframerate()

            if rate != TARGET_RATE:
                return f"Unsupported sample rate: {rate} Hz (expected {TARGET_RATE} Hz)."
            # We will quantize to 16-bit PCM for intermediate processing and playback
            fmt = pyaudio.paInt16
            if sampwidth not in (2, 3, 4):
                return f"Unsupported sample width: {sampwidth*8} bits"

            out_index = select_output_device(p, "pulse", device_index, device_name)

            stream = p.open(format=fmt, channels=channels, rate=TARGET_RATE,
                            output=True, output_device_index=out_index,
                            frames_per_buffer=frames_per_buffer)

            writer = None
            if save_out:
                writer = wave.open(save_out, "wb")
                writer.setnchannels(channels)
                writer.setsampwidth(2)  # 16-bit intermediate storage
                writer.setframerate(rate)

            total_frames = wf.getnframes()
            played_frames = 0
            # entropy timelines (append one char per processed block)
            timeline_val = ""
            timeline_spec = ""

            # Lazy import numpy for faster entropy; fall back to stdlib if missing
            try:
                import numpy as _np
                _USE_NP = True
            except Exception:
                from collections import Counter as _Counter
                _USE_NP = False

            def entropy_bits_from_block(raw16: bytes) -> float:
                """Compute Shannon entropy (bits) over sample values of the block.
                Treats each 16-bit sample (all channels) as a symbol.
                Max entropy is log2(N) where N = number of samples in the block.
                """
                if _USE_NP:
                    arr = _np.frombuffer(raw16, dtype='<i2')
                    if arr.size == 0:
                        return 0.0
                    vals, counts = _np.unique(arr, return_counts=True)
                    p = counts.astype(_np.float64) / arr.size
                    H = -_np.sum(p * _np.log2(p, where=p>0, out=_np.zeros_like(p)))
                    return float(H)
                else:
                    # Fallback: Python unpack + Counter
                    import struct
                    n = len(raw16) // 2
                    if n == 0:
                        return 0.0
                    it = struct.iter_unpack('<h', raw16)
                    counts = {}
                    for (v,) in it:
                        counts[v] = counts.get(v, 0) + 1
                    total = sum(counts.values())
                    from math import log2
                    H = -sum((c/total)*log2(c/total) for c in counts.values())
                    return H

            def _decode_block_to_mono_float(raw16: bytes):
                """Decode raw PCM block to mono float32 in [-1,1]. Uses NumPy if available."""
                if not _USE_NP:
                    return None
                arr = _np.frombuffer(raw16, dtype='<i2')
                if arr.size == 0:
                    return _np.zeros(0, dtype=_np.float32)
                if channels > 1:
                    arr = arr.reshape(-1, channels).mean(axis=1)
                return arr.astype(_np.float32) / 32768.0

            def spectral_entropy_bits_from_block(raw16: bytes) -> float:
                if not (_USE_NP and ENABLE_SPECTRAL_ENTROPY):
                    return 0.0
                x = _decode_block_to_mono_float(raw16)
                if x is None or x.size == 0:
                    return 0.0
                win = _np.hanning(x.size).astype(_np.float32)
                X = _np.fft.rfft(x * win)
                P = (X.real**2 + X.imag**2)
                Ps = P.sum()
                if Ps <= 0.0:
                    return 0.0
                p = P / Ps
                H = -_np.sum(_np.where(p > 0, p * _np.log2(p), 0.0))
                return float(H)

            def spectral_entropy_from_dct_int(coeffs_i16) -> float:
                """Spectral entropy over DCT coefficient energy distribution (per block)."""
                if not (_USE_NP and ENABLE_SPECTRAL_ENTROPY):
                    return 0.0
                if coeffs_i16.size == 0:
                    return 0.0
                c = coeffs_i16.astype(_np.float64)
                if c.ndim == 1:
                    E = c * c
                else:
                    E = _np.sum(c * c, axis=1)  # sum over channels
                Etot = float(_np.sum(E))
                if Etot <= 0.0:
                    return 0.0
                p = E / Etot
                H = -_np.sum(_np.where(p > 0, p * _np.log2(p), 0.0))
                return float(H)

            def _fmt_mmmss(sec: float) -> str:
                sec = max(0.0, float(sec))
                m = int(sec // 60)
                s = int(sec % 60)
                return f"{m:03d}:{s:02d}"

            last_print_len = [0]  # mutable holder to track previous line length

            def draw_entropy_bars(H_val: float, H_spec: float,
                                  progress: float | None = None,
                                  elapsed_sec: float | None = None,
                                  total_sec: float | None = None,
                                  tline_val: str | None = None,
                                  tline_spec: str | None = None) -> None:
                from math import log2
                Hval_max = log2(max(2, frames_per_buffer * channels))
                Nbins = (frames_per_buffer // 2) + 1
                Hspec_max = log2(max(2, Nbins))
                def bar(h, hmax):
                    h = max(0.0, min(h, hmax))
                    fill = int((h / hmax) * BAR_WIDTH) if hmax > 0 else 0
                    return '[' + '#' * fill + '-' * (BAR_WIDTH - fill) + ']'
                # Generate fixed-width, multi-line UI and update in-place if TTY
                is_tty = sys.stdout.isatty()
                t_val = (tline_val or "").ljust(ENTROPY_TRACK_WIDTH)
                t_spec = (tline_spec or "").ljust(ENTROPY_TRACK_WIDTH)
                # Progress bar parts
                if progress is not None:
                    pfill = int(max(0.0, min(1.0, progress)) * PROGRESS_BAR_WIDTH)
                    pbar = '[' + '=' * pfill + '-' * (PROGRESS_BAR_WIDTH - pfill) + ']'
                    pct = f" {progress*100:6.1f}%"
                else:
                    pbar = '[' + '-' * PROGRESS_BAR_WIDTH + ']'
                    pct = "       "
                if elapsed_sec is not None and total_sec is not None:
                    tstamp = f" {_fmt_mmmss(elapsed_sec)}/{_fmt_mmmss(total_sec)}"
                else:
                    tstamp = " 000:00/000:00"

                line1 = f"Val {bar(H_val, Hval_max)} {H_val:6.2f}b  Max~{Hval_max:4.1f}b"
                line2 = f"Spec{bar(H_spec, Hspec_max)} {H_spec:6.2f}b  Max~{Hspec_max:4.1f}b"
                line3 = f"Prog {pbar}{pct}{tstamp}"
                line4 = f"EVal [{t_val}]"
                line5 = f"ESpc [{t_spec}]"

                if is_tty:
                    cols = shutil.get_terminal_size(fallback=(120, 24)).columns
                    # Ensure we don't wrap: clip to terminal width
                    line1_c = line1[:cols]
                    line2_c = line2[:cols]
                    line3_c = line3[:cols]
                    line4_c = line4[:cols]
                    line5_c = line5[:cols]
                    # Move cursor up to rewrite the 5 UI lines after the first print
                    if last_print_len[0] > 0:
                        sys.stdout.write(f"\x1b[5F")  # cursor 5 lines up
                    # Clear and print each line
                    for ln in (line1_c, line2_c, line3_c, line4_c, line5_c):
                        sys.stdout.write("\x1b[2K" + ln + "\n")  # clear line, then write and newline
                    last_print_len[0] = 1  # mark as initialized
                    sys.stdout.flush()
                else:
                    # Fallback: single-line carriage return
                    msg = line1 + "  " + line2 + "  " + line3
                    pad = max(0, last_print_len[0] - len(msg))
                    last_print_len[0] = max(last_print_len[0], len(msg))
                    sys.stdout.write('\r' + msg + (' ' * pad))
                    sys.stdout.flush()

            # ------------------------ decoding + quantization helpers ------------------------
            def _decode_any_to_int16(raw: bytes) -> bytes:
                """Decode source block (16/24/32-bit) to little-endian int16 bytes."""
                if _USE_NP:
                    if sampwidth == 2:
                        # Already int16
                        return raw
                    elif sampwidth == 3:
                        b = _np.frombuffer(raw, dtype=_np.uint8)
                        if b.size == 0:
                            return b.tobytes()
                        trip = b.reshape(-1, 3).astype(_np.uint32)
                        val = (trip[:, 0] | (trip[:, 1] << 8) | (trip[:, 2] << 16)).astype(_np.int32)
                        neg = (trip[:, 2] & 0x80) != 0
                        val[neg] |= _np.int32(-1 << 24)
                        # scale 24->16
                        val16 = _np.clip(_np.round(val / 256.0), -32768, 32767).astype('<i2')
                        return val16.tobytes()
                    else:  # 32-bit
                        val = _np.frombuffer(raw, dtype='<i4')
                        if val.size == 0:
                            return b""
                        val16 = _np.clip(_np.round(val / 65536.0), -32768, 32767).astype('<i2')
                        return val16.tobytes()
                else:
                    import struct
                    if sampwidth == 2:
                        return raw
                    elif sampwidth == 3:
                        it = struct.iter_unpack('<3s', raw)
                        out = bytearray()
                        for (bs,) in it:
                            b0, b1, b2 = bs[0], bs[1], bs[2]
                            v = b0 | (b1 << 8) | (b2 << 16)
                            if b2 & 0x80:
                                v |= -1 << 24
                            # scale to 16-bit
                            v16 = max(-32768, min(32767, int(round(v / 256.0))))
                            out += struct.pack('<h', v16)
                        return bytes(out)
                    else:
                        it = struct.iter_unpack('<i', raw)
                        out = bytearray()
                        for (v,) in it:
                            v16 = max(-32768, min(32767, int(round(v / 65536.0))))
                            out += struct.pack('<h', v16)
                        return bytes(out)

            # DCT encode/decode helpers
            def _dct_encode_int16_block(block16: bytes):
                """Return (payload_i16_bytes, coeffs_i16_array, scales_per_channel)."""
                if not _USE_NP:
                    return block16, None, None
                arr = _np.frombuffer(block16, dtype='<i2')
                if arr.size == 0:
                    return b"", _np.zeros((0, channels), dtype='<i2'), _np.ones(channels, dtype=_np.float64)
                frames = arr.size // channels
                x = (arr.reshape(frames, channels).astype(_np.float64)) / 32768.0
                C = dct(x, type=2, norm='ortho', axis=0)
                scales = _np.maximum(1e-12, _np.max(_np.abs(C), axis=0))
                Cn = C / scales  # [-1,1]
                Ci16 = _np.clip(_np.round(Cn * 32767.0), -32768, 32767).astype('<i2')
                return Ci16.tobytes(), Ci16, scales

            def _dct_decode_int16_block(coeffs_i16, scales):
                if not _USE_NP:
                    return coeffs_i16  # passthrough
                if coeffs_i16 is None or coeffs_i16.size == 0:
                    return b""
                frames = coeffs_i16.shape[0]
                C_rec = coeffs_i16.astype(_np.float64) / 32767.0
                C_rec = C_rec * scales  # broadcast per-channel
                x_rec = idct(C_rec, type=2, norm='ortho', axis=0)
                y = _np.clip(_np.round(x_rec * 32767.0), -32768, 32767).astype('<i2')
                return y.tobytes()

            try:
                while True:
                    data = wf.readframes(frames_per_buffer)
                    if not data:
                        break
                    # Decode to int16 -> DCT encode (normalize to 16-bit coeffs) -> IDCT decode
                    block16 = _decode_any_to_int16(data)
                    payload_bytes, coeffs_i16, scales = _dct_encode_int16_block(block16)
                    # Decoder uses 'scales' to reverse normalization
                    out_bytes = _dct_decode_int16_block(coeffs_i16, scales)
                    stream.write(out_bytes)
                    if writer:
                        writer.writeframesraw(out_bytes)
                    played_frames += len(data) // (channels * sampwidth)
                    # compute and draw entropy bars per block
                    H_val = entropy_bits_from_block(payload_bytes)
                    # Spectral entropy from DCT energy distribution
                    if coeffs_i16 is not None:
                        coeffs_for_spec = coeffs_i16
                    else:
                        coeffs_for_spec = None
                    H_spec = spectral_entropy_from_dct_int(coeffs_for_spec) 
                    prog = min(1.0, played_frames / total_frames) if total_frames else None
                    elapsed = played_frames / rate if rate else 0.0
                    total = total_frames / rate if rate else 0.0
                    # Update entropy timeline characters
                    # map entropy to a density character
                    def e_char(h: float, hmax: float) -> str:
                        if hmax <= 0:
                            return ' '
                        n = max(0.0, min(1.0, h / hmax))
                        idx = int(round(n * (len(ENTROPY_CHARSET) - 1)))
                        return ENTROPY_CHARSET[idx]
                    from math import log2
                    Hval_max = log2(max(2, frames_per_buffer * channels))
                    Nbins = (frames_per_buffer // 2) + 1
                    Hspec_max = log2(max(2, Nbins))
                    timeline_val = (timeline_val + e_char(H_val, Hval_max))[-ENTROPY_TRACK_WIDTH:]
                    timeline_spec = (timeline_spec + e_char(H_spec, Hspec_max))[-ENTROPY_TRACK_WIDTH:]
                    draw_entropy_bars(H_val, H_spec, progress=prog, elapsed_sec=elapsed, total_sec=total,
                                      tline_val=timeline_val, tline_spec=timeline_spec)
            finally:
                stream.stop_stream()
                stream.close()
                if writer:
                    writer.close()
            # Newline after progress bar
            sys.stdout.write('\n')
            return ""
        except Exception as e:
            return f"{e}\n{cap.data}"
        finally:
            p.terminate()


def main():
    if LIST_DEVICES:
        with capture_stderr():
            pa = pyaudio.PyAudio()
            try:
                for line in list_output_devices(pa):
                    print(line)
            finally:
                pa.terminate()
        return

    err = stream_wav(WAV_PATH, frames_per_buffer=FRAMES_PER_BUFFER,
                     device_index=DEVICE_INDEX, device_name=DEVICE_NAME,
                     save_out=SAVE_OUT)
    if err:
        sys.stderr.write(err)
        sys.exit(1)


if __name__ == "__main__":
    main()

    
