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

import pyaudio


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
FRAMES_PER_BUFFER = 2048

# Device selection (choose one or keep both as None)
# If DEVICE_INDEX is not None, it takes precedence.
DEVICE_INDEX: int | None = None
DEVICE_NAME: str | None = None   # e.g., "pulse"; None keeps default PulseAudio preference

# Optional: write-through to a new WAV file while playing (None to disable)
SAVE_OUT: str | None = None  # e.g., "out.wav"

# Set True to print device list and exit
LIST_DEVICES = False


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
            fmt = None
            if sampwidth == 2:
                fmt = pyaudio.paInt16
            elif sampwidth == 3:
                # 24‑bit packed; many devices accept this
                fmt = pyaudio.paInt24
            elif sampwidth == 4:
                fmt = pyaudio.paInt32
            else:
                return f"Unsupported sample width: {sampwidth*8} bits"

            out_index = select_output_device(p, "pulse", device_index, device_name)

            stream = p.open(format=fmt, channels=channels, rate=TARGET_RATE,
                            output=True, output_device_index=out_index,
                            frames_per_buffer=frames_per_buffer)

            writer = None
            if save_out:
                writer = wave.open(save_out, "wb")
                writer.setnchannels(channels)
                writer.setsampwidth(sampwidth)
                writer.setframerate(rate)

            try:
                while True:
                    data = wf.readframes(frames_per_buffer)
                    if not data:
                        break
                    stream.write(data)
                    if writer:
                        writer.writeframesraw(data)
            finally:
                stream.stop_stream()
                stream.close()
                if writer:
                    writer.close()
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

    
