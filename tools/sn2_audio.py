"""Shared audio-monitoring helpers for the Supernova II test scripts.

dBFS math + a time-indexed ring buffer for pulling "what came in on the
audio interface during this MIDI note" out of a live capture stream.
"""

from __future__ import annotations

import threading
import time

import numpy as np


def dbfs(rms: float) -> float:
    """RMS of a [-1, 1]-scaled signal, in dB relative to full scale."""
    return 20.0 * np.log10(max(rms, 1e-12))


def rms_of(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class RingBuffer:
    """Fixed-length multi-channel audio ring buffer indexed by wall-clock time."""

    def __init__(self, seconds: float, samplerate: int, channels: int):
        self.samplerate = samplerate
        self.channels = channels
        self.n = int(seconds * samplerate)
        self.buf = np.zeros((self.n, channels), dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0
        self.t0 = None  # wall-clock time of sample 0, set on first write
        self.lock = threading.Lock()

    def write(self, block: np.ndarray, now: float):
        with self.lock:
            if self.t0 is None:
                self.t0 = now - block.shape[0] / self.samplerate
            n = block.shape[0]
            end = self.write_pos + n
            if end <= self.n:
                self.buf[self.write_pos:end] = block
            else:
                first = self.n - self.write_pos
                self.buf[self.write_pos:] = block[:first]
                self.buf[: end - self.n] = block[first:]
            self.write_pos = end % self.n
            self.total_written += n

    def read_window(self, start_time: float, duration: float) -> np.ndarray | None:
        """Best-effort read of samples covering [start_time, start_time+duration)."""
        with self.lock:
            if self.t0 is None:
                return None
            start_sample = int((start_time - self.t0) * self.samplerate)
            n_samples = int(duration * self.samplerate)
            end_sample = start_sample + n_samples
            oldest_available = max(0, self.total_written - self.n)
            if start_sample < oldest_available or end_sample > self.total_written:
                return None  # window not (fully) captured yet, or already overwritten
            out = np.empty((n_samples, self.channels), dtype=np.float32)
            for i in range(n_samples):
                idx = (start_sample + i) % self.n
                out[i] = self.buf[idx]
            return out


class AudioMonitor:
    def __init__(self, device, samplerate: int, channels: int, buffer_seconds: float = 30.0):
        import sounddevice as sd

        self.sd = sd
        self.samplerate = samplerate
        self.ring = RingBuffer(buffer_seconds, samplerate, channels)
        self.stream = sd.InputStream(
            device=device,
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        self.ring.write(indata.copy(), time.time())

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def measure(self, input_channels: list[int], start_time: float, duration: float,
                settle: float = 0.15, timeout: float = 5.0) -> tuple[float, float] | None:
        """Poll until the window is available, then return (peak_dbfs, rms_dbfs)
        across the given input channels."""
        deadline = time.time() + duration + settle + timeout
        window = None
        while time.time() < deadline:
            window = self.ring.read_window(start_time, duration)
            if window is not None:
                break
            time.sleep(0.05)
        if window is None:
            return None
        sel = window[:, input_channels]
        peak = float(np.max(np.abs(sel))) if sel.size else 0.0
        rms = rms_of(sel)
        return dbfs(peak), dbfs(rms)


def run_selftest() -> bool:
    """Pure-logic checks — no MIDI/audio hardware required."""
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}: {name}")
        ok = ok and cond

    check("silence is very negative dBFS", dbfs(0.0) < -100)
    check("full scale is ~0 dBFS", abs(dbfs(1.0)) < 1e-6)
    check("half amplitude is about -6dBFS", abs(dbfs(0.5) - (-6.02)) < 0.1)

    sr = 1000
    rb = RingBuffer(seconds=2.0, samplerate=sr, channels=2)
    t_start = 100.0
    tone = np.stack(
        [np.sin(2 * np.pi * 50 * np.arange(sr) / sr), np.zeros(sr)], axis=1
    ).astype(np.float32)

    window = rb.read_window(t_start, 1.0)
    check("window before any data returns None", window is None)

    # callback fires once the block finishes capturing, so "now" marks the
    # END of the block: a 1s block handed in at t_start+1.0 covers
    # [t_start, t_start+1.0).
    rb.write(tone, t_start + 1.0)
    window = rb.read_window(t_start, 1.0)
    check("window matches written block shape", window is not None and window.shape == tone.shape)
    if window is not None:
        check("recovered samples match written tone", np.allclose(window, tone, atol=1e-6))
        tone_peak = dbfs(float(np.max(np.abs(window[:, 0]))))
        silent_rms = dbfs(rms_of(window[:, 1]))
        check("channel 0 (tone) peak near 0dBFS", tone_peak > -1.0)
        check("channel 1 (silence) rms very low", silent_rms < -100)

    rb2 = RingBuffer(seconds=1.0, samplerate=sr, channels=1)
    block = np.ones((sr, 1), dtype=np.float32)
    rb2.write(block, 200.0)   # covers [199, 200)
    rb2.write(block, 201.0)   # covers [200, 201) -> overwrites first block
    stale = rb2.read_window(199.0, 1.0)
    check("overwritten window is reported unavailable", stale is None)
    fresh = rb2.read_window(200.0, 1.0)
    check("most recent window is still available", fresh is not None)

    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_selftest() else 1)
