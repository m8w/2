#!/usr/bin/env python3
"""Sweep every Program in a Novation Supernova II library and flag silent patches.

Intended use: load a Performance (e.g. C000) whose Parts are set to distinct,
non-Global MIDI channels ("lanes"). Each lane independently steps through a
range of Program banks/numbers on its own MIDI channel — Bank Select LSB
(CC32) + Program Change, per the Supernova II MIDI implementation — holds a
test chord, and records+analyzes the audio arriving on that lane's assigned
input channel(s) of a USB audio interface. Anything below the silence
threshold gets flagged in the CSV report as a patch that produced no audible
output.

Setup required on the Supernova II before running:
  - Global MIDI channel set to a channel NOT used by any lane below (e.g. 16).
  - Each Part under test given its own MIDI channel (Part Edit > MIDI menu),
    matching --lane's CH value, and NOT set to "Global" or "Omni".
  - Each Part's Program Change filter enabled (Global Menu, not Disabled).
  - For true parallel lanes, route each Part to its own output pair
    (Part Edit > Output > "Part outputs") and wire each pair into a separate
    input channel on the interface, matching --lane's INPUTS value. Running a
    single lane at a time on a shared stereo bus also works and needs no
    special output routing.

Example — two lanes (Parts on MIDI ch 1 and 2, routed to interface inputs
0/1 and 2/3), sweeping Program banks A-D, one program every 20s, second lane
starting 20s after the first:

    python3 supernova_patch_sweep.py \\
        --list-devices                       # find exact names first
    python3 supernova_patch_sweep.py \\
        --midi-port "Supernova" --audio-device "Scarlett" \\
        --lane 1:0,1:A,B,C,D \\
        --lane 2:2,3:A,B,C,D \\
        --stagger 20 --step 20

Dependencies (install on the machine actually connected to the hardware):
    pip install mido python-rtmidi sounddevice numpy
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

BANK_LSB = {"A": 4, "B": 5, "C": 6, "D": 7, "E": 8, "F": 9, "G": 10, "H": 11}


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


@dataclass
class Lane:
    name: str
    channel: int             # MIDI channel, 1-16
    input_channels: list[int]  # 0-based audio input channel indices
    banks: list[str]
    programs: range = field(default_factory=lambda: range(128))


@dataclass
class Result:
    lane: str
    channel: int
    bank: str
    program: int
    peak_dbfs: float
    rms_dbfs: float
    silent: bool
    timestamp: str


def parse_lane(spec: str) -> Lane:
    # "CH:INPUTS:BANKS" e.g. "1:0,1:A,B,C,D"
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--lane must be CH:INPUTS:BANKS, got {spec!r}")
    ch_s, inputs_s, banks_s = parts
    channel = int(ch_s)
    input_channels = [int(x) for x in inputs_s.split(",")]
    banks = [b.strip().upper() for b in banks_s.split(",")]
    for b in banks:
        if b not in BANK_LSB:
            raise argparse.ArgumentTypeError(f"unknown bank {b!r}, expected one of {sorted(BANK_LSB)}")
    return Lane(name=f"ch{channel}", channel=channel, input_channels=input_channels, banks=banks)


class MidiPort:
    """Thin wrapper so lane runners don't need to import mido directly, and so
    --dry-run can substitute a no-op port for testing without hardware."""

    def send_program(self, channel: int, bank: str, program: int):
        raise NotImplementedError

    def note_on(self, channel: int, notes: list[int], velocity: int):
        raise NotImplementedError

    def note_off(self, channel: int, notes: list[int]):
        raise NotImplementedError


class RealMidiPort(MidiPort):
    def __init__(self, port_name_substring: str):
        import mido

        self.mido = mido
        candidates = [n for n in mido.get_output_names() if port_name_substring.lower() in n.lower()]
        if not candidates:
            raise SystemExit(
                f"No MIDI output port matching {port_name_substring!r}. "
                f"Available: {mido.get_output_names()}"
            )
        self.port = mido.open_output(candidates[0])

    def send_program(self, channel: int, bank: str, program: int):
        ch0 = channel - 1
        self.port.send(self.mido.Message("control_change", channel=ch0, control=32, value=BANK_LSB[bank]))
        self.port.send(self.mido.Message("program_change", channel=ch0, program=program))

    def note_on(self, channel: int, notes: list[int], velocity: int):
        ch0 = channel - 1
        for n in notes:
            self.port.send(self.mido.Message("note_on", channel=ch0, note=n, velocity=velocity))

    def note_off(self, channel: int, notes: list[int]):
        ch0 = channel - 1
        for n in notes:
            self.port.send(self.mido.Message("note_off", channel=ch0, note=n, velocity=0))


class DryRunMidiPort(MidiPort):
    def send_program(self, channel, bank, program):
        pass

    def note_on(self, channel, notes, velocity):
        pass

    def note_off(self, channel, notes):
        pass


class LaneRunner(threading.Thread):
    def __init__(self, lane: Lane, midi: MidiPort, monitor, args, results: list, start_delay: float):
        super().__init__(daemon=True)
        self.lane = lane
        self.midi = midi
        self.monitor = monitor
        self.args = args
        self.results = results
        self.results_lock = threading.Lock()
        self.start_delay = start_delay
        self.stop_flag = threading.Event()

    def run(self):
        if self.start_delay > 0:
            time.sleep(self.start_delay)
        args = self.args
        for bank in self.lane.banks:
            for program in self.lane.programs:
                if self.stop_flag.is_set():
                    return
                t0 = time.time()
                self.midi.send_program(self.lane.channel, bank, program)
                time.sleep(args.settle)
                note_time = time.time()
                self.midi.note_on(self.lane.channel, args.chord, args.velocity)
                time.sleep(args.note_hold)
                self.midi.note_off(self.lane.channel, args.chord)

                measurement = None
                if self.monitor is not None:
                    measurement = self.monitor.measure(
                        self.lane.input_channels, note_time, args.note_hold + 0.3
                    )
                peak, rms = measurement if measurement else (float("-inf"), float("-inf"))
                silent = rms < args.threshold
                result = Result(
                    lane=self.lane.name,
                    channel=self.lane.channel,
                    bank=bank,
                    program=program,
                    peak_dbfs=peak,
                    rms_dbfs=rms,
                    silent=silent,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                )
                with self.results_lock:
                    self.results.append(result)
                flag = "SILENT" if silent else "ok"
                print(f"[{self.lane.name}] {bank}{program:03d}  peak={peak:6.1f}dBFS  rms={rms:6.1f}dBFS  {flag}")

                elapsed = time.time() - t0
                time.sleep(max(0.0, args.step - elapsed))


def run_selftest() -> bool:
    """Pure-logic checks for the ring buffer and dBFS math — no MIDI/audio
    hardware required. Run with --selftest."""
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}: {name}")
        ok = ok and cond

    # dBFS math
    check("silence is very negative dBFS", dbfs(0.0) < -100)
    check("full scale is ~0 dBFS", abs(dbfs(1.0)) < 1e-6)
    check("half amplitude is about -6dBFS", abs(dbfs(0.5) - (-6.02)) < 0.1)

    # ring buffer round-trip
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

    # wrap-around behaviour: write more than buffer length, oldest data should
    # become unavailable
    rb2 = RingBuffer(seconds=1.0, samplerate=sr, channels=1)
    block = np.ones((sr, 1), dtype=np.float32)
    rb2.write(block, 200.0)   # covers [199, 200)
    rb2.write(block, 201.0)   # covers [200, 201) -> overwrites first block
    stale = rb2.read_window(199.0, 1.0)
    check("overwritten window is reported unavailable", stale is None)
    fresh = rb2.read_window(200.0, 1.0)
    check("most recent window is still available", fresh is not None)

    # lane spec parsing
    lane = parse_lane("2:2,3:A,B")
    check("lane channel parsed", lane.channel == 2)
    check("lane inputs parsed", lane.input_channels == [2, 3])
    check("lane banks parsed", lane.banks == ["A", "B"])
    try:
        parse_lane("2:2,3:Z")
        check("invalid bank rejected", False)
    except argparse.ArgumentTypeError:
        check("invalid bank rejected", True)

    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-devices", action="store_true", help="list MIDI/audio devices and exit")
    p.add_argument("--selftest", action="store_true", help="run offline logic checks and exit")
    p.add_argument("--midi-port", help="substring to match a MIDI output port name")
    p.add_argument("--audio-device", help="substring or index to match an input audio device")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--lane", action="append", default=[], type=parse_lane,
                    help="CH:INPUTS:BANKS, repeatable, e.g. 1:0,1:A,B,C,D")
    p.add_argument("--stagger", type=float, default=20.0, help="seconds between each lane's start")
    p.add_argument("--step", type=float, default=20.0, help="seconds allotted per program")
    p.add_argument("--settle", type=float, default=0.3, help="seconds to wait after Program Change before the note")
    p.add_argument("--note-hold", type=float, default=2.5, help="seconds the test chord is held")
    p.add_argument("--chord", type=int, nargs="+", default=[60, 64, 67], help="MIDI note numbers to trigger")
    p.add_argument("--velocity", type=int, default=100)
    p.add_argument("--threshold", type=float, default=-50.0, help="rms dBFS below which a program is flagged silent")
    p.add_argument("--out", default=None, help="CSV report path (default: sweep_results_<timestamp>.csv)")
    p.add_argument("--dry-run", action="store_true", help="don't touch real MIDI/audio; useful to rehearse timing")
    args = p.parse_args()

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    if args.list_devices:
        try:
            import mido

            print("MIDI outputs:")
            for n in mido.get_output_names():
                print(f"  {n}")
        except ImportError:
            print("mido not installed (pip install mido python-rtmidi)")
        try:
            import sounddevice as sd

            print("\nAudio devices:")
            print(sd.query_devices())
        except ImportError:
            print("sounddevice not installed (pip install sounddevice)")
        return

    if not args.lane:
        p.error("at least one --lane is required (or use --selftest / --list-devices)")

    max_input_channel = max(c for lane in args.lane for c in lane.input_channels)

    midi: MidiPort
    monitor = None
    if args.dry_run:
        midi = DryRunMidiPort()
        print("Dry run: no MIDI or audio devices will be touched.")
    else:
        if not args.midi_port:
            p.error("--midi-port is required unless --dry-run")
        midi = RealMidiPort(args.midi_port)
        monitor = AudioMonitor(args.audio_device, args.samplerate, max_input_channel + 1)
        monitor.start()

    results: list[Result] = []
    runners = [
        LaneRunner(lane, midi, monitor, args, results, start_delay=i * args.stagger)
        for i, lane in enumerate(args.lane)
    ]
    for r in runners:
        r.start()
    try:
        for r in runners:
            r.join()
    except KeyboardInterrupt:
        print("\nInterrupted, stopping lanes...")
        for r in runners:
            r.stop_flag.set()
        for r in runners:
            r.join(timeout=5)
    finally:
        if monitor is not None:
            monitor.stop()

    out_path = args.out or f"sweep_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lane", "channel", "bank", "program", "peak_dbfs", "rms_dbfs", "silent", "timestamp"])
        for r in results:
            w.writerow([r.lane, r.channel, r.bank, r.program, f"{r.peak_dbfs:.1f}", f"{r.rms_dbfs:.1f}", r.silent, r.timestamp])

    silent = [r for r in results if r.silent]
    print(f"\nWrote {out_path} ({len(results)} programs tested, {len(silent)} flagged silent)")
    for r in silent:
        print(f"  SILENT: {r.lane} ch{r.channel} {r.bank}{r.program:03d}")


if __name__ == "__main__":
    main()
