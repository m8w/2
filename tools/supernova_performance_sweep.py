#!/usr/bin/env python3
"""Sweep every Performance in a Novation Supernova II Performance bank
(e.g. C000-C127) and flag any that produce no audible output.

Performance selection is fundamentally different from Program selection: a
Bank Select (CC32) + Program Change for a Performance bank is only
recognised on the Supernova II's Global MIDI channel (per the "BANK
MESSAGES" table in the manual), and it swaps the *whole unit* — all 8 Parts
at once. So unlike supernova_patch_sweep.py's per-Part "lanes", this script
is a single sequential loop: select a Performance, trigger notes, measure,
move on.

Because a Performance's 8 Parts can each be set to listen on a different
MIDI channel (Global, Omni, or 1-16), and you may not know each Performance's
per-Part channel assignments in advance, this script broadcasts the test
chord across a configurable set of channels (default: 1-16) so a Part is
triggered regardless of which channel it's listening on. This trades a
little precision (you get "the Performance made no sound on any of these
channels", not "which Part failed") for not missing real silence just
because of a channel mismatch.

One-time setup on the Supernova II:
  - Note the Global MIDI channel (Global Menu page 1) and pass it with
    --global-channel — Bank Select/Program Change for Performances only
    works on that channel.
  - Nothing else needs to change; Parts can stay however they're configured.

Example — sweep Performance bank C, all 128 slots, 20s each:

    python3 supernova_performance_sweep.py --list-devices
    python3 supernova_performance_sweep.py \\
        --midi-port "Supernova" --audio-device "USB Audio CODEC" \\
        --global-channel 1 --bank C --step 20

Dependencies: pip install mido python-rtmidi sounddevice numpy
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime

from sn2_audio import AudioMonitor, run_selftest as run_audio_selftest

# Performance bank -> Bank Select LSB (CC32 value). Recognised ONLY on the
# Global MIDI channel (manual, "BANK MESSAGES" table).
PERF_BANK_LSB = {"A": 1, "B": 2, "C": 3, "D": 4}

# A chord spanning 3 octaves, so a Performance whose Parts have narrow/split
# key Ranges is still likely to have at least one Part triggered, reducing
# false "silent" flags caused by range gaps rather than an actually broken
# patch.
DEFAULT_CHORD = [36, 40, 43, 48, 52, 55, 60, 64, 67, 72, 76, 79]


@dataclass
class Result:
    bank: str
    performance: int
    peak_dbfs: float
    rms_dbfs: float
    silent: bool
    timestamp: str


class MidiOut:
    def select_performance(self, global_channel: int, bank: str, number: int):
        raise NotImplementedError

    def note_on(self, channels: list[int], notes: list[int], velocity: int):
        raise NotImplementedError

    def note_off(self, channels: list[int], notes: list[int]):
        raise NotImplementedError


class RealMidiOut(MidiOut):
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

    def select_performance(self, global_channel: int, bank: str, number: int):
        ch0 = global_channel - 1
        self.port.send(self.mido.Message("control_change", channel=ch0, control=32, value=PERF_BANK_LSB[bank]))
        self.port.send(self.mido.Message("program_change", channel=ch0, program=number))

    def note_on(self, channels: list[int], notes: list[int], velocity: int):
        for ch in channels:
            ch0 = ch - 1
            for n in notes:
                self.port.send(self.mido.Message("note_on", channel=ch0, note=n, velocity=velocity))

    def note_off(self, channels: list[int], notes: list[int]):
        for ch in channels:
            ch0 = ch - 1
            for n in notes:
                self.port.send(self.mido.Message("note_off", channel=ch0, note=n, velocity=0))


class DryRunMidiOut(MidiOut):
    def select_performance(self, global_channel, bank, number):
        pass

    def note_on(self, channels, notes, velocity):
        pass

    def note_off(self, channels, notes):
        pass


def sweep(midi: MidiOut, monitor, args) -> list[Result]:
    results: list[Result] = []
    for number in range(args.start, args.end + 1):
        t0 = time.time()
        midi.select_performance(args.global_channel, args.bank, number)
        time.sleep(args.settle)

        note_time = time.time()
        midi.note_on(args.note_channels, args.chord, args.velocity)
        time.sleep(args.note_hold)
        midi.note_off(args.note_channels, args.chord)

        measurement = None
        if monitor is not None:
            measurement = monitor.measure(args.input_channels, note_time, args.note_hold + 0.3)
        peak, rms = measurement if measurement else (float("-inf"), float("-inf"))
        silent = rms < args.threshold
        result = Result(
            bank=args.bank,
            performance=number,
            peak_dbfs=peak,
            rms_dbfs=rms,
            silent=silent,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        results.append(result)
        flag = "SILENT" if silent else "ok"
        print(f"{args.bank}{number:03d}  peak={peak:6.1f}dBFS  rms={rms:6.1f}dBFS  {flag}")

        elapsed = time.time() - t0
        time.sleep(max(0.0, args.step - elapsed))
    return results


def run_selftest() -> bool:
    ok = run_audio_selftest()

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}: {name}")
        ok = ok and cond

    check("Perf bank A is Bank Select LSB 1", PERF_BANK_LSB["A"] == 1)
    check("Perf bank C is Bank Select LSB 3", PERF_BANK_LSB["C"] == 3)
    check("Perf bank D is Bank Select LSB 4", PERF_BANK_LSB["D"] == 4)

    class Args:
        bank = "C"
        start = 0
        end = 2
        global_channel = 1
        note_channels = [1, 2]
        chord = [60, 64, 67]
        velocity = 100
        settle = 0.01
        note_hold = 0.01
        step = 0.02
        threshold = -50.0
        input_channels = [0, 1]

    results = sweep(DryRunMidiOut(), None, Args())
    check("dry-run sweep covers start..end inclusive", len(results) == 3)
    check("dry-run sweep numbers in order", [r.performance for r in results] == [0, 1, 2])
    check("dry-run sweep flags silent (no monitor)", all(r.silent for r in results))

    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-devices", action="store_true", help="list MIDI/audio devices and exit")
    p.add_argument("--selftest", action="store_true", help="run offline logic checks and exit")
    p.add_argument("--midi-port", help="substring to match a MIDI output port name")
    p.add_argument("--audio-device", help="substring or index to match an input audio device")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--input-channels", type=int, nargs="+", default=[0, 1],
                    help="0-based audio input channel indices to monitor")
    p.add_argument("--global-channel", type=int, required=False, default=1,
                    help="Supernova II's Global MIDI channel (1-16) — must match Global Menu page 1")
    p.add_argument("--bank", choices=sorted(PERF_BANK_LSB), default="C", help="Performance bank to sweep")
    p.add_argument("--start", type=int, default=0, help="first Performance number (0-127)")
    p.add_argument("--end", type=int, default=127, help="last Performance number, inclusive (0-127)")
    p.add_argument("--note-channels", type=int, nargs="+", default=list(range(1, 17)),
                    help="MIDI channels to broadcast the test chord on (default: all 16)")
    p.add_argument("--step", type=float, default=20.0, help="seconds allotted per Performance")
    p.add_argument("--settle", type=float, default=0.5, help="seconds to wait after Program Change before the note")
    p.add_argument("--note-hold", type=float, default=3.0, help="seconds the test chord is held")
    p.add_argument("--chord", type=int, nargs="+", default=DEFAULT_CHORD, help="MIDI note numbers to trigger")
    p.add_argument("--velocity", type=int, default=100)
    p.add_argument("--threshold", type=float, default=-50.0, help="rms dBFS below which a Performance is flagged silent")
    p.add_argument("--out", default=None, help="CSV report path (default: perf_sweep_<bank>_<timestamp>.csv)")
    p.add_argument("--dry-run", action="store_true", help="don't touch real MIDI/audio; useful to rehearse timing")
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)

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

    if not (0 <= args.start <= args.end <= 127):
        p.error("--start/--end must satisfy 0 <= start <= end <= 127")

    midi: MidiOut
    monitor = None
    if args.dry_run:
        midi = DryRunMidiOut()
        print("Dry run: no MIDI or audio devices will be touched.")
    else:
        if not args.midi_port:
            p.error("--midi-port is required unless --dry-run")
        midi = RealMidiOut(args.midi_port)
        monitor = AudioMonitor(args.audio_device, args.samplerate, max(args.input_channels) + 1)
        monitor.start()

    try:
        results = sweep(midi, monitor, args)
    finally:
        if monitor is not None:
            monitor.stop()

    out_path = args.out or f"perf_sweep_{args.bank}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bank", "performance", "peak_dbfs", "rms_dbfs", "silent", "timestamp"])
        for r in results:
            w.writerow([r.bank, r.performance, f"{r.peak_dbfs:.1f}", f"{r.rms_dbfs:.1f}", r.silent, r.timestamp])

    silent = [r for r in results if r.silent]
    print(f"\nWrote {out_path} ({len(results)} performances tested, {len(silent)} flagged silent)")
    for r in silent:
        print(f"  SILENT: {r.bank}{r.performance:03d}")


if __name__ == "__main__":
    main()
