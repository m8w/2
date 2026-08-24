#!/usr/bin/env python3
"""Sweep a Novation Supernova II library and flag anything producing no
audible output. Covers all three "kinds" of sound the manual defines
(p.27, "About Programs"/"About Drum Maps"): --target performance
(C000-C127 etc.), --target program (raw Programs, A000-H127), and
--target drum (Drum Maps a000-h048).

All three are selected the same way — Bank Select (CC32) + Program Change,
recognised ONLY on the Supernova II's Global MIDI channel (manual, "BANK
MESSAGES" table, p.166) — so this is always a single sequential loop, never
parallel lanes: selecting a new slot swaps what the whole unit is doing.

Performances and Programs are tested the same way: select slot N, trigger a
chord, measure, move to N+1. Drum Maps are different — a Drum Map is not
128 alternate sounds, it's ~49 *simultaneously* active sounds, one per key
from C1 to B4 (manual p.26). So for --target drum, the bank is selected
ONCE, then the sweep steps through individual MIDI notes one at a time
(chords don't make sense here — each note is a different underlying sound).

One-time setup on the Supernova II:
  - Note the Global MIDI channel (Global Menu page 1) and pass it with
    --global-channel — Bank Select/Program Change only works there.

Examples:

    python3 supernova_performance_sweep.py --list-devices

    python3 supernova_performance_sweep.py --target performance --bank C \\
        --midi-port "microKORG XL MIDI OUT" --audio-device "USB Audio CODEC" \\
        --global-channel 16

    python3 supernova_performance_sweep.py --target program --bank A \\
        --midi-port "microKORG XL MIDI OUT" --audio-device "USB Audio CODEC" \\
        --global-channel 16

    python3 supernova_performance_sweep.py --target drum --bank a \\
        --midi-port "microKORG XL MIDI OUT" --audio-device "USB Audio CODEC" \\
        --global-channel 16

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

# Bank Select LSB (CC32 value) per target, per the manual's "BANK MESSAGES"
# table (p.166): 0=Favourites, 1-4=Performance banks A-D, 5-12=Program banks
# A-H, 13-14=Arp Mono/Poly, 15-17=Arp User U/V/W, 18-25=Drum Maps a-h.
BANK_LSB = {
    "performance": {"A": 1, "B": 2, "C": 3, "D": 4},
    "program": {"A": 5, "B": 6, "C": 7, "D": 8, "E": 9, "F": 10, "G": 11, "H": 12},
    "drum": {"a": 18, "b": 19, "c": 20, "d": 21, "e": 22, "f": 23, "g": 24, "h": 25},
}

# A chord spanning 3 octaves, so a Performance whose Parts have narrow/split
# key Ranges is still likely to have at least one Part triggered, reducing
# false "silent" flags caused by range gaps rather than an actually broken
# patch. Also used for raw Program testing.
DEFAULT_CHORD = [36, 40, 43, 48, 52, 55, 60, 64, 67, 72, 76, 79]

# Drum Maps assign one Program per note from C1 to B4 (manual p.26) — note
# numbers per the Supernova II's own convention, confirmed elsewhere in the
# manual ("Drum played as" range is C-2 to G8, i.e. MIDI note 0 = C-2).
DRUM_NOTE_START = 36  # C1
DRUM_NOTE_END = 83    # B4

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(n: int) -> str:
    octave = n // 12 - 2
    return f"{_NOTE_NAMES[n % 12]}{octave}"


@dataclass
class Result:
    target: str
    bank: str
    number: int          # Performance/Program number, or (for drum) the MIDI note tested
    label: str
    peak_dbfs: float
    rms_dbfs: float
    silent: bool
    timestamp: str


class MidiOut:
    def select_slot(self, global_channel: int, target: str, bank: str, number: int):
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

    def select_slot(self, global_channel: int, target: str, bank: str, number: int):
        ch0 = global_channel - 1
        self.port.send(self.mido.Message("control_change", channel=ch0, control=32, value=BANK_LSB[target][bank]))
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
    def select_slot(self, global_channel, target, bank, number):
        pass

    def note_on(self, channels, notes, velocity):
        pass

    def note_off(self, channels, notes):
        pass


def _measure_and_log(midi, monitor, args, note_time, notes_held, target, bank, number, label) -> Result:
    measurement = None
    if monitor is not None:
        measurement = monitor.measure(args.input_channels, note_time, args.note_hold + 0.3)
    peak, rms = measurement if measurement else (float("-inf"), float("-inf"))
    silent = rms < args.threshold
    result = Result(
        target=target,
        bank=bank,
        number=number,
        label=label,
        peak_dbfs=peak,
        rms_dbfs=rms,
        silent=silent,
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
    flag = "SILENT" if silent else "ok"
    print(f"{label}  peak={peak:6.1f}dBFS  rms={rms:6.1f}dBFS  {flag}")
    return result


def sweep_performance_or_program(midi: MidiOut, monitor, args) -> list[Result]:
    results: list[Result] = []
    for number in range(args.start, args.end + 1):
        t0 = time.time()
        midi.select_slot(args.global_channel, args.target, args.bank, number)
        time.sleep(args.settle)

        note_time = time.time()
        midi.note_on(args.note_channels, args.chord, args.velocity)
        time.sleep(args.note_hold)
        midi.note_off(args.note_channels, args.chord)

        label = f"{args.bank}{number:03d}"
        results.append(_measure_and_log(midi, monitor, args, note_time, args.chord, args.target, args.bank, number, label))

        elapsed = time.time() - t0
        time.sleep(max(0.0, args.step - elapsed))
    return results


def sweep_drum(midi: MidiOut, monitor, args) -> list[Result]:
    midi.select_slot(args.global_channel, "drum", args.bank, 0)
    time.sleep(args.settle)

    results: list[Result] = []
    for note in range(args.note_start, args.note_end + 1):
        t0 = time.time()
        note_time = time.time()
        midi.note_on(args.note_channels, [note], args.velocity)
        time.sleep(args.note_hold)
        midi.note_off(args.note_channels, [note])

        label = f"{args.bank} {note_name(note)} (note {note})"
        results.append(_measure_and_log(midi, monitor, args, note_time, [note], "drum", args.bank, note, label))

        elapsed = time.time() - t0
        time.sleep(max(0.0, args.step - elapsed))
    return results


def sweep(midi: MidiOut, monitor, args) -> list[Result]:
    if args.target == "drum":
        return sweep_drum(midi, monitor, args)
    return sweep_performance_or_program(midi, monitor, args)


def run_selftest() -> bool:
    ok = run_audio_selftest()

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}: {name}")
        ok = ok and cond

    check("Perf bank A is Bank Select LSB 1", BANK_LSB["performance"]["A"] == 1)
    check("Perf bank C is Bank Select LSB 3", BANK_LSB["performance"]["C"] == 3)
    check("Prog bank A is Bank Select LSB 5", BANK_LSB["program"]["A"] == 5)
    check("Prog bank H is Bank Select LSB 12", BANK_LSB["program"]["H"] == 12)
    check("Drum bank a is Bank Select LSB 18", BANK_LSB["drum"]["a"] == 18)
    check("Drum bank h is Bank Select LSB 25", BANK_LSB["drum"]["h"] == 25)

    check("note_name(0) is C-2 (manual's MIDI-0 reference point)", note_name(0) == "C-2")
    check("note_name(127) is G8", note_name(127) == "G8")
    check("note_name(36) is C1 (Drum Map range start)", note_name(36) == "C1")
    check("note_name(83) is B4 (Drum Map range end)", note_name(83) == "B4")

    class PerfArgs:
        target = "performance"
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

    results = sweep(DryRunMidiOut(), None, PerfArgs())
    check("performance dry-run covers start..end inclusive", len(results) == 3)
    check("performance dry-run numbers in order", [r.number for r in results] == [0, 1, 2])
    check("performance dry-run flags silent (no monitor)", all(r.silent for r in results))
    check("performance dry-run labels look like C000", results[0].label == "C000")

    class ProgArgs(PerfArgs):
        target = "program"
        bank = "A"

    results = sweep(DryRunMidiOut(), None, ProgArgs())
    check("program dry-run labels look like A000", results[0].label == "A000")

    class DrumArgs(PerfArgs):
        target = "drum"
        bank = "a"
        note_start = 36
        note_end = 38

    results = sweep(DryRunMidiOut(), None, DrumArgs())
    check("drum dry-run covers note_start..note_end inclusive", len(results) == 3)
    check("drum dry-run notes in order", [r.number for r in results] == [36, 37, 38])
    check("drum dry-run label includes note name", "C1" in results[0].label)

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
    p.add_argument("--global-channel", type=int, default=1,
                    help="Supernova II's Global MIDI channel (1-16) — must match Global Menu page 1")
    p.add_argument("--target", choices=["performance", "program", "drum"], default="performance",
                    help="what to sweep: whole Performances, raw Programs, or one Drum Map's notes")
    p.add_argument("--bank", default=None,
                    help="bank letter for --target: A-D (performance), A-H (program), a-h (drum)")
    p.add_argument("--start", type=int, default=0, help="[performance/program] first number (0-127)")
    p.add_argument("--end", type=int, default=127, help="[performance/program] last number, inclusive (0-127)")
    p.add_argument("--note-start", type=int, default=DRUM_NOTE_START, help="[drum] first MIDI note to test")
    p.add_argument("--note-end", type=int, default=DRUM_NOTE_END, help="[drum] last MIDI note to test, inclusive")
    p.add_argument("--note-channels", type=int, nargs="+", default=None,
                    help="MIDI channels to trigger notes on (default: Global channel only, "
                         "or all 16 for --target performance)")
    p.add_argument("--step", type=float, default=None, help="seconds allotted per slot/note (default: 20 for "
                                                              "performance/program, 2 for drum)")
    p.add_argument("--settle", type=float, default=None, help="seconds to wait after selecting a slot before playing")
    p.add_argument("--note-hold", type=float, default=None, help="seconds a note/chord is held")
    p.add_argument("--chord", type=int, nargs="+", default=DEFAULT_CHORD,
                    help="[performance/program] MIDI note numbers to trigger")
    p.add_argument("--velocity", type=int, default=100)
    p.add_argument("--threshold", type=float, default=-50.0, help="rms dBFS below which a slot is flagged silent")
    p.add_argument("--out", default=None, help="CSV report path (default: sweep_<target>_<bank>_<timestamp>.csv)")
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

    if args.bank is None:
        p.error("--bank is required (unless using --selftest or --list-devices)")
    if args.bank not in BANK_LSB[args.target]:
        p.error(f"--bank {args.bank!r} isn't valid for --target {args.target}; "
                f"choose from {sorted(BANK_LSB[args.target])}")
    if args.target in ("performance", "program") and not (0 <= args.start <= args.end <= 127):
        p.error("--start/--end must satisfy 0 <= start <= end <= 127")
    if args.target == "drum" and not (0 <= args.note_start <= args.note_end <= 127):
        p.error("--note-start/--note-end must satisfy 0 <= note_start <= note_end <= 127")

    if args.note_channels is None:
        args.note_channels = list(range(1, 17)) if args.target == "performance" else [args.global_channel]
    if args.step is None:
        args.step = 20.0 if args.target in ("performance", "program") else 2.0
    if args.settle is None:
        args.settle = 0.5 if args.target == "performance" else (0.3 if args.target == "program" else 0.1)
    if args.note_hold is None:
        args.note_hold = 3.0 if args.target == "performance" else (2.0 if args.target == "program" else 0.4)

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

    out_path = args.out or f"sweep_{args.target}_{args.bank}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "bank", "number", "label", "peak_dbfs", "rms_dbfs", "silent", "timestamp"])
        for r in results:
            w.writerow([r.target, r.bank, r.number, r.label, f"{r.peak_dbfs:.1f}", f"{r.rms_dbfs:.1f}", r.silent, r.timestamp])

    silent = [r for r in results if r.silent]
    print(f"\nWrote {out_path} ({len(results)} tested, {len(silent)} flagged silent)")
    for r in silent:
        print(f"  SILENT: {r.label}")


if __name__ == "__main__":
    main()
