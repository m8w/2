#!/usr/bin/env python3
"""Sweep raw Programs and/or Drum Maps on a Novation Supernova II by
changing what a specific Performance Part is playing — never leaving
Performance Mode — and flag anything producing no audible output.

Why this script exists alongside supernova_performance_sweep.py: that
script's --target program/--target drum switch the WHOLE UNIT into plain
Program Mode to test a slot. Program Mode has no Output routing control at
all (manual, "Output - Button: This button only works in Performance
Mode.", p.135) — it always uses the unit's fixed default output pair. If
that pair is the one that's broken/unavailable (e.g. outputs 1&2 dead, only
outputs 3&4 wired to your interface), whole-unit Program Mode testing is
useless: everything comes back silent regardless of whether the patch
itself is fine, because the audio never reaches your interface at all.

The fix is this script: stay inside a Performance whose Parts are already
routed to a working output pair (Output button > "Part outputs", a
Performance/Part-level setting, p.135), and change only that Part's
Program via a Program Change on the Part's OWN MIDI channel — Part Output
routing survives that unchanged (manual: "Any Program from any one of the
Program Banks or any Drum Map can be assigned to any Part of a
Performance.", p.41). So if your outputs 1&2 are dead and you're routing
Parts to 3&4 instead, THIS is the script to use for testing raw Programs
and Drum Maps, not supernova_performance_sweep.py's program/drum targets.

Program banks (A-H) and Drum Map banks (a-h, lower-case per the manual's
own convention) can both be swept, mixed in the same --lane if you like.
They behave differently once selected, same as in supernova_performance_
sweep.py: a Program bank steps through Program Change numbers with a test
chord; a Drum Map is selected ONCE then swept note-by-note (C1-B4) since a
Drum Map is ~49 simultaneously active sounds, one per key, not 128
alternate sounds.

Multiple lanes (MIDI channels) can run in parallel IF each Part has its own
working, separately-wired Output pair — otherwise (e.g. everything forced
onto a single shared pair like 3&4) run ONE lane at a time, since a shared
audio bus can't distinguish which lane's note produced what you're hearing.

Setup required on the Supernova II before running:
  - Load a Performance whose Part(s) under test are already routed to a
    working Output pair (Output button > "Part outputs").
  - Global MIDI channel set to a channel NOT used by any lane below (e.g. 16).
  - Each Part under test given its own MIDI channel (Part Edit > MIDI menu),
    matching --lane's CH value, and NOT set to "Global" or "Omni".
  - Each Part's Program Change filter enabled (Global Menu, not Disabled).

Examples:

    python3 supernova_patch_sweep.py --list-devices

    python3 supernova_patch_sweep.py \\
        --midi-port "microKORG XL MIDI OUT" --audio-device "USB Audio CODEC" \\
        --lane 1:0,1:A,B,C,D,E,F,G,H

    python3 supernova_patch_sweep.py \\
        --midi-port "microKORG XL MIDI OUT" --audio-device "USB Audio CODEC" \\
        --lane 1:0,1:a,b,c,d,e,f,g,h --drum-step 2

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

from sn2_audio import (
    AudioMonitor,
    DRUM_NOTE_END,
    DRUM_NOTE_START,
    note_name,
    run_selftest as run_audio_selftest,
)

# Bank Select LSB (CC32 value), per the Supernova II MIDI implementation
# chart ("BANK MESSAGES"): 0=Favourites, 1-4=Perf banks A-D, 5-12=Prog banks
# A-H, 18-25=Drum Maps a-h (lower-case, per the manual's own convention to
# avoid confusion with Program banks A-H).
BANK_LSB = {
    "A": 5, "B": 6, "C": 7, "D": 8, "E": 9, "F": 10, "G": 11, "H": 12,
    "a": 18, "b": 19, "c": 20, "d": 21, "e": 22, "f": 23, "g": 24, "h": 25,
}


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
    number: int          # Program number, or (for a drum bank) the MIDI note tested
    label: str
    peak_dbfs: float
    rms_dbfs: float
    silent: bool
    timestamp: str


def parse_lane(spec: str) -> Lane:
    # "CH:INPUTS:BANKS" e.g. "1:0,1:A,B,C,D" or "1:0,1:a,b,c,d" (mixed OK)
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--lane must be CH:INPUTS:BANKS, got {spec!r}")
    ch_s, inputs_s, banks_s = parts
    channel = int(ch_s)
    input_channels = [int(x) for x in inputs_s.split(",")]
    banks = [b.strip() for b in banks_s.split(",")]
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

    def _log(self, bank, number, label, note_time, hold):
        args = self.args
        measurement = None
        if self.monitor is not None:
            measurement = self.monitor.measure(self.lane.input_channels, note_time, hold + 0.3)
        peak, rms = measurement if measurement else (float("-inf"), float("-inf"))
        silent = rms < args.threshold
        result = Result(
            lane=self.lane.name,
            channel=self.lane.channel,
            bank=bank,
            number=number,
            label=label,
            peak_dbfs=peak,
            rms_dbfs=rms,
            silent=silent,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        with self.results_lock:
            self.results.append(result)
        flag = "SILENT" if silent else "ok"
        print(f"[{self.lane.name}] {label}  peak={peak:6.1f}dBFS  rms={rms:6.1f}dBFS  {flag}")

    def _run_program_bank(self, bank):
        args = self.args
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

            self._log(bank, program, f"{bank}{program:03d}", note_time, args.note_hold)

            elapsed = time.time() - t0
            time.sleep(max(0.0, args.step - elapsed))

    def _run_drum_bank(self, bank, note_start=DRUM_NOTE_START, note_end=DRUM_NOTE_END):
        args = self.args
        self.midi.send_program(self.lane.channel, bank, 0)
        time.sleep(args.settle)

        for note in range(note_start, note_end + 1):
            if self.stop_flag.is_set():
                return
            t0 = time.time()
            note_time = time.time()
            self.midi.note_on(self.lane.channel, [note], args.velocity)
            time.sleep(args.drum_note_hold)
            self.midi.note_off(self.lane.channel, [note])

            self._log(bank, note, f"{bank} {note_name(note)} (note {note})", note_time, args.drum_note_hold)

            elapsed = time.time() - t0
            time.sleep(max(0.0, args.drum_step - elapsed))

    def run(self):
        if self.start_delay > 0:
            time.sleep(self.start_delay)
        for bank in self.lane.banks:
            if self.stop_flag.is_set():
                return
            if bank.islower():
                self._run_drum_bank(bank)
            else:
                self._run_program_bank(bank)


def run_selftest() -> bool:
    """Pure-logic checks for lane parsing, plus the shared audio-math checks
    from sn2_audio — no MIDI/audio hardware required. Run with --selftest."""
    ok = run_audio_selftest()

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}: {name}")
        ok = ok and cond

    check("Program bank A is Bank Select LSB 5 (not 4)", BANK_LSB["A"] == 5)
    check("Program bank H is Bank Select LSB 12", BANK_LSB["H"] == 12)
    check("Drum bank a is Bank Select LSB 18", BANK_LSB["a"] == 18)
    check("Drum bank h is Bank Select LSB 25", BANK_LSB["h"] == 25)

    lane = parse_lane("2:2,3:A,B")
    check("lane channel parsed", lane.channel == 2)
    check("lane inputs parsed", lane.input_channels == [2, 3])
    check("lane banks parsed", lane.banks == ["A", "B"])
    try:
        parse_lane("2:2,3:Z")
        check("invalid bank rejected", False)
    except argparse.ArgumentTypeError:
        check("invalid bank rejected", True)

    mixed_lane = parse_lane("1:0,1:A,a")
    check("mixed-case lane keeps Program and Drum bank distinct", mixed_lane.banks == ["A", "a"])

    class Args:
        chord = [60, 64, 67]
        velocity = 100
        settle = 0.01
        note_hold = 0.01
        step = 0.02
        drum_note_hold = 0.01
        drum_step = 0.02
        threshold = -50.0

    results: list[Result] = []
    lane = Lane(name="ch1", channel=1, input_channels=[0, 1], banks=["A"], programs=range(3))
    runner = LaneRunner(lane, DryRunMidiPort(), None, Args(), results, start_delay=0)
    runner._run_program_bank("A")
    check("program-bank dry-run covers 3 programs", len(results) == 3)
    check("program-bank dry-run labels look like A000", results[0].label == "A000")
    check("program-bank dry-run flags silent (no monitor)", all(r.silent for r in results))

    results.clear()
    drum_lane = Lane(name="ch1", channel=1, input_channels=[0, 1], banks=["a"])
    runner = LaneRunner(drum_lane, DryRunMidiPort(), None, Args(), results, start_delay=0)
    runner._run_drum_bank("a", note_start=36, note_end=38)  # a few notes, not the full C1-B4 range
    check("drum-bank dry-run covers 3 notes", len(results) == 3)
    check("drum-bank dry-run notes in order", [r.number for r in results] == [36, 37, 38])
    check("drum-bank dry-run label includes note name", "C1" in results[0].label)

    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-devices", action="store_true", help="list MIDI/audio devices and exit")
    p.add_argument("--selftest", action="store_true", help="run offline logic checks and exit")
    p.add_argument("--midi-port", help="substring to match a MIDI output port name")
    p.add_argument("--audio-device", help="substring or index to match an input audio device")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--lane", action="append", default=[], type=parse_lane,
                    help="CH:INPUTS:BANKS, repeatable, e.g. 1:0,1:A,B,C,D or 1:0,1:a,b,c,d (drum, lower-case)")
    p.add_argument("--stagger", type=float, default=20.0, help="seconds between each lane's start")
    p.add_argument("--step", type=float, default=20.0, help="[Program banks] seconds allotted per program")
    p.add_argument("--settle", type=float, default=0.3, help="seconds to wait after Program Change before the note")
    p.add_argument("--note-hold", type=float, default=2.5, help="[Program banks] seconds the test chord is held")
    p.add_argument("--chord", type=int, nargs="+", default=[60, 64, 67], help="[Program banks] MIDI notes to trigger")
    p.add_argument("--drum-step", type=float, default=2.0, help="[Drum banks] seconds allotted per note")
    p.add_argument("--drum-note-hold", type=float, default=0.4, help="[Drum banks] seconds each note is held")
    p.add_argument("--velocity", type=int, default=100)
    p.add_argument("--threshold", type=float, default=-50.0, help="rms dBFS below which a slot is flagged silent")
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
    if len(args.lane) > 1:
        print("Warning: multiple lanes only give unambiguous results if each lane's Part "
              "is routed to its own separately-wired Output pair. If everything shares one "
              "output pair, run one --lane at a time instead.", file=sys.stderr)

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
        w.writerow(["lane", "channel", "bank", "number", "label", "peak_dbfs", "rms_dbfs", "silent", "timestamp"])
        for r in results:
            w.writerow([r.lane, r.channel, r.bank, r.number, r.label, f"{r.peak_dbfs:.1f}", f"{r.rms_dbfs:.1f}", r.silent, r.timestamp])

    silent = [r for r in results if r.silent]
    dead_list_path = out_path.rsplit(".", 1)[0] + "_dead.txt"
    with open(dead_list_path, "w") as f:
        for r in silent:
            f.write(r.label + "\n")

    print(f"\nWrote {out_path} ({len(results)} tested, {len(silent)} flagged silent)")
    print(f"Wrote {dead_list_path} — steer Performance Parts away from these:")
    for r in silent:
        print(f"  SILENT: [{r.lane}] {r.label}")


if __name__ == "__main__":
    main()
