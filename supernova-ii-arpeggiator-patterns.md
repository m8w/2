# Novation Supernova II — Arpeggiator Patterns: Create New, Keep the Old

Reference notes distilled from `snvkii_manual_os20.pdf` (Supernova II OS 2.0 manual)
and `snvii_os14_addendum.pdf` (OS 1.4 addendum, which added the 2nd/3rd user banks).
Page numbers below refer to the OS 2.0 manual.

## The short version

The Supernova II Keyboard has **three** banks of 128 user arpeggiator patterns
(`U`, `V`, `W` — Nova II only has `U`/`V`). Patterns `000–063` in each bank are
Monophonic, `064–127` are Polyphonic. Because there are three whole banks, the
safest way to "make new patterns without losing the old ones" is:

1. **Back up what's there now** (SysEx dump to a computer/DAW), in case of a
   mistake.
2. **Write your new patterns into a bank/slot you aren't using** (e.g. bank `V`
   or `W`, or any empty numbered slot in `U`) instead of overwriting the ones
   you already rely on.

Editing itself is destructive the instant you hit **Write** — there's no
"save as" — so backup-first is the only real undo.

## Step 0 — Back up the existing patterns (do this first)

Writing a pattern into memory overwrites that slot permanently (manual, p.38,
same warning applies to patterns). Before touching anything, dump your current
patterns out over MIDI to a computer, DAW, or MIDI data filer:

1. Connect the Supernova II's MIDI **Out** to your computer/recorder's MIDI **In**.
2. Press **Global**, then use Page Up/Down to reach **Global Menu page 2**
   ("Sysex transmission").
3. Turn the lower Data knob to select what to dump. For a full backup of every
   user pattern bank, do this three times (once per bank) or use `Total data`
   for everything:
   - `Patt bank U`
   - `Patt bank V`
   - `Patt bank W` (Supernova II only)
   - (or `All arp patterns` / `Single arp pattern` for a narrower dump)
4. Arm your computer/DAW to record incoming MIDI/SysEx.
5. Press the **MIDI** button (in the Part Edit section) to start the dump —
   this only works while the Sysex transmission page is showing (p.48).
6. Save the captured SysEx file somewhere safe, labelled with today's date.

To restore later: set **Global Menu page 3** ("Sysex reception") to
`Normal (Rx as sent)`, then play the saved SysEx file back into the
Supernova II's MIDI In — each bank dump restores to the bank it came from.

## Step 1 — Pick a destination that won't clobber existing work

- Go to **Global Menu page 18** ("User pattern"). Use the Bank Up/Down buttons
  to pick bank `U`, `V`, or `W`.
- Pick a pattern **number you aren't already using** — e.g. if you only use
  bank `U`, build new patterns in `V` or `W` and leave `U` untouched. If you
  must reuse a bank, pick a slot number known to be blank/unused.
- Make a written note of exactly which Program(s)/Performance(s) currently
  reference the old pattern number, so reassigning later doesn't get confused.
- Also confirm **Memory Protect** is `Off` (**Global Menu page 7**) — without
  this, Write is blocked and nothing new can be saved anyway.

## Step 2 — Assign the destination pattern to the Arpeggiator you're editing

1. Select the Program you want the new pattern on.
2. Turn the Arpeggiator on (**Arp On/Off**).
3. Open the **Arpeggiator Menu → page 1** ("Pattern Bank" / "Pattern no.").
4. Set **Pattern Bank** to `User(U)`, `User(V)`, or `User(W)` and **Pattern no.**
   to the same slot you picked in Step 1. This makes the Arp Menu and the
   Global Menu edit pages point at the same pattern buffer.

## Step 3 — Choose Monophonic or Polyphonic

- **Monophonic** (patterns `000–063`): plays one note per step, in whatever
  order you assign — classic up/down/random arp behaviour, up to 64 steps.
- **Polyphonic** (patterns `064–127`): plays the whole held chord at every
  step, each step optionally transposing it — good for chord-stab/gater
  patterns.

Pick a number in the right half of the bank for the type you want.

## Step 4 — Program the pattern (Global Menu pages 18–20)

- **Page 18**: set `User pattern` (bank+number, same as Step 2) and
  `No of steps` (1–64). Start small (e.g. 8) while learning.
- **Page 19**: for each step, use the upper Data knob to choose which field to
  edit — `Step`, `Note`, `Vel.`, or `Gate` — and the lower Data knob to set its
  value:
  - `Note`: for Monophonic, `1–12` = which note of the currently-held chord
    (1 = lowest note played, counting up). For Polyphonic, `-36`..`+36` =
    semitone transposition applied to the whole chord at that step.
  - `Vel.`: `1–127`, velocity for that step.
  - `Gate`: `Norm` (plays for one step), `Tie` (ties into previous step),
    `Rest` (silent), `Glide` (portamento into the next step — needs `Note`,
    only available when Gate is `Norm` or `Glide`).
  - Bank Up/Down from page 19 jump to **Insert**/**Delete** sub-pages; Page
    Up/Down from there reaches **Rotate** (shift the whole pattern forward/back
    by N steps).
- Alternative: set **Global Menu page 17** ("Arp pattern editing via kbd") to
  `On`, then play notes on the keyboard (with sustain pedal for rest/tie/glide)
  to enter Note/Velocity/Gate step-by-step live instead of dialing each value.

These edits only live in the pattern *buffer* until written — nothing is
permanent yet.

## Step 5 — Write it to memory

- With any pattern-edit page showing, press **Write**. Confirm **Memory
  Protect** is `Off` (page 7) or this does nothing.
- This commits the buffer to the bank/slot you selected in Step 1 — this step
  is the destructive one, which is exactly why it's pointed at an unused slot.

## Step 6 — Point Programs/Performances at the new pattern

Update the Arpeggiator Menu's `Pattern Bank` / `Pattern no.` on whichever
Programs or Performance Parts should use the new pattern, then **Write** those
Programs/Performances too if you want the assignment to stick after power-off.
The old patterns are untouched in their original bank/slot the whole time.

## Restoring factory patterns (optional, if you want a clean slate to build on)

**Global Menu page 8** ("Restore from ROM") → set to `One patt` or `All patts`,
press **Write**. When restoring "All arp patterns," a second page lets you
choose the destination bank (`U`/`V`/`W`) — so you can, e.g., copy the factory
pattern set into bank `W` as a fresh base to customize, without disturbing
whatever you've built in `U`/`V`.

## Gotchas from the manual worth knowing up front

- Max 12 distinct note values in a Monophonic pattern; if you play more or
  fewer notes than the pattern references, turn `Fill In` (Arp Menu page 4) to
  `On` so the arp intelligently reuses steps instead of playing silence.
- Polyphonic transposition range is `-36` to `+36` semitones; `Fill In` does
  nothing in Polyphonic mode.
- `Glide` gate only works when the Program/Part's polyphony is set to `Mono`.
- Multiple arpeggiators (up to 8, one per Performance Part) can run different
  patterns/time signatures at once, but they all share one master clock — the
  Speed knob affects every running arpeggiator together.

## Building a Performance with 8 different arpeggiators

A Performance is 8 Parts, each a full copy of a Program (own Oscillators,
Filter, Effects **and** Arpeggiator, p.28). To get 8 independently-arpeggiated
Parts:

1. Press **Performance**, pick a Performance to start from/overwrite (or an
   empty slot), e.g. the factory `Multi Ch 1-8` template (Perf A126) which
   already has each Part on its own MIDI channel — a good base for layering
   or multitimbral use.
2. For **each of the 8 Parts** (press the **Part 1**…**Part 8** button):
   - Assign a Program with **Bank + keypad/Prog Up-Down** (pick any patch —
     see "Changing patches" below).
   - Turn that Part's **Arp On/Off** on (front panel, applies to the
     currently-selected Part).
   - Press **Special** (Part Edit section) → page 1 → set **"Arp bank &
     pattern used"** to `Part`. This is the key step: it lets this Part use
     its *own* pattern choice instead of whatever pattern is baked into the
     Program, so you don't need 8 separate Programs just to get 8 different
     patterns (p.140).
   - Press **Arp Menu** → page 1 → set **Pattern Bank**/**Pattern no.** to a
     *different* value for each Part — e.g. Part 1 = `Mono 000`, Part 2 =
     `Mono 037`, Part 3 = `Poly 071`, Part 4 = one of your own `User(U/V/W)`
     patterns from the earlier steps in this doc, etc. Since there are 128
     factory Mono + 128 factory Poly + up to 384 user patterns to draw from,
     "random" in practice means: pick 8 numbers without thinking too hard, or
     literally roll dice / use a random number generator to choose bank +
     pattern number for each Part.
   - Optionally vary **Sync** (Arp Menu page 5) per Part so they interlock at
     different subdivisions instead of all ticking in lockstep — they still
     share one master clock/Speed (p.28), so this is where the polyrhythmic
     variation comes from, not from independent tempos.
   - Optionally set **Range** (Part Edit section) per Part if you want a
     keyboard split/layer rather than all 8 stacked on the same notes.
3. Press **Write**, choose the destination Performance slot, name it, and
   confirm. Since writing is destructive (p.43), pick a free slot if you want
   to keep whatever Performance was there before.

Result: playing a chord (or holding notes across the whole keyboard, if
Ranges overlap) triggers up to 8 simultaneous, differently-patterned
arpeggios — a good starting point for generative/evolving textures.

## Changing which patches (Programs) a Performance uses

Yes — each Part's Program assignment is fully editable, independently of the
other Parts, without leaving Performance mode:

1. Press **Performance**, select the Performance.
2. Press the **Part** button for the Part whose patch you want to change
   (e.g. **Part 3**). The display switches to show that Part's currently
   assigned Program.
3. Use the **Bank** button + **keypad** (or Prog Up/Down) to pick a different
   Program from any of the Program banks (or a Drum Map) — this replaces the
   patch used by that Part only; the other 7 Parts are untouched (p.41).
4. Repeat for any other Parts you want to repatch.
5. Press **Write** to save the change into the Performance. You'll be asked
   `Update progs? No/Yes/Each` — leave this at `No` unless you specifically
   want to also overwrite the underlying Program memories with any knob
   tweaks you made while auditioning (p.44); `No` just saves which
   Program-slot each Part points to.

Note: switching a Part's Program also reloads that Program's own Arpeggiator
pattern/settings into the Part *unless* you've set "Arp bank & pattern used"
to `Part` for that Part (see above) — with that set, your Performance-level
pattern choice survives a patch change instead of being overwritten by the
new Program's default pattern.

## Testing that a patch is actually producing sound via USB into the Mac mini

I don't have physical access to your Supernova II or your Mac mini from this
session, so I can't run this test for you — but here's the checklist to
verify it yourself:

**Signal path**: the Supernova II Keyboard has no built-in USB audio output —
its outputs are analog (1/4" jacks, 8 outputs in 4 stereo pairs, p.135). So
"feeding it back via USB" means it's going: Supernova II analog out → cable →
a **USB audio interface** → USB → Mac mini. Confirm which interface is in the
chain before debugging further.

1. **Check the Supernova II side**
   - Confirm the Part(s) you're testing are set to **Part outputs 1 & 2**
     (Output button, p.135) unless you've deliberately routed them elsewhere
     — outputs 3–8 need separate cables into the interface too.
   - Play/trigger the arpeggiator and watch the Supernova II's own output
     level (if it has meters) or just confirm the Program isn't muted/soloed
     to a different Part (Mute/Solo buttons, p.42).
2. **Check the cabling** — Supernova II output 1 (L) and 2 (R) into the
   correct **input** channels 1/2 on the USB interface, not an output.
3. **On the Mac mini**:
   - Open **Audio MIDI Setup** (Applications → Utilities). Confirm the USB
     interface is listed and its input channels show activity (small level
     meters) when you play a note on the Supernova II.
   - Or: **System Settings → Sound → Input**, select the interface, and watch
     the input level bar while playing.
   - Or, to actually hear it: open **QuickTime Player → File → New Audio
     Recording**, click the dropdown arrow next to the record button, select
     the interface as the input source, and you'll hear/see the input live
     without needing to actually record.
   - In a DAW (Logic/GarageBand/Ableton), create an audio track, set its input
     to the interface's channel(s), and enable input monitoring.
4. **No signal?** Common culprits: interface's input gain/trim turned down,
   wrong input channel selected in the app vs. which physical jack you used,
   a TRS/TS cable mismatch, the interface set to a different sample rate than
   expected, or macOS's selected **input device** not matching the interface
   at all (another device, e.g. the Mac's built-in mic, may be selected
   instead).

If you tell me the specific USB audio interface model, I can give exact menu
names for its control panel/driver, if it has one.

## Automated sweep: testing every patch/Performance/Drum sound for silence

Manually stepping through every slot by hand and listening for silence
doesn't scale — Performances alone are 128 slots, Programs are 1024, Drum
Maps add ~392 more (8 maps × 49 sounds). `tools/supernova_performance_sweep.py`
covers all three via `--target performance|program|drum`; a second script,
`tools/supernova_patch_sweep.py`, covers a different, narrower case (see
below). Both flag anything whose recorded audio never rises above a silence
threshold and write a CSV report.

**Confirmed working on your hardware**: the SN2 has no native USB MIDI —
it's daisy-chained through the microKORG XL's DIN ports, which is the actual
USB-MIDI bridge to the Mac mini. The port that reaches the SN2 is
`microKORG XL MIDI OUT` (sending to this CoreMIDI destination drives the
microKORG's physical MIDI OUT jack, wired into the SN2's MIDI IN). Your
Global MIDI channel is `16`. A one-shot test (`Perf C000` selection) already
confirmed this routing works.

> **Correction**: an earlier version of this doc/script had the Program
> Bank Select values off by one (used `A=4..H=11`; the manual's own "BANK
> MESSAGES" table gives `A=5..H=12`, since LSB values 1-4 are Performance
> banks A-D and Program banks start at 5). Both scripts now source
> `BANK_LSB` directly from that table (p.166), checked by `--selftest`.

### How the three targets differ

All three select a slot the same way — Bank Select (CC32) + Program Change
on the **Global MIDI channel only** (manual, "BANK MESSAGES" table, p.166)
— so this is always a single sequential sweep, not parallel lanes: changing
the slot swaps what the whole unit is doing.

- **`--target performance`** (`C000`-`C127` etc.): select Performance `N`,
  broadcast a wide test chord across all 16 MIDI channels (so it doesn't
  matter which channel any given Part happens to listen on), measure, move
  on. Silence here means the *whole Performance* (all 8 Parts together)
  produced nothing.
- **`--target program`** (`A000`-`H127`): same idea, one raw Program at a
  time — this switches the whole unit into Program Mode for that single
  patch. Notes are sent on the Global channel only (Program Mode always
  listens there).
- **`--target drum`** (`a000`-`h048`): fundamentally different shape. A Drum
  Map isn't 49 alternate sounds, it's ~49 *simultaneously* active sounds,
  one per key from C1 to B4 (manual p.26). So the bank is selected **once**,
  then the sweep steps through individual **MIDI notes** one at a time —
  chords don't make sense here, since each note is a different underlying
  sound.

### Running it

```bash
pip install mido python-rtmidi sounddevice numpy

python3 tools/supernova_performance_sweep.py --list-devices

python3 tools/supernova_performance_sweep.py --selftest

python3 tools/supernova_performance_sweep.py --target performance --bank C \
    --midi-port "microKORG XL MIDI OUT" --audio-device "USB Audio CODEC" \
    --global-channel 16

python3 tools/supernova_performance_sweep.py --target program --bank ALL \
    --midi-port "microKORG XL MIDI OUT" --audio-device "USB Audio CODEC" \
    --global-channel 16

python3 tools/supernova_performance_sweep.py --target drum --bank ALL \
    --midi-port "microKORG XL MIDI OUT" --audio-device "USB Audio CODEC" \
    --global-channel 16
```

`--bank` accepts a single letter, a comma-separated list (`--bank A,B,C`), or
`ALL` (every valid bank for that `--target`) — so the `program --bank ALL`
run above sweeps all 1024 Programs (`A000`-`H127`) in one sitting instead of
8 separate invocations.

Rough timing at the defaults: Performances/Programs are 20s/slot (128 slots
≈ 43 minutes per bank — all 8 Program banks in one `--bank ALL` run ≈ 5.7
hours); Drum Maps are 2s/note (48 notes ≈ 96 seconds per map, all 8 maps
≈ 13 minutes). That's long enough you'll likely want to split it across a
few sittings — narrow any run with `--start`/`--end` (performance/program)
or `--note-start`/`--note-end` (drum), or just pass a subset of banks
(`--bank A,B,C`) per sitting.

Every run writes two files: `sweep_<target>_<banks>_<timestamp>.csv` (full
detail — `target, bank, number, label, peak_dbfs, rms_dbfs, silent,
timestamp`) and `..._dead.txt` (just the labels flagged silent, one per
line, e.g. `A017`) — that second file is your running "steer clear of
these" list for reprogramming Performance Parts. Re-running later (e.g.
after narrowing with `--start`/`--end` to double-check a few) overwrites
that run's own files; if you want one master list across multiple sittings,
just `cat` the `_dead.txt` files together and dedupe.

**Caveats these scripts can't remove**:
- *Performance* sweep: broadcasting across all 16 channels only proves
  *something* on some Part responded — if a Part's **Range** excludes every
  chord note, that Part alone looks silent even if it's fine. The flag is on
  the whole Performance, not a specific Part.
- *Drum* sweep: some notes in the C1-B4 range are legitimately meant to be
  unmapped/empty in a given kit — a `SILENT` flag there just means "nothing
  assigned or it's not sounding," which you'll want to eyeball against what
  that Drum Map is supposed to contain rather than treat as automatically
  broken.

Treat every `SILENT` result as "go look at this one," not automatically
"this patch is broken."

### Fixing something flagged silent

Common causes, from the Performance/Part Edit/Program sections of the
manual:

- **Polyphony = Off** for a Part (Polyphony menu) — the "no Program
  assigned" state (p.42).
- **Part Muted** — mute state is saved with the Performance (p.42); press
  Mute then the Part button to check/clear it.
- **Part Volume** at/near 0, or **Output** routed to a pair (3-8) you're
  not monitoring (Output menu, p.135).
- **Range** excludes every note you're testing with.
- A raw Program itself is genuinely silent (e.g. all Oscillator mix levels
  at 0) — rebuild or replace it.
- A Drum Map slot has nothing assigned to that note, or its Program's own
  Oscillator levels are at 0 (Drum Map Programs are edited exactly like
  normal Programs, p.3409-3431 of the manual text).

### The other script: `supernova_patch_sweep.py`

This one is for a narrower case: auditioning raw Programs **per Performance
Part**, independently, potentially in parallel across multiple Parts at
once (multi-channel "lanes") — e.g. building fresh Parts for a new
Performance and checking each candidate patch before assigning it, without
leaving Performance mode. It matches "channel 1 changes patch, channel 2
changes a patch ~20s later, keep going" only when the thing changing
per-channel is a Part's Program, not a whole-unit Program Mode switch.

```bash
python3 tools/supernova_patch_sweep.py --list-devices
python3 tools/supernova_patch_sweep.py --selftest
python3 tools/supernova_patch_sweep.py --dry-run --lane 1:0,1:A,B,C,D --lane 2:2,3:A,B,C,D

python3 tools/supernova_patch_sweep.py \
    --midi-port "microKORG XL MIDI OUT" --audio-device "USB Audio CODEC" \
    --lane 1:0,1:A,B,C,D --lane 2:2,3:A,B,C,D \
    --stagger 20 --step 20
```

(Last command: channel 1 → interface inputs 0/1, channel 2 → inputs 2/3,
sweep Program banks A-D on both, 20s per program, channel 2 starting 20s
after channel 1. Setup for this one is different from the Global-channel
scripts above — see the script's own `--help`/docstring: it needs each
Part given its own non-Global MIDI channel first.)

Both scripts share `tools/sn2_audio.py` for the audio-capture/dBFS logic —
`--selftest` on either exercises that shared code too.

I can't run these against your actual Supernova II / interface from this
session — I verified the pure logic (ring-buffer timing, dBFS math, Bank
Select LSB values against the manual's table, the C1-B4/note-name math for
Drum Maps, CLI parsing, sweep scheduling for all three targets) with
`--selftest` and `--dry-run`, catching a couple of my own bugs along the
way (the off-by-one bank table, and an argparse `--bank required=True`
that would have blocked `--selftest`/`--list-devices` outright), but the
real proof is a live run on your hardware — which you've now confirmed
routes correctly.
