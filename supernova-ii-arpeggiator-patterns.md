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

## Automated sweep: testing every patch in a library for silent output

Manually stepping through every Program by hand (128 per bank × up to 8
banks) and listening for silence doesn't scale. `tools/supernova_patch_sweep.py`
automates it: on one or more MIDI channels ("lanes"), it steps through a
Program bank one number at a time, triggers a chord, records the audio
arriving on that lane's assigned interface input channel(s), and flags any
program whose recorded level never rises above a silence threshold.

This matches the "channel 1 changes patch, channel 2 changes a patch ~20s
later, keep going, for every program in the library" idea directly: each
`--lane` is one MIDI channel/Performance Part, `--stagger` offsets when each
lane starts (so their very first note-onsets don't land on top of each
other), and `--step` is the ~20s given to each program before moving to the
next.

### One-time setup on the Supernova II

1. Load Performance **C000** (or whichever Performance you're testing from).
2. Set the **Global MIDI channel** (Global Menu page 1) to a channel none of
   your test lanes will use, e.g. `16`.
3. For each Part you're going to sweep (e.g. Part 1, Part 2): press its Part
   button, then **MIDI** menu page 1, set **MIDI channel** to a fixed value
   (`1`, `2`, ...) — not `Global`, not `Omni`.
4. Confirm the **Program Change filter** (Global Menu) isn't set to
   `Disabled` for those channels, or Program Changes will be ignored.
5. For genuinely parallel lanes (testing 2+ channels at once with unambiguous
   per-channel audio): give each Part its own **Part outputs** pair (Output
   button, e.g. Part 1 → outputs 1&2, Part 2 → outputs 3&4) and wire each
   pair into separate input channels on your USB interface. Running one lane
   at a time on a shared stereo bus needs none of this.

### Running it

```bash
pip install mido python-rtmidi sounddevice numpy

# find your exact MIDI/audio device names first
python3 tools/supernova_patch_sweep.py --list-devices

# offline check of the script's own logic, no hardware needed
python3 tools/supernova_patch_sweep.py --selftest

# rehearse timing without touching real MIDI/audio
python3 tools/supernova_patch_sweep.py --dry-run --lane 1:0,1:A,B,C,D --lane 2:2,3:A,B,C,D

# the real run: channel 1 -> interface inputs 0/1, channel 2 -> inputs 2/3,
# sweep Program banks A-D on both, 20s per program, channel 2 starts 20s
# after channel 1
python3 tools/supernova_patch_sweep.py \
    --midi-port "Supernova" --audio-device "Scarlett" \
    --lane 1:0,1:A,B,C,D --lane 2:2,3:A,B,C,D \
    --stagger 20 --step 20
```

Each run writes a timestamped CSV (`lane, channel, bank, program, peak_dbfs,
rms_dbfs, silent, timestamp`) and prints a live line per program plus a final
summary of everything flagged `SILENT`. Tune `--threshold` (default -50 dBFS
RMS) if your interface's noise floor makes that too sensitive or not
sensitive enough, and `--chord`/`--velocity`/`--note-hold` if a patch needs a
different trigger to speak (e.g. a higher velocity to clear a velocity
switch, or a longer hold for a slow attack).

I can't run this against your actual Supernova II / interface from here — I
verified the script's own logic (ring-buffer timing, dBFS math, CLI parsing,
multi-lane scheduling) with `--selftest` and `--dry-run`, but the real
proof is a live run on your hardware.
