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
