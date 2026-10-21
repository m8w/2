#!/usr/bin/env python3
"""Build random SoundFont 2 (.sf2) files out of a pile of .wav files.

Each generated font picks a random subset of the input waves, maps them onto
the keyboard in a random way (key splits, velocity layers, or a drum-style
one-key-per-sample kit) and gives every zone random loop points, envelope,
filter, pan, tuning and LFO settings.  Pure standard library; ffmpeg is used
only as a fallback for wavs Python's `wave` module cannot read (float, ADPCM...).

    random_soundfont.py samples/ -n 5 -o fonts/            # 5 random fonts
    random_soundfont.py kick.wav pad.wav --mode layer      # layered font
    random_soundfont.py samples/ --seed 42                 # reproducible
"""
import argparse
import array
import math
import random
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

# SF2 generator operators
G_VIB_LFO_TO_PITCH, G_FILTER_FC, G_FILTER_Q = 6, 8, 9
G_CHORUS, G_REVERB, G_PAN = 15, 16, 17
G_DELAY_VIB, G_FREQ_VIB = 23, 24
G_ATTACK, G_HOLD, G_DECAY, G_SUSTAIN, G_RELEASE = 34, 35, 36, 37, 38
G_INSTRUMENT, G_KEYRANGE, G_VELRANGE, G_ATTEN = 41, 43, 44, 48
G_COARSE, G_FINE, G_SAMPLEID, G_SAMPLEMODES, G_ROOT = 51, 52, 53, 54, 58

TAIL = 46  # mandatory zero padding after every sample, in frames
MODES = ("split", "layer", "drum")

# (attack, decay, sustain cB, release) seconds ranges for a few "characters"
CHARACTERS = {
    "pad":   dict(att=(0.2, 2.5),   dec=(0.5, 3),   sus=(0, 200),   rel=(0.5, 4),   loop=0.9),
    "pluck": dict(att=(0.001, 0.02), dec=(0.2, 1.5), sus=(200, 800), rel=(0.1, 0.8), loop=0.3),
    "perc":  dict(att=(0.0005, 0.005), dec=(0.05, 0.6), sus=(600, 1440), rel=(0.05, 0.4), loop=0.0),
    "drone": dict(att=(0.5, 4),     dec=(1, 5),     sus=(0, 60),    rel=(1, 6),     loop=1.0),
    "glitch": dict(att=(0.0005, 0.3), dec=(0.01, 2), sus=(0, 1000), rel=(0.01, 2),  loop=0.5),
}


def tc(seconds):
    """Seconds -> SF2 timecents."""
    return max(-12000, min(8000, round(1200 * math.log2(max(seconds, 1e-4)))))


# ---------------------------------------------------------------- wav input --

def read_pcm(path):
    """Return (mono 16-bit samples as list[int], sample rate)."""
    with wave.open(str(path), "rb") as w:
        nch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 1:
        data = [(b - 128) << 8 for b in raw]
    elif width == 2:
        a = array.array("h", raw[: len(raw) // 2 * 2])
        if sys.byteorder == "big":
            a.byteswap()
        data = list(a)
    elif width == 3:
        data = [int.from_bytes(raw[i:i + 3], "little", signed=True) >> 8
                for i in range(0, len(raw) - 2, 3)]
    elif width == 4:
        a = array.array("i", raw[: len(raw) // 4 * 4])
        if sys.byteorder == "big":
            a.byteswap()
        data = [v >> 16 for v in a]
    else:
        raise wave.Error(f"unsupported sample width {width}")
    if nch > 1:
        n = len(data) // nch
        chans = [data[c::nch][:n] for c in range(nch)]
        data = [sum(f) // nch for f in zip(*chans)]
    return data, rate


def load_wav(path):
    try:
        return read_pcm(path)
    except (wave.Error, EOFError):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "c.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path),
                            "-acodec", "pcm_s16le", str(out)], check=True)
            return read_pcm(out)


def prepare(path, max_seconds):
    data, rate = load_wav(path)
    data = data[: int(max_seconds * rate)]
    if len(data) < 64:
        raise ValueError("too short")
    peak = max(1, max(abs(v) for v in data))
    gain = 0.9 * 32767 / peak
    data = [max(-32768, min(32767, round(v * gain))) for v in data]
    return data, max(400, min(rate, 50000))


def zero_crossings(data, lo, hi):
    return [i for i in range(max(lo, 1), min(hi, len(data) - 1)) if data[i - 1] < 0 <= data[i]]


def pick_loop(rng, data):
    """Random loop region snapped to upward zero crossings; None if too short."""
    n = len(data)
    if n < 2000:
        return None
    a = zero_crossings(data, int(n * rng.uniform(0.1, 0.5)), int(n * 0.55))
    b = zero_crossings(data, int(n * rng.uniform(0.6, 0.8)), n - 64)
    if not a or not b:
        return None
    start, end = rng.choice(a), rng.choice(b)
    return (start, end) if end - start > 64 else None


# ------------------------------------------------------------ font building --

def zone_gens(rng, character, loop):
    c = CHARACTERS[character]
    g = []
    g.append((G_ATTACK, tc(rng.uniform(*c["att"]))))
    g.append((G_DECAY, tc(rng.uniform(*c["dec"]))))
    g.append((G_SUSTAIN, round(rng.uniform(*c["sus"]))))
    g.append((G_RELEASE, tc(rng.uniform(*c["rel"]))))
    if rng.random() < 0.6:
        g.append((G_FILTER_FC, rng.randint(4500, 13500)))
        g.append((G_FILTER_Q, rng.randint(0, 300)))
    g.append((G_PAN, rng.randint(-350, 350)))
    g.append((G_FINE, rng.randint(-40, 40)))
    g.append((G_REVERB, rng.randint(0, 500)))
    g.append((G_CHORUS, rng.randint(0, 300)))
    if rng.random() < 0.3:
        g.append((G_DELAY_VIB, tc(rng.uniform(0.01, 1.0))))
        g.append((G_FREQ_VIB, round(1200 * math.log2(rng.uniform(3, 8) / 8.176))))
        g.append((G_VIB_LFO_TO_PITCH, rng.randint(5, 60)))
    if loop and rng.random() < c["loop"]:
        g.append((G_SAMPLEMODES, 1))
    return g


def layout(rng, mode, n):
    """Yield (keylo, keyhi, vello, velhi, root) per sample."""
    if mode == "drum":
        return [(36 + i, 36 + i, 1, 127, 36 + i) for i in range(n)]
    if mode == "layer":
        cuts = [1] + sorted(rng.sample(range(2, 127), n - 1)) + [128]
        return [(0, 127, cuts[i], cuts[i + 1] - 1, max(0, min(127, 60 + rng.randint(-12, 12))))
                for i in range(n)]
    cuts = [0] + sorted(rng.sample(range(8, 120), n - 1)) + [128]
    out = []
    for i in range(n):
        lo, hi = cuts[i], cuts[i + 1] - 1
        root = (lo + hi) // 2
        if rng.random() < 0.5:
            root = max(0, min(127, root + rng.randint(-12, 12)))
        out.append((lo, hi, 1, 127, root))
    return out


def chunk(tag, body):
    return tag + struct.pack("<I", len(body)) + body + b"\0" * (len(body) & 1)


def lst(tag, *chunks):
    return chunk(b"LIST", tag + b"".join(chunks))


def name20(s):
    return s.encode("ascii", "replace")[:19].ljust(20, b"\0")


def gen_bytes(oper, amount):
    if oper in (G_KEYRANGE, G_VELRANGE):
        return struct.pack("<HBB", oper, *amount)
    return struct.pack("<Hh", oper, amount)


def build_sf2(title, samples, zones):
    """samples: [(name, data, rate, root, loop|None)]; zones: [gen list ending in sampleID]."""
    smpl, shdr, pos = bytearray(), bytearray(), 0
    for name, data, rate, root, loop in samples:
        smpl += struct.pack(f"<{len(data)}h", *data) + b"\0" * (2 * TAIL)
        ls, le = loop or (16, len(data) - 16)
        shdr += struct.pack("<20sIIIIIBbHH", name20(name), pos, pos + len(data),
                            pos + ls, pos + le, rate, root, 0, 0, 1)
        pos += len(data) + TAIL
    shdr += struct.pack("<20sIIIIIBbHH", name20("EOS"), 0, 0, 0, 0, 0, 0, 0, 0, 0)

    ibag, igen = bytearray(), bytearray()
    for gens in zones:
        ibag += struct.pack("<HH", len(igen) // 4, 0)
        igen += b"".join(gen_bytes(*g) for g in gens)
    ibag += struct.pack("<HH", len(igen) // 4, 0)
    igen += b"\0" * 4

    info = lst(b"INFO",
               chunk(b"ifil", struct.pack("<HH", 2, 1)),
               chunk(b"isng", b"EMU8000\0"),
               chunk(b"INAM", title.encode("ascii", "replace")[:250].ljust(len(title) + 2 & ~1, b"\0")))
    sdta = lst(b"sdta", chunk(b"smpl", bytes(smpl)))
    pdta = lst(b"pdta",
               chunk(b"phdr", struct.pack("<20sHHHIII", name20(title), 0, 0, 0, 0, 0, 0)
                     + struct.pack("<20sHHHIII", name20("EOP"), 0, 0, 1, 0, 0, 0)),
               chunk(b"pbag", struct.pack("<HHHH", 0, 0, 1, 0)),
               chunk(b"pmod", b"\0" * 10),
               chunk(b"pgen", gen_bytes(G_INSTRUMENT, 0) + b"\0" * 4),
               chunk(b"inst", struct.pack("<20sH", name20(title), 0)
                     + struct.pack("<20sH", name20("EOI"), len(zones))),
               chunk(b"ibag", bytes(ibag)),
               chunk(b"imod", b"\0" * 10),
               chunk(b"igen", bytes(igen)),
               chunk(b"shdr", bytes(shdr)))
    return chunk(b"RIFF", b"sfbk" + info + sdta + pdta)


def random_font(rng, pool, title, mode, lo_n, hi_n, picks=None):
    """Build one font. `picks` (pool entries) re-uses given wavs; otherwise a random subset."""
    mode = mode or rng.choice(MODES)
    if picks is None:
        cap = {"layer": 126, "drum": 92}.get(mode, 100)
        picks = rng.sample(pool, min(len(pool), cap, rng.randint(lo_n, hi_n)))
    character = rng.choice(list(CHARACTERS))
    slots = layout(rng, mode, len(picks))
    samples, zones, info = [], [], []
    for i, ((path, data, rate), (klo, khi, vlo, vhi, root)) in enumerate(zip(picks, slots)):
        loop = None if mode == "drum" else pick_loop(rng, data)
        samples.append((Path(path).stem, data, rate, root, loop))
        gens = [(G_KEYRANGE, (klo, khi))]
        if (vlo, vhi) != (1, 127):
            gens.append((G_VELRANGE, (vlo, vhi)))
        gens += zone_gens(rng, "perc" if mode == "drum" else character, loop is not None)
        if mode == "drum":
            gens.append((G_ATTEN, rng.randint(0, 120)))
        gens.append((G_SAMPLEID, i))
        zones.append(gens)
        info.append(dict(path=Path(path), data=data, rate=rate, keys=(klo, khi), vel=(vlo, vhi),
                         root=root, loop=loop, gens={g[0]: g[1] for g in gens}))
    return dict(blob=build_sf2(title, samples, zones), title=title, mode=mode,
                character=character, picks=picks, zones=info)


def write_details(path, font):
    lines = [f"Name: {font['title']}", f"Layout: {font['mode']}", f"Character: {font['character']}", ""]
    for i, z in enumerate(font["zones"], 1):
        lines.append(f"Zone {i}: {z['path'].name}  keys {z['keys'][0]}-{z['keys'][1]} "
                     f"(root {z['root']})  velocity {z['vel'][0]}-{z['vel'][1]}  "
                     f"{'looped' if z['gens'].get(G_SAMPLEMODES) else 'one-shot'}")
    lines += ["", "Source wavs (used by --reroll):"] + [f"wav: {z['path'].resolve()}" for z in font["zones"]]
    path.write_text("\n".join(lines) + "\n")


def render_preview(font, path, rate=44100):
    """Rough audition of a font: one note per zone, mixed to a stereo wav.
    Applies pitch, pan and the volume envelope only (no filter, vibrato or reverb)."""
    events = []
    for z in sorted(font["zones"], key=lambda z: z["keys"]):
        key = (z["keys"][0] + z["keys"][1]) // 2 if font["mode"] != "layer" else 60
        events.append((key, (z["vel"][0] + z["vel"][1]) // 2, z))
    step, hold, tail = 0.55, 0.5, 1.5
    total = int((step * len(events) + hold + tail) * rate)
    left, right = [0.0] * total, [0.0] * total
    for n, (key, vel, z) in enumerate(events):
        g, data = z["gens"], z["data"]
        ratio = 2 ** ((key - z["root"]) / 12 + g.get(G_FINE, 0) / 1200) * z["rate"] / rate
        sec = lambda op, d: 2 ** (g.get(op, d) / 1200)
        att, dec, rel = sec(G_ATTACK, -12000), sec(G_DECAY, -12000), sec(G_RELEASE, -12000)
        sus = 10 ** (-g.get(G_SUSTAIN, 0) / 200)
        looped = z["loop"] if g.get(G_SAMPLEMODES) else None
        pan = (g.get(G_PAN, 0) / 500 + 1) / 2
        lg, rg = math.cos(pan * math.pi / 2), math.sin(pan * math.pi / 2)
        amp = (vel / 127) ** 2 * 10 ** (-g.get(G_ATTEN, 0) / 200) * 0.5
        start, pos, t, last, rel_level = int(n * step * rate), 0.0, 0.0, len(data) - 1, None
        for i in range(start, min(total, start + int((hold + min(rel, 2.0)) * rate))):
            if looped and pos >= looped[1]:
                pos -= looped[1] - looped[0]
            if pos >= last:
                break
            if t < hold:
                env = t / att if t < att else (1 - (1 - sus) * min(1, (t - att) / dec))
                rel_level = env
            else:
                env = rel_level * max(0.0, 1 - (t - hold) / rel)
            j = int(pos)
            v = (data[j] + (data[j + 1] - data[j]) * (pos - j)) / 32768 * env * amp
            left[i] += v * lg
            right[i] += v * rg
            pos += ratio
            t += 1 / rate
    peak = max(1e-9, max(map(abs, left + right)))
    scale = 0.9 * 32767 / peak if peak > 0.9 else 32767
    frames = array.array("h")
    for l, r in zip(left, right):
        frames.append(int(l * scale))
        frames.append(int(r * scale))
    if sys.byteorder == "big":
        frames.byteswap()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames.tobytes())


# --------------------------------------------------------------------- main --

def ask(prompt, default=""):
    try:
        ans = input(f"{prompt}{f' [{default}]' if default != '' else ''}: ").strip()
    except EOFError:
        sys.exit()
    return ans or str(default)


def ask_range(prompt, default, limit):
    """'5' -> (5, 5); '3-8' -> (3, 8); clamped to 1..limit."""
    while True:
        txt = ask(prompt, default).replace(" ", "")
        try:
            lo, _, hi = txt.partition("-")
            lo, hi = int(lo), int(hi or lo)
            return max(1, min(lo, hi, limit)), max(1, min(max(lo, hi), limit))
        except ValueError:
            print("  please type a number like 5, or a range like 3-8")


def find_wavs(paths):
    files = []
    for p in paths:
        if p.is_dir():
            files += [f for f in sorted(p.rglob("*")) if f.suffix.lower() == ".wav"
                      and not f.name.startswith(".")]
        else:
            files.append(p)
    return files


def safe_name(title):
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip() or "font"


def save_font(font, out, preview):
    stem = safe_name(font["title"])
    (out / f"{stem}.sf2").write_bytes(font["blob"])
    write_details(out / f"{stem}.txt", font)
    if preview:
        render_preview(font, out / f"{stem}_preview.wav")
    print(f"{stem}.sf2  {font['mode']}/{font['character']}  {len(font['picks'])} samples: "
          f"{', '.join(p[0].name for p in font['picks'])}")


def reroll_picks(details, pool_by_path, max_seconds):
    picks = []
    for line in details.read_text().splitlines():
        if line.startswith("wav: "):
            p = Path(line[5:])
            if p not in pool_by_path:
                pool_by_path[p] = (p, *prepare(p, max_seconds))
            picks.append(pool_by_path[p])
    if not picks:
        sys.exit(f"no 'wav:' lines in {details}")
    return picks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help=".wav files and/or folders (omit to be asked)")
    ap.add_argument("-o", "--out", help="output folder (default: 'SoundFonts' inside the wav folder)")
    ap.add_argument("-n", "--count", type=int, help="how many fonts to make")
    ap.add_argument("--name", help="font name (several fonts get ' 1', ' 2'...); default Random####")
    ap.add_argument("--mode", choices=MODES, help="keyboard layout (default: random per font)")
    ap.add_argument("--min-samples", type=int, help="fewest wavs per font")
    ap.add_argument("--max-samples", type=int, help="most wavs per font")
    ap.add_argument("--max-seconds", type=float, default=20, help="trim each wav to this length")
    ap.add_argument("--no-preview", action="store_true", help="skip the *_preview.wav audition files")
    ap.add_argument("--reroll", type=Path, metavar="DETAILS.txt",
                    help="keep the wavs listed in a font's .txt but re-randomize everything else")
    ap.add_argument("--seed", type=int, help="seed for reproducible output")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    preview = not args.no_preview

    if args.reroll:
        picks = reroll_picks(args.reroll, {}, args.max_seconds)
        out = Path(args.out) if args.out else args.reroll.parent
        out.mkdir(parents=True, exist_ok=True)
        base = args.name or args.reroll.stem
        for i in range(args.count or 1):
            title = f"{base} r{rng.randrange(10**3):03d}"
            save_font(random_font(rng, None, title, args.mode, 0, 0, picks), out, preview)
        return

    interactive = not args.inputs
    if interactive:
        print("Random SoundFont maker\n")
        while True:
            # drag-and-drop in Terminal pastes a path with backslash escapes / quotes
            txt = ask("Folder with your wav files (drag it here)", ".").strip("'\"")
            folder = Path(txt.replace("\\ ", " ")).expanduser()
            if folder.is_dir() and find_wavs([folder]):
                break
            print("  no wav files found there, try again")
        args.inputs = [str(folder)]

    pool = []
    for f in find_wavs(map(Path, args.inputs)):
        try:
            pool.append((f, *prepare(f, args.max_seconds)))
        except Exception as e:  # keep going past unreadable files
            print(f"skip {f}: {e}", file=sys.stderr)
    if not pool:
        sys.exit("no usable wav files")
    print(f"Found {len(pool)} usable wav files.")

    if interactive:
        lo_n, hi_n = ask_range("How many wavs per font? (a number like 5, or a range like 3-8)",
                               f"3-{min(12, len(pool))}", len(pool))
        args.count = ask_range("How many fonts to make?", 1, 999)[0]
        args.name = ask("Name for the fonts (Enter = Random1234 style)", "") or None
        mode = ask("Layout: split / layer / drum / random", "random").lower()
        args.mode = mode if mode in MODES else None
        preview = ask("Make a short preview .wav of each font? (y/n)", "y").lower().startswith("y")
    else:
        lo_n = args.min_samples or 3
        hi_n = args.max_samples or max(lo_n, 12)
        args.count = args.count or 1
        lo_n = max(1, min(lo_n, hi_n))

    first = Path(args.inputs[0])
    out = Path(args.out) if args.out else (first if first.is_dir() else first.parent) / "SoundFonts"
    out.mkdir(parents=True, exist_ok=True)
    by_path = {p[0]: p for p in pool}
    fonts = {}
    for i in range(args.count):
        if args.name:
            title = args.name if args.count == 1 else f"{args.name} {i + 1}"
        else:
            title = f"Random{rng.randrange(10**4):04d}"
        font = random_font(rng, pool, title, args.mode, lo_n, hi_n)
        fonts[safe_name(title)] = font
        save_font(font, out, preview)
    print(f"\nDone. Fonts are in: {out}")

    while interactive:  # keep the wavs of a font you almost like, re-randomize the rest
        pick = ask("\nRe-roll a font? Type its name to get a variation, or press Enter to finish", "")
        if not pick:
            break
        pick = pick.removesuffix(".sf2")
        if pick not in fonts:
            print("  names:", ", ".join(fonts))
            continue
        old = fonts[pick]
        title = f"{old['title'].split(' r')[0]} r{rng.randrange(10**3):03d}"
        font = random_font(rng, None, title, args.mode, 0, 0, old["picks"])
        fonts[safe_name(title)] = font
        save_font(font, out, preview)


if __name__ == "__main__":
    main()
