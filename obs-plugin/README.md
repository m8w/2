# Molten Gold Visualizer (OBS plugin)

A native OBS Studio plugin that registers two video sources:

- **Molten Gold Visualizer** — the domain-warped fractal gold/bronze shader.
- **Beat Rings Visualizer** — expanding rings that pulse on detected beats.

Each source has an **Audio source to react to** dropdown (populated from your
existing OBS sources — Mic, Desktop Audio, a Media Source, etc.). The plugin
taps that source's raw audio, extracts bass/mid/treble energy and beat onsets
in real time (via biquad filters, no FFT dependency), and feeds them into the
GPU shader as it renders. To layer the two — the "OBS-style mixing" part —
add both sources to a scene and use OBS's own **scene item blending modes**
(right-click a source in the Sources list → *Blending Mode* → *Additive*)
instead of a custom compositor; that's the idiomatic OBS way to layer visuals
and it interacts correctly with everything else in your scene.

## What's been verified vs. not

This was built in a headless sandbox with no display and no OBS Studio GUI
install, so:

- **Verified here:** the C code compiles and links cleanly against real
  `libobs` headers/library (`libobs-dev` 30.0.2) with `-Wall -Wextra -Werror`,
  and the resulting `.so` exports the correct module entry points
  (`obs_module_load`, `obs_current_module`, etc.) with all symbols resolving
  against `libobs.so` — no missing/undefined references.
- **Not verified here:** the `.effect` shader files actually compiling on a
  real GPU driver, and the plugin loading/rendering inside a running OBS
  Studio window. I don't have a display or OBS install in this environment to
  do that. The effect files' syntax was written to match libobs's own shipped
  `.effect` files exactly (checked against `/usr/share/obs/libobs/*.effect`
  on this machine) and reuse math already confirmed correct in
  `golden-flow/index.html`'s GLSL, but the first real test needs to happen on
  your machine. If something doesn't compile, send me the OBS log
  (`Help > Log Files > View Current Log`) and I'll fix it from there.

## Building

You'll need OBS's development headers. On Ubuntu/Debian:

```sh
sudo apt install libobs-dev cmake build-essential
```

On other platforms, follow the [obs-studio build guide](https://github.com/obsproject/obs-studio/wiki) for your OS, or build against an existing OBS Studio install's SDK.

```sh
cd obs-plugin
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces `build/obs-molten-gold-visualizer.so` (Linux) — on macOS/Windows
the equivalent `.dylib`/`.dll` — plus the `data/` directory (locale + shader
files) that must ship alongside it.

## Installing

**Linux** (per-user plugin directory):

```sh
mkdir -p ~/.config/obs-studio/plugins/obs-molten-gold-visualizer/bin/64bit
mkdir -p ~/.config/obs-studio/plugins/obs-molten-gold-visualizer/data
cp build/obs-molten-gold-visualizer.so ~/.config/obs-studio/plugins/obs-molten-gold-visualizer/bin/64bit/
cp -r data/* ~/.config/obs-studio/plugins/obs-molten-gold-visualizer/data/
```

(`cmake --install build` does the same thing on Linux — the `CMakeLists.txt`
already points at this path.)

**macOS:**

```sh
mkdir -p ~/Library/Application\ Support/obs-studio/plugins/obs-molten-gold-visualizer/bin
mkdir -p ~/Library/Application\ Support/obs-studio/plugins/obs-molten-gold-visualizer/data
cp build/obs-molten-gold-visualizer.dylib ~/Library/Application\ Support/obs-studio/plugins/obs-molten-gold-visualizer/bin/
cp -r data/* ~/Library/Application\ Support/obs-studio/plugins/obs-molten-gold-visualizer/data/
```

**Windows:** copy `obs-molten-gold-visualizer.dll` into
`%ProgramData%\obs-studio\plugins\obs-molten-gold-visualizer\bin\64bit\` and
`data\` into the sibling `data\` folder, mirroring the Linux layout.

Restart OBS Studio after installing.

## Using it

1. In a scene, click **+** under Sources → **Molten Gold Visualizer**.
2. In its properties, pick an **Audio source to react to** (e.g. your Mic or
   Desktop Audio), set width/height to match your canvas, and adjust
   **Audio reactivity**.
3. Optionally add a **Beat Rings Visualizer** source on top, pick the same
   audio source, then right-click it in the Sources list → **Blending Mode**
   → **Additive** to layer it over the gold visual.
4. Click **Regenerate pattern** in a source's properties to reseed its noise
   field for a different look.

## Source layout

```
obs-plugin/
  CMakeLists.txt
  src/
    plugin-main.c        module entry point, registers both source types
    visualizer-source.c  shared obs_source_info implementation
    audio-bands.c         audio tap + biquad band-energy + beat detection
  data/
    locale/en-US.ini
    shaders/molten_gold.effect
    shaders/beat_rings.effect
```
