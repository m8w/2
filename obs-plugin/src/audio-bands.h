#pragma once

#include <obs-module.h>
#include <util/threading.h>

/*
 * Taps a user-chosen OBS audio source and extracts smoothed bass/mid/treble
 * band energy plus a beat-onset pulse, mirroring the band-energy logic used
 * by the browser visualizer (visualizer/audio.js) but implemented with RBJ
 * biquad filters instead of an FFT, since libobs hands us raw float-planar
 * PCM rather than a ready-made frequency analyser.
 */

struct biquad {
	float b0, b1, b2, a1, a2;
	float z1, z2; /* transposed direct form II state */
};

struct audio_bands {
	pthread_mutex_t lock;

	obs_source_t *tapped_source; /* source we're currently hooked to (not owned) */
	char *tapped_name;

	struct biquad bass_f;
	struct biquad mid_f;
	struct biquad treble_f;
	uint32_t filters_sample_rate;

	/* smoothed 0..1 outputs, read by the render thread */
	float bass, mid, treble, amp;
	float beat_pulse;

	/* beat detector state */
	float bass_running_avg;
	float beat_cooldown;
};

void audio_bands_init(struct audio_bands *b);
void audio_bands_free(struct audio_bands *b);

/* Hook/unhook the audio capture callback. Pass NULL name to unhook. */
void audio_bands_set_source(struct audio_bands *b, const char *source_name);

/* Called once per video frame on the render thread to decay the beat pulse. */
void audio_bands_tick(struct audio_bands *b, float seconds);

/* Thread-safe snapshot for the render thread. */
void audio_bands_get(struct audio_bands *b, float *bass, float *mid, float *treble, float *amp, float *beat);
