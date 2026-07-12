#include "audio-bands.h"

#include <math.h>
#include <string.h>
#include <obs.h>
#include <util/platform.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- RBJ biquad design (Audio EQ Cookbook), one section per band ---- */

static void biquad_reset(struct biquad *f)
{
	f->z1 = 0.0f;
	f->z2 = 0.0f;
}

static void biquad_design_lowpass(struct biquad *f, float freq, float q, float sample_rate)
{
	float w0 = 2.0f * (float)M_PI * freq / sample_rate;
	float alpha = sinf(w0) / (2.0f * q);
	float cosw0 = cosf(w0);

	float b0 = (1.0f - cosw0) / 2.0f;
	float b1 = 1.0f - cosw0;
	float b2 = (1.0f - cosw0) / 2.0f;
	float a0 = 1.0f + alpha;
	float a1 = -2.0f * cosw0;
	float a2 = 1.0f - alpha;

	f->b0 = b0 / a0;
	f->b1 = b1 / a0;
	f->b2 = b2 / a0;
	f->a1 = a1 / a0;
	f->a2 = a2 / a0;
	biquad_reset(f);
}

static void biquad_design_highpass(struct biquad *f, float freq, float q, float sample_rate)
{
	float w0 = 2.0f * (float)M_PI * freq / sample_rate;
	float alpha = sinf(w0) / (2.0f * q);
	float cosw0 = cosf(w0);

	float b0 = (1.0f + cosw0) / 2.0f;
	float b1 = -(1.0f + cosw0);
	float b2 = (1.0f + cosw0) / 2.0f;
	float a0 = 1.0f + alpha;
	float a1 = -2.0f * cosw0;
	float a2 = 1.0f - alpha;

	f->b0 = b0 / a0;
	f->b1 = b1 / a0;
	f->b2 = b2 / a0;
	f->a1 = a1 / a0;
	f->a2 = a2 / a0;
	biquad_reset(f);
}

static void biquad_design_bandpass(struct biquad *f, float center_freq, float q, float sample_rate)
{
	float w0 = 2.0f * (float)M_PI * center_freq / sample_rate;
	float alpha = sinf(w0) / (2.0f * q);
	float cosw0 = cosf(w0);

	float b0 = alpha;
	float b1 = 0.0f;
	float b2 = -alpha;
	float a0 = 1.0f + alpha;
	float a1 = -2.0f * cosw0;
	float a2 = 1.0f - alpha;

	f->b0 = b0 / a0;
	f->b1 = b1 / a0;
	f->b2 = b2 / a0;
	f->a1 = a1 / a0;
	f->a2 = a2 / a0;
	biquad_reset(f);
}

static inline float biquad_process(struct biquad *f, float x)
{
	/* transposed direct form II */
	float y = f->b0 * x + f->z1;
	f->z1 = f->b1 * x - f->a1 * y + f->z2;
	f->z2 = f->b2 * x - f->a2 * y;
	return y;
}

static void design_filters(struct audio_bands *b, uint32_t sample_rate)
{
	if (b->filters_sample_rate == sample_rate)
		return;
	biquad_design_lowpass(&b->bass_f, 150.0f, 0.707f, (float)sample_rate);
	biquad_design_bandpass(&b->mid_f, 560.0f, 0.5f, (float)sample_rate);
	biquad_design_highpass(&b->treble_f, 2500.0f, 0.707f, (float)sample_rate);
	b->filters_sample_rate = sample_rate;
}

/* ---- capture callback: runs on OBS's audio thread ---- */

static void audio_capture_cb(void *param, obs_source_t *source, const struct audio_data *audio_data, bool muted)
{
	UNUSED_PARAMETER(source);
	struct audio_bands *b = param;
	if (audio_data->frames == 0)
		return;

	struct obs_audio_info info;
	if (!obs_get_audio_info(&info))
		return;

	pthread_mutex_lock(&b->lock);
	design_filters(b, info.samples_per_sec);

	const float *samples = (const float *)audio_data->data[0];
	uint32_t n = audio_data->frames;

	float bass_sq = 0.0f, mid_sq = 0.0f, treble_sq = 0.0f, amp_sq = 0.0f;
	if (!muted && samples) {
		for (uint32_t i = 0; i < n; i++) {
			float x = samples[i];
			float bass_y = biquad_process(&b->bass_f, x);
			float mid_y = biquad_process(&b->mid_f, x);
			float treble_y = biquad_process(&b->treble_f, x);
			bass_sq += bass_y * bass_y;
			mid_sq += mid_y * mid_y;
			treble_sq += treble_y * treble_y;
			amp_sq += x * x;
		}
	}

	float raw_bass = sqrtf(bass_sq / (float)n);
	float raw_mid = sqrtf(mid_sq / (float)n);
	float raw_treble = sqrtf(treble_sq / (float)n);
	float raw_amp = sqrtf(amp_sq / (float)n);

	/* the same asymmetric attack/release smoothing as the web version */
	const float attack = 0.55f, release = 0.08f;
	b->bass += (raw_bass - b->bass) * (raw_bass > b->bass ? attack : release);
	b->mid += (raw_mid - b->mid) * (raw_mid > b->mid ? attack : release);
	b->treble += (raw_treble - b->treble) * (raw_treble > b->treble ? attack : release);
	b->amp += (raw_amp - b->amp) * (raw_amp > b->amp ? attack : release);

	float dt = (float)n / (float)info.samples_per_sec;
	b->bass_running_avg += (raw_bass - b->bass_running_avg) * fminf(1.0f, dt / 1.0f);
	b->beat_cooldown = fmaxf(0.0f, b->beat_cooldown - dt);
	if (raw_bass > b->bass_running_avg * 1.35f + 0.02f && b->beat_cooldown <= 0.0f) {
		b->beat_pulse = 1.0f;
		b->beat_cooldown = 0.12f;
	}

	pthread_mutex_unlock(&b->lock);
}

void audio_bands_init(struct audio_bands *b)
{
	memset(b, 0, sizeof(*b));
	pthread_mutex_init(&b->lock, NULL);
}

void audio_bands_free(struct audio_bands *b)
{
	audio_bands_set_source(b, NULL);
	pthread_mutex_destroy(&b->lock);
	bfree(b->tapped_name);
	b->tapped_name = NULL;
}

void audio_bands_set_source(struct audio_bands *b, const char *source_name)
{
	if (b->tapped_source) {
		obs_source_remove_audio_capture_callback(b->tapped_source, audio_capture_cb, b);
		obs_source_release(b->tapped_source);
		b->tapped_source = NULL;
	}
	bfree(b->tapped_name);
	b->tapped_name = NULL;

	if (source_name && *source_name) {
		obs_source_t *src = obs_get_source_by_name(source_name);
		if (src) {
			obs_source_add_audio_capture_callback(src, audio_capture_cb, b);
			b->tapped_source = src; /* holds the ref returned by obs_get_source_by_name */
			b->tapped_name = bstrdup(source_name);
		}
	}
}

void audio_bands_tick(struct audio_bands *b, float seconds)
{
	pthread_mutex_lock(&b->lock);
	b->beat_pulse *= powf(0.02f, seconds);
	pthread_mutex_unlock(&b->lock);
}

void audio_bands_get(struct audio_bands *b, float *bass, float *mid, float *treble, float *amp, float *beat)
{
	pthread_mutex_lock(&b->lock);
	*bass = b->bass;
	*mid = b->mid;
	*treble = b->treble;
	*amp = b->amp;
	*beat = b->beat_pulse;
	pthread_mutex_unlock(&b->lock);
}
