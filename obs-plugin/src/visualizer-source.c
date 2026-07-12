#include "visualizer-source.h"
#include "audio-bands.h"

#include <string.h>
#include <obs.h>
#include <util/platform.h>
#include <util/dstr.h>
#include <graphics/vec2.h>

struct visualizer_data {
	obs_source_t *source;
	const struct visualizer_type_data *type_data;

	gs_effect_t *effect;
	gs_eparam_t *p_resolution;
	gs_eparam_t *p_time;
	gs_eparam_t *p_seed;
	gs_eparam_t *p_bass;
	gs_eparam_t *p_mid;
	gs_eparam_t *p_treble;
	gs_eparam_t *p_amp;
	gs_eparam_t *p_beat;
	gs_eparam_t *p_reactivity;

	uint32_t width, height;
	float reactivity;
	float seed;
	float elapsed;

	struct audio_bands bands;
	char *audio_source_name;
};

static float random_seed(void)
{
	return (float)(os_gettime_ns() % 1000000000ULL) / 1000000.0f;
}

static const char *visualizer_get_name(void *type_data)
{
	const struct visualizer_type_data *td = type_data;
	return td->display_name;
}

static void visualizer_load_effect(struct visualizer_data *v)
{
	struct dstr path = {0};
	dstr_copy(&path, "shaders/");
	dstr_cat(&path, v->type_data->effect_filename);
	char *full_path = obs_module_file(path.array);
	dstr_free(&path);

	if (!full_path) {
		blog(LOG_ERROR, "[%s] could not locate shader file: %s", v->type_data->id,
		     v->type_data->effect_filename);
		return;
	}

	obs_enter_graphics();
	char *error_string = NULL;
	v->effect = gs_effect_create_from_file(full_path, &error_string);
	if (!v->effect) {
		blog(LOG_ERROR, "[%s] failed to compile effect %s: %s", v->type_data->id, full_path,
		     error_string ? error_string : "(unknown error)");
	} else {
		v->p_resolution = gs_effect_get_param_by_name(v->effect, "uResolution");
		v->p_time = gs_effect_get_param_by_name(v->effect, "uTime");
		v->p_seed = gs_effect_get_param_by_name(v->effect, "uSeed");
		v->p_bass = gs_effect_get_param_by_name(v->effect, "uBass");
		v->p_mid = gs_effect_get_param_by_name(v->effect, "uMid");
		v->p_treble = gs_effect_get_param_by_name(v->effect, "uTreble");
		v->p_amp = gs_effect_get_param_by_name(v->effect, "uAmp");
		v->p_beat = gs_effect_get_param_by_name(v->effect, "uBeat");
		v->p_reactivity = gs_effect_get_param_by_name(v->effect, "uReactivity");
	}
	obs_leave_graphics();

	bfree(error_string);
	bfree(full_path);
}

static void visualizer_update(void *data, obs_data_t *settings)
{
	struct visualizer_data *v = data;

	v->width = (uint32_t)obs_data_get_int(settings, "width");
	v->height = (uint32_t)obs_data_get_int(settings, "height");
	v->reactivity = (float)obs_data_get_double(settings, "reactivity");

	const char *audio_source = obs_data_get_string(settings, "audio_source");
	bool changed = !v->audio_source_name || strcmp(v->audio_source_name, audio_source) != 0;
	if (changed) {
		audio_bands_set_source(&v->bands, audio_source);
		bfree(v->audio_source_name);
		v->audio_source_name = bstrdup(audio_source);
	}
}

static void *visualizer_create(obs_data_t *settings, obs_source_t *source)
{
	struct visualizer_data *v = bzalloc(sizeof(struct visualizer_data));
	v->source = source;
	v->type_data = obs_source_get_type_data(source);
	v->seed = random_seed();

	audio_bands_init(&v->bands);
	visualizer_load_effect(v);
	visualizer_update(v, settings);

	return v;
}

static void visualizer_destroy(void *data)
{
	struct visualizer_data *v = data;

	audio_bands_free(&v->bands);

	obs_enter_graphics();
	gs_effect_destroy(v->effect);
	obs_leave_graphics();

	bfree(v->audio_source_name);
	bfree(v);
}

static uint32_t visualizer_get_width(void *data)
{
	return ((struct visualizer_data *)data)->width;
}

static uint32_t visualizer_get_height(void *data)
{
	return ((struct visualizer_data *)data)->height;
}

static void visualizer_get_defaults2(void *type_data, obs_data_t *settings)
{
	UNUSED_PARAMETER(type_data);
	obs_data_set_default_string(settings, "audio_source", "");
	obs_data_set_default_int(settings, "width", 1280);
	obs_data_set_default_int(settings, "height", 720);
	obs_data_set_default_double(settings, "reactivity", 1.0);
}

static bool enum_audio_sources_cb(void *param, obs_source_t *src)
{
	obs_property_t *prop = param;
	uint32_t flags = obs_source_get_output_flags(src);
	if (flags & OBS_SOURCE_AUDIO) {
		const char *name = obs_source_get_name(src);
		obs_property_list_add_string(prop, name, name);
	}
	return true;
}

static bool regenerate_seed_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	struct visualizer_data *v = data;
	v->seed = random_seed();
	return false;
}

static obs_properties_t *visualizer_get_properties2(void *data, void *type_data)
{
	UNUSED_PARAMETER(type_data);
	obs_properties_t *props = obs_properties_create();

	obs_property_t *list = obs_properties_add_list(props, "audio_source", "Audio source to react to",
							OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(list, "(none)", "");
	obs_enum_sources(enum_audio_sources_cb, list);

	obs_properties_add_int(props, "width", "Width", 16, 7680, 2);
	obs_properties_add_int(props, "height", "Height", 16, 4320, 2);
	obs_properties_add_float_slider(props, "reactivity", "Audio reactivity", 0.0, 2.0, 0.01);
	obs_properties_add_button2(props, "regenerate_seed", "Regenerate pattern", regenerate_seed_clicked, data);

	return props;
}

static void visualizer_video_tick(void *data, float seconds)
{
	struct visualizer_data *v = data;
	v->elapsed += seconds;
	audio_bands_tick(&v->bands, seconds);
}

static void visualizer_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct visualizer_data *v = data;
	if (!v->effect || v->width == 0 || v->height == 0)
		return;

	float bass, mid, treble, amp, beat;
	audio_bands_get(&v->bands, &bass, &mid, &treble, &amp, &beat);

	struct vec2 res;
	vec2_set(&res, (float)v->width, (float)v->height);

	gs_effect_set_vec2(v->p_resolution, &res);
	gs_effect_set_float(v->p_time, v->elapsed);
	gs_effect_set_float(v->p_seed, v->seed);
	gs_effect_set_float(v->p_bass, bass);
	gs_effect_set_float(v->p_mid, mid);
	gs_effect_set_float(v->p_treble, treble);
	gs_effect_set_float(v->p_amp, amp);
	gs_effect_set_float(v->p_beat, beat);
	gs_effect_set_float(v->p_reactivity, v->reactivity);

	while (gs_effect_loop(v->effect, "Draw")) {
		gs_draw_sprite(NULL, 0, v->width, v->height);
	}
}

void visualizer_register_source_type(const struct visualizer_type_data *type_data)
{
	struct obs_source_info info = {0};
	info.id = type_data->id;
	info.type = OBS_SOURCE_TYPE_INPUT;
	info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW;
	info.get_name = visualizer_get_name;
	info.create = visualizer_create;
	info.destroy = visualizer_destroy;
	info.get_width = visualizer_get_width;
	info.get_height = visualizer_get_height;
	info.get_defaults2 = visualizer_get_defaults2;
	info.get_properties2 = visualizer_get_properties2;
	info.update = visualizer_update;
	info.video_tick = visualizer_video_tick;
	info.video_render = visualizer_video_render;
	info.type_data = (void *)type_data;
	info.icon_type = OBS_ICON_TYPE_COLOR;

	obs_register_source_s(&info, sizeof(info));
}
