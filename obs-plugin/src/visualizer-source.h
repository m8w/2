#pragma once

#include <obs-module.h>

/*
 * Registers one obs_source_info per shader (molten gold / beat rings).
 * Both share the same create/destroy/render/property machinery; only the
 * .effect filename and display name differ, passed in as "type_data".
 */

struct visualizer_type_data {
	const char *id;
	const char *display_name;
	const char *effect_filename;
};

void visualizer_register_source_type(const struct visualizer_type_data *type_data);
