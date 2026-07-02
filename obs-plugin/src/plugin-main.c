#include <obs-module.h>
#include "visualizer-source.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-molten-gold-visualizer", "en-US")

static const struct visualizer_type_data gold_type = {
	.id = "molten_gold_visualizer_source",
	.display_name = "Molten Gold Visualizer",
	.effect_filename = "molten_gold.effect",
};

static const struct visualizer_type_data rings_type = {
	.id = "beat_rings_visualizer_source",
	.display_name = "Beat Rings Visualizer",
	.effect_filename = "beat_rings.effect",
};

bool obs_module_load(void)
{
	visualizer_register_source_type(&gold_type);
	visualizer_register_source_type(&rings_type);

	blog(LOG_INFO, "[obs-molten-gold-visualizer] loaded (Molten Gold + Beat Rings sources)");
	return true;
}

void obs_module_unload(void)
{
	blog(LOG_INFO, "[obs-molten-gold-visualizer] unloaded");
}
