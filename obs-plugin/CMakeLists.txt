cmake_minimum_required(VERSION 3.16)
project(obs-molten-gold-visualizer VERSION 1.0.0)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(libobs REQUIRED)

add_library(obs-molten-gold-visualizer MODULE
	src/plugin-main.c
	src/audio-bands.c
	src/visualizer-source.c
)

target_link_libraries(obs-molten-gold-visualizer PRIVATE OBS::libobs)

set_target_properties(obs-molten-gold-visualizer PROPERTIES
	PREFIX ""
)

if(UNIX AND NOT APPLE)
	install(TARGETS obs-molten-gold-visualizer
		LIBRARY DESTINATION "$ENV{HOME}/.config/obs-studio/plugins/obs-molten-gold-visualizer/bin/64bit"
	)
	install(DIRECTORY data/
		DESTINATION "$ENV{HOME}/.config/obs-studio/plugins/obs-molten-gold-visualizer/data"
	)
endif()
