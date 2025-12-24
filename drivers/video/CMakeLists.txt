# SPDX-License-Identifier: Apache-2.0

zephyr_library_include_directories(${ZEPHYR_BASE}/drivers/video)
zephyr_library_sources_ifdef(CONFIG_VIDEO_AMEBAPRO2 video_amebapro2.c video_amebapro2_controller.c)
