# SPDX-License-Identifier: Apache-2.0

if(CONFIG_DISK_DRIVERS)

zephyr_library_sources_ifdef(CONFIG_SDMMC_AMEBAPRO2 sdmmc_amebapro2.c)

endif()
