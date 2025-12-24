# SPDX-License-Identifier: Apache-2.0

zephyr_library_include_directories(${ZEPHYR_BASE}/drivers/spi)
zephyr_library_sources_ifdef(CONFIG_SPI_AMEBAPRO2 spi_amebapro2.c)
