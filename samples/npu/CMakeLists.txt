# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(npu)

target_sources(app PRIVATE
	src/main.c
	src/model_loader/model_flash_loader.c
	src/model_tensor_process/scrfd_postprocess.c
	src/model_test_input/wls_rgb888_planar.c
)
