/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Flash-resident NN model loader for the Ameba NPU driver.
 */

#ifndef ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_MODEL_LOADER_H_
#define ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_MODEL_LOADER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct npu_flash_model {
	void *alloc;
	uint8_t *data;
	size_t size;
};

int npu_flash_model_load(const char *filename, struct npu_flash_model *model);
void npu_flash_model_unload(struct npu_flash_model *model);

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_MODEL_LOADER_H_ */
