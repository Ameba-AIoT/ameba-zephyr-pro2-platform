/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PLATFORM_SAMPLES_NPU_SRC_MODEL_LOADER_MODEL_FLASH_LOADER_H_
#define PLATFORM_SAMPLES_NPU_SRC_MODEL_LOADER_MODEL_FLASH_LOADER_H_

#include <stddef.h>
#include <stdint.h>

struct npu_flash_model {
	void *alloc;
	uint8_t *data;
	size_t size;
};

int npu_flash_model_load(const char *filename, struct npu_flash_model *model);
void npu_flash_model_unload(struct npu_flash_model *model);

#endif /* PLATFORM_SAMPLES_NPU_SRC_MODEL_LOADER_MODEL_FLASH_LOADER_H_ */
