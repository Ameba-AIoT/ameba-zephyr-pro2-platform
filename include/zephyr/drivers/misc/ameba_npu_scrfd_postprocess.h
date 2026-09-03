/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief SCRFD output tensor postprocessing for the Ameba NPU driver.
 */

#ifndef ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_SCRFD_POSTPROCESS_H_
#define ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_SCRFD_POSTPROCESS_H_

#include <zephyr/drivers/misc/ameba_npu.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SCRFD_OUTPUT_COUNT 9U

struct scrfd_tensor {
	const void *data;
	struct ameba_npu_buffer_param param;
};

int scrfd_postprocess_and_log(const struct scrfd_tensor tensors[], uint32_t count);

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_SCRFD_POSTPROCESS_H_ */
