/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PLATFORM_SAMPLES_NPU_SRC_MODEL_TENSOR_PROCESS_SCRFD_POSTPROCESS_H_
#define PLATFORM_SAMPLES_NPU_SRC_MODEL_TENSOR_PROCESS_SCRFD_POSTPROCESS_H_

#include <zephyr/drivers/misc/ameba_npu.h>

#define SCRFD_OUTPUT_COUNT 9U

struct scrfd_tensor {
	const void *data;
	struct ameba_npu_buffer_param param;
};

int scrfd_postprocess_and_log(const struct scrfd_tensor tensors[], uint32_t count);

#endif /* PLATFORM_SAMPLES_NPU_SRC_MODEL_TENSOR_PROCESS_SCRFD_POSTPROCESS_H_ */
