/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief YOLOv4-tiny output tensor postprocessing for the Ameba NPU driver.
 */

#ifndef ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_YOLOV4_TINY_POSTPROCESS_H_
#define ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_YOLOV4_TINY_POSTPROCESS_H_

#include <stddef.h>

#include <zephyr/drivers/misc/ameba_npu.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YOLOV4_TINY_INPUT_WIDTH  416U
#define YOLOV4_TINY_INPUT_HEIGHT 416U
#define YOLOV4_TINY_OUTPUT_COUNT 2U

struct yolov4_tiny_tensor {
	const void *data;
	size_t size;
	struct ameba_npu_buffer_param param;
};

int yolov4_tiny_postprocess_and_log(const struct yolov4_tiny_tensor tensors[],
									uint32_t count);

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_YOLOV4_TINY_POSTPROCESS_H_ */
