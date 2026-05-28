/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @brief Public API for the Ameba NPU driver.
 */

#ifndef ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_H_
#define ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Maximum number of tensor dimensions reported by the NPU runtime.
 */
#define AMEBA_NPU_MAX_DIMS 6

/**
 * @brief Maximum tensor name size reserved by the NPU runtime.
 */
#define AMEBA_NPU_NAME_SIZE 64

/**
 * @brief Maximum number of input or output tensors supported per model.
 */
#define AMEBA_NPU_MAX_IO 16

enum ameba_npu_network_source {
	AMEBA_NPU_NETWORK_FROM_NONE = 0x00,
	AMEBA_NPU_NETWORK_FROM_FILE = 0x01,
	AMEBA_NPU_NETWORK_FROM_MEMORY = 0x02,
	AMEBA_NPU_NETWORK_FROM_FLASH = 0x04,
};

enum ameba_npu_buffer_format {
	AMEBA_NPU_BUFFER_FORMAT_FP32 = 0,
	AMEBA_NPU_BUFFER_FORMAT_FP16 = 1,
	AMEBA_NPU_BUFFER_FORMAT_UINT8 = 2,
	AMEBA_NPU_BUFFER_FORMAT_INT8 = 3,
	AMEBA_NPU_BUFFER_FORMAT_UINT16 = 4,
	AMEBA_NPU_BUFFER_FORMAT_INT16 = 5,
	AMEBA_NPU_BUFFER_FORMAT_CHAR = 6,
	AMEBA_NPU_BUFFER_FORMAT_BFP16 = 7,
	AMEBA_NPU_BUFFER_FORMAT_INT32 = 8,
	AMEBA_NPU_BUFFER_FORMAT_UINT32 = 9,
	AMEBA_NPU_BUFFER_FORMAT_INT64 = 10,
	AMEBA_NPU_BUFFER_FORMAT_UINT64 = 11,
	AMEBA_NPU_BUFFER_FORMAT_FP64 = 12,
	AMEBA_NPU_BUFFER_FORMAT_INT4 = 13,
	AMEBA_NPU_BUFFER_FORMAT_UINT4 = 14,
	AMEBA_NPU_BUFFER_FORMAT_BOOL8 = 16,
};

enum ameba_npu_buffer_quantize_format {
	AMEBA_NPU_BUFFER_QUANTIZE_NONE = 0,
	AMEBA_NPU_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT = 1,
	AMEBA_NPU_BUFFER_QUANTIZE_TF_ASYMM = 2,
};

struct ameba_npu_buffer_param {
	uint32_t dim_count;
	uint32_t dim_size[AMEBA_NPU_MAX_DIMS];
	enum ameba_npu_buffer_format data_format;
	enum ameba_npu_buffer_quantize_format quant_format;
	union {
		struct {
			int32_t fixed_point_pos;
		} dfp;
		struct {
			float scale;
			int32_t zero_point;
		} affine;
	} quant_data;
};

struct ameba_npu_inference_profile {
	uint32_t inference_time;
	uint32_t total_cycle;
};

/**
 * @brief NPU model handle.
 *
 * Holds all state for a loaded NBG model, including the runtime network
 * handle and the pre-allocated input/output DMA buffers. Callers may allocate
 * this structure on the stack or statically; it must not be freed while
 * inference is in progress.
 */
struct ameba_npu {
	void *network;
	void *inputs[AMEBA_NPU_MAX_IO];
	void *outputs[AMEBA_NPU_MAX_IO];
	uint32_t input_count;
	uint32_t output_count;
	bool prepared;
};

uint32_t ameba_npu_get_version(void);
int ameba_npu_init(void);
int ameba_npu_deinit(void);

/**
 * @brief Load an NBG model from flash (XIP).
 *
 * Convenience wrapper around ameba_npu_load_from_source() that always
 * uses @ref AMEBA_NPU_NETWORK_FROM_FLASH as the source. The @p data
 * pointer must remain valid for the lifetime of the model.
 *
 * @param model    Model handle to initialise; must not be NULL.
 * @param data     Pointer to the NBG binary in flash address space.
 * @param data_size Size of the NBG binary in bytes.
 *
 * @retval 0 on success.
 * @retval -EINVAL if any argument is NULL or @p data_size is zero.
 * @retval negative errno on runtime or buffer allocation failure.
 */
int ameba_npu_load(struct ameba_npu *model, const void *data,
				   size_t data_size);

/**
 * @brief Load an NBG model from the specified memory source.
 *
 * Initialises @p model, flushes the D-cache for the model data if
 * required by @p source, creates the runtime network, allocates input/output
 * buffers, and prepares the network for inference.
 *
 * On failure the handle is left zeroed and all partially allocated
 * resources are released.
 *
 * @param model     Model handle to initialise; must not be NULL.
 * @param data      Pointer to the NBG binary.
 * @param data_size Size of the NBG binary in bytes.
 * @param source    Memory source type (@ref ameba_npu_network_source).
 *
 * @retval 0 on success.
 * @retval -EINVAL if any argument is NULL or @p data_size is zero.
 * @retval negative errno on runtime or buffer allocation failure.
 */
int ameba_npu_load_from_source(struct ameba_npu *model,
							   const void *data, size_t data_size,
							   enum ameba_npu_network_source source);

/**
 * @brief Unload a model and release all associated resources.
 *
 * Finishes the runtime network, destroys all input/output buffers, and destroys
 * the network handle. The @p model handle is zeroed on return and may be reused
 * with ameba_npu_load().
 *
 * @param model Model handle to unload; must not be NULL.
 *
 * @retval 0 on success.
 * @retval -EINVAL if @p model is NULL.
 * @retval negative errno on runtime teardown failure.
 */
int ameba_npu_unload(struct ameba_npu *model);

/**
 * @brief Return the number of input tensors exposed by the model.
 *
 * @param model Model handle; may be NULL (returns 0).
 *
 * @return Number of input tensors, or 0 if @p model is NULL.
 */
uint32_t ameba_npu_get_input_count(const struct ameba_npu *model);

/**
 * @brief Return the number of output tensors exposed by the model.
 *
 * @param model Model handle; may be NULL (returns 0).
 *
 * @return Number of output tensors, or 0 if @p model is NULL.
 */
uint32_t ameba_npu_get_output_count(const struct ameba_npu *model);

/**
 * @brief Return the byte size of an input tensor buffer.
 *
 * @param model Model handle.
 * @param index Zero-based input tensor index.
 *
 * @return Byte size of the input buffer, or 0 on invalid arguments.
 */
size_t ameba_npu_get_input_size(const struct ameba_npu *model,
								uint32_t index);

/**
 * @brief Return the byte size of an output tensor buffer.
 *
 * @param model Model handle.
 * @param index Zero-based output tensor index.
 *
 * @return Byte size of the output buffer, or 0 on invalid arguments.
 */
size_t ameba_npu_get_output_size(const struct ameba_npu *model,
								 uint32_t index);

/**
 * @brief Copy data into an input tensor buffer.
 *
 * Maps the input DMA buffer, zeroes it, copies up to @p size bytes from
 * @p data, then flushes the D-cache before unmapping. If @p data is NULL
 * and @p size is 0 the buffer is zeroed without copying.
 *
 * @param model Model handle.
 * @param index Zero-based input tensor index.
 * @param data  Source data to copy; may be NULL only when @p size is 0.
 * @param size  Number of bytes to copy; must not exceed the tensor size.
 *
 * @retval 0 on success.
 * @retval -EINVAL if arguments are invalid or index is out of range.
 * @retval -ENOSPC if @p size exceeds the tensor buffer size.
 * @retval negative errno on map/flush failure.
 */
int ameba_npu_set_input(struct ameba_npu *model, uint32_t index,
						const void *data, size_t size);

/**
 * @brief Run inference on the loaded model.
 *
 * Triggers synchronous execution of the neural network. If @p profile is
 * non-NULL, timing and cycle-count information is written to it after
 * the run completes.
 *
 * @param model   Model handle; network must be prepared.
 * @param profile Optional profiling result; may be NULL.
 *
 * @retval 0 on success.
 * @retval -EINVAL if @p model is NULL or not prepared.
 * @retval negative errno on runtime run failure.
 */
int ameba_npu_inference(struct ameba_npu *model,
						struct ameba_npu_inference_profile *profile);

/**
 * @brief Copy output tensor data to a caller-supplied buffer.
 *
 * Invalidates the D-cache for the output buffer, maps it, copies up to
 * @p size bytes to @p data, then unmaps. If @p data is NULL or @p size
 * is 0 the cache is still invalidated and the actual size is written to
 * @p output_size.
 *
 * @param model       Model handle.
 * @param index       Zero-based output tensor index.
 * @param data        Destination buffer; may be NULL to query size only.
 * @param size        Capacity of @p data in bytes.
 * @param output_size Optional pointer to receive the actual tensor size;
 *                    may be NULL.
 *
 * @retval 0 on success.
 * @retval -EINVAL if arguments are invalid or index is out of range.
 * @retval negative errno on flush or map failure.
 */
int ameba_npu_get_output(struct ameba_npu *model, uint32_t index,
						 void *data, size_t size, size_t *output_size);

/**
 * @brief Query the quantization and format metadata of an output tensor.
 *
 * Retrieves data format, quantization scheme, and associated parameters
 * (fixed-point position or affine scale/zero-point) from the runtime network.
 * The result can be passed to post-processing code to perform correct
 * dequantization.
 *
 * @param model Model handle.
 * @param index Zero-based output tensor index.
 * @param param Pointer to the structure to fill with tensor metadata.
 *
 * @retval 0 on success.
 * @retval -EINVAL if any argument is NULL or index is out of range.
 * @retval negative errno on runtime query failure.
 */
int ameba_npu_get_output_param(const struct ameba_npu *model, uint32_t index,
							   struct ameba_npu_buffer_param *param);

/**
 * @brief Map an output tensor buffer for zero-copy read access.
 *
 * Invalidates the D-cache for the output buffer, then returns a direct
 * pointer to the tensor data. The caller must call ameba_npu_unmap_output()
 * when done. The pointer is valid only between map and unmap; inference
 * must not be started while the buffer is mapped.
 *
 * @param model Model handle.
 * @param index Zero-based output tensor index.
 *
 * @return Pointer to the mapped tensor data, or NULL on error.
 */
void *ameba_npu_map_output(struct ameba_npu *model, uint32_t index);

/**
 * @brief Unmap an output tensor buffer previously mapped with
 *        ameba_npu_map_output().
 *
 * @param model Model handle.
 * @param index Zero-based output tensor index.
 *
 * @retval 0 on success.
 * @retval -EINVAL if arguments are invalid or index is out of range.
 * @retval negative errno on runtime unmap failure.
 */
int ameba_npu_unmap_output(struct ameba_npu *model, uint32_t index);

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_INCLUDE_DRIVERS_MISC_AMEBA_NPU_H_ */
