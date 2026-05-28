/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <errno.h>
#include <string.h>

#define DT_DRV_COMPAT realtek_amebapro2_npu

#include <soc.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/misc/ameba_npu.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include "hal_sys_ctrl.h"
#include <ameba_nn.h>

LOG_MODULE_REGISTER(ameba_npu, CONFIG_AMEBA_NPU_LOG_LEVEL);

struct ameba_npu_config {
	void *base_addr;
	uint32_t irq_num;
};

static bool valid_model(const struct ameba_npu *model)
{
	return model != NULL && model->network != NULL;
}

static NN_NetworkSourceTypeDef to_nn_network_source(enum ameba_npu_network_source source)
{
	return (NN_NetworkSourceTypeDef)source;
}

static NN_BufferFormatTypeDef to_nn_buffer_format(enum ameba_npu_buffer_format format)
{
	return (NN_BufferFormatTypeDef)format;
}

static NN_BufferQuantizeFormatTypeDef to_nn_quant_format(
	enum ameba_npu_buffer_quantize_format format)
{
	return (NN_BufferQuantizeFormatTypeDef)format;
}

static enum ameba_npu_buffer_format to_ameba_buffer_format(NN_BufferFormatTypeDef format)
{
	return (enum ameba_npu_buffer_format)format;
}

static enum ameba_npu_buffer_quantize_format to_ameba_quant_format(
	NN_BufferQuantizeFormatTypeDef format)
{
	return (enum ameba_npu_buffer_quantize_format)format;
}

static void copy_tensor_param_from_nn(const NN_BufferParamTypeDef *src,
									  struct ameba_npu_buffer_param *dst)
{
	memset(dst, 0, sizeof(*dst));
	dst->dim_count = src->dim_count;
	for (uint32_t i = 0; i < src->dim_count && i < AMEBA_NPU_MAX_DIMS; i++) {
		dst->dim_size[i] = src->dim_size[i];
	}
	dst->data_format = to_ameba_buffer_format(src->data_format);
	dst->quant_format = to_ameba_quant_format(src->quant_format);

	if (dst->quant_format == AMEBA_NPU_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
		dst->quant_data.dfp.fixed_point_pos =
			src->quant_data.dfp.fixed_point_pos;
	} else if (dst->quant_format == AMEBA_NPU_BUFFER_QUANTIZE_TF_ASYMM) {
		dst->quant_data.affine.scale = src->quant_data.affine.scale;
		dst->quant_data.affine.zero_point = src->quant_data.affine.zero_point;
	}
}

static int buffer_params_from_tensor(const NN_BufferParamTypeDef *tensor,
									 NN_BufferCreateParamTypeDef *params)
{
	if (tensor == NULL || params == NULL || tensor->dim_count > AMEBA_NPU_MAX_DIMS) {
		return -EINVAL;
	}

	memset(params, 0, sizeof(*params));
	params->num_of_dims = tensor->dim_count;
	for (uint32_t i = 0; i < tensor->dim_count; i++) {
		params->sizes[i] = tensor->dim_size[i];
	}
	params->data_format = to_nn_buffer_format((enum ameba_npu_buffer_format)tensor->data_format);
	params->quant_format = to_nn_quant_format(
							   (enum ameba_npu_buffer_quantize_format)tensor->quant_format);
	params->memory_type = NN_BUFFER_MEMORY_TYPE_DEFAULT;

	if (tensor->quant_format == NN_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
		params->quant_data.dfp.fixed_point_pos =
			tensor->quant_data.dfp.fixed_point_pos;
	} else if (tensor->quant_format == NN_BUFFER_QUANTIZE_TF_ASYMM) {
		params->quant_data.affine.scale = tensor->quant_data.affine.scale;
		params->quant_data.affine.zero_point = tensor->quant_data.affine.zero_point;
	}

	return 0;
}

static int create_buffer_from_tensor(const NN_BufferParamTypeDef *tensor, void **buffer)
{
	NN_BufferCreateParamTypeDef params;
	NN_BufferTypeDef nn_buffer = NULL;
	int ret;

	ret = buffer_params_from_tensor(tensor, &params);
	if (ret != 0) {
		return ret;
	}

	ret = NN_CreateBuffer(&params, &nn_buffer);
	if (ret == 0) {
		*buffer = nn_buffer;
	}

	return ret;
}

static void destroy_buffers(void *buffers[], uint32_t count)
{
	for (uint32_t i = 0; i < count; i++) {
		if (buffers[i] != NULL) {
			NN_BufferTypeDef buffer = buffers[i];

			(void)NN_DestroyBuffer(&buffer);
			buffers[i] = NULL;
		}
	}
}

static int create_io_buffers(struct ameba_npu *model)
{
	NN_BufferParamTypeDef tensor;
	int ret;

	ret = NN_QueryNetwork(model->network, NN_NETWORK_PROP_INPUT_COUNT,
						  &model->input_count);
	if (ret != 0) {
		return ret;
	}

	ret = NN_QueryNetwork(model->network, NN_NETWORK_PROP_OUTPUT_COUNT,
						  &model->output_count);
	if (ret != 0) {
		return ret;
	}

	if (model->input_count > AMEBA_NPU_MAX_IO ||
		model->output_count > AMEBA_NPU_MAX_IO) {
		LOG_ERR("IO count exceeds limit: in=%u out=%u max=%u",
				model->input_count, model->output_count, AMEBA_NPU_MAX_IO);
		return -ENOSPC;
	}

	for (uint32_t i = 0; i < model->input_count; i++) {
		ret = NN_QueryInputParam(model->network, i, &tensor);
		if (ret != 0) {
			return ret;
		}

		ret = create_buffer_from_tensor(&tensor, &model->inputs[i]);
		if (ret != 0) {
			return ret;
		}
	}

	for (uint32_t i = 0; i < model->output_count; i++) {
		ret = NN_QueryOutputParam(model->network, i, &tensor);
		if (ret != 0) {
			return ret;
		}

		ret = create_buffer_from_tensor(&tensor, &model->outputs[i]);
		if (ret != 0) {
			return ret;
		}
	}

	return 0;
}

static const struct device *ameba_npu_get_device(void)
{
#if DT_HAS_COMPAT_STATUS_OKAY(DT_DRV_COMPAT)
	return DEVICE_DT_GET_ONE(realtek_amebapro2_npu);
#else
	return NULL;
#endif
}

static int ameba_npu_device_init(const struct device *dev)
{
	const struct ameba_npu_config *config = dev->config;

	LOG_DBG("Ameba NPU DTS info: base_address=%p irq=%u",
			config->base_addr, config->irq_num);

#if defined(NN_BASE)
	if ((uintptr_t)config->base_addr != (uintptr_t)NN_BASE) {
		LOG_WRN("NPU DTS base %p differs from HAL NN_BASE 0x%08x",
				config->base_addr, (uint32_t)NN_BASE);
	}
#endif

	if (config->irq_num != (uint32_t)NN_IRQn) {
		LOG_WRN("NPU DTS IRQ %u differs from HAL NN_IRQn %u",
				config->irq_num, (uint32_t)NN_IRQn);
	}

	return 0;
}

static int ameba_npu_check_device_ready(void)
{
	const struct device *dev = ameba_npu_get_device();

	if (dev == NULL || !device_is_ready(dev)) {
		LOG_ERR("Ameba NPU device is not ready");
		return -ENODEV;
	}

	return 0;
}

static void ameba_npu_hardware_init(void)
{
	LOG_INF("Enabling Ameba NPU hardware clock");
	hal_sys_peripheral_en(NN_SYS, ENABLE);
	hal_sys_set_clk(NN_SYS, NN_500M);
	LOG_INF("Ameba NPU clock: 0x%x", hal_sys_get_clk(NN_SYS));
}

uint32_t ameba_npu_get_version(void)
{
	return NN_GetVersion();
}

int ameba_npu_init(void)
{
	int ret;

	ret = ameba_npu_check_device_ready();
	if (ret != 0) {
		return ret;
	}

	ameba_npu_hardware_init();

	return NN_Init();
}

int ameba_npu_deinit(void)
{
	int ret;

	ret = ameba_npu_check_device_ready();
	if (ret != 0) {
		return ret;
	}

	return NN_DeInit();
}

#define AMEBA_NPU_DEVICE_INIT(n)                                                                  \
	static const struct ameba_npu_config ameba_npu_config_##n = {                            \
		.base_addr = (void *)DT_INST_REG_ADDR(n),                                        \
		.irq_num = DT_INST_IRQN(n),                                                      \
	};                                                                                         \
                                                                                                   \
	DEVICE_DT_INST_DEFINE(n, ameba_npu_device_init, NULL, NULL,                            \
			      &ameba_npu_config_##n, POST_KERNEL,                              \
			      CONFIG_KERNEL_INIT_PRIORITY_DEFAULT, NULL);

DT_INST_FOREACH_STATUS_OKAY(AMEBA_NPU_DEVICE_INIT)

int ameba_npu_load(struct ameba_npu *model, const void *data, size_t data_size)
{
	return ameba_npu_load_from_source(model, data, data_size,
									  AMEBA_NPU_NETWORK_FROM_FLASH);
}

int ameba_npu_load_from_source(struct ameba_npu *model, const void *data,
							   size_t data_size, enum ameba_npu_network_source source)
{
	const void *network_data = data;
	uintptr_t model_base = (uintptr_t)data;
	NN_NetworkTypeDef network = NULL;
	int ret;

	if (model == NULL || data == NULL || data_size == 0U) {
		return -EINVAL;
	}

	memset(model, 0, sizeof(*model));

	if (source != AMEBA_NPU_NETWORK_FROM_FILE) {
		NN_FlushDCache(&model_base, &data_size, 1);
	}

	if (source == AMEBA_NPU_NETWORK_FROM_FLASH) {
		network_data = &data;
	}

	ret = NN_CreateNetwork(network_data, data_size, to_nn_network_source(source),
						   &network);
	if (ret != 0) {
		return ret;
	}
	model->network = network;

	ret = create_io_buffers(model);
	if (ret != 0) {
		goto fail;
	}

	ret = NN_PrepareNetwork(model->network);
	if (ret != 0) {
		goto fail;
	}
	model->prepared = true;

	for (uint32_t i = 0; i < model->output_count; i++) {
		ret = NN_SetOutput(model->network, i, model->outputs[i]);
		if (ret != 0) {
			goto fail;
		}
	}

	return 0;

fail:
	(void)ameba_npu_unload(model);
	return ret;
}

int ameba_npu_unload(struct ameba_npu *model)
{
	int ret = 0;
	int cleanup_ret;

	if (model == NULL) {
		return -EINVAL;
	}

	if (model->prepared && model->network != NULL) {
		ret = NN_FinishNetwork(model->network);
		model->prepared = false;
	}

	destroy_buffers(model->inputs, model->input_count);
	destroy_buffers(model->outputs, model->output_count);

	if (model->network != NULL) {
		NN_NetworkTypeDef network = model->network;

		cleanup_ret = NN_DestroyNetwork(&network);
		if (ret == 0) {
			ret = cleanup_ret;
		}
	}

	memset(model, 0, sizeof(*model));

	return ret;
}

uint32_t ameba_npu_get_input_count(const struct ameba_npu *model)
{
	return model == NULL ? 0U : model->input_count;
}

uint32_t ameba_npu_get_output_count(const struct ameba_npu *model)
{
	return model == NULL ? 0U : model->output_count;
}

size_t ameba_npu_get_input_size(const struct ameba_npu *model, uint32_t index)
{
	if (!valid_model(model) || index >= model->input_count) {
		return 0;
	}

	return NN_GetBufferSize(model->inputs[index]);
}

size_t ameba_npu_get_output_size(const struct ameba_npu *model, uint32_t index)
{
	if (!valid_model(model) || index >= model->output_count) {
		return 0;
	}

	return NN_GetBufferSize(model->outputs[index]);
}

int ameba_npu_set_input(struct ameba_npu *model, uint32_t index,
						const void *data, size_t size)
{
	void *mapped;
	size_t input_size;
	int ret;

	if (!valid_model(model) || index >= model->input_count) {
		return -EINVAL;
	}

	if (data == NULL && size != 0U) {
		return -EINVAL;
	}

	input_size = NN_GetBufferSize(model->inputs[index]);
	if (data != NULL && size > input_size) {
		return -ENOSPC;
	}

	mapped = NN_MapBuffer(model->inputs[index]);
	if (mapped == NULL) {
		return -ENOMEM;
	}

	memset(mapped, 0, input_size);
	if (data != NULL && size > 0U) {
		memcpy(mapped, data, size);
	}

	ret = NN_UnmapBuffer(model->inputs[index]);
	if (ret != 0) {
		return ret;
	}

	ret = NN_FlushBuffer(model->inputs[index], NN_BUFFER_OPER_FLUSH);
	if (ret != 0) {
		return ret;
	}

	return NN_SetInput(model->network, index, model->inputs[index]);
}

int ameba_npu_inference(struct ameba_npu *model,
						struct ameba_npu_inference_profile *profile)
{
	int ret;

	if (!valid_model(model) || !model->prepared) {
		return -EINVAL;
	}

	ret = NN_RunNetwork(model->network);
	if (ret != 0) {
		return ret;
	}

	if (profile == NULL) {
		return 0;
	}

	memset(profile, 0, sizeof(*profile));

	return NN_QueryNetwork(model->network, NN_NETWORK_PROP_PROFILING, profile);
}

int ameba_npu_get_output_param(const struct ameba_npu *model, uint32_t index,
							   struct ameba_npu_buffer_param *param)
{
	NN_BufferParamTypeDef nn_param;
	int ret;

	if (!valid_model(model) || index >= model->output_count || param == NULL) {
		return -EINVAL;
	}

	ret = NN_QueryOutputParam(model->network, index, &nn_param);
	if (ret != 0) {
		return ret;
	}

	copy_tensor_param_from_nn(&nn_param, param);

	return 0;
}

void *ameba_npu_map_output(struct ameba_npu *model, uint32_t index)
{
	int ret;

	if (!valid_model(model) || index >= model->output_count) {
		return NULL;
	}

	ret = NN_FlushBuffer(model->outputs[index], NN_BUFFER_OPER_INVALIDATE);
	if (ret != 0) {
		LOG_ERR("Failed to invalidate output[%u]: %d", index, ret);
		return NULL;
	}

	return NN_MapBuffer(model->outputs[index]);
}

int ameba_npu_unmap_output(struct ameba_npu *model, uint32_t index)
{
	if (!valid_model(model) || index >= model->output_count) {
		return -EINVAL;
	}

	return NN_UnmapBuffer(model->outputs[index]);
}

int ameba_npu_get_output(struct ameba_npu *model, uint32_t index,
						 void *data, size_t size, size_t *output_size)
{
	void *mapped;
	size_t actual_size;
	size_t copy_size;
	int ret;

	if (!valid_model(model) || index >= model->output_count) {
		return -EINVAL;
	}

	if (data == NULL && size != 0U) {
		return -EINVAL;
	}

	actual_size = NN_GetBufferSize(model->outputs[index]);
	if (output_size != NULL) {
		*output_size = actual_size;
	}

	ret = NN_FlushBuffer(model->outputs[index], NN_BUFFER_OPER_INVALIDATE);
	if (ret != 0) {
		return ret;
	}

	if (data == NULL || size == 0U) {
		return 0;
	}

	mapped = NN_MapBuffer(model->outputs[index]);
	if (mapped == NULL) {
		return -ENOMEM;
	}

	copy_size = MIN(size, actual_size);
	memcpy(data, mapped, copy_size);

	return NN_UnmapBuffer(model->outputs[index]);
}
