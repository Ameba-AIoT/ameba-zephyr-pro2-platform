/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Ameba NPU detection sample
 * ==========================
 *
 * This sample runs either SCRFD face detection or YOLOv4-tiny object
 * detection through the same Zephyr ameba_npu lifecycle. The shell command
 * selects the model at run time:
 *
 *   uart:~$ npu run face
 *   uart:~$ npu run yolo
 *
 * Omitting the model keeps the original behavior and runs face detection.
 * Both models consume RGB888 planar data. SCRFD uses the embedded 576x320
 * wls image, while YOLOv4-tiny uses the embedded 416x416 horses image.
 *
 * Model binaries under src/model_binary are inputs for the NN_MDL flash image
 * packaging flow; they are not linked into the Zephyr application image.
 */

#include <errno.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/drivers/misc/ameba_npu.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/shell/shell.h>
#include <zephyr/sys/atomic.h>

#include <zephyr/drivers/misc/ameba_npu_model_loader.h>
#include <zephyr/drivers/misc/ameba_npu_scrfd_postprocess.h>
#include <zephyr/drivers/misc/ameba_npu_yolov4_tiny_postprocess.h>

LOG_MODULE_REGISTER(npu_sample, LOG_LEVEL_INF);

#define NPU_SAMPLE_INFERENCE_RUNS 1U
#define NPU_SAMPLE_STACK_SIZE     10240
#define NPU_SAMPLE_PRIORITY       5

enum npu_sample_model_type {
	NPU_SAMPLE_MODEL_SCRFD,
	NPU_SAMPLE_MODEL_YOLOV4_TINY,
};

struct npu_sample_model_config {
	enum npu_sample_model_type type;
	const char *command_name;
	const char *display_name;
	const char *flash_filename;
	uint32_t input_width;
	uint32_t input_height;
};

static const struct npu_sample_model_config scrfd_config = {
	.type = NPU_SAMPLE_MODEL_SCRFD,
	.command_name = "face",
	.display_name = "SCRFD face detection",
	.flash_filename = "scrfd.nb",
	.input_width = 576U,
	.input_height = 320U,
};

static const struct npu_sample_model_config yolov4_tiny_config = {
	.type = NPU_SAMPLE_MODEL_YOLOV4_TINY,
	.command_name = "yolo",
	.display_name = "YOLOv4-tiny object detection",
	.flash_filename = "yolov4_tiny.nb",
	.input_width = YOLOV4_TINY_INPUT_WIDTH,
	.input_height = YOLOV4_TINY_INPUT_HEIGHT,
};

K_THREAD_STACK_DEFINE(npu_sample_stack, NPU_SAMPLE_STACK_SIZE);

static struct k_thread npu_sample_thread;
static k_tid_t npu_sample_tid;
static atomic_t npu_sample_running;

/* RGB888 planar test image generated from model_test_input/wls.jpg. */
extern const uint8_t wls_rgb888_planar[];
extern const size_t wls_rgb888_planar_size;
extern const uint32_t wls_rgb888_planar_width;
extern const uint32_t wls_rgb888_planar_height;

/* RGB888 planar test image generated from model_test_input/horses_416x416.jpg. */
extern const uint8_t horses_rgb888_planar[];
extern const size_t horses_rgb888_planar_size;
extern const uint32_t horses_rgb888_planar_width;
extern const uint32_t horses_rgb888_planar_height;

static const struct npu_sample_model_config *find_model_config(const char *name)
{
	if (name == NULL || strcmp(name, scrfd_config.command_name) == 0 ||
		strcmp(name, "scrfd") == 0) {
		return &scrfd_config;
	}

	if (strcmp(name, yolov4_tiny_config.command_name) == 0 ||
		strcmp(name, "yolov4") == 0 ||
		strcmp(name, "yolov4-tiny") == 0) {
		return &yolov4_tiny_config;
	}

	return NULL;
}

static int set_sample_inputs(struct ameba_npu *model,
							 const struct npu_sample_model_config *config)
{
	uint32_t input_count = ameba_npu_get_input_count(model);
	size_t expected_size = (size_t)config->input_width * config->input_height * 3U;
	const uint8_t *input_data;
	size_t input_size;
	uint32_t input_width;
	uint32_t input_height;
	size_t tensor_size;
	int ret;

	if (input_count == 0U) {
		LOG_ERR("Model does not expose any input tensors");
		return -EINVAL;
	}

	switch (config->type) {
	case NPU_SAMPLE_MODEL_SCRFD:
		input_data = wls_rgb888_planar;
		input_size = wls_rgb888_planar_size;
		input_width = wls_rgb888_planar_width;
		input_height = wls_rgb888_planar_height;
		break;
	case NPU_SAMPLE_MODEL_YOLOV4_TINY:
		input_data = horses_rgb888_planar;
		input_size = horses_rgb888_planar_size;
		input_width = horses_rgb888_planar_width;
		input_height = horses_rgb888_planar_height;
		break;
	default:
		return -EINVAL;
	}

	if (input_size != expected_size || input_width != config->input_width ||
		input_height != config->input_height) {
		LOG_ERR("%s test image mismatch: image=%ux%u size=%u expected=%ux%u size=%u",
				config->display_name, input_width, input_height, (uint32_t)input_size,
				config->input_width, config->input_height, (uint32_t)expected_size);
		return -EINVAL;
	}

	tensor_size = ameba_npu_get_input_size(model, 0);
	if (tensor_size != expected_size) {
		LOG_ERR("%s input size mismatch: image=%u tensor=%u",
				config->display_name, (uint32_t)input_size, (uint32_t)tensor_size);
		return -EINVAL;
	}

	LOG_INF("Use %s test input: %ux%u RGB888 planar",
			config->command_name, input_width, input_height);

	ret = ameba_npu_set_input(model, 0, input_data, input_size);
	if (ret != 0) {
		LOG_ERR("Failed to set input[0]: %d", ret);
		return ret;
	}

	for (uint32_t i = 1; i < input_count; i++) {
		LOG_INF("Set input[%u] to zero data, size=%u", i,
				(uint32_t)ameba_npu_get_input_size(model, i));
		ret = ameba_npu_set_input(model, i, NULL, 0);
		if (ret != 0) {
			LOG_ERR("Failed to set input[%u]: %d", i, ret);
			return ret;
		}
	}

	return 0;
}

static int run_inference(struct ameba_npu *model)
{
	struct ameba_npu_inference_profile profile;
	int ret;

	for (uint32_t i = 0; i < NPU_SAMPLE_INFERENCE_RUNS; i++) {
		ret = ameba_npu_inference(model, &profile);
		if (ret != 0) {
			LOG_ERR("Inference %u/%u failed: %d", i + 1U,
					NPU_SAMPLE_INFERENCE_RUNS, ret);
			return ret;
		}

		LOG_INF("Inference %u/%u time: %u us, total_cycle=%u", i + 1U,
				NPU_SAMPLE_INFERENCE_RUNS, profile.inference_time,
				profile.total_cycle);
	}

	return 0;
}

static int run_scrfd_postprocess(struct ameba_npu *model)
{
	struct scrfd_tensor tensors[SCRFD_OUTPUT_COUNT];
	uint32_t mapped_count = 0U;
	int ret = 0;
	int cleanup_ret;

	if (ameba_npu_get_output_count(model) < SCRFD_OUTPUT_COUNT) {
		LOG_ERR("SCRFD requires %u outputs, model has %u",
				SCRFD_OUTPUT_COUNT, ameba_npu_get_output_count(model));
		return -EINVAL;
	}

	for (uint32_t i = 0; i < SCRFD_OUTPUT_COUNT; i++) {
		ret = ameba_npu_get_output_param(model, i, &tensors[i].param);
		if (ret != 0) {
			LOG_ERR("Failed to get SCRFD output[%u] param: %d", i, ret);
			goto out_unmap;
		}

		tensors[i].data = ameba_npu_map_output(model, i);
		if (tensors[i].data == NULL) {
			LOG_ERR("Failed to map SCRFD output[%u]", i);
			ret = -EIO;
			goto out_unmap;
		}

		mapped_count++;
		LOG_INF("Output[%u]: format=%d quant=%d size=%u", i,
				tensors[i].param.data_format, tensors[i].param.quant_format,
				(uint32_t)ameba_npu_get_output_size(model, i));
	}

	ret = scrfd_postprocess_and_log(tensors, SCRFD_OUTPUT_COUNT);

out_unmap:
	for (uint32_t i = 0; i < mapped_count; i++) {
		cleanup_ret = ameba_npu_unmap_output(model, i);
		if (cleanup_ret != 0) {
			LOG_ERR("Failed to unmap output[%u]: %d", i, cleanup_ret);
			if (ret == 0) {
				ret = cleanup_ret;
			}
		}
	}

	return ret;
}

static int run_yolov4_tiny_postprocess(struct ameba_npu *model)
{
	struct yolov4_tiny_tensor tensors[YOLOV4_TINY_OUTPUT_COUNT];
	uint32_t mapped_count = 0U;
	int ret = 0;
	int cleanup_ret;

	if (ameba_npu_get_output_count(model) != YOLOV4_TINY_OUTPUT_COUNT) {
		LOG_ERR("YOLOv4-tiny requires %u outputs, model has %u",
				YOLOV4_TINY_OUTPUT_COUNT, ameba_npu_get_output_count(model));
		return -EINVAL;
	}

	for (uint32_t i = 0; i < YOLOV4_TINY_OUTPUT_COUNT; i++) {
		ret = ameba_npu_get_output_param(model, i, &tensors[i].param);
		if (ret != 0) {
			LOG_ERR("Failed to get YOLO output[%u] param: %d", i, ret);
			goto out_unmap;
		}

		tensors[i].size = ameba_npu_get_output_size(model, i);
		tensors[i].data = ameba_npu_map_output(model, i);
		if (tensors[i].data == NULL) {
			LOG_ERR("Failed to map YOLO output[%u]", i);
			ret = -EIO;
			goto out_unmap;
		}

		mapped_count++;
		LOG_INF("Output[%u]: dims=%ux%ux%u format=%d quant=%d size=%u", i,
				tensors[i].param.dim_size[0], tensors[i].param.dim_size[1],
				tensors[i].param.dim_size[2], tensors[i].param.data_format,
				tensors[i].param.quant_format, (uint32_t)tensors[i].size);
	}

	ret = yolov4_tiny_postprocess_and_log(tensors, YOLOV4_TINY_OUTPUT_COUNT);

out_unmap:
	for (uint32_t i = 0; i < mapped_count; i++) {
		cleanup_ret = ameba_npu_unmap_output(model, i);
		if (cleanup_ret != 0) {
			LOG_ERR("Failed to unmap output[%u]: %d", i, cleanup_ret);
			if (ret == 0) {
				ret = cleanup_ret;
			}
		}
	}

	return ret;
}

static int run_postprocess(struct ameba_npu *model,
						   const struct npu_sample_model_config *config)
{
	switch (config->type) {
	case NPU_SAMPLE_MODEL_SCRFD:
		return run_scrfd_postprocess(model);
	case NPU_SAMPLE_MODEL_YOLOV4_TINY:
		return run_yolov4_tiny_postprocess(model);
	default:
		return -EINVAL;
	}
}

static int npu_sample_run(const struct npu_sample_model_config *config)
{
	struct ameba_npu model = {0};
	struct npu_flash_model flash_model = {0};
	int ret;
	int cleanup_ret;

	LOG_INF("Ameba NPU sample on %s", CONFIG_BOARD_TARGET);
	LOG_INF("Selected model: %s (%s)", config->display_name, config->flash_filename);
	LOG_INF("NPU runtime version: 0x%08x", ameba_npu_get_version());

	ret = ameba_npu_init();
	if (ret != 0) {
		LOG_ERR("Failed to initialize NPU: %d", ret);
		return ret;
	}

	ret = npu_flash_model_load(config->flash_filename, &flash_model);
	if (ret != 0) {
		LOG_ERR("Failed to load %s from flash: %d", config->flash_filename, ret);
		goto out_deinit;
	}

	ret = ameba_npu_load(&model, flash_model.data, flash_model.size);
	if (ret != 0) {
		LOG_ERR("Failed to load NPU model: %d", ret);
		goto out_flash_model;
	}

	LOG_INF("Model loaded: inputs=%u outputs=%u",
			ameba_npu_get_input_count(&model), ameba_npu_get_output_count(&model));

	ret = set_sample_inputs(&model, config);
	if (ret == 0) {
		ret = run_inference(&model);
	}
	if (ret == 0) {
		ret = run_postprocess(&model, config);
	}

	cleanup_ret = ameba_npu_unload(&model);
	if (cleanup_ret != 0) {
		LOG_ERR("Failed to unload NPU model: %d", cleanup_ret);
		if (ret == 0) {
			ret = cleanup_ret;
		}
	}

out_flash_model:
	npu_flash_model_unload(&flash_model);

out_deinit:
	cleanup_ret = ameba_npu_deinit();
	if (cleanup_ret != 0) {
		LOG_ERR("Failed to deinitialize NPU: %d", cleanup_ret);
		if (ret == 0) {
			ret = cleanup_ret;
		}
	}

	return ret;
}

static void npu_sample_task(void *arg1, void *arg2, void *arg3)
{
	const struct npu_sample_model_config *config = arg1;
	int ret;

	ARG_UNUSED(arg2);
	ARG_UNUSED(arg3);

	ret = npu_sample_run(config);
	if (ret != 0) {
		LOG_ERR("%s sample failed: %d", config->display_name, ret);
	} else {
		LOG_INF("%s sample completed", config->display_name);
	}

	atomic_clear(&npu_sample_running);
}

static int cmd_npu_run(const struct shell *shell, size_t argc, char **argv)
{
	const struct npu_sample_model_config *config =
		find_model_config(argc >= 2U ? argv[1] : NULL);

	if (config == NULL) {
		shell_error(shell, "Unknown model '%s'; use face or yolo", argv[1]);
		return -EINVAL;
	}

	if (!atomic_cas(&npu_sample_running, 0, 1)) {
		shell_print(shell, "NPU sample is already running");
		return -EBUSY;
	}

	if (npu_sample_tid != NULL) {
		(void)k_thread_join(npu_sample_tid, K_FOREVER);
	}

	npu_sample_tid = k_thread_create(&npu_sample_thread, npu_sample_stack,
									 NPU_SAMPLE_STACK_SIZE, npu_sample_task,
									 (void *)config, NULL, NULL,
									 NPU_SAMPLE_PRIORITY, 0, K_NO_WAIT);
	k_thread_name_set(npu_sample_tid, "npu_sample");

	shell_print(shell, "NPU sample started: %s", config->display_name);
	return 0;
}

static int cmd_npu_list(const struct shell *shell, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	shell_print(shell, "%s: %s, input %ux%u, flash file %s",
				scrfd_config.command_name, scrfd_config.display_name, scrfd_config.input_width,
				scrfd_config.input_height, scrfd_config.flash_filename);
	shell_print(shell, "%s: %s, input %ux%u, flash file %s",
				yolov4_tiny_config.command_name, yolov4_tiny_config.display_name,
				yolov4_tiny_config.input_width,
				yolov4_tiny_config.input_height, yolov4_tiny_config.flash_filename);

	return 0;
}

SHELL_STATIC_SUBCMD_SET_CREATE(sub_npu,
							   SHELL_CMD_ARG(run, NULL,
									   "Run detection [face|yolo] (default: face)",
									   cmd_npu_run, 1, 1),
							   SHELL_CMD(list, NULL, "List supported NPU models", cmd_npu_list),
							   SHELL_SUBCMD_SET_END);

SHELL_CMD_REGISTER(npu, &sub_npu, "NPU detection commands", NULL);

int main(void)
{
	LOG_INF("Ameba NPU detection sample on %s", CONFIG_BOARD_TARGET);
	LOG_INF("Run 'npu run face' or 'npu run yolo' to start inference");

	return 0;
}
