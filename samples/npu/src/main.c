/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Ameba NPU Sample — SCRFD Face Detection
 * ========================================
 * This sample demonstrates how to run a neural network model on the
 * AmebaPro2 NPU using the Zephyr ameba_npu driver. The model used is
 * SCRFD-500M, a lightweight face detection network that accepts a
 * 576x320 RGB888 planar image and outputs bounding boxes with landmarks.
 *
 * Source layout
 * -------------
 *   src/
 *     main.c                              <- you are here; NPU lifecycle and orchestration
 *     model_loader/
 *       model_flash_loader.c/.h          <- loads an NBG model from the NN_MDL flash partition
 *     model_tensor_process/
 *       scrfd_postprocess.c/.h           <- SCRFD output decoding, NMS, result logging
 *     model_test_input/
 *       wls_rgb888_planar.c              <- test image embedded as a C array
 *       wls.jpg                          <- source JPEG (reference only, not compiled)
 *     model_binary/
 *       scrfd_500m_bnkps_576x320_u8.nb  <- NBG model (embedded build; see CMakeLists.txt)
 *
 * Adapting this sample to your own model
 * ---------------------------------------
 *  1. Flash partition  — pack your .nb file into the NN_MDL partition and
 *     update the filename passed to npu_flash_model_load().
 *  2. Input tensor     — replace wls_rgb888_planar with your own image data
 *     in set_sample_inputs(). Check ameba_npu_get_input_size() to confirm
 *     the expected byte count matches your image.
 *  3. Post-processing  — replace run_scrfd_postprocess() and the files
 *     under model_tensor_process/ with your own decoder.
 *     Use ameba_npu_get_output_param() to retrieve quantization metadata,
 *     and ameba_npu_map_output() / ameba_npu_unmap_output() for zero-copy
 *     access to the raw output buffers.
 *
 * Running the sample
 * ------------------
 * The inference is triggered via the Zephyr shell to allow repeated runs
 * without reflashing:
 *
 *   uart:~$ npu run
 *
 * Inference runs in a dedicated thread (NPU_SAMPLE_PRIORITY) so the shell
 * stays responsive. A second 'npu run' while inference is active returns
 * -EBUSY.
 */

#include <errno.h>
#include <stdint.h>

#include <zephyr/drivers/misc/ameba_npu.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/shell/shell.h>
#include <zephyr/sys/atomic.h>

#include "model_loader/model_flash_loader.h"
#include "model_tensor_process/scrfd_postprocess.h"

LOG_MODULE_REGISTER(npu_sample, LOG_LEVEL_INF);

/*
 * Number of consecutive inference runs per invocation. Increase to
 * measure average latency across multiple passes.
 */
#define NPU_SAMPLE_INFERENCE_RUNS 1

/* Inference thread configuration. */
#define NPU_SAMPLE_STACK_SIZE 10240
#define NPU_SAMPLE_PRIORITY   5

K_THREAD_STACK_DEFINE(npu_sample_stack, NPU_SAMPLE_STACK_SIZE);

static struct k_thread npu_sample_thread;
static k_tid_t npu_sample_tid;
static atomic_t npu_sample_running;

/*
 * Test input image — RGB888 planar layout (all R pixels, then all G, then
 * all B), 576x320. Generated from wls.jpg by model_test_input/wls_rgb888_planar.c.
 *
 * To use a different image, replace the declarations below and the
 * corresponding source file in model_test_input/.
 */
extern const uint8_t wls_rgb888_planar[];
extern const size_t wls_rgb888_planar_size;
extern const uint32_t wls_rgb888_planar_width;
extern const uint32_t wls_rgb888_planar_height;

/*
 * set_sample_inputs - copy the test image into the model's input tensor(s).
 *
 * Input[0] receives the test image. Any additional inputs (uncommon for
 * SCRFD but possible in other models) are zeroed automatically.
 *
 * Adapt this function when using a different model or a live image source
 * such as the video driver. The key API call is ameba_npu_set_input(),
 * which handles DMA buffer mapping and D-cache flushing internally.
 */
static int set_sample_inputs(struct ameba_npu *model)
{
	uint32_t input_count = ameba_npu_get_input_count(model);
	int ret;

	if (input_count == 0U) {
		LOG_ERR("Model does not expose any input tensors");
		return -EINVAL;
	}

	/* Verify the test image matches the model's expected input size. */
	if (wls_rgb888_planar_size != ameba_npu_get_input_size(model, 0)) {
		LOG_ERR("wls input size mismatch: image=%u tensor=%u",
				(uint32_t)wls_rgb888_planar_size,
				(uint32_t)ameba_npu_get_input_size(model, 0));
		return -EINVAL;
	}

	LOG_INF("Set input[0] to wls RGB888 planar image: %ux%u, size=%u",
			wls_rgb888_planar_width, wls_rgb888_planar_height,
			(uint32_t)wls_rgb888_planar_size);

	ret = ameba_npu_set_input(model, 0, wls_rgb888_planar, wls_rgb888_planar_size);
	if (ret != 0) {
		LOG_ERR("Failed to set input[0]: %d", ret);
		return ret;
	}

	/* Zero any extra inputs the model may expose. */
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

/*
 * run_inference - trigger NPU execution and log timing.
 *
 * ameba_npu_inference() is synchronous: it blocks until the hardware
 * finishes. The profile struct captures wall-clock time (microseconds)
 * and total NPU cycle count — useful for benchmarking.
 *
 * For repeated benchmarking, increase NPU_SAMPLE_INFERENCE_RUNS.
 */
static int run_inference(struct ameba_npu *model)
{
	struct ameba_npu_inference_profile profile;
	int ret;

	for (uint32_t i = 0; i < NPU_SAMPLE_INFERENCE_RUNS; i++) {
		ret = ameba_npu_inference(model, &profile);
		if (ret != 0) {
			LOG_ERR("Inference %u/%u failed: %d", i + 1,
					NPU_SAMPLE_INFERENCE_RUNS, ret);
			return ret;
		}

		LOG_INF("Inference %u/%u time: %u us, total_cycle=%u", i + 1,
				NPU_SAMPLE_INFERENCE_RUNS, profile.inference_time,
				profile.total_cycle);
	}

	return 0;
}

/*
 * run_scrfd_postprocess - map output tensors and run SCRFD post-processing.
 *
 * This function owns all NPU output buffer lifecycle:
 *   1. ameba_npu_get_output_param() — retrieve quantization metadata so
 *      the decoder knows how to dequantize each output.
 *   2. ameba_npu_map_output()       — obtain a zero-copy pointer to the
 *      output buffer (also invalidates D-cache automatically).
 *   3. scrfd_postprocess_and_log()  — decode boxes/landmarks, run NMS,
 *      and print detected faces. This call does not touch any NPU APIs.
 *   4. ameba_npu_unmap_output()     — release the buffer mapping.
 *
 * When replacing SCRFD with a different network:
 *   - Update SCRFD_OUTPUT_COUNT in model_tensor_process/scrfd_postprocess.h.
 *   - Replace scrfd_postprocess_and_log() with your own decoder that
 *     accepts an array of scrfd_tensor (data pointer + buffer param).
 */
static int run_scrfd_postprocess(struct ameba_npu *model)
{
	struct scrfd_tensor tensors[SCRFD_OUTPUT_COUNT];
	uint32_t mapped_count = 0;
	int ret = 0;
	int cleanup_ret;

	if (ameba_npu_get_output_count(model) < SCRFD_OUTPUT_COUNT) {
		LOG_ERR("SCRFD requires %u outputs, model has %u",
				SCRFD_OUTPUT_COUNT, ameba_npu_get_output_count(model));
		return -EINVAL;
	}

	for (uint32_t i = 0; i < SCRFD_OUTPUT_COUNT; i++) {
		struct ameba_npu_buffer_param param;
		void *data;

		/* Query quantization metadata for this output. */
		ret = ameba_npu_get_output_param(model, i, &param);
		if (ret != 0) {
			LOG_ERR("Failed to get output[%u] param: %d", i, ret);
			goto out_unmap;
		}

		/* Map the output buffer for zero-copy access. */
		data = ameba_npu_map_output(model, i);
		if (data == NULL) {
			LOG_ERR("Failed to map output[%u]", i);
			ret = -EIO;
			goto out_unmap;
		}

		mapped_count++;
		tensors[i].data = data;
		tensors[i].param = param;
		LOG_INF("Output[%u]: format=%d quant=%d size=%u", i,
				param.data_format, param.quant_format,
				(uint32_t)ameba_npu_get_output_size(model, i));
	}

	/* Hand off to the post-processor; no NPU API calls from here on. */
	ret = scrfd_postprocess_and_log(tensors, SCRFD_OUTPUT_COUNT);

out_unmap:
	/* Always unmap every buffer that was successfully mapped. */
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

/*
 * npu_sample_run - top-level inference flow.
 *
 * Implements the full model lifecycle in one function:
 *
 *   ameba_npu_init()
 *     └─ npu_flash_model_load()   load NBG binary from NN_MDL partition
 *         └─ ameba_npu_load()     create network + allocate I/O buffers
 *             ├─ set_sample_inputs()     fill input tensor(s)
 *             ├─ run_inference()         execute on NPU hardware
 *             └─ run_scrfd_postprocess() decode outputs + log results
 *         ameba_npu_unload()      free network and I/O buffers
 *   npu_flash_model_unload()      free flash model copy
 *   ameba_npu_deinit()
 *
 * Resources are released in reverse order even on partial failure;
 * cleanup errors are reported but do not override the original error code.
 */
static int npu_sample_run(void)
{
	struct ameba_npu model;
	struct npu_flash_model flash_model;
	int ret;
	int cleanup_ret;

	LOG_INF("Ameba NPU sample on %s", CONFIG_BOARD_TARGET);
	LOG_INF("NPU runtime version: 0x%08x", ameba_npu_get_version());

	ret = ameba_npu_init();
	if (ret != 0) {
		LOG_ERR("Failed to initialize NPU: %d", ret);
		return ret;
	}

	/*
	 * Load the NBG model from the NN_MDL flash partition by filename.
	 * The file must be packed into the partition at flash time using
	 * the Realtek image tool. See model_loader/model_flash_loader.h
	 * for the partition layout expected by the loader.
	 */
	ret = npu_flash_model_load("scrfd.nb", &flash_model);
	if (ret != 0) {
		LOG_ERR("Failed to load model from flash: %d", ret);
		goto out_deinit;
	}

	/*
	 * Create the NPU network from the loaded NBG data and allocate
	 * DMA-capable input/output buffers. The flash_model buffer must remain
	 * valid until ameba_npu_unload() is called.
	 */
	ret = ameba_npu_load(&model, flash_model.data, flash_model.size);
	if (ret != 0) {
		LOG_ERR("Failed to load NPU model: %d", ret);
		goto out_deinit;
	}

	LOG_INF("Model loaded: inputs=%u outputs=%u",
			ameba_npu_get_input_count(&model),
			ameba_npu_get_output_count(&model));

	ret = set_sample_inputs(&model);
	if (ret != 0) {
		goto out_unload;
	}

	ret = run_inference(&model);
	if (ret != 0) {
		goto out_unload;
	}

	ret = run_scrfd_postprocess(&model);

out_unload:
	cleanup_ret = ameba_npu_unload(&model);
	if (cleanup_ret != 0) {
		LOG_ERR("Failed to unload NPU model: %d", cleanup_ret);
		if (ret == 0) {
			ret = cleanup_ret;
		}
	}

out_deinit:
	npu_flash_model_unload(&flash_model);

	cleanup_ret = ameba_npu_deinit();
	if (cleanup_ret != 0) {
		LOG_ERR("Failed to deinitialize NPU: %d", cleanup_ret);
		if (ret == 0) {
			ret = cleanup_ret;
		}
	}

	return ret;
}

/*
 * npu_sample_task - thread entry point for inference.
 *
 * Wraps npu_sample_run() and clears the running flag when done so
 * a subsequent 'npu run' shell command can start a new inference.
 */
static void npu_sample_task(void *arg1, void *arg2, void *arg3)
{
	int ret;

	ARG_UNUSED(arg1);
	ARG_UNUSED(arg2);
	ARG_UNUSED(arg3);

	ret = npu_sample_run();
	if (ret != 0) {
		LOG_ERR("NPU sample failed: %d", ret);
	} else {
		LOG_INF("NPU sample completed");
	}

	atomic_clear(&npu_sample_running);
}

/*
 * cmd_npu_run - shell handler for 'npu run'.
 *
 * Spawns npu_sample_task in a dedicated thread. Returns -EBUSY if a
 * previous run has not yet finished. The thread is joined (non-blocking)
 * on re-entry to reclaim its resources before creating a new one.
 */
static int cmd_npu_run(const struct shell *shell, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	if (!atomic_cas(&npu_sample_running, 0, 1)) {
		shell_print(shell, "NPU sample is already running");
		return -EBUSY;
	}

	if (npu_sample_tid != NULL) {
		(void)k_thread_join(npu_sample_tid, K_NO_WAIT);
	}

	npu_sample_tid = k_thread_create(&npu_sample_thread, npu_sample_stack,
									 NPU_SAMPLE_STACK_SIZE, npu_sample_task,
									 NULL, NULL, NULL, NPU_SAMPLE_PRIORITY, 0,
									 K_NO_WAIT);
	k_thread_name_set(npu_sample_tid, "npu_sample");

	shell_print(shell, "NPU sample started");

	return 0;
}

SHELL_STATIC_SUBCMD_SET_CREATE(sub_npu,
							   SHELL_CMD(run, NULL, "Run NPU sample", cmd_npu_run),
							   SHELL_SUBCMD_SET_END);

SHELL_CMD_REGISTER(npu, &sub_npu, "NPU commands", NULL);

int main(void)
{
	LOG_INF("Ameba NPU sample on %s", CONFIG_BOARD_TARGET);
	LOG_INF("Run 'npu run' to start inference");

	return 0;
}
