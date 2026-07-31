/*
 * Copyright (c) 2024 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/video.h>
#include <zephyr/sys/printk.h>
#include <zephyr/logging/log.h>
#include <zephyr/shell/shell.h>
#include <strings.h>
#include <zephyr/drivers/video-controls.h>

#include <zephyr/storage/disk_access.h>
#include <zephyr/fs/fs.h>
#include <ff.h>
#include <video_api.h>
#include <eip_api.h>
#include <md_api.h>
#include <avcodec.h>

/* ===== EIP Motion Detection status ===== */
#define EIP_MD_STOP  0
#define EIP_MD_START 1
#define EIP_MD_SET_STOP 2

static const md_config_t eip_md_default_config = {
	.adapt_mode = 0,
	.adapt_level = 1.1,
	.adapt_step = 30,
	.adapt_thr_max = 10,
	.bg_mode = 0,
	.detect_interval = 1,
	.his_resolution = 5,
	.his_threshold = 50,
	.his_step = 200,
	.md_obj_sensitivity = 90,
	.md_time_filter_interval = 3,
	.md_trigger_block_threshold = 0,
	.block_base_thr = 1,
	.block_lum_thr = 3,
};

#define VIDEO_CLOSE 0
#define VIDEO_OPEN  1

LOG_MODULE_REGISTER(video_capture, LOG_LEVEL_DBG);

#define MOUNT_POINT      "/SD:"
#define RECORD_FRAME_LEN 300

#define STACKSIZE   16*1024
#define PRIORITY    5
#define NUM_THREADS 3

#define VIDEO_BUF_SIZE (256 * 1024)

K_THREAD_STACK_ARRAY_DEFINE(thread_stacks, NUM_THREADS, STACKSIZE);

struct thread_context {
	struct k_thread thread_data;
	k_tid_t tid;
	struct k_sem my_sem;
	struct k_sem *other_sem;
	const char *name;
	k_thread_stack_t *stack_ptr;
	int video_status;
};

#define EIP_COL 32
#define EIP_ROW 32
/* Per-channel EIP state (MD is one feature of EIP) */
struct eip_context {
	md_context_t *motion_detect_ctx;      /* libmd allocate context */
	md_config_t md_config;                 /* MD config (sensitivity, threshold, etc.) */
	md_result_t md_result;                 /* MD result (motion count, positions) */
	eip_param_t eip_params;                /* EIP params (image size, grid) */
	eip_Y_data_t Y_data;                   /* Y data for MD processing */
	eip_statis_infor_t statis_info;        /* Statistic info (debug) */
	eip_ae_stable_t ae_stable;             /* AE stable check */
	unsigned long md_time0;                /* FPS timing */
	int en_ae_stable;                      /* 1: check AE stable before MD */
	volatile int md_status;                /* EIP_MD_STOP / START / SET_STOP */
};

struct video_sample_context {
	struct thread_context vthread[NUM_THREADS];
	struct eip_context eip;
	FATFS fat_fs;
	int sdcard_init;
	struct fs_file_t file;
	int record_seq; /* sequential index for SD card file naming */
	uint32_t video_sd_buf_pos;
	struct fs_mount_t fat_mount;
	uint32_t video_record_channel;
	volatile int record_pending;

	__aligned(32) uint8_t video_buffer[VIDEO_BUF_SIZE];
};

static struct video_sample_context ctxs;

void mem_dump(const void *addr, size_t len)
{
	const uint8_t *p = (const uint8_t *)addr;
	size_t i;

	for (i = 0; i < len; i++) {
		if (i % 16 == 0) {
			printk("%08x: ", (unsigned int)((uintptr_t)(p + i)));
		}

		printk("%02x ", p[i]);

		if (i % 16 == 15 || i == len - 1) {
			printk("\n");
		}
	}
}

int video_get_channel(const char *dev_name)
{
	int video_index = 0;

	if (sscanf(dev_name, "%*[^0-9]%d", &video_index) == 1) {
		LOG_INF("Video index %d", video_index);
	} else {
		return -EINVAL;
	}
	return video_index;
}

static int video_sd_card_init(void)
{
	int rc = 0;

	if (ctxs.sdcard_init) {
		goto EXIT;
	}

	ctxs.fat_mount.type = FS_FATFS;
	ctxs.fat_mount.mnt_point = MOUNT_POINT;
	ctxs.fat_mount.fs_data = &ctxs.fat_fs;
	ctxs.fat_mount.storage_dev = "SD";

	rc = disk_access_init(ctxs.fat_mount.storage_dev);
	if (rc != 0) {
		LOG_ERR("SD disk init failed (%d)", rc);
		goto EXIT;
	} else {
		LOG_INF("SD disk init (%d)", rc);
	}

	rc = fs_mount(&ctxs.fat_mount);
	if (rc < 0) {
		LOG_ERR("Failed to mount filesystem (%d)", rc);
		disk_access_ioctl(ctxs.fat_mount.storage_dev, DISK_IOCTL_CTRL_DEINIT, NULL);
		goto EXIT;
	}
	fs_file_t_init(&ctxs.file);
	ctxs.sdcard_init = 1;
	LOG_INF("Filesystem mounted at %s", MOUNT_POINT);
EXIT:
	return rc;
}

static void video_sd_write(struct fs_file_t *file, char *buf, uint32_t len)
{
	int offset = 0;

	while (len > 0) {
		if (ctxs.video_sd_buf_pos + len >= VIDEO_BUF_SIZE) {
			memcpy(ctxs.video_buffer + ctxs.video_sd_buf_pos, buf + offset,
				   VIDEO_BUF_SIZE - ctxs.video_sd_buf_pos);
			fs_write(file, ctxs.video_buffer, VIDEO_BUF_SIZE);
			offset += VIDEO_BUF_SIZE - ctxs.video_sd_buf_pos;
			len -= VIDEO_BUF_SIZE - ctxs.video_sd_buf_pos;
			ctxs.video_sd_buf_pos = 0;
		} else {
			memcpy(ctxs.video_buffer + ctxs.video_sd_buf_pos, buf + offset, len);
			ctxs.video_sd_buf_pos = ctxs.video_sd_buf_pos + len;
			len = 0;
		}
	}
}

static void video_sd_flush_buf(struct fs_file_t *file)
{
	if (ctxs.video_sd_buf_pos != 0) {
		fs_write(file, ctxs.video_buffer, ctxs.video_sd_buf_pos);
		ctxs.video_sd_buf_pos = 0;
	}
}

static void eip_init(struct eip_context *eip, int video_channel,
					 const struct video_format *fmt)
{
	eip->motion_detect_ctx = (md_context_t *)malloc(sizeof(md_context_t));
	if (!eip->motion_detect_ctx) {
		LOG_ERR("Failed to allocate MD context for ch%d", video_channel);
		eip->md_status = EIP_MD_STOP;
		return;
	}

	memset(eip->motion_detect_ctx, 0, sizeof(md_context_t));

	eip->eip_params.image_width = fmt->width;
	eip->eip_params.image_height = fmt->height;
	eip->eip_params.eip_row = EIP_ROW;
	eip->eip_params.eip_col = EIP_COL;

	memcpy(&eip->md_config, &eip_md_default_config, sizeof(md_config_t));
	for (int i = 0; i < MD_MASK_ROW * MD_MASK_COL; i++) {
		eip->md_config.md_mask[i] = 1;
	}
	eip->en_ae_stable = 0;
	eip->md_status = EIP_MD_STOP;
	memset(&eip->ae_stable, 0, sizeof(eip_ae_stable_t));
	LOG_INF("EIP context ready for ch%d (%dx%d, grid %dx%d)", video_channel, fmt->width, fmt->height, eip->eip_params.eip_row, eip->eip_params.eip_col);
}

static void eip_process(struct eip_context *eip, int video_channel,
						struct video_buffer *vbuf, const struct video_format *fmt)
{
	/* Handle stop signal */
	if (eip->md_status == EIP_MD_SET_STOP) {
		eip->md_status = EIP_MD_STOP;
		LOG_INF("[EIP] stopped ch%d", video_channel);
	}

	if (eip->md_status != EIP_MD_START || !eip->motion_detect_ctx) {
		return;
	}

	/* AE not stable yet: skip processing, but keep frame count */
	if (eip->en_ae_stable && !eip->ae_stable.stable) {
		eip->ae_stable.stable = eip_check_ae_stable(&eip->ae_stable);
		if (!eip->ae_stable.stable) {
			eip->motion_detect_ctx->count++;
			return;
		}
	}

	if (eip->motion_detect_ctx->count % eip->md_config.detect_interval == 0) {
		eip_gen_Y_data(&eip->eip_params, (unsigned char *)vbuf->buffer, AV_CODEC_ID_NV12, &eip->Y_data);
		eip_gen_statistic_data(&eip->eip_params, &eip->Y_data, &eip->statis_info);
	}

	/* First frame: init background model */
	if (eip->motion_detect_ctx->count == 0) {
		md_initial(eip->motion_detect_ctx, (md_param_t *)&eip->eip_params, &eip->md_config);
		md_initial_bgmodel(eip->motion_detect_ctx, (md_param_t *)&eip->eip_params, &eip->Y_data);
		md_show_config(eip->motion_detect_ctx, (md_param_t *)&eip->eip_params, &eip->md_config);
		LOG_INF("[EIP] background model initialized on ch%d", video_channel);
	}

	/* Run motion detection */
	if (eip->motion_detect_ctx->count % eip->md_config.detect_interval == 0) {
		motion_detect(eip->motion_detect_ctx, (md_param_t *)&eip->eip_params, &eip->md_config, &eip->Y_data, &eip->md_result);

		if (eip->motion_detect_ctx->count >= eip->md_config.detect_interval * 1024) {
			eip->motion_detect_ctx->count = eip->md_config.detect_interval;
		}

		/* Check for motion objects */
		if (eip->md_result.motion_cnt) {
			LOG_INF("[EIP] MOTION ch%d objects=%d", video_channel, eip->md_result.motion_cnt);
			md_pos_t *p = &eip->md_result.md_pos[0];
			LOG_INF("[EIP] obj[0] (%.2f,%.2f)-(%.2f,%.2f)", (double)p->xmin, (double)p->ymin, (double)p->xmax, (double)p->ymax);
		}
	}

	/* FPS every 128 frames */
	if (eip->motion_detect_ctx->count % 128 == 0) {
		unsigned long now = vbuf->timestamp;
		if (eip->md_time0 != 0 && now > eip->md_time0) {
			float fps = 128.0f * 1000.0f / (float)(now - eip->md_time0);
			LOG_INF("[EIP] FPS = %0.2f", (double)fps);
		}
		eip->md_time0 = now;
	}

	eip->motion_detect_ctx->count++;
}

static void video_task(void *param, void *param1, void *param2)
{
	struct video_buffer buffers[16], *vbuf;
	struct video_format fmt;
	struct video_caps caps;
	unsigned int frame = 0;
	int i = 0;
	struct thread_context *ctx = (struct thread_context *)param;
	const char *video_dev_name = (const char *)ctx->name;
	const struct device *video = NULL;
	int video_channel = 0;
	(void)param1;
	(void)param2;

	struct fs_file_t record_file;
	int recording = 0;
	int record_count = 0;

	video = device_get_binding(video_dev_name);
	if (video == NULL) {
		LOG_ERR("Video device %s not found", video_dev_name);
		goto EXIT;
	}

	if (video_get_caps(video, &caps)) {
		LOG_ERR("Unable to retrieve video capabilities");
		goto EXIT;
	}

	LOG_INF("- Capabilities:\n");
	while (caps.format_caps[i].pixelformat) {
		const struct video_format_cap *fcap = &caps.format_caps[i];

		LOG_INF("  %c%c%c%c width [%u; %u; %u] height [%u; %u; %u]\n",
				(char)fcap->pixelformat, (char)(fcap->pixelformat >> 8),
				(char)(fcap->pixelformat >> 16), (char)(fcap->pixelformat >> 24),
				fcap->width_min, fcap->width_max, fcap->width_step, fcap->height_min,
				fcap->height_max, fcap->height_step);
		i++;
	}

	if (video_get_format(video, &fmt)) {
		LOG_ERR("Unable to retrieve video format");
		goto EXIT;
	}

	if (video_set_format(video, &fmt)) {
		LOG_ERR("Unable to set video format");
		goto EXIT;
	}

	LOG_INF("- Default format: %c%c%c%c %ux%u\n", (char)fmt.pixelformat,
			(char)(fmt.pixelformat >> 8), (char)(fmt.pixelformat >> 16),
			(char)(fmt.pixelformat >> 24), fmt.width, fmt.height);

	memset(buffers, 0, sizeof(buffers));
	for (i = 0; i < ARRAY_SIZE(buffers); i++) {
		video_enqueue(video, &buffers[i]);
	}

	video_channel = video_get_channel(ctx->name);

	if (video_stream_start(video, VIDEO_BUF_TYPE_OUTPUT)) {
		LOG_ERR("Unable to start capture (interface)");
		goto EXIT;
	}

	LOG_INF("Capture started\n");

	ctx->video_status = VIDEO_OPEN;

	if (video_channel == 2) {
		eip_init(&ctxs.eip, video_channel, &fmt);
	}

	while (1) {
		int err;

		err = video_dequeue(video, &vbuf, K_MSEC(500));
		if (err) {
			LOG_ERR("Unable to dequeue video buf");
			goto EXIT;
		}
		frame++;
		if (frame % 30 == 0) {
			if (!ctxs.sdcard_init || (ctxs.video_record_channel != video_channel)) {
				LOG_INF("\rGot frame %u! size: %u; timestamp %u ms", frame, vbuf->size, vbuf->timestamp);
				mem_dump(vbuf->buffer, 16);
			}
		}

		if (video_channel == 2) {
			eip_process(&ctxs.eip, video_channel, vbuf, &fmt);
		}

		/* Record trigger: NV12 single frame or H264/H265 multi-frame */
		if (ctxs.record_pending && ctxs.sdcard_init && ctxs.video_record_channel == video_channel) {

			if (fmt.pixelformat == VIDEO_PIX_FMT_NV12) {
				/* NV12: single frame snapshot */
				char snap_filename[32];
				struct fs_file_t snap_file;

				fs_file_t_init(&snap_file);
				snprintf(snap_filename, sizeof(snap_filename), "/SD:/snap_%d.nv12", ctxs.record_seq);
				if (fs_open(&snap_file, snap_filename, FS_O_CREATE | FS_O_WRITE) == 0) {
					fs_write(&snap_file, vbuf->buffer, vbuf->size);
					fs_close(&snap_file);
					LOG_INF("Snapshot saved: %s (%u bytes)", snap_filename, vbuf->size);
					ctxs.record_seq++;
				} else {
					LOG_ERR("Failed to open snapshot file");
				}
				ctxs.record_pending = 0;

				/* Deinit SD after NV12 snapshot */
				fs_unmount(&ctxs.fat_mount);
				disk_access_ioctl(ctxs.fat_mount.storage_dev, DISK_IOCTL_CTRL_DEINIT, NULL);
				ctxs.sdcard_init = 0;
				LOG_INF("SD unmounted after snapshot\n");

			} else {
				/* H264/H265: start multi-frame recording */
				const char *ext = (fmt.pixelformat == VIDEO_PIX_FMT_H264) ? "h264" : "h265";
				char rec_filename[32];

				fs_file_t_init(&record_file);
				snprintf(rec_filename, sizeof(rec_filename), "/SD:/video_%d.%s", ctxs.record_seq, ext);
				if (fs_open(&record_file, rec_filename, FS_O_CREATE | FS_O_WRITE) == 0) {
					recording = 1;
					record_count = 1;
					ctxs.record_pending = 0;
					ctxs.video_sd_buf_pos = 0;
					video_sd_write(&record_file, vbuf->buffer, vbuf->size);
					LOG_INF("Recording started: %s (%u/%u)\n", rec_filename, record_count, RECORD_FRAME_LEN);
				} else {
					LOG_ERR("Failed to open recording file");
					ctxs.record_pending = 0;
				}
			}
		}

		/* H264/H265: continue writing frames while recording */
		if (recording) {
			if (record_count < RECORD_FRAME_LEN) {
				video_sd_write(&record_file, vbuf->buffer, vbuf->size);
				record_count++;
				if (record_count % 30 == 0) {
					LOG_INF("Recording frame %u/%u\n", record_count, RECORD_FRAME_LEN);
				}
			}
			if (record_count >= RECORD_FRAME_LEN) {
				video_sd_flush_buf(&record_file);
				fs_close(&record_file);
				recording = 0;
				ctxs.record_seq++;
				/* Deinit SD */
				fs_unmount(&ctxs.fat_mount);
				disk_access_ioctl(ctxs.fat_mount.storage_dev, DISK_IOCTL_CTRL_DEINIT, NULL);
				ctxs.sdcard_init = 0;
				LOG_INF("Recording complete (%u frames), SD unmounted\n",
						RECORD_FRAME_LEN);
			}
		}
		err = video_enqueue(video, vbuf);
		if (err) {
			LOG_ERR("Unable to requeue video buf");
			goto EXIT;
		}
	}
EXIT:
	if (recording) {
		video_sd_flush_buf(&record_file);
		fs_close(&record_file);
		LOG_INF("Recording file closed (task exit)\r\n");
	}
	LOG_INF("The video task is closed\r\n");
	ctx->video_status = VIDEO_CLOSE;
}

static void video_thread_init(const char *video_dev_name)
{
	int video_index = 0;

	if (strcasecmp(video_dev_name, "video_0") == 0) {
		video_index = 0;
	} else if (strcasecmp(video_dev_name, "video_1") == 0) {
		video_index = 1;
	} else if (strcasecmp(video_dev_name, "video_2") == 0) {
		video_index = 2;
	}

	ctxs.vthread[video_index].tid = k_thread_create(
										&ctxs.vthread[video_index].thread_data, thread_stacks[video_index], STACKSIZE,
										video_task, &ctxs.vthread[video_index], NULL, NULL, PRIORITY, 0, K_NO_WAIT);
}

int main(void)
{
	static const char *const video_names[NUM_THREADS] = {"video_0", "video_1", "video_2"};

	LOG_INF("amebapro2 video example\r\n");
	LOG_INF("Enter the video start video_0/video_1/video_2 to run the video\r\n");
	LOG_INF("Enter the video stop video_0/video_1/video_2 to stop the video\r\n");

	for (int i = 0; i < NUM_THREADS; i++) {
		ctxs.vthread[i].name = video_names[i];
	}

	return 0;
}

static bool is_valid_video_device(const char *dev_name)
{
	return (strcmp(dev_name, "video_0") == 0) || (strcmp(dev_name, "video_1") == 0) || (strcmp(dev_name, "video_2") == 0);
}

static int cmd_video_start(const struct shell *shell, size_t argc, char **argv)
{
	const char *dev_name = argv[1];
	int video_index = 0;

	if (argc != 2) {
		shell_print(shell, "Usage: video start <device_name>, device name should be "
					"video_0, video_1 or video_2");
		return -EINVAL;
	}
	LOG_INF("dev_name %s", dev_name);

	if (!is_valid_video_device(dev_name)) {
		shell_error(shell, "Invalid device name '%s'. Allowed: video_0, video_1, video_2", dev_name);
		return -EINVAL;
	}

	video_index = video_get_channel(dev_name);

	if (ctxs.vthread[video_index].video_status == VIDEO_CLOSE) {
		video_thread_init(dev_name);
		shell_print(shell, "Video started");
	} else {
		shell_print(shell, "Video is open status\r\n");
	}
	return 0;
}

static int cmd_video_stop(const struct shell *shell, size_t argc, char **argv)
{
	const struct device *video = NULL;
	const char *dev_name = NULL;
	int video_index = 0;

	if (argc != 2) {
		shell_print(shell, "Usage: video stop <device_name> video_0~video_2");
		return -EINVAL;
	}

	dev_name = argv[1];

	if (!is_valid_video_device(dev_name)) {
		shell_error(shell, "Invalid device name '%s'. Allowed: video_0, video_1, video_2", dev_name);
		return -EINVAL;
	}

	video_index = video_get_channel(dev_name);

	LOG_INF("Video index %d", video_index);

	video = device_get_binding(dev_name);

	if (video == NULL) {
		LOG_ERR("Video device %s not found", dev_name);
		return 0;
	}

	if (ctxs.vthread[video_index].video_status == VIDEO_OPEN) {
		video_stream_stop(video, VIDEO_BUF_TYPE_OUTPUT);
		shell_print(shell, "Video stopped");
	} else {
		shell_print(shell, "Video is close status\r\n");
	}
	return 0;
}

static int cmd_video_ctrl(const struct shell *shell, size_t argc, char **argv)
{
	int ret = 0;
	const struct device *video = NULL;
	const char *subcmd;
	int val;
	const char *dev_name;
	int video_index;
	struct video_control control;

	if (argc != 4) {
		shell_error(shell, "Usage: video cmd <operation> <value> video_0/video_1/video_2");
		return -EINVAL;
	}

	subcmd = argv[1];
	val = strtol(argv[2], NULL, 0);
	dev_name = argv[3];

	if (!is_valid_video_device(dev_name)) {
		shell_error(shell, "Invalid device name '%s'. Allowed: video_0, video_1, video_2", dev_name);
		return -EINVAL;
	}

	video_index = video_get_channel(dev_name);

	video = device_get_binding(dev_name);

	if (video == NULL) {
		LOG_ERR("Video device %s not found", dev_name);
		return 0;
	}

	if (ctxs.vthread[video_index].video_status != VIDEO_OPEN) {
		shell_print(shell, "The video is not open\r\n");
		return -EINVAL;
	}

	if (val < 0) {
		shell_error(shell, "Invalid value: %s", argv[2]);
		return -EINVAL;
	}

	if (strcasecmp(subcmd, "gop") == 0) {
		shell_print(shell, "GOP command received with value: %d (0x%x)", val, val);
		control.id = VIDEO_CID_VENDOR_GOP;
		control.val = val;
		ret = video_set_ctrl(video, &control);
	} else if (strcasecmp(subcmd, "fps") == 0) {
		control.id = VIDEO_CID_VENDOR_FPS;
		control.val = val;
		shell_print(shell, "FPS command received with value: %d (0x%x)", val, val);
		ret = video_set_ctrl(video, &control);
	} else if (strcasecmp(subcmd, "ispfps") == 0) {
		control.id = VIDEO_CID_VENDOR_ISPFPS;
		control.val = val;
		shell_print(shell, "ISPFPS command received with value: %d (0x%x)", val, val);
		ret = video_set_ctrl(video, &control);
	} else if (strcasecmp(subcmd, "bps") == 0) {
		control.id = VIDEO_CID_VENDOR_BPS;
		control.val = val;
		shell_print(shell, "BSP command received with value: %d (0x%x)", val, val);
		ret = video_set_ctrl(video, &control);
	} else if (strcasecmp(subcmd, "forcei") == 0) {
		control.id = VIDEO_CID_VENDOR_FORCE_IFRAME;
		control.val = val;
		shell_print(shell, "forcei command received with value: %d (0x%x)", val, val);
		ret = video_set_ctrl(video, &control);
	} else {
		shell_error(shell, "Unknown cmd: %s", subcmd);
		return -EINVAL;
	}
	return 0;
}

static int cmd_video_record_init(const struct shell *shell, size_t argc, char **argv)
{
	if (argc == 2) {
		const char *dev_name = argv[1];

		if (!is_valid_video_device(dev_name)) {
			shell_error(shell, "Invalid device name '%s'. Allowed: video_0, video_1, video_2",
						dev_name);
			return -EINVAL;
		}
		ctxs.video_record_channel = video_get_channel(dev_name);
	}

	video_sd_card_init();
	shell_print(shell, "Enable the sdcard, record channel video_%u", ctxs.video_record_channel);
	ctxs.record_pending = 1;
	return 0;
}

static int cmd_video_md(const struct shell *shell, size_t argc, char **argv)
{
	int video_index = 0;
	const char *dev_name = NULL;
	const char *subcmd = NULL;

	if (argc < 3) {
		shell_print(shell, "Usage: video md <start|stop|sensitivity|ae_stable|status> [value] <video_N>");
		return -EINVAL;
	}

	subcmd = argv[1];
	dev_name = argv[argc - 1];

	if (!is_valid_video_device(dev_name)) {
		shell_error(shell, "Invalid device name '%s'. Allowed: video_0, video_1, video_2", dev_name);
		return -EINVAL;
	}

	video_index = video_get_channel(dev_name);

	if (video_index != 2) {
		shell_print(shell, "MD is only available on video_2 (NV12)");
		return 0;
	}

	if (ctxs.vthread[video_index].video_status != VIDEO_OPEN) {
		shell_print(shell, "video_2 is not running. Start video_2 first.");
		return -EINVAL;
	}

	struct eip_context *eip = &ctxs.eip;

	if (strcmp(subcmd, "start") == 0) {
		if (!eip->motion_detect_ctx) {
			shell_print(shell, "MD context not available");
			return -EINVAL;
		}
		eip->md_status = EIP_MD_START;
		shell_print(shell, "MD started on video_2");
	} else if (strcmp(subcmd, "stop") == 0) {
		eip->md_status = EIP_MD_SET_STOP;
		shell_print(shell, "MD stop requested on video_2");
	} else if (strcmp(subcmd, "sensitivity") == 0) {
		if (argc < 4) {
			shell_print(shell, "Usage: video md sensitivity <0-100> video_2");
			return -EINVAL;
		}
		int val = strtol(argv[2], NULL, 0);
		if (val < 0 || val > 100) {
			shell_error(shell, "Sensitivity must be 0-100");
			return -EINVAL;
		}
		eip->md_config.md_obj_sensitivity = val;
		shell_print(shell, "MD sensitivity set to %d", val);
	} else if (strcmp(subcmd, "ae_stable") == 0) {
		if (argc < 4) {
			shell_print(shell, "Usage: video md ae_stable <0|1> video_2");
			return -EINVAL;
		}
		int val = strtol(argv[2], NULL, 0);
		eip->en_ae_stable = val ? 1 : 0;
		if (eip->en_ae_stable) {
			eip->ae_stable.stable = 0;
			eip->ae_stable.last_ae_etgain = 0;
			eip->ae_stable.timestamp = 0;
		}
		shell_print(shell, "MD AE stable check %s", eip->en_ae_stable ? "enabled" : "disabled");
	} else if (strcmp(subcmd, "status") == 0) {
		shell_print(shell, "--- MD status on video_2 ---");
		shell_print(shell, "  status: %s", eip->md_status == EIP_MD_START ? "START" :
					eip->md_status == EIP_MD_STOP ? "STOP" : "STOPPING");
		shell_print(shell, "  sensitivity: %d", eip->md_config.md_obj_sensitivity);
		shell_print(shell, "  AE stable check: %s", eip->en_ae_stable ? "on" : "off");
		shell_print(shell, "  detect interval: %d", eip->md_config.detect_interval);
		shell_print(shell, "  trigger threshold: %d", eip->md_config.md_trigger_block_threshold);
		if (eip->motion_detect_ctx) {
			shell_print(shell, "  frame count: %d", eip->motion_detect_ctx->count);
			shell_print(shell, "  trigger blocks: %d", eip->motion_detect_ctx->md_trigger_block);
		}
	} else {
		shell_error(shell, "Unknown MD subcmd: %s", subcmd);
		return -EINVAL;
	}
	return 0;
}

SHELL_STATIC_SUBCMD_SET_CREATE(sub_video, SHELL_CMD(start, NULL, "Start video", cmd_video_start),
							   SHELL_CMD(stop, NULL, "Stop video", cmd_video_stop),
							   SHELL_CMD(cmd, NULL, "Video cmd", cmd_video_ctrl),
							   SHELL_CMD(record, NULL, "Record video [video_0|video_1|video_2]", cmd_video_record_init),
							   SHELL_CMD(md, NULL, "Motion detection (video_2 only) <start|stop|sensitivity|ae_stable|status>", cmd_video_md),
							   SHELL_SUBCMD_SET_END);

SHELL_CMD_REGISTER(video, &sub_video, "Video commands", NULL);
