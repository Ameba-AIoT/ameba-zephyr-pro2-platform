/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/drivers/misc/ameba_npu_scrfd_postprocess.h>

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

LOG_MODULE_REGISTER(scrfd_postprocess, LOG_LEVEL_INF);

#define SCRFD_INPUT_WIDTH 576U
#define SCRFD_INPUT_HEIGHT 320U
#define SCRFD_MAX_FACE_CANDIDATES 256U
#define SCRFD_SCORE_THRESHOLD 0.5f
#define SCRFD_NMS_THRESHOLD 0.3f

struct scrfd_box {
	float x;
	float y;
	float w;
	float h;
	float score;
	float landmarks[10];
	int class_idx;
	bool invalid;
};

static struct scrfd_box face_boxes[SCRFD_MAX_FACE_CANDIDATES];
static struct scrfd_box *face_box_ptrs[SCRFD_MAX_FACE_CANDIDATES];
static uint32_t face_box_count;
static bool face_boxes_full;

static float clamp_float(float value, float min_value, float max_value)
{
	if (value < min_value) {
		return min_value;
	}

	if (value > max_value) {
		return max_value;
	}

	return value;
}

static size_t element_size(enum ameba_npu_buffer_format format)
{
	switch (format) {
	case AMEBA_NPU_BUFFER_FORMAT_FP32:
	case AMEBA_NPU_BUFFER_FORMAT_INT32:
	case AMEBA_NPU_BUFFER_FORMAT_UINT32:
		return 4U;
	case AMEBA_NPU_BUFFER_FORMAT_FP16:
	case AMEBA_NPU_BUFFER_FORMAT_BFP16:
	case AMEBA_NPU_BUFFER_FORMAT_INT16:
	case AMEBA_NPU_BUFFER_FORMAT_UINT16:
		return 2U;
	case AMEBA_NPU_BUFFER_FORMAT_UINT8:
	case AMEBA_NPU_BUFFER_FORMAT_INT8:
	case AMEBA_NPU_BUFFER_FORMAT_CHAR:
	case AMEBA_NPU_BUFFER_FORMAT_BOOL8:
	default:
		return 1U;
	}
}

static float dfp_to_float(int32_t value, int32_t fixed_point_pos)
{
	if (fixed_point_pos >= 0) {
		return (float)value / (float)(1 << fixed_point_pos);
	}

	return (float)value * (float)(1 << -fixed_point_pos);
}

static float fp16_to_float(uint16_t value)
{
	uint32_t sign = (uint32_t)(value & 0x8000U) << 16;
	uint32_t exponent = (value >> 10) & 0x1FU;
	uint32_t mantissa = value & 0x03FFU;
	uint32_t out;
	float result;

	if (exponent == 0U) {
		if (mantissa == 0U) {
			out = sign;
		} else {
			exponent = 1U;
			while ((mantissa & 0x0400U) == 0U) {
				mantissa <<= 1;
				exponent--;
			}
			mantissa &= 0x03FFU;
			out = sign | ((exponent + 112U) << 23) | (mantissa << 13);
		}
	} else if (exponent == 0x1FU) {
		out = sign | 0x7F800000U | (mantissa << 13);
	} else {
		out = sign | ((exponent + 112U) << 23) | (mantissa << 13);
	}

	memcpy(&result, &out, sizeof(result));
	return result;
}

static float bfp16_to_float(uint16_t value)
{
	uint32_t out = (uint32_t)value << 16;
	float result;

	memcpy(&result, &out, sizeof(result));
	return result;
}

static int32_t read_signed_value(const void *addr, enum ameba_npu_buffer_format format)
{
	switch (format) {
	case AMEBA_NPU_BUFFER_FORMAT_INT16: {
		int16_t value;

		memcpy(&value, addr, sizeof(value));
		return value;
	}
	case AMEBA_NPU_BUFFER_FORMAT_INT32: {
		int32_t value;

		memcpy(&value, addr, sizeof(value));
		return value;
	}
	case AMEBA_NPU_BUFFER_FORMAT_INT8:
	default:
		return *(const int8_t *)addr;
	}
}

static uint32_t read_unsigned_value(const void *addr, enum ameba_npu_buffer_format format)
{
	switch (format) {
	case AMEBA_NPU_BUFFER_FORMAT_UINT16: {
		uint16_t value;

		memcpy(&value, addr, sizeof(value));
		return value;
	}
	case AMEBA_NPU_BUFFER_FORMAT_UINT32: {
		uint32_t value;

		memcpy(&value, addr, sizeof(value));
		return value;
	}
	case AMEBA_NPU_BUFFER_FORMAT_UINT8:
	default:
		return *(const uint8_t *)addr;
	}
}

static float tensor_value(const struct scrfd_tensor *tensor, uint32_t index)
{
	const struct ameba_npu_buffer_param *param = &tensor->param;
	const uint8_t *addr = (const uint8_t *)tensor->data +
						  ((size_t)index * element_size(param->data_format));

	switch (param->quant_format) {
	case AMEBA_NPU_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT:
		return dfp_to_float(read_signed_value(addr, param->data_format),
							param->quant_data.dfp.fixed_point_pos);
	case AMEBA_NPU_BUFFER_QUANTIZE_TF_ASYMM:
		return ((float)read_unsigned_value(addr, param->data_format) -
				(float)param->quant_data.affine.zero_point) *
			   param->quant_data.affine.scale;
	case AMEBA_NPU_BUFFER_QUANTIZE_NONE:
	default:
		break;
	}

	switch (param->data_format) {
	case AMEBA_NPU_BUFFER_FORMAT_FP32: {
		float value;

		memcpy(&value, addr, sizeof(value));
		return value;
	}
	case AMEBA_NPU_BUFFER_FORMAT_FP16: {
		uint16_t value;

		memcpy(&value, addr, sizeof(value));
		return fp16_to_float(value);
	}
	case AMEBA_NPU_BUFFER_FORMAT_BFP16: {
		uint16_t value;

		memcpy(&value, addr, sizeof(value));
		return bfp16_to_float(value);
	}
	case AMEBA_NPU_BUFFER_FORMAT_INT8:
	case AMEBA_NPU_BUFFER_FORMAT_INT16:
	case AMEBA_NPU_BUFFER_FORMAT_INT32:
		return (float)read_signed_value(addr, param->data_format);
	default:
		return (float)read_unsigned_value(addr, param->data_format);
	}
}

static void add_face_box(float score, uint32_t stride, uint32_t cx, uint32_t cy,
						 uint32_t anchor_index, const struct scrfd_tensor *score_out,
						 const struct scrfd_tensor *bbox_out,
						 const struct scrfd_tensor *kps_out)
{
	struct scrfd_box *box;
	float l = tensor_value(bbox_out, anchor_index * 4U + 0U);
	float t = tensor_value(bbox_out, anchor_index * 4U + 1U);
	float r = tensor_value(bbox_out, anchor_index * 4U + 2U);
	float b = tensor_value(bbox_out, anchor_index * 4U + 3U);
	float x1 = (((float)cx - l) * (float)stride) / (float)SCRFD_INPUT_WIDTH;
	float y1 = (((float)cy - t) * (float)stride) / (float)SCRFD_INPUT_HEIGHT;
	float x2 = (((float)cx + r) * (float)stride) / (float)SCRFD_INPUT_WIDTH;
	float y2 = (((float)cy + b) * (float)stride) / (float)SCRFD_INPUT_HEIGHT;

	ARG_UNUSED(score_out);

	if (face_box_count >= ARRAY_SIZE(face_boxes)) {
		face_boxes_full = true;
		return;
	}

	x1 = clamp_float(x1, 0.0f, 1.0f);
	y1 = clamp_float(y1, 0.0f, 1.0f);
	x2 = clamp_float(x2, 0.0f, 1.0f);
	y2 = clamp_float(y2, 0.0f, 1.0f);

	box = &face_boxes[face_box_count++];
	box->x = x1;
	box->y = y1;
	box->w = MAX(0.0f, x2 - x1);
	box->h = MAX(0.0f, y2 - y1);
	box->score = score;
	box->class_idx = 0;
	box->invalid = false;

	for (uint32_t j = 0; j < 5U; j++) {
		float kps_x = tensor_value(kps_out, anchor_index * 10U + 2U * j);
		float kps_y = tensor_value(kps_out, anchor_index * 10U + 2U * j + 1U);

		box->landmarks[2U * j] =
			(((float)cx + kps_x) * (float)stride) / (float)SCRFD_INPUT_WIDTH;
		box->landmarks[2U * j + 1U] =
			(((float)cy + kps_y) * (float)stride) / (float)SCRFD_INPUT_HEIGHT;
	}
}

static void generate_bboxes_single_stride(uint32_t stride,
		const struct scrfd_tensor *score_out,
		const struct scrfd_tensor *bbox_out,
		const struct scrfd_tensor *kps_out)
{
	const uint32_t num_anchors = 2U;
	uint32_t grid_w = SCRFD_INPUT_WIDTH / stride;
	uint32_t grid_h = SCRFD_INPUT_HEIGHT / stride;

	for (uint32_t cy = 0; cy < grid_h; cy++) {
		for (uint32_t cx = 0; cx < grid_w; cx++) {
			for (uint32_t anchor = 0; anchor < num_anchors; anchor++) {
				uint32_t index = ((cy * grid_w) + cx) * num_anchors + anchor;
				float score = tensor_value(score_out, index);

				if (score > SCRFD_SCORE_THRESHOLD) {
					add_face_box(score, stride, cx, cy, index,
								 score_out, bbox_out, kps_out);
				}
			}
		}
	}
}

static float box_intersection(const struct scrfd_box *a, const struct scrfd_box *b)
{
	float left = MAX(a->x, b->x);
	float right = MIN(a->x + a->w, b->x + b->w);
	float top = MAX(a->y, b->y);
	float bottom = MIN(a->y + a->h, b->y + b->h);
	float width = right - left;
	float height = bottom - top;

	if (width <= 0.0f || height <= 0.0f) {
		return 0.0f;
	}

	return width * height;
}

static float box_iou(const struct scrfd_box *a, const struct scrfd_box *b)
{
	float intersection = box_intersection(a, b);
	float union_area = (a->w * a->h) + (b->w * b->h) - intersection;

	if (union_area <= 0.0f) {
		return 0.0f;
	}

	return intersection / union_area;
}

static int score_compare_desc(const void *pa, const void *pb)
{
	const struct scrfd_box *a = *(const struct scrfd_box * const *)pa;
	const struct scrfd_box *b = *(const struct scrfd_box * const *)pb;

	if (a->score < b->score) {
		return 1;
	}

	if (a->score > b->score) {
		return -1;
	}

	return 0;
}

static void do_nms(void)
{
	for (uint32_t i = 0; i < face_box_count; i++) {
		face_box_ptrs[i] = &face_boxes[i];
	}

	qsort(face_box_ptrs, face_box_count, sizeof(face_box_ptrs[0]), score_compare_desc);

	for (uint32_t i = 0; i < face_box_count; i++) {
		struct scrfd_box *a = face_box_ptrs[i];

		if (a->invalid) {
			continue;
		}

		for (uint32_t j = i + 1U; j < face_box_count; j++) {
			struct scrfd_box *b = face_box_ptrs[j];

			if (!b->invalid && box_iou(a, b) > SCRFD_NMS_THRESHOLD) {
				b->invalid = true;
			}
		}
	}
}

static uint32_t normalized_to_pixel(float value, uint32_t limit)
{
	float scaled = clamp_float(value, 0.0f, 1.0f) * (float)limit;

	return (uint32_t)(scaled + 0.5f);
}

static void log_faces(void)
{
	uint32_t valid_count = 0;

	for (uint32_t i = 0; i < face_box_count; i++) {
		if (!face_boxes[i].invalid) {
			valid_count++;
		}
	}

	LOG_INF("Detected %u face(s)", valid_count);

	for (uint32_t i = 0, out_idx = 0; i < face_box_count; i++) {
		const struct scrfd_box *box = &face_boxes[i];
		uint32_t x1;
		uint32_t y1;
		uint32_t x2;
		uint32_t y2;
		uint32_t score_permille;

		if (box->invalid) {
			continue;
		}

		x1 = normalized_to_pixel(box->x, SCRFD_INPUT_WIDTH);
		y1 = normalized_to_pixel(box->y, SCRFD_INPUT_HEIGHT);
		x2 = normalized_to_pixel(box->x + box->w, SCRFD_INPUT_WIDTH);
		y2 = normalized_to_pixel(box->y + box->h, SCRFD_INPUT_HEIGHT);
		score_permille = (uint32_t)(clamp_float(box->score, 0.0f, 1.0f) *
									1000.0f + 0.5f);

		LOG_INF("Face[%u]: score=%u.%03u x1=%u y1=%u x2=%u y2=%u w=%u h=%u",
				out_idx, score_permille / 1000U, score_permille % 1000U,
				x1, y1, x2, y2, x2 - x1, y2 - y1);
		out_idx++;
	}

	if (face_boxes_full) {
		LOG_WRN("Face candidate buffer full; results may be truncated");
	}
}

int scrfd_postprocess_and_log(const struct scrfd_tensor tensors[], uint32_t count)
{
	if (count < SCRFD_OUTPUT_COUNT) {
		LOG_ERR("SCRFD postprocess expects %u outputs, got %u",
				SCRFD_OUTPUT_COUNT, count);
		return -EINVAL;
	}

	face_box_count = 0;
	face_boxes_full = false;
	memset(face_boxes, 0, sizeof(face_boxes));

	generate_bboxes_single_stride(8U,  &tensors[0], &tensors[3], &tensors[6]);
	generate_bboxes_single_stride(16U, &tensors[1], &tensors[4], &tensors[7]);
	generate_bboxes_single_stride(32U, &tensors[2], &tensors[5], &tensors[8]);

	do_nms();
	log_faces();

	return 0;
}
