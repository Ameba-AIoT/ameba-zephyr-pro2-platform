/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * YOLOv4-tiny decoder adapted from the AmebaPro2 FreeRTOS model_yolo.c
 * implementation. The NPU output tensors use CHW storage with dimensions
 * reported as [grid width, grid height, 3 * (5 + class count)].
 */

#include <zephyr/drivers/misc/ameba_npu_yolov4_tiny_postprocess.h>

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

LOG_MODULE_REGISTER(yolov4_tiny_postprocess, LOG_LEVEL_INF);

#define YOLOV4_TINY_CLASS_COUNT        80U
#define YOLOV4_TINY_ANCHORS_PER_OUTPUT 3U
#define YOLOV4_TINY_MAX_CANDIDATES     256U
#define YOLOV4_TINY_CONFIDENCE_THRESH  0.5f
#define YOLOV4_TINY_NMS_THRESH         0.3f

struct yolo_box {
	float x;
	float y;
	float w;
	float h;
	float score;
	uint32_t class_idx;
	bool invalid;
};

/*
 * Anchor order follows yolov4-tiny.cfg and the FreeRTOS reference:
 * three anchors for the 13x13 output, then three for the 26x26 output.
 */
static const float yolov4_tiny_anchors[6][2] = {
	{81.0f, 82.0f}, {135.0f, 169.0f}, {344.0f, 319.0f},
	{23.0f, 27.0f}, {37.0f, 58.0f}, {81.0f, 82.0f},
};

static const char *const coco_class_names[YOLOV4_TINY_CLASS_COUNT] = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
	"truck", "boat", "traffic light", "fire hydrant", "stop sign",
	"parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
	"cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
	"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
	"surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
	"knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
	"broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
	"couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
	"mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
	"teddy bear", "hair drier", "toothbrush",
};

static struct yolo_box yolo_boxes[YOLOV4_TINY_MAX_CANDIDATES];
static struct yolo_box *yolo_box_ptrs[YOLOV4_TINY_MAX_CANDIDATES];
static uint32_t yolo_box_count;
static bool yolo_boxes_full;

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
		return 1U;
	default:
		return 0U;
	}
}

static float dfp_to_float(int32_t value, int32_t fixed_point_pos)
{
	float scale = 1.0f;

	if (fixed_point_pos >= 0) {
		for (int32_t i = 0; i < fixed_point_pos; i++) {
			scale *= 0.5f;
		}
	} else {
		for (int32_t i = 0; i > fixed_point_pos; i--) {
			scale *= 2.0f;
		}
	}

	return (float)value * scale;
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
	case AMEBA_NPU_BUFFER_FORMAT_CHAR:
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
	case AMEBA_NPU_BUFFER_FORMAT_BOOL8:
	default:
		return *(const uint8_t *)addr;
	}
}

static bool is_signed_format(enum ameba_npu_buffer_format format)
{
	return format == AMEBA_NPU_BUFFER_FORMAT_INT8 ||
		   format == AMEBA_NPU_BUFFER_FORMAT_INT16 ||
		   format == AMEBA_NPU_BUFFER_FORMAT_INT32 ||
		   format == AMEBA_NPU_BUFFER_FORMAT_CHAR;
}

static float tensor_value(const struct yolov4_tiny_tensor *tensor, size_t index)
{
	const struct ameba_npu_buffer_param *param = &tensor->param;
	const uint8_t *addr = (const uint8_t *)tensor->data +
						  (index * element_size(param->data_format));

	if (param->quant_format == AMEBA_NPU_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
		return dfp_to_float(read_signed_value(addr, param->data_format),
							param->quant_data.dfp.fixed_point_pos);
	}

	if (param->quant_format == AMEBA_NPU_BUFFER_QUANTIZE_TF_ASYMM) {
		float raw_value = is_signed_format(param->data_format) ?
						  (float)read_signed_value(addr, param->data_format) :
						  (float)read_unsigned_value(addr, param->data_format);

		return (raw_value - (float)param->quant_data.affine.zero_point) *
			   param->quant_data.affine.scale;
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
	default:
		return is_signed_format(param->data_format) ?
			   (float)read_signed_value(addr, param->data_format) :
			   (float)read_unsigned_value(addr, param->data_format);
	}
}

static float sigmoid(float value)
{
	value = clamp_float(value, -16.0f, 16.0f);
	return 1.0f / (1.0f + expf(-value));
}

static size_t yolo_data_index(uint32_t grid_w, uint32_t grid_h,
							  uint32_t class_count, uint32_t x, uint32_t y,
							  uint32_t anchor, uint32_t entry)
{
	return (((size_t)(5U + class_count) * anchor + entry) * grid_h * grid_w) +
		   ((size_t)y * grid_w) + x;
}

static void add_candidate(const struct yolov4_tiny_tensor *tensor,
						  const float anchors[YOLOV4_TINY_ANCHORS_PER_OUTPUT][2],
						  uint32_t grid_w, uint32_t grid_h, uint32_t class_count,
						  uint32_t x, uint32_t y, uint32_t anchor)
{
	float objectness = sigmoid(tensor_value(tensor,
											yolo_data_index(grid_w, grid_h, class_count, x, y, anchor, 4U)));

	if (objectness < YOLOV4_TINY_CONFIDENCE_THRESH) {
		return;
	}

	for (uint32_t class_idx = 0; class_idx < class_count; class_idx++) {
		float class_probability = sigmoid(tensor_value(tensor,
										  yolo_data_index(grid_w, grid_h, class_count, x, y, anchor,
												  5U + class_idx)));
		float score = objectness * class_probability;
		struct yolo_box *box;
		float center_x;
		float center_y;
		float width;
		float height;
		float x1;
		float y1;

		if (score < YOLOV4_TINY_CONFIDENCE_THRESH) {
			continue;
		}

		if (yolo_box_count >= ARRAY_SIZE(yolo_boxes)) {
			yolo_boxes_full = true;
			return;
		}

		center_x = ((float)x + sigmoid(tensor_value(tensor,
									   yolo_data_index(grid_w, grid_h, class_count, x, y, anchor, 0U)))) /
				   (float)grid_w;
		center_y = ((float)y + sigmoid(tensor_value(tensor,
									   yolo_data_index(grid_w, grid_h, class_count, x, y, anchor, 1U)))) /
				   (float)grid_h;
		width = expf(clamp_float(tensor_value(tensor,
											  yolo_data_index(grid_w, grid_h, class_count, x, y, anchor, 2U)),
								 -16.0f, 16.0f)) * anchors[anchor][0] /
				(float)YOLOV4_TINY_INPUT_WIDTH;
		height = expf(clamp_float(tensor_value(tensor,
											   yolo_data_index(grid_w, grid_h, class_count, x, y, anchor, 3U)),
								  -16.0f, 16.0f)) * anchors[anchor][1] /
				 (float)YOLOV4_TINY_INPUT_HEIGHT;

		x1 = clamp_float(center_x - width * 0.5f, 0.0f, 1.0f);
		y1 = clamp_float(center_y - height * 0.5f, 0.0f, 1.0f);

		box = &yolo_boxes[yolo_box_count++];
		box->x = x1;
		box->y = y1;
		box->w = width;
		box->h = height;
		box->score = score;
		box->class_idx = class_idx;
		box->invalid = false;
	}
}

static float box_intersection(const struct yolo_box *a, const struct yolo_box *b)
{
	float width = MIN(a->x + a->w, b->x + b->w) - MAX(a->x, b->x);
	float height = MIN(a->y + a->h, b->y + b->h) - MAX(a->y, b->y);

	if (width <= 0.0f || height <= 0.0f) {
		return 0.0f;
	}

	return width * height;
}

static float box_iou(const struct yolo_box *a, const struct yolo_box *b)
{
	float intersection = box_intersection(a, b);
	float union_area = a->w * a->h + b->w * b->h - intersection;

	return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static float box_diou(const struct yolo_box *a, const struct yolo_box *b)
{
	float contain_x1 = MIN(a->x, b->x);
	float contain_y1 = MIN(a->y, b->y);
	float contain_x2 = MAX(a->x + a->w, b->x + b->w);
	float contain_y2 = MAX(a->y + a->h, b->y + b->h);
	float diagonal_x = contain_x2 - contain_x1;
	float diagonal_y = contain_y2 - contain_y1;
	float diagonal_squared = diagonal_x * diagonal_x + diagonal_y * diagonal_y;
	float center_dx = (a->x + a->w * 0.5f) - (b->x + b->w * 0.5f);
	float center_dy = (a->y + a->h * 0.5f) - (b->y + b->h * 0.5f);
	float center_distance_squared = center_dx * center_dx + center_dy * center_dy;
	float overlap = box_iou(a, b);

	if (diagonal_squared <= 0.0f) {
		return overlap;
	}

	return overlap - powf(center_distance_squared / diagonal_squared, 0.6f);
}

static int score_compare_desc(const void *pa, const void *pb)
{
	const struct yolo_box *a = *(const struct yolo_box * const *)pa;
	const struct yolo_box *b = *(const struct yolo_box * const *)pb;

	if (a->score < b->score) {
		return 1;
	}

	if (a->score > b->score) {
		return -1;
	}

	return 0;
}

static void do_class_aware_diou_nms(uint32_t class_count)
{
	for (uint32_t class_idx = 0; class_idx < class_count; class_idx++) {
		uint32_t class_box_count = 0;

		for (uint32_t i = 0; i < yolo_box_count; i++) {
			if (yolo_boxes[i].class_idx == class_idx) {
				yolo_box_ptrs[class_box_count++] = &yolo_boxes[i];
			}
		}

		qsort(yolo_box_ptrs, class_box_count, sizeof(yolo_box_ptrs[0]),
			  score_compare_desc);

		for (uint32_t i = 0; i < class_box_count; i++) {
			struct yolo_box *a = yolo_box_ptrs[i];

			if (a->invalid) {
				continue;
			}

			for (uint32_t j = i + 1U; j < class_box_count; j++) {
				struct yolo_box *b = yolo_box_ptrs[j];

				if (!b->invalid && box_diou(a, b) > YOLOV4_TINY_NMS_THRESH) {
					b->invalid = true;
				}
			}
		}
	}
}

static uint32_t normalized_to_pixel(float value, uint32_t limit)
{
	return (uint32_t)(clamp_float(value, 0.0f, 1.0f) * (float)limit + 0.5f);
}

static void log_objects(void)
{
	uint32_t valid_count = 0;

	for (uint32_t i = 0; i < yolo_box_count; i++) {
		if (!yolo_boxes[i].invalid) {
			valid_count++;
		}
	}

	LOG_INF("Detected %u object(s)", valid_count);

	for (uint32_t i = 0, out_idx = 0; i < yolo_box_count; i++) {
		const struct yolo_box *box = &yolo_boxes[i];
		uint32_t score_permille;
		uint32_t x1;
		uint32_t y1;
		uint32_t x2;
		uint32_t y2;

		if (box->invalid) {
			continue;
		}

		score_permille = (uint32_t)(clamp_float(box->score, 0.0f, 1.0f) *
									1000.0f + 0.5f);
		x1 = normalized_to_pixel(box->x, YOLOV4_TINY_INPUT_WIDTH);
		y1 = normalized_to_pixel(box->y, YOLOV4_TINY_INPUT_HEIGHT);
		x2 = normalized_to_pixel(box->x + box->w, YOLOV4_TINY_INPUT_WIDTH);
		y2 = normalized_to_pixel(box->y + box->h, YOLOV4_TINY_INPUT_HEIGHT);

		LOG_INF("Object[%u]: class=%u (%s) score=%u.%03u x1=%u y1=%u x2=%u y2=%u",
				out_idx, box->class_idx, coco_class_names[box->class_idx],
				score_permille / 1000U, score_permille % 1000U,
				x1, y1, x2, y2);
		out_idx++;
	}

	if (yolo_boxes_full) {
		LOG_WRN("YOLO candidate buffer full; results may be truncated");
	}
}

static int validate_and_decode_output(const struct yolov4_tiny_tensor *tensor,
									  bool *seen_13, bool *seen_26)
{
	const struct ameba_npu_buffer_param *param = &tensor->param;
	const float (*anchors)[2];
	size_t elem_size = element_size(param->data_format);
	size_t required_size;
	uint32_t grid_w;
	uint32_t grid_h;
	uint32_t channels;
	uint32_t class_count;

	if (tensor->data == NULL || elem_size == 0U || param->dim_count < 3U) {
		LOG_ERR("Unsupported YOLO output tensor metadata");
		return -EINVAL;
	}

	grid_w = param->dim_size[0];
	grid_h = param->dim_size[1];
	channels = param->dim_size[2];

	if (channels < YOLOV4_TINY_ANCHORS_PER_OUTPUT * 5U ||
		channels % YOLOV4_TINY_ANCHORS_PER_OUTPUT != 0U) {
		LOG_ERR("Invalid YOLO output depth: %u", channels);
		return -EINVAL;
	}

	class_count = channels / YOLOV4_TINY_ANCHORS_PER_OUTPUT - 5U;
	if (class_count != YOLOV4_TINY_CLASS_COUNT) {
		LOG_ERR("YOLO output has %u classes; expected %u",
				class_count, YOLOV4_TINY_CLASS_COUNT);
		return -EINVAL;
	}

	required_size = (size_t)grid_w * grid_h * channels * elem_size;
	if (required_size > tensor->size) {
		LOG_ERR("YOLO output buffer too small: need=%u have=%u",
				(uint32_t)required_size, (uint32_t)tensor->size);
		return -EINVAL;
	}

	if (grid_w == 13U && grid_h == 13U && !*seen_13) {
		anchors = &yolov4_tiny_anchors[0];
		*seen_13 = true;
	} else if (grid_w == 26U && grid_h == 26U && !*seen_26) {
		anchors = &yolov4_tiny_anchors[3];
		*seen_26 = true;
	} else {
		LOG_ERR("Unexpected or duplicate YOLO output grid: %ux%u", grid_w, grid_h);
		return -EINVAL;
	}

	for (uint32_t x = 0; x < grid_w; x++) {
		for (uint32_t y = 0; y < grid_h; y++) {
			for (uint32_t anchor = 0; anchor < YOLOV4_TINY_ANCHORS_PER_OUTPUT;
				 anchor++) {
				add_candidate(tensor, anchors, grid_w, grid_h, class_count,
							  x, y, anchor);
			}
		}
	}

	return 0;
}

int yolov4_tiny_postprocess_and_log(const struct yolov4_tiny_tensor tensors[],
									uint32_t count)
{
	bool seen_13 = false;
	bool seen_26 = false;
	int ret;

	if (count < YOLOV4_TINY_OUTPUT_COUNT) {
		LOG_ERR("YOLOv4-tiny postprocess expects %u outputs, got %u",
				YOLOV4_TINY_OUTPUT_COUNT, count);
		return -EINVAL;
	}

	yolo_box_count = 0U;
	yolo_boxes_full = false;
	memset(yolo_boxes, 0, sizeof(yolo_boxes));

	for (uint32_t i = 0; i < YOLOV4_TINY_OUTPUT_COUNT; i++) {
		ret = validate_and_decode_output(&tensors[i], &seen_13, &seen_26);
		if (ret != 0) {
			return ret;
		}
	}

	if (!seen_13 || !seen_26) {
		LOG_ERR("YOLOv4-tiny requires one 13x13 and one 26x26 output");
		return -EINVAL;
	}

	do_class_aware_diou_nms(YOLOV4_TINY_CLASS_COUNT);
	log_objects();

	return 0;
}
