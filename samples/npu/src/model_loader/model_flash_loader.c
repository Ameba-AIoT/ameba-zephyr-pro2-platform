/*
 * Copyright (c) 2026 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "model_flash_loader.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

LOG_MODULE_DECLARE(npu_sample, LOG_LEVEL_INF);

#define NPU_FLASH_BASE 0x08000000UL
#define NPU_FLASH_PARTITION_TABLE_OFFSET 0x2000UL
#define NPU_FLASH_MANIFEST_SIZE 4096U
#define NPU_FLASH_IMAGE_HEADER_SIZE 128U
#define NPU_FLASH_NN_MODEL_TYPE_ID 0x81CFU
#define NPU_FLASH_NN_IMAGE_TYPE_ID 0xF8E0U
#define NPU_FLASH_MODEL_ALIGNMENT 4096U
#define NPU_FWFS_MAX_FILES 32U

struct nor_part_hdr {
	uint8_t rec_num;
	uint8_t bl_p_idx;
	uint8_t bl_s_idx;
	uint8_t fw1_idx;
	uint8_t fw2_idx;
	uint8_t iq_idx;
	uint8_t nn_m_idx;
	uint8_t mp_idx;
	uint8_t keycert1_idx;
	uint8_t keycert2_idx;
	uint8_t fcs_idx;
	uint8_t resv1[5];
	uint16_t ota_trap;
	uint16_t mp_trap;
	uint32_t udl;
	uint8_t resv2[8];
} __attribute__((packed));

struct nor_part_rec {
	uint32_t start_addr;
	uint32_t length;
	uint16_t type_id;
	uint8_t resv1[5];
	uint8_t valid;
	uint8_t resv2[16];
} __attribute__((packed));

struct fw_img_hdr {
	uint32_t imglen;
	uint32_t nxtoffset;
	uint16_t type_id;
	uint16_t nxt_type_id;
	uint8_t sec_enc_ctrl;
	uint8_t resv[115];
} __attribute__((packed));

struct fwfs_file {
	char filename[40];
	uint32_t filelen;
	uint32_t offset;
} __attribute__((packed));

struct fwfs_folder {
	char tag[12];
	uint32_t file_cnt;
	struct fwfs_file files[NPU_FWFS_MAX_FILES];
} __attribute__((packed));

static const uint8_t *flash_ptr(uint32_t offset)
{
	return (const uint8_t *)(NPU_FLASH_BASE + offset);
}

static void *align_ptr(void *ptr, size_t alignment)
{
	uintptr_t value = (uintptr_t)ptr;

	return (void *)ROUND_UP(value, alignment);
}

static const struct nor_part_rec *find_nn_partition(void)
{
	const struct nor_part_hdr *hdr =
		(const struct nor_part_hdr *)flash_ptr(NPU_FLASH_PARTITION_TABLE_OFFSET);
	const struct nor_part_rec *rec =
		(const struct nor_part_rec *)(flash_ptr(NPU_FLASH_PARTITION_TABLE_OFFSET) +
									  sizeof(*hdr));
	uint32_t rec_num = hdr->rec_num;

	if (rec_num == 0U || rec_num > 64U) {
		rec_num = 64U;
	}

	for (uint32_t i = 0; i < rec_num; i++) {
		if (rec[i].type_id == 0U || rec[i].type_id == 0xFFFFU) {
			break;
		}

		if (rec[i].type_id == NPU_FLASH_NN_MODEL_TYPE_ID &&
			rec[i].valid != 0U) {
			return &rec[i];
		}
	}

	return NULL;
}

static int find_fwfs_file(const struct fwfs_folder *folder, const char *filename,
						  const struct fwfs_file **file)
{
	uint32_t file_count;

	if (memcmp(folder->tag, "FWFSDIR", 7) != 0) {
		LOG_ERR("NN_MDL FWFS directory tag is invalid");
		return -EINVAL;
	}

	file_count = MIN(folder->file_cnt, NPU_FWFS_MAX_FILES);
	for (uint32_t i = 0; i < file_count; i++) {
		if (strncmp(folder->files[i].filename, filename,
					sizeof(folder->files[i].filename)) == 0) {
			*file = &folder->files[i];
			return 0;
		}
	}

	LOG_ERR("Model file '%s' was not found in NN_MDL partition", filename);
	return -ENOENT;
}

int npu_flash_model_load(const char *filename, struct npu_flash_model *model)
{
	const struct nor_part_rec *partition;
	const struct fw_img_hdr *image_header;
	const struct fwfs_folder *folder;
	const struct fwfs_file *file = NULL;
	const uint8_t *partition_base;
	const uint8_t *content_base;
	const uint8_t *model_base;
	void *model_alloc;
	uint8_t *model_copy;
	uint32_t content_size;
	int ret;

	if (filename == NULL || model == NULL) {
		return -EINVAL;
	}

	memset(model, 0, sizeof(*model));

	partition = find_nn_partition();
	if (partition == NULL) {
		LOG_ERR("NN_MDL partition was not found");
		return -ENOENT;
	}

	partition_base = flash_ptr(partition->start_addr);
	if (memcmp(partition_base, "RTL8735B", 8) != 0) {
		LOG_ERR("NN_MDL partition manifest is invalid");
		return -EINVAL;
	}

	image_header = (const struct fw_img_hdr *)(partition_base +
				   NPU_FLASH_MANIFEST_SIZE);
	if (image_header->type_id != NPU_FLASH_NN_IMAGE_TYPE_ID) {
		LOG_ERR("NN_MDL image type mismatch: 0x%04x", image_header->type_id);
		return -EINVAL;
	}

	content_base = partition_base + NPU_FLASH_MANIFEST_SIZE +
				   NPU_FLASH_IMAGE_HEADER_SIZE;
	content_size = image_header->imglen;
	if (content_size == 0U ||
		content_size > partition->length - NPU_FLASH_MANIFEST_SIZE -
		NPU_FLASH_IMAGE_HEADER_SIZE) {
		LOG_ERR("NN_MDL image length is invalid: %u", content_size);
		return -EINVAL;
	}

	folder = (const struct fwfs_folder *)content_base;
	ret = find_fwfs_file(folder, filename, &file);
	if (ret != 0) {
		return ret;
	}

	if (file->filelen == 0U || file->offset > content_size ||
		file->filelen > content_size - file->offset) {
		LOG_ERR("Model file range is invalid: offset=%u size=%u content=%u",
				file->offset, file->filelen, content_size);
		return -EINVAL;
	}

	model_base = content_base + file->offset;
	model_alloc = malloc(file->filelen + NPU_FLASH_MODEL_ALIGNMENT - 1U);
	if (model_alloc == NULL) {
		LOG_ERR("Failed to allocate %u bytes for 4KB-aligned model",
				file->filelen);
		return -ENOMEM;
	}

	model_copy = align_ptr(model_alloc, NPU_FLASH_MODEL_ALIGNMENT);
	memcpy(model_copy, model_base, file->filelen);

	model->alloc = model_alloc;
	model->data = model_copy;
	model->size = file->filelen;

	LOG_INF("Loaded %s from NN_MDL flash partition: flash=0x%08x size=%u",
			filename, (uint32_t)(NPU_FLASH_BASE + partition->start_addr +
								 NPU_FLASH_MANIFEST_SIZE +
								 NPU_FLASH_IMAGE_HEADER_SIZE + file->offset),
			(uint32_t)model->size);

	return 0;
}

void npu_flash_model_unload(struct npu_flash_model *model)
{
	if (model == NULL) {
		return;
	}

	free(model->alloc);
	memset(model, 0, sizeof(*model));
}
