/*
 * Copyright (c) 2024 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ZEPHYR_INCLUDE_DT_BINDINGS_VIDEO_AMEBAPRO2_VIDEO_H_
#define ZEPHYR_INCLUDE_DT_BINDINGS_VIDEO_AMEBAPRO2_VIDEO_H_

/* encode_type values for default-video-type DTS property */
#define AMEBAPRO2_VIDEO_TYPE_HEVC  0
#define AMEBAPRO2_VIDEO_TYPE_H264  1
#define AMEBAPRO2_VIDEO_TYPE_JPEG  2
#define AMEBAPRO2_VIDEO_TYPE_NV12  3

/* VOE output mode values for default-out-mode DTS property */
#define AMEBAPRO2_VIDEO_OUT_MODE_DEFAULT    0
#define AMEBAPRO2_VIDEO_OUT_MODE_CONTINUOUS 2

#endif /* ZEPHYR_INCLUDE_DT_BINDINGS_VIDEO_AMEBAPRO2_VIDEO_H_ */
