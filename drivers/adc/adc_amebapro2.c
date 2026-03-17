/*
 * Copyright (c) 2024 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#define DT_DRV_COMPAT realtek_amebapro2_adc

/* Include <soc.h> before <ameba_soc.h> to avoid redefining unlikely() macro */
#include <soc.h>
#include <zephyr/drivers/adc.h>
#include <zephyr/drivers/pinctrl.h>
#include <zephyr/drivers/clock_control.h>
#include <zephyr/logging/log.h>
#include "hal.h"
#include "hal_adc.h"

LOG_MODULE_REGISTER(adc_ameba, CONFIG_ADC_LOG_LEVEL);

static uint8_t analogin_init_flag = 0;
static hal_adc_adapter_t analogin_con_adpt;

/* PIN_F0 : PID_ADC0 */
/* PIN_F1 : PID_ADC1 */
/* PIN_F2 : PID_ADC2 */
/* PIN_F3 : PID_ADC3 */
/* PIN_A0 : PID_ADC4 */
/* PIN_A1 : PID_ADC5 */
/* PIN_A2 : PID_ADC6 */
/* PIN_A3 : PID_ADC7 */

struct adc_ameba_config {
	const uint8_t channel_count;
};

struct adc_ameba_data {
	uint16_t meas_ref_internal;
	uint16_t *buffer;
};

static int adc_ameba_read(const struct device *dev, const struct adc_sequence *seq)
{
	struct adc_ameba_data *data = dev->data;
	uint16_t anain16;

	anain16 = hal_adc_single_read(&analogin_con_adpt, seq->channels);

	/* Store result */
	data->buffer = (uint16_t *)seq->buffer;
	data->buffer[0] = anain16;

	return 0;
}

static int adc_ameba_channel_setup(const struct device *dev, const struct adc_channel_cfg *cfg)
{
	const struct adc_ameba_config *conf = dev->config;

	if (cfg->channel_id >= conf->channel_count) {
		LOG_ERR("Unsupported channel id '%d' \n\r ", cfg->channel_id);
		return -ENOTSUP;
	}

	if (cfg->differential) {
		LOG_ERR("Differential channels are not supported");
		return -ENOTSUP;
	}

	LOG_DBG("adc_ameba_channel_setup");

	if (!analogin_init_flag) {
		memset(&analogin_con_adpt, 0x00, sizeof(analogin_con_adpt));
		hal_adc_load_default(&analogin_con_adpt);
		/* set pin enable flag */
		analogin_con_adpt.plft_dat.pin_en.w |= ((uint32_t)0x1 << cfg->channel_id);
		if (hal_adc_init(&analogin_con_adpt) != HAL_OK) {
			LOG_ERR("analogin initialization failed\n");
		} else {
			analogin_init_flag = 1;
			k_sleep(K_MSEC(20));
			hal_adc_set_in_type_all((hal_adc_adapter_t *)&analogin_con_adpt, HP_ADC_INPUT_ALL_SINGLE);
			hal_adc_set_cvlist((hal_adc_adapter_t *)&analogin_con_adpt, (uint8_t *)&analogin_init_flag, 1);
		}
	} else {
		/* module initialized but pin was NOT */
		if ((analogin_con_adpt.plft_dat.pin_en.w & ((uint32_t)0x1 << cfg->channel_id)) == 0) {
			LOG_DBG("module initialized; now initializing pin for ADC%d\r\n", cfg->channel_id);
			analogin_con_adpt.plft_dat.pin_en.w |= ((uint32_t)0x1 << cfg->channel_id);
			hal_adc_pin_init(&analogin_con_adpt);
		}
	}

	return 0;
}

static int adc_ameba_init(const struct device *dev)
{
	return 0;
}

static const struct adc_driver_api api_ameba_driver_api = {
	.channel_setup = adc_ameba_channel_setup,
	.read = adc_ameba_read,
	.ref_internal = 3300,
};



static const struct adc_ameba_config adc_config = {
	.channel_count = 8,
};

static struct adc_ameba_data adc_data = {
	.meas_ref_internal = 3300,
};

DEVICE_DT_INST_DEFINE(0, adc_ameba_init, NULL, &adc_data, &adc_config, POST_KERNEL,
					  CONFIG_ADC_INIT_PRIORITY, &api_ameba_driver_api);
