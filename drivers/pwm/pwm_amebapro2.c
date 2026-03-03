/*
 * Copyright (c) 2024 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#define DT_DRV_COMPAT realtek_amebapro2_pwm

/* Include <soc.h> before <ameba_soc.h> to avoid redefining unlikely() macro */
#include <soc.h>
#include <zephyr/drivers/pwm.h>
#include <zephyr/drivers/pwm/pwm_ameba.h>
#include <zephyr/drivers/pinctrl.h>
#include <zephyr/drivers/clock_control.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include "hal.h"
#include "hal_timer.h"
#include "hal_pwm.h"

LOG_MODULE_REGISTER(pwm_ameba, CONFIG_PWM_LOG_LEVEL);

static u8 timer_for_pwm_temp[] = {GTimer0, GTimer1, GTimer2, GTimer3, GTimer4, GTimer5, GTimer6, GTimer7, 0xff};  /* the timer ID list those can be used as PWM tick source*/
static hal_pwm_comm_adapter_t pwm_com_handler;
static hal_timer_group_adapter_t _timer_group0;
static hal_pwm_adapter_t pwm_c0, pwm_c1, pwm_c2, pwm_c8, pwm_c9;

struct pwm_ameba_data {
	uint16_t prescale;
	bool CC_polarity;
};

struct pwm_ameba_config {
	uint32_t clock_frequency;
};

static int pwm_ameba_get_cycles_per_sec(const struct device *dev, uint32_t channel_idx,
										uint64_t *cycles)
{
	const struct pwm_ameba_config *config = dev->config;

	*cycles = config->clock_frequency;

	return 0;
}

static int pwm_ameba_set_cycles(const struct device *dev, uint32_t channel_idx,
								uint32_t period_cycles, uint32_t pulse_cycles, pwm_flags_t flags)
{
	uint32_t period_us;
	uint32_t duty_us;

	/* 40 MHz PWM clock: cycles -> us */
	period_us = period_cycles / 40;
	duty_us = pulse_cycles / 40;

	if (channel_idx == 0) {
		hal_pwm_set_duty(&pwm_c0, period_us, duty_us, 0);
	} else if (channel_idx == 1) {
		hal_pwm_set_duty(&pwm_c1, period_us, duty_us, 0);
	} else if (channel_idx == 2) {
		hal_pwm_set_duty(&pwm_c2, period_us, duty_us, 0);
	} else if (channel_idx == 8) {
		hal_pwm_set_duty(&pwm_c8, period_us, duty_us, 0);
	} else if (channel_idx == 9) {
		hal_pwm_set_duty(&pwm_c9, period_us, duty_us, 0);
	} else {
		LOG_ERR("Invalid PWM channel_idx (%d)\n", channel_idx);
	}

	return 0;
}

int pwm_ameba_init(const struct device *dev)
{
	hal_timer_clock_init(0, ENABLE);
	hal_timer_group_init(&_timer_group0, 0);
	hal_timer_group_sclk_sel(&_timer_group0, GTimerSClk_40M);  /* Default 40MHz */
	hal_pwm_clk_sel(PWM_Sclk_40M);

	hal_pwm_clock_init(ENABLE);
	hal_pwm_comm_init(&pwm_com_handler);
	hal_pwm_comm_tick_source_list(timer_for_pwm_temp);

	hal_pwm_init(&pwm_c0, 0, 0, 0);  /* PIN_F6 */
	hal_pwm_set_duty(&pwm_c0, 50, 0, 0); /* 20khz 0% */
	hal_pwm_enable(&pwm_c0);

	hal_pwm_init(&pwm_c1, 1, 0, 0);  /* PIN_F7 */
	hal_pwm_set_duty(&pwm_c1, 50, 0, 0); /* 20khz 0% */
	hal_pwm_enable(&pwm_c1);

	hal_pwm_init(&pwm_c2, 2, 0, 0);  /* PIN_F8 */
	hal_pwm_set_duty(&pwm_c2, 50, 0, 0); /* 20khz 0% */
	hal_pwm_enable(&pwm_c2);

	hal_pwm_init(&pwm_c8, 8, 0, 0);  /* PIN_F14 */
	hal_pwm_set_duty(&pwm_c8, 50, 0, 0); /* 20khz 0% */
	hal_pwm_enable(&pwm_c8);

	hal_pwm_init(&pwm_c9, 9, 0, 0);  /* PIN_F15 */
	hal_pwm_set_duty(&pwm_c9, 50, 0, 0); /* 20khz 0% */
	hal_pwm_enable(&pwm_c9);
	return 0;
}

static const struct pwm_driver_api pwm_ameba_api = {
	.set_cycles = pwm_ameba_set_cycles,
	.get_cycles_per_sec = pwm_ameba_get_cycles_per_sec,
};

static struct pwm_ameba_data pwm_ameba_data;

static const struct pwm_ameba_config pwm_ameba_config = {
	.clock_frequency = 40000000, /* 40M */
};

DEVICE_DT_INST_DEFINE(0, pwm_ameba_init, NULL, &pwm_ameba_data,
					  &pwm_ameba_config, POST_KERNEL, CONFIG_PWM_INIT_PRIORITY,
					  &pwm_ameba_api);
