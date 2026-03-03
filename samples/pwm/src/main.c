/*
 * Copyright (c) 2024 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/drivers/pwm.h>
#include <zephyr/device.h>

static const struct device *pwm_dev = DEVICE_DT_GET(DT_NODELABEL(pwm));

#define PWM_CHANNEL_0 0
#define PWM_CHANNEL_1 1
#define PWM_CHANNEL_2 2
#define PWM_CHANNEL_8 8
#define PWM_CHANNEL_9 9

/* 40 MHz clock, 20 kHz PWM */
uint32_t period_cycles = 2000;   /* 40MHz / 20kHz */
uint32_t pulse_cycles = 1000;    /* 50% duty */
uint64_t cycles_per_sec;

void set_pwm_channel(uint32_t channel, uint32_t pulse_cycles)
{
	int ret = pwm_set_cycles(pwm_dev, channel, period_cycles, pulse_cycles, 0);
	if (ret) {
		printf("Failed to set PWM channel %d: %d\n", channel, ret);
	}
}

static void test_pwm(bool async)
{
	if (!device_is_ready(pwm_dev)) {
		printf("PWM device not ready\n");
		return;
	}

	pwm_get_cycles_per_sec(pwm_dev, 0, &cycles_per_sec);
	printf("PWM clock = %llu Hz\n", cycles_per_sec);

	set_pwm_channel(PWM_CHANNEL_0, 400); /* 20% */
	set_pwm_channel(PWM_CHANNEL_1, 800); /* 40% */
	set_pwm_channel(PWM_CHANNEL_2, 1200); /* 60% */
	set_pwm_channel(PWM_CHANNEL_8, 1600); /* 80% */
	set_pwm_channel(PWM_CHANNEL_9, 2000); /* 100% */

	printf("flash pwm finish\r\n");
}

int main(void)
{
	printf("pwm example! %s\n", CONFIG_BOARD_TARGET);

	test_pwm(false);

	return 0;
}
