/*
 * Copyright (c) 2024 Realtek Semiconductor Corp.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <zephyr/kernel.h>
#include <zephyr/drivers/adc.h>
#include <zephyr/device.h>

static const struct device *adc_dev = DEVICE_DT_GET(DT_NODELABEL(adc));

/* PIN_F0 : PID_ADC0 */
/* PIN_F1 : PID_ADC1 */
/* PIN_F2 : PID_ADC2 */
/* PIN_F3 : PID_ADC3 */
/* PIN_A0 : PID_ADC4 */
/* PIN_A1 : PID_ADC5 */
/* PIN_A2 : PID_ADC6 */
/* PIN_A3 : PID_ADC7 */

void set_adc_channel(uint8_t ch)
{
	struct adc_channel_cfg channel_cfg = {
		.channel_id = ch,
	};

	adc_channel_setup(adc_dev, &channel_cfg);
}

#define ADC_SAMPLE_COUNT 3

void read_adc_channel(uint8_t ch)
{
	uint16_t sample_buffer[ADC_SAMPLE_COUNT];
	struct adc_sequence sequence = {
		.channels = BIT(ch),
		.buffer = sample_buffer,
		.buffer_size = sizeof(sample_buffer),
	};
	uint32_t sum = 0;

	adc_read(adc_dev, &sequence);

	for (int i = 0; i < ADC_SAMPLE_COUNT; i++) {
		printk("ADC channel: %d, sample[%d]: %u\n", ch, i, sample_buffer[i]);
		sum += sample_buffer[i];
	}
	printk("ADC channel: %d, average (%d samples): %u\n", ch, ADC_SAMPLE_COUNT,
		   (uint32_t)(sum / ADC_SAMPLE_COUNT));
}

static void test_adc(bool async)
{
	if (!device_is_ready(adc_dev)) {
		printk("ADC device not ready\n");
		return;
	}

	set_adc_channel(6);
	set_adc_channel(7);

	while (1) {
		read_adc_channel(6);
		k_sleep(K_MSEC(1000));
		read_adc_channel(7);
		k_sleep(K_MSEC(1000));
	}

	printf("adc  finish\r\n");
}

int main(void)
{
	printf("adc example! %s\n", CONFIG_BOARD_TARGET);

	test_adc(false);

	return 0;
}

