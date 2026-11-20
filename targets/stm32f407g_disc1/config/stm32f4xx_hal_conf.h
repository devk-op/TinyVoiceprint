// Copyright 2026 The Pigweed Authors
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//
// Forked from pigweed's targets/stm32f429i_disc1_stm32cube/config, adapted
// for STM32F407VG. Unlike the F429 target, this file only enables HAL
// modules the project actually needs (audio DSP over I2S2/I2S3, I2C-driven
// codec control, PDM decimation via CRC, plus core/clock/GPIO/DMA/timer/UART
// plumbing). This matters here, not just for size: every module's HAL header
// is compiled against STM32F407xx regardless of whether it's used, and the
// STM32F407 die lacks several peripherals the F429 die has (LTDC, DSI,
// DMA2D, SDRAM, SAI, QSPI, FMPI2C/FMPSMBUS, DFSDM, CEC, LPTIM). Enabling
// those here would fail to compile against the F407 CMSIS device header, so
// -- unlike the F429 target's config, which enables nearly everything -- this
// file is intentionally minimal.

#pragma once

#include <stdint.h>

/* Clock setup */
#define HSI_VALUE 16000000U
#define LSI_VALUE 32000U

// The F407G-DISC1 has an 8MHz external crystal (X2). Verify against the
// specific board revision if clock startup fails.
#define HSE_VALUE 8000000U
#define HSE_STARTUP_TIMEOUT 100U

// The F407G-DISC1 does not populate an LSE crystal.
#define LSE_VALUE 0U
#define LSE_STARTUP_TIMEOUT 5000U

#define EXTERNAL_CLOCK_VALUE 0U

/* HAL Config */
#define TICK_INT_PRIORITY 0x0FU
#define USE_RTOS 0U
#define PREFETCH_ENABLE 1U
#define INSTRUCTION_CACHE_ENABLE 1U
#define DATA_CACHE_ENABLE 1U

#define assert_param(expr) ((void)0U)

/** HAL Headers: comment out defines + include to remove **/
/* primary HAL headers */
#define HAL_CORTEX_MODULE_ENABLED
#include "stm32f4xx_hal_cortex.h"

#define HAL_DMA_MODULE_ENABLED
#include "stm32f4xx_hal_dma.h"

#define HAL_EXTI_MODULE_ENABLED
#include "stm32f4xx_hal_exti.h"

#define HAL_GPIO_MODULE_ENABLED
#include "stm32f4xx_hal_gpio.h"

#define HAL_RCC_MODULE_ENABLED
#include "stm32f4xx_hal_rcc.h"

#define HAL_FLASH_MODULE_ENABLED
#include "stm32f4xx_hal_flash.h"

#define HAL_PWR_MODULE_ENABLED
#include "stm32f4xx_hal_pwr.h"

#define HAL_TIM_MODULE_ENABLED
#define USE_HAL_TIM_REGISTER_CALLBACKS 0U
#include "stm32f4xx_hal_tim.h"

/* Audio DSP path: I2S2 (mic) / I2S3 (headphone DAC) */
#define HAL_I2S_MODULE_ENABLED
#define USE_HAL_I2S_REGISTER_CALLBACKS 0U
#include "stm32f4xx_hal_i2s.h"

/* CS43L22 control interface */
#define HAL_I2C_MODULE_ENABLED
#define USE_HAL_I2C_REGISTER_CALLBACKS 0U
#include "stm32f4xx_hal_i2c.h"

/* LIS3DSH accelerometer on SPI1 (PA5/PA6/PA7, CS on PE3) */
#define HAL_SPI_MODULE_ENABLED
#define USE_HAL_SPI_REGISTER_CALLBACKS 0U
#include "stm32f4xx_hal_spi.h"

/* Required by ST's PDM2PCM decimation filter */
#define HAL_CRC_MODULE_ENABLED
#include "stm32f4xx_hal_crc.h"

/* pw_sys_io_stm32cube logging backend */
#define HAL_UART_MODULE_ENABLED
#define USE_HAL_UART_REGISTER_CALLBACKS 0U
#include "stm32f4xx_hal_uart.h"

// Modules deliberately left disabled (not used by this project, and several
// -- LTDC, DSI, DMA2D, SDRAM, SAI, QSPI, FMPI2C/FMPSMBUS, DFSDM, CEC, LPTIM --
// don't exist on the STM32F407 die): ADC, CAN, CEC, CRYP, DAC, DCMI, DFSDM,
// DMA2D, DSI, ETH, FMPI2C, FMPSMBUS, HASH, HCD, IRDA, IWDG, LPTIM, LTDC, MMC,
// NAND, NOR, PCCARD, PCD, QSPI, RNG, RTC, SAI, SD, SDRAM, SMARTCARD, SMBUS,
// SPDIFRX, SPI, SRAM, USART, WWDG.
