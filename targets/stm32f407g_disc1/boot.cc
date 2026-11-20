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
// Forked from pigweed's targets/stm32f429i_disc1_stm32cube/boot.cc.
// SystemClock_Config() is rewritten for the F407VG's 168MHz max (vs. the
// F429's 180MHz): different PLL dividers, and no over-drive mode -- that PWR
// feature doesn't exist on the STM32F407 die at all (F42x/43x-only), so the
// F429 target's HAL_PWREx_EnableOverDrive() call is removed rather than kept
// as dead code.

#include "pw_boot/boot.h"

#include <array>

#include "FreeRTOS.h"
#include "pw_assert/check.h"
#include "pw_boot_cortex_m/boot.h"
#include "pw_malloc/malloc.h"
#include "pw_memory/busy_wait_forever.h"
#include "pw_preprocessor/compiler.h"
#include "pw_string/util.h"
#include "pw_sys_io_stm32cube/init.h"
#include "stm32f4xx.h"
#include "task.h"

// Initializes clock to its max, 168MHz, from the board's 8MHz HSE. Note that
// this naming follows CubeMX's naming out of convention. It's not required
// that this target provides a symbol named SystemClock_Config. This function
// shares the same purpose as the symbol of the same name that is generated
// by CubeMX.
extern "C" void SystemClock_Config() {
  RCC_OscInitTypeDef RCC_OscInitStruct = {};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  // (HSE / PLLM) * PLLN / PLLP = (8MHz / 8) * 336 / 2 = 168MHz SYSCLK.
  // PLLQ = 7 gives a 48MHz USB/SDIO/RNG clock, unused today but standard.
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
    pw_boot_PostMain();
  }

  // No over-drive step here: unlike the F429/F439, the F407 die doesn't
  // support over-drive mode at all, and 168MHz doesn't need it in any case
  // (over-drive on F42x/43x is only required above 168MHz).

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  // APB1 max is 42MHz, APB2 max is 84MHz; both dividers hit those caps
  // exactly at 168MHz HCLK.
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK) {
    pw_boot_PostMain();
  }

  // TODO: configure PLLI2S (separate from the main PLL above) once the
  // audio capture/output drivers land -- it feeds the I2S2/I2S3 bit clocks
  // independently of SYSCLK/HCLK.
}

// FreeRTOS owns SysTick (SysTick_Handler is its port handler), so the HAL's
// own 1ms tick (uwTick) never advanced: every HAL timeout (SPI/I2C/UART) could
// never expire, and getRunTimeCounterValue() below always returned 0. The
// FreeRTOS tick runs at 1kHz (configTICK_RATE_HZ), matching HAL's 1ms tick.
extern "C" void vApplicationTickHook(void) { HAL_IncTick(); }

// Sleeps the core whenever no task is ready. Any interrupt wakes it (the
// 1 kHz tick at the latest), and peripheral clocks keep running, so audio
// and USB are unaffected. Without this the idle task spins at full speed
// and the chip draws its full run current even when it has nothing to do.
extern "C" void vApplicationIdleHook(void) { __WFI(); }

// Functions needed when configGENERATE_RUN_TIME_STATS is on.
extern "C" void configureTimerForRunTimeStats(void) {}
extern "C" unsigned long getRunTimeCounterValue(void) { return uwTick; }

extern "C" void pw_boot_PreStaticMemoryInit() {}

extern "C" void pw_boot_PreStaticConstructorInit() {
  // Provided by STMicroelectronics SDK. Can be configured to be provided
  // elsewhere by changing pw_third_party_stm32cube_CMSIS_INIT.
  SystemInit();

  // Provided by the STMicroelectronics SDK.
  HAL_Init();

  // Typically provided by CubeMX codegen, SystemClock_Config() is instead
  // provided as part of this target.
  SystemClock_Config();

#if PW_MALLOC_ACTIVE
  pw_MallocInit(&pw_boot_heap_low_addr, &pw_boot_heap_high_addr);
#endif  // PW_MALLOC_ACTIVE
  pw_sys_io_Init();
}

// TODO(amontanez): pw_boot_PreMainInit() should get renamed to
// pw_boot_FinalizeBoot or similar when main() is removed.
// Provided by the application: creates its tasks. Runs before the scheduler
// starts, so it must not block.
extern "C" void device_ml_AppInit();

extern "C" void pw_boot_PreMainInit() {
  device_ml_AppInit();
  vTaskStartScheduler();
  PW_UNREACHABLE;
}

// This `main()` stub prevents another main function from being linked since
// this target deliberately doesn't run `main()`.
int main() {}

extern "C" PW_NO_RETURN void pw_boot_PostMain() {
  // In case main() returns, just sit here until the device is reset.
  pw::BusyWaitForever();
}
