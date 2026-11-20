// Copyright 2021 The Pigweed Authors
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
// Unchanged from pigweed's targets/stm32f429i_disc1_stm32cube/vector_table.c:
// the core exception handlers and TIM6_DAC timebase IRQ are identical across
// the whole STM32F4 family (shared RM0090 reference manual). Peripheral IRQ
// entries for I2S2/I2S3 DMA and I2C1 will be added here once the audio
// capture/output/codec drivers need them.

#include <stdbool.h>

#include "pw_boot/boot.h"
#include "pw_boot_cortex_m/boot.h"
#include "pw_memory/busy_wait_forever.h"
#include "pw_preprocessor/compiler.h"
#include "stm32f4xx.h"

// Default handler to insert into the ARMv7-M vector table (below).
// This function exists for convenience. If a device isn't doing what you
// expect, it might have hit a fault and ended up here.
static void DefaultFaultHandler(void) {
  // Wait for debugger to attach.
  pw_BusyWaitForever();
}

// This is the device's interrupt vector table. It's not referenced in any
// code because the platform (STM32F4xx) expects this table to be present at the
// beginning of flash. The exact address is specified in the pw_boot_armv7m
// configuration as part of the target config.
//
// For more information, see ARMv7-M Architecture Reference Manual DDI 0403E.b
// section B1.5.3.

// This typedef is for convenience when building the vector table. With the
// exception of SP_main (0th entry in the vector table), all the entries of the
// vector table are function pointers.
typedef void (*InterruptHandler)(void);

// This is the timer interrupt handler implemented by the stm32cubef4 timer
// template.
void TIM6_DAC_IRQHandler(void);

// Interrupt handlers critical for OS operation.
void SVC_Handler(void);
void PendSV_Handler(void);
void SysTick_Handler(void);

// Peripheral interrupts owned by application modules (the mic DMA in
// //modules/mic, USB in //modules/usb_audio). Weak so an app that doesn't
// link one of those drivers still builds; an unexpected interrupt then spins
// in DefaultFaultHandler like any other unhandled exception.
void DMA1_Stream3_IRQHandler(void) PW_ALIAS(DefaultFaultHandler);
void OTG_FS_IRQHandler(void) PW_ALIAS(DefaultFaultHandler);

PW_KEEP_IN_SECTION(".vector_table")
const InterruptHandler vector_table[] = {
    // The starting location of the stack pointer.
    // This address is NOT an interrupt handler/function pointer, it is simply
    // the address that the main stack pointer should be initialized to. The
    // value is reinterpret casted because it needs to be in the vector table.
    [0] = (InterruptHandler)(&pw_boot_stack_high_addr),

    // Reset handler, dictates how to handle reset interrupt. This is the
    // address that the Program Counter (PC) is initialized to at boot.
    [1] = pw_boot_Entry,

    // NMI handler.
    [2] = DefaultFaultHandler,
    // HardFault handler.
    [3] = DefaultFaultHandler,
    // 4-6: Specialized fault handlers.
    // 7-10: Reserved.
    // SVCall handler.
    [11] = SVC_Handler,
    // DebugMon handler.
    [12] = DefaultFaultHandler,
    // 13: Reserved.
    // PendSV handler.
    [14] = PendSV_Handler,
    // SysTick handler.
    [15] = SysTick_Handler,
    // stm32f4xx_hal sys-tick handler.
    [TIM6_DAC_IRQn + 16] = TIM6_DAC_IRQHandler,
    // SPI2/I2S2 RX DMA: the onboard PDM mic.
    [DMA1_Stream3_IRQn + 16] = DMA1_Stream3_IRQHandler,
    // USB OTG FS (micro-USB connector CN5).
    [OTG_FS_IRQn + 16] = OTG_FS_IRQHandler,
};
