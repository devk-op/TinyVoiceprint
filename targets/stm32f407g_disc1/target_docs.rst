.. _target-stm32f407g-disc1:

=================
stm32f407g_disc1
=================
Target support for the STM32F407G-DISC1 (STM32F407VGT6, Cortex-M4F, 168MHz,
1MB flash, 128KB contiguous SRAM + 64KB non-contiguous CCM RAM), forked from
upstream Pigweed's ``targets/stm32f429i_disc1_stm32cube``. Uses STM32Cube
HAL + FreeRTOS + ``pw_system``, built with Bazel.

Differences from the F429 target this was forked from
=======================================================
- ``STM32F407xx`` product define instead of ``STM32F429xx`` (same
  ``cmsis_device_f4`` / ``stm32f4xx_hal_driver`` packages cover both --
  F405/415/407/417/427/437/429/439 share the RM0090 reference manual).
- 1MB flash linker region (``PW_BOOT_FLASH_SIZE=1024K``), not 2MB -- the
  F407VG has 1MB flash, unlike the F429ZI/ZG die the upstream target assumes.
- ``SystemClock_Config()`` targets 168MHz (the F407's max), not 180MHz, with
  different PLL dividers and no over-drive step -- that PWR feature doesn't
  exist on the F407 die at all.
- ``config/stm32f4xx_hal_conf.h`` only enables the HAL modules this project
  needs (core/clock/GPIO/DMA/timer/UART/I2C/I2S/CRC), not nearly everything
  like the F429 target's config does. Several modules the F429 config enables
  (LTDC, DSI, DMA2D, SDRAM, SAI, QSPI, FMPI2C/FMPSMBUS, DFSDM, CEC, LPTIM)
  don't exist on the F407 die and would fail to compile if enabled here.
- ``PW_BOOT_RAM_SIZE=128K``, not 192K. The F429 has 192K genuinely
  contiguous at ``0x20000000`` (SRAM1+SRAM2+SRAM3). The F407 only has 128K
  contiguous there (SRAM1+SRAM2); its remaining 64K "CCM" RAM lives at the
  separate, non-contiguous, non-DMA-accessible address ``0x10000000`` and
  isn't usable as a simple extension of the main region. Using 192K here
  (initially copied uncritically from the F429 target) put the linker
  script's stack region 64K past the end of real memory and hard-faulted on
  the very first stack push during boot -- caught by flashing to real
  hardware and diagnosing the resulting HardFault via GDB/OpenOCD (BFAR
  pointed squarely into the nonexistent region once precise bus faults were
  forced via ``ACTLR.DISDEFWBUF``).

Flashing
========
.. code-block:: sh

   bazel build --config=stm32f407g_disc1 //modules/app
   targets/stm32f407g_disc1/py/flash.py bazel-bin/modules/app/app

Open items
==========
- HSE crystal (8MHz assumed) and CS43L22/I2S pin wiring are taken from the
  well-known F4-Discovery schematic, not re-verified against this specific
  unit -- flag any clock startup or codec I2C failures for a schematic
  recheck.
- No on-device unit test wiring yet (unlike the F429 target's
  ``unit_test_app``); add ``pw_unit_test`` backend/main flags here if that
  becomes useful.
- The 64K CCM RAM at ``0x10000000`` is currently unused. It's core-only (no
  DMA), which makes it a poor fit for PDM/I2S DMA buffers but a reasonable
  place for the FreeRTOS stack region or scratch DSP buffers later, if the
  128K main SRAM gets tight.
