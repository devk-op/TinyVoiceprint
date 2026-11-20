// Driver for the STM32F407G-DISC1's onboard MP45DT02 PDM MEMS microphone.
//
// The mic is wired to I2S2 (CLK on PB10, DATA on PC3). I2S2 runs as a master
// receiver purely to generate a 1.024 MHz clock and shift the 1-bit stream
// into 16-bit words, which DMA1 Stream 3 copies into a circular buffer. Each
// half of that buffer holds 2 ms of PDM; the DMA interrupt wakes the waiting
// task, which hands back that half for decimation.

#pragma once

#include <cstddef>
#include <cstdint>

#include "pw_status/status.h"

namespace device_ml::mic {

class Mp45dt02 {
 public:
  // PDM words in each block returned by WaitForBlock(): 2 ms at 1.024 MHz.
  static constexpr size_t kBlockWords = 128;

  // Configures the I2S clock (PLLI2S), pins, I2S2 and DMA, then starts
  // continuous capture. Must be called from the task that will call
  // WaitForBlock(), after the scheduler has started.
  pw::Status Start();

  // Blocks until the next 2 ms block of PDM is ready and returns it. The
  // block stays valid for 2 ms, until DMA wraps around and overwrites it.
  // Returns nullptr on timeout (capture stalled).
  const uint16_t* WaitForBlock();

  // Blocks that were overwritten before WaitForBlock() picked them up,
  // meaning the consumer fell behind. Should stay at zero.
  uint32_t dropped_blocks() const { return dropped_blocks_; }

 private:
  uint32_t dropped_blocks_ = 0;
};

}  // namespace device_ml::mic
