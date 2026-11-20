// Driver for the STM32F407G-DISC1's onboard accelerometer.
//
// Board revision MB997D (current) carries an ST LIS3DSH; the older MB997B
// carried an LIS302DL with a different register map. Init() reads WHO_AM_I
// and only proceeds on a LIS3DSH, so a wrong-revision board fails loudly
// instead of returning garbage.

#pragma once

#include <cstdint>

#include "pw_status/status.h"

namespace device_ml::accel {

enum class Chip { kUnknown, kLis3dsh, kLis302dl };

struct Sample {
  int32_t x_mg;
  int32_t y_mg;
  int32_t z_mg;
};

// At the default +/-2g full scale the LIS3DSH reports 0.06 mg per LSB.
// Kept free of hardware access so it can be unit tested on the host.
constexpr int32_t RawToMilliG(int16_t raw) {
  return (static_cast<int32_t>(raw) * 6) / 100;
}

class Lis3dsh {
 public:
  // Sets up SPI1 and the chip-select GPIO, checks WHO_AM_I, and configures
  // 100 Hz output with block data update and all three axes enabled.
  pw::Status Init();

  // Reads one X/Y/Z sample. Init() must have succeeded first.
  pw::Status Read(Sample& out);

  Chip chip() const { return chip_; }

 private:
  pw::Status ReadRegister(uint8_t reg, uint8_t& value);
  pw::Status WriteRegister(uint8_t reg, uint8_t value);

  Chip chip_ = Chip::kUnknown;
};

}  // namespace device_ml::accel
