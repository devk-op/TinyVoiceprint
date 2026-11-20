// The STM32F407's last flash sector (sector 11: 128 KB at 0x080E0000) as a
// voice_store::Flash. The linker script stops the firmware before it, so
// flashing new firmware leaves the stored voiceprint alone.

#pragma once

#include "modules/voice_store/voice_store.h"

namespace device_ml::voice_store {

class Stm32Sector11 : public Flash {
 public:
  static constexpr uint32_t kAddress = 0x080E0000;
  static constexpr size_t kBytes = 128 * 1024;

  const volatile uint32_t* words() const override {
    return reinterpret_cast<const volatile uint32_t*>(kAddress);
  }
  size_t size_words() const override { return kBytes / 4; }
  bool Erase() override;
  bool Program(size_t offset_words, const uint32_t* data, size_t count) override;
};

// Tag for the voiceprint network compiled into this firmware.
uint32_t CurrentModelTag();

}  // namespace device_ml::voice_store
