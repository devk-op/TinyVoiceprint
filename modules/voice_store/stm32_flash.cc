#include "modules/voice_store/stm32_flash.h"

#include "modules/voiceprint/voiceprint_model.h"
#include "stm32f4xx.h"

namespace device_ml::voice_store {
namespace {

constexpr uint32_t kSector = 11;
constexpr uint32_t kErrors = FLASH_SR_SOP | FLASH_SR_WRPERR | FLASH_SR_PGAERR | FLASH_SR_PGPERR | FLASH_SR_PGSERR;

void WaitIdle() {
  while (FLASH->SR & FLASH_SR_BSY) {
  }
}

void Unlock() {
  WaitIdle();
  if (FLASH->CR & FLASH_CR_LOCK) {
    FLASH->KEYR = 0x45670123;
    FLASH->KEYR = 0xCDEF89AB;
  }
  FLASH->SR = kErrors | FLASH_SR_EOP;  // Clear stale flags (write 1 to clear).
}

void Lock() { FLASH->CR |= FLASH_CR_LOCK; }

// The flash data cache can hold words read before an erase or write.
void ResetDataCache() {
  FLASH->ACR &= ~FLASH_ACR_DCEN;
  FLASH->ACR |= FLASH_ACR_DCRST;
  FLASH->ACR &= ~FLASH_ACR_DCRST;
  FLASH->ACR |= FLASH_ACR_DCEN;
}

}  // namespace

bool Stm32Sector11::Erase() {
  Unlock();
  // 32-bit parallelism (valid at the board's 3 V supply).
  FLASH->CR = FLASH_CR_PSIZE_1 | FLASH_CR_SER | (kSector << FLASH_CR_SNB_Pos);
  FLASH->CR |= FLASH_CR_STRT;
  WaitIdle();
  const bool ok = (FLASH->SR & kErrors) == 0;
  FLASH->CR = 0;
  Lock();
  ResetDataCache();
  return ok;
}

bool Stm32Sector11::Program(size_t offset_words, const uint32_t* data, size_t count) {
  if (offset_words + count > size_words()) {
    return false;
  }
  Unlock();
  FLASH->CR = FLASH_CR_PSIZE_1 | FLASH_CR_PG;
  volatile uint32_t* dest = reinterpret_cast<volatile uint32_t*>(kAddress) + offset_words;
  bool ok = true;
  for (size_t i = 0; i < count && ok; ++i) {
    dest[i] = data[i];
    WaitIdle();
    ok = (FLASH->SR & kErrors) == 0;
  }
  FLASH->CR = 0;
  Lock();
  ResetDataCache();
  return ok;
}

uint32_t CurrentModelTag() {
  // Any retrained or re-exported model has different weights, so a hash of
  // the final layer's bias and the input scale tells models apart.
  uint32_t tag = Fnv1a(voiceprint::model::kFcBias, sizeof(voiceprint::model::kFcBias));
  return Fnv1a(&voiceprint::model::kInputScale, sizeof(voiceprint::model::kInputScale), tag);
}

}  // namespace device_ml::voice_store
