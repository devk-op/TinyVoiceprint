#include "modules/voice_store/voice_store.h"

#include <cstring>

namespace device_ml::voice_store {

uint32_t Fnv1a(const void* data, size_t bytes, uint32_t hash) {
  const auto* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    hash = (hash ^ p[i]) * 16777619u;
  }
  return hash;
}

namespace {

uint32_t RecordChecksum(const volatile uint32_t* record) {
  uint32_t copy[VoiceStore::kRecordWords - 1];
  for (size_t i = 0; i < VoiceStore::kRecordWords - 1; ++i) {
    copy[i] = record[i];
  }
  return Fnv1a(copy, sizeof(copy));
}

}  // namespace

bool VoiceStore::Valid(size_t slot) const {
  const volatile uint32_t* r = flash_.words() + slot * kRecordWords;
  return r[0] == kMagic && r[kRecordWords - 1] == RecordChecksum(r);
}

bool VoiceStore::Erased(size_t slot) const {
  const volatile uint32_t* r = flash_.words() + slot * kRecordWords;
  for (size_t i = 0; i < kRecordWords; ++i) {
    if (r[i] != 0xFFFFFFFFu) {
      return false;
    }
  }
  return true;
}

void VoiceStore::Scan() {
  have_newest_ = false;
  next_ = slots();
  // Records are appended in order, so everything after the first fully
  // erased slot is erased too. A slot that is neither valid nor erased
  // (an interrupted write, or data from other firmware) is skipped.
  for (size_t slot = 0; slot < slots(); ++slot) {
    if (Erased(slot)) {
      next_ = slot;
      return;
    }
    if (Valid(slot)) {
      have_newest_ = true;
      newest_ = slot;
    }
  }
}

void VoiceStore::Prepare() {
  Scan();
  if (next_ < slots()) {
    return;
  }
  // Full: keep only the newest record.
  uint32_t keep[kRecordWords];
  const bool keep_it = have_newest_;
  if (keep_it) {
    const volatile uint32_t* r = flash_.words() + newest_ * kRecordWords;
    for (size_t i = 0; i < kRecordWords; ++i) {
      keep[i] = r[i];
    }
  }
  if (!flash_.Erase()) {
    return;
  }
  if (keep_it) {
    flash_.Program(0, keep, kRecordWords);
  }
  Scan();
}

bool VoiceStore::Load(float* voice) const {
  if (!have_newest_) {
    return false;
  }
  const volatile uint32_t* r = flash_.words() + newest_ * kRecordWords;
  if (r[1] != model_tag_ || r[2] != kVoice) {
    return false;
  }
  for (size_t i = 0; i < kDim; ++i) {
    const uint32_t word = r[3 + i];
    std::memcpy(&voice[i], &word, sizeof(word));
  }
  return true;
}

bool VoiceStore::Append(Kind kind, const float* voice) {
  if (next_ >= slots()) {
    // Only if ~160 saves happen without a reboot: erase now, which stalls
    // the CPU (and audio) for about a second. The record about to be
    // written replaces whatever was there.
    if (!flash_.Erase()) {
      return false;
    }
    Scan();
  }
  uint32_t record[kRecordWords] = {};
  record[0] = kMagic;
  record[1] = model_tag_;
  record[2] = kind;
  if (voice != nullptr) {
    std::memcpy(&record[3], voice, kDim * sizeof(float));
  }
  record[kRecordWords - 1] = Fnv1a(record, (kRecordWords - 1) * sizeof(uint32_t));
  const size_t slot = next_;
  ++next_;  // Even if programming fails, never write this slot twice.
  if (!flash_.Program(slot * kRecordWords, record, kRecordWords) || !Valid(slot)) {
    return false;
  }
  have_newest_ = true;
  newest_ = slot;
  return true;
}

bool VoiceStore::Save(const float* voice) { return Append(kVoice, voice); }

bool VoiceStore::Clear() { return Append(kCleared, nullptr); }

}  // namespace device_ml::voice_store
