// Keeps the enrolled voiceprint in flash, so it survives reset, power-off
// and flashing new firmware.
//
// Storage is one flash sector used as an append-only log of fixed-size
// records. Each save or clear appends a record; the newest valid record
// wins. The sector is erased only when it is full (once every ~160
// enrollments), and Prepare() does that at boot, before audio starts,
// because erasing stalls the CPU for about a second.
//
// Each record carries a tag of the model that produced it. After the model
// changes, an old voiceprint no longer means anything, so Load() ignores
// it and the person enrolls again.
//
// Flash access goes through the Flash interface, so the logic runs in host
// tests with a RAM fake.

#pragma once

#include <cstddef>
#include <cstdint>

namespace device_ml::voice_store {

inline constexpr size_t kDim = 192;  // voiceprint::Voiceprint::kEmbeddingDim

class Flash {
 public:
  virtual ~Flash() = default;
  // The sector, readable as words. Erased words read 0xFFFFFFFF.
  virtual const volatile uint32_t* words() const = 0;
  virtual size_t size_words() const = 0;
  virtual bool Erase() = 0;
  // Programs words that are currently erased.
  virtual bool Program(size_t offset_words, const uint32_t* data, size_t count) = 0;
};

class VoiceStore {
 public:
  // Record layout, in words: magic, model tag, kind, kDim floats, checksum.
  static constexpr size_t kRecordWords = 3 + kDim + 1;
  static constexpr uint32_t kMagic = 0x56505231;  // "VPR1"

  VoiceStore(Flash& flash, uint32_t model_tag) : flash_(flash), model_tag_(model_tag) {}

  // Call once at boot, before anything time-critical runs: finds the newest
  // record and, if no free slot is left (or the sector holds foreign data),
  // erases the sector and rewrites the newest record.
  void Prepare();

  // Copies the stored voiceprint for this model into `voice`; false if
  // there is none (never enrolled, cleared, or enrolled with another model).
  bool Load(float* voice) const;

  bool Save(const float* voice);
  bool Clear();

  size_t free_slots() const { return slots() - next_; }

 private:
  enum Kind : uint32_t { kVoice = 1, kCleared = 2 };

  size_t slots() const { return flash_.size_words() / kRecordWords; }
  bool Valid(size_t slot) const;
  bool Erased(size_t slot) const;
  void Scan();
  bool Append(Kind kind, const float* voice);

  Flash& flash_;
  uint32_t model_tag_;
  size_t next_ = 0;  // First free slot.
  bool have_newest_ = false;
  size_t newest_ = 0;
};

// FNV-1a over raw bytes, used for record checksums and model tags.
uint32_t Fnv1a(const void* data, size_t bytes, uint32_t hash = 2166136261u);

}  // namespace device_ml::voice_store
