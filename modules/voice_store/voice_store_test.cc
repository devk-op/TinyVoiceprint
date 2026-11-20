#include "modules/voice_store/voice_store.h"

#include <algorithm>
#include <vector>

#include "pw_unit_test/framework.h"

namespace device_ml::voice_store {
namespace {

// Behaves like NOR flash: erase sets all bits, programming can only clear
// bits, and programming a word twice is an error.
class FakeFlash : public Flash {
 public:
  explicit FakeFlash(size_t words) : data_(words, 0xFFFFFFFFu) {}
  const volatile uint32_t* words() const override { return data_.data(); }
  size_t size_words() const override { return data_.size(); }
  bool Erase() override {
    std::fill(data_.begin(), data_.end(), 0xFFFFFFFFu);
    ++erases;
    return true;
  }
  bool Program(size_t offset, const uint32_t* data, size_t count) override {
    if (fail_program) {
      return false;
    }
    for (size_t i = 0; i < count; ++i) {
      if (data_[offset + i] != 0xFFFFFFFFu) {
        return false;
      }
      data_[offset + i] = data[i];
    }
    return true;
  }
  uint32_t& at(size_t i) { return data_[i]; }

  int erases = 0;
  bool fail_program = false;

 private:
  std::vector<uint32_t> data_;
};

std::vector<float> Voice(float seed) {
  std::vector<float> v(kDim);
  for (size_t i = 0; i < kDim; ++i) {
    v[i] = seed + 0.001f * static_cast<float>(i);
  }
  return v;
}

constexpr size_t kSlots = 5;
constexpr uint32_t kTag = 1234;

TEST(VoiceStore, EmptyHasNothing) {
  FakeFlash flash(kSlots * VoiceStore::kRecordWords);
  VoiceStore store(flash, kTag);
  store.Prepare();
  float out[kDim];
  EXPECT_FALSE(store.Load(out));
  EXPECT_EQ(flash.erases, 0);
}

TEST(VoiceStore, SurvivesRestart) {
  FakeFlash flash(kSlots * VoiceStore::kRecordWords);
  const auto voice = Voice(0.5f);
  {
    VoiceStore store(flash, kTag);
    store.Prepare();
    ASSERT_TRUE(store.Save(voice.data()));
  }
  VoiceStore after_reboot(flash, kTag);
  after_reboot.Prepare();
  float out[kDim];
  ASSERT_TRUE(after_reboot.Load(out));
  EXPECT_TRUE(std::equal(out, out + kDim, voice.begin()));
}

TEST(VoiceStore, NewestWinsAndClearForgets) {
  FakeFlash flash(kSlots * VoiceStore::kRecordWords);
  VoiceStore store(flash, kTag);
  store.Prepare();
  ASSERT_TRUE(store.Save(Voice(0.1f).data()));
  ASSERT_TRUE(store.Save(Voice(0.2f).data()));
  float out[kDim];
  ASSERT_TRUE(store.Load(out));
  EXPECT_FLOAT_EQ(out[0], 0.2f);
  ASSERT_TRUE(store.Clear());
  EXPECT_FALSE(store.Load(out));
  VoiceStore after_reboot(flash, kTag);
  after_reboot.Prepare();
  EXPECT_FALSE(after_reboot.Load(out));
}

TEST(VoiceStore, OtherModelIsIgnored) {
  FakeFlash flash(kSlots * VoiceStore::kRecordWords);
  VoiceStore old_model(flash, kTag);
  old_model.Prepare();
  ASSERT_TRUE(old_model.Save(Voice(0.3f).data()));
  VoiceStore new_model(flash, kTag + 1);
  new_model.Prepare();
  float out[kDim];
  EXPECT_FALSE(new_model.Load(out));
}

TEST(VoiceStore, FullSectorIsCompactedAtBoot) {
  FakeFlash flash(kSlots * VoiceStore::kRecordWords);
  {
    VoiceStore store(flash, kTag);
    store.Prepare();
    for (size_t i = 0; i < kSlots; ++i) {
      ASSERT_TRUE(store.Save(Voice(static_cast<float>(i)).data()));
    }
    EXPECT_EQ(store.free_slots(), 0u);
  }
  VoiceStore after_reboot(flash, kTag);
  after_reboot.Prepare();
  EXPECT_EQ(flash.erases, 1);
  EXPECT_EQ(after_reboot.free_slots(), kSlots - 1);
  float out[kDim];
  ASSERT_TRUE(after_reboot.Load(out));
  EXPECT_FLOAT_EQ(out[0], static_cast<float>(kSlots - 1));
}

TEST(VoiceStore, SaveWhenFullStillWorks) {
  FakeFlash flash(kSlots * VoiceStore::kRecordWords);
  VoiceStore store(flash, kTag);
  store.Prepare();
  for (size_t i = 0; i <= kSlots; ++i) {
    ASSERT_TRUE(store.Save(Voice(static_cast<float>(i)).data()));
  }
  float out[kDim];
  ASSERT_TRUE(store.Load(out));
  EXPECT_FLOAT_EQ(out[0], static_cast<float>(kSlots));
}

TEST(VoiceStore, ForeignDataIsErasedAtBoot) {
  FakeFlash flash(kSlots * VoiceStore::kRecordWords);
  for (size_t i = 0; i < flash.size_words(); ++i) {
    flash.at(i) = 0x12345678u;  // Left by some other firmware.
  }
  VoiceStore store(flash, kTag);
  store.Prepare();
  EXPECT_EQ(flash.erases, 1);
  float out[kDim];
  EXPECT_FALSE(store.Load(out));
  ASSERT_TRUE(store.Save(Voice(0.7f).data()));
  ASSERT_TRUE(store.Load(out));
}

TEST(VoiceStore, CorruptRecordFallsBackToPreviousOne) {
  FakeFlash flash(kSlots * VoiceStore::kRecordWords);
  {
    VoiceStore store(flash, kTag);
    store.Prepare();
    ASSERT_TRUE(store.Save(Voice(0.1f).data()));
    ASSERT_TRUE(store.Save(Voice(0.2f).data()));
  }
  flash.at(VoiceStore::kRecordWords + 10) &= 0x0000FFFFu;  // Damage the newest.
  VoiceStore after_reboot(flash, kTag);
  after_reboot.Prepare();
  float out[kDim];
  ASSERT_TRUE(after_reboot.Load(out));
  EXPECT_FLOAT_EQ(out[0], 0.1f);
}

TEST(VoiceStore, FailedWriteReportsFailure) {
  FakeFlash flash(kSlots * VoiceStore::kRecordWords);
  VoiceStore store(flash, kTag);
  store.Prepare();
  flash.fail_program = true;
  EXPECT_FALSE(store.Save(Voice(0.4f).data()));
  float out[kDim];
  EXPECT_FALSE(store.Load(out));
}

}  // namespace
}  // namespace device_ml::voice_store
