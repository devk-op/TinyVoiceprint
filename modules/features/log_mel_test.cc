#include "modules/features/log_mel.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "modules/features/log_mel_golden.h"
#include "modules/features/log_mel_selftest.h"
#include "pw_unit_test/framework.h"

namespace device_ml::features {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<int16_t> Tone(double hz, double amplitude, size_t count) {
  std::vector<int16_t> pcm(count);
  for (size_t n = 0; n < count; ++n) {
    pcm[n] = static_cast<int16_t>(
        std::lrint(amplitude * 32767.0 *
                   std::sin(2.0 * kPi * hz * n / LogMel::kSampleRate)));
  }
  return pcm;
}

std::vector<LogMel::Frame> Extract(LogMel& log_mel,
                                   const std::vector<int16_t>& pcm) {
  std::vector<LogMel::Frame> frames;
  log_mel.Process(pcm.data(), pcm.size(), [&frames](const LogMel::Frame& f) {
    frames.push_back(f);
  });
  return frames;
}

TEST(LogMel, MatchesPythonReference) {
  LogMel log_mel;
  const SelfTestResult result = RunLogMelSelfTest(log_mel);
  EXPECT_EQ(result.frames, golden::kNumFrames);
  // float32 here vs float64 in the reference. Measured error is ~2e-6 on
  // features spanning about -2.4..+6.2; this leaves room for libm differences
  // between platforms while still catching any real mismatch.
  EXPECT_LT(result.max_abs_error, 1e-4f);
}

TEST(LogMel, FrameTiming) {
  LogMel log_mel;
  // One frame after kFrameLength samples, then one per kHop.
  EXPECT_EQ(Extract(log_mel, std::vector<int16_t>(399)).size(), 0u);
  EXPECT_EQ(Extract(log_mel, std::vector<int16_t>(1)).size(), 1u);
  EXPECT_EQ(Extract(log_mel, std::vector<int16_t>(159)).size(), 0u);
  EXPECT_EQ(Extract(log_mel, std::vector<int16_t>(1)).size(), 1u);
  EXPECT_EQ(Extract(log_mel, std::vector<int16_t>(1600)).size(), 10u);
}

TEST(LogMel, SilenceGivesLogFloor) {
  LogMel log_mel;
  const auto frames = Extract(log_mel, std::vector<int16_t>(800));
  ASSERT_FALSE(frames.empty());
  for (float value : frames.back()) {
    EXPECT_NEAR(value, std::log(1e-6f), 1e-4f);
  }
}

TEST(LogMel, HigherToneLandsInHigherBand) {
  auto peak_band = [](double hz) {
    LogMel log_mel;
    const auto frames = Extract(log_mel, Tone(hz, 0.5, 1600));
    const LogMel::Frame& frame = frames.back();
    size_t best = 0;
    for (size_t m = 1; m < LogMel::kNumMels; ++m) {
      if (frame[m] > frame[best]) {
        best = m;
      }
    }
    return best;
  };
  const size_t low = peak_band(300.0);
  const size_t mid = peak_band(1500.0);
  const size_t high = peak_band(5000.0);
  EXPECT_LT(low, mid);
  EXPECT_LT(mid, high);
}

TEST(LogMel, StreamingInChunksMatchesOneCall) {
  const auto pcm = Tone(700.0, 0.3, 2000);
  LogMel whole;
  const auto expected = Extract(whole, pcm);

  LogMel chunked;
  std::vector<LogMel::Frame> actual;
  // 32 samples at a time, as the firmware's 2 ms audio blocks deliver them.
  for (size_t offset = 0; offset < pcm.size(); offset += 32) {
    chunked.Process(&pcm[offset], 32, [&actual](const LogMel::Frame& f) {
      actual.push_back(f);
    });
  }
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]);
  }
}

}  // namespace
}  // namespace device_ml::features
