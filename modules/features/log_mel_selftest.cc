#include "modules/features/log_mel_selftest.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "modules/features/log_mel_golden.h"

namespace device_ml::features {

static_assert(golden::kNumMels == LogMel::kNumMels);

int16_t GoldenPcmGenerator::Next() {
  lcg_ = lcg_ * 1664525u + 1013904223u;
  const int32_t noise = static_cast<int32_t>((lcg_ >> 16) & 0xFFFF) - 32768;
  phase_a_ = (phase_a_ + 1802) & 0xFFFF;  // ~440 Hz
  phase_b_ = (phase_b_ + 12288) & 0xFFFF;  // 3 kHz
  const int32_t tri_a = std::abs(static_cast<int32_t>(phase_a_) - 32768) - 16384;
  const int32_t tri_b = std::abs(static_cast<int32_t>(phase_b_) - 32768) - 16384;
  // Arithmetic shifts, to match Python's floor division on negatives.
  return static_cast<int16_t>((noise >> 3) + (tri_a >> 1) + (tri_b >> 2));
}

SelfTestResult RunLogMelSelfTest(LogMel& log_mel) {
  log_mel.Reset();
  GoldenPcmGenerator generator;
  SelfTestResult result;
  for (size_t n = 0; n < golden::kNumSamples; ++n) {
    const int16_t sample = generator.Next();
    log_mel.Process(&sample, 1, [&result](const LogMel::Frame& frame) {
      if (result.frames < golden::kNumFrames) {
        for (size_t m = 0; m < LogMel::kNumMels; ++m) {
          const float error =
              std::fabs(frame[m] - golden::kExpected[result.frames][m]);
          result.max_abs_error = std::max(result.max_abs_error, error);
        }
      }
      ++result.frames;
    });
  }
  log_mel.Reset();
  return result;
}

}  // namespace device_ml::features
