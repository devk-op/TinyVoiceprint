// Checks LogMel against golden values from the Python reference
// (tools/host/log_mel_reference.py). Used by the host unit test and run on
// the chip at boot, so the same check covers both builds.

#pragma once

#include <cstddef>
#include <cstdint>

#include "modules/features/log_mel.h"

namespace device_ml::features {

// Regenerates the deterministic test signal of golden_pcm() in the Python
// reference: two triangle-wave tones plus pseudo-random noise, built from
// integer arithmetic only so both languages produce identical samples.
class GoldenPcmGenerator {
 public:
  int16_t Next();

 private:
  uint32_t lcg_ = 12345;
  uint32_t phase_a_ = 0;
  uint32_t phase_b_ = 0;
};

struct SelfTestResult {
  size_t frames = 0;  // Frames produced; should equal golden::kNumFrames.
  float max_abs_error = 0.0f;  // Largest |computed - expected|, in log units.
};

// Runs the golden signal through `log_mel` (which it resets first) and
// compares every output value with the reference.
SelfTestResult RunLogMelSelfTest(LogMel& log_mel);

}  // namespace device_ml::features
