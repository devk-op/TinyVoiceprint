// Checks Voiceprint against the integer reference model in
// tools/train/tcs_integer.py, on one real window of speech. Used by the host
// unit test and run on the chip at boot.

#pragma once

#include <cstdint>

#include "modules/voiceprint/voiceprint.h"

namespace device_ml::voiceprint {

struct SelfTestResult {
  bool input_matches = false;  // int8 input identical to the reference's.
  bool conv_matches = false;  // Last convolution output identical.
  float max_abs_error = 1.0f;  // Largest voiceprint difference, per value.
  float similarity = 0.0f;  // Voiceprint vs the reference's; 1 is identical.
};

SelfTestResult RunVoiceprintSelfTest(Voiceprint& voiceprint);

}  // namespace device_ml::voiceprint
