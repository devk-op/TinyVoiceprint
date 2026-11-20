#include "modules/voiceprint/voiceprint.h"

#include "modules/voiceprint/voiceprint_golden.h"
#include "modules/voiceprint/voiceprint_selftest.h"
#include "pw_unit_test/framework.h"

namespace device_ml::voiceprint {
namespace {

Voiceprint voiceprint;  // Too large for the test's stack.

TEST(Voiceprint, MatchesIntegerReference) {
  const SelfTestResult result = RunVoiceprintSelfTest(voiceprint);
  EXPECT_TRUE(result.input_matches);
  EXPECT_TRUE(result.conv_matches);
  EXPECT_LT(result.max_abs_error, 1e-4f);
  EXPECT_GT(result.similarity, 0.9999f);
}

TEST(Voiceprint, OutputIsUnitLength) {
  float embedding[Voiceprint::kEmbeddingDim];
  voiceprint.Run(golden::kLogMel, embedding);
  EXPECT_NEAR(Similarity(embedding, embedding), 1.0f, 1e-4f);
}

TEST(Voiceprint, IgnoresConstantOffsetPerBand) {
  // A fixed gain or microphone response adds a constant to each mel band in
  // the log domain; the mean subtraction should remove it.
  static float shifted[Voiceprint::kFrames * Voiceprint::kMels];
  for (size_t t = 0; t < Voiceprint::kFrames; ++t) {
    for (size_t m = 0; m < Voiceprint::kMels; ++m) {
      const size_t i = t * Voiceprint::kMels + m;
      shifted[i] = golden::kLogMel[i] + 0.1f * static_cast<float>(m) - 2.0f;
    }
  }
  float original[Voiceprint::kEmbeddingDim];
  float moved[Voiceprint::kEmbeddingDim];
  voiceprint.Run(golden::kLogMel, original);
  voiceprint.Run(shifted, moved);
  EXPECT_GT(Similarity(original, moved), 0.99f);
}

}  // namespace
}  // namespace device_ml::voiceprint
