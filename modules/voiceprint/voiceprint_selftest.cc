#include "modules/voiceprint/voiceprint_selftest.h"

#include <algorithm>
#include <cmath>

#include "modules/voiceprint/voiceprint_golden.h"

namespace device_ml::voiceprint {
namespace {

uint32_t Fnv1a(const uint8_t* data, size_t size) {
  uint32_t hash = 0x811C9DC5u;
  for (size_t i = 0; i < size; ++i) {
    hash = (hash ^ data[i]) * 0x01000193u;
  }
  return hash;
}

}  // namespace

SelfTestResult RunVoiceprintSelfTest(Voiceprint& voiceprint) {
  SelfTestResult result;
  float embedding[Voiceprint::kEmbeddingDim];

  voiceprint.QuantizeInput(golden::kLogMel);
  result.input_matches =
      Fnv1a(reinterpret_cast<const uint8_t*>(voiceprint.input()),
            Voiceprint::kFrames * Voiceprint::kMels) == golden::kInputHash;
  voiceprint.RunConvolutions();
  result.conv_matches =
      voiceprint.conv_output_bytes() == golden::kConvOutputBytes &&
      Fnv1a(voiceprint.conv_output(), voiceprint.conv_output_bytes()) ==
          golden::kConvOutputHash;
  voiceprint.PoolAndEmbed(embedding);

  result.max_abs_error = 0.0f;
  for (size_t i = 0; i < Voiceprint::kEmbeddingDim; ++i) {
    result.max_abs_error = std::max(
        result.max_abs_error, std::fabs(embedding[i] - golden::kEmbedding[i]));
  }
  result.similarity = Similarity(embedding, golden::kEmbedding);
  return result;
}

}  // namespace device_ml::voiceprint
