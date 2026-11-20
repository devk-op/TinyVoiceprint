// Tiny speaker encoder: 1.5 s of log-mel features in, a unit-length
// voiceprint out. Two voiceprints of the same person point in nearly the
// same direction, so their dot product is the match score.
//
// The network (the "tcs" model trained by tools/train/spk.py: 1-D
// time-channel separable convolutions in residual blocks) runs in 8-bit
// integer arithmetic. tools/train/tcs_integer.py generates its weights
// (voiceprint_model.*) and holds the Python reference this code must match
// exactly through the convolutions.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "modules/voiceprint/voiceprint_model.h"

namespace device_ml::voiceprint {

class Voiceprint {
 public:
  static constexpr size_t kFrames = model::kFrames;
  static constexpr size_t kMels = model::kMels;
  static constexpr size_t kEmbeddingDim = model::kEmbeddingDim;

  // log_mel: kFrames rows of kMels values, oldest frame first.
  // embedding: kEmbeddingDim values, unit length.
  void Run(const float* log_mel, float* embedding);

  // The steps of Run(), exposed so tests can check each against the
  // reference. Call in this order.
  void QuantizeInput(const float* log_mel);
  // The same, with the window split in two pieces (as it is in a ring
  // buffer): `first_frames` rows at `first`, then the rest at `second`.
  void QuantizeInput(const float* first,
                     size_t first_frames,
                     const float* second);
  void RunConvolutions();
  void PoolAndEmbed(float* embedding);

  // Valid after QuantizeInput() and until RunConvolutions().
  const int8_t* input() const {
    return reinterpret_cast<const int8_t*>(input_);
  }
  // Valid after RunConvolutions(): (time, channel) order.
  const uint8_t* conv_output() const { return conv_output_; }
  size_t conv_output_bytes() const { return out_time_ * model::kChannels; }

 private:
  static constexpr size_t kBufferBytes =
      std::max({kFrames * kMels, model::kOutFrames * model::kChannels,
                model::kOutFrames * kMels});
  // About 24 KB together. The input buffer is reused once the stem has run.
  uint8_t input_[kBufferBytes];
  uint8_t buffer_x_[kBufferBytes];
  uint8_t buffer_d_[kBufferBytes];
  uint8_t buffer_y_[kBufferBytes];
  const uint8_t* conv_output_ = nullptr;
  size_t out_time_ = 0;
};

// Dot product of two unit-length voiceprints: 1 is identical, around 0 is
// unrelated.
float Similarity(const float* a, const float* b);

}  // namespace device_ml::voiceprint
