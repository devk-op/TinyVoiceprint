#include "modules/voice_match/voice_match.h"

#include <algorithm>
#include <cmath>

#include "modules/voiceprint/voiceprint_model.h"

namespace device_ml::voice_match {
namespace {

void Normalize(float* v, size_t n) {
  float norm = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    norm += v[i] * v[i];
  }
  const float inverse = 1.0f / std::sqrt(norm + 1e-12f);
  for (size_t i = 0; i < n; ++i) {
    v[i] *= inverse;
  }
}

}  // namespace

void VoiceMatch::AddFrame(const float* log_mel) {
  const size_t slot = frames_ % kRingFrames;
  float level = 0.0f;
  for (size_t m = 0; m < kMels; ++m) {
    ring_[slot][m] = log_mel[m];
    level += log_mel[m];
  }
  level /= static_cast<float>(kMels);
  // The background level follows quiet frames down at once and creeps up
  // slowly (about 0.2 dB a second), so speech doesn't drag it up.
  floor_ = std::min(floor_ + 0.0005f, level);
  active_[slot] = level > floor_ + kSpeechMargin ? 1 : 0;
  frames_ = frames_ + 1;  // Publish the frame last.
}

void VoiceMatch::AddRecent(const float* embedding) {
  std::copy_n(embedding, kDim, recent_[recent_next_]);
  recent_next_ = (recent_next_ + 1) % kSmoothWindows;
  recent_count_ = std::min(recent_count_ + 1, kSmoothWindows);
}

void VoiceMatch::ResetRecent() {
  recent_count_ = 0;
  recent_next_ = 0;
}

VoiceMatch::Result VoiceMatch::Step() {
  const uint32_t newest = frames_;
  if (newest < kWindowFrames) {
    return Result::kNotReady;
  }
  if (enrolling_ && ++enroll_steps_ > kEnrollTimeoutSteps) {
    enrolling_ = false;
    if (enroll_count_ < kEnrollMinWindows) {
      return Result::kEnrollFailed;
    }
    std::copy_n(enroll_sum_, kDim, enrolled_voice_);
    Normalize(enrolled_voice_, kDim);
    enrolled_ = true;
    ResetRecent();
    return Result::kEnrolled;
  }

  const size_t start = (newest - kWindowFrames) % kRingFrames;
  const size_t first_frames = std::min(kWindowFrames, kRingFrames - start);
  uint32_t active = 0;
  uint32_t tail_active = 0;
  for (size_t i = 0; i < kWindowFrames; ++i) {
    const uint8_t frame_active = active_[(start + i) % kRingFrames];
    active += frame_active;
    if (i >= kWindowFrames - kTailFrames) {
      tail_active += frame_active;
    }
  }
  if (static_cast<float>(active) <
          kSpeechShare * static_cast<float>(kWindowFrames) ||
      static_cast<float>(tail_active) <
          kTailShare * static_cast<float>(kTailFrames)) {
    ResetRecent();
    return enrolling_ ? Result::kEnrolling : Result::kSilence;
  }

  float embedding[kDim];
  voiceprint_.QuantizeInput(ring_[start], first_frames, ring_[0]);
  voiceprint_.RunConvolutions();
  voiceprint_.PoolAndEmbed(embedding);

  if (enrolling_) {
    for (size_t i = 0; i < kDim; ++i) {
      enroll_sum_[i] += embedding[i];
    }
    if (++enroll_count_ < kEnrollWindows) {
      return Result::kEnrolling;
    }
    enroll_steps_ = kEnrollTimeoutSteps;  // Finish on the next step.
    return Result::kEnrolling;
  }
  if (!enrolled_) {
    return Result::kSpeech;
  }

  AddRecent(embedding);
  float smoothed[kDim] = {};
  for (size_t w = 0; w < recent_count_; ++w) {
    for (size_t i = 0; i < kDim; ++i) {
      smoothed[i] += recent_[w][i];
    }
  }
  Normalize(smoothed, kDim);
  score_ = voiceprint::Similarity(smoothed, enrolled_voice_);
  // Loud non-speech (a chair squeak) passes the loudness gate and comes out
  // near the average voiceprint, which sits fairly close to everyone; such a
  // window is closer to the average than to the enrolled voice.
  average_score_ =
      voiceprint::Similarity(smoothed, voiceprint::model::kAverageVoice);
  return score_ >= threshold_ && score_ >= average_score_ ? Result::kMatch
                                                          : Result::kOther;
}

bool VoiceMatch::RunNetworkOnNewestWindow() {
  const uint32_t newest = frames_;
  if (newest < kWindowFrames) {
    return false;
  }
  const size_t start = (newest - kWindowFrames) % kRingFrames;
  const size_t first_frames = std::min(kWindowFrames, kRingFrames - start);
  float embedding[kDim];
  voiceprint_.QuantizeInput(ring_[start], first_frames, ring_[0]);
  voiceprint_.RunConvolutions();
  voiceprint_.PoolAndEmbed(embedding);
  return true;
}

void VoiceMatch::StartEnrollment() {
  std::fill_n(enroll_sum_, kDim, 0.0f);
  enroll_count_ = 0;
  enroll_steps_ = 0;
  enrolling_ = true;
}

void VoiceMatch::SetEnrollment(const float* voice) {
  std::copy_n(voice, kDim, enrolled_voice_);
  Normalize(enrolled_voice_, kDim);
  enrolling_ = false;
  enrolled_ = true;
  ResetRecent();
  score_ = 0.0f;
}

void VoiceMatch::ClearEnrollment() {
  enrolling_ = false;
  enrolled_ = false;
  ResetRecent();
  score_ = 0.0f;
}

}  // namespace device_ml::voice_match
