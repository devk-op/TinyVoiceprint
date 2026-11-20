// Decides, a couple of times a second, whether the enrolled person is
// speaking.
//
// The audio side calls AddFrame() with each 10 ms log-mel frame. A slower
// task calls Step(), which runs the voiceprint network on the newest 1.5 s
// and compares the result with the enrolled voiceprint. Silence is detected
// from loudness alone, so the network only runs when someone is talking.
//
// AddFrame() and Step() may be called from different tasks, provided the
// AddFrame() caller can preempt Step() and not the other way round.

#pragma once

#include <cstddef>
#include <cstdint>

#include "modules/voiceprint/voiceprint.h"

namespace device_ml::voice_match {

class VoiceMatch {
 public:
  static constexpr size_t kMels = voiceprint::Voiceprint::kMels;
  static constexpr size_t kWindowFrames = voiceprint::Voiceprint::kFrames;
  static constexpr size_t kDim = voiceprint::Voiceprint::kEmbeddingDim;

  // A frame counts as speech when its level is this far above the tracked
  // background (natural-log power, averaged over mel bands; 1.5 is about
  // 6.5 dB), and a window when this share of its frames do.
  static constexpr float kSpeechMargin = 1.5f;
  static constexpr float kSpeechShare = 0.4f;
  // The window also counts as silence once fewer than kTailShare of its
  // newest kTailFrames (0.3 s) are speech, so the decision drops soon after
  // the speaker stops instead of waiting for the window to empty.
  static constexpr size_t kTailFrames = 30;
  static constexpr float kTailShare = 0.2f;
  // Score at or above which a window is attributed to the enrolled voice,
  // provided it is also at least as close to the enrolled voice as to the
  // average voice (model::kAverageVoice).
  // For model tcs_noise, simulated with tools/train/board_sim.py on board
  // recordings (smoothed decisions): VoxCeleb1 replayed through the board,
  // 39 speakers, false reject 8.1% and false accept 22.8% at 0.35 (7.3% and
  // 23.7% without the average-voice check). Live (2026-11-19, free-talk
  // enrolment): a video of another speaker was accepted in 4 of 66 speech
  // decisions; chair squeaks scored 0.35-0.49 and matched without the check
  // (offline the check rejects all of them). Model-specific: re-measure when
  // the model changes.
  static constexpr float kDefaultThreshold = 0.35f;
  // Decisions use the average of this many recent speech windows.
  static constexpr size_t kSmoothWindows = 3;
  // Enrollment averages this many speech windows, and gives up after
  // kEnrollTimeoutSteps steps unless it has at least kEnrollMinWindows.
  static constexpr uint32_t kEnrollWindows = 20;
  static constexpr uint32_t kEnrollMinWindows = 6;
  static constexpr uint32_t kEnrollTimeoutSteps = 60;

  enum class Result : uint8_t {
    kNotReady,  // Less than one window of audio so far.
    kSilence,
    kSpeech,  // Someone is speaking, nobody is enrolled.
    kOther,  // Someone other than the enrolled voice.
    kMatch,  // The enrolled voice.
    kEnrolling,
    kEnrolled,  // Enrollment just completed.
    kEnrollFailed,  // Too little speech before the timeout.
  };

  // One log-mel frame of kMels values.
  void AddFrame(const float* log_mel);

  // Examines the newest window. Takes as long as the network does when
  // there is speech; returns quickly otherwise.
  Result Step();

  // Runs the network on the newest window regardless of silence or
  // enrollment, for timing and energy measurement. Returns false before
  // the first full window.
  bool RunNetworkOnNewestWindow();

  void StartEnrollment();
  void ClearEnrollment();
  // Uses a voiceprint saved earlier (see modules/voice_store) as the
  // enrolled voice.
  void SetEnrollment(const float* voice);
  // The enrolled voiceprint (kDim values, unit length); valid when enrolled().
  const float* enrolled_voice() const { return enrolled_voice_; }

  // The network instance, for running its self-test before audio starts.
  voiceprint::Voiceprint& voiceprint() { return voiceprint_; }

  bool enrolled() const { return enrolled_; }
  bool enrolling() const { return enrolling_; }
  // Smoothed score of the last speech window against the enrolled voice.
  float score() const { return score_; }
  // Latest smoothed score against the average voice.
  float average_score() const { return average_score_; }
  float threshold() const { return threshold_; }
  void set_threshold(float threshold) { threshold_ = threshold; }
  uint32_t frames() const { return frames_; }
  uint32_t enroll_windows() const { return enroll_count_; }

 private:
  // A little longer than the window, so frames that arrive while Step() is
  // reading the window land outside it.
  static constexpr size_t kRingFrames = kWindowFrames + 16;

  void AddRecent(const float* embedding);
  void ResetRecent();

  float ring_[kRingFrames][kMels];
  uint8_t active_[kRingFrames];
  volatile uint32_t frames_ = 0;
  float floor_ = 1e9f;  // Tracked background level.

  voiceprint::Voiceprint voiceprint_;
  float recent_[kSmoothWindows][kDim];
  size_t recent_count_ = 0;
  size_t recent_next_ = 0;

  float enrolled_voice_[kDim];
  float enroll_sum_[kDim];
  uint32_t enroll_count_ = 0;
  uint32_t enroll_steps_ = 0;
  bool enrolling_ = false;
  bool enrolled_ = false;
  float score_ = 0.0f;
  float average_score_ = 0.0f;
  float threshold_ = kDefaultThreshold;
};

}  // namespace device_ml::voice_match
