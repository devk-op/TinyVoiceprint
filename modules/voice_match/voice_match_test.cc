#include "modules/voice_match/voice_match.h"

#include <algorithm>

#include "modules/voiceprint/voiceprint_golden.h"
#include "pw_unit_test/framework.h"

namespace device_ml::voice_match {
namespace {

using Result = VoiceMatch::Result;

constexpr size_t kMels = VoiceMatch::kMels;
constexpr size_t kWindow = VoiceMatch::kWindowFrames;

void AddQuiet(VoiceMatch& match, size_t frames) {
  float quiet[kMels];
  for (float& value : quiet) {
    value = -12.0f;
  }
  for (size_t i = 0; i < frames; ++i) {
    match.AddFrame(quiet);
  }
}

// The golden window of real speech, optionally with the mel bands reversed
// to stand in for a different voice.
void AddSpeech(VoiceMatch& match, bool reversed = false) {
  for (size_t t = 0; t < kWindow; ++t) {
    float frame[kMels];
    for (size_t m = 0; m < kMels; ++m) {
      const size_t source = reversed ? kMels - 1 - m : m;
      frame[m] = voiceprint::golden::kLogMel[t * kMels + source];
    }
    match.AddFrame(frame);
  }
}

void Enroll(VoiceMatch& match) {
  match.StartEnrollment();
  Result result = Result::kEnrolling;
  for (uint32_t i = 0; i < VoiceMatch::kEnrollWindows + 2; ++i) {
    AddSpeech(match);
    result = match.Step();
    if (result != Result::kEnrolling) {
      break;
    }
  }
  ASSERT_EQ(result, Result::kEnrolled);
}

VoiceMatch match;  // Too large for the test's stack.

TEST(VoiceMatch, NotReadyUntilOneWindow) {
  match = VoiceMatch();
  AddQuiet(match, kWindow - 1);
  EXPECT_EQ(match.Step(), Result::kNotReady);
}

TEST(VoiceMatch, QuietIsSilence) {
  match = VoiceMatch();
  AddQuiet(match, 2 * kWindow);
  EXPECT_EQ(match.Step(), Result::kSilence);
}

TEST(VoiceMatch, SpeechWithoutEnrollment) {
  match = VoiceMatch();
  AddQuiet(match, kWindow);
  AddSpeech(match);
  EXPECT_EQ(match.Step(), Result::kSpeech);
}

TEST(VoiceMatch, EnrolledVoiceMatchesAndAnotherDoesNot) {
  match = VoiceMatch();
  AddQuiet(match, kWindow);
  Enroll(match);
  EXPECT_TRUE(match.enrolled());

  AddSpeech(match);
  EXPECT_EQ(match.Step(), Result::kMatch);
  EXPECT_GT(match.score(), 0.99f);

  AddQuiet(match, kWindow);
  EXPECT_EQ(match.Step(), Result::kSilence);

  AddSpeech(match, /*reversed=*/true);
  EXPECT_EQ(match.Step(), Result::kOther);
  EXPECT_LT(match.score(), VoiceMatch::kDefaultThreshold);
}

TEST(VoiceMatch, RestoredEnrollmentMatchesLikeTheOriginal) {
  match = VoiceMatch();
  AddQuiet(match, kWindow);
  Enroll(match);
  float saved[VoiceMatch::kDim];
  std::copy_n(match.enrolled_voice(), VoiceMatch::kDim, saved);

  match = VoiceMatch();  // As after a reset.
  AddQuiet(match, kWindow);
  EXPECT_FALSE(match.enrolled());
  match.SetEnrollment(saved);
  EXPECT_TRUE(match.enrolled());
  AddSpeech(match);
  EXPECT_EQ(match.Step(), Result::kMatch);
  EXPECT_GT(match.score(), 0.99f);
  AddQuiet(match, kWindow);
  EXPECT_EQ(match.Step(), Result::kSilence);
  AddSpeech(match, /*reversed=*/true);
  EXPECT_EQ(match.Step(), Result::kOther);
}

TEST(VoiceMatch, SilenceSoonAfterSpeechStops) {
  match = VoiceMatch();
  AddQuiet(match, kWindow);
  AddSpeech(match);
  EXPECT_EQ(match.Step(), Result::kSpeech);
  // 0.4 s of quiet: most of the window is still speech, but its end isn't.
  AddQuiet(match, 40);
  EXPECT_EQ(match.Step(), Result::kSilence);
}

TEST(VoiceMatch, EnrollmentFailsWithoutSpeech) {
  match = VoiceMatch();
  AddQuiet(match, kWindow);
  match.StartEnrollment();
  Result result = Result::kEnrolling;
  for (uint32_t i = 0; i <= VoiceMatch::kEnrollTimeoutSteps + 1; ++i) {
    AddQuiet(match, 50);
    result = match.Step();
    if (result != Result::kEnrolling) {
      break;
    }
  }
  EXPECT_EQ(result, Result::kEnrollFailed);
  EXPECT_FALSE(match.enrolled());
}

}  // namespace
}  // namespace device_ml::voice_match
