#include "modules/pdm/pdm_decimator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "pw_unit_test/framework.h"

namespace device_ml::pdm {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kPdmRateHz = 1024000.0;
constexpr double kPcmRateHz = 16000.0;

// Samples skipped before measuring, to let the FIR and DC blocker settle.
constexpr size_t kSettleSamples = 2000;

// Encodes `signal(t)` (in [-1, 1]) as PDM with a 2nd-order sigma-delta
// modulator, the same kind of encoding the mic does in hardware. Packs bits
// MSB-first into 16-bit words, matching what I2S DMA delivers.
template <typename Signal>
std::vector<uint16_t> EncodePdm(Signal signal, size_t num_pcm_samples) {
  const size_t num_bits = num_pcm_samples * PdmDecimator::kWordsPerSample * 16;
  std::vector<uint16_t> words(num_bits / 16, 0);
  double integrator1 = 0.0;
  double integrator2 = 0.0;
  double feedback = 0.0;
  for (size_t n = 0; n < num_bits; ++n) {
    integrator1 += signal(n / kPdmRateHz) - feedback;
    integrator2 += integrator1 - feedback;
    const bool bit = integrator2 >= 0.0;
    feedback = bit ? 1.0 : -1.0;
    if (bit) {
      words[n / 16] |= static_cast<uint16_t>(1u << (15 - n % 16));
    }
  }
  return words;
}

std::vector<int16_t> Decimate(PdmDecimator& decimator,
                              const std::vector<uint16_t>& pdm) {
  std::vector<int16_t> pcm(pdm.size() / PdmDecimator::kWordsPerSample);
  decimator.Process(pdm.data(), pdm.size(), pcm.data());
  return pcm;
}

// Least-squares amplitude of a sinusoid at `freq_hz` in pcm[kSettleSamples:].
double ToneAmplitude(const std::vector<int16_t>& pcm, double freq_hz) {
  double in_phase = 0.0;
  double quadrature = 0.0;
  const size_t count = pcm.size() - kSettleSamples;
  for (size_t n = kSettleSamples; n < pcm.size(); ++n) {
    const double angle = 2.0 * kPi * freq_hz * n / kPcmRateHz;
    in_phase += pcm[n] * std::cos(angle);
    quadrature += pcm[n] * std::sin(angle);
  }
  return 2.0 * std::hypot(in_phase, quadrature) / count;
}

double Rms(const std::vector<int16_t>& pcm) {
  double sum_sq = 0.0;
  for (size_t n = kSettleSamples; n < pcm.size(); ++n) {
    sum_sq += static_cast<double>(pcm[n]) * pcm[n];
  }
  return std::sqrt(sum_sq / (pcm.size() - kSettleSamples));
}

TEST(PdmDecimator, PassbandToneKeepsItsLevel) {
  PdmDecimator decimator(/*gain=*/1.0f);
  const auto pdm = EncodePdm(
      [](double t) { return 0.5 * std::sin(2.0 * kPi * 1000.0 * t); }, 8000);
  const auto pcm = Decimate(decimator, pdm);

  const double expected = 0.5 * 32767.0;
  EXPECT_NEAR(ToneAmplitude(pcm, 1000.0), expected, 0.03 * expected);

  // Almost all energy should be the tone itself: a pure sine of amplitude A
  // has RMS A/sqrt(2). Anything beyond that is noise or distortion.
  const double tone_rms = ToneAmplitude(pcm, 1000.0) / std::sqrt(2.0);
  const double residual_rms =
      std::sqrt(std::max(0.0, Rms(pcm) * Rms(pcm) - tone_rms * tone_rms));
  EXPECT_LT(residual_rms, 0.01 * tone_rms);
}

TEST(PdmDecimator, ToneAboveNyquistIsRejected) {
  // 12 kHz would alias down to 4 kHz if the low-pass let it through.
  PdmDecimator decimator(/*gain=*/1.0f);
  const auto pdm = EncodePdm(
      [](double t) { return 0.5 * std::sin(2.0 * kPi * 12000.0 * t); }, 8000);
  const auto pcm = Decimate(decimator, pdm);

  EXPECT_LT(Rms(pcm), 0.001 * 0.5 * 32767.0);  // At least 60 dB down.
}

TEST(PdmDecimator, DcOffsetIsRemoved) {
  PdmDecimator decimator(/*gain=*/1.0f);
  const auto pdm = EncodePdm([](double) { return 0.25; }, 8000);
  const auto pcm = Decimate(decimator, pdm);

  EXPECT_LT(Rms(pcm), 30.0);
}

TEST(PdmDecimator, LoudInputSaturatesInsteadOfWrapping) {
  PdmDecimator decimator(/*gain=*/4.0f);
  const auto pdm = EncodePdm(
      [](double t) { return 0.5 * std::sin(2.0 * kPi * 1000.0 * t); }, 8000);
  const auto pcm = Decimate(decimator, pdm);

  const auto [min_it, max_it] =
      std::minmax_element(pcm.begin() + kSettleSamples, pcm.end());
  EXPECT_EQ(*max_it, 32767);
  EXPECT_EQ(*min_it, -32768);
  // A wrapped sample would be a sign flip at the peak; the clipped tone
  // still has the right fundamental phase and roughly 2x+ the level.
  EXPECT_GT(ToneAmplitude(pcm, 1000.0), 1.0 * 32767.0);
}

TEST(PdmDecimator, ResetClearsState) {
  PdmDecimator decimator(/*gain=*/1.0f);
  const auto pdm = EncodePdm(
      [](double t) { return 0.5 * std::sin(2.0 * kPi * 1000.0 * t); }, 4000);
  const auto first = Decimate(decimator, pdm);
  decimator.Reset();
  const auto second = Decimate(decimator, pdm);
  EXPECT_EQ(first, second);
}

}  // namespace
}  // namespace device_ml::pdm
