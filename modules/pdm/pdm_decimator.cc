#include "modules/pdm/pdm_decimator.h"

#include <cmath>

namespace device_ml::pdm {
namespace {

constexpr float kPi = 3.14159265358979f;

// Rates through the chain: 1.024 MHz PDM -> 64 kHz after the CIC -> 16 kHz.
constexpr float kCicOutputRateHz = 64000.0f;
constexpr float kFirCutoffHz = 7200.0f;

// CIC gain is (decimation)^(order) = 16^5 = 2^20.
constexpr float kCicScale = 1.0f / static_cast<float>(1u << 20);

// Pole of the DC blocker; ~13 Hz corner at 16 kHz. Removes the mic's DC
// offset without touching speech.
constexpr float kDcPole = 0.995f;

}  // namespace

PdmDecimator::PdmDecimator(float gain) : gain_(gain) {
  // Blackman-windowed sinc low-pass, normalized to unity DC gain. Computed at
  // startup instead of baked in so the filter stays readable and tweakable.
  const float fc = kFirCutoffHz / kCicOutputRateHz;
  const float center = (kFirTaps - 1) / 2.0f;
  float sum = 0.0f;
  for (size_t n = 0; n < kFirTaps; ++n) {
    const float t = static_cast<float>(n) - center;
    const float sinc =
        t == 0.0f ? 2.0f * fc : std::sin(2.0f * kPi * fc * t) / (kPi * t);
    const float phase = 2.0f * kPi * static_cast<float>(n) / (kFirTaps - 1);
    const float window =
        0.42f - 0.5f * std::cos(phase) + 0.08f * std::cos(2.0f * phase);
    taps_[n] = sinc * window;
    sum += taps_[n];
  }
  for (float& tap : taps_) {
    tap /= sum;
  }
}

void PdmDecimator::Reset() {
  integrators_.fill(0);
  comb_delays_.fill(0);
  history_.fill(0.0f);
  pos_ = 0;
  phase_ = 0;
  dc_prev_in_ = 0.0f;
  dc_prev_out_ = 0.0f;
}

float PdmDecimator::CicWord(uint16_t word) {
  uint32_t i0 = integrators_[0];
  uint32_t i1 = integrators_[1];
  uint32_t i2 = integrators_[2];
  uint32_t i3 = integrators_[3];
  uint32_t i4 = integrators_[4];
  for (int bit = 15; bit >= 0; --bit) {
    // PDM 1 -> +1, 0 -> -1 (0xFFFFFFFF in two's complement).
    i0 += ((word >> bit) & 1u) ? 1u : 0xFFFFFFFFu;
    i1 += i0;
    i2 += i1;
    i3 += i2;
    i4 += i3;
  }
  integrators_ = {i0, i1, i2, i3, i4};

  uint32_t value = i4;
  for (uint32_t& delay : comb_delays_) {
    const uint32_t diff = value - delay;
    delay = value;
    value = diff;
  }
  return static_cast<float>(static_cast<int32_t>(value)) * kCicScale;
}

void PdmDecimator::Process(const uint16_t* pdm,
                           size_t num_words,
                           int16_t* out) {
  for (size_t i = 0; i < num_words; ++i) {
    const float sample = CicWord(pdm[i]);
    pos_ = (pos_ == 0) ? kFirTaps - 1 : pos_ - 1;
    history_[pos_] = sample;
    history_[pos_ + kFirTaps] = sample;

    if (++phase_ < kWordsPerSample) {
      continue;
    }
    phase_ = 0;

    float filtered = 0.0f;
    const float* recent = &history_[pos_];
    for (size_t k = 0; k < kFirTaps; ++k) {
      filtered += taps_[k] * recent[k];
    }

    const float dc_blocked = filtered - dc_prev_in_ + kDcPole * dc_prev_out_;
    dc_prev_in_ = filtered;
    dc_prev_out_ = dc_blocked;

    float scaled = dc_blocked * gain_ * 32767.0f;
    if (scaled > 32767.0f) {
      scaled = 32767.0f;
    } else if (scaled < -32768.0f) {
      scaled = -32768.0f;
    }
    *out++ = static_cast<int16_t>(std::lrintf(scaled));
  }
}

}  // namespace device_ml::pdm
