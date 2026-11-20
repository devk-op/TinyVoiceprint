// Converts a 1-bit PDM bitstream into 16-bit PCM.
//
// The onboard MP45DT02 mic is clocked at 1.024 MHz and emits one bit per
// clock. This decimates by 64 to 16 kHz in two stages:
//   1. A 5th-order CIC (sinc^5) filter decimating by 16, one output per
//      16-bit word of PDM -> 64 kHz.
//   2. A 128-tap windowed-sinc FIR low-pass (7.2 kHz cutoff) decimating by 4
//      -> 16 kHz.
// followed by a DC-blocking high-pass and a fixed gain.
//
// Pure computation with no hardware access, so it runs and is tested on the
// host as well as on the device.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace device_ml::pdm {

class PdmDecimator {
 public:
  // 16-bit PDM words consumed per PCM sample produced.
  static constexpr size_t kWordsPerSample = 4;

  // `gain` scales the output, where 1.0 maps a full-scale PDM signal (all
  // ones) to full-scale PCM. The MP45DT02 reads -26 dBFS at 94 dB SPL, so
  // conversational speech is far below full scale without extra gain.
  explicit PdmDecimator(float gain = 16.0f);

  // Consumes `num_words` PDM words (MSB is the earliest bit) and writes
  // num_words / kWordsPerSample samples to `out`. `num_words` must be a
  // multiple of kWordsPerSample. Output saturates at the int16 range.
  void Process(const uint16_t* pdm, size_t num_words, int16_t* out);

  // Clears all filter state, as if freshly constructed.
  void Reset();

 private:
  static constexpr size_t kCicOrder = 5;
  static constexpr size_t kFirTaps = 128;

  // Runs the CIC over one 16-bit word; returns the 64 kHz sample scaled to
  // [-1, 1].
  float CicWord(uint16_t word);

  // CIC state. Integrators wrap modulo 2^32, which is exact for a CIC as
  // long as the final output (gain 16^5 = 2^20) fits, and it does.
  std::array<uint32_t, kCicOrder> integrators_{};
  std::array<uint32_t, kCicOrder> comb_delays_{};

  // FIR history, stored twice so the newest kFirTaps samples are always
  // contiguous at history_[pos_] without wrapping.
  std::array<float, kFirTaps> taps_{};
  std::array<float, 2 * kFirTaps> history_{};
  size_t pos_ = 0;
  size_t phase_ = 0;

  // DC blocker state.
  float dc_prev_in_ = 0.0f;
  float dc_prev_out_ = 0.0f;

  float gain_;
};

}  // namespace device_ml::pdm
