// Streaming log-mel spectrogram: the input features for the on-device model.
//
// Must stay numerically equivalent to tools/host/log_mel_reference.py, which
// is the definition the model is trained against. Per frame:
//   - 400 samples (25 ms at 16 kHz), advanced by 160 samples (10 ms)
//   - int16 scaled to [-1, 1), times a periodic Hann window
//   - zero-padded 512-point FFT, power spectrum of bins 0..256
//   - 40 triangular mel filters (HTK scale, unnormalised), 80 Hz to 7600 Hz
//   - natural log of (band energy + 1e-6)
//
// Pure computation with no hardware access, so it runs and is tested on the
// host as well as on the device.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace device_ml::features {

class LogMel {
 public:
  static constexpr size_t kSampleRate = 16000;
  static constexpr size_t kFrameLength = 400;
  static constexpr size_t kHop = 160;
  static constexpr size_t kFftSize = 512;
  static constexpr size_t kNumBins = kFftSize / 2 + 1;
  static constexpr size_t kNumMels = 40;

  using Frame = std::array<float, kNumMels>;

  LogMel();

  // Consumes `count` samples of 16 kHz PCM and calls `on_frame(const Frame&)`
  // once for every feature frame completed: the first after kFrameLength
  // samples, then one per kHop samples. The Frame is only valid during the
  // call.
  template <typename OnFrame>
  void Process(const int16_t* pcm, size_t count, OnFrame&& on_frame) {
    for (size_t i = 0; i < count; ++i) {
      if (Push(pcm[i])) {
        on_frame(frame_);
      }
    }
  }

  // Discards buffered samples, as if freshly constructed.
  void Reset() { fill_ = 0; }

 private:
  // Appends one sample; returns true if that completed a frame, which is
  // then in frame_.
  bool Push(int16_t sample);
  void ComputeFrame();

  // Each mel filter is a triangle over a contiguous run of FFT bins, and
  // neighbouring triangles overlap, so every bin feeds at most two filters:
  // the one rising through it and the one falling through it.
  struct BinWeights {
    uint8_t rising_filter;  // kNumMels if none
    uint8_t falling_filter;  // kNumMels if none
    float rising_weight;
    float falling_weight;
  };

  std::array<float, kFrameLength> window_;
  std::array<float, kFftSize / 2> twiddle_cos_;
  std::array<float, kFftSize / 2> twiddle_sin_;
  std::array<uint16_t, kFftSize> bit_reversed_;
  std::array<BinWeights, kNumBins> bin_weights_;

  std::array<int16_t, kFrameLength> samples_;
  size_t fill_ = 0;

  std::array<float, kFftSize> real_;
  std::array<float, kFftSize> imag_;
  Frame frame_;
};

}  // namespace device_ml::features
