#include "modules/features/log_mel.h"

#include <algorithm>
#include <cmath>

namespace device_ml::features {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMelLowHz = 80.0;
constexpr double kMelHighHz = 7600.0;
constexpr float kLogFloor = 1e-6f;
constexpr float kInt16Scale = 1.0f / 32768.0f;

constexpr size_t kOverlap = LogMel::kFrameLength - LogMel::kHop;

// HTK mel scale.
double HzToMel(double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }
double MelToHz(double mel) {
  return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

}  // namespace

// The tables are built once at startup rather than stored as constants: it
// keeps the definition readable next to the Python reference, and the
// one-off double-precision maths is negligible.
LogMel::LogMel() {
  for (size_t n = 0; n < kFrameLength; ++n) {
    window_[n] = static_cast<float>(
        0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(n) / kFrameLength));
  }

  for (size_t k = 0; k < kFftSize / 2; ++k) {
    const double angle = 2.0 * kPi * static_cast<double>(k) / kFftSize;
    twiddle_cos_[k] = static_cast<float>(std::cos(angle));
    twiddle_sin_[k] = static_cast<float>(std::sin(angle));
  }

  size_t bits = 0;
  while ((size_t{1} << bits) < kFftSize) {
    ++bits;
  }
  for (size_t i = 0; i < kFftSize; ++i) {
    size_t reversed = 0;
    for (size_t b = 0; b < bits; ++b) {
      reversed |= ((i >> b) & 1u) << (bits - 1 - b);
    }
    bit_reversed_[i] = static_cast<uint16_t>(reversed);
  }

  // kNumMels triangles share kNumMels + 2 edges, evenly spaced in mel.
  std::array<double, kNumMels + 2> edges_hz;
  const double mel_low = HzToMel(kMelLowHz);
  const double mel_step = (HzToMel(kMelHighHz) - mel_low) / (kNumMels + 1);
  for (size_t i = 0; i < edges_hz.size(); ++i) {
    edges_hz[i] = MelToHz(mel_low + mel_step * static_cast<double>(i));
  }
  for (size_t bin = 0; bin < kNumBins; ++bin) {
    BinWeights& weights = bin_weights_[bin];
    weights = {kNumMels, kNumMels, 0.0f, 0.0f};
    const double hz = static_cast<double>(bin) * kSampleRate / kFftSize;
    if (hz < edges_hz.front() || hz >= edges_hz.back()) {
      continue;
    }
    // Between edges k and k+1 the bin is on the rising side of filter k and
    // the falling side of filter k-1.
    const size_t k = static_cast<size_t>(
        std::upper_bound(edges_hz.begin(), edges_hz.end(), hz) -
        edges_hz.begin() - 1);
    const double span = edges_hz[k + 1] - edges_hz[k];
    if (k < kNumMels) {
      weights.rising_filter = static_cast<uint8_t>(k);
      weights.rising_weight = static_cast<float>((hz - edges_hz[k]) / span);
    }
    if (k >= 1) {
      weights.falling_filter = static_cast<uint8_t>(k - 1);
      weights.falling_weight =
          static_cast<float>((edges_hz[k + 1] - hz) / span);
    }
  }
}

bool LogMel::Push(int16_t sample) {
  samples_[fill_++] = sample;
  if (fill_ < kFrameLength) {
    return false;
  }
  ComputeFrame();
  std::copy(samples_.begin() + kHop, samples_.end(), samples_.begin());
  fill_ = kOverlap;
  return true;
}

void LogMel::ComputeFrame() {
  // Window, zero-pad to the FFT size, and load in bit-reversed order ready
  // for the in-place transform.
  for (size_t i = 0; i < kFftSize; ++i) {
    const size_t target = bit_reversed_[i];
    real_[target] = i < kFrameLength
                        ? static_cast<float>(samples_[i]) * kInt16Scale *
                              window_[i]
                        : 0.0f;
    imag_[target] = 0.0f;
  }

  // Iterative radix-2 FFT.
  for (size_t half = 1; half < kFftSize; half *= 2) {
    const size_t twiddle_step = kFftSize / (2 * half);
    for (size_t start = 0; start < kFftSize; start += 2 * half) {
      for (size_t j = 0; j < half; ++j) {
        const float cos = twiddle_cos_[j * twiddle_step];
        const float sin = twiddle_sin_[j * twiddle_step];
        const size_t a = start + j;
        const size_t b = a + half;
        const float product_real = real_[b] * cos + imag_[b] * sin;
        const float product_imag = imag_[b] * cos - real_[b] * sin;
        real_[b] = real_[a] - product_real;
        imag_[b] = imag_[a] - product_imag;
        real_[a] += product_real;
        imag_[a] += product_imag;
      }
    }
  }

  Frame energy{};
  for (size_t bin = 0; bin < kNumBins; ++bin) {
    const BinWeights& weights = bin_weights_[bin];
    const float power = real_[bin] * real_[bin] + imag_[bin] * imag_[bin];
    if (weights.rising_filter < kNumMels) {
      energy[weights.rising_filter] += power * weights.rising_weight;
    }
    if (weights.falling_filter < kNumMels) {
      energy[weights.falling_filter] += power * weights.falling_weight;
    }
  }
  for (size_t m = 0; m < kNumMels; ++m) {
    frame_[m] = std::log(energy[m] + kLogFloor);
  }
}

}  // namespace device_ml::features
