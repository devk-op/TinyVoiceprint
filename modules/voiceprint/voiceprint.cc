#include "modules/voiceprint/voiceprint.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__ARM_FEATURE_DSP)
#include <arm_acle.h>
#endif

namespace device_ml::voiceprint {
namespace {

using model::Layer;

// Scales a 32-bit accumulator to the 8-bit range [low, high]:
// (acc * multiplier + 2^(shift-1)) >> shift. The generator guarantees
// shift >= 33, so the rounding term has no low word and the shift can start
// from the product's high word; arithmetic shifts compose, so this equals
// the 64-bit expression exactly.
inline int32_t Rescale(int32_t acc,
                       int32_t multiplier,
                       uint8_t shift,
                       int32_t low,
                       int32_t high) {
  int32_t hi = static_cast<int32_t>(
      (static_cast<int64_t>(acc) * multiplier) >> 32);
  hi += int32_t{1} << (shift - 33);
  return std::clamp(hi >> (shift - 32), low, high);
}

// Convolution along time of each channel separately, zero padded at both
// ends; int8 out. `In` is int8 for the network input, uint8 elsewhere.
// One channel at a time, so its taps, multiplier and shift stay in
// registers.
template <typename In>
size_t Depthwise(const Layer& layer,
                 const In* in,
                 size_t in_time,
                 int8_t* out) {
  constexpr int kTaps = static_cast<int>(model::kTaps);
  constexpr int kPad = kTaps / 2;
  const int channels = layer.in_channels;
  const int stride = layer.stride;
  const int time = static_cast<int>(in_time);
  const int out_time = (time + 2 * kPad - kTaps) / stride + 1;
  for (int c = 0; c < channels; ++c) {
    int32_t w[kTaps];
    for (int k = 0; k < kTaps; ++k) {
      w[k] = layer.dw[k * channels + c];
    }
    const int32_t multiplier = layer.dw_multiplier[c];
    const uint8_t shift = layer.dw_shift[c];
    const In* x = in + c;
    int8_t* o = out + c;
    for (int ot = 0; ot < out_time; ++ot, o += channels) {
      const int t0 = ot * stride - kPad;
      int32_t acc = 0;
      if (t0 >= 0 && t0 + kTaps <= time) {
        const In* row = x + t0 * channels;
        for (int k = 0; k < kTaps; ++k) {
          acc += static_cast<int32_t>(row[k * channels]) * w[k];
        }
      } else {
        for (int k = 0; k < kTaps; ++k) {
          const int t = t0 + k;
          if (t >= 0 && t < time) {
            acc += static_cast<int32_t>(x[t * channels]) * w[k];
          }
        }
      }
      *o = static_cast<int8_t>(Rescale(acc, multiplier, shift, -128, 127));
    }
  }
  return static_cast<size_t>(out_time);
}

#if defined(__ARM_FEATURE_DSP)
// Four int8 values as two pairs of int16: (b0, b2) and (b1, b3).
inline void Unpack(const int8_t* p, uint32_t& even, uint32_t& odd) {
  uint32_t word;
  std::memcpy(&word, p, 4);
  even = __sxtb16(word);
  odd = __sxtb16(__ror(word, 8));
}
#endif

// acc[i][j] += dot(in row i, weight row j) over `count` channels, for two
// time steps and two output channels. Integer sums, so the order of
// additions does not change the result.
inline void Dot2x2(const int8_t* in0,
                   const int8_t* in1,
                   const int8_t* w0,
                   const int8_t* w1,
                   size_t count,
                   int32_t acc[2][2]) {
  size_t i = 0;
#if defined(__ARM_FEATURE_DSP)
  // SMLAD does two 16-bit multiply-adds in one cycle.
  int32_t a00 = acc[0][0], a01 = acc[0][1], a10 = acc[1][0], a11 = acc[1][1];
  for (; i + 4 <= count; i += 4) {
    uint32_t x0e, x0o, x1e, x1o, w0e, w0o, w1e, w1o;
    Unpack(in0 + i, x0e, x0o);
    Unpack(in1 + i, x1e, x1o);
    Unpack(w0 + i, w0e, w0o);
    Unpack(w1 + i, w1e, w1o);
    a00 = __smlad(x0e, w0e, __smlad(x0o, w0o, a00));
    a01 = __smlad(x0e, w1e, __smlad(x0o, w1o, a01));
    a10 = __smlad(x1e, w0e, __smlad(x1o, w0o, a10));
    a11 = __smlad(x1e, w1e, __smlad(x1o, w1o, a11));
  }
  acc[0][0] = a00, acc[0][1] = a01, acc[1][0] = a10, acc[1][1] = a11;
#endif
  for (; i < count; ++i) {
    acc[0][0] += in0[i] * w0[i];
    acc[0][1] += in0[i] * w1[i];
    acc[1][0] += in1[i] * w0[i];
    acc[1][1] += in1[i] * w1[i];
  }
}

// Mixes channels at each time step. The output is uint8 after a ReLU and
// int8 without one; both are stored as bytes. Works on two time steps and
// two output channels at a time, so each load is used twice.
void Pointwise(const Layer& layer,
               const int8_t* in,
               size_t time,
               uint8_t* out) {
  const size_t cin = layer.in_channels;
  const size_t cout = layer.out_channels;
  const int32_t low = layer.relu ? 0 : -128;
  const int32_t high = layer.relu ? 255 : 127;
  // Each weight is read once per pair of time steps; from flash that costs
  // wait states, so copy the layer's weights to RAM first.
  static int8_t weights[model::kChannels * model::kChannels];
  std::memcpy(weights, layer.pw, cin * cout);
  for (size_t t = 0; t < time; t += 2) {
    // An odd last step is computed twice and the copy discarded.
    const bool pair = t + 1 < time;
    const int8_t* in0 = in + t * cin;
    const int8_t* in1 = pair ? in0 + cin : in0;
    uint8_t* out0 = out + t * cout;
    uint8_t* out1 = out0 + cout;
    for (size_t oc = 0; oc < cout; oc += 2) {
      const int8_t* w0 = weights + oc * cin;
      const int8_t* w1 = w0 + cin;
      int32_t acc[2][2] = {{layer.bias[oc], layer.bias[oc + 1]},
                           {layer.bias[oc], layer.bias[oc + 1]}};
      Dot2x2(in0, in1, w0, w1, cin, acc);
      for (size_t j = 0; j < 2; ++j) {
        const int32_t m = layer.pw_multiplier[oc + j];
        const uint8_t s = layer.pw_shift[oc + j];
        out0[oc + j] = static_cast<uint8_t>(Rescale(acc[0][j], m, s, low, high));
        if (pair) {
          out1[oc + j] =
              static_cast<uint8_t>(Rescale(acc[1][j], m, s, low, high));
        }
      }
    }
  }
}

// Residual add and ReLU, in place: x = relu(x + y) at the block's scale.
void AddResidual(const model::Residual& r,
                 uint8_t* x,
                 const int8_t* y,
                 size_t count) {
  const int64_t round = int64_t{1} << (r.shift - 1);
  for (size_t i = 0; i < count; ++i) {
    const int64_t sum = static_cast<int64_t>(x[i]) * r.skip_multiplier +
                        static_cast<int64_t>(y[i]) * r.y_multiplier + round;
    x[i] = static_cast<uint8_t>(std::clamp<int64_t>(sum >> r.shift, 0, 255));
  }
}

}  // namespace

void Voiceprint::QuantizeInput(const float* log_mel) {
  QuantizeInput(log_mel, kFrames, nullptr);
}

void Voiceprint::QuantizeInput(const float* first,
                               size_t first_frames,
                               const float* second) {
  const auto row = [=](size_t t) {
    return t < first_frames ? first + t * kMels
                            : second + (t - first_frames) * kMels;
  };
  // Subtract each mel band's mean over the window, which removes a fixed
  // microphone response, then round to int8.
  float mean[kMels] = {};
  for (size_t t = 0; t < kFrames; ++t) {
    const float* frame = row(t);
    for (size_t m = 0; m < kMels; ++m) {
      mean[m] += frame[m];
    }
  }
  for (float& value : mean) {
    value /= static_cast<float>(kFrames);
  }
  const float inverse_scale = 1.0f / model::kInputScale;
  int8_t* out = reinterpret_cast<int8_t*>(input_);
  for (size_t t = 0; t < kFrames; ++t) {
    const float* frame = row(t);
    for (size_t m = 0; m < kMels; ++m) {
      const float scaled = (frame[m] - mean[m]) * inverse_scale + 0.5f;
      *out++ = static_cast<int8_t>(
          std::clamp(std::floor(scaled), -128.0f, 127.0f));
    }
  }
}

void Voiceprint::RunConvolutions() {
  // x holds each block's input (and finally the output); the depthwise
  // result goes to d, and pointwise results alternate between y and z.
  uint8_t* x = buffer_x_;
  int8_t* d = reinterpret_cast<int8_t*>(buffer_d_);
  uint8_t* y = buffer_y_;
  uint8_t* z = input_;  // Free once the stem has read the input.
  size_t time = Depthwise(model::kLayers[0],
                          reinterpret_cast<const int8_t*>(input_), kFrames, d);
  Pointwise(model::kLayers[0], d, time, x);
  for (size_t b = 0; b < model::kNumBlocks; ++b) {
    const uint8_t* in = x;
    uint8_t* out = y;
    for (size_t l = 0; l < model::kLayersPerBlock; ++l) {
      const Layer& layer = model::kLayers[1 + b * model::kLayersPerBlock + l];
      Depthwise(layer, in, time, d);
      Pointwise(layer, d, time, out);
      in = out;
      out = (out == y) ? z : y;
    }
    AddResidual(model::kResiduals[b], x, reinterpret_cast<const int8_t*>(in),
                time * model::kChannels);
  }
  conv_output_ = x;
  out_time_ = time;
}

void Voiceprint::PoolAndEmbed(float* embedding) {
  // Summarise each channel over time by its mean and spread, as 8-bit
  // values: all the means first, then all the spreads.
  uint8_t* pooled = buffer_y_;
  const size_t channels = model::kChannels;
  const float steps = static_cast<float>(out_time_);
  for (size_t c = 0; c < channels; ++c) {
    uint32_t sum = 0;
    uint32_t squares = 0;
    const uint8_t* x = conv_output_ + c;
    for (size_t t = 0; t < out_time_; ++t, x += channels) {
      sum += *x;
      squares += static_cast<uint32_t>(*x) * *x;
    }
    const float total = static_cast<float>(sum);
    const float mean = total / steps;
    const float variance =
        std::max(static_cast<float>(squares) - total * mean, 0.0f) /
        (steps - 1.0f);
    const float values[2] = {mean, std::sqrt(variance)};
    for (size_t k = 0; k < 2; ++k) {
      const float scaled = values[k] * model::kPoolRescale + 0.5f;
      pooled[k * channels + c] =
          static_cast<uint8_t>(std::clamp(std::floor(scaled), 0.0f, 255.0f));
    }
  }

  float norm = 0.0f;
  const int8_t* w = model::kFcWeights;
  for (size_t o = 0; o < kEmbeddingDim; ++o, w += model::kPooledDim) {
    int32_t acc = 0;
    for (size_t i = 0; i < model::kPooledDim; ++i) {
      acc += static_cast<int32_t>(pooled[i]) * w[i];
    }
    embedding[o] = static_cast<float>(acc) * model::kFcScale + model::kFcBias[o];
    norm += embedding[o] * embedding[o];
  }
  const float inverse = 1.0f / std::sqrt(norm + 1e-12f);
  for (size_t o = 0; o < kEmbeddingDim; ++o) {
    embedding[o] *= inverse;
  }
}

void Voiceprint::Run(const float* log_mel, float* embedding) {
  QuantizeInput(log_mel);
  RunConvolutions();
  PoolAndEmbed(embedding);
}

float Similarity(const float* a, const float* b) {
  float dot = 0.0f;
  for (size_t i = 0; i < Voiceprint::kEmbeddingDim; ++i) {
    dot += a[i] * b[i];
  }
  return dot;
}

}  // namespace device_ml::voiceprint
