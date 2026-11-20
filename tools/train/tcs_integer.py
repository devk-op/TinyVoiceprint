# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "soundfile", "torch<2.9"]
# ///
"""The 8-bit tcs network in the chip's integer arithmetic.

quantize.QuantizedTCS simulates the 8-bit network in floating point. This
class does the same sums in integers, the way the firmware will, and is the
reference the C++ is tested against (as export_c.IntegerModel is for v3).

Number formats (scales chosen by quantize.py):
  input        int8 after mean subtraction, real = q * scale.input
  depthwise    int8 weights per channel, no bias; output int8
  pointwise    int8 weights per output channel, int32 bias at
               (input scale x weight scale); output uint8 after a ReLU,
               int8 without one (the last layer of each block)
  residual     uint8 = clamp((skip * a + y * b + 2^(s-1)) >> s, 0, 255),
               which is the add and the ReLU in one step
  rescale      out = (acc * multiplier + 2^(shift-1)) >> shift, clamped
  pooling      float32 mean and spread over time, then uint8
  embedding    int32 dot products, float32 scale and bias, unit length

Usage:
  uv run tools/train/tcs_integer.py tcs_long
compares this against the float simulation on test-clean crops; add
--write to generate modules/voiceprint/voiceprint_model.{h,cc} and
voiceprint_golden.h from it.
"""

import argparse
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import export_c  # noqa: E402
import quantize  # noqa: E402
import spk  # noqa: E402

_OUT = export_c._OUT


def _multiplier(real: float) -> tuple[int, int]:
    """real ~= multiplier / 2^shift with multiplier in [2^30, 2^31)."""
    fraction, exponent = math.frexp(real)
    multiplier = round(fraction * (1 << 31))
    if multiplier == 1 << 31:
        multiplier //= 2
        exponent += 1
    shift = 31 - exponent
    if not 1 <= shift <= 62:
        raise SystemExit(f"error: scale ratio {real} is out of range")
    return multiplier, shift


def _rescale(acc: np.ndarray, multiplier: np.ndarray, shift: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.clip((acc * multiplier + (np.int64(1) << (shift - 1))) >> shift, low, high)


class IntegerTCS:
    """Takes [batch, frames, mels] float features, like the float model."""

    def __init__(self, q: quantize.QuantizedTCS):
        s = q.scales
        self.input_scale = np.float32(s["input"])
        self.layers = []
        named = [("stem", q.stem)] + [(f"b{b}.l{i}", l) for b, block in enumerate(q.blocks) for i, l in enumerate(block)]
        scale_in = s["input"]
        for key, layer in named:
            if key.endswith(".l0"):
                scale_in = s["stem.pw"] if key == "b0.l0" else s[f"b{int(key[1:].split('.')[0]) - 1}.out"]
            dw_scale = layer["dw_scale"].numpy().astype(np.float64)
            pw_scale = layer["pw_scale"].numpy().astype(np.float64)
            dw_pairs = [_multiplier(scale_in * w / s[key + ".dw"]) for w in dw_scale]
            pw_pairs = [_multiplier(s[key + ".dw"] * w / s[key + ".pw"]) for w in pw_scale]
            self.layers.append(
                {
                    "key": key,
                    "stride": layer["stride"][0],
                    "relu": layer["relu"],
                    "dw": layer["dw_q"].numpy()[:, 0, :, 0].T.astype(np.int64),  # [tap][channel]
                    "dw_mul": np.array([p[0] for p in dw_pairs], dtype=np.int64),
                    "dw_shift": np.array([p[1] for p in dw_pairs], dtype=np.int64),
                    "pw": layer["pw_q"].numpy()[:, :, 0, 0].astype(np.int64),  # [out][in]
                    "bias": torch.round(layer["bias"] / (s[key + ".dw"] * layer["pw_scale"])).numpy().astype(np.int64),
                    "pw_mul": np.array([p[0] for p in pw_pairs], dtype=np.int64),
                    "pw_shift": np.array([p[1] for p in pw_pairs], dtype=np.int64),
                }
            )
            scale_in = s[key + ".pw"]
        # Residual add: both inputs rescaled to the block's output scale with
        # one shared shift, chosen so the larger multiplier fits in 31 bits.
        self.residuals = []
        for b, block in enumerate(q.blocks):
            skip_scale = s["stem.pw"] if b == 0 else s[f"b{b - 1}.out"]
            y_scale = s[f"b{b}.l{len(block) - 1}.pw"]
            ratios = (skip_scale / s[f"b{b}.out"], y_scale / s[f"b{b}.out"])
            _, shift = _multiplier(max(ratios))
            a, c = (round(r * (1 << shift)) for r in ratios)
            assert max(a, c) < 1 << 31 and 255 * a + 128 * c < 1 << 62
            self.residuals.append((len(block), np.int64(a), np.int64(c), np.int64(shift)))
        self.last_scale = np.float32(s[f"b{len(q.blocks) - 1}.out"])
        self.pooled_scale = np.float32(s["pooled"])
        self.fc_weight = q.fc_weight_q.numpy().astype(np.int64)  # [out][in]
        self.fc_scale = np.float32(s["pooled"] * float(q.fc_weight_scale))
        self.fc_bias = q.fc_bias.numpy().astype(np.float32)

    def quantize_input(self, features: np.ndarray) -> np.ndarray:
        """[batch, frames, mels] float32 -> int8 values (held in int64)."""
        features = features.astype(np.float32)
        # Sequential float32 sum per band, the order the chip adds in.
        total = np.zeros((features.shape[0], features.shape[2]), dtype=np.float32)
        for frame in range(features.shape[1]):
            total += features[:, frame]
        centred = features - (total / np.float32(features.shape[1]))[:, None, :]
        scaled = centred * (np.float32(1.0) / self.input_scale) + np.float32(0.5)
        return np.clip(np.floor(scaled), -128, 127).astype(np.int64)

    @staticmethod
    def layer(x: np.ndarray, layer: dict) -> np.ndarray:
        """One depthwise + pointwise layer; x is [batch, time, channels]."""
        taps, stride = layer["dw"].shape[0], layer["stride"]
        pad = taps // 2
        out_t = (x.shape[1] + 2 * pad - taps) // stride + 1
        padded = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
        acc = np.zeros((x.shape[0], out_t, x.shape[2]), dtype=np.int64)
        for k in range(taps):
            acc += padded[:, k : k + stride * (out_t - 1) + 1 : stride] * layer["dw"][k]
        x = _rescale(acc, layer["dw_mul"], layer["dw_shift"], -128, 127)
        acc = x @ layer["pw"].T + layer["bias"]
        low, high = (0, 255) if layer["relu"] else (-128, 127)
        return _rescale(acc, layer["pw_mul"], layer["pw_shift"], low, high)

    def convolutions(self, x: np.ndarray) -> np.ndarray:
        """int8 input [batch, frames, mels] -> uint8 [batch, time, channels]."""
        x = self.layer(x, self.layers[0])
        i = 1
        for count, a, c, shift in self.residuals:
            y = x
            for layer in self.layers[i : i + count]:
                y = self.layer(y, layer)
            i += count
            x = np.clip((x * a + y * c + (np.int64(1) << (shift - 1))) >> shift, 0, 255)
        return x

    def pool(self, x: np.ndarray) -> np.ndarray:
        """Mean and spread over time of each channel, as uint8 values."""
        steps = x.shape[1]
        total = x.sum(axis=1).astype(np.float32)
        squares = (x * x).sum(axis=1).astype(np.float32)
        mean = total / np.float32(steps)
        variance = np.maximum(squares - total * mean, np.float32(0.0)) / np.float32(steps - 1)
        both = np.concatenate([mean, np.sqrt(variance)], axis=1)
        scaled = both * (self.last_scale / self.pooled_scale) + np.float32(0.5)
        return np.clip(np.floor(scaled), 0, 255).astype(np.int64)

    def embed(self, pooled: np.ndarray) -> np.ndarray:
        out = (pooled @ self.fc_weight.T).astype(np.float32) * self.fc_scale + self.fc_bias
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def __call__(self, features):
        """Accepts a torch tensor too, so it can stand in for the model."""
        as_torch = isinstance(features, torch.Tensor)
        x = features.numpy() if as_torch else features
        out = self.embed(self.pool(self.convolutions(self.quantize_input(x))))
        return torch.from_numpy(out) if as_torch else out


def average_voice(model, count: int = 3000) -> np.ndarray:
    """Unit mean voiceprint of `count` 1.5 s crops from held-out Common Voice
    test speakers (cv-test). voice_match.cc rejects a window that is closer
    to this "average voice" than to the enrolled one, which stops loud
    non-speech sounds (they land near the average) from matching."""
    split = spk.Split("cv-test")
    rng = np.random.default_rng(0)
    chosen = rng.choice(np.flatnonzero(split.lengths >= spk.CROP_FRAMES), count, replace=False)
    crops = np.stack([split.crop(int(u), int(rng.integers(0, split.lengths[u] - spk.CROP_FRAMES + 1))) for u in chosen])
    with torch.no_grad():
        prints = np.concatenate([np.asarray(model(crops[i : i + 256])) for i in range(0, len(crops), 256)])
    mean = prints.mean(axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def write_model(model: IntegerTCS, name: str, average: np.ndarray) -> None:
    arr, flt = export_c._array, export_c._float
    stem, rest = model.layers[0], model.layers[1:]
    per_block = model.residuals[0][0]
    assert all(r[0] == per_block for r in model.residuals)
    channels = rest[0]["pw"].shape[0]
    # voiceprint.cc's Rescale() starts from the product's high word.
    assert all(min(layer["dw_shift"].min(), layer["pw_shift"].min()) >= 33 for layer in model.layers)
    # voiceprint.cc's Pointwise() does two output channels at a time.
    assert all(layer["pw"].shape[0] % 2 == 0 for layer in model.layers)
    taps = stem["dw"].shape[0]
    out_frames = (spk.CROP_FRAMES + 2 * (taps // 2) - taps) // stem["stride"] + 1
    header = f"""// Generated by tools/train/tcs_integer.py from model "{name}". Do not edit.

#pragma once

#include <cstddef>
#include <cstdint>

namespace device_ml::voiceprint::model {{

inline constexpr size_t kFrames = {spk.CROP_FRAMES};
inline constexpr size_t kMels = {spk.ref.NUM_MELS};
inline constexpr size_t kEmbeddingDim = {len(model.fc_bias)};
inline constexpr size_t kChannels = {channels};
inline constexpr size_t kPooledDim = {model.fc_weight.shape[1]};
inline constexpr size_t kTaps = {taps};  // Depthwise kernel length, in time.
inline constexpr size_t kStemStride = {stem["stride"]};
inline constexpr size_t kOutFrames = {out_frames};  // After the stem.
inline constexpr size_t kNumBlocks = {len(model.residuals)};
inline constexpr size_t kLayersPerBlock = {per_block};
inline constexpr size_t kNumLayers = 1 + kNumBlocks * kLayersPerBlock;

// One depthwise convolution along time, then a pointwise one. Each has a
// per-channel rescale: out = (acc * multiplier + 2^(shift-1)) >> shift.
struct Layer {{
  uint8_t stride;
  bool relu;  // Output uint8 (0..255) if true, else int8.
  uint16_t in_channels;
  uint16_t out_channels;
  const int8_t* dw;  // [tap][in]
  const int32_t* dw_multiplier;
  const uint8_t* dw_shift;
  const int8_t* pw;  // [out][in]
  const int32_t* bias;
  const int32_t* pw_multiplier;
  const uint8_t* pw_shift;
}};

// End of a block: out = clamp((skip * a + y * b + 2^(shift-1)) >> shift,
// 0, 255), the residual add and its ReLU in one step.
struct Residual {{
  int32_t skip_multiplier;
  int32_t y_multiplier;
  uint8_t shift;
}};

extern const Layer kLayers[kNumLayers];  // Stem, then each block's layers.
extern const Residual kResiduals[kNumBlocks];

inline constexpr float kInputScale = {flt(model.input_scale)};
// Last block's scale divided by the pooled vector's scale.
inline constexpr float kPoolRescale = {flt(model.last_scale / model.pooled_scale)};
extern const int8_t kFcWeights[kEmbeddingDim * kPooledDim];  // [out][in]
inline constexpr float kFcScale = {flt(model.fc_scale)};
extern const float kFcBias[kEmbeddingDim];
// Unit mean voiceprint of held-out speakers (cv-test); see average_voice().
extern const float kAverageVoice[kEmbeddingDim];

}}  // namespace device_ml::voiceprint::model
"""
    source = [
        f'// Generated by tools/train/tcs_integer.py from model "{name}". Do not edit.\n',
        '#include "modules/voiceprint/voiceprint_model.h"\n',
        "namespace device_ml::voiceprint::model {\nnamespace {\n",
    ]
    rows = []
    for i, layer in enumerate(model.layers):
        source.append(arr("int8_t", f"kDw{i}", layer["dw"], 20))
        source.append(arr("int32_t", f"kDwMultiplier{i}", layer["dw_mul"], 6))
        source.append(arr("uint8_t", f"kDwShift{i}", layer["dw_shift"], 20))
        source.append(arr("int8_t", f"kPw{i}", layer["pw"], 20))
        source.append(arr("int32_t", f"kBias{i}", layer["bias"], 8))
        source.append(arr("int32_t", f"kPwMultiplier{i}", layer["pw_mul"], 6))
        source.append(arr("uint8_t", f"kPwShift{i}", layer["pw_shift"], 20))
        rows.append(
            f"    {{{layer['stride']}, {'true' if layer['relu'] else 'false'}, {layer['dw'].shape[1]}, {layer['pw'].shape[0]}, "
            f"kDw{i}, kDwMultiplier{i}, kDwShift{i}, kPw{i}, kBias{i}, kPwMultiplier{i}, kPwShift{i}}},"
        )
    source.append("}  // namespace\n")
    source.append("const Layer kLayers[kNumLayers] = {\n" + "\n".join(rows) + "\n};\n")
    residuals = [f"    {{{a}, {c}, {s}}}," for _, a, c, s in model.residuals]
    source.append("const Residual kResiduals[kNumBlocks] = {\n" + "\n".join(residuals) + "\n};\n")
    source.append(arr("int8_t", "kFcWeights", model.fc_weight, 20))
    source.append(arr("float", "kFcBias", [flt(v) for v in model.fc_bias], 6))
    source.append(arr("float", "kAverageVoice", [flt(v) for v in average], 6))
    source.append("}  // namespace device_ml::voiceprint::model\n")
    (_OUT / "voiceprint_model.h").write_text(header)
    (_OUT / "voiceprint_model.cc").write_text("\n".join(source))


def write_golden(model: IntegerTCS) -> None:
    """The same dev-clean window export_c.py uses, with this model's outputs."""
    dev = spk.Split("dev-clean")
    rng = np.random.default_rng(7)
    utterance = int(rng.choice(np.flatnonzero(dev.lengths >= spk.CROP_FRAMES)))
    features = dev.crop(utterance, int(rng.integers(0, dev.lengths[utterance] - spk.CROP_FRAMES + 1)))
    export_c.write_golden(model, features)
    path = _OUT / "voiceprint_golden.h"
    text = path.read_text().replace("export_c.py", "tcs_integer.py").replace("(time, mel, channel)", "(time, channel)")
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="A tcs model under data/models, e.g. tcs_long")
    parser.add_argument("--crops", type=int, default=2000)
    parser.add_argument("--write", action="store_true", help="Generate the firmware's model files")
    args = parser.parse_args()

    torch.manual_seed(0)
    q = quantize.quantize(quantize.load_float(args.model), quantize.calibration_crops(1000), 99.99)
    integer = IntegerTCS(q)
    split = spk.Split("test-clean")
    rng = np.random.default_rng(1)
    eligible = np.flatnonzero(split.lengths >= spk.CROP_FRAMES)
    chosen = rng.choice(eligible, size=min(args.crops, len(eligible)), replace=False)
    crops = np.stack([split.crop(int(u), int(rng.integers(0, split.lengths[u] - spk.CROP_FRAMES + 1))) for u in chosen])
    with torch.no_grad():
        sim = quantize.embeddings(q, crops)
    exact = np.concatenate([integer(crops[i : i + 256]) for i in range(0, len(crops), 256)])
    cos = (sim * exact).sum(axis=1)
    print(f"{len(crops)} crops: cosine(integer, float simulation) min {cos.min():.5f}  mean {cos.mean():.6f}")
    if args.write:
        write_model(integer, args.model, average_voice(integer))
        write_golden(integer)
        print(f"Wrote {_OUT}/voiceprint_model.h, voiceprint_model.cc, voiceprint_golden.h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
