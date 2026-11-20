# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "soundfile", "torch<2.9"]
# ///
"""Writes an 8-bit student into the firmware as C++ data, with a test vector.

quantize.py simulates the 8-bit network in floating point. This goes one
step further: `IntegerModel` is the network in integer arithmetic exactly as
modules/voiceprint/voiceprint.cc computes it on the chip, operation for
operation. It is the reference the C++ is tested against.

Number formats (all chosen by quantize.py):
  input        int8, real value = q * input_scale (after mean subtraction)
  activations  uint8, real value = q * layer scale (ReLU makes them >= 0)
  weights      int8, one scale per output channel
  bias         int32 at (input scale * weight scale)
  rescale      out = (acc * multiplier + 2^(shift-1)) >> shift, clamped 0..255

Writes modules/voiceprint/voiceprint_model.{h,cc} and voiceprint_golden.h.

The firmware now runs the tcs network instead (tools/train/tcs_integer.py);
this matches modules/voiceprint/voiceprint.cc as of commit ad81096.

Usage:
  uv run tools/train/export_c.py v1
"""

import argparse
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quantize  # noqa: E402
import spk  # noqa: E402

_OUT = spk._ROOT / "modules" / "voiceprint"
_KINDS = {"full": "kFull3x3", "depthwise": "kDepthwise3x3", "pointwise": "kPointwise"}


def _multiplier(real: float) -> tuple[int, int]:
    """real ~= multiplier / 2^shift with multiplier in [2^30, 2^31)."""
    fraction, exponent = math.frexp(real)
    multiplier = round(fraction * (1 << 31))
    if multiplier == 1 << 31:
        multiplier //= 2
        exponent += 1
    return multiplier, 31 - exponent


class IntegerModel:
    """The 8-bit network in the chip's integer arithmetic."""

    def __init__(self, q: quantize.QuantizedStudent):
        self.input_scale = np.float32(q.input_scale)
        self.layers = []
        scale_in = q.input_scale
        for layer in q.layers:
            weight = layer["weight_q"].numpy().astype(np.int32)  # [out, in/groups, kh, kw]
            out_ch = weight.shape[0]
            if weight.shape[2] == 1:
                kind, stored = "pointwise", weight[:, :, 0, 0]  # [out][in]
            elif layer["groups"] == 1:
                kind, stored = "full", weight[:, 0].reshape(out_ch, 9)  # [out][tap]
            else:
                kind, stored = "depthwise", weight[:, 0].reshape(out_ch, 9).T  # [tap][channel]
            weight_scale = layer["weight_scale"].numpy().astype(np.float64)
            pairs = [_multiplier(scale_in * s / layer["output_scale"]) for s in weight_scale]
            self.layers.append(
                {
                    "kind": kind,
                    "stride": tuple(layer["stride"]),
                    "in_ch": weight.shape[1] * layer["groups"],
                    "out_ch": out_ch,
                    "weight": np.ascontiguousarray(stored),
                    "bias": torch.round(layer["bias"] / (scale_in * layer["weight_scale"])).numpy().astype(np.int64),
                    "multiplier": np.array([p[0] for p in pairs], dtype=np.int64),
                    "shift": np.array([p[1] for p in pairs], dtype=np.int64),
                }
            )
            scale_in = layer["output_scale"]
        self.last_scale = np.float32(scale_in)
        self.pooled_scale = np.float32(q.pooled_scale)
        self.fc_weight = q.fc_weight_q.numpy().astype(np.int32)  # [out][in]
        self.fc_scale = np.float32(q.pooled_scale * float(q.fc_weight_scale))
        self.fc_bias = q.fc_bias.numpy().astype(np.float32)

    def quantize_input(self, features: np.ndarray) -> np.ndarray:
        """[batch, frames, mels] float32 -> int8 values (held in int32)."""
        features = features.astype(np.float32)
        # Sequential float32 sum per band, the order the chip adds in.
        total = np.zeros((features.shape[0], features.shape[2]), dtype=np.float32)
        for frame in range(features.shape[1]):
            total += features[:, frame]
        centred = features - (total / np.float32(features.shape[1]))[:, None, :]
        scaled = centred * (np.float32(1.0) / self.input_scale) + np.float32(0.5)
        return np.clip(np.floor(scaled), -128, 127).astype(np.int32)

    def convolutions(self, x: np.ndarray) -> np.ndarray:
        """[batch, frames, mels] int -> [batch, time, mel, channels] uint8 values."""
        x = x[..., None].astype(np.int64)
        for layer in self.layers:
            if layer["kind"] == "pointwise":
                acc = x @ layer["weight"].T.astype(np.int64)
            else:
                st, sm = layer["stride"]
                out_t = (x.shape[1] - 1) // st + 1
                out_m = (x.shape[2] - 1) // sm + 1
                padded = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
                acc = np.zeros((x.shape[0], out_t, out_m, layer["out_ch"]), dtype=np.int64)
                for kt in range(3):
                    for km in range(3):
                        tap = padded[:, kt : kt + st * (out_t - 1) + 1 : st, km : km + sm * (out_m - 1) + 1 : sm]
                        if layer["kind"] == "full":
                            acc += tap * layer["weight"][:, kt * 3 + km]
                        else:
                            acc += tap * layer["weight"][kt * 3 + km]
            acc += layer["bias"]
            shift = layer["shift"]
            x = np.clip((acc * layer["multiplier"] + (1 << (shift - 1))) >> shift, 0, 255)
        return x

    def pool(self, x: np.ndarray) -> np.ndarray:
        """Mean and spread over time of each (channel, mel), as uint8 values."""
        steps = x.shape[1]
        total = x.sum(axis=1).astype(np.float32)  # [batch, mel, channel]
        squares = (x * x).sum(axis=1).astype(np.float32)
        mean = total / np.float32(steps)
        variance = np.maximum(squares - total * mean, np.float32(0.0)) / np.float32(steps - 1)
        spread = np.sqrt(variance)
        # Feature order matches the float model: channel-major, then mel.
        both = np.concatenate([mean.transpose(0, 2, 1).reshape(len(x), -1), spread.transpose(0, 2, 1).reshape(len(x), -1)], axis=1)
        scaled = both * (self.last_scale / self.pooled_scale) + np.float32(0.5)
        return np.clip(np.floor(scaled), 0, 255).astype(np.int32)

    def embed(self, pooled: np.ndarray) -> np.ndarray:
        out = (pooled @ self.fc_weight.T).astype(np.float32) * self.fc_scale + self.fc_bias
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return self.embed(self.pool(self.convolutions(self.quantize_input(features))))


def fnv1a(data: bytes) -> int:
    value = 0x811C9DC5
    for byte in data:
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def _array(ctype: str, name: str, values, per_line: int = 16, fmt: str = "{}") -> str:
    flat = np.asarray(values).ravel()
    rows = [", ".join(fmt.format(v) for v in flat[i : i + per_line]) for i in range(0, len(flat), per_line)]
    body = ",\n    ".join(rows)
    return f"const {ctype} {name}[{len(flat)}] = {{\n    {body},\n}};\n"


def _float(value) -> str:
    text = f"{float(value):.9g}"
    if "." not in text and "e" not in text:
        text += ".0"  # "-7f" isn't a valid C++ literal; "-7.0f" is.
    return text + "f"


def write_model(model: IntegerModel, name: str, sizes: list[int]) -> None:
    layers = model.layers
    # Layer outputs alternate between two buffers; the int8 input starts in B.
    buffer_a = max(sizes[1::2])
    buffer_b = max(sizes[0::2])
    header = f"""// Generated by tools/train/export_c.py from model "{name}". Do not edit.

#pragma once

#include <cstddef>
#include <cstdint>

namespace device_ml::voiceprint::model {{

inline constexpr size_t kFrames = {spk.CROP_FRAMES};
inline constexpr size_t kMels = {spk.ref.NUM_MELS};
inline constexpr size_t kEmbeddingDim = {len(model.fc_bias)};
inline constexpr size_t kPooledDim = {model.fc_weight.shape[1]};
inline constexpr size_t kNumLayers = {len(layers)};
inline constexpr size_t kMaxChannels = {max(layer["out_ch"] for layer in layers)};
// Layer outputs alternate between two buffers, A then B; the input uses B.
inline constexpr size_t kBufferABytes = {buffer_a};
inline constexpr size_t kBufferBBytes = {buffer_b};

enum class Kind : uint8_t {{ kFull3x3, kDepthwise3x3, kPointwise }};

struct Layer {{
  Kind kind;
  uint8_t stride_time;
  uint8_t stride_mel;
  uint16_t in_channels;
  uint16_t out_channels;
  // kFull3x3: [out][9]; kDepthwise3x3: [9][channel]; kPointwise: [out][in].
  const int8_t* weights;
  const int32_t* bias;
  // Per output channel: out = (acc * multiplier + 2^(shift-1)) >> shift.
  const int32_t* multiplier;
  const uint8_t* shift;
}};

extern const Layer kLayers[kNumLayers];

inline constexpr float kInputScale = {_float(model.input_scale)};
// Last layer's scale divided by the pooled vector's scale.
inline constexpr float kPoolRescale = {_float(model.last_scale / model.pooled_scale)};
extern const int8_t kFcWeights[kEmbeddingDim * kPooledDim];  // [out][in]
inline constexpr float kFcScale = {_float(model.fc_scale)};
extern const float kFcBias[kEmbeddingDim];

}}  // namespace device_ml::voiceprint::model
"""
    source = [
        f'// Generated by tools/train/export_c.py from model "{name}". Do not edit.\n',
        '#include "modules/voiceprint/voiceprint_model.h"\n',
        "namespace device_ml::voiceprint::model {\nnamespace {\n",
    ]
    for i, layer in enumerate(layers):
        source.append(_array("int8_t", f"kWeights{i}", layer["weight"], 20))
        source.append(_array("int32_t", f"kBias{i}", layer["bias"], 8))
        source.append(_array("int32_t", f"kMultiplier{i}", layer["multiplier"], 6))
        source.append(_array("uint8_t", f"kShift{i}", layer["shift"], 20))
    source.append("}  // namespace\n")
    rows = [
        f"    {{Kind::{_KINDS[layer['kind']]}, {layer['stride'][0]}, {layer['stride'][1]}, {layer['in_ch']}, "
        f"{layer['out_ch']}, kWeights{i}, kBias{i}, kMultiplier{i}, kShift{i}}},"
        for i, layer in enumerate(layers)
    ]
    source.append("const Layer kLayers[kNumLayers] = {\n" + "\n".join(rows) + "\n};\n")
    source.append(_array("int8_t", "kFcWeights", model.fc_weight, 20))
    source.append(_array("float", "kFcBias", [_float(v) for v in model.fc_bias], 6))
    source.append("}  // namespace device_ml::voiceprint::model\n")
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "voiceprint_model.h").write_text(header)
    (_OUT / "voiceprint_model.cc").write_text("\n".join(source))


def write_golden(model: IntegerModel, features: np.ndarray) -> None:
    """One real 1.5 s window with the outputs the chip must reproduce."""
    quantized = model.quantize_input(features[None])
    conv = model.convolutions(quantized)
    embedding = model.embed(model.pool(conv))[0]
    text = f"""// Generated by tools/train/export_c.py. Do not edit.
//
// One 1.5 s window of speech (LibriSpeech dev-clean) as log-mel features,
// with what the integer reference model computes from it.

#pragma once

#include <cstddef>
#include <cstdint>

namespace device_ml::voiceprint::golden {{

// FNV-1a hash of the int8 network input, and of the last convolution's
// output bytes in (time, mel, channel) order.
inline constexpr uint32_t kInputHash = 0x{fnv1a(quantized.astype(np.int8).tobytes()):08x}u;
inline constexpr uint32_t kConvOutputHash = 0x{fnv1a(conv.astype(np.uint8).tobytes()):08x}u;
inline constexpr size_t kConvOutputBytes = {conv.size};

inline {_array("float", "kLogMel", [_float(v) for v in features.ravel()], 6)}
inline {_array("float", "kEmbedding", [_float(v) for v in embedding], 6)}
}}  // namespace device_ml::voiceprint::golden
"""
    (_OUT / "voiceprint_golden.h").write_text(text)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="Model under data/models, e.g. v1")
    parser.add_argument("--calibration-crops", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(0)
    float_model = quantize.load_float(args.name)
    simulated = quantize.QuantizedStudent(float_model, quantize.calibration_crops(args.calibration_crops))
    model = IntegerModel(simulated)

    trials = spk.Trials(spk.Split("test-clean"), 30)
    crops = trials.crops()
    original = quantize.embeddings(float_model, crops)
    integer = np.concatenate([model(crops[i : i + 64]) for i in range(0, len(crops), 64)])
    print(f"{args.name}: test-clean error rate, float {100 * trials.eer(original):.2f}%  integer {100 * trials.eer(integer):.2f}%")
    print(f"  integer vs float voiceprint, average similarity {float(np.mean(np.sum(original * integer, axis=1))):.4f}")

    dev = spk.Split("dev-clean")
    rng = np.random.default_rng(7)
    utterance = int(rng.choice(np.flatnonzero(dev.lengths >= spk.CROP_FRAMES)))
    golden = dev.crop(utterance, int(rng.integers(0, dev.lengths[utterance] - spk.CROP_FRAMES + 1)))

    write_model(model, args.name, quantize.activation_bytes(float_model))
    write_golden(model, golden)
    print(f"Wrote {_OUT}/voiceprint_model.h, voiceprint_model.cc, voiceprint_golden.h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
