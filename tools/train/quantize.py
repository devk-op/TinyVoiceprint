# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "soundfile", "torch<2.9"]
# ///
"""Converts a trained student to 8-bit and measures what that costs.

The chip runs the network with 8-bit integers (as Arm's CMSIS-NN does):
weights are int8 with one scale per output channel, and each layer's output
is int8 with one scale for the layer. This tool

  1. folds each batch-norm into the convolution before it,
  2. picks the scales from a batch of calibration speech,
  3. simulates the 8-bit network on the Mac and reports its error rate next
     to the original's, and
  4. saves the integers and scales to data/models/NAME.int8.npz.

What stays in floating point on the chip, because it is tiny: subtracting
the per-band mean from the input window, the mean/spread pooling, and the
final length normalisation.

The simulation rounds every stored value exactly as the chip will, but
computes each layer in floating point from those rounded values, so the
chip's results can differ in the last bit. That gets checked once the C
code exists.

Usage:
  uv run tools/train/quantize.py v1
  uv run tools/train/quantize.py v2_w32 --calibration-crops 2000
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spk  # noqa: E402

_MODELS = spk._MODELS


def fold_batch_norms(model: spk.Student) -> list[dict]:
    """The network as a list of plain convolutions (weight, bias, stride,
    groups), each followed by ReLU, with batch-norm folded in."""
    modules = [m for m in model.features.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
    layers = []
    for conv, norm in zip(modules[0::2], modules[1::2]):
        gain = norm.weight / torch.sqrt(norm.running_var + norm.eps)
        layers.append(
            {
                "weight": (conv.weight * gain[:, None, None, None]).detach(),
                "bias": (norm.bias - norm.running_mean * gain).detach(),
                "stride": conv.stride,
                "padding": conv.padding,
                "groups": conv.groups,
            }
        )
    return layers


def _round_to(values: torch.Tensor, scale, low: int, high: int) -> torch.Tensor:
    """Integers the chip would store for `values` at this scale."""
    return torch.clamp(torch.round(values / scale), low, high)


class QuantizedStudent(nn.Module):
    """8-bit simulation with the same call signature as spk.Student."""

    def __init__(self, model: spk.Student, calibration: torch.Tensor, percentile: float = 99.99):
        super().__init__()
        self.layers = fold_batch_norms(model)
        self.fc_weight = model.embed.weight.detach()
        self.fc_bias = model.embed.bias.detach()

        # Per-output-channel int8 weights for convolutions; one scale for the
        # fully connected layer (CMSIS-NN's fully connected is per-layer).
        for layer in self.layers:
            peak = layer["weight"].abs().amax(dim=(1, 2, 3)).clamp_min(1e-8)
            layer["weight_scale"] = peak / 127.0
            layer["weight_q"] = _round_to(layer["weight"], layer["weight_scale"][:, None, None, None], -127, 127)
        self.fc_weight_scale = self.fc_weight.abs().max() / 127.0
        self.fc_weight_q = _round_to(self.fc_weight, self.fc_weight_scale, -127, 127)

        # Activation scales: run calibration speech through the folded
        # float network and take a high percentile of each layer's output,
        # so a few extreme values don't waste the 8-bit range.
        def high(x: torch.Tensor) -> float:
            sample = x.flatten()
            sample = sample[torch.randperm(len(sample))[:2_000_000]]
            return float(torch.quantile(sample.abs(), percentile / 100.0))

        x = self._centre(calibration)
        self.input_scale = high(x) / 127.0  # Signed: the input is centred on zero.
        for layer in self.layers:
            x = torch.relu(self._conv(x, layer, layer["weight"], layer["bias"]))
            layer["output_scale"] = high(x) / 255.0  # Unsigned range after ReLU.
        self.pooled_scale = high(self._pool(x)) / 255.0

    @staticmethod
    def _centre(features: torch.Tensor) -> torch.Tensor:
        return (features - features.mean(dim=1, keepdim=True)).unsqueeze(1)

    @staticmethod
    def _conv(x, layer, weight, bias):
        return nn.functional.conv2d(x, weight, bias, stride=layer["stride"], padding=layer["padding"], groups=layer["groups"])

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).flatten(2)
        return torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self._centre(features)
        x = _round_to(x, self.input_scale, -128, 127) * self.input_scale
        scale_in = self.input_scale
        for layer in self.layers:
            weight = layer["weight_q"] * layer["weight_scale"][:, None, None, None]
            # Biases are int32 at (input scale x weight scale), as in CMSIS-NN.
            bias_scale = scale_in * layer["weight_scale"]
            bias = torch.round(layer["bias"] / bias_scale) * bias_scale
            x = torch.relu(self._conv(x, layer, weight, bias))
            x = _round_to(x, layer["output_scale"], 0, 255) * layer["output_scale"]
            scale_in = layer["output_scale"]
        pooled = _round_to(self._pool(x), self.pooled_scale, 0, 255) * self.pooled_scale
        out = pooled @ (self.fc_weight_q * self.fc_weight_scale).T + self.fc_bias
        return nn.functional.normalize(out, dim=1)

    def export(self, path: pathlib.Path) -> int:
        """Saves integers and scales; returns the weight bytes on the chip."""
        arrays = {
            "input_scale": np.float32(self.input_scale),
            "pooled_scale": np.float32(self.pooled_scale),
            "fc_weight": self.fc_weight_q.numpy().astype(np.int8),
            "fc_weight_scale": np.float32(self.fc_weight_scale),
            "fc_bias": self.fc_bias.numpy().astype(np.float32),
        }
        size = arrays["fc_weight"].size + 4 * arrays["fc_bias"].size
        scale_in = self.input_scale
        for i, layer in enumerate(self.layers):
            bias_scale = scale_in * layer["weight_scale"]
            arrays[f"conv{i}_weight"] = layer["weight_q"].numpy().astype(np.int8)
            arrays[f"conv{i}_weight_scale"] = layer["weight_scale"].numpy().astype(np.float32)
            arrays[f"conv{i}_bias"] = torch.round(layer["bias"] / bias_scale).numpy().astype(np.int32)
            arrays[f"conv{i}_output_scale"] = np.float32(layer["output_scale"])
            arrays[f"conv{i}_stride"] = np.array(layer["stride"])
            arrays[f"conv{i}_groups"] = np.int32(layer["groups"])
            # Per channel: int8 weights, an int32 bias, and an int32
            # multiplier plus a shift byte for rescaling.
            size += arrays[f"conv{i}_weight"].size + 9 * len(layer["weight_scale"])
            scale_in = layer["output_scale"]
        np.savez(path, **arrays)
        return size


class QuantizedTCS(nn.Module):
    """8-bit simulation of baselines.TCS, with the same call signature.

    Every tensor the chip would store is rounded: the centred input and the
    output of each depthwise convolution are signed int8 (no ReLU follows
    them), each pointwise output is uint8 after a ReLU or int8 without one
    (the last layer of a block, before the residual add), and each block's
    output after the add and ReLU is uint8. Weights are int8 per output
    channel, batch-norm folded into the pointwise convolution, biases int32
    at (input scale x weight scale).
    """

    def __init__(self, model: nn.Module, calibration: torch.Tensor, percentile: float = 99.99):
        super().__init__()
        self.percentile = percentile
        self.stem = self._fold(model.features[0])
        self.blocks = [[self._fold(layer) for layer in block.layers] for block in model.features[1:]]
        self.fc_weight = model.embed.weight.detach()
        self.fc_bias = model.embed.bias.detach()
        for layer in [self.stem] + [layer for block in self.blocks for layer in block]:
            for kind in ("dw", "pw"):
                weight = layer[kind]
                scale = weight.abs().amax(dim=(1, 2, 3)).clamp_min(1e-8) / 127.0
                layer[kind + "_scale"] = scale
                layer[kind + "_q"] = _round_to(weight, scale[:, None, None, None], -127, 127)
        self.fc_weight_scale = self.fc_weight.abs().max() / 127.0
        self.fc_weight_q = _round_to(self.fc_weight, self.fc_weight_scale, -127, 127)

        self.scales: dict[str, float] = {}
        self._calibrating = True
        self._run(calibration)
        self._calibrating = False

    @staticmethod
    def _fold(layer: nn.Module) -> dict:
        depthwise, pointwise, norm = layer[0], layer[1], layer[2]
        gain = norm.weight / torch.sqrt(norm.running_var + norm.eps)
        return {
            "dw": depthwise.weight.detach(),
            "stride": depthwise.stride,
            "padding": depthwise.padding,
            "pw": (pointwise.weight * gain[:, None, None, None]).detach(),
            "bias": (norm.bias - norm.running_mean * gain).detach(),
            "relu": len(layer) > 3,
        }

    def _act(self, x: torch.Tensor, key: str, signed: bool) -> torch.Tensor:
        """Records a scale while calibrating; otherwise rounds as the chip would."""
        if self._calibrating:
            sample = x.flatten()
            sample = sample[torch.randperm(len(sample))[:2_000_000]]
            peak = float(torch.quantile(sample.abs(), self.percentile / 100.0))
            self.scales[key] = max(peak, 1e-8) / (127.0 if signed else 255.0)
            return x
        scale = self.scales[key]
        x = _round_to(x, scale, -128, 127) if signed else _round_to(x, scale, 0, 255)
        return x * scale

    def _layer(self, x: torch.Tensor, layer: dict, key: str, scale_in: float) -> tuple[torch.Tensor, float]:
        quant = not self._calibrating
        dw = layer["dw_q"] * layer["dw_scale"][:, None, None, None] if quant else layer["dw"]
        x = nn.functional.conv2d(x, dw, None, stride=layer["stride"], padding=layer["padding"], groups=dw.shape[0])
        x = self._act(x, key + ".dw", signed=True)
        scale_in = self.scales.get(key + ".dw", 1.0)
        pw = layer["pw_q"] * layer["pw_scale"][:, None, None, None] if quant else layer["pw"]
        bias = layer["bias"]
        if quant:
            bias_scale = scale_in * layer["pw_scale"]
            bias = torch.round(bias / bias_scale) * bias_scale
        x = nn.functional.conv2d(x, pw, bias)
        if layer["relu"]:
            x = torch.relu(x)
        x = self._act(x, key + ".pw", signed=not layer["relu"])
        return x, self.scales.get(key + ".pw", 1.0)

    def _run(self, features: torch.Tensor) -> torch.Tensor:
        x = (features - features.mean(dim=1, keepdim=True)).transpose(1, 2).unsqueeze(3)
        x = self._act(x, "input", signed=True)
        x, scale = self._layer(x, self.stem, "stem", self.scales.get("input", 1.0))
        for b, block in enumerate(self.blocks):
            skip, y, s = x, x, scale
            for i, layer in enumerate(block):
                y, s = self._layer(y, layer, f"b{b}.l{i}", s)
            x = self._act(torch.relu(skip + y), f"b{b}.out", signed=False)
            scale = self.scales.get(f"b{b}.out", 1.0)
        x = x.squeeze(3).transpose(1, 2)  # [batch, time, channels]
        pooled = self._act(torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1), "pooled", signed=False)
        fc = self.fc_weight if self._calibrating else self.fc_weight_q * self.fc_weight_scale
        return nn.functional.normalize(pooled @ fc.T + self.fc_bias, dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self._run(features)

    def export(self, path: pathlib.Path) -> int:
        """Saves integers and scales; returns the weight bytes on the chip."""
        arrays = {f"scale.{k}": np.float32(v) for k, v in self.scales.items()}
        arrays["fc_weight"] = self.fc_weight_q.numpy().astype(np.int8)
        arrays["fc_weight_scale"] = np.float32(self.fc_weight_scale)
        arrays["fc_bias"] = self.fc_bias.numpy().astype(np.float32)
        size = arrays["fc_weight"].size + 4 * arrays["fc_bias"].size
        named = [("stem", self.stem)] + [(f"b{b}.l{i}", l) for b, block in enumerate(self.blocks) for i, l in enumerate(block)]
        for key, layer in named:
            scale_in = self.scales[key + ".dw"]
            arrays[key + ".dw"] = layer["dw_q"].numpy().astype(np.int8)
            arrays[key + ".dw_scale"] = layer["dw_scale"].numpy().astype(np.float32)
            arrays[key + ".pw"] = layer["pw_q"].numpy().astype(np.int8)
            arrays[key + ".pw_scale"] = layer["pw_scale"].numpy().astype(np.float32)
            arrays[key + ".bias"] = torch.round(layer["bias"] / (scale_in * layer["pw_scale"])).numpy().astype(np.int32)
            arrays[key + ".stride"] = np.array(layer["stride"])
            arrays[key + ".relu"] = np.int32(layer["relu"])
            # Depthwise: int8 weights plus a multiplier and shift per channel.
            # Pointwise: int8 weights, int32 bias, multiplier and shift.
            size += arrays[key + ".dw"].size + 5 * arrays[key + ".dw"].shape[0]
            size += arrays[key + ".pw"].size + 9 * arrays[key + ".pw"].shape[0]
        np.savez(path, **arrays)
        return size


def quantize(model: nn.Module, calibration: torch.Tensor, percentile: float) -> nn.Module:
    if isinstance(model, spk.Student):
        return QuantizedStudent(model, calibration, percentile)
    if type(model).__name__ == "TCS":
        return QuantizedTCS(model, calibration, percentile)
    raise SystemExit(f"error: no 8-bit simulation for {type(model).__name__}")


def load_float(name: str) -> nn.Module:
    saved = torch.load(_MODELS / f"{name}.pt", map_location="cpu")
    model = spk.make_model(saved.get("arch", "student"), saved["width"])
    model.load_state_dict(saved["state"])
    return model.eval()


def calibration_crops(count: int, seed: int = 0) -> torch.Tensor:
    """Training speech, half clean and half with room/noise effects, so the
    scales cover both."""
    rng = np.random.default_rng(seed)
    split = spk.Split("train-clean-100")
    eligible = np.flatnonzero(split.lengths >= spk.CROP_FRAMES)
    utterances = rng.choice(eligible, size=count, replace=False)
    crops = np.stack(
        [split.crop(int(u), int(rng.integers(0, split.lengths[u] - spk.CROP_FRAMES + 1))) for u in utterances]
    )
    half = count // 2
    crops[half:] = spk.augment(crops[half:], rng, room=True)
    return torch.from_numpy(crops)


def embeddings(model: nn.Module, crops: np.ndarray) -> np.ndarray:
    return np.concatenate([model(torch.from_numpy(crops[i : i + 256])).numpy() for i in range(0, len(crops), 256)])


def activation_bytes(model: spk.Student) -> list[int]:
    """Size of each layer's int8 output for one 1.5 s window."""
    sizes = [spk.CROP_FRAMES * spk.ref.NUM_MELS]
    x = torch.zeros(1, 1, spk.CROP_FRAMES, spk.ref.NUM_MELS)
    with torch.no_grad():
        for layer in fold_batch_norms(model):
            x = QuantizedStudent._conv(x, layer, layer["weight"], layer["bias"])
            sizes.append(x.numel())
    return sizes


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="Model under data/models, e.g. v1")
    parser.add_argument("--calibration-crops", type=int, default=1000)
    parser.add_argument("--percentile", type=float, default=99.99, help="Activation range percentile")
    parser.add_argument("--split", default="test-clean")
    parser.add_argument("--per-speaker", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(0)
    model = load_float(args.name)
    quantized = quantize(model, calibration_crops(args.calibration_crops), args.percentile)

    split = spk.Split(args.split)
    trials = spk.Trials(split, args.per_speaker)
    crops = trials.crops()
    rng = np.random.default_rng(1)
    conditions = {
        "clean": crops,
        "noise 10 dB quieter": spk.mix_log_mel(crops, spk.random_noise_log_mel(rng, len(crops)), np.full(len(crops), 10.0)),
        "room and device effects": spk.room_effects(crops, rng),
    }

    print(f"{args.name}: error rate on {args.split}, {len(trials.same)} trials")
    print(f"  {'condition':<26}{'float':>8}{'8-bit':>8}")
    agreement = []
    for label, data in conditions.items():
        original = embeddings(model, data)
        eight_bit = embeddings(quantized, data)
        agreement.append(float(np.mean(np.sum(original * eight_bit, axis=1))))
        print(f"  {label:<26}{100 * trials.eer(original):7.2f}%{100 * trials.eer(eight_bit):7.2f}%")
    print(f"  8-bit voiceprint vs float voiceprint, average similarity: {np.mean(agreement):.4f}")

    out = _MODELS / f"{args.name}.int8.npz"
    weight_bytes = quantized.export(out)
    if isinstance(model, spk.Student):
        sizes = activation_bytes(model)
        # Layers run one after another, so working memory is the largest
        # input+output pair held at once.
        working = max(a + b for a, b in zip(sizes, sizes[1:]))
        print(f"  weights on the chip: {weight_bytes / 1024:.0f} KB   working memory per window: {working / 1024:.0f} KB")
        print(f"  layer output sizes (bytes): {sizes}")
    else:
        print(f"  weights on the chip: {weight_bytes / 1024:.0f} KB   (working memory depends on the streaming code)")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
