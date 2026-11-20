# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "soundfile", "torch<2.9"]
# ///
"""What each model would cost on the STM32F407, from its layer shapes.

For every convolution it records the 8-bit input and output sizes of one
1.5 s window, then reports:

  weights     int8 weights plus, per output channel, an int32 bias and a
              multiplier/shift (9 bytes), as the board stores them.
  ping-pong   largest input + output held at once. This is how the board
              runs v3 today: each layer reads one buffer and writes the other.
  in place    the same, but depthwise and pointwise layers overwrite their
              input (they can, with a small row or pixel buffer), so only
              layers that change the shape need two buffers. A lower bound
              on what any careful implementation needs.
  est. time   multiply-adds scaled by v3's measured speed on the board
              (174 ms for one window). A rough estimate: the kernels differ.

The board has 128 KB of main RAM; the current firmware leaves about 56 KB
for network buffers (the rest is audio, USB, tasks and stacks). The
separate 64 KB core-coupled RAM (CCM) is not counted; using it would mean
splitting tensors across two memory regions.

Usage:
  uv run tools/train/chip_cost.py v3 base_dscnn base_bcresnet
"""

import argparse
import pathlib
import sys

import torch
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spk  # noqa: E402

V3_MS_PER_WINDOW = 174.0  # Measured on the board, network only.
V3_NAME = "v3"
FREE_RAM_BYTES = 56 * 1024


def layer_shapes(model: nn.Module) -> list[dict]:
    layers = []

    def hook(module, inputs, output):
        kernel = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
        layers.append(
            {
                "in": inputs[0][0].numel(),
                "out": output[0].numel(),
                "macs": output[0].numel() * kernel,
                "weights": module.weight.numel() + 9 * module.out_channels,
                # Same shape in and out: depthwise or pointwise, can overwrite its input.
                "in_place": tuple(inputs[0].shape) == tuple(output.shape),
            }
        )

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model.eval()(torch.zeros(1, spk.CROP_FRAMES, spk.ref.NUM_MELS))
    for h in handles:
        h.remove()
    return layers


def cost(name: str) -> dict:
    saved = torch.load(spk._MODELS / f"{name}.pt", map_location="cpu")
    model = spk.make_model(saved.get("arch", "student"), saved["width"])
    model.load_state_dict(saved["state"])
    layers = layer_shapes(model)
    params, macs = spk.model_cost(model)
    head = model.embed
    return {
        "name": name,
        "params": params,
        "macs": macs,
        "weights": sum(l["weights"] for l in layers) + head.weight.numel() + 4 * head.out_features,
        "largest": max(l["out"] for l in layers),
        "ping_pong": max(l["in"] + l["out"] for l in layers),
        "in_place": max(l["out"] if l["in_place"] else l["in"] + l["out"] for l in layers),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()

    v3_macs = cost(V3_NAME)["macs"]
    print(f"{'model':<15}{'params':>9}{'MACs':>9}{'weights':>10}{'largest':>10}"
          f"{'ping-pong':>11}{'in place':>10}{'est. time':>11}  fits in {FREE_RAM_BYTES // 1024} KB?")
    for name in args.models:
        c = cost(name)
        ms = V3_MS_PER_WINDOW * c["macs"] / v3_macs
        fits = "yes" if c["ping_pong"] <= FREE_RAM_BYTES else ("only in place" if c["in_place"] <= FREE_RAM_BYTES else "no")
        print(f"{name:<15}{c['params'] / 1e3:8.1f}k{c['macs'] / 1e6:8.2f}M{c['weights'] / 1024:8.0f}KB"
              f"{c['largest'] / 1024:8.1f}KB{c['ping_pong'] / 1024:9.1f}KB{c['in_place'] / 1024:8.1f}KB"
              f"{ms:9.0f}ms  {fits}")
    print("\nest. time: scaled from v3's measured 174 ms per window; decisions come every 400 ms.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
