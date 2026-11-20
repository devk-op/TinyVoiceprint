"""Published small audio networks, adapted as speaker encoders, for comparison.

HyVox (arXiv 2611.03735) used compact keyword-spotting networks as
embedded speaker-encoder baselines. These are the same two families,
reimplemented here so they can be trained with exactly our recipe
(spk.py: distillation from the same teacher, same data and augmentation);
only the architecture differs.

Each takes [batch, CROP_FRAMES, NUM_MELS] log-mel, subtracts the per-band
mean like spk.Student, and returns a unit 192-d voiceprint. As in the
original papers, features are averaged over frequency at the end; the time
axis then gets the same mean/spread pooling and linear head as spk.Student.

  dscnn     DS-CNN-S (Zhang et al., "Hello Edge", 2017): a strided 10x4
            convolution, then four depthwise-separable 3x3 blocks, 64
            channels throughout.
  bcresnet  BC-ResNet-1 (Kim et al., "Broadcasted Residual Learning",
            Interspeech 2021): frequency-wise depthwise convolutions with
            a broadcast time-wise branch. Adapted: plain batch-norm in
            place of sub-spectral norm, and our pooling head.

  tcs       1-D time-channel-separable convolutions with residual blocks,
            after QuartzNet as used for speaker verification by Balian et
            al. (ICASSP 2021), sized for the F407. Mel bands are the input
            channels, so each layer sees all frequencies and convolves only
            along time; every frame is computed once, which suits streaming.
            Adapted: ReLU for max-feature-map, our mean/spread pooling for
            GhostVLAD, 40 mel bands, fewer and narrower layers.

These are reimplementations from the papers' descriptions; parameter and
compute counts are reported by spk.model_cost, not copied from the papers.
"""

import torch
from torch import nn

TEACHER_DIM = 192


def _pool_and_embed(x: torch.Tensor, embed: nn.Linear) -> torch.Tensor:
    """[batch, channels, time, mel] -> unit voiceprint: average over mel
    (the originals' global average pooling), then mean/spread over time."""
    x = x.mean(dim=3).transpose(1, 2)  # [batch, time, channels]
    pooled = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)
    return nn.functional.normalize(embed(pooled), dim=1)


class DSCNN(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        c = channels
        layers: list[nn.Module] = [
            nn.Conv2d(1, c, (10, 4), stride=(2, 2), padding=(4, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        ]
        for _ in range(4):
            layers += [
                nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
        self.features = nn.Sequential(*layers)
        self.embed = nn.Linear(2 * c, TEACHER_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=1, keepdim=True)
        return _pool_and_embed(self.features(x.unsqueeze(1)), self.embed)


class _BCBlock(nn.Module):
    """Broadcasted residual block. Layout is [batch, ch, time, mel]."""

    def __init__(self, cin: int, cout: int, mel_stride: int, dilation: int):
        super().__init__()
        self.transition = cin != cout or mel_stride != 1
        if self.transition:
            self.project = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        # f2: frequency-wise depthwise 3 (along mel).
        self.freq = nn.Sequential(
            nn.Conv2d(cout, cout, (1, 3), stride=(1, mel_stride), padding=(0, 1), groups=cout, bias=False),
            nn.BatchNorm2d(cout),
        )
        # f1: on the mel-averaged signal, time-wise depthwise 3 (dilated).
        self.time = nn.Sequential(
            nn.Conv2d(cout, cout, (3, 1), padding=(dilation, 0), dilation=(dilation, 1), groups=cout, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 1, bias=False),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transition:
            x = self.project(x)
        aux = self.freq(x)
        out = self.time(aux.mean(dim=3, keepdim=True)) + aux  # Broadcast over mel.
        if not self.transition:
            out = out + x
        return torch.relu(out)


class BCResNet(nn.Module):
    def __init__(self, multiplier: int = 1):
        super().__init__()
        widths = [int(w * multiplier) for w in (8, 12, 16, 20)]
        repeats = (2, 2, 4, 4)
        mel_strides = (1, 2, 2, 1)
        dilations = (1, 2, 4, 8)
        stem = 16 * multiplier
        layers: list[nn.Module] = [
            nn.Conv2d(1, stem, 5, stride=(1, 2), padding=2, bias=False),
            nn.BatchNorm2d(stem),
            nn.ReLU(inplace=True),
        ]
        cin = stem
        for width, count, stride, dilation in zip(widths, repeats, mel_strides, dilations):
            for i in range(count):
                layers.append(_BCBlock(cin, width, stride if i == 0 else 1, dilation))
                cin = width
        out = 32 * multiplier
        layers += [nn.Conv2d(cin, out, 1, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.embed = nn.Linear(2 * out, TEACHER_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=1, keepdim=True)
        return _pool_and_embed(self.features(x.unsqueeze(1)), self.embed)


class _TCSLayer(nn.Sequential):
    """Depthwise convolution along time, then pointwise; [batch, c, time, 1]."""

    def __init__(self, cin: int, cout: int, kernel: int, stride: int = 1, relu: bool = True):
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cin, (kernel, 1), stride=(stride, 1), padding=(kernel // 2, 0), groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class _TCSBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, repeats: int):
        super().__init__()
        self.layers = nn.Sequential(
            *[_TCSLayer(channels, channels, kernel, relu=i + 1 < repeats) for i in range(repeats)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.layers(x))


class TCS(nn.Module):
    """Stem at half the frame rate (50 per second), then `blocks` residual
    blocks of `repeats` layers each."""

    def __init__(self, channels: int = 80, blocks: int = 5, repeats: int = 3, kernel: int = 5, mels: int = 40):
        super().__init__()
        self.features = nn.Sequential(
            _TCSLayer(mels, channels, kernel, stride=2),
            *[_TCSBlock(channels, kernel, repeats) for _ in range(blocks)],
        )
        self.embed = nn.Linear(2 * channels, TEACHER_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=1, keepdim=True)
        x = self.features(x.transpose(1, 2).unsqueeze(3))  # [batch, channels, time, 1]
        return _pool_and_embed(x, self.embed)
