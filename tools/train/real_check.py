# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "sounddevice", "soundfile", "torch<2.9"]
# ///
"""Scores models on real recordings made with the board's microphone.

A small sanity check to sit beside the LibriSpeech numbers. The target is
enrolled from recordings/enroll_krathi.wav; windows of the same person from
other sessions should match, and windows of voices played through a
loudspeaker (a video, a synthetic voice) should not. Turn labels come from
the *.diarization.json files written by tools/host/voices.py.

A name ending in .int8 uses the 8-bit simulation from quantize.py, and
.int (tcs only) the chip's integer arithmetic from tcs_integer.py.

Usage:
  uv run tools/train/real_check.py v1 v1.int8 v2_w32
"""

import json
import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tools" / "host"))
sys.path.insert(0, str(_ROOT / "tools" / "train"))
import duration_eval  # noqa: E402
import live  # noqa: E402
import quantize  # noqa: E402
import spk  # noqa: E402

_RECORDINGS = _ROOT / "recordings"
_ENROLL = "enroll_krathi"
_TARGET = [("me", None), ("claude_talk", "krathi"), ("conv_take1", "krathi"), ("me_and_video", "krathi")]
_OTHERS = [("listen_last", None), ("claude_only", None), ("claude_talk", "claude"), ("conv_take1", "claude"), ("me_and_video", "speaker 1")]


def windows(name: str, who: str | None = None) -> list[np.ndarray]:
    """Speech windows of a recording, optionally only one speaker's turns."""
    pcm = live.read_wav(_RECORDINGS / f"{name}.wav")
    floor = live.background_db(pcm)
    spans = [(0.0, len(pcm) / live._RATE)]
    if who:
        turns = json.loads((_RECORDINGS / f"{name}.diarization.json").read_text())
        spans = [(turn["start"], turn["end"]) for turn in turns if turn["speaker"] == who]
    found = []
    for begin, end in spans:
        for start in range(int(begin * live._RATE), int(end * live._RATE) - live._WINDOW + 1, live._HOP):
            window = pcm[start : start + live._WINDOW]
            if live.is_speech(window, floor):
                found.append(window)
    return found


_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


@torch.no_grad()
def main() -> int:
    names = sys.argv[1:] or ["v1"]
    enroll = np.stack(windows(_ENROLL))
    target = np.stack([w for name, who in _TARGET for w in windows(name, who)])
    others = np.stack([w for name, who in _OTHERS for w in windows(name, who)])
    print(f"{len(target)} windows of the enrolled speaker from other sessions, {len(others)} of loudspeaker voices")
    for name in names:
        model = duration_eval.load(name)
        voice = live.unit(live.embed(model, enroll).mean(axis=0))
        same = live.embed(model, target) @ voice
        different = live.embed(model, others) @ voice
        eer = spk.equal_error_rate(
            np.concatenate([same, different]),
            np.concatenate([np.ones(len(same), bool), np.zeros(len(different), bool)]),
        )
        print(f"  {name:<12} speaker mean {same.mean():.2f}   others mean {different.mean():.2f}   error rate {100 * eer:.1f}%")
        passes = "  ".join(f"{th:.2f}: {100 * (same >= th).mean():.0f}%/{100 * (different >= th).mean():.0f}%" for th in _THRESHOLDS)
        print(f"  {'':<12} pass rate speaker/others at {passes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
