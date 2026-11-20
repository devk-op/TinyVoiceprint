# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "sounddevice", "soundfile", "speechbrain>=1.0", "torch<2.9", "torchaudio<2.9"]
# ///
"""Error rate as a function of how much speech each decision uses.

The board makes a decision from several overlapping 1.5 s windows averaged
together (VoiceMatch::kSmoothWindows). This measures how the error rate
falls as more windows are averaged, the way the board would do it:

  * dataset splits: per utterance, K windows 0.4 s apart (the board's
    decision interval), voiceprints averaged; every pair of utterances
    from different recordings is a trial. The same utterances are used at
    every K, so only the amount of speech changes. The large teacher model
    is scored on the same stretch of audio for reference.
  * real board recordings (tools/train/real_check.py): the enrolled
    speaker vs loudspeaker voices, averaging K consecutive windows 0.25 s
    apart within one speaker's turn.

A model name ending in .int8 uses the 8-bit simulation (quantize.py),
which matches the board's integer network.

Usage:
  uv run tools/train/duration_eval.py v3.int8
  uv run tools/train/duration_eval.py v3.int8 --splits cv-test --no-teacher
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tools" / "train"))
sys.path.insert(0, str(_ROOT / "tools" / "host"))
import live  # noqa: E402
import quantize  # noqa: E402
import real_check  # noqa: E402
import spk  # noqa: E402

_HOP_FRAMES = 40  # 0.4 s between decisions on the board.
_WINDOWS = (1, 2, 3, 5, 9)  # 1.5, 1.9, 2.3, 3.1, 4.7 s of speech.
_REAL_HOP_S = 0.25  # Window spacing in real_check.windows().
_REAL_WINDOWS = (1, 4, 7, 13)  # 1.5, 2.25, 3.0, 4.5 s.


def load(name: str):
    """name, name.int8 (8-bit float simulation) or name.int (tcs only: the
    chip's integer arithmetic, tcs_integer.py)."""
    base = name.removesuffix(".int8").removesuffix(".int")
    model = quantize.load_float(base)
    if name != base:
        torch.manual_seed(0)
        model = quantize.quantize(model, quantize.calibration_crops(1000), 99.99)
    if name.endswith(".int"):
        import tcs_integer

        model = tcs_integer.IntegerTCS(model)
    return model


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def pick_utterances(split: spk.Split, per_speaker: int, rng: np.random.Generator) -> np.ndarray:
    """Utterances long enough for the largest K, up to per_speaker each."""
    need = spk.CROP_FRAMES + _HOP_FRAMES * (max(_WINDOWS) - 1)
    chosen = []
    for speaker in range(split.num_speakers):
        own = np.flatnonzero((split.speaker == speaker) & (split.lengths >= need))
        if len(own) >= 2:
            chosen += rng.choice(own, size=min(per_speaker, len(own)), replace=False).tolist()
    return np.array(sorted(chosen))


@torch.no_grad()
def dataset_curve(model, split_name: str, per_speaker: int, teacher) -> None:
    split = spk.Split(split_name)
    rng = np.random.default_rng(0)
    utterances = pick_utterances(split, per_speaker, rng)
    need = spk.CROP_FRAMES + _HOP_FRAMES * (max(_WINDOWS) - 1)
    starts = np.array([rng.integers(0, split.lengths[u] - need + 1) for u in utterances])
    labels = split.speaker[utterances]
    upper = np.triu_indices(len(utterances), k=1)
    same = labels[upper[0]] == labels[upper[1]]

    # Voiceprint of every window position once; averages are taken from these.
    crops = np.stack(
        [split.crop(int(u), int(s) + _HOP_FRAMES * k) for u, s in zip(utterances, starts) for k in range(max(_WINDOWS))]
    )
    prints = quantize.embeddings(model, crops).reshape(len(utterances), max(_WINDOWS), -1)

    print(f"\n{split_name}: {len(utterances)} utterances from {len(set(labels.tolist()))} speakers, {len(same)} trials")
    print(f"  {'speech per decision':<22}{'student':>9}{'teacher':>9}")
    for k in _WINDOWS:
        emb = unit(prints[:, :k].mean(axis=1))
        scores = np.einsum("ij,ij->i", emb[upper[0]], emb[upper[1]])
        student = spk.equal_error_rate(scores, same)
        teacher_text = ""
        if teacher is not None:
            seconds_frames = spk.CROP_FRAMES + _HOP_FRAMES * (k - 1)
            clips = []
            for u, s in zip(utterances, starts):
                pcm = spk.read_pcm(str(split.paths[u]))
                clips.append(pcm[s * spk.ref.HOP : s * spk.ref.HOP + spk.ref.FRAME_LENGTH + spk.ref.HOP * (seconds_frames - 1)])
            t_emb = np.concatenate([teacher.embed(clips[i : i + 48]) for i in range(0, len(clips), 48)])
            t_scores = np.einsum("ij,ij->i", t_emb[upper[0]], t_emb[upper[1]])
            teacher_text = f"{100 * spk.equal_error_rate(t_scores, same):8.2f}%"
        seconds = (spk.CROP_FRAMES + _HOP_FRAMES * (k - 1)) / 100
        print(f"  {seconds:4.1f} s ({k} window{'s' if k > 1 else ''}){'':<7}{100 * student:8.2f}%{teacher_text}")


@torch.no_grad()
def real_curve(model) -> None:
    def spans(name, who):
        """Windows grouped by turn, so averages never mix two turns."""
        out = []
        recording = real_check._RECORDINGS / f"{name}.wav"
        if who is None:
            out.append(real_check.windows(name))
        else:
            import json

            pcm = live.read_wav(recording)
            floor = live.background_db(pcm)
            for turn in json.loads((real_check._RECORDINGS / f"{name}.diarization.json").read_text()):
                if turn["speaker"] != who:
                    continue
                found = []
                for start in range(int(turn["start"] * live._RATE), int(turn["end"] * live._RATE) - live._WINDOW + 1, live._HOP):
                    window = pcm[start : start + live._WINDOW]
                    if live.is_speech(window, floor):
                        found.append(window)
                out.append(found)
        return [s for s in out if s]

    def prints(groups):
        return [live.embed(model, np.stack(g)) for g in groups]

    voice = unit(live.embed(model, np.stack(real_check.windows(real_check._ENROLL))).mean(axis=0))
    target = prints([g for name, who in real_check._TARGET for g in spans(name, who)])
    others = prints([g for name, who in real_check._OTHERS for g in spans(name, who)])
    print("\nreal board recordings (enrolled speaker vs loudspeaker voices)")
    print(f"  {'speech per decision':<22}{'error':>8}{'decisions':>12}")
    for k in _REAL_WINDOWS:
        def scores(groups):
            return np.array([float(unit(g[i : i + k].mean(axis=0)) @ voice) for g in groups for i in range(len(g) - k + 1)])

        s_t, s_o = scores(target), scores(others)
        if len(s_t) == 0 or len(s_o) == 0:
            continue
        eer = spk.equal_error_rate(np.concatenate([s_t, s_o]), np.concatenate([np.ones(len(s_t), bool), np.zeros(len(s_o), bool)]))
        seconds = 1.5 + _REAL_HOP_S * (k - 1)
        print(f"  {seconds:4.2f} s ({k} windows){'':<7}{100 * eer:7.1f}%{len(s_t):>6} + {len(s_o)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="Model under data/models; add .int8 for the board's 8-bit version")
    parser.add_argument("--splits", nargs="*", default=["cv-test", "test-clean"])
    parser.add_argument("--per-speaker", type=int, default=20)
    parser.add_argument("--no-teacher", action="store_true")
    parser.add_argument("--no-real", action="store_true")
    args = parser.parse_args()

    model = load(args.name)
    teacher = None if args.no_teacher else spk.Teacher()
    print(f"model {args.name}")
    for split in args.splits:
        dataset_curve(model, split, args.per_speaker, teacher)
    if not args.no_real:
        real_curve(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
