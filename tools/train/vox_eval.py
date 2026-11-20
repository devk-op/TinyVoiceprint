# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "sounddevice", "soundfile", "speechbrain>=1.0", "torch<2.9", "torchaudio<2.9"]
# ///
"""Scores models on VoxCeleb1-O, the standard speaker-verification benchmark.

Uses the official cleaned trial list (veri_test2.txt: 37,611 trials over
4,708 clips from 40 speakers). Data: VoxCeleb1 test audio (Nagrani et al.,
2017; CC BY 4.0), unpacked under data/voxceleb/wav/.

Students are scored the way the board decides: 1.5 s windows every 0.4 s,
voiceprints averaged. Reported both over the whole clip (as other papers
do) and over only the first N seconds (as the board would with N seconds of
speech). The teacher is scored on the same audio. Metrics: equal error
rate, and minimum detection cost (minDCF, p_target 0.01, the VoxSRC
convention).

A model name ending in .int8 uses the 8-bit simulation that matches the
board's integer network.

Usage:
  uv run tools/train/vox_eval.py v3.int8 v3 --teacher
  uv run tools/train/vox_eval.py base_bcresnet --seconds 2.3
  uv run tools/train/vox_eval.py v3.int8 --teacher --replay

--replay scores the clips recorded through the board's microphone by
tools/host/vox_replay.py, with every pair of those clips as a trial. The
same trials are scored on the original files (to separate the effect of
the room and microphone from the smaller subset), and with one side
original and one side replayed.
"""

import argparse
import pathlib
import sys

import numpy as np
import soundfile as sf
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tools" / "train"))
import duration_eval  # noqa: E402
import quantize  # noqa: E402
import spk  # noqa: E402

_VOX = _ROOT / "data" / "voxceleb"
_REPLAY = _ROOT / "recordings" / "vox_replay"
_HOP_FRAMES = 40  # 0.4 s, the board's decision interval.
# Voiceprints per clip, for tools/train/stats.py (confidence intervals,
# paired comparisons, error rates at a threshold) without re-running models.
PRINTS = _ROOT / "data" / "models" / "prints"


def save_prints(model: str, test: str, prints: np.ndarray) -> None:
    PRINTS.mkdir(parents=True, exist_ok=True)
    np.save(PRINTS / f"{model}__{test}.npy", prints.astype(np.float32))


def min_dcf(scores: np.ndarray, same: np.ndarray, p_target: float = 0.01) -> float:
    """Normalised minimum detection cost (C_miss = C_fa = 1)."""
    order = np.argsort(scores)
    same = same[order]
    n_target, n_nontarget = same.sum(), (~same).sum()
    # Threshold just above each score: misses grow, false alarms shrink.
    miss = np.concatenate([[0], np.cumsum(same)]) / n_target
    false_alarm = 1 - np.concatenate([[0], np.cumsum(~same)]) / n_nontarget
    cost = p_target * miss + (1 - p_target) * false_alarm
    return float(cost.min() / min(p_target, 1 - p_target))


def read_trials() -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    lines = (_VOX / "veri_test2.txt").read_text().split("\n")
    rows = [line.split() for line in lines if line.strip()]
    clips = sorted({r[1] for r in rows} | {r[2] for r in rows})
    index = {c: i for i, c in enumerate(clips)}
    same = np.array([r[0] == "1" for r in rows])
    a = np.array([index[r[1]] for r in rows])
    b = np.array([index[r[2]] for r in rows])
    return same, clips, a, b


def load_audio(clips: list[str], paths: list[str] | None = None) -> list[np.ndarray]:
    audio = []
    missing = 0
    for i, clip in enumerate(clips):
        path = pathlib.Path(paths[i]) if paths else _VOX / "wav" / clip
        if not path.exists():
            missing += 1
            audio.append(None)
            continue
        pcm, rate = sf.read(str(path), dtype="int16")
        assert rate == 16000 and pcm.ndim == 1, path
        audio.append(pcm)
    if missing:
        raise SystemExit(f"error: {missing} of {len(clips)} trial clips missing under {_VOX / 'wav'}")
    return audio


@torch.no_grad()
def student_prints(model, audio: list[np.ndarray], seconds: float | None) -> np.ndarray:
    out = []
    for pcm in audio:
        if seconds is not None:
            pcm = pcm[: int(seconds * 16000)]
        feats = spk.log_mel_torch(torch.from_numpy(pcm.astype(np.int16))[None])[0].numpy()
        starts = range(0, max(1, len(feats) - spk.CROP_FRAMES + 1), _HOP_FRAMES)
        crops = np.stack([feats[s : s + spk.CROP_FRAMES] for s in starts]).astype(np.float32)
        prints = quantize.embeddings(model, crops)
        out.append(duration_eval.unit(prints.mean(axis=0)))
    return np.stack(out)


@torch.no_grad()
def teacher_prints(teacher, audio: list[np.ndarray], seconds: float | None) -> np.ndarray:
    clips = [pcm if seconds is None else pcm[: int(seconds * 16000)] for pcm in audio]
    order = np.argsort([len(c) for c in clips])  # Similar lengths per batch.
    out = np.zeros((len(clips), spk.TEACHER_DIM), dtype=np.float32)
    batch = []
    for i in order:
        batch.append(i)
        if sum(len(clips[j]) for j in batch) > 60 * 16000 or i == order[-1]:
            out[batch] = teacher.embed([clips[j] for j in batch])
            batch = []
            if teacher.device.type == "mps":
                torch.mps.empty_cache()
    return out


def replay_trials() -> tuple[np.ndarray, list[str], list[str], np.ndarray, np.ndarray]:
    """All pairs of the replayed clips: (same, clips, replayed paths, a, b)."""
    names = sorted(p.name for p in _REPLAY.glob("id*.wav"))
    if not names:
        raise SystemExit(f"error: no replayed clips under {_REPLAY}; run tools/host/vox_replay.py")
    # id10270_OhfKF8FSq3Y_00003.wav -> id10270/OhfKF8FSq3Y/00003.wav (video ids may contain "_").
    clips = []
    for n in names:
        speaker, rest = n.split("_", 1)
        video, clip = rest.rsplit("_", 1)
        clips.append(f"{speaker}/{video}/{clip}")
    a, b = np.triu_indices(len(clips), k=1)
    speakers = np.array([c.split("/")[0] for c in clips])
    return speakers[a] == speakers[b], clips, [str(_REPLAY / n) for n in names], a, b


def report(label: str, prints: np.ndarray, a: np.ndarray, b: np.ndarray, same: np.ndarray,
           prints_b: np.ndarray | None = None) -> None:
    scores = np.einsum("ij,ij->i", prints[a], (prints if prints_b is None else prints_b)[b])
    eer = spk.equal_error_rate(scores, same)
    print(f"  {label:<32} EER {100 * eer:6.2f}%   minDCF {min_dcf(scores, same):.3f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="Models under data/models; add .int8 for the board's 8-bit version")
    parser.add_argument("--teacher", action="store_true", help="Also score the teacher (ECAPA-TDNN)")
    parser.add_argument("--seconds", type=float, nargs="*", default=[1.5, 2.3, 4.7], help="Also score on only the first N seconds")
    parser.add_argument("--replay", action="store_true", help="Score the clips replayed through the board instead")
    args = parser.parse_args()
    if args.replay:
        return main_replay(args)

    same, clips, a, b = read_trials()
    audio = load_audio(clips)
    lengths = np.array([len(x) for x in audio]) / 16000
    print(f"VoxCeleb1-O: {len(same)} trials ({same.sum()} same speaker), {len(clips)} clips, "
          f"median {np.median(lengths):.1f} s")
    durations: list[float | None] = [None] + list(args.seconds)

    def name(seconds):
        return "whole clip" if seconds is None else f"first {seconds:g} s"

    for model_name in args.models:
        print(f"\n{model_name}")
        model = duration_eval.load(model_name)
        for seconds in durations:
            prints = student_prints(model, audio, seconds)
            save_prints(model_name, "vox1o" if seconds is None else f"vox1o_{seconds:g}s", prints)
            report(name(seconds), prints, a, b, same)
    if args.teacher:
        print("\nteacher ECAPA-TDNN (~20M parameters)")
        teacher = spk.Teacher()
        for seconds in durations:
            prints = teacher_prints(teacher, audio, seconds)
            save_prints("teacher", "vox1o" if seconds is None else f"vox1o_{seconds:g}s", prints)
            report(name(seconds), prints, a, b, same)
    return 0


def main_replay(args) -> int:
    same, clips, paths, a, b = replay_trials()
    original = load_audio(clips)
    replayed = load_audio(clips, paths)
    print(f"VoxCeleb1 replayed through the board: {len(clips)} clips, {len({c.split('/')[0] for c in clips})} "
          f"speakers, {len(same)} trials ({same.sum()} same speaker); scored over the whole clip")

    def run(label, embed):
        print(f"\n{label}")
        clean, board = embed(original), embed(replayed)
        save_prints(label.split()[0], "replay_original", clean)
        save_prints(label.split()[0], "replay_board", board)
        report("original files", clean, a, b, same)
        report("one original, one through board", clean, a, b, same, board)
        report("both through board", board, a, b, same)

    for model_name in args.models:
        model = duration_eval.load(model_name)
        run(model_name, lambda audio: student_prints(model, audio, None))
    if args.teacher:
        teacher = spk.Teacher()
        run("teacher (ECAPA-TDNN, ~20M parameters)", lambda audio: teacher_prints(teacher, audio, None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
