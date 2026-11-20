# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "soundfile", "torch<2.9"]
# ///
"""Confidence intervals, paired comparisons and threshold error rates.

Works from the voiceprints vox_eval.py saves under data/models/prints, so
nothing is re-run. Tests:
  vox1o, vox1o_1.5s, ...   VoxCeleb1-O (37,611 trials, 40 speakers)
  replay                   VoxCeleb1 clips replayed through the board, both
                           sides through the board (400 clips, 40 speakers)

Intervals come from a speaker-level bootstrap: the test speakers are
resampled with replacement and each drawn speaker brings all trials whose
first clip is theirs. Trials sharing a speaker are not independent, so
resampling single trials would make the intervals too narrow. Comparisons
are paired: both models are scored on the same resamples, and the interval
is for the difference.

Usage:
  uv run tools/train/stats.py tcs_long.int v3.int8
  uv run tools/train/stats.py tcs_long.int v3.int8 --compare --test replay
  uv run tools/train/stats.py tcs_long tcs_long_s1 tcs_long_s2 --seeds
  uv run tools/train/stats.py tcs_long.int --threshold 0.35 --test replay
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spk  # noqa: E402
import vox_eval  # noqa: E402


def trials(test: str):
    """(same, a, b, speaker of each trial's first clip, prints file suffix)."""
    if test == "replay":
        same, clips, _, a, b = vox_eval.replay_trials()
        suffix = "replay_board"
    else:
        same, clips, a, b = vox_eval.read_trials()
        suffix = test
    speakers = np.array([c.split("/")[0] for c in clips])
    return same, a, b, speakers[a], suffix


def scores(model: str, suffix: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    path = vox_eval.PRINTS / f"{model}__{suffix}.npy"
    if not path.exists():
        raise SystemExit(f"error: no saved voiceprints {path.name}; run vox_eval.py {model}" + (" --replay" if "replay" in suffix else ""))
    prints = np.load(path)
    return np.einsum("ij,ij->i", prints[a], prints[b])


def resamples(groups: np.ndarray, count: int, seed: int = 0) -> list[np.ndarray]:
    """Trial indices for each bootstrap resample of the speakers."""
    rng = np.random.default_rng(seed)
    names = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in names]
    return [np.concatenate([members[i] for i in rng.integers(0, len(names), len(names))]) for _ in range(count)]


def eer(s: np.ndarray, same: np.ndarray) -> float:
    return 100 * spk.equal_error_rate(s, same)


def interval(values: np.ndarray) -> str:
    low, high = np.percentile(values, [2.5, 97.5])
    return f"[{low:.2f}, {high:.2f}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--test", default="vox1o")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--compare", action="store_true", help="Paired difference of each model against the first")
    parser.add_argument("--seeds", action="store_true", help="The models are seeds of one recipe: mean and spread")
    parser.add_argument("--threshold", type=float, nargs="*", help="False accept / false reject at these scores")
    args = parser.parse_args()

    same, a, b, groups, suffix = trials(args.test)
    all_scores = {m: scores(m, suffix, a, b) for m in args.models}
    draws = resamples(groups, args.bootstrap)
    print(f"{args.test}: {len(same)} trials, {len(np.unique(groups))} speakers, "
          f"{args.bootstrap} speaker-level bootstrap resamples, 95% intervals")

    boot = {}
    for m, s in all_scores.items():
        boot[m] = np.array([eer(s[d], same[d]) for d in draws])
        print(f"  {m:<20} EER {eer(s, same):6.2f}%  {interval(boot[m])}")

    if args.compare and len(args.models) > 1:
        base = args.models[0]
        print(f"\nPaired difference, model minus {base} (negative = lower error than {base}):")
        for m in args.models[1:]:
            diff = boot[m] - boot[base]
            print(f"  {m:<20} {eer(all_scores[m], same) - eer(all_scores[base], same):+6.2f} points  "
                  f"{interval(diff)}   lower in {100 * np.mean(diff < 0):.0f}% of resamples")

    if args.seeds:
        values = np.array([eer(s, same) for s in all_scores.values()])
        print(f"\nAcross {len(values)} seeds: mean {values.mean():.2f}%, standard deviation {values.std(ddof=1):.2f}, "
              f"range {values.min():.2f}-{values.max():.2f}")

    for th in args.threshold or []:
        print(f"\nAt score threshold {th:g} (whole-clip voiceprints, not the board's per-window decisions):")
        for m, s in all_scores.items():
            fa = np.mean(s[~same] >= th) * 100
            fr = np.mean(s[same] < th) * 100
            fa_b = [np.mean(s[d][~same[d]] >= th) * 100 for d in draws]
            fr_b = [np.mean(s[d][same[d]] < th) * 100 for d in draws]
            print(f"  {m:<20} false accept {fa:5.2f}% {interval(np.array(fa_b))}   false reject {fr:5.2f}% {interval(np.array(fr_b))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
