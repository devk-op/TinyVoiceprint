# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "soundfile", "torch<2.9"]
# ///
"""The board's voice-match decisions, replayed on recordings.

Copies modules/voice_match/voice_match.cc step by step: log-mel frames (the
chip's features), the background level that follows quiet frames down and
creeps up 0.0005 per frame, a frame counting as speech 1.5 above it, a
decision every 0.4 s on the newest 1.5 s, made only if at least 40% of its
frames and 20% of its newest 0.3 s are speech, the voiceprint from the
chip's exact integer network (tcs_integer.py), the average of the last
three speech windows' voiceprints, and the threshold. Enrolment averages
the first 20 speech windows, as the board does.

Reports false reject (the enrolled speaker's speech decisions that do not
match) and false accept (other speakers' speech decisions that match), per
decision, at the board's threshold.

Tests:
  replay    VoxCeleb1 clips replayed through the board (40 speakers, 10
            clips each): each speaker enrolled from their first clip, then
            every other clip of every speaker run through the decisions.
  study     The volunteer recordings in recordings/study/<ID>/: each person
            enrolled from <room>_enroll.wav; their quiet, far, noise and
            soft recordings are target trials, everyone else's are
            impostor trials. The overlap recording (their voice over a
            talk video) is reported separately. --enrol-from free enrols from
            the first 20 speech windows of the quiet step instead.

Usage:
  uv run tools/train/board_sim.py replay
  uv run tools/train/board_sim.py study --threshold 0.30 0.35 0.40
"""

import argparse
import pathlib
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import duration_eval  # noqa: E402
import spk  # noqa: E402
import vox_eval  # noqa: E402

_MODEL = "tcs_noise.int"
_STEP = 40  # Frames between decisions (0.4 s).
_WINDOW = spk.CROP_FRAMES
_MARGIN, _SHARE, _TAIL, _TAIL_SHARE = 1.5, 0.4, 30, 0.2
_SMOOTH, _ENROLL = 3, 20
_STUDY = spk._ROOT / "recordings" / "study"
_STUDY_TARGET = ("quiet", "far", "noise", "soft")


# The firmware's model::kAverageVoice for the model in use (main() sets it);
# None turns the average-voice check off.
AVERAGE: np.ndarray | None = None


def speech_windows(pcm: np.ndarray) -> np.ndarray:
    """[n, frames, mels] features of the windows the board would score, in
    order, plus a marker (-1 row index) where a silent decision resets the
    smoothing. Returns (windows, reset_before) as arrays."""
    feats = spk.log_mel_torch(torch.from_numpy(pcm.astype(np.int16))[None])[0].numpy()
    level = feats.mean(axis=1)
    floor, active = np.float32(1e9), np.zeros(len(feats), bool)
    for i, value in enumerate(level):
        floor = min(floor + np.float32(0.0005), value)
        active[i] = value > floor + _MARGIN
    windows, reset = [], []
    pending_reset = True
    for end in range(_WINDOW, len(feats) + 1, _STEP):
        a = active[end - _WINDOW : end]
        if a.sum() < _SHARE * _WINDOW or a[-_TAIL:].sum() < _TAIL_SHARE * _TAIL:
            pending_reset = True
            continue
        windows.append(feats[end - _WINDOW : end])
        reset.append(pending_reset)
        pending_reset = False
    if not windows:
        return np.zeros((0, _WINDOW, feats.shape[1]), np.float32), np.zeros(0, bool)
    return np.stack(windows).astype(np.float32), np.array(reset)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.sqrt((v * v).sum(axis=-1, keepdims=True) + 1e-12)


class Recording:
    def __init__(self, model, pcm: np.ndarray):
        windows, self.reset = speech_windows(pcm)
        self.prints = model(windows) if len(windows) else np.zeros((0, 192), np.float32)

    def enrolment(self) -> np.ndarray | None:
        if len(self.prints) < 6:
            return None
        return unit(self.prints[:_ENROLL].sum(axis=0))

    def scores(self, voice: np.ndarray) -> np.ndarray:
        """Smoothed score at each speech decision. With AVERAGE set (as the
        firmware does), a decision closer to the average voice than to
        `voice` gets -1, so it is rejected at any threshold."""
        out, recent = [], []
        for p, reset in zip(self.prints, self.reset):
            if reset:
                recent = []
            recent = (recent + [p])[-_SMOOTH:]
            smoothed = unit(np.sum(recent, axis=0))
            score = float(smoothed @ voice)
            if AVERAGE is not None and score < float(smoothed @ AVERAGE):
                score = -1.0
            out.append(score)
        return np.array(out)


def report(label: str, target: np.ndarray, impostor: np.ndarray, thresholds) -> None:
    print(f"\n{label}: {len(target)} target and {len(impostor)} impostor speech decisions")
    for th in thresholds:
        fr = 100 * np.mean(target < th) if len(target) else float("nan")
        fa = 100 * np.mean(impostor >= th) if len(impostor) else float("nan")
        print(f"  threshold {th:.2f}   false reject {fr:5.1f}%   false accept {fa:5.2f}%")
    if len(target) and len(impostor):
        both = np.concatenate([target, impostor])
        same = np.concatenate([np.ones(len(target), bool), np.zeros(len(impostor), bool)])
        print(f"  equal error rate over decisions {100 * spk.equal_error_rate(both, same):.2f}%")


def run_replay(model, thresholds) -> None:
    _, clips, paths, _, _ = vox_eval.replay_trials()
    speakers = np.array([c.split("/")[0] for c in clips])
    audio = vox_eval.load_audio(clips, paths)
    recordings = [Recording(model, pcm) for pcm in audio]
    target, impostor, skipped = [], [], 0
    for speaker in np.unique(speakers):
        own = np.flatnonzero(speakers == speaker)
        voice = recordings[own[0]].enrolment()
        if voice is None:
            skipped += 1
            continue
        for i, rec in enumerate(recordings):
            if i == own[0]:
                continue
            (target if speakers[i] == speaker else impostor).append(rec.scores(voice))
    if skipped:
        print(f"  {skipped} speakers skipped: fewer than 6 speech windows in their enrolment clip")
    report(f"Replayed VoxCeleb1, {len(np.unique(speakers)) - skipped} speakers", np.concatenate(target), np.concatenate(impostor), thresholds)


def run_study(model, thresholds, enrol_from: str = "reading") -> None:
    people = sorted(p for p in _STUDY.iterdir() if p.is_dir()) if _STUDY.exists() else []
    if not people:
        raise SystemExit(f"error: no recordings under {_STUDY}")
    if len(people) < 2:
        print("Only one person recorded: false reject only; false accept needs a second person.")
    recs: dict[tuple[str, str], Recording] = {}
    for person in people:
        for wav in sorted(person.glob("*.wav")):
            pcm, rate = sf.read(str(wav), dtype="int16")
            assert rate == 16000, wav
            step = wav.stem.split("_", 1)[1]
            recs[(person.name, step)] = Recording(model, pcm)
    target, impostor, overlap = [], [], []
    per_step = {s: [] for s in _STUDY_TARGET}
    per_pair: dict[tuple[str, str], list[np.ndarray]] = {}
    for person in people:
        # "reading": the enrol step (a paragraph read aloud). "free": the first
        # 20 speech windows of the quiet step (free talk), which are then left
        # out of that step's test decisions.
        enrol = recs.get((person.name, "enroll" if enrol_from == "reading" else "quiet"))
        voice = enrol.enrolment() if enrol else None
        if voice is None:
            print(f"  {person.name}: no usable enrolment, skipped")
            continue
        for (who, step), rec in recs.items():
            if step == "enroll":
                continue
            s = rec.scores(voice)
            if enrol_from == "free" and who == person.name and step == "quiet":
                s = s[_ENROLL:]
            if who != person.name:
                impostor.append(s)
                per_pair.setdefault((person.name, who), []).append(s)
            elif step == "overlap":
                overlap.append(s)
            elif step in per_step:
                target.append(s)
                per_step[step].append(s)
    def joined(parts):
        return np.concatenate(parts) if parts else np.zeros(0)

    report(f"Volunteers, {len(people)} people, enrolled from {enrol_from}", joined(target), joined(impostor), thresholds)
    for step, scores in per_step.items():
        if scores:
            s = np.concatenate(scores)
            print(f"  {step:<8} {len(s):3d} decisions, false reject at each threshold: " + "  ".join(f"{100 * np.mean(s < th):.1f}%" for th in thresholds))
    if overlap:
        s = np.concatenate(overlap)
        print(f"  overlap  {len(s):3d} decisions (own voice over a video), matched at each threshold: " + "  ".join(f"{100 * np.mean(s >= th):.1f}%" for th in thresholds))
    for (enrolled, who), scores in sorted(per_pair.items()):
        s = np.concatenate(scores)
        print(f"  {who} against {enrolled}'s voiceprint, {len(s):3d} decisions, false accept at each threshold: " + "  ".join(f"{100 * np.mean(s >= th):.1f}%" for th in thresholds))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", choices=["replay", "study"])
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.25, 0.30, 0.35, 0.40, 0.45])
    parser.add_argument("--enrol-from", choices=["reading", "free"], default="reading",
                        help="study: enrol from the read paragraph, or from the start of the free-talk quiet step")
    parser.add_argument("--no-average-check", action="store_true",
                        help="Score as the firmware did before the average-voice check")
    args = parser.parse_args()
    model = duration_eval.load(args.model)
    global AVERAGE
    if not args.no_average_check:
        import tcs_integer

        AVERAGE = tcs_integer.average_voice(model)
    with torch.no_grad():
        if args.test == "replay":
            run_replay(model, args.threshold)
        else:
            run_study(model, args.threshold, args.enrol_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())
