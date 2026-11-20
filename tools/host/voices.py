# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy",
#   "scikit-learn",
#   "scipy",
#   "silero-vad",
#   "soundfile",
#   "speechbrain>=1.0",
#   # speechbrain 1.0 still calls torchaudio APIs removed in 2.9.
#   "torch<2.9",
#   "torchaudio<2.9",
# ]
# ///
"""Host-side voice analysis for recordings from the board's USB mic.

Stage 1 of the project: everything here runs on the Mac, on recorded files.

  enroll    Learn a person's voice from a clip, so they can be named later.
  diarize   Work out who spoke when (and name them, if enrolled).
  separate  Split two overlapping voices into separate tracks, and
            optionally remix them with a different volume per voice.
  focus     Make one person louder and everyone else quieter, using the
            who-spoke-when timeline. Any number of speakers, full audio
            quality; it can't untangle moments where people talk at once.

Pipeline pieces (all open models, downloaded on first use):
  - Silero VAD            finds where there is speech at all
  - ECAPA-TDNN            turns a short clip into a 192-number "voiceprint"
    (SpeechBrain)         where clips of the same person land close together
  - SepFormer             neural network that un-mixes two voices; works at
    (SpeechBrain)         8 kHz, so separated tracks are telephone quality

Usage:
  uv run tools/host/voices.py enroll alice recordings/alice.wav
  uv run tools/host/voices.py diarize recordings/meeting.wav
  uv run tools/host/voices.py separate recordings/two_people.wav --gain alice=+6 --gain 2=-20
  uv run tools/host/voices.py focus recordings/meeting.wav --on alice -n 3
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio

_RATE = 16000
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VOICES_DIR = _ROOT / "recordings" / "voices"
_MODEL_CACHE = pathlib.Path.home() / ".cache" / "device_ml" / "models"

# SepFormer doesn't run on Apple's MPS backend, and the CPU is fast enough for
# clips, so keep every model on the CPU.
_RUN_OPTS = {"device": "cpu"}

# The separation model works on 8 kHz audio. On a two-voice test clip it
# scored ~21 dB SI-SDR against ~1 dB for the 16 kHz sepformer-whamr16k, so
# telephone-band output is the better trade for now.
_SEPARATION_RATE = 8000

# Voiceprint windows: long enough to characterise a voice, short enough to
# catch speaker changes.
_WINDOW_S = 1.5
_HOP_S = 0.5
_MIN_SEGMENT_S = 0.3
# Shortest window whose voiceprint is trusted to tell speakers apart.
_MIN_RELIABLE_S = 1.0

# Cosine distance (1 - similarity) above which two groups of windows are
# treated as different people when the speaker count isn't given.
_CLUSTER_THRESHOLD = 0.6
# Minimum cosine similarity to an enrolled voiceprint to use that name.
_IDENTIFY_THRESHOLD = 0.35

# `focus`: extra time kept around each focused turn, and the fade length
# when the volume changes.
_FOCUS_PAD_S = 0.15
_FOCUS_RAMP_S = 0.03

# Level every recording is normalised to before analysis.
_TARGET_PEAK = 0.5

# The board's mic picks up low-frequency rumble (desk vibration, fans) that
# carries no speech information.
_HIGHPASS_HZ = 80.0


def load_audio(path: pathlib.Path) -> torch.Tensor:
    """Returns mono float audio at 16 kHz, rumble filtered out, level normalised.

    Reads audio files directly; anything else (video files such as .mp4 or
    .mov, .m4a, ...) has its audio track extracted with ffmpeg.
    """
    if not path.is_file():
        raise SystemExit(f"error: {path} does not exist")
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(audio.mean(axis=1))
        if rate != _RATE:
            wav = torchaudio.functional.resample(wav, rate, _RATE)
    except sf.LibsndfileError:
        wav = _decode_with_ffmpeg(path)
    wav = torchaudio.functional.highpass_biquad(wav, _RATE, _HIGHPASS_HZ)
    # The models were trained on speech at normal recording levels, and the
    # board's mic records quietly; without this the speech detector misses
    # soft or distant talkers. The 99.9th percentile ignores stray clicks.
    peak = torch.quantile(wav.abs(), 0.999) if len(wav) else torch.tensor(0.0)
    if peak > 0:
        wav = (wav * (_TARGET_PEAK / peak)).clamp(-1.0, 1.0)
    return wav


def _decode_with_ffmpeg(path: pathlib.Path) -> torch.Tensor:
    if shutil.which("ffmpeg") is None:
        raise SystemExit(f"error: reading {path.suffix} files needs ffmpeg (brew install ffmpeg)")
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(_RATE), "-f", "f32le", "-"],
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        raise SystemExit(f"error: ffmpeg found no audio in {path}: {result.stderr.decode().strip()}")
    return torch.from_numpy(np.frombuffer(result.stdout, dtype=np.float32).copy())


class Voiceprints:
    """ECAPA-TDNN speaker embeddings, unit length so dot product = cosine."""

    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(_MODEL_CACHE / "spkrec-ecapa-voxceleb"),
            run_opts=_RUN_OPTS,
        )

    def embed(self, clips: list[torch.Tensor]) -> np.ndarray:
        """One voiceprint per clip. Clips may differ in length."""
        # Batch equal-length clips together; the rest go one at a time.
        by_length: dict[int, list[int]] = {}
        for index, clip in enumerate(clips):
            by_length.setdefault(len(clip), []).append(index)
        result = [None] * len(clips)
        with torch.no_grad():
            for indices in by_length.values():
                batch = torch.stack([clips[i] for i in indices])
                vectors = self._model.encode_batch(batch).squeeze(1)
                vectors = torch.nn.functional.normalize(vectors, dim=1)
                for i, vector in zip(indices, vectors):
                    result[i] = vector.numpy()
        return np.stack(result)

    def embed_whole(self, wav: torch.Tensor) -> np.ndarray:
        """Average voiceprint over all the speech in `wav`."""
        windows = [wav[a:b] for a, b in speech_windows(wav)]
        if not windows:
            raise SystemExit("error: no speech found in the clip")
        mean = self.embed(windows).mean(axis=0)
        return mean / np.linalg.norm(mean)


def speech_segments(wav: torch.Tensor) -> list[tuple[int, int]]:
    """(start, end) sample ranges that contain speech, via Silero VAD."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    stamps = get_speech_timestamps(
        wav, load_silero_vad(), sampling_rate=_RATE, min_silence_duration_ms=200
    )
    return [(s["start"], s["end"]) for s in stamps]


def speech_windows(wav: torch.Tensor) -> list[tuple[int, int]]:
    """Overlapping fixed-size windows covering every speech segment."""
    window, hop = int(_WINDOW_S * _RATE), int(_HOP_S * _RATE)
    windows = []
    for start, end in speech_segments(wav):
        if end - start < int(_MIN_SEGMENT_S * _RATE):
            continue
        if end - start <= window:
            windows.append((start, end))
            continue
        position = start
        while position + window < end:
            windows.append((position, position + window))
            position += hop
        windows.append((end - window, end))  # Final window flush with the end.
    return windows


def load_enrolled() -> dict[str, np.ndarray]:
    return {path.stem: np.load(path) for path in sorted(_VOICES_DIR.glob("*.npy"))}


def name_speakers(centroids: np.ndarray, enrolled: dict[str, np.ndarray]) -> list[str]:
    """Names each centroid after its best enrolled match, else 'speaker N'.

    Each enrolled name is used at most once, best matches first.
    """
    names = [f"speaker {i + 1}" for i in range(len(centroids))]
    if not enrolled:
        return names
    enrolled_names = list(enrolled)
    similarity = centroids @ np.stack([enrolled[n] for n in enrolled_names]).T
    taken_rows, taken_cols = set(), set()
    for flat in np.argsort(similarity, axis=None)[::-1]:
        row, col = np.unravel_index(flat, similarity.shape)
        if similarity[row, col] < _IDENTIFY_THRESHOLD:
            break
        if row in taken_rows or col in taken_cols:
            continue
        names[row] = enrolled_names[col]
        taken_rows.add(row)
        taken_cols.add(col)
    return names


def cmd_enroll(args: argparse.Namespace) -> int:
    voiceprints = Voiceprints()
    vectors = [voiceprints.embed_whole(load_audio(path)) for path in args.wav]
    mean = np.mean(vectors, axis=0)
    _VOICES_DIR.mkdir(parents=True, exist_ok=True)
    out = _VOICES_DIR / f"{args.name}.npy"
    np.save(out, mean / np.linalg.norm(mean))
    print(f"Enrolled {args.name!r} from {len(args.wav)} clip(s) -> {out.relative_to(_ROOT)}")
    return 0


def diarize(wav: torch.Tensor, voiceprints: Voiceprints, num_speakers: int | None, threshold: float):
    """Returns ([(start_s, end_s, cluster)], centroids)."""
    from sklearn.cluster import AgglomerativeClustering

    windows = speech_windows(wav)
    if not windows:
        return [], np.zeros((0, 192), dtype=np.float32)
    vectors = voiceprints.embed([wav[a:b] for a, b in windows])

    # Only windows long enough for a trustworthy voiceprint get to define the
    # speakers; shorter fragments then join whichever speaker they're closest
    # to, so that a half-second "yeah" can't become a person of its own.
    reliable = np.array([b - a >= int(_MIN_RELIABLE_S * _RATE) for a, b in windows])
    if reliable.sum() < 2:
        labels = np.zeros(len(windows), dtype=int)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=min(num_speakers, int(reliable.sum())) if num_speakers else None,
            distance_threshold=None if num_speakers else threshold,
            metric="cosine",
            linkage="average",
        )
        reliable_labels = clustering.fit_predict(vectors[reliable])
        anchors = np.stack(
            [vectors[reliable][reliable_labels == k].mean(axis=0) for k in range(reliable_labels.max() + 1)]
        )
        labels = (vectors @ anchors.T).argmax(axis=1)
        labels[reliable] = reliable_labels

    # Renumber clusters in order of first appearance.
    order = {label: i for i, label in enumerate(dict.fromkeys(labels))}
    labels = np.array([order[label] for label in labels])
    centroids = np.stack([vectors[labels == k].mean(axis=0) for k in range(len(order))])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    # Windows overlap, so let each 100 ms frame take the majority label of
    # the windows covering it, then merge runs of frames into turns.
    frame = int(0.1 * _RATE)
    votes = np.zeros((len(wav) // frame + 1, len(order)), dtype=int)
    for (start, end), label in zip(windows, labels):
        votes[start // frame : max(start // frame + 1, end // frame), label] += 1
    frame_label = np.where(votes.sum(axis=1) > 0, votes.argmax(axis=1), -1)

    turns = []
    for start, end in speech_segments(wav):
        first, last = start // frame, max(start // frame + 1, end // frame)
        run_start = first
        for i in range(first, last + 1):
            if i == last or frame_label[i] != frame_label[run_start]:
                if frame_label[run_start] >= 0:
                    turns.append(
                        (
                            max(start, run_start * frame) / _RATE,
                            min(end, i * frame) / _RATE,
                            int(frame_label[run_start]),
                        )
                    )
                run_start = i
    return turns, centroids


def score_against_truth(turns, names: list[str], truth_path: pathlib.Path) -> float:
    """Fraction of ground-truth speech time attributed to the right person.

    Predicted names are matched to true names by best overlap, so this works
    with or without enrollment.
    """
    from scipy.optimize import linear_sum_assignment

    truth = json.loads(truth_path.read_text())
    true_names = sorted({t["speaker"] for t in truth})
    step = 0.01
    total = int(max(t["end"] for t in truth) / step) + 1
    true_frames = np.full(total, -1)
    for t in truth:
        true_frames[int(t["start"] / step) : int(t["end"] / step)] = true_names.index(t["speaker"])
    predicted = np.full(total, -1)
    for start, end, cluster in turns:
        predicted[int(start / step) : min(total, int(end / step))] = cluster

    overlap = np.zeros((len(names), len(true_names)))
    for p in range(len(names)):
        for t in range(len(true_names)):
            overlap[p, t] = np.sum((predicted == p) & (true_frames == t))
    rows, cols = linear_sum_assignment(-overlap)
    return overlap[rows, cols].sum() / np.sum(true_frames >= 0)


def cmd_diarize(args: argparse.Namespace) -> int:
    wav = load_audio(args.wav)
    voiceprints = Voiceprints()
    turns, centroids = diarize(wav, voiceprints, args.speakers, args.threshold)
    if not turns:
        print("No speech found.")
        return 0
    names = name_speakers(centroids, load_enrolled())

    print(f"\n{args.wav.name}: {len(wav) / _RATE:.1f}s, {len(names)} speaker(s)\n")
    for start, end, cluster in turns:
        print(f"  {start:7.2f} - {end:7.2f}  {names[cluster]}")
    print()
    for cluster, name in enumerate(names):
        seconds = sum(end - start for start, end, c in turns if c == cluster)
        print(f"  {name:<12} {seconds:6.1f}s")

    out = args.wav.with_suffix(".diarization.json")
    out.write_text(
        json.dumps(
            [{"speaker": names[c], "start": round(s, 2), "end": round(e, 2)} for s, e, c in turns],
            indent=2,
        )
        + "\n"
    )
    print(f"\nSaved {out}")

    if args.truth:
        accuracy = score_against_truth(turns, names, args.truth)
        print(f"Correctly attributed speech time vs {args.truth.name}: {100 * accuracy:.1f}%")
    return 0


def cmd_focus(args: argparse.Namespace) -> int:
    wav = load_audio(args.wav)
    turns, centroids = diarize(wav, Voiceprints(), args.speakers, args.threshold)
    if not turns:
        print("No speech found.")
        return 0
    names = name_speakers(centroids, load_enrolled())

    wanted = {name.lower() for name in args.on}
    targets = {i for i, name in enumerate(names) if name.lower() in wanted}
    missing = wanted - {names[i].lower() for i in targets}
    if missing:
        raise SystemExit(
            f"error: {', '.join(sorted(missing))} not found in this recording. "
            f"Speakers found: {', '.join(names)}"
        )

    # Per-sample gain: boosted during the chosen speakers' turns, turned down
    # everywhere else (other speakers and the background between turns).
    boost = 10 ** (args.boost / 20)
    cut = 10 ** (args.others / 20)
    gain = torch.full_like(wav, cut)
    pad = int(_FOCUS_PAD_S * _RATE)  # VAD trims word edges slightly.
    for start, end, cluster in turns:
        if cluster in targets:
            gain[max(0, int(start * _RATE) - pad) : int(end * _RATE) + pad] = boost
    # Other speakers win the padding where turns touch.
    for start, end, cluster in turns:
        if cluster not in targets:
            gain[int(start * _RATE) : int(end * _RATE)] = cut
    # Ramp between levels instead of stepping, which would click.
    ramp = int(_FOCUS_RAMP_S * _RATE)
    kernel = torch.ones(1, 1, ramp) / ramp
    gain = torch.nn.functional.conv1d(
        torch.nn.functional.pad(gain.view(1, 1, -1), (ramp // 2, ramp - ramp // 2 - 1), mode="replicate"),
        kernel,
    ).view(-1)

    focused = (wav * gain).clamp(-1.0, 1.0)
    label = "+".join(sorted(names[i].replace(" ", "") for i in targets))
    out = args.wav.with_name(f"{args.wav.stem}.focus-{label}.wav")
    sf.write(out, focused.numpy(), _RATE, subtype="PCM_16")

    print(f"\n{args.wav.name}: {len(wav) / _RATE:.1f}s, {len(names)} speaker(s)\n")
    for cluster, name in enumerate(names):
        mask = torch.zeros(len(wav), dtype=torch.bool)
        for start, end, c in turns:
            if c == cluster:
                mask[int(start * _RATE) : int(end * _RATE)] = True
        before = wav[mask].square().mean().sqrt()
        after = focused[mask].square().mean().sqrt()
        change = 20 * torch.log10(after / before).item()
        role = "FOCUS" if cluster in targets else "turned down"
        print(f"  {name:<12} {mask.sum().item() / _RATE:5.1f}s  {change:+6.1f} dB  ({role})")
    print(f"\nSaved {out}")
    return 0


def parse_gains(specs: list[str]) -> dict[str, float]:
    gains = {}
    for spec in specs:
        key, _, value = spec.partition("=")
        if not value:
            raise SystemExit(f"error: --gain expects NAME=dB or INDEX=dB, got {spec!r}")
        gains[key] = float(value)
    return gains


def cmd_separate(args: argparse.Namespace) -> int:
    from speechbrain.inference.separation import SepformerSeparation

    wav = load_audio(args.wav)
    separator = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-libri2mix",
        savedir=str(_MODEL_CACHE / "sepformer-libri2mix"),
        run_opts=_RUN_OPTS,
    )
    narrowband = torchaudio.functional.resample(wav, _RATE, _SEPARATION_RATE)
    with torch.no_grad():
        sources = separator.separate_batch(narrowband.unsqueeze(0))[0].T  # [n_src, time]
    sources = torchaudio.functional.resample(sources, _SEPARATION_RATE, _RATE)
    # The model's output scale is arbitrary. Rescale the tracks so that they
    # add back up to the original mix (least squares); a 0 dB remix then
    # sounds like the input, and --gain values mean what they say.
    length = min(sources.shape[1], len(wav))
    sources, wav = sources[:, :length], wav[:length]
    scales = torch.linalg.lstsq(sources.T, wav.unsqueeze(1)).solution
    sources = sources * scales

    voiceprints = Voiceprints()
    centroids = np.stack([voiceprints.embed_whole(source) for source in sources])
    names = name_speakers(centroids, load_enrolled())

    gains = parse_gains(args.gain)
    remix = torch.zeros_like(sources[0])
    print()
    for index, (source, name) in enumerate(zip(sources, names), start=1):
        out = args.wav.with_name(f"{args.wav.stem}.{name.replace(' ', '')}.wav")
        sf.write(out, source.numpy(), _RATE, subtype="PCM_16")
        gain_db = gains.get(name, gains.get(str(index), 0.0))
        remix += source * 10 ** (gain_db / 20)
        print(f"  track {index}: {name:<12} {gain_db:+5.1f} dB  -> {out.name}")

    if gains:
        out = args.wav.with_name(f"{args.wav.stem}.remix.wav")
        sf.write(out, remix.clamp(-1, 1).numpy(), _RATE, subtype="PCM_16")
        print(f"\n  remix -> {out.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    enroll = commands.add_parser("enroll", help="Learn a person's voice")
    enroll.add_argument("name")
    enroll.add_argument("wav", type=pathlib.Path, nargs="+", help="Clip(s) of only that person speaking")
    enroll.set_defaults(run=cmd_enroll)

    diarize_parser = commands.add_parser("diarize", help="Who spoke when")
    diarize_parser.add_argument("wav", type=pathlib.Path)
    diarize_parser.add_argument("-n", "--speakers", type=int, help="Number of speakers, if known (more reliable)")
    diarize_parser.add_argument("--threshold", type=float, default=_CLUSTER_THRESHOLD, help="Clustering distance threshold when --speakers isn't given")
    diarize_parser.add_argument("--truth", type=pathlib.Path, help="Ground-truth JSON to score against")
    diarize_parser.set_defaults(run=cmd_diarize)

    separate = commands.add_parser("separate", help="Split two overlapping voices")
    separate.add_argument("wav", type=pathlib.Path)
    separate.add_argument("--gain", action="append", default=[], metavar="WHO=dB", help="Volume change for a voice in the remix, by enrolled name or track number; repeatable")
    separate.set_defaults(run=cmd_separate)

    focus = commands.add_parser("focus", help="Make one person louder and everyone else quieter")
    focus.add_argument("wav", type=pathlib.Path)
    focus.add_argument("--on", action="append", required=True, metavar="NAME", help="Who to focus on: an enrolled name, or e.g. 'speaker 2'; repeatable")
    focus.add_argument("--boost", type=float, default=6.0, help="Volume change in dB for the focused speaker(s) (default +6)")
    focus.add_argument("--others", type=float, default=-20.0, help="Volume change in dB for everything else (default -20)")
    focus.add_argument("-n", "--speakers", type=int, help="Number of speakers, if known (more reliable)")
    focus.add_argument("--threshold", type=float, default=_CLUSTER_THRESHOLD, help="Clustering distance threshold when --speakers isn't given")
    focus.set_defaults(run=cmd_focus)

    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
