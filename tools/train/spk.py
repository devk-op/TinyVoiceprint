# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "numpy",
#   "scipy",
#   "soundfile",
#   "speechbrain>=1.0",
#   # speechbrain 1.0 still calls torchaudio APIs removed in 2.9.
#   "torch<2.9",
#   "torchaudio<2.9",
# ]
# ///
"""Trains the tiny on-device speaker encoder by distillation.

A small CNN (the "student", ~100k parameters, sized for a Cortex-M4) learns to
reproduce the voiceprints of a large pre-trained model (the "teacher",
SpeechBrain's ECAPA-TDNN, ~20M parameters) from 1.5 s of speech. The student's
input is the log-mel features the firmware computes (modules/features), as
defined by tools/host/log_mel_reference.py.

Data: LibriSpeech (CC BY 4.0), unpacked under data/librispeech/LibriSpeech.

  selfcheck   Confirm the fast feature code here matches the reference.
  prepare     Per split: cache log-mel features and teacher voiceprints.
  cv-select   Choose Common Voice clips (also CC0) and list them to unpack.
  prepare-cv  The same caching for the chosen Common Voice clips.
  prepare-vox The same caching for VoxCeleb1 dev or VoxCeleb2 dev (CC BY 4.0).
  train       Train a student on a prepared split.
  eval        Speaker-verification error (EER) of a student and the teacher.

Usage:
  uv run tools/train/spk.py selfcheck
  uv run tools/train/spk.py prepare train-clean-100
  uv run tools/train/spk.py prepare train-clean-100 --speed 0.9
  uv run tools/train/spk.py prepare test-clean
  uv run tools/train/spk.py train --split train-clean-100 train-clean-100@0.9 --augment --name v1
  uv run tools/train/spk.py eval --name v0 --split test-clean
"""

import argparse
import json
import multiprocessing
import pathlib
import subprocess
import sys
import time
import warnings

import numpy as np
import soundfile as sf
import torch
from torch import nn

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tools" / "host"))
import log_mel_reference as ref  # noqa: E402

_LIBRISPEECH = _ROOT / "data" / "librispeech" / "LibriSpeech"
# Mozilla Common Voice (CC0), unpacked language folder: clips/ and *.tsv.
_COMMONVOICE = _ROOT / "data" / "commonvoice" / "cv-corpus-27.0-2026-09-11" / "en"
# VoxCeleb1 dev (CC BY 4.0): wav/<speaker>/<video>/<clip>.wav. Its speakers
# are disjoint from the VoxCeleb1-O test speakers used by vox_eval.py.
_VOXCELEB_DEV = _ROOT / "data" / "voxceleb" / "dev" / "wav"
_VOX_PREFIX = "vox1-dev/"
# VoxCeleb2 dev (CC BY 4.0): .m4a (AAC) clips under <speaker>/<video>/, in
# whatever folders the zip unpacks to. Disjoint from VoxCeleb1 entirely.
# Decoded with ffmpeg, which libsndfile can't replace for AAC.
_VOXCELEB2_DEV = _ROOT / "data" / "voxceleb" / "vox2_dev"
_VOX2_PREFIX = "vox2-dev/"
_CACHE = _ROOT / "data" / "features"
_MODELS = _ROOT / "data" / "models"
_TEACHER_CACHE = pathlib.Path.home() / ".cache" / "device_ml" / "models" / "spkrec-ecapa-voxceleb"

TEACHER_DIM = 192
# Frames of context the student sees: 150 frames = 1.5 s. Short enough to
# react within a conversational turn, long enough to characterise a voice.
CROP_FRAMES = 150
CROP_SAMPLES = ref.FRAME_LENGTH + ref.HOP * (CROP_FRAMES - 1)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- features


def log_mel_torch(pcm: torch.Tensor) -> torch.Tensor:
    """Batched log-mel, same definition as ref.log_mel.

    pcm: [batch, samples] holding int16-range values. Returns
    [batch, frames, NUM_MELS].
    """
    frames = pcm.to(torch.float32).unfold(1, ref.FRAME_LENGTH, ref.HOP) / 32768.0
    n = torch.arange(ref.FRAME_LENGTH, dtype=torch.float64)
    window = (0.5 - 0.5 * torch.cos(2.0 * torch.pi * n / ref.FRAME_LENGTH)).to(torch.float32)
    power = torch.fft.rfft(frames * window, n=ref.FFT_SIZE).abs() ** 2
    bank = torch.from_numpy(ref.mel_filterbank().T.astype(np.float32))
    return torch.log(power @ bank + ref.LOG_FLOOR)


def cmd_selfcheck(_: argparse.Namespace) -> int:
    rng = np.random.default_rng(0)
    worst = 0.0
    for pcm in (ref.golden_pcm(8000), (rng.standard_normal(8000) * 3000).astype(np.int16)):
        expected = ref.log_mel(pcm)
        actual = log_mel_torch(torch.from_numpy(pcm).unsqueeze(0))[0].numpy()
        worst = max(worst, float(np.abs(actual - expected).max()))
    ok = worst < 1e-3
    print(f"torch features vs reference: max abs error {worst:.2e} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ----------------------------------------------------------------- prepare


def _speed_ratio(speed: float) -> tuple[int, int]:
    """(up, down) resampling ratio that plays audio `speed` times faster."""
    down = round(speed * 10)
    assert abs(down / 10 - speed) < 1e-9 and down > 0, "speed must be a multiple of 0.1"
    return 10, down


def _load_features(job: tuple[str, float]):
    """Worker: decode one utterance; return (log-mel float16, int16 audio).

    A speed other than 1 resamples the audio so it plays faster or slower,
    which shifts pitch and tempo together. The result sounds like a different
    person, so each speed is treated as a new set of speakers.
    """
    path, speed = job
    pcm, rate = sf.read(path, dtype="int16")
    assert rate == ref.SAMPLE_RATE and pcm.ndim == 1, path
    if speed != 1.0:
        from scipy.signal import resample_poly

        up, down = _speed_ratio(speed)
        pcm = np.clip(np.rint(resample_poly(pcm.astype(np.float64), up, down)), -32768, 32767).astype(np.int16)
    return ref.log_mel(pcm).astype(np.float16), pcm


def _decode_ffmpeg(path: pathlib.Path) -> np.ndarray:
    """16 kHz mono int16 audio from any file ffmpeg can read."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
               "-f", "s16le", "-ac", "1", "-ar", str(ref.SAMPLE_RATE), "-"]
    out = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i2").copy()


def read_pcm(path: str) -> np.ndarray:
    """16 kHz mono int16 audio for a path as stored in a split's index:
    a LibriSpeech .flac relative to its root, or a Common Voice .mp3 name.
    """
    if path.startswith(_VOX2_PREFIX):
        return _decode_ffmpeg(_VOXCELEB2_DEV / path[len(_VOX2_PREFIX) :])
    if path.startswith(_VOX_PREFIX):
        pcm, rate = sf.read(str(_VOXCELEB_DEV / path[len(_VOX_PREFIX) :]), dtype="int16")
        assert rate == ref.SAMPLE_RATE and pcm.ndim == 1, path
        return pcm
    if path.endswith(".mp3"):
        audio, rate = sf.read(str(_COMMONVOICE / "clips" / path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != ref.SAMPLE_RATE:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(rate, ref.SAMPLE_RATE)
            audio = resample_poly(audio, ref.SAMPLE_RATE // g, rate // g)
        return np.clip(np.rint(audio * 32768.0), -32768, 32767).astype(np.int16)
    pcm, rate = sf.read(str(_LIBRISPEECH / path), dtype="int16")
    assert rate == ref.SAMPLE_RATE and pcm.ndim == 1, path
    return pcm


def _load_cv_features(path: str):
    """Worker: decode one Common Voice clip; (log-mel float16, int16 audio),
    or None if the file can't be decoded or is too short."""
    try:
        pcm = read_pcm(path)
    except Exception:  # A few clips in a set this size are corrupt.
        return None
    if len(pcm) < CROP_SAMPLES:
        return None
    return ref.log_mel(pcm).astype(np.float16), pcm


def _split_name(split: str, speed: float) -> str:
    return split if speed == 1.0 else f"{split}@{speed:g}"


def _split_dir(split: str) -> pathlib.Path:
    return _CACHE / split


class Teacher:
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device()
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(_TEACHER_CACHE),
            run_opts={"device": str(self.device)},
        )

    @torch.no_grad()
    def embed(self, clips: list[np.ndarray]) -> np.ndarray:
        """int16 clips of any lengths -> [n, TEACHER_DIM] unit vectors."""
        longest = max(len(c) for c in clips)
        batch = torch.zeros(len(clips), longest)
        for i, clip in enumerate(clips):
            batch[i, : len(clip)] = torch.from_numpy(clip.astype(np.float32) / 32768.0)
        lengths = torch.tensor([len(c) / longest for c in clips])
        out = self._model.encode_batch(batch.to(self.device), lengths.to(self.device)).squeeze(1)
        return nn.functional.normalize(out, dim=1).cpu().numpy()


def cmd_prepare(args: argparse.Namespace) -> int:
    source = _LIBRISPEECH / args.split
    if not source.is_dir():
        raise SystemExit(f"error: {source} not found; download and unpack LibriSpeech {args.split} first")
    name = _split_name(args.split, args.speed)
    out = _split_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    up, down = _speed_ratio(args.speed)

    # Sorted by size so that teacher batches hold utterances of similar
    # length and little time is spent on padding.
    paths = sorted(source.glob("*/*/*.flac"), key=lambda p: p.stat().st_size)
    speakers = sorted({p.parts[-3] for p in paths})
    speaker_index = {name: i for i, name in enumerate(speakers)}
    # resample_poly returns ceil(n * up / down) samples.
    samples = np.array([-(-sf.info(p).frames * up // down) for p in paths], dtype=np.int64)
    lengths = (samples - ref.FRAME_LENGTH) // ref.HOP + 1
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    print(f"{name}: {len(paths)} utterances, {len(speakers)} speakers, {offsets[-1] / 100 / 3600:.1f} h")

    features = np.lib.format.open_memmap(
        out / "features.npy", mode="w+", dtype=np.float16, shape=(int(offsets[-1]), ref.NUM_MELS)
    )
    teacher = Teacher()
    teacher_out = np.zeros((len(paths), TEACHER_DIM), dtype=np.float32)

    start = time.time()
    pending: list[tuple[int, np.ndarray]] = []

    def flush():
        if pending:
            teacher_out[[i for i, _ in pending]] = teacher.embed([clip for _, clip in pending])
            pending.clear()
            if teacher.device.type == "mps":
                torch.mps.empty_cache()

    with multiprocessing.Pool() as pool:
        jobs = ((str(path), args.speed) for path in paths)
        for i, (feats, pcm) in enumerate(pool.imap(_load_features, jobs, chunksize=8)):
            assert len(feats) == lengths[i], paths[i]
            features[offsets[i] : offsets[i + 1]] = feats
            # Batches are padded to their longest clip, so bound the padded
            # total rather than the clip count: long utterances would
            # otherwise exhaust GPU memory.
            longest = max(len(pcm), max((len(clip) for _, clip in pending), default=0))
            if pending and longest * (len(pending) + 1) > args.batch_seconds * ref.SAMPLE_RATE:
                flush()
            pending.append((i, pcm))
            if (i + 1) % 500 == 0 or i + 1 == len(paths):
                rate = offsets[i + 1] / 100 / (time.time() - start)
                print(f"  {i + 1}/{len(paths)}  ({rate:.0f}x real time)", flush=True)
    flush()
    features.flush()

    np.savez(
        out / "index.npz",
        offsets=offsets,
        speaker=np.array([speaker_index[p.parts[-3]] for p in paths], dtype=np.int32),
        teacher=teacher_out,
        paths=np.array([str(p.relative_to(_LIBRISPEECH)) for p in paths]),
        speakers=np.array(speakers),
    )
    print(f"Wrote {out} in {time.time() - start:.0f}s")
    return 0


def _cv_selection(args: argparse.Namespace) -> dict[str, list[tuple[str, str, int]]]:
    """Picks Common Voice clips: {"cv-train" | "cv-test": [(clip, speaker, ms)]}.

    Keeps validated clips of 3-11 s with no down-votes, from speakers with at
    least --min-clips of them, at most --cap per speaker (so no speaker
    dominates, and a mislabelled speaker does limited harm). --test-speakers
    of those speakers are held out entirely as cv-test.
    """
    import collections
    import csv

    csv.field_size_limit(10**9)
    durations = {}
    with open(_COMMONVOICE / "clip_durations.tsv") as f:
        next(f)
        for line in f:
            clip, ms = line.rstrip("\n").split("\t")
            durations[clip] = int(ms)
    by_speaker = collections.defaultdict(list)
    with open(_COMMONVOICE / "validated.tsv", newline="") as f:
        rows = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(rows)
        speaker_col, path_col, down_col = (header.index(c) for c in ("client_id", "path", "down_votes"))
        for row in rows:
            ms = durations.get(row[path_col], 0)
            if 3000 <= ms <= 11000 and int(row[down_col] or 0) == 0:
                by_speaker[row[speaker_col]].append((row[path_col], ms))

    rng = np.random.default_rng(args.seed)
    speakers = sorted(s for s, clips in by_speaker.items() if len(clips) >= args.min_clips)
    held_out = set(rng.choice(speakers, size=args.test_speakers, replace=False).tolist())
    selection: dict[str, list[tuple[str, str, int]]] = {"cv-train": [], "cv-test": []}
    for speaker in speakers:
        clips = sorted(by_speaker[speaker])
        keep = rng.choice(len(clips), size=min(len(clips), args.cap), replace=False)
        split = "cv-test" if speaker in held_out else "cv-train"
        selection[split] += [(clips[i][0], speaker, clips[i][1]) for i in sorted(keep)]
    return selection


def cmd_cv_select(args: argparse.Namespace) -> int:
    """Writes the list of clips to unpack from the Common Voice archive."""
    selection = _cv_selection(args)
    prefix = f"{_COMMONVOICE.parent.name}/{_COMMONVOICE.name}/clips/"
    out = _COMMONVOICE / "selected_clips.txt"
    with open(out, "w") as f:
        for clips in selection.values():
            f.writelines(prefix + clip + "\n" for clip, _, _ in clips)
    for name, clips in selection.items():
        hours = sum(ms for _, _, ms in clips) / 3.6e6
        print(f"{name}: {len(clips)} clips, {len({s for _, s, _ in clips})} speakers, {hours:.0f} h")
    print(f"Wrote {out}\nUnpack with: tar -xzf ARCHIVE -T {out}   (run in {_COMMONVOICE.parent.parent})")
    return 0


def cmd_prepare_cv(args: argparse.Namespace) -> int:
    """Caches features and teacher voiceprints for cv-train or cv-test."""
    clips = _cv_selection(args)[args.split]
    clips = [c for c in clips if (_COMMONVOICE / "clips" / c[0]).is_file()]
    if not clips:
        raise SystemExit(f"error: no selected clips under {_COMMONVOICE / 'clips'}; run cv-select and unpack first")
    clips.sort(key=lambda c: c[2])  # By length, so teacher batches pad little.
    speakers = sorted({s for _, s, _ in clips})
    speaker_index = {name: i for i, name in enumerate(speakers)}
    out = _split_dir(args.split)
    out.mkdir(parents=True, exist_ok=True)

    # Exact lengths aren't known until each MP3 is decoded, so reserve an
    # upper bound and record the real offsets as clips arrive.
    reserve = sum(ms // 10 + 20 for _, _, ms in clips)
    print(f"{args.split}: {len(clips)} clips, {len(speakers)} speakers, about {reserve / 100 / 3600:.0f} h")
    features = np.lib.format.open_memmap(
        out / "features.npy", mode="w+", dtype=np.float16, shape=(reserve, ref.NUM_MELS)
    )
    teacher = Teacher()
    kept: list[int] = []  # Index into `clips` of each stored utterance.
    offsets = [0]
    embeddings: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    start = time.time()

    def flush():
        if pending:
            embeddings.append(teacher.embed(pending))
            pending.clear()
            if teacher.device.type == "mps":
                torch.mps.empty_cache()

    def decoded(pool):
        # Pool.imap runs ahead of the consumer without limit. MP3 clips
        # decode far faster than the teacher embeds them, so the backlog
        # filled the Mac's memory; hand the pool a bounded slice at a time.
        names = [clip for clip, _, _ in clips]
        for begin in range(0, len(names), 2000):
            yield from pool.imap(_load_cv_features, names[begin : begin + 2000], chunksize=16)

    with multiprocessing.Pool() as pool:
        for i, result in enumerate(decoded(pool)):
            if result is not None and offsets[-1] + len(result[0]) <= reserve:
                feats, pcm = result
                longest = max(len(pcm), max((len(clip) for clip in pending), default=0))
                if pending and longest * (len(pending) + 1) > args.batch_seconds * ref.SAMPLE_RATE:
                    flush()
                features[offsets[-1] : offsets[-1] + len(feats)] = feats
                offsets.append(offsets[-1] + len(feats))
                kept.append(i)
                pending.append(pcm)
            if (i + 1) % 2000 == 0 or i + 1 == len(clips):
                rate = offsets[-1] / 100 / (time.time() - start)
                print(f"  {i + 1}/{len(clips)}  kept {len(kept)}  ({rate:.0f}x real time)", flush=True)
    flush()
    features.flush()

    np.savez(
        out / "index.npz",
        offsets=np.array(offsets, dtype=np.int64),
        speaker=np.array([speaker_index[clips[i][1]] for i in kept], dtype=np.int32),
        teacher=np.concatenate(embeddings),
        paths=np.array([clips[i][0] for i in kept]),
        speakers=np.array(speakers),
    )
    print(f"Wrote {out}: {len(kept)} clips, {offsets[-1] / 100 / 3600:.1f} h, in {time.time() - start:.0f}s")
    return 0


def _load_vox_features(path: str):
    """Worker: decode one VoxCeleb clip; (log-mel float16, int16 audio)."""
    pcm = read_pcm(path)
    return ref.log_mel(pcm).astype(np.float16), pcm


def _samples(path: str) -> int:
    """Worker: decoded length of one clip in samples, or 0 if it won't decode."""
    try:
        return len(read_pcm(path))
    except (subprocess.CalledProcessError, OSError):
        return 0


def cmd_prepare_vox(args: argparse.Namespace) -> int:
    """Caches features and teacher voiceprints for VoxCeleb1 or VoxCeleb2 dev."""
    split = f"{args.dataset}-dev"
    if args.dataset == "vox1":
        root, prefix = _VOXCELEB_DEV, _VOX_PREFIX
        files = sorted(root.glob("id*/*/*.wav"))
    else:
        root, prefix = _VOXCELEB2_DEV, _VOX2_PREFIX
        files = sorted(root.rglob("*.m4a"))
    if not files:
        raise SystemExit(f"error: no clips under {root}; unpack the {args.dataset} dev zip there")
    test_speakers = {p.name for p in (_ROOT / "data" / "voxceleb" / "wav").glob("id*")}
    speakers = sorted({f.parts[-3] for f in files})
    overlap = test_speakers & set(speakers)
    if overlap:
        raise SystemExit(f"error: {len(overlap)} dev speakers are also test speakers, e.g. {sorted(overlap)[:3]}")
    speaker_index = {name: i for i, name in enumerate(speakers)}
    names = [prefix + str(f.relative_to(root)) for f in files]
    if args.dataset == "vox1":
        frames = np.array([sf.info(str(f)).frames for f in files], dtype=np.int64)
    else:
        # AAC has no cheap exact length, so decode everything once to measure.
        print(f"{split}: measuring {len(files)} clips", flush=True)
        with multiprocessing.Pool() as pool:
            frames = np.array(list(pool.imap(_samples, names, chunksize=64)), dtype=np.int64)
        bad = int(np.sum(frames < CROP_SAMPLES))
        if bad:
            print(f"  skipping {bad} clips that won't decode or are shorter than {CROP_SAMPLES / ref.SAMPLE_RATE:.2f} s")
        keep = np.flatnonzero(frames >= CROP_SAMPLES)
        files, names, frames = [files[i] for i in keep], [names[i] for i in keep], frames[keep]
    order = np.argsort(frames, kind="stable")  # By length, so teacher batches pad little.
    files = [files[i] for i in order]
    names = [names[i] for i in order]
    lengths = (frames[order] - ref.FRAME_LENGTH) // ref.HOP + 1
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    out = _split_dir(split)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{split}: {len(files)} clips, {len(speakers)} speakers, {offsets[-1] / 100 / 3600:.0f} h "
          f"(no overlap with the {len(test_speakers)} test speakers)")

    features = np.lib.format.open_memmap(
        out / "features.npy", mode="w+", dtype=np.float16, shape=(int(offsets[-1]), ref.NUM_MELS)
    )
    teacher = Teacher()
    teacher_out = np.zeros((len(files), TEACHER_DIM), dtype=np.float32)
    pending: list[tuple[int, np.ndarray]] = []
    start = time.time()

    def flush():
        if pending:
            teacher_out[[i for i, _ in pending]] = teacher.embed([clip for _, clip in pending])
            pending.clear()
            if teacher.device.type == "mps":
                torch.mps.empty_cache()

    def decoded(pool):
        # Bounded slices: an unbounded imap backlog filled the Mac's memory
        # on Common Voice (see cmd_prepare_cv).
        for begin in range(0, len(names), 2000):
            yield from pool.imap(_load_vox_features, names[begin : begin + 2000], chunksize=16)

    with multiprocessing.Pool() as pool:
        for i, (feats, pcm) in enumerate(decoded(pool)):
            assert len(feats) == lengths[i], names[i]
            features[offsets[i] : offsets[i + 1]] = feats
            longest = max(len(pcm), max((len(clip) for _, clip in pending), default=0))
            if pending and longest * (len(pending) + 1) > args.batch_seconds * ref.SAMPLE_RATE:
                flush()
            pending.append((i, pcm))
            if (i + 1) % 2000 == 0 or i + 1 == len(files):
                rate = offsets[i + 1] / 100 / (time.time() - start)
                print(f"  {i + 1}/{len(files)}  ({rate:.0f}x real time)", flush=True)
    flush()
    features.flush()

    np.savez(
        out / "index.npz",
        offsets=offsets,
        speaker=np.array([speaker_index[f.parts[-3]] for f in files], dtype=np.int32),
        teacher=teacher_out,
        paths=np.array(names),
        speakers=np.array(speakers),
    )
    print(f"Wrote {out} in {time.time() - start:.0f}s")
    return 0


class Split:
    def __init__(self, name: str):
        directory = _split_dir(name)
        if not (directory / "index.npz").is_file():
            raise SystemExit(f"error: split {name!r} isn't prepared; run: spk.py prepare {name}")
        index = np.load(directory / "index.npz")
        self.name = name
        self.features = np.load(directory / "features.npy", mmap_mode="r")
        self.offsets = index["offsets"]
        self.lengths = np.diff(self.offsets)
        self.speaker = index["speaker"]
        self.teacher = index["teacher"]
        self.paths = index["paths"]
        self.num_speakers = int(self.speaker.max()) + 1

    def crop(self, utterance: int, start_frame: int, frames: int = CROP_FRAMES) -> np.ndarray:
        begin = self.offsets[utterance] + start_frame
        return np.asarray(self.features[begin : begin + frames], dtype=np.float32)


class TrainingSet:
    """One or more prepared splits used together.

    Speaker numbers are offset per split, so a speed-changed copy of a split
    contributes its own speakers rather than more data for existing ones.
    """

    def __init__(self, names: list[str]):
        self.splits = [Split(name) for name in names]
        self.lengths = np.concatenate([split.lengths for split in self.splits])
        self.teacher = np.concatenate([split.teacher for split in self.splits])
        speaker_base = np.cumsum([0] + [split.num_speakers for split in self.splits])
        self.speaker = np.concatenate([split.speaker + base for split, base in zip(self.splits, speaker_base)])
        self.num_speakers = int(speaker_base[-1])
        self.hours = sum(split.offsets[-1] for split in self.splits) / 100 / 3600
        self._owner = np.concatenate([np.full(len(split.lengths), i) for i, split in enumerate(self.splits)])
        self._local = np.concatenate([np.arange(len(split.lengths)) for split in self.splits])

    def crop(self, utterance: int, start_frame: int, frames: int = CROP_FRAMES) -> np.ndarray:
        return self.splits[self._owner[utterance]].crop(int(self._local[utterance]), start_frame, frames)


def mix_log_mel(clean: np.ndarray, interference: np.ndarray, snr_db: np.ndarray) -> np.ndarray:
    """Adds interference to log-mel features at the given SNR per item.

    Two sounds that aren't correlated add in power, so mixing mel *powers*
    approximates mixing the waveforms, and lets training mix from the cached
    features without touching audio. clean, interference: [batch, frames,
    mels] log power; snr_db: [batch].
    """
    clean_power = np.exp(clean)
    other_power = np.exp(interference)
    gain = clean_power.mean(axis=(1, 2)) / (other_power.mean(axis=(1, 2)) * 10.0 ** (snr_db / 10.0))
    return np.log(clean_power + other_power * gain[:, None, None] + ref.LOG_FLOOR).astype(np.float32)


def random_noise_log_mel(rng: np.random.Generator, batch: int, frames: int = CROP_FRAMES) -> np.ndarray:
    """Synthetic background noise as log-mel features.

    Each item gets its own random spectral shape (a tilt from rumbly to
    hissy plus a few broad bumps), a slow loudness wander, and the
    frame-to-frame flutter that real noise has. Stands in for a recorded
    noise set until one is added.
    """
    mel = np.linspace(-1.0, 1.0, ref.NUM_MELS)
    tilt = rng.uniform(-3.0, 3.0, size=(batch, 1)) * mel
    bumps = np.zeros((batch, ref.NUM_MELS))
    for _ in range(3):
        centre = rng.uniform(-1.0, 1.0, size=(batch, 1))
        bumps += rng.uniform(-1.5, 1.5, size=(batch, 1)) * np.exp(-((mel - centre) ** 2) / 0.1)
    wander = np.cumsum(rng.normal(0.0, 0.05, size=(batch, frames, 1)), axis=1)
    flutter = np.log(rng.gamma(4.0, 0.25, size=(batch, frames, ref.NUM_MELS)))
    return ((tilt + bumps)[:, None, :] + wander + flutter).astype(np.float32)


def room_effects(crops: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Makes clean close-microphone features sound like a room and a cheap
    microphone or loudspeaker, working on the features.

    Echo: each mel band's power gets a decaying tail of what came before,
    with a random decay time (0.15-0.8 s to fall 60 dB, shorter for high
    bands) and a random strength relative to the direct sound (-5 to
    +15 dB). Device colouring: a random roll-off at the low and high ends
    and a few broad bumps. Self-noise: a flat noise floor 15-40 dB below
    the speech, which also stops the colouring being undone exactly by the
    model's mean subtraction.
    """
    batch = len(crops)
    mel = np.linspace(0.0, 1.0, ref.NUM_MELS)
    power = np.exp(crops.astype(np.float64))

    decay_time = rng.uniform(0.15, 0.8, size=(batch, 1)) * (1.0 - rng.uniform(0.0, 0.5, size=(batch, 1)) * mel)
    decay = 10.0 ** (-6.0 * (ref.HOP / ref.SAMPLE_RATE) / decay_time)  # Per-frame power decay.
    strength = 10.0 ** (-rng.uniform(-5.0, 15.0, size=(batch, 1)) / 10.0)
    tail = np.zeros((batch, ref.NUM_MELS))
    echoed = np.empty_like(power)
    for frame in range(power.shape[1]):
        echoed[:, frame] = power[:, frame] + strength * (1.0 - decay) * tail
        tail = decay * tail + power[:, frame]

    low_edge = rng.uniform(0.0, 0.25, size=(batch, 1))
    high_edge = rng.uniform(0.6, 1.0, size=(batch, 1))
    colour = -rng.uniform(20.0, 80.0, size=(batch, 1)) * np.maximum(0.0, low_edge - mel)
    colour -= rng.uniform(10.0, 60.0, size=(batch, 1)) * np.maximum(0.0, mel - high_edge)
    for _ in range(3):
        centre = rng.uniform(0.0, 1.0, size=(batch, 1))
        colour += rng.uniform(-6.0, 6.0, size=(batch, 1)) * np.exp(-((mel - centre) ** 2) / 0.02)
    echoed *= 10.0 ** (colour[:, None, :] / 10.0)

    floor = echoed.mean(axis=(1, 2)) * 10.0 ** (-rng.uniform(15.0, 40.0, size=batch) / 10.0)
    echoed += floor[:, None, None] * rng.gamma(4.0, 0.25, size=echoed.shape)
    return np.log(echoed + ref.LOG_FLOOR).astype(np.float32)


def augment(crops: np.ndarray, rng: np.random.Generator, room: bool = False) -> np.ndarray:
    """Leaves 40% of the batch clean; adds noise to 30% and a second,
    quieter voice (another crop from the batch) to 30%. With `room`, 80% of
    the batch first gets room and device effects (see room_effects).
    """
    batch = len(crops)
    if room:
        roomy = np.flatnonzero(rng.random(batch) < 0.8)
        crops = crops.copy()
        crops[roomy] = room_effects(crops[roomy], rng)
    choice = rng.random(batch)
    out = crops.copy()
    noisy = np.flatnonzero((choice >= 0.4) & (choice < 0.7))
    if len(noisy):
        out[noisy] = mix_log_mel(crops[noisy], random_noise_log_mel(rng, len(noisy), crops.shape[1]), rng.uniform(0.0, 20.0, len(noisy)))
    babble = np.flatnonzero(choice >= 0.7)
    if len(babble):
        others = crops[rng.permutation(batch)[: len(babble)]]
        out[babble] = mix_log_mel(crops[babble], others, rng.uniform(5.0, 20.0, len(babble)))
    return out


# ------------------------------------------------------------------- model


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, cin: int, cout: int, stride: tuple[int, int]):
        super().__init__(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class Student(nn.Module):
    """Tiny speaker encoder: [batch, CROP_FRAMES, NUM_MELS] -> unit voiceprint.

    Only layer types Arm's CMSIS-NN provides in 8-bit (conv, depthwise conv,
    fully connected), so it can run on a Cortex-M4. Its output lives in the
    teacher's 192-dimensional space, so a voice can be enrolled by either
    model.
    """

    def __init__(self, width: int = 16):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(1, w, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            DepthwiseSeparable(w, 2 * w, (2, 2)),
            DepthwiseSeparable(2 * w, 3 * w, (1, 2)),
            DepthwiseSeparable(3 * w, 4 * w, (1, 1)),
            DepthwiseSeparable(4 * w, 4 * w, (1, 1)),
        )
        mel_out = ref.NUM_MELS // 8  # Three stride-2 steps along the mel axis.
        self.embed = nn.Linear(2 * 4 * w * mel_out, TEACHER_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Subtract each mel band's mean over the window. This removes a fixed
        # microphone/channel response (a constant offset in the log domain),
        # which differs between the training recordings and the board's mic.
        x = x - x.mean(dim=1, keepdim=True)
        x = self.features(x.unsqueeze(1))  # [batch, channels, time, mel]
        x = x.permute(0, 2, 1, 3).flatten(2)  # [batch, time, channels * mel]
        # Statistics pooling: summarise the window by per-feature mean and
        # spread, giving a fixed-size vector whatever happens over time.
        pooled = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)
        return nn.functional.normalize(self.embed(pooled), dim=1)


ARCHITECTURES = ("student", "dscnn", "bcresnet", "tcs")


def make_model(arch: str = "student", width: int = 16) -> nn.Module:
    """Our Student, or a published baseline body (baselines.py) with the
    same input, pooling and output. `width` sizes the Student, and the
    tcs network's channels (5 per unit of width)."""
    if arch == "student":
        return Student(width)
    import baselines

    if arch == "tcs":
        return baselines.TCS(channels=5 * width)
    return {"dscnn": baselines.DSCNN, "bcresnet": baselines.BCResNet}[arch]()


def load_model(path: pathlib.Path, map_location="cpu") -> nn.Module:
    saved = torch.load(path, map_location=map_location)
    model = make_model(saved.get("arch", "student"), saved["width"])
    model.load_state_dict(saved["state"])
    return model


def model_cost(model: nn.Module) -> tuple[int, int]:
    """(parameters, multiply-accumulates per CROP_FRAMES window)."""
    macs = 0

    def hook(module, _inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            macs += output[0].numel() * kernel
        elif isinstance(module, nn.Linear):
            macs += module.in_features * module.out_features

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, CROP_FRAMES, ref.NUM_MELS, device=next(model.parameters()).device))
    model.train(was_training)
    for handle in handles:
        handle.remove()
    return sum(p.numel() for p in model.parameters()), macs


# -------------------------------------------------------------- evaluation


def equal_error_rate(scores: np.ndarray, same: np.ndarray) -> float:
    """Error rate at the threshold where false accepts equal false rejects."""
    order = np.argsort(-scores)
    same = same[order]
    false_accept = np.cumsum(~same) / max(1, (~same).sum())
    false_reject = 1.0 - np.cumsum(same) / max(1, same.sum())
    crossing = np.argmin(np.abs(false_accept - false_reject))
    return float((false_accept[crossing] + false_reject[crossing]) / 2)


class Trials:
    """Fixed verification trials: one 1.5 s crop from each of `per_speaker`
    utterances per speaker, scored against every crop from other utterances.
    """

    def __init__(self, split: Split, per_speaker: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        eligible = np.flatnonzero(split.lengths >= CROP_FRAMES)
        self.utterance, self.start = [], []
        for speaker in range(split.num_speakers):
            own = eligible[split.speaker[eligible] == speaker]
            for utterance in rng.choice(own, size=min(per_speaker, len(own)), replace=False):
                self.utterance.append(int(utterance))
                self.start.append(int(rng.integers(0, split.lengths[utterance] - CROP_FRAMES + 1)))
        self.split = split
        labels = split.speaker[self.utterance]
        upper = np.triu_indices(len(labels), k=1)
        self._pairs = upper
        self.same = labels[upper[0]] == labels[upper[1]]

    def crops(self) -> np.ndarray:
        return np.stack([self.split.crop(u, s) for u, s in zip(self.utterance, self.start)])

    def audio(self) -> list[np.ndarray]:
        clips = []
        for utterance, start in zip(self.utterance, self.start):
            pcm = read_pcm(str(self.split.paths[utterance]))
            clips.append(pcm[start * ref.HOP : start * ref.HOP + CROP_SAMPLES])
        return clips

    def eer(self, embeddings: np.ndarray) -> float:
        scores = np.einsum("ij,ij->i", embeddings[self._pairs[0]], embeddings[self._pairs[1]])
        return equal_error_rate(scores, self.same)


@torch.no_grad()
def student_embeddings(model: Student, crops: np.ndarray) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = [model(torch.from_numpy(crops[i : i + 256]).to(dev)).cpu().numpy() for i in range(0, len(crops), 256)]
    return np.concatenate(out)


# ------------------------------------------------------------------- train


def cmd_train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = device()
    train = TrainingSet(args.split)
    trials = Trials(Split(args.eval_split), args.eval_per_speaker) if args.eval_split else None
    eval_crops = trials.crops() if trials else None

    model = make_model(args.arch, args.width).to(dev)
    params, macs = model_cost(model)
    print(f"student: {params / 1e3:.1f}k parameters, {macs / 1e6:.2f}M MACs per 1.5 s window, device {dev}")
    print(
        f"train: {' + '.join(args.split)}, {train.num_speakers} speakers, {train.hours:.1f} h"
        f"{', with noise/second-voice augmentation' if args.augment or args.room else ''}"
        f"{' and room effects' if args.room else ''}"
    )

    # Cosine classifier over the training speakers; used in training only.
    classifier = nn.Linear(TEACHER_DIM, train.num_speakers, bias=False).to(dev)
    optimizer = torch.optim.AdamW([*model.parameters(), *classifier.parameters()], lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * args.steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)

    frames = args.crop_frames
    eligible = np.flatnonzero(train.lengths >= frames)
    # Longer utterances contain more distinct crops, so sample them more.
    weights = (train.lengths[eligible] - frames + 1).astype(np.float64)
    weights /= weights.sum()

    real_bank = None
    if args.real_noise:
        import real_noise

        real_bank = real_noise.Bank()
        print("augmentation: real noise, music, 1-5 voice babble and measured rooms (real_noise.py)")

    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        totals = np.zeros(2)
        for _ in range(args.steps):
            utterances = rng.choice(eligible, size=args.batch, p=weights)
            starts = rng.integers(0, train.lengths[utterances] - frames + 1)
            crops = np.stack([train.crop(u, s, frames) for u, s in zip(utterances, starts)])
            if real_bank is not None:
                crops = real_noise.augment(crops, rng, real_bank, mix_log_mel, room_effects)
            elif args.augment or args.room:
                crops = augment(crops, rng, room=args.room)
            x = torch.from_numpy(crops).to(dev)
            target = torch.from_numpy(train.teacher[utterances]).to(dev)
            labels = torch.from_numpy(train.speaker[utterances]).long().to(dev)

            embedding = model(x)
            distill = (1.0 - (embedding * target).sum(dim=1)).mean()
            cosine = embedding @ nn.functional.normalize(classifier.weight, dim=1).T
            if args.margin > 0.0:
                # Additive angular margin (ArcFace): the right speaker must win
                # by an angle of `margin`. Ramped up over the first tenth of
                # training, which is unstable otherwise.
                margin = args.margin * min(1.0, step / (0.1 * total_steps))
                own = cosine.gather(1, labels[:, None]).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                cosine = cosine.scatter(1, labels[:, None], torch.cos(torch.acos(own) + margin))
            classify = nn.functional.cross_entropy(args.scale * cosine, labels)
            loss = args.distill_weight * distill + args.classify_weight * classify
            step += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            totals += [distill.item(), classify.item()]

        entry = {
            "epoch": epoch,
            "distill": totals[0] / args.steps,
            "classify": totals[1] / args.steps,
            "seconds": round(time.time() - start, 1),
        }
        if trials:
            entry["eer"] = trials.eer(student_embeddings(model, eval_crops))
        history.append(entry)
        eer = f"  {args.eval_split} EER {100 * entry['eer']:.2f}%" if trials else ""
        print(
            f"epoch {epoch:3d}  distill {entry['distill']:.4f}  classify {entry['classify']:.3f}"
            f"{eer}  ({entry['seconds']:.0f}s)",
            flush=True,
        )

    _MODELS.mkdir(parents=True, exist_ok=True)
    settings = {key: value for key, value in vars(args).items() if key != "run"}
    torch.save({"state": model.state_dict(), "width": args.width, "arch": args.arch}, _MODELS / f"{args.name}.pt")
    (_MODELS / f"{args.name}.json").write_text(
        json.dumps({"args": settings, "parameters": params, "macs": macs, "history": history}, indent=2) + "\n"
    )
    print(f"Saved {_MODELS / args.name}.pt")
    return 0


def cmd_prepare_noise(_: argparse.Namespace) -> int:
    import real_noise

    real_noise.prepare(log_mel_torch)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    dev = device()
    checkpoint = torch.load(_MODELS / f"{args.name}.pt", map_location=dev)
    model = make_model(checkpoint.get("arch", "student"), checkpoint["width"]).to(dev)
    model.load_state_dict(checkpoint["state"])
    params, macs = model_cost(model)

    split = Split(args.split)
    trials = Trials(split, args.per_speaker)
    crops = trials.crops()
    student = trials.eer(student_embeddings(model, crops))
    teacher = Teacher()
    clips = trials.audio()
    teacher_embeddings = np.concatenate([teacher.embed(clips[i : i + 64]) for i in range(0, len(clips), 64)])
    teacher_crop = trials.eer(teacher_embeddings)

    print(f"\n{args.split}: {len(trials.utterance)} crops of 1.5 s, {len(trials.same)} trials ({int(trials.same.sum())} same-speaker)")
    print(f"  student {args.name:<10} {params / 1e3:7.1f}k params  {macs / 1e6:5.2f}M MACs   EER {100 * student:5.2f}%")
    print(f"  teacher ECAPA-TDNN  {'~20000':>7}k params                 EER {100 * teacher_crop:5.2f}%   (same 1.5 s crops)")

    # Harder conditions, student only: the mixing is done on features, which
    # the teacher (a waveform model) can't take.
    rng = np.random.default_rng(1)
    labels = split.speaker[trials.utterance]
    for snr in args.second_voice_snr:
        # Pair each crop with one from a different speaker.
        partner = np.array([rng.choice(np.flatnonzero(labels != label)) for label in labels])
        mixed = mix_log_mel(crops, crops[partner], np.full(len(crops), float(snr)))
        print(f"  student {args.name:<10} with a second voice {snr:>2} dB quieter:      EER {100 * trials.eer(student_embeddings(model, mixed)):5.2f}%")
    for snr in args.noise_snr:
        mixed = mix_log_mel(crops, random_noise_log_mel(rng, len(crops)), np.full(len(crops), float(snr)))
        print(f"  student {args.name:<10} with noise {snr:>2} dB quieter:               EER {100 * trials.eer(student_embeddings(model, mixed)):5.2f}%")
    # Every crop in its own simulated room and device.
    roomy = room_effects(crops, rng)
    print(f"  student {args.name:<10} with room and device effects:          EER {100 * trials.eer(student_embeddings(model, roomy)):5.2f}%")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("selfcheck", help="Check the feature code against the reference").set_defaults(run=cmd_selfcheck)

    prepare = commands.add_parser("prepare", help="Cache features and teacher voiceprints for a split")
    prepare.add_argument("split", help="LibriSpeech split, e.g. train-clean-100")
    prepare.add_argument("--batch-seconds", type=float, default=80.0, help="Audio per teacher batch, after padding")
    prepare.add_argument("--speed", type=float, default=1.0, help="Playback speed, e.g. 0.9 or 1.1; saved as SPLIT@SPEED")
    prepare.set_defaults(run=cmd_prepare)

    for name, run, text in (
        ("cv-select", cmd_cv_select, "List the Common Voice clips to unpack"),
        ("prepare-cv", cmd_prepare_cv, "Cache features and teacher voiceprints for Common Voice"),
    ):
        cv = commands.add_parser(name, help=text)
        if name == "prepare-cv":
            cv.add_argument("split", choices=["cv-train", "cv-test"])
            cv.add_argument("--batch-seconds", type=float, default=80.0)
        cv.add_argument("--min-clips", type=int, default=20, help="Use speakers with at least this many clean clips")
        cv.add_argument("--cap", type=int, default=30, help="Clips kept per speaker")
        cv.add_argument("--test-speakers", type=int, default=200, help="Speakers held out as cv-test")
        cv.add_argument("--seed", type=int, default=0)
        cv.set_defaults(run=run)

    vox = commands.add_parser("prepare-vox", help="Cache features and teacher voiceprints for VoxCeleb1 or 2 dev")
    vox.add_argument("--dataset", choices=("vox1", "vox2"), default="vox1")
    vox.add_argument("--batch-seconds", type=float, default=80.0)
    vox.set_defaults(run=cmd_prepare_vox)

    train = commands.add_parser("train", help="Train a student")
    train.add_argument("--split", required=True, nargs="+", help="Prepared split(s), e.g. train-clean-100 train-clean-100@0.9")
    train.add_argument("--augment", action="store_true", help="Mix noise and second voices into training crops")
    train.add_argument("--room", action="store_true", help="Also add simulated echo and device colouring (implies --augment)")
    train.add_argument("--name", required=True, help="Name to save the model under")
    train.add_argument("--eval-split", help="Prepared split to report EER on after each epoch")
    train.add_argument("--eval-per-speaker", type=int, default=20)
    train.add_argument("--width", type=int, default=16, help="Channel multiplier; sets the model size")
    train.add_argument("--arch", choices=ARCHITECTURES, default="student", help="Our Student or a published baseline (baselines.py)")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--steps", type=int, default=500, help="Batches per epoch")
    train.add_argument("--batch", type=int, default=256)
    train.add_argument("--lr", type=float, default=3e-3)
    train.add_argument("--classify-weight", type=float, default=0.2)
    train.add_argument("--distill-weight", type=float, default=1.0, help="0 trains on speaker labels alone")
    train.add_argument("--margin", type=float, default=0.0, help="Additive angular margin of the classifier, radians")
    train.add_argument("--crop-frames", type=int, default=CROP_FRAMES, help="Training crop length; 100 frames per second")
    train.add_argument("--scale", type=float, default=30.0, help="Logit scale of the cosine classifier")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--real-noise", action="store_true", help="Real noise, music, crowd and rooms (real_noise.py) instead of --augment/--room")
    train.set_defaults(run=cmd_train)

    noise = commands.add_parser("prepare-noise", help="Features of MUSAN noise/music and room responses for --real-noise")
    noise.set_defaults(run=cmd_prepare_noise)

    evaluate = commands.add_parser("eval", help="EER of a student and the teacher")
    evaluate.add_argument("--name", required=True)
    evaluate.add_argument("--split", default="test-clean")
    evaluate.add_argument("--per-speaker", type=int, default=30)
    evaluate.add_argument("--second-voice-snr", type=int, nargs="*", default=[10], help="Also test with another speaker mixed in this many dB quieter")
    evaluate.add_argument("--noise-snr", type=int, nargs="*", default=[10], help="Also test with synthetic noise this many dB quieter")
    evaluate.set_defaults(run=cmd_eval)

    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
