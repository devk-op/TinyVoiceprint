# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "sounddevice", "soundfile"]
# ///
"""Plays VoxCeleb1 test clips through the Mac's speaker and records them
with the board's microphone, so the benchmark can be scored on audio that
went through the real room, microphone and on-board audio pipeline.

Picks a fixed subset (seeded): N clips of 4-10 s from each of the 40 test
speakers. Each recording is lined up with the original by cross-correlation
and cut to the original's length, then saved to
recordings/vox_replay/<speaker>_<video>_<clip>.wav with the same name
pattern as the original (git-ignored; VoxCeleb is CC BY 4.0).

Re-running skips clips already recorded, so it can be stopped and resumed.

Usage:
  uv run tools/host/vox_replay.py --test     # 3 clips, prints levels
  uv run tools/host/vox_replay.py            # the full subset
"""

import argparse
import collections
import json
import pathlib
import subprocess
import sys
import threading
import time

import numpy as np
import scipy.signal
import sounddevice as sd
import soundfile as sf

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VOX = _ROOT / "data" / "voxceleb" / "wav"
_OUT = _ROOT / "recordings" / "vox_replay"
_RATE = 16000
_GAP_SECONDS = 0.6  # Silence after each clip so its echo dies down.
_MAX_LAG = int(0.5 * _RATE)


def pick_clips(per_speaker: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    by_speaker = collections.defaultdict(list)
    for path in sorted(_VOX.glob("id*/*/*.wav")):
        seconds = sf.info(str(path)).frames / _RATE
        if 4.0 <= seconds <= 10.0:
            by_speaker[path.parts[-3]].append(str(path.relative_to(_VOX)))
    chosen = []
    for speaker in sorted(by_speaker):
        clips = by_speaker[speaker]
        chosen += list(rng.choice(clips, size=min(per_speaker, len(clips)), replace=False))
    return chosen


def find_device(name: str, direction: str) -> int:
    key = "max_input_channels" if direction == "input" else "max_output_channels"
    for index, device in enumerate(sd.query_devices()):
        if name.lower() in device["name"].lower() and device[key] > 0:
            return index
    raise SystemExit(f"error: no {direction} device matching {name!r}:\n{sd.query_devices()}")


class Recorder:
    """Records the board continuously; hands out stretches by sample index."""

    def __init__(self, device: int):
        self.chunks: list[np.ndarray] = []
        self.count = 0
        self.lock = threading.Lock()
        self.stream = sd.InputStream(device=device, channels=1, samplerate=_RATE, dtype="float32",
                                     blocksize=_RATE // 50, callback=self._on_audio)

    def _on_audio(self, indata, frames, _time, status):
        with self.lock:
            self.chunks.append(indata[:, 0].copy())
            self.count += frames

    def take(self, start: int, end: int) -> np.ndarray:
        with self.lock:
            audio = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
            base = self.count - len(audio)
            # Keep memory bounded: drop audio before `start`.
            keep = max(0, start - base)
            self.chunks = [audio[keep:]]
        return audio[start - base : end - base]


def align(original: np.ndarray, recorded: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Cuts `recorded` to line up with `original`; returns (audio, lag, peak
    correlation). Correlation is on the first 3 s, normalised to 0..1."""
    a = original[: 3 * _RATE].astype(np.float64)
    b = recorded[: len(a) + _MAX_LAG].astype(np.float64)
    corr = scipy.signal.correlate(b, a, mode="valid", method="fft")[: _MAX_LAG + 1]
    lag = int(np.argmax(np.abs(corr)))
    norm = np.linalg.norm(a) * np.linalg.norm(b[lag : lag + len(a)]) + 1e-9
    return recorded[lag : lag + len(original)], lag, float(abs(corr[lag]) / norm)


def to_int16(audio: np.ndarray) -> np.ndarray:
    return np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-speaker", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test", action="store_true", help="Record 3 clips into a scratch folder and print levels")
    parser.add_argument("--input", default="device_ml")
    parser.add_argument("--output", default="Mac mini Speakers")
    args = parser.parse_args()

    out_dir = _OUT / "_test" if args.test else _OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = pick_clips(args.per_speaker, args.seed)
    if args.test:
        clips = clips[:: len(clips) // 3][:3]
    todo = [c for c in clips if not (out_dir / c.replace("/", "_")).exists()]
    seconds = sum(sf.info(str(_VOX / c)).frames / _RATE + _GAP_SECONDS for c in todo)
    volume = subprocess.run(["osascript", "-e", "output volume of (get volume settings)"],
                            capture_output=True, text=True).stdout.strip()
    print(f"{len(clips)} clips from {len({c.split('/')[0] for c in clips})} speakers, "
          f"{len(todo)} to record, about {seconds / 60:.0f} min. Mac output volume {volume}.")
    (out_dir / "manifest.json").write_text(json.dumps({"seed": args.seed, "per_speaker": args.per_speaker,
                                                       "clips": clips, "volume": volume}, indent=1))

    output = find_device(args.output, "output")
    recorder = Recorder(find_device(args.input, "input"))
    log = []
    with recorder.stream:
        time.sleep(1.0)  # Let the board's stream settle.
        for i, clip in enumerate(todo):
            original, rate = sf.read(str(_VOX / clip), dtype="float32")
            assert rate == _RATE
            start = recorder.count
            sd.play(original, _RATE, device=output, blocking=True)
            time.sleep(_GAP_SECONDS)
            recorded = recorder.take(start, recorder.count)
            aligned, lag, peak = align(original, recorded)
            if len(aligned) < len(original):
                print(f"  {clip}: recording too short, skipped", flush=True)
                continue
            clipped = float(np.mean(np.abs(aligned) > 0.99))
            if clipped > 0.01:
                # The board's audio has gone wrong (it happened once after a
                # debugger connection); stop rather than save garbage.
                print(f"error: {clip}: {100 * clipped:.0f}% of samples clipped; stopping. "
                      "Reset the board and run again (finished clips are kept).", flush=True)
                return 1
            sf.write(str(out_dir / clip.replace("/", "_")), to_int16(aligned), _RATE, subtype="PCM_16")
            level = 20 * np.log10(np.sqrt(np.mean(aligned**2)) + 1e-9)
            log.append(peak)
            if args.test or i % 20 == 0:
                print(f"  {i + 1}/{len(todo)} {clip}  delay {1000 * lag / _RATE:.0f} ms  "
                      f"level {level:.0f} dBFS  match {peak:.2f}", flush=True)
    if log:
        print(f"Done. Median match with the original {np.median(log):.2f} (1 = identical waveform).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
