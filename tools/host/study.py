# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "sounddevice"]
# ///
"""Records one participant for the multi-speaker study, step by step.

Follows docs/recording_protocol.md. Each step shows what to say, counts
down, records raw audio from the board's microphone (no filtering, so any
model can be scored on it later), and checks the level. Files go to
recordings/study/<ID>/, which git ignores.

Usage:
  uv run tools/host/study.py P01 --room office
  uv run tools/host/study.py P01 --room kitchen --only noise far
  uv run tools/host/study.py --list-steps
"""

import argparse
import datetime
import json
import pathlib
import queue
import sys
import time
import wave

import numpy as np
import sounddevice as sd

_RATE = 16000
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_STUDY = _ROOT / "recordings" / "study"

_READING = (
    "When the sunlight strikes raindrops in the air, they act as a prism and "
    "form a rainbow. The rainbow is a division of white light into many "
    "beautiful colors. These take the shape of a long round arch, with its "
    "path high above, and its two ends apparently beyond the horizon."
)

# (name, seconds, distance, background, what to tell the participant)
_STEPS = (
    ("enroll", 20, "30 cm", "quiet", "Read this aloud at a normal pace, twice if you finish early:\n\n" + _READING),
    ("quiet", 60, "30 cm", "quiet", "Talk freely: describe your day, your work, or a recent trip. Pauses are fine."),
    ("far", 45, "1 m", "quiet", "Move back to about 1 metre. Keep talking freely, in your normal voice."),
    ("noise", 45, "30 cm", "noise", "Start the background noise (cafe sounds from a phone, 1 m away). Talk freely."),
    ("overlap", 45, "30 cm", "speech", "Start a talk video on a speaker 1 m away. Talk over it and pause now and then."),
    ("soft", 30, "30 cm", "quiet", "Talk quietly, as if someone is asleep nearby (not a whisper)."),
)


def _find_device(name: str) -> int:
    for index, device in enumerate(sd.query_devices()):
        if name.lower() in device["name"].lower() and device["max_input_channels"] > 0:
            return index
    raise SystemExit(f"error: no input device matching {name!r}. Is the board's micro-USB port (CN5) connected?")


def _record(device: int, seconds: float) -> np.ndarray:
    blocks: "queue.Queue[np.ndarray]" = queue.Queue()

    def on_audio(indata, frames, when, status):
        blocks.put(indata[:, 0].copy())

    total = int(seconds * _RATE)
    got: list[np.ndarray] = []
    count = 0
    with sd.InputStream(device=device, channels=1, samplerate=_RATE, dtype="float32", blocksize=_RATE // 10, callback=on_audio):
        while count < total:
            block = blocks.get()
            got.append(block)
            count += len(block)
            level = 20 * np.log10(max(float(np.max(np.abs(block))), 1e-6))
            bar = "#" * int(30 * min(1.0, max(0.0, (level + 60) / 60)))
            print(f"\r  {count / _RATE:5.1f}/{seconds:.0f}s [{bar:<30}] {level:6.1f} dBFS", end="", flush=True)
    print()
    # Exact float-to-int16, so the board's match bit (lowest bit) survives.
    audio = np.concatenate(got)[:total]
    return np.round(audio * 32768.0).clip(-32768, 32767).astype(np.int16)


def _save(path: pathlib.Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_RATE)
        wav.writeframes(pcm.tobytes())


def _level_report(pcm: np.ndarray) -> tuple[float, float, str]:
    """(loudest 10 ms frame in dBFS, share of frames with speech, verdict)."""
    frames = pcm[: len(pcm) // 160 * 160].astype(np.float32).reshape(-1, 160) / 32768.0
    levels = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    floor = float(np.percentile(levels, 10))
    speech = float(np.mean(levels > floor + 10))
    peak = float(np.max(levels))
    if peak > -3:
        verdict = "TOO LOUD (clipping): move back a little and redo"
    elif peak < -40:
        verdict = "TOO QUIET: move closer or speak up and redo"
    elif speech < 0.3:
        verdict = "LITTLE SPEECH: talk more continuously and redo"
    else:
        verdict = "ok"
    return peak, speech, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("participant", nargs="?", help="Participant ID, e.g. P01 (no names)")
    parser.add_argument("--room", help="Short room label, e.g. office, kitchen, cafe")
    parser.add_argument("--only", nargs="+", metavar="STEP", help="Record only these steps")
    parser.add_argument("--device", default="device_ml")
    parser.add_argument("--list-steps", action="store_true")
    args = parser.parse_args()

    if args.list_steps:
        for name, seconds, distance, background, _ in _STEPS:
            print(f"{name:<8} {seconds:3d} s  {distance:<6} {background}")
        return 0
    if not args.participant or not args.room:
        parser.error("give a participant ID and --room")

    device = _find_device(args.device)
    out = _STUDY / args.participant
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"participant": args.participant, "takes": []}

    steps = [s for s in _STEPS if not args.only or s[0] in args.only]
    for name, seconds, distance, background, instructions in steps:
        while True:
            print(f"\n=== {name}: {seconds} s, mic distance {distance}, background: {background} ===")
            print(instructions)
            if input("\nPress Enter to start (s = skip): ").strip().lower() == "s":
                break
            for n in (3, 2, 1):
                print(f"  {n}...", flush=True)
                time.sleep(1)
            print("  Recording.")
            pcm = _record(device, seconds)
            peak, speech, verdict = _level_report(pcm)
            print(f"  loudest {peak:.1f} dBFS, speech in {100 * speech:.0f}% of frames -> {verdict}")
            if verdict != "ok" and input("  Redo this step? [Y/n] ").strip().lower() != "n":
                continue
            path = out / f"{args.room}_{name}.wav"
            _save(path, pcm)
            meta["takes"].append(
                {
                    "file": path.name,
                    "step": name,
                    "room": args.room,
                    "distance": distance,
                    "background": background,
                    "seconds": seconds,
                    "peak_dbfs": round(peak, 1),
                    "speech_share": round(speech, 2),
                    "recorded": datetime.datetime.now().isoformat(timespec="seconds"),
                }
            )
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"  Saved {path.relative_to(_ROOT)}")
            break

    print(f"\nDone. Files in {out.relative_to(_ROOT)}; details in meta.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
