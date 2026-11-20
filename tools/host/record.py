# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "sounddevice"]
# ///
"""Records from the board's USB microphone to a WAV file.

The board enumerates as a standard USB audio input named "device_ml mic"
(16 kHz, 16-bit mono), so this is a thin wrapper around the OS audio stack
with a live level meter to confirm the mic is picking up sound.

Usage:
  uv run tools/host/record.py --list
  uv run tools/host/record.py                  # until Ctrl-C
  uv run tools/host/record.py -t 30 -o clip.wav
  uv run tools/host/record.py -t 30 --only-me -o mine.wav

--only-me keeps only the audio the board attributes to the enrolled voice.
The decision is made on the chip and travels in the lowest bit of each
sample; this script only cuts. Because the chip needs about a second of
speech before it decides, each kept stretch also includes the 1.2 s before
it. Add --keep-timing to silence the rest instead of cutting it out.
"""

import argparse
import datetime
import pathlib
import queue
import sys
import wave

import numpy as np
import sounddevice as sd

_SAMPLE_RATE = 16000
_DEFAULT_DEVICE = "device_ml"
_RECORDINGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "recordings"


def _find_input_device(name: str) -> int:
    for index, device in enumerate(sd.query_devices()):
        if name.lower() in device["name"].lower() and device["max_input_channels"] > 0:
            return index
    raise SystemExit(
        f"error: no input device matching {name!r}. Is the board plugged in "
        "via the micro-USB port (CN5)? Try --list."
    )


def _meter(block: np.ndarray, width: int = 40) -> str:
    peak = int(np.max(np.abs(block))) if block.size else 0
    dbfs = 20 * np.log10(max(peak, 1) / 32768)
    filled = int(width * max(0.0, (dbfs + 60) / 60))  # -60..0 dBFS
    return f"[{'#' * filled}{' ' * (width - filled)}] {dbfs:6.1f} dBFS"


_FRAME = 160  # 10 ms
_PRE_ROLL_FRAMES = 120  # 1.2 s kept before each stretch the chip marks.


def keep_only_marked(audio: np.ndarray, keep_timing: bool) -> tuple[np.ndarray, float]:
    """Returns (audio with unmarked parts removed or silenced, seconds kept)."""
    frames = len(audio) // _FRAME
    marks = (audio[: frames * _FRAME] & 1).reshape(frames, _FRAME).mean(axis=1) > 0.5
    keep = marks.copy()
    for onset in np.flatnonzero(marks[1:] & ~marks[:-1]) + 1:
        keep[max(0, onset - _PRE_ROLL_FRAMES) : onset] = True
    if marks.size and marks[0]:
        keep[0] = True
    mask = np.repeat(keep, _FRAME)
    body = audio[: frames * _FRAME]
    kept_seconds = float(keep.sum()) * _FRAME / _SAMPLE_RATE
    return (np.where(mask, body, 0).astype(np.int16) if keep_timing else body[mask]), kept_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-t", "--seconds", type=float, default=0, help="Recording length; 0 records until Ctrl-C")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="WAV path (default: recordings/<timestamp>.wav)")
    parser.add_argument("-d", "--device", default=_DEFAULT_DEVICE, help="Substring of the input device name")
    parser.add_argument("--list", action="store_true", help="List audio devices and exit")
    parser.add_argument("--only-me", action="store_true", help="Keep only what the board marks as the enrolled voice")
    parser.add_argument("--keep-timing", action="store_true", help="With --only-me: silence other audio instead of cutting it")
    args = parser.parse_args()

    if args.list:
        print(sd.query_devices())
        return 0

    device = _find_input_device(args.device)
    output = args.output or _RECORDINGS_DIR / f"{datetime.datetime.now():%Y%m%d-%H%M%S}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)

    blocks: "queue.Queue[np.ndarray]" = queue.Queue()

    def on_audio(indata, frames, time, status):  # Runs on the audio thread.
        if status:
            print(f"\n{status}", file=sys.stderr)
        # Captured as float and converted here: the audio stack's own
        # float-to-int16 step is not exact, and would corrupt the mark bit.
        blocks.put(np.round(indata * 32768.0).clip(-32768, 32767).astype(np.int16))

    target_frames = int(args.seconds * _SAMPLE_RATE)
    recorded = 0
    print(f"Recording from {sd.query_devices(device)['name']!r} to {output}" + ("" if target_frames else " (Ctrl-C to stop)"))

    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_SAMPLE_RATE)
        try:
            with sd.InputStream(device=device, channels=1, samplerate=_SAMPLE_RATE, dtype="float32", blocksize=_SAMPLE_RATE // 10, callback=on_audio):
                while not target_frames or recorded < target_frames:
                    block = blocks.get()
                    if target_frames:
                        block = block[: target_frames - recorded]
                    wav.writeframes(block.tobytes())
                    recorded += len(block)
                    print(f"\r{recorded / _SAMPLE_RATE:7.1f}s {_meter(block)}", end="", flush=True)
        except KeyboardInterrupt:
            pass

    if args.only_me:
        with wave.open(str(output), "rb") as wav:
            audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        mine, kept = keep_only_marked(audio, args.keep_timing)
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(_SAMPLE_RATE)
            wav.writeframes(mine.tobytes())
        print(f"\nRecorded {recorded / _SAMPLE_RATE:.1f}s, kept {kept:.1f}s as the enrolled voice -> {output}")
        return 0
    print(f"\nSaved {recorded / _SAMPLE_RATE:.1f}s to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
