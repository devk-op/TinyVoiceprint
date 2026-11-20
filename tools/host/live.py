# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "sounddevice", "soundfile", "torch<2.9"]
# ///
"""Live voice match with the tiny model, using the board as the microphone.

The Mac stands in for the chip here: it runs the same features
(tools/host/log_mel_reference.py) and the small "student" network trained by
tools/train/spk.py, on 1.5 s windows, and says whose voice it hears. Nothing
large runs: no ECAPA, no neural voice-activity detector. Silence is detected
from loudness alone, as the firmware will.

  enroll   Learn a voice from ~15 s of speech (from the board or a WAV file).
  listen   Live: show who is speaking.
  score    Same decision over a WAV file, printed as a timeline.
  voices   List enrolled voices.

Usage:
  uv run tools/host/live.py enroll krathi
  uv run tools/host/live.py enroll claude --from recordings/claude_only.wav
  uv run tools/host/live.py listen
  uv run tools/host/live.py score recordings/claude_talk.wav
"""

import argparse
import collections
import pathlib
import queue
import sys
import time

import numpy as np
import soundfile as sf
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tools" / "train"))
import spk  # noqa: E402  (also puts log_mel_reference on the path)
import log_mel_reference as ref  # noqa: E402

_RATE = ref.SAMPLE_RATE
_WINDOW = spk.CROP_SAMPLES  # 1.5 s, the length the model was trained on.
_HOP = _RATE // 4  # A new decision every 0.25 s.
_MODELS = _ROOT / "data" / "models"
_VOICES = _ROOT / "recordings" / "voices_tiny"
_DEFAULT_DEVICE = "device_ml"

# A 10 ms frame counts as speech when it is this much louder than the
# background level, and a window counts as speech when at least this share
# of its frames do.
_SPEECH_MARGIN_DB = 10.0
_SPEECH_SHARE = 0.4
# Cosine similarity above which a window is attributed to an enrolled voice.
# Measured on 40 unseen LibriSpeech speakers with model v1 (11 s enrollment,
# 4-window decisions): the enrolled speaker averages 0.77 and a stranger
# 0.37; the two error rates cross near 0.58 at about 4%. Override with
# --threshold.
_MATCH_THRESHOLD = 0.60
# Decisions are made on the average of this many recent speech windows
# (4 windows at a 0.25 s hop span about 2.25 s of audio).
_SMOOTH_WINDOWS = 4
_MIN_ENROLL_WINDOWS = 6


def load_model(name: str) -> spk.Student:
    path = _MODELS / f"{name}.pt"
    if not path.exists():
        raise SystemExit(f"error: no model at {path}. Train one with tools/train/spk.py.")
    saved = torch.load(path, map_location="cpu")
    model = spk.make_model(saved.get("arch", "student"), saved["width"])
    model.load_state_dict(saved["state"])
    return model.eval()


def read_wav(path: pathlib.Path) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="int16", always_2d=True)
    if rate != _RATE:
        raise SystemExit(f"error: {path} is {rate} Hz; this tool needs {_RATE} Hz (record.py output is).")
    return audio[:, 0]


def frame_levels_db(pcm: np.ndarray) -> np.ndarray:
    """Loudness of each 10 ms frame, in dB relative to full scale."""
    usable = len(pcm) // ref.HOP * ref.HOP
    frames = pcm[:usable].astype(np.float32).reshape(-1, ref.HOP) / 32768.0
    return 10.0 * np.log10(np.mean(frames**2, axis=1) + 1e-10)


def is_speech(window: np.ndarray, floor_db: float) -> bool:
    levels = frame_levels_db(window)
    return float(np.mean(levels > floor_db + _SPEECH_MARGIN_DB)) >= _SPEECH_SHARE


def background_db(pcm: np.ndarray) -> float:
    """Background level: the quietest tenth of the frames."""
    return float(np.percentile(frame_levels_db(pcm), 10))


@torch.no_grad()
def embed(model: spk.Student, windows: np.ndarray) -> np.ndarray:
    """[n, _WINDOW] int16 -> [n, 192] unit voiceprints."""
    features = spk.log_mel_torch(torch.from_numpy(np.ascontiguousarray(windows)))
    return model(features).numpy()


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / (np.linalg.norm(vector) + 1e-9)


def voices_dir(model_name: str) -> pathlib.Path:
    return _VOICES / model_name


def load_voices(model_name: str) -> dict[str, np.ndarray]:
    return {path.stem: np.load(path) for path in sorted(voices_dir(model_name).glob("*.npy"))}


# ------------------------------------------------------------------ capture


def find_input_device(name: str) -> int:
    import sounddevice as sd

    for index, device in enumerate(sd.query_devices()):
        if name.lower() in device["name"].lower() and device["max_input_channels"] > 0:
            return index
    raise SystemExit(
        f"error: no input device matching {name!r}. Is the board plugged in via the micro-USB port (CN5)?"
    )


def open_stream(device_name: str, blocks: "queue.Queue[np.ndarray]"):
    import sounddevice as sd

    def on_audio(indata, frames, when, status):  # Runs on the audio thread.
        blocks.put(indata[:, 0].copy())

    return sd.InputStream(
        device=find_input_device(device_name),
        channels=1,
        samplerate=_RATE,
        dtype="int16",
        blocksize=_HOP,
        callback=on_audio,
    )


def record(device_name: str, seconds: float) -> np.ndarray:
    blocks: "queue.Queue[np.ndarray]" = queue.Queue()
    collected: list[np.ndarray] = []
    total = int(seconds * _RATE)
    got = 0
    with open_stream(device_name, blocks):
        while got < total:
            block = blocks.get()
            collected.append(block)
            got += len(block)
            level = float(np.max(frame_levels_db(block)))
            bar = "#" * int(30 * min(1.0, max(0.0, (level + 70) / 60)))
            print(f"\r  {got / _RATE:5.1f}/{seconds:.0f}s [{bar:<30}]", end="", flush=True)
    print()
    return np.concatenate(collected)[:total]


# ----------------------------------------------------------------- commands


def cmd_enroll(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    if args.source:
        pcm = read_wav(args.source)
    else:
        print(f"Speak normally for {args.seconds:.0f} seconds...")
        pcm = record(args.device, args.seconds)
        keep = _ROOT / "recordings" / f"enroll_{args.name}.wav"
        sf.write(str(keep), pcm, _RATE, subtype="PCM_16")
        print(f"  kept the recording at {keep}")

    floor = background_db(pcm)
    starts = range(0, len(pcm) - _WINDOW + 1, _HOP * 2)
    windows = [pcm[s : s + _WINDOW] for s in starts]
    windows = [w for w in windows if is_speech(w, floor)]
    if len(windows) < _MIN_ENROLL_WINDOWS:
        print(
            f"error: found only {len(windows)} windows of speech; need {_MIN_ENROLL_WINDOWS}. "
            "Speak closer to the board or for longer.",
            file=sys.stderr,
        )
        return 1

    prints = embed(model, np.stack(windows))
    voice = unit(prints.mean(axis=0))
    out = voices_dir(args.model) / f"{args.name}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, voice)

    consistency = float(np.mean(prints @ voice))
    print(f"Enrolled {args.name!r} from {len(windows)} speech windows (self-similarity {consistency:.2f}).")
    for other, vector in load_voices(args.model).items():
        if other != args.name:
            print(f"  similarity to {other}: {float(voice @ vector):.2f}")
    return 0


def cmd_voices(args: argparse.Namespace) -> int:
    voices = load_voices(args.model)
    if not voices:
        print(f"No voices enrolled for model {args.model!r}.")
    for name in voices:
        print(name)
    return 0


class Matcher:
    """Turns a stream of 1.5 s windows into who-is-speaking decisions."""

    def __init__(self, model: spk.Student, voices: dict[str, np.ndarray], threshold: float):
        self.model = model
        self.names = list(voices)
        self.voices = np.stack([voices[name] for name in self.names])
        self.threshold = threshold
        self.recent: collections.deque[np.ndarray] = collections.deque(maxlen=_SMOOTH_WINDOWS)

    def step(self, window: np.ndarray, floor_db: float) -> tuple[str, dict[str, float]]:
        """Returns (label, score per voice). label is a name, 'other' or 'silence'."""
        if not is_speech(window, floor_db):
            self.recent.clear()
            return "silence", {}
        self.recent.append(embed(self.model, window[None])[0])
        scores = self.voices @ unit(np.mean(self.recent, axis=0))
        best = int(np.argmax(scores))
        label = self.names[best] if scores[best] >= self.threshold else "other"
        return label, dict(zip(self.names, map(float, scores)))


def _require_voices(model_name: str) -> dict[str, np.ndarray]:
    voices = load_voices(model_name)
    if not voices:
        raise SystemExit("error: no voices enrolled. Run: uv run tools/host/live.py enroll NAME")
    return voices


def _format_scores(scores: dict[str, float]) -> str:
    return "  ".join(f"{name} {value:+.2f}" for name, value in scores.items())


def cmd_score(args: argparse.Namespace) -> int:
    matcher = Matcher(load_model(args.model), _require_voices(args.model), args.threshold)
    pcm = read_wav(args.wav)
    floor = background_db(pcm)

    spans: list[list] = []  # [label, start_s, end_s]
    for start in range(0, len(pcm) - _WINDOW + 1, _HOP):
        label, scores = matcher.step(pcm[start : start + _WINDOW], floor)
        end = (start + _WINDOW) / _RATE
        if args.verbose:
            print(f"{end:7.2f}s  {label:<10} {_format_scores(scores)}")
        if spans and spans[-1][0] == label:
            spans[-1][2] = end
        else:
            spans.append([label, end - _HOP / _RATE, end])

    totals: dict[str, float] = collections.defaultdict(float)
    for label, begin, end in spans:
        totals[label] += end - begin
        if not args.verbose and end - begin >= args.min_span:
            print(f"{begin:7.1f}s - {end:6.1f}s  {label}")
    print("Totals: " + ", ".join(f"{label} {seconds:.1f}s" for label, seconds in sorted(totals.items())))
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    matcher = Matcher(load_model(args.model), _require_voices(args.model), args.threshold)
    blocks: "queue.Queue[np.ndarray]" = queue.Queue()
    buffer = np.zeros(_WINDOW, dtype=np.int16)
    # Background level: quietest tenth of the last 10 s of 10 ms frames.
    levels: collections.deque[float] = collections.deque(maxlen=1000)
    filled = 0
    current = None
    began = time.time()
    heard: list[np.ndarray] = []

    print(f"Listening for: {', '.join(matcher.names)}   (threshold {args.threshold:.2f}, Ctrl-C to stop)")
    try:
        with open_stream(args.device, blocks):
            while True:
                block = blocks.get()
                heard.append(block)
                buffer = np.concatenate([buffer[len(block) :], block])
                filled += len(block)
                levels.extend(frame_levels_db(block))
                if filled < _WINDOW:
                    continue
                floor = float(np.percentile(levels, 10))
                label, scores = matcher.step(buffer, floor)

                shown = {"silence": "...", "other": "someone else"}.get(label, label.upper())
                stamp = time.time() - began
                if label != current:
                    # Keep a line for each change so the history stays on screen.
                    print(f"\r{stamp:7.1f}s  {shown:<14} {_format_scores(scores):<60}")
                    current = label
                print(f"\r{stamp:7.1f}s  {shown:<14} {_format_scores(scores):<60}", end="", flush=True)
    except KeyboardInterrupt:
        print()
    if heard:
        # Keep the audio so a session can be replayed with `score`.
        kept = _ROOT / "recordings" / "listen_last.wav"
        sf.write(str(kept), np.concatenate(heard), _RATE, subtype="PCM_16")
        print(f"Saved what was heard to {kept}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="v1", help="Name of a model under data/models (default: v1)")
    sub = parser.add_subparsers(dest="command", required=True)

    enroll = sub.add_parser("enroll", help="Learn a voice")
    enroll.add_argument("name")
    enroll.add_argument("--from", dest="source", type=pathlib.Path, help="Enroll from a 16 kHz WAV instead of the mic")
    enroll.add_argument("-t", "--seconds", type=float, default=15.0)
    enroll.add_argument("-d", "--device", default=_DEFAULT_DEVICE)
    enroll.set_defaults(run=cmd_enroll)

    listen = sub.add_parser("listen", help="Live voice match")
    listen.add_argument("--threshold", type=float, default=_MATCH_THRESHOLD)
    listen.add_argument("-d", "--device", default=_DEFAULT_DEVICE)
    listen.set_defaults(run=cmd_listen)

    score = sub.add_parser("score", help="Voice match over a WAV file")
    score.add_argument("wav", type=pathlib.Path)
    score.add_argument("--threshold", type=float, default=_MATCH_THRESHOLD)
    score.add_argument("--min-span", type=float, default=0.5, help="Hide spans shorter than this many seconds")
    score.add_argument("-v", "--verbose", action="store_true", help="Print every 0.25 s decision with scores")
    score.set_defaults(run=cmd_score)

    voices = sub.add_parser("voices", help="List enrolled voices")
    voices.set_defaults(run=cmd_voices)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
