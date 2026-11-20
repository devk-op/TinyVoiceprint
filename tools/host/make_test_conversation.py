# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "soundfile"]
# ///
"""Builds synthetic multi-speaker test audio with known ground truth.

Uses macOS text-to-speech voices as stand-in speakers so the diarization,
identification and separation steps in voices.py can be checked against a
known answer before trusting them on real recordings.

Writes to recordings/synthetic/:
  enroll_<name>.wav     one clip per speaker, for `voices.py enroll`
  conversation.wav      turn-taking conversation with pauses
  conversation.json     ground truth: who spoke when
  overlap.wav           two speakers talking at the same time
  overlap_<name>.wav    each speaker's clean part of overlap.wav

Usage:
  uv run tools/host/make_test_conversation.py
"""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

_RATE = 16000
_OUT = pathlib.Path(__file__).resolve().parents[2] / "recordings" / "synthetic"

# name -> macOS voice. Two female, two male, different accents.
_SPEAKERS = {
    "samantha": "Samantha",
    "daniel": "Daniel",
    "karen": "Karen",
    "rishi": "Rishi",
}

_ENROLL_TEXT = (
    "This is my voice sample for enrollment. I am reading a few sentences so "
    "the system can learn what I sound like. The quick brown fox jumps over "
    "the lazy dog, and then walks slowly back home across the quiet field."
)

# (speaker, text, pause after in seconds)
_CONVERSATION = [
    ("samantha", "Hi everyone, thanks for joining. Shall we start with the plan for this week?", 0.6),
    ("daniel", "Sure. I finished the firmware for the microphone yesterday, and it streams audio over USB now.", 0.5),
    ("karen", "That is great news. Did you manage to test it with a real recording?", 0.4),
    ("daniel", "Yes, I recorded ten seconds and the speech was very clear.", 0.7),
    ("rishi", "I have a question about the next step. Are we running the models on the laptop first?", 0.5),
    ("samantha", "Exactly. We start on the laptop, and later we move the small parts onto the chip.", 0.4),
    ("rishi", "Okay, that makes sense to me.", 0.3),
    ("karen", "I can bring some cafeteria noise recordings tomorrow, so we can try it in a harder setting.", 0.6),
    ("samantha", "Perfect. Let us meet again on Thursday and compare the results.", 0.5),
    ("daniel", "Thursday works for me. See you all then.", 0.3),
]

_OVERLAP = {
    "samantha": "I think we should measure the delay first, because that decides whether people can use it in a real conversation.",
    "daniel": "The second microphone will arrive next week, and then we can finally try to focus on one direction only.",
}


def _say(voice: str, text: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "tts.wav"
        subprocess.run(
            ["say", "-v", voice, "-o", str(path), f"--data-format=LEI16@{_RATE}", text],
            check=True,
        )
        audio, rate = sf.read(path, dtype="float32")
    assert rate == _RATE
    return audio


def _write(name: str, audio: np.ndarray) -> None:
    sf.write(_OUT / name, np.clip(audio, -1.0, 1.0), _RATE, subtype="PCM_16")


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)

    for name, voice in _SPEAKERS.items():
        _write(f"enroll_{name}.wav", _say(voice, _ENROLL_TEXT))

    pieces = [np.zeros(int(0.5 * _RATE), dtype=np.float32)]
    truth = []
    position = len(pieces[0])
    for name, text, pause in _CONVERSATION:
        speech = _say(_SPEAKERS[name], text)
        truth.append(
            {
                "speaker": name,
                "start": round(position / _RATE, 3),
                "end": round((position + len(speech)) / _RATE, 3),
            }
        )
        pieces += [speech, np.zeros(int(pause * _RATE), dtype=np.float32)]
        position += len(speech) + int(pause * _RATE)
    conversation = np.concatenate(pieces)
    _write("conversation.wav", conversation)
    (_OUT / "conversation.json").write_text(json.dumps(truth, indent=2) + "\n")

    parts = {name: _say(_SPEAKERS[name], text) for name, text in _OVERLAP.items()}
    length = max(len(part) for part in parts.values())
    mix = np.zeros(length, dtype=np.float32)
    for name, part in parts.items():
        padded = np.pad(part, (0, length - len(part)))
        _write(f"overlap_{name}.wav", padded)
        mix += padded
    _write("overlap.wav", 0.5 * mix)

    print(f"Wrote {len(list(_OUT.iterdir()))} files to {_OUT}")
    print(f"conversation.wav: {len(conversation) / _RATE:.1f}s, {len(truth)} turns, {len(_SPEAKERS)} speakers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
