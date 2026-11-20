# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "soundfile", "torch<2.9", "librosa", "ai-edge-litert"]
# ///
"""Scores Duffy's (2020) published microcontroller models on our tests.

T. Duffy, "Edge Device Speaker Verification", master's thesis, City College
of New York, 2020; code and trained models at
github.com/tduffy000/ccny-masters-thesis, cloned to data/external/duffy_thesis.
The thesis reports EER on LibriSpeech and VCTK only; this puts the same
models on VoxCeleb1-O and the board replay, next to ours.

Features follow his tensorflow/features.py and main.py: librosa voice
activity split at top_db=30, 1.2 s windows with 50% overlap inside each
voiced interval, a 40-band mel power spectrogram (n_fft 512, 25 ms frames,
10 ms hop) and log10(S + 1e-6), no mean normalisation. His code ran on 2020
librosa, whose framing padded by reflection, so pad_mode="reflect" is set
explicitly. A clip's voiceprint is the normalised mean of its windows'.
A clip with no voiced interval of 1.2 s falls back to windows over the
whole clip (our addition; his pipeline drops such clips).

Usage:
  uv run tools/train/duffy_eval.py --check   # LibriSpeech test-clean sanity check
  uv run tools/train/duffy_eval.py           # VoxCeleb1-O and replay voiceprints
then tools/train/stats.py duffy_32k ... as for our models.
"""

import argparse
import pathlib
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spk  # noqa: E402
import vox_eval  # noqa: E402

_REPO = spk._ROOT / "data" / "external" / "duffy_thesis" / "tensorflow" / "frozen_models" / "tiny"
# The three convolutional models of the thesis, by serialized size. The
# 32 kB one is the model compiled into his Arduino sketch.
MODELS = {
    "duffy_20k": _REPO / "1606700489" / "model.tflite",
    "duffy_32k": _REPO / "1605672336" / "model.tflite",
    "duffy_108k": _REPO / "1605647312" / "quant_model.tflite",
}
_SR = 16000
_WINDOW = int(1.2 * _SR)
_HOP = _WINDOW // 2


def windows(pcm: np.ndarray) -> list[np.ndarray]:
    y = pcm.astype(np.float32) / 32768.0
    out = []
    for left, right in librosa.effects.split(y, top_db=30):
        if right - left >= _WINDOW:
            out += list(librosa.util.frame(y[left:right], frame_length=_WINDOW, hop_length=_HOP, axis=0))
    if not out:
        if len(y) < _WINDOW:
            y = np.pad(y, (0, _WINDOW - len(y)))
        out = list(librosa.util.frame(y, frame_length=_WINDOW, hop_length=_HOP, axis=0))
    return out


def features(window: np.ndarray) -> np.ndarray:
    s = librosa.feature.melspectrogram(
        y=np.ascontiguousarray(window), sr=_SR, n_fft=512, n_mels=40, win_length=400, hop_length=160, pad_mode="reflect"
    )
    return np.log10(s + 1e-6).astype(np.float32)  # [40, 121]


class Model:
    def __init__(self, path: pathlib.Path):
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]["index"]
        self.output = self.interpreter.get_output_details()[0]["index"]

    def __call__(self, feature: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input, feature[None])
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output)[0].copy()


def clip_prints(models: dict[str, Model], audio: list[np.ndarray]) -> dict[str, np.ndarray]:
    prints = {name: [] for name in models}
    for i, pcm in enumerate(audio):
        feats = [features(w) for w in windows(pcm)]
        for name, model in models.items():
            e = np.stack([model(f) for f in feats]).mean(axis=0)
            prints[name].append(e / (np.linalg.norm(e) + 1e-12))
        if i % 500 == 0:
            print(f"  {i}/{len(audio)} clips", flush=True)
    return {name: np.stack(p) for name, p in prints.items()}


def check(models: dict[str, Model]) -> None:
    """All pairs of 10 utterances from each LibriSpeech test-clean speaker."""
    root = spk._ROOT / "data" / "librispeech" / "LibriSpeech" / "test-clean"
    rng = np.random.default_rng(0)
    paths, speakers = [], []
    for speaker in sorted(root.iterdir()):
        files = sorted(speaker.glob("*/*.flac"))
        for f in rng.choice(files, size=min(10, len(files)), replace=False):
            paths.append(f)
            speakers.append(speaker.name)
    audio = [sf.read(str(p), dtype="int16")[0] for p in paths]
    speakers = np.array(speakers)
    a, b = np.triu_indices(len(paths), k=1)
    same = speakers[a] == speakers[b]
    print(f"LibriSpeech test-clean: {len(set(speakers))} speakers, {len(paths)} utterances, {len(a)} pairs")
    for name, prints in clip_prints(models, audio).items():
        scores = np.einsum("ij,ij->i", prints[a], prints[b])
        print(f"  {name:<12} EER {100 * spk.equal_error_rate(scores, same):.2f}%  (thesis: 10.7-11.1% on LibriSpeech test)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    models = {name: Model(path) for name, path in MODELS.items()}
    if args.check:
        check(models)
        return 0

    same, clips, a, b = vox_eval.read_trials()
    for name, prints in clip_prints(models, vox_eval.load_audio(clips)).items():
        vox_eval.save_prints(name, "vox1o", prints)
        vox_eval.report(f"{name} VoxCeleb1-O", prints, a, b, same)
    same, clips, paths, a, b = vox_eval.replay_trials()
    for test, audio in (("replay_original", vox_eval.load_audio(clips)), ("replay_board", vox_eval.load_audio(clips, paths))):
        for name, prints in clip_prints(models, audio).items():
            vox_eval.save_prints(name, test, prints)
            vox_eval.report(f"{name} {test}", prints, a, b, same)
    return 0


if __name__ == "__main__":
    sys.exit(main())
