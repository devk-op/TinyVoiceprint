"""Real noise, music, crowd babble and measured room echo for training.

spk.py's default augmentation uses synthetic noise, one interfering voice
and simulated echo. With `spk.py train --real-noise` it uses this instead:

  noise    MUSAN noise recordings (openslr.org/17, CC BY 4.0), mixed at
           -5 to 20 dB SNR, so sometimes louder than the speech
  music    MUSAN music, 0 to 20 dB SNR
  babble   1 to 5 other training voices together, 0 to 20 dB SNR
  echo     measured and simulated room impulse responses (openslr.org/28,
           Apache 2.0), applied to the features as below; the simulated
           room and device effects of spk.room_effects stay in the mix

Everything works on log-mel features, as spk.augment does: uncorrelated
sounds add in power, and a room's echo is applied per mel band as a
convolution over frames with the impulse response's energy in that band
per 10 ms frame. Phase within a frame is ignored, which is what makes it
cheap enough to do per batch.

Prepare once (reads data/noise/musan and data/noise/RIRS_NOISES):
  uv run tools/train/spk.py prepare-noise
"""

import pathlib

import numpy as np
import soundfile as sf
import torch

import log_mel_reference as ref

_ROOT = pathlib.Path(__file__).resolve().parents[2]
NOISE_DIR = _ROOT / "data" / "noise"
BANK = NOISE_DIR / "bank.npz"
_RIR_FRAMES = 80  # 0.8 s of echo tail.
_MUSIC_HOURS = 8.0  # MUSAN has 42 h of music; a sample is enough.


def _log_mel(pcm: np.ndarray, log_mel_torch) -> np.ndarray:
    out = []
    for start in range(0, len(pcm), 16000 * 60):  # A minute at a time.
        piece = pcm[start : start + 16000 * 60 + ref.FRAME_LENGTH]
        if len(piece) >= ref.FRAME_LENGTH:
            out.append(log_mel_torch(torch.from_numpy(piece.astype(np.float32))[None])[0].numpy())
    return np.concatenate(out) if out else np.zeros((0, ref.NUM_MELS), np.float32)


def _read(path: pathlib.Path) -> np.ndarray:
    pcm, rate = sf.read(str(path), dtype="float32", always_2d=True)
    pcm = pcm[:, 0]
    if rate != ref.SAMPLE_RATE:
        raise SystemExit(f"error: {path} is {rate} Hz, expected {ref.SAMPLE_RATE}")
    return pcm * 32768.0


def _rir_kernel(rir: np.ndarray) -> np.ndarray | None:
    """[_RIR_FRAMES, mels] energy per band per frame, direct sound = 1."""
    rir = rir[int(np.argmax(np.abs(rir))) :]  # Start at the direct sound.
    rir = np.pad(rir, (0, _RIR_FRAMES * ref.HOP))[: _RIR_FRAMES * ref.HOP]
    bank = ref.mel_filterbank()  # [mels, bins]
    frames = rir.reshape(_RIR_FRAMES, ref.HOP)
    power = np.abs(np.fft.rfft(frames, n=ref.FFT_SIZE)) ** 2 @ bank.T  # [frames, mels]
    if not np.all(power[0] > 0):
        return None
    return (power / power[0]).astype(np.float32)


def prepare(log_mel_torch, rng_seed: int = 0) -> None:
    rng = np.random.default_rng(rng_seed)
    musan = NOISE_DIR / "musan"
    rirs = NOISE_DIR / "RIRS_NOISES"
    for needed in (musan, rirs):
        if not needed.is_dir():
            raise SystemExit(f"error: {needed} missing; unpack the openslr downloads there")

    noise = [_log_mel(_read(p), log_mel_torch) for p in sorted((musan / "noise").rglob("*.wav"))]
    music_files = sorted((musan / "music").rglob("*.wav"))
    rng.shuffle(music_files)
    music, frames = [], 0
    for p in music_files:
        if frames >= _MUSIC_HOURS * 3600 * 100:
            break
        music.append(_log_mel(_read(p), log_mel_torch))
        frames += len(music[-1])

    real = [p for p in sorted((rirs / "real_rirs_isotropic_noises").glob("*.wav")) if "noise" not in p.name]
    simulated = sorted((rirs / "simulated_rirs").rglob("*.wav"))
    chosen = real + [simulated[i] for i in rng.choice(len(simulated), size=min(3000, len(simulated)), replace=False)]
    kernels = [k for k in (_rir_kernel(_read(p) / 32768.0) for p in chosen) if k is not None]

    def pack(parts):
        lengths = np.array([len(x) for x in parts])
        keep = lengths >= 200  # At least 2 s.
        parts = [x for x, k in zip(parts, keep) if k]
        return np.concatenate(parts).astype(np.float32), np.cumsum([0] + [len(x) for x in parts])

    noise_feats, noise_offsets = pack(noise)
    music_feats, music_offsets = pack(music)
    np.savez(
        BANK,
        noise=noise_feats,
        noise_offsets=noise_offsets,
        music=music_feats,
        music_offsets=music_offsets,
        rirs=np.stack(kernels),
        real_rirs=np.int32(len(real)),
    )
    print(
        f"noise: {len(noise_offsets) - 1} files, {len(noise_feats) / 360000:.1f} h; "
        f"music: {len(music_offsets) - 1} files, {len(music_feats) / 360000:.1f} h; "
        f"room responses: {len(kernels)} ({len(real)} measured). Wrote {BANK}"
    )


class Bank:
    def __init__(self):
        if not BANK.exists():
            raise SystemExit(f"error: {BANK} missing; run: spk.py prepare-noise")
        data = np.load(BANK)
        self.noise, self.noise_offsets = data["noise"], data["noise_offsets"]
        self.music, self.music_offsets = data["music"], data["music_offsets"]
        self.rirs = data["rirs"]

    @staticmethod
    def _crops(feats, offsets, rng, batch: int, frames: int) -> np.ndarray:
        files = rng.integers(0, len(offsets) - 1, batch)
        out = np.empty((batch, frames, feats.shape[1]), np.float32)
        for i, f in enumerate(files):
            begin, end = offsets[f], offsets[f + 1]
            length = end - begin
            if length >= frames:
                s = begin + rng.integers(0, length - frames + 1)
                out[i] = feats[s : s + frames]
            else:  # Loop short files.
                idx = begin + (np.arange(frames) % length)
                out[i] = feats[idx]
        return out

    def noise_crops(self, rng, batch, frames):
        return self._crops(self.noise, self.noise_offsets, rng, batch, frames)

    def music_crops(self, rng, batch, frames):
        return self._crops(self.music, self.music_offsets, rng, batch, frames)

    def echo(self, crops: np.ndarray, rng) -> np.ndarray:
        """Convolves each item's mel powers over time with a random room."""
        kernels = self.rirs[rng.integers(0, len(self.rirs), len(crops))]  # [b, K, mels]
        power = np.exp(crops.astype(np.float64))
        n = crops.shape[1] + kernels.shape[1]
        spectrum = np.fft.rfft(power, n=n, axis=1) * np.fft.rfft(kernels, n=n, axis=1)
        echoed = np.fft.irfft(spectrum, n=n, axis=1)[:, : crops.shape[1]]
        return np.log(np.maximum(echoed, 0.0) + ref.LOG_FLOOR).astype(np.float32)


def augment(crops: np.ndarray, rng: np.random.Generator, bank: Bank, mix, room_effects) -> np.ndarray:
    """Room first (40% measured or simulated room responses, 40% the
    simulated room and device effects, 20% none), then interference: 25%
    clean, 30% noise, 10% music, 35% babble of 1-5 voices."""
    batch, frames = len(crops), crops.shape[1]
    crops = crops.copy()
    r = rng.random(batch)
    measured = np.flatnonzero(r < 0.4)
    simulated = np.flatnonzero((r >= 0.4) & (r < 0.8))
    if len(measured):
        crops[measured] = bank.echo(crops[measured], rng)
    if len(simulated):
        crops[simulated] = room_effects(crops[simulated], rng)

    out = crops.copy()
    c = rng.random(batch)
    noisy = np.flatnonzero((c >= 0.25) & (c < 0.55))
    if len(noisy):
        out[noisy] = mix(crops[noisy], bank.noise_crops(rng, len(noisy), frames), rng.uniform(-5.0, 20.0, len(noisy)))
    musical = np.flatnonzero((c >= 0.55) & (c < 0.65))
    if len(musical):
        out[musical] = mix(crops[musical], bank.music_crops(rng, len(musical), frames), rng.uniform(0.0, 20.0, len(musical)))
    babble = np.flatnonzero(c >= 0.65)
    if len(babble):
        voices = rng.integers(1, 6, len(babble))
        crowd = np.empty((len(babble), frames, crops.shape[2]), np.float32)
        for i, count in enumerate(voices):
            pool = np.delete(np.arange(batch), babble[i])  # Never the speaker themself.
            others = rng.choice(pool, size=count, replace=False)
            crowd[i] = np.log(np.exp(crops[others].astype(np.float64)).sum(axis=0) + ref.LOG_FLOOR)
        out[babble] = mix(crops[babble], crowd, rng.uniform(0.0, 20.0, len(babble)))
    return out
