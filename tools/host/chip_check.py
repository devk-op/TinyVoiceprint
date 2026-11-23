# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "scipy", "soundfile", "sounddevice", "torch<2.9"]
# ///
"""Checks the chip's features and integer network against the Python reference
on many real windows, not just the boot self-test's one.

For each 1.5 s window of board-captured audio, the PCM is written over SWD to
the chip's CCM RAM and command 3 makes the firmware compute log-mel features
and run the network exactly as on live audio (main.cc, RunChipCheck). The
chip's int8-input hash, convolution-output hash and voiceprint are compared
with tools/train/tcs_integer.py run on features from the same samples, computed
by the same Python front end the off-chip evaluations use.

Usage (board connected, current firmware flashed):
  uv run tools/host/chip_check.py              # 200 replay + 40 pilot windows
  uv run tools/host/chip_check.py --replay 50 --pilot 0
"""

import argparse
import pathlib
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tools" / "train"))
sys.path.insert(0, str(_ROOT / "tools" / "host"))
import duration_eval  # noqa: E402
import read_perf  # noqa: E402
import spk  # noqa: E402

_ELF = _ROOT / "bazel-bin" / "modules" / "app" / "app"
_CCM = 0x10000000
_SAMPLES = 400 + 149 * 160  # Exactly 150 log-mel frames.
_DIM = 192
_TCL_PORT = 6667


def fnv1a(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


class OpenOcd:
    """One OpenOCD session over its Tcl port, so each window costs a few calls."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            ["openocd", "-f", str(read_perf._OPENOCD_CFG), "-c", f"tcl_port {_TCL_PORT}",
             "-c", "telnet_port disabled", "-c", "gdb_port disabled", "-c", "init"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):
            try:
                self.sock = socket.create_connection(("127.0.0.1", _TCL_PORT), timeout=10)
                return
            except OSError:
                time.sleep(0.2)
        raise SystemExit("error: OpenOCD did not start; is the board connected?")

    def cmd(self, text: str) -> str:
        self.sock.sendall(text.encode() + b"\x1a")
        out = b""
        while not out.endswith(b"\x1a"):
            chunk = self.sock.recv(65536)
            if not chunk:
                raise SystemExit("error: OpenOCD closed the connection")
            out += chunk
        return out[:-1].decode(errors="replace")

    def words(self, address: int, count: int) -> list[int]:
        text = self.cmd(f"capture {{mdw 0x{address:08x} {count}}}")
        words = []
        for line in text.splitlines():
            m = re.match(r"^0x[0-9a-f]{8}:((?: [0-9a-f]{8})+)\s*$", line.strip())
            if m:
                words += [int(w, 16) for w in m.group(1).split()]
        if len(words) != count:
            raise SystemExit(f"error: read {len(words)} of {count} words at 0x{address:08x}")
        return words

    def close(self) -> None:
        try:
            self.cmd(read_perf._RELEASE_SLEEP)
            self.cmd("shutdown")
        finally:
            self.proc.wait(timeout=10)


def windows(replay: int, pilot: int, seed: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = []
    files = sorted((_ROOT / "recordings" / "vox_replay").glob("*.wav"))
    for path in rng.choice(files, size=min(replay, len(files)), replace=False):
        pcm = sf.read(str(path), dtype="int16")[0]
        start = int(rng.integers(0, len(pcm) - _SAMPLES + 1))
        out.append((f"replay/{path.name}@{start}", pcm[start : start + _SAMPLES]))
    pilot_files = sorted((_ROOT / "recordings" / "study" / "P00").glob("*.wav"))
    for k in range(pilot):
        path = pilot_files[k % len(pilot_files)]
        pcm = sf.read(str(path), dtype="int16")[0]
        start = int(rng.integers(0, len(pcm) - _SAMPLES + 1))
        out.append((f"pilot/{path.name}@{start}", pcm[start : start + _SAMPLES]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=int, default=200, help="Windows from the replay recordings")
    parser.add_argument("--pilot", type=int, default=40, help="Windows from the pilot recordings")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    check = read_perf._symbol_address(_ELF, "device_ml_check")
    command = read_perf._symbol_address(_ELF, "device_ml_voice")  # `command` is the first field.
    model = duration_eval.load("tcs_noise.int")
    items = windows(args.replay, args.pilot, args.seed)
    ocd = OpenOcd()
    rows = []
    try:
        done = ocd.words(check, 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            blob = pathlib.Path(tmp) / "pcm.bin"
            for i, (name, pcm) in enumerate(items):
                blob.write_bytes(pcm.astype("<i2").tobytes())
                ocd.cmd(f"load_image {blob} 0x{_CCM:08x} bin")
                ocd.cmd(f"mww 0x{command:08x} 3")
                deadline = time.time() + 10
                while True:
                    words = ocd.words(check, 8 + _DIM)
                    if words[0] != done:
                        break
                    if time.time() > deadline:
                        raise SystemExit("error: the chip did not run the check; is the new firmware flashed?")
                    time.sleep(0.1)
                done = words[0]
                frames, in_hash, conv_hash, conv_bytes, cycles, in_addr = words[1:7]
                chip = np.array([struct.unpack("<f", struct.pack("<I", w))[0] for w in words[8:]], np.float32)

                feats = spk.log_mel_torch(torch.from_numpy(pcm.astype(np.int16))[None])[0].numpy()[None]
                q = model.quantize_input(feats)
                conv = model.convolutions(q)
                ref = model.embed(model.pool(conv))[0].astype(np.float32)
                row = {
                    "name": name,
                    "frames_ok": frames == 150 and feats.shape[1] == 150,
                    "input": in_hash == fnv1a(q.astype(np.int8).tobytes()),
                    "conv": conv_hash == fnv1a(conv.astype(np.uint8).tobytes()) and conv_bytes == conv.size,
                    "max_diff": float(np.max(np.abs(chip - ref))),
                    "cos": float(chip @ ref / (np.linalg.norm(chip) * np.linalg.norm(ref))),
                    "ms": 1000.0 * cycles / 168e6,
                }
                if not row["input"]:  # Count the int8 inputs that differ.
                    raw = ocd.words(in_addr, 150 * 40 // 4)
                    got = np.frombuffer(struct.pack(f"<{len(raw)}I", *raw), np.int8).astype(np.int64)
                    diff = got - q[0].ravel()
                    row["input_diffs"] = int(np.count_nonzero(diff))
                    row["input_maxdiff"] = int(np.max(np.abs(diff)))
                rows.append(row)
                flag = "ok" if row["input"] and row["conv"] else "MISMATCH"
                print(f"{i + 1:4d}/{len(items)} {flag:<8} max|diff| {row['max_diff']:.2e}  {name}", flush=True)
    finally:
        ocd.close()

    n = len(rows)
    inp = sum(r["input"] for r in rows)
    conv = sum(r["conv"] for r in rows)
    print(f"\n{n} windows: int8 input identical {inp}/{n}, convolution output identical {conv}/{n}, "
          f"frames correct {sum(r['frames_ok'] for r in rows)}/{n}")
    print(f"voiceprint max |chip - reference| {max(r['max_diff'] for r in rows):.2e}, "
          f"min cosine {min(r['cos'] for r in rows):.8f}, time {np.mean([r['ms'] for r in rows]):.1f} ms per window")
    bad = [r for r in rows if not r["input"]]
    if bad:
        print(f"input mismatches: {len(bad)}; int8 values differing per window: "
              f"{sorted(r['input_diffs'] for r in bad)}; max difference {max(r['input_maxdiff'] for r in bad)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
