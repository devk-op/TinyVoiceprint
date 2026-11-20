#!/usr/bin/env python3
"""Reads the firmware's timing and self-test counters off the board.

The board has no log output, so the app keeps its measurements in a global
struct (DeviceMlPerf in modules/app/main.cc) and this script reads that
memory over SWD with OpenOCD while the firmware keeps running.

Usage:
  bazel build --config=stm32f407g_disc1 //modules/app
  targets/stm32f407g_disc1/py/flash.py bazel-bin/modules/app/app
  tools/host/read_perf.py                 # after it has run for a few seconds
  tools/host/read_perf.py --enroll        # then speak for about 10 seconds
  tools/host/read_perf.py --watch 20      # live voice-match state for 20 s
"""

import argparse
import pathlib
import re
import struct
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_OPENOCD_CFG = _ROOT / "targets" / "stm32f407g_disc1" / "py" / "openocd_stm32f407g_disc1.cfg"
# OpenOCD sets DBGMCU_CR's DBG_SLEEP/STOP/STANDBY bits when it connects, which
# keep the core clocks running while the chip sleeps. Only a power cycle clears
# them, so clear them before disconnecting or current readings come out high.
_RELEASE_SLEEP = "mww 0xE0042004 0"
_SYMBOL = "device_ml_perf"

# Must match DeviceMlPerf in modules/app/main.cc.
_FIELDS = (
    "selftest_frames",
    "selftest_max_error",  # float
    "selftest_cycles",
    "cpu_hz",
    "blocks",
    "frames",
    "decimate_min",
    "decimate_max",
    "frame_min",
    "frame_max",
    "decimate_sum_low",
    "decimate_sum_high",
    "features_sum_low",
    "features_sum_high",
    "voiceprint_input_ok",
    "voiceprint_conv_ok",
    "voiceprint_max_error",  # float
    "voiceprint_similarity",  # float
    "voiceprint_cycles",
)
# Must match DeviceMlVoice in modules/app/main.cc.
_VOICE_SYMBOL = "device_ml_voice"
_VOICE_FIELDS = (
    "command",
    "threshold",  # float
    "result",
    "score",  # float
    "enrolled",
    "enroll_windows",
    "steps",
    "speech_steps",
    "match_steps",
    "step_cycles_last",
    "step_cycles_max",
    "enroll_failures",
    "mode",
    "network_runs",
    "stored",
    "store_free_slots",
)
# DeviceMlVoice::stored, in order.
_STORED = ("not saved", "loaded from flash at boot", "saved to flash", "cleared from flash", "FLASH WRITE FAILED")
# VoiceMatch::Result, in order.
_RESULTS = ("not ready", "silence", "speech", "someone else", "MATCH", "enrolling", "enrolled", "enroll failed")
_BLOCK_SECONDS = 0.002
_EXPECTED_SELFTEST_FRAMES = 18
_SELFTEST_TOLERANCE = 1e-4


def _symbol_address(elf: pathlib.Path, symbol: str = _SYMBOL) -> int:
    out = subprocess.run(["nm", str(elf)], check=True, capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == symbol:
            return int(parts[0], 16)
    raise SystemExit(f"error: {symbol} not found in {elf}; is this the current app build?")


def _as_float(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _write_word(address: int, value: int) -> None:
    result = subprocess.run(
        ["openocd", "-f", str(_OPENOCD_CFG), "-c", "init", "-c", f"mww 0x{address:08x} 0x{value:08x}", "-c", _RELEASE_SLEEP, "-c", "shutdown"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"error: could not write to the board over SWD:\n{result.stderr.strip()}")


def _voice(elf: pathlib.Path) -> dict:
    words = _read_words(_symbol_address(elf, _VOICE_SYMBOL), len(_VOICE_FIELDS))
    voice = dict(zip(_VOICE_FIELDS, words))
    voice["threshold"] = _as_float(voice["threshold"])
    voice["score"] = _as_float(voice["score"])
    return voice


def _voice_line(voice: dict, hz: int) -> str:
    result = _RESULTS[voice["result"]] if voice["result"] < len(_RESULTS) else "?"
    last = 1000.0 * voice["step_cycles_last"] / hz
    worst = 1000.0 * voice["step_cycles_max"] / hz
    return (
        f"{result:<14} score {voice['score']:+.2f} (threshold {voice['threshold']:.2f})   "
        f"enrolled {'yes' if voice['enrolled'] else 'no'} ({voice['enroll_windows']} windows)   "
        f"decisions {voice['steps']}, with speech {voice['speech_steps']}, matched {voice['match_steps']}   "
        f"network {last:.0f} ms (max {worst:.0f} ms)   "
        f"voiceprint {_STORED[voice['stored']] if voice['stored'] < len(_STORED) else '?'}"
    )


def _read_words(address: int, count: int) -> list[int]:
    result = subprocess.run(
        ["openocd", "-f", str(_OPENOCD_CFG), "-c", "init", "-c", f"mdw 0x{address:08x} {count}", "-c", _RELEASE_SLEEP, "-c", "shutdown"],
        capture_output=True,
        text=True,
    )
    words = []
    for line in (result.stdout + result.stderr).splitlines():
        match = re.match(r"^0x[0-9a-f]{8}:((?: [0-9a-f]{8})+)\s*$", line)
        if match:
            words += [int(word, 16) for word in match.group(1).split()]
    if len(words) != count:
        raise SystemExit(f"error: could not read the board over SWD:\n{result.stderr.strip()}")
    return words


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("elf", nargs="?", type=pathlib.Path, default=_ROOT / "bazel-bin" / "modules" / "app" / "app")
    parser.add_argument("--enroll", action="store_true", help="Start enrollment (same as pressing the user button)")
    parser.add_argument("--clear", action="store_true", help="Forget the enrolled voice")
    parser.add_argument("--threshold", type=float, help="Set the match threshold")
    parser.add_argument(
        "--mode",
        type=int,
        choices=(0, 1, 2),
        help="Energy test mode: 0 normal, 1 listen only (no network), 2 network back to back",
    )
    parser.add_argument("--rate", type=float, metavar="SECONDS", help="Count network runs over this long and print the rate")
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="Print the voice-match state repeatedly for this long")
    args = parser.parse_args()

    voice_address = _symbol_address(args.elf, _VOICE_SYMBOL)
    if args.enroll or args.clear:
        _write_word(voice_address, 1 if args.enroll else 2)
        print("Enrollment started: speak for about 10 seconds." if args.enroll else "Enrolled voice cleared.")
        return 0
    if args.mode is not None:
        _write_word(voice_address + 4 * _VOICE_FIELDS.index("mode"), args.mode)
        print(f"Mode set to {args.mode} ({['normal', 'listen only', 'network back to back'][args.mode]}).")
        return 0
    if args.rate:
        import time

        hz = _read_words(_symbol_address(args.elf) + 4 * _FIELDS.index("cpu_hz"), 1)[0]
        first = _voice(args.elf)
        began = time.time()
        time.sleep(args.rate)
        last = _voice(args.elf)
        elapsed = time.time() - began
        runs = last["network_runs"] - first["network_runs"]
        print(
            f"mode {last['mode']}: {runs} network runs in {elapsed:.1f} s = {runs / elapsed:.2f} per second; "
            f"last run {1000.0 * last['step_cycles_last'] / hz:.1f} ms"
        )
        return 0
    if args.threshold is not None:
        _write_word(voice_address + 4, struct.unpack("<I", struct.pack("<f", args.threshold))[0])
        print(f"Threshold set to {args.threshold:.2f}.")
        return 0
    if args.watch:
        import time

        hz = _read_words(_symbol_address(args.elf) + 4 * _FIELDS.index("cpu_hz"), 1)[0]
        # Each read takes a fraction of a second over SWD, so this view lags
        # the LEDs a little. Only changes are printed.
        began = time.time()
        shown = None
        try:
            while time.time() - began < args.watch:
                voice = _voice(args.elf)
                key = (voice["result"], round(voice["score"], 2), voice["enroll_windows"])
                if key != shown:
                    print(f"{time.time() - began:5.1f}s  {_voice_line(voice, hz)}", flush=True)
                    shown = key
        except KeyboardInterrupt:
            pass
        return 0

    words = _read_words(_symbol_address(args.elf), len(_FIELDS))
    perf = dict(zip(_FIELDS, words))
    perf["selftest_max_error"] = struct.unpack("<f", struct.pack("<I", perf["selftest_max_error"]))[0]
    for key in ("voiceprint_max_error", "voiceprint_similarity"):
        perf[key] = struct.unpack("<f", struct.pack("<I", perf[key]))[0]
    decimate_sum = perf["decimate_sum_low"] | (perf["decimate_sum_high"] << 32)
    features_sum = perf["features_sum_low"] | (perf["features_sum_high"] << 32)

    hz = perf["cpu_hz"]
    if hz == 0 or perf["blocks"] == 0:
        raise SystemExit("error: counters are empty; the firmware hasn't started processing audio yet")
    elapsed = perf["blocks"] * _BLOCK_SECONDS
    cycles_available = elapsed * hz

    def ms(cycles: float) -> float:
        return 1000.0 * cycles / hz

    selftest_ok = (
        perf["selftest_frames"] == _EXPECTED_SELFTEST_FRAMES and perf["selftest_max_error"] < _SELFTEST_TOLERANCE
    )
    print(f"CPU clock:            {hz / 1e6:.0f} MHz")
    print(f"Running for:          {elapsed:.1f} s of audio ({perf['blocks']} blocks, {perf['frames']} feature frames)")
    print()
    print("On-chip feature self-test (vs Python reference)")
    print(f"  frames:             {perf['selftest_frames']} (expected {_EXPECTED_SELFTEST_FRAMES})")
    print(f"  max abs error:      {perf['selftest_max_error']:.2e} log units  ->  {'PASS' if selftest_ok else 'FAIL'}")
    print(f"  cost, idle system:  {ms(perf['selftest_cycles'] / max(1, perf['selftest_frames'])):.3f} ms per frame")
    print()
    voiceprint_ok = (
        perf["voiceprint_input_ok"] == 1 and perf["voiceprint_conv_ok"] == 1 and perf["voiceprint_max_error"] < 1e-4
    )
    exact = "identical" if perf["voiceprint_input_ok"] == 1 and perf["voiceprint_conv_ok"] == 1 else "DIFFERENT"
    print("On-chip voiceprint network self-test (vs Python integer reference)")
    print(f"  8-bit input and convolution output: {exact}")
    print(
        f"  voiceprint:         max abs error {perf['voiceprint_max_error']:.2e}, "
        f"similarity {perf['voiceprint_similarity']:.6f}  ->  {'PASS' if voiceprint_ok else 'FAIL'}"
    )
    window_ms = ms(perf["voiceprint_cycles"])
    print(f"  cost, idle system:  {window_ms:.1f} ms per 1.5 s window")
    print()
    print("Live audio path")
    print(
        f"  PDM decimation:     {100 * decimate_sum / cycles_available:5.2f}% CPU   "
        f"(per 2 ms block: min {ms(perf['decimate_min']):.3f} ms, max {ms(perf['decimate_max']):.3f} ms)"
    )
    print(
        f"  log-mel features:   {100 * features_sum / cycles_available:5.2f}% CPU   "
        f"(per frame: min {ms(perf['frame_min']):.3f} ms, max {ms(perf['frame_max']):.3f} ms)"
    )
    left = 1 - (decimate_sum + features_sum) / cycles_available
    print(f"  left for the model: {100 * left:5.2f}% CPU (before USB and RTOS overhead)")
    print()
    print("Voice match")
    print("  " + _voice_line(_voice(args.elf), hz))
    return 0 if selftest_ok and voiceprint_ok else 1


if __name__ == "__main__":
    sys.exit(main())
