#!/usr/bin/env python3
"""Flashes a built ELF to the STM32F407G-DISC1 over OpenOCD/ST-LINK.

Usage:
  bazel build --config=stm32f407g_disc1 //modules/app
  targets/stm32f407g_disc1/py/flash.py bazel-bin/modules/app/app
"""

import argparse
import pathlib
import subprocess
import sys

_CFG = pathlib.Path(__file__).parent / "openocd_stm32f407g_disc1.cfg"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=pathlib.Path, help="Path to the built ELF file")
    args = parser.parse_args()

    if not args.elf.is_file():
        print(f"error: {args.elf} does not exist", file=sys.stderr)
        return 1

    cmd = [
        "openocd",
        "-f",
        str(_CFG),
        "-c",
        f"program {args.elf} verify reset",
        # OpenOCD sets DBGMCU_CR's debug-in-sleep bits on connect, which keep
        # the core clocks on while the chip sleeps. Clear them so the chip's
        # current is the real figure without needing a power cycle.
        "-c",
        "mww 0xE0042004 0",
        "-c",
        "shutdown",
    ]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
