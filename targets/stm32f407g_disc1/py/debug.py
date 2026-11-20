#!/usr/bin/env python3
"""Starts an OpenOCD/GDB debug session against the STM32F407G-DISC1.

Without any real logging path wired up yet (see target_docs.rst -- this
board's ST-LINK doesn't appear to bridge a UART to the target), GDB over
SWD is the primary way to observe device behavior during development:
breakpoints, register/memory reads, single-stepping.

Starts OpenOCD in the background (gdbserver on :3333) and drops you into an
interactive arm-none-eabi-gdb session attached to it, with breakpoints
pre-armed at the application entry point and the fault handler.

Usage:
  bazel build --config=stm32f407g_disc1 //modules/app
  targets/stm32f407g_disc1/py/debug.py bazel-bin/modules/app/app
"""

import argparse
import atexit
import pathlib
import subprocess
import sys
import time

_CFG = pathlib.Path(__file__).parent / "openocd_stm32f407g_disc1.cfg"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=pathlib.Path, help="Path to the built ELF file")
    args = parser.parse_args()

    if not args.elf.is_file():
        print(f"error: {args.elf} does not exist", file=sys.stderr)
        return 1

    ocd = subprocess.Popen(
        ["openocd", "-f", str(_CFG)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(ocd.terminate)
    time.sleep(1)  # let OpenOCD open the gdbserver port before GDB connects.

    gdb_commands = [
        "target extended-remote localhost:3333",
        "monitor reset halt",
        "break pw::system::UserAppInit",
        "break DefaultFaultHandler",
        "echo \\n--- Breakpoints armed at UserAppInit and DefaultFaultHandler.\\n"
        "--- 'monitor reset run' then 'continue' to boot; 'bt'/'info registers' to inspect.\\n",
    ]
    cmd = ["arm-none-eabi-gdb", str(args.elf)]
    for c in gdb_commands:
        cmd += ["-ex", c]

    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
