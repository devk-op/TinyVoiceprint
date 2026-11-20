#!/usr/bin/env python3
"""Unpacks only the chosen clips from the Common Voice archive.

`tar -T LIST` compares every archive entry against every name in the list,
which for ~2 million entries and ~280,000 names takes most of a day. This
reads the archive once and looks names up in a set.

Usage (after spk.py cv-select):
  python3 tools/train/cv_unpack.py data/commonvoice/cv-27.0-en.tar.gz \\
      data/commonvoice/cv-corpus-27.0-2026-09-11/en/selected_clips.txt
"""

import pathlib
import sys
import tarfile
import time


def main() -> int:
    archive, listing = map(pathlib.Path, sys.argv[1:3])
    root = archive.parent
    wanted = set(listing.read_text().split())
    print(f"{len(wanted)} clips wanted", flush=True)
    start = time.time()
    seen = written = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            seen += 1
            if member.isfile() and member.name in wanted:
                out = root / member.name
                if not (out.exists() and out.stat().st_size == member.size):
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(tar.extractfile(member).read())
                written += 1
            if seen % 100_000 == 0:
                print(f"  {seen} entries read, {written} clips unpacked ({time.time() - start:.0f}s)", flush=True)
    print(f"Done: {written} of {len(wanted)} clips in {time.time() - start:.0f}s", flush=True)
    return 0 if written == len(wanted) else 1


if __name__ == "__main__":
    sys.exit(main())
