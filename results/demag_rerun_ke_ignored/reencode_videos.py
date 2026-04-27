#!/usr/bin/env python
"""Re-encode every video under results/demag_rerun/videos/ to H.264 baseline
mp4 with yuv420p for maximum player compatibility.

Accepts both .mp4 and .webm inputs (anything ffmpeg can read).  Outputs
always land as .mp4 in the same folder; the original file is replaced in
place (backed up to *.orig once, not overwritten on re-run).

  ffmpeg -i <in> -c:v libx264 -profile:v baseline -level 3.0 \
         -pix_fmt yuv420p -preset veryfast -movflags +faststart \
         -crf 20 -an <out>.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent / "videos"

FFMPEG_ARGS = [
    "-c:v", "libx264",
    "-profile:v", "baseline",
    "-level", "3.0",
    "-pix_fmt", "yuv420p",
    "-preset", "veryfast",
    "-movflags", "+faststart",
    "-crf", "20",
    "-an",                # no audio stream
    "-y",                 # overwrite output
]


def reencode(src: Path, dst: Path) -> bool:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-i", str(src), *FFMPEG_ARGS, str(dst)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {src.name}: ffmpeg exit {e.returncode}")
        return False
    size = dst.stat().st_size
    if size < 10_000:
        print(f"[WARN] {dst.name}: output suspiciously small ({size} B)")
        return False
    print(f"[OK]   {dst.name}  ({size / 1024:.0f} KB)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                        help="directory containing videos to re-encode")
    parser.add_argument("--backup", action="store_true",
                        help="save original as <name>.orig before replacing")
    parser.add_argument("--patterns", nargs="+",
                        default=["*.mp4", "*.webm"],
                        help="glob patterns to include (default mp4+webm)")
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"[ERROR] not a directory: {args.dir}")
        sys.exit(1)

    sources: list[Path] = []
    for pat in args.patterns:
        sources.extend(sorted(args.dir.glob(pat)))
    # Skip already-backed-up originals.
    sources = [p for p in sources if not p.name.endswith(".orig")]
    if not sources:
        print(f"[info] nothing to re-encode in {args.dir}")
        return

    n_ok = 0
    for src in sources:
        dst = src.with_suffix(".mp4")
        tmp = dst.with_name(dst.name + ".tmp.mp4")

        if args.backup:
            orig = src.with_suffix(src.suffix + ".orig")
            if not orig.exists():
                shutil.copy2(src, orig)

        if not reencode(src, tmp):
            tmp.unlink(missing_ok=True)
            continue

        # atomic replace in place
        tmp.replace(dst)
        # if input had a different extension (e.g. .webm), drop the original
        if src != dst and src.exists():
            src.unlink()
        n_ok += 1

    print(f"\n{n_ok}/{len(sources)} re-encoded to H.264 baseline mp4 "
          f"in {args.dir}")


if __name__ == "__main__":
    main()
