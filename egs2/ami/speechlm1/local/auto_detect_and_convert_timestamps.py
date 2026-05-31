#!/usr/bin/env python3
"""
auto_detect_and_convert_timestamps.py

Batch version (multiple inputs):

- Detect timestamps inside brackets [] as either:
    * mm:ss(.d)  (contains ':')
    * seconds    (no ':')
  Works even for MIXED lines like: [58.2, 1:09.1]

- Convert to pure seconds.
- Add chunk offset parsed from filename pattern: *_<start>s_<end>s*.txt
- Optionally drop lines whose (offset-adjusted) end time exceeds global_end.

Usage examples:
  python auto_detect_and_convert_timestamps.py in1.txt in2.txt --out_dir out
  python auto_detect_and_convert_timestamps.py *.txt --out_dir out --drop_after_end
"""

import re
import argparse
import os
import pdb
from typing import Tuple


# Match any bracketed time pair: [t1, t2] + rest
BRACKET_RE = re.compile(
    r"""
    ^\s*
    \[
        (?P<t1>[^,\]]+)
        \s*,\s*
        (?P<t2>[^\]]+)
    \]
    (?P<rest>.*)
    $
    """,
    re.VERBOSE,
)

# Find "_<start>s_<end>s" anywhere in filename
FNAME_TIME_RE = re.compile(r"_(?P<start>\d+(?:\.\d+)?)s_(?P<end>\d+(?:\.\d+)?)s", re.IGNORECASE)


def parse_time_token_to_seconds(t: str) -> float:
    """Parse a single token that may be 'mm:ss(.d)' or 'ss(.d)'."""
    t = t.strip()
    if ":" in t:
        m, s = t.split(":", 1)
        return 60.0 * float(m) + float(s)
    return float(t)


def parse_offset_and_end_from_filename(path: str) -> Tuple[float, float]:
    """
    Parse chunk start/end from filename like:
      EN2002a.Mix-Headset_240.0s_540.0s.wav_temp0.txt
    Returns (offset=start, global_end=end).
    """
    base = os.path.basename(path)
    m = FNAME_TIME_RE.search(base)
    if not m:
        raise ValueError(f"Cannot parse '_<start>s_<end>s' from filename: {base}")
    return float(m.group("start")), float(m.group("end"))


def make_output_path(in_path: str, out_dir: str, suffix: str) -> str:
    base = os.path.basename(in_path)
    root, ext = os.path.splitext(base)
    root = root.replace(".wav_temp0", ".global_time")
    return os.path.join(out_dir, f"{root}{suffix}{ext if ext else '.txt'}")


def convert_one_file(
    in_path: str,
    out_path: str,
    precision: int,
    drop_after_end: bool,
) -> None:
    fmt = f"{{:.{precision}f}}"

    offset, global_end = parse_offset_and_end_from_filename(in_path)

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.rstrip("\n")

            m = BRACKET_RE.match(line)
            if not m:
                fout.write(raw)
                continue

            try:
                start = parse_time_token_to_seconds(m.group("t1")) + offset
                end = parse_time_token_to_seconds(m.group("t2")) + offset
            except ValueError:
                # If time tokens are weird, keep line unchanged
                fout.write(raw)
                continue

            if drop_after_end and end > global_end:
                continue

            fout.write(f"[{fmt.format(start)}, {fmt.format(end)}]{m.group('rest')}\n")

    print(f"OK: {in_path} -> {out_path} (offset={offset}, end={global_end})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Input transcript files (one or many)")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--suffix", default="", help="Suffix added before extension for output files")
    ap.add_argument("--precision", type=int, default=1, help="Decimal precision for output timestamps")
    ap.add_argument(
        "--drop_after_end",
        action="store_true",
        help="If set, drop lines whose adjusted end time > chunk end time parsed from filename.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for in_path in args.inputs:
        out_path = make_output_path(in_path, args.out_dir, args.suffix)
        convert_one_file(
            in_path=in_path,
            out_path=out_path,
            precision=args.precision,
            drop_after_end=args.drop_after_end,
        )

    print(f"Done. Wrote outputs to directory: {args.out_dir}")


if __name__ == "__main__":
    main()
