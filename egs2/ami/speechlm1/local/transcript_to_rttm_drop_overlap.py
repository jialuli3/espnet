#!/usr/bin/env python3
"""
transcript_to_rttm_drop_overlap.py

Convert global-timestamp transcripts to RTTM, dropping overlap-duplicated regions.

Input line format:
  [start, end] Speaker G1: text
  [start, end] (Laughter)   # ignored

Filenames must contain chunk times:
  *_<chunk_start>s_<chunk_end>s*.txt

Overlap handling (default overlap_sec=60):
  - First chunk: keep [chunk_start, chunk_end]
  - Subsequent chunks: keep [chunk_start + overlap_sec, chunk_end]
This removes duplicated overlap regions across chunk boundaries.

RTTM format:
  SPEAKER <file_id> 1 <tbeg> <tdur> <NA> <NA> <spk> <NA> <NA>
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

TIME_RE = re.compile(r"_(?P<start>\d+(?:\.\d+)?)s_(?P<end>\d+(?:\.\d+)?)s", re.IGNORECASE)
LINE_RE = re.compile(r"^\s*\[(?P<start>-?\d+(?:\.\d+)?),\s*(?P<end>-?\d+(?:\.\d+)?)\]\s*(?P<rest>.*)$")
SPEAKER_RE = re.compile(r"^\s*Speaker\s+(?P<spk>\S+)\s*:\s*(?P<txt>.*)$")


@dataclass
class Seg:
    start: float
    end: float
    spk: str


def parse_chunk_times(path: str) -> Tuple[float, float]:
    base = os.path.basename(path)
    m = TIME_RE.search(base)
    if not m:
        raise ValueError(f"Cannot parse _<start>s_<end>s from filename: {base}")
    return float(m.group("start")), float(m.group("end"))


def read_segments_from_transcript(path: str) -> List[Seg]:
    segs: List[Seg] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = LINE_RE.match(line)
            if not m:
                continue
            s = float(m.group("start"))
            e = float(m.group("end"))
            rest = m.group("rest").strip()

            sm = SPEAKER_RE.match(rest)
            if not sm:
                continue  # ignore non-speaker lines
            spk = sm.group("spk")
            if e <= s:
                continue
            segs.append(Seg(s, e, spk))
    return segs


def clip_to_window(segs: List[Seg], win_start: float, win_end: float) -> List[Seg]:
    out: List[Seg] = []
    for g in segs:
        s = max(g.start, win_start)
        e = min(g.end, win_end)
        if e > s:
            out.append(Seg(s, e, g.spk))
    return out


def merge_adjacent(segs: List[Seg], merge_gap: float) -> List[Seg]:
    """Merge same-speaker segments if they overlap or are within merge_gap seconds."""
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: (x.start, x.end))
    merged = [segs[0]]
    for g in segs[1:]:
        last = merged[-1]
        if g.spk == last.spk and g.start <= last.end + merge_gap:
            last.end = max(last.end, g.end)
        else:
            merged.append(g)
    return merged


def write_rttm(segs: List[Seg], out_path: str, file_id: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for g in sorted(segs, key=lambda x: (x.start, x.end)):
            tbeg = g.start
            tdur = g.end - g.start
            f.write(f"SPEAKER {file_id} 1 {tbeg:.3f} {tdur:.3f} <NA> <NA> {g.spk} <NA> <NA>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="+", help="Input transcript files (global timestamps)")
    ap.add_argument("--out_rttm", required=True, help="Output RTTM path")
    ap.add_argument("--file_id", default="recording", help="RTTM file_id (e.g., EN2002a)")
    ap.add_argument("--overlap_sec", type=float, default=60.0, help="Chunk overlap length in seconds")
    ap.add_argument("--merge_gap", type=float, default=0.0, help="Merge same-speaker segments if gap <= this")
    args = ap.parse_args()

    # sort by chunk start time
    items = []
    for p in args.transcripts:
        cs, ce = parse_chunk_times(p)
        items.append((cs, ce, p))
    items.sort(key=lambda x: x[0])

    all_kept: List[Seg] = []

    for i, (cs, ce, p) in enumerate(items):
        segs = read_segments_from_transcript(p)

        # Drop duplicated overlap region for every chunk except the first
        win_start = cs if i == 0 else cs + args.overlap_sec
        win_end = ce

        segs = clip_to_window(segs, win_start, win_end)
        all_kept.extend(segs)

    if args.merge_gap > 0:
        all_kept = merge_adjacent(all_kept, args.merge_gap)

    write_rttm(all_kept, args.out_rttm, args.file_id)
    print(f"Done. Wrote RTTM: {args.out_rttm} (segments={len(all_kept)})")


if __name__ == "__main__":
    main()
