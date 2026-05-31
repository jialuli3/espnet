#!/usr/bin/env python3
"""
Chunk a WAV file into fixed-length segments with overlap.

Example:
  python chunk_wav_overlap.py input.wav out_dir --chunk_sec 300 --overlap_sec 30
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf


def chunk_wav(
    wav_path: str,
    out_dir: str,
    chunk_sec: float = 300.0,
    overlap_sec: float = 30.0,
    prefix: str = "",
) -> None:
    wav_path = str(wav_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = sf.info(wav_path)
    sr = info.samplerate
    subtype = info.subtype  # preserve bit depth/encoding if possible

    y, sr_read = sf.read(wav_path, always_2d=True)
    assert sr_read == sr, f"Samplerate mismatch: info={sr}, read={sr_read}"

    n_samples = y.shape[0]
    total_sec = n_samples / sr

    if overlap_sec < 0 or chunk_sec <= 0:
        raise ValueError("chunk_sec must be > 0 and overlap_sec must be >= 0")
    if overlap_sec >= chunk_sec:
        raise ValueError("overlap_sec must be < chunk_sec")

    chunk_len = int(round(chunk_sec * sr))
    hop_len = int(round((chunk_sec - overlap_sec) * sr))

    base = prefix if prefix else Path(wav_path).stem

    idx = 0
    start = 0
    while start < n_samples:
        end = min(start + chunk_len, n_samples)
        seg = y[start:end]

        start_t = start / sr
        end_t = end / sr

        out_name = f"{base}_{start_t:0.1f}s_{end_t:0.1f}s.wav"
        out_path = out_dir / out_name

        sf.write(out_path, seg, sr, subtype=subtype)

        idx += 1
        if end == n_samples:
            break
        start += hop_len

    print(f"Wrote {idx} chunks to: {out_dir}")
    print(f"Input duration: {total_sec:.2f}s, chunk={chunk_sec}s, overlap={overlap_sec}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="Path to input .wav")
    ap.add_argument("out_dir", help="Output directory")
    ap.add_argument("--chunk_sec", type=float, default=300.0, help="Chunk length in seconds (default: 300)")
    ap.add_argument("--overlap_sec", type=float, default=30.0, help="Overlap in seconds (default: 30)")
    ap.add_argument("--prefix", type=str, default="", help="Optional output filename prefix")
    args = ap.parse_args()

    if not os.path.isfile(args.wav):
        raise FileNotFoundError(args.wav)

    chunk_wav(args.wav, args.out_dir, args.chunk_sec, args.overlap_sec, args.prefix)


if __name__ == "__main__":
    main()
