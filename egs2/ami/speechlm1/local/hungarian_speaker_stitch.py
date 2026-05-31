#!/usr/bin/env python3
"""
hungarian_speaker_stitch_sentence1to1.py

What you asked for (sentence-based, one-by-one):
- Use ONLY the 60s overlap windows:
    prev chunk: last overlap_sec seconds
    cur  chunk: first overlap_sec seconds
- Do NOT aggregate "all text per speaker".
- Instead: match INDIVIDUAL utterances (sentences) across the overlap.
- If a single utterance from local speaker L matches well to an utterance spoken by
  some previous GLOBAL speaker G, then L -> G (that best sentence match drives the mapping).

Constraints:
- Global K is fixed as max #speakers appearing in any chunk.
- Mapping within each chunk is injective (no many-to-one local->global):
    each local speaker in a chunk gets a unique global ID (when K >= #locals).

Dependencies:
  pip install numpy scipy scikit-learn

Input transcript format:
  [start, end] Speaker A: text
  [start, end] (Laughter)   # ignored for matching

Filename must contain chunk time range:
  *_0.0s_300.0s*.txt
"""

from __future__ import annotations
import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pdb

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


TIME_RE = re.compile(r"_(?P<start>\d+(?:\.\d+)?)s_(?P<end>\d+(?:\.\d+)?)s", re.IGNORECASE)
LINE_RE = re.compile(r"^\s*\[(?P<start>-?\d+(?:\.\d+)?),\s*(?P<end>-?\d+(?:\.\d+)?)\]\s*(?P<rest>.*)$")
SPEAKER_RE = re.compile(r"^\s*Speaker\s+(?P<spk>[A-Za-z0-9_]+)\s*:\s*(?P<txt>.*)$")


@dataclass
class Utterance:
    start: float
    end: float
    speaker: Optional[str]  # local or global ID depending on stage
    text: str


def parse_chunk_times(path: str) -> Tuple[float, float]:
    m = TIME_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse chunk times from filename: {path}")
    return float(m.group("start")), float(m.group("end"))


def read_transcript(path: str) -> List[Utterance]:
    utts: List[Utterance] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            start = float(m.group("start"))
            end = float(m.group("end"))
            rest = m.group("rest").strip()

            sm = SPEAKER_RE.match(rest)
            if sm:
                utts.append(Utterance(start, end, sm.group("spk"), sm.group("txt").strip()))
            else:
                utts.append(Utterance(start, end, None, rest))
    return utts


def speakers_in_chunk(utts: List[Utterance]) -> List[str]:
    return sorted({u.speaker for u in utts if u.speaker is not None})


def is_chunk_relative_times(utts: List[Utterance], chunk_dur: float) -> bool:
    if not utts:
        return True
    starts = np.array([u.start for u in utts], dtype=float)
    ends = np.array([u.end for u in utts], dtype=float)
    return (np.nanmin(starts) >= -1.0) and (np.nanmax(ends) <= chunk_dur + 2.0)


def select_overlap_window_utts(
    raw_utts: List[Utterance],
    chunk_start: float,
    chunk_end: float,
    overlap_sec: float,
    which: str,  # "prev" or "cur"
) -> List[Utterance]:
    """
    Use timestamps ONLY to select boundary overlap windows.
    Converts chunk-relative times to absolute for window filtering ONLY.
    Returned utterances have absolute times (not used for alignment).
    """
    dur = chunk_end - chunk_start
    rel = is_chunk_relative_times(raw_utts, dur)

    if which == "prev":
        win_start = max(chunk_end - overlap_sec, chunk_start)
        win_end = chunk_end
    elif which == "cur":
        win_start = chunk_start
        win_end = min(chunk_start + overlap_sec, chunk_end)
    else:
        raise ValueError("which must be 'prev' or 'cur'")

    out: List[Utterance] = []
    for u in raw_utts:
        if u.speaker is None:
            continue
        if not u.text.strip():
            continue
        abs_start = u.start + chunk_start if rel else u.start
        abs_end = u.end + chunk_start if rel else u.end
        if abs_end <= win_start or abs_start >= win_end:
            continue
        out.append(Utterance(abs_start, abs_end, u.speaker, u.text))
    return out


def joint_tfidf_cosine_matrix(texts_a: List[str], texts_b: List[str]) -> np.ndarray:
    """Joint TF-IDF on A+B, L2 norm, cosine sim = dot."""
    all_texts = texts_a + texts_b
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, stop_words="english")
    X = vec.fit_transform(all_texts).tocsr().astype(np.float32)

    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    X = X.multiply(1.0 / norms[:, None]).tocsr()

    A = X[: len(texts_a), :]
    B = X[len(texts_a) :, :]
    return (A @ B.T).toarray().astype(np.float32)


def build_local_to_global_candidates_from_sentence_matches(
    prev_ov_global: List[Utterance],   # speaker = GLOBAL ID
    cur_ov_local: List[Utterance],     # speaker = LOCAL ID
    min_words: int,
    min_sim: float,
) -> Dict[str, Tuple[str, float]]:
    """
    For each CURRENT local speaker L, find the single best (global speaker G) supported by
    ANY ONE utterance pair (u_prev, u_cur). No speaker-level aggregation.

    Returns:
      best_for_local[L] = (G, score)
    where score is the best utterance-utterance similarity achieved for that (L,G) choice.
    """
    # filter short utterances (recommended to avoid "yeah/ok")
    prev_idx = [i for i, u in enumerate(prev_ov_global) if len(u.text.split()) >= min_words]
    cur_idx = [j for j, u in enumerate(cur_ov_local) if len(u.text.split()) >= min_words]

    if not prev_idx or not cur_idx:
        return {}

    prev_texts = [prev_ov_global[i].text for i in prev_idx]
    cur_texts = [cur_ov_local[j].text for j in cur_idx]

    sim = joint_tfidf_cosine_matrix(prev_texts, cur_texts)  # (P, C)

    # For each local speaker, keep best global speaker based on ANY sentence match
    best_for_local: Dict[str, Tuple[str, float]] = {}

    for pi, i in enumerate(prev_idx):
        g = prev_ov_global[i].speaker  # global
        if g is None:
            continue
        for cj, j in enumerate(cur_idx):
            l = cur_ov_local[j].speaker  # local
            if l is None:
                continue
            s = float(sim[pi, cj])
            if s < min_sim:
                continue
            # candidate: local l matches to global g with score s
            if (l not in best_for_local) or (s > best_for_local[l][1]):
                best_for_local[l] = (g, s)

    return best_for_local


def greedy_injective_assignment(
    cur_locals: List[str],
    global_pool: List[str],
    best_for_local: Dict[str, Tuple[str, float]],
) -> Dict[str, str]:
    """
    One-by-one mapping with injectivity:
    - Create candidate edges (local -> global, score)
    - Sort by score desc
    - Assign if neither local nor global used
    - Remaining locals get any unused globals (unique)

    Assumes len(global_pool) >= len(cur_locals) (true if K defined as max locals per chunk).
    """
    # candidate edges sorted by confidence
    edges: List[Tuple[float, str, str]] = []  # (score, local, global)
    for l, (g, s) in best_for_local.items():
        edges.append((s, l, g))
    edges.sort(reverse=True)

    assigned: Dict[str, str] = {}
    used_globals = set()

    for s, l, g in edges:
        if l in assigned:
            continue
        if g in used_globals:
            continue
        assigned[l] = g
        used_globals.add(g)

    # Fill the rest uniquely with unused globals
    unused = [g for g in global_pool if g not in used_globals]
    ui = 0
    for l in cur_locals:
        if l in assigned:
            continue
        if ui >= len(unused):
            # Should not happen if pool big enough
            assigned[l] = global_pool[0]
        else:
            assigned[l] = unused[ui]
            ui += 1

    # Safety (injective if pool >= locals)
    if len(set(assigned.values())) != len(assigned):
        raise RuntimeError("Many-to-one mapping occurred; check K >= #locals and assignment logic.")
    return assigned


def relabel_transcript_lines(raw_utts: List[Utterance], local_to_global: Dict[str, str]) -> List[str]:
    out: List[str] = []
    for u in raw_utts:
        if u.speaker is None:
            out.append(f"[{u.start:.1f}, {u.end:.1f}] {u.text}")
        else:
            out.append(f"[{u.start:.1f}, {u.end:.1f}] Speaker {local_to_global.get(u.speaker, u.speaker)}: {u.text}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="+", help="Chunk transcript files in chronological order")
    ap.add_argument("--out_dir", default="stitched_sentence1to1", help="Output directory")
    ap.add_argument("--overlap_sec", type=float, default=60.0, help="Overlap window length (seconds)")
    ap.add_argument("--min_words", type=int, default=3, help="Ignore utterances shorter than this word count")
    ap.add_argument("--min_sim", type=float, default=0.25, help="Min cosine sim for an utterance match to count")
    ap.add_argument("--global_prefix", default="G", help="Global speaker prefix (G1..GK)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load chunks
    chunks: List[Tuple[str, Tuple[float, float], List[Utterance]]] = []
    for p in args.transcripts:
        t0, t1 = parse_chunk_times(p)
        utts = read_transcript(p)
        chunks.append((p, (t0, t1), utts))

    if not chunks:
        raise RuntimeError("No transcripts provided.")

    # K = max #speakers in any chunk
    K = max(len(speakers_in_chunk(utts)) for _, _, utts in chunks)
    if K <= 0:
        raise RuntimeError("No speakers found in transcripts.")
    global_pool = [f"{args.global_prefix}{i}" for i in range(1, K + 1)]

    local_to_global_all: List[Dict[str, str]] = []

    # First chunk: assign locals arbitrarily to G1.. (unique)
    p0, (s0, e0), u0 = chunks[0]
    locals0 = speakers_in_chunk(u0)
    if len(locals0) > K:
        locals0 = locals0[:K]
    map0 = {ls: global_pool[i] for i, ls in enumerate(locals0)}
    local_to_global_all.append(map0)

    out0 = os.path.join(args.out_dir, os.path.basename(p0).replace(".txt", ".global.txt"))
    with open(out0, "w", encoding="utf-8") as f:
        f.write("\n".join(relabel_transcript_lines(u0, map0)) + "\n")

    # Iterate chunks
    for idx in range(1, len(chunks)):
        prev_path, (ps, pe), prev_raw = chunks[idx - 1]
        cur_path, (cs, ce), cur_raw = chunks[idx]

        prev_ov = select_overlap_window_utts(prev_raw, ps, pe, args.overlap_sec, which="prev")
        cur_ov = select_overlap_window_utts(cur_raw, cs, ce, args.overlap_sec, which="cur")

        # Convert prev overlap local speakers -> global speakers (from previous mapping)
        prev_map = local_to_global_all[idx - 1]
        prev_ov_global: List[Utterance] = []
        for u in prev_ov:
            g = prev_map.get(u.speaker)
            if g is None:
                continue
            prev_ov_global.append(Utterance(u.start, u.end, g, u.text))

        pdb.set_trace()
        # Build per-local best (global, score) using single utterance matches only
        best_for_local = build_local_to_global_candidates_from_sentence_matches(
            prev_ov_global, cur_ov, min_words=args.min_words, min_sim=args.min_sim
        )

        cur_locals = speakers_in_chunk(cur_raw)
        if len(cur_locals) > K:
            cur_locals = cur_locals[:K]

        # One-by-one, injective assignment based on best single sentence matches
        cur_map = greedy_injective_assignment(cur_locals, global_pool, best_for_local)
        local_to_global_all.append(cur_map)

        # write output
        out_txt = os.path.join(args.out_dir, os.path.basename(cur_path).replace(".txt", ".global.txt"))
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(relabel_transcript_lines(cur_raw, cur_map)) + "\n")

        out_map = os.path.join(args.out_dir, os.path.basename(cur_path).replace(".txt", ".map.txt"))
        with open(out_map, "w", encoding="utf-8") as f:
            f.write(f"# K={K}, overlap_sec={args.overlap_sec}, min_words={args.min_words}, min_sim={args.min_sim}\n")
            f.write(f"# prev_overlap_utts={len(prev_ov_global)}, cur_overlap_utts={len(cur_ov)}\n")
            f.write("# local -> global (with best single-sentence evidence if available)\n")
            for l in cur_locals:
                if l in best_for_local:
                    g, s = best_for_local[l]
                    f.write(f"{l}\t{cur_map[l]}\tbest_evidence={g}:{s:.3f}\n")
                else:
                    f.write(f"{l}\t{cur_map[l]}\tbest_evidence=None\n")

    print(f"Done. K={K}. Wrote stitched transcripts to: {args.out_dir}")


if __name__ == "__main__":
    main()
