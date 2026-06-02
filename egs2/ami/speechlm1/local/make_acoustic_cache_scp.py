#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
from pathlib import Path

import kaldiio
import numpy as np


SEGMENT_RE = re.compile(r"^(?P<recording>.+)-(?P<start>\d+)-(?P<end>\d+)$")


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build oracle acoustic speaker-cache codec_ssl scp from RTTM. "
            "For each session, the script selects fixed non-overlap enrollment "
            "segments and writes those codec/SSL features as conditioning cache."
        )
    )
    parser.add_argument("--rttm", required=True, type=Path)
    parser.add_argument("--codec_scp", required=True, type=Path)
    parser.add_argument(
        "--utt_list",
        type=Path,
        help=(
            "Optional Kaldi-style file whose first column defines the utterances "
            "that should receive cache entries. The full codec_scp is still used "
            "to locate enrollment chunks."
        ),
    )
    parser.add_argument("--output_scp", required=True, type=Path)
    parser.add_argument("--output_ark", required=True, type=Path)
    parser.add_argument("--max_speakers", default=5, type=int)
    parser.add_argument("--enrollment_duration", default=5.0, type=float)
    parser.add_argument("--codec_token_per_frame", default=9, type=int)
    parser.add_argument(
        "--frame_rate",
        default=None,
        type=float,
        help=(
            "Optional frames-per-second override. By default the script infers "
            "fps from each selected codec chunk as num_frames / chunk_duration."
        ),
    )
    parser.add_argument("--min_frames", default=4, type=int)
    parser.add_argument("--max_cache_frames", default=0, type=int)
    parser.add_argument("--max_examples", default=0, type=int)
    return parser


def parse_utt_id(utt_id):
    match = SEGMENT_RE.match(utt_id)
    if match is None:
        raise ValueError(f"Unexpected utterance id format: {utt_id}")
    return match.group("recording"), float(match.group("start")), float(match.group("end"))


def session_id(recording):
    return recording


def read_utt_list(path):
    utt_ids = []
    with path.open() as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            utt_ids.append(line.split(maxsplit=1)[0])
    return utt_ids


def read_codec_scp(path, max_examples, utt_list=None):
    utt2example = {}
    chunks_by_recording = defaultdict(list)
    with path.open() as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            utt_id, ark_path = line.split(maxsplit=1)
            recording, start, end = parse_utt_id(utt_id)
            utt2example[utt_id] = (utt_id, recording, start, end, ark_path)
            chunks_by_recording[recording].append((start, end, utt_id, ark_path))

    for chunks in chunks_by_recording.values():
        chunks.sort(key=lambda item: (item[0], item[1], item[2]))

    if utt_list is None:
        examples = list(utt2example.values())
    else:
        examples = []
        missing = 0
        for utt_id in utt_list:
            if utt_id in utt2example:
                examples.append(utt2example[utt_id])
            else:
                missing += 1
        if missing:
            print(f"WARNING: {missing} utt_list entries were missing from {path}")

    if max_examples > 0:
        examples = examples[:max_examples]
    return examples, chunks_by_recording


def read_rttm(path):
    turns_by_session = defaultdict(list)
    turns_by_recording = defaultdict(list)
    with path.open() as reader:
        for line in reader:
            fields = line.strip().split()
            if len(fields) < 8 or fields[0] != "SPEAKER":
                continue
            recording = fields[1]
            start = float(fields[3])
            end = start + float(fields[4])
            speaker = fields[7]
            if end <= start:
                continue
            turn = {
                "recording": recording,
                "session": session_id(recording),
                "start": start,
                "end": end,
                "speaker": speaker,
            }
            turns_by_session[turn["session"]].append(turn)
            turns_by_recording[recording].append(turn)

    for turns in turns_by_session.values():
        turns.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    for turns in turns_by_recording.values():
        turns.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return turns_by_session, turns_by_recording


def subtract_intervals(start, end, blockers):
    segments = [(start, end)]
    for block_start, block_end in blockers:
        next_segments = []
        for seg_start, seg_end in segments:
            if block_end <= seg_start or block_start >= seg_end:
                next_segments.append((seg_start, seg_end))
                continue
            if block_start > seg_start:
                next_segments.append((seg_start, min(block_start, seg_end)))
            if block_end < seg_end:
                next_segments.append((max(block_end, seg_start), seg_end))
        segments = next_segments
        if not segments:
            break
    return [(seg_start, seg_end) for seg_start, seg_end in segments if seg_end > seg_start]


def clean_speaker_segments(turn, turns_by_recording):
    blockers = [
        (other["start"], other["end"])
        for other in turns_by_recording[turn["recording"]]
        if other["speaker"] != turn["speaker"]
        and other["end"] > turn["start"]
        and other["start"] < turn["end"]
    ]
    return subtract_intervals(turn["start"], turn["end"], blockers)


def speaker_order(turns):
    first_seen = {}
    for turn in turns:
        first_seen.setdefault(turn["speaker"], (turn["start"], turn["speaker"]))
    return [speaker for speaker, _ in sorted(first_seen.items(), key=lambda item: item[1])]


def find_chunk(chunks_by_recording, recording, start, end):
    chunks = chunks_by_recording.get(recording, [])
    for chunk_start, chunk_end, utt_id, ark_path in chunks:
        if chunk_start <= start and chunk_end >= end:
            return chunk_start, chunk_end, utt_id, ark_path

    best = None
    best_overlap = 0.0
    for chunk_start, chunk_end, utt_id, ark_path in chunks:
        overlap = min(chunk_end, end) - max(chunk_start, start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = (chunk_start, chunk_end, utt_id, ark_path)
    return best


def load_codec(path, codec_token_per_frame):
    value = kaldiio.load_mat(path)
    value = np.asarray(value, dtype=np.int32)
    if value.ndim == 1:
        value = value.reshape(-1, codec_token_per_frame)
    return value


def slice_codec(codec, rel_start, rel_end, frame_rate, min_frames, max_cache_frames):
    beg = max(0, int(round(rel_start * frame_rate)))
    num_frames = int(round((rel_end - rel_start) * frame_rate))
    fin = min(codec.shape[0], beg + num_frames)
    if fin - beg < min_frames:
        return None
    segment = codec[beg:fin]
    if max_cache_frames > 0 and segment.shape[0] > max_cache_frames:
        segment = segment[:max_cache_frames]
    return segment


def safe_key(text):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def select_enrollments(
    turns_by_session,
    turns_by_recording,
    chunks_by_recording,
    enrollment_duration,
    max_speakers,
):
    selected = defaultdict(list)
    warnings = []

    for sess, turns in turns_by_session.items():
        for speaker in speaker_order(turns)[:max_speakers]:
            speaker_turns = [turn for turn in turns if turn["speaker"] == speaker]
            clean_candidates = []
            for turn in speaker_turns:
                for clean_start, clean_end in clean_speaker_segments(
                    turn, turns_by_recording
                ):
                    clean_duration = clean_end - clean_start
                    if clean_duration <= 0:
                        continue
                    clean_candidates.append(
                        (turn["recording"], clean_start, clean_end, clean_duration)
                    )

            clean_candidates.sort(key=lambda item: (-item[3], item[1], item[2]))

            selected_parts = []
            selected_duration = 0.0

            for recording, clean_start, clean_end, _ in clean_candidates:
                if selected_duration >= enrollment_duration:
                    break
                remaining = enrollment_duration - selected_duration
                part_end = min(clean_end, clean_start + remaining)
                chunk = find_chunk(
                    chunks_by_recording, recording, clean_start, part_end
                )
                if chunk is None:
                    continue
                selected_parts.append((recording, clean_start, part_end, chunk))
                selected_duration += part_end - clean_start
                if selected_duration >= enrollment_duration:
                    break

            if not selected_parts:
                warnings.append(f"{sess}/{speaker}: no non-overlap speech found")
                continue

            selected[sess].append((speaker, selected_parts))
            if selected_duration < enrollment_duration - 1e-3:
                warnings.append(
                    f"{sess}/{speaker}: only found {selected_duration:.2f}s "
                    f"non-overlap speech, shorter than requested "
                    f"{enrollment_duration:.2f}s"
                )

    return selected, warnings


def main():
    args = get_parser().parse_args()

    utt_list = read_utt_list(args.utt_list) if args.utt_list is not None else None
    examples, chunks_by_recording = read_codec_scp(
        args.codec_scp, args.max_examples, utt_list
    )
    turns_by_session, turns_by_recording = read_rttm(args.rttm)
    enrollments, warnings = select_enrollments(
        turns_by_session,
        turns_by_recording,
        chunks_by_recording,
        args.enrollment_duration,
        args.max_speakers,
    )

    ark_values = {}
    cache_keys_by_session = defaultdict(list)
    empty_key = "empty_cache"
    ark_values[empty_key] = np.zeros((args.codec_token_per_frame,), dtype=np.int32)

    for sess, sess_enrollments in sorted(enrollments.items()):
        for idx, (speaker, parts) in enumerate(sess_enrollments, 1):
            segments = []
            for recording, start, end, chunk in parts:
                chunk_start, chunk_end, chunk_utt, ark_path = chunk
                codec = load_codec(ark_path, args.codec_token_per_frame)
                if args.frame_rate is None:
                    chunk_duration = chunk_end - chunk_start
                    frame_rate = codec.shape[0] / chunk_duration
                else:
                    frame_rate = args.frame_rate
                segment = slice_codec(
                    codec,
                    start - chunk_start,
                    end - chunk_start,
                    frame_rate,
                    args.min_frames,
                    args.max_cache_frames,
                )
                if segment is not None:
                    segments.append(segment)

            if not segments:
                warnings.append(
                    f"{sess}/{speaker}: selected segment too short after codec slicing"
                )
                continue
            segment = np.concatenate(segments, axis=0)
            key = (
                f"{safe_key(sess)}_slot{idx}_{safe_key(speaker)}_"
                f"{len(parts)}parts_{int(round(sum(p[2] - p[1] for p in parts) * 100))}"
            )
            ark_values[key] = segment.reshape(-1)
            cache_keys_by_session[sess].append(key)

    args.output_ark.parent.mkdir(parents=True, exist_ok=True)
    args.output_scp.parent.mkdir(parents=True, exist_ok=True)
    generated_scp = str(args.output_scp) + ".segments"
    kaldiio.save_ark(str(args.output_ark), ark_values, scp=generated_scp)

    segment_paths = {}
    with open(generated_scp) as reader:
        for line in reader:
            key, path = line.strip().split(maxsplit=1)
            segment_paths[key] = path

    with args.output_scp.open("w") as writer:
        for utt_id, recording, _, _, _ in examples:
            keys = cache_keys_by_session.get(session_id(recording), [empty_key])
            paths = [segment_paths[key] for key in keys]
            writer.write(f"{utt_id} {' '.join(paths)}\n")

    print(
        f"Wrote {len(examples)} utterances, "
        f"{sum(len(v) for v in cache_keys_by_session.values())} enrollment segments, "
        f"{len(cache_keys_by_session)} sessions to {args.output_scp}"
    )
    for warning in warnings[:50]:
        print(f"WARNING: {warning}")
    if len(warnings) > 50:
        print(f"WARNING: ... {len(warnings) - 50} more warnings")


if __name__ == "__main__":
    main()
