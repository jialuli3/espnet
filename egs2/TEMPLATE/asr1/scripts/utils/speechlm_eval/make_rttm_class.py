import os
import re
import argparse
from collections import Counter
from typing import List, Dict
from scipy.io.wavfile import read
from scipy.signal import medfilt
import numpy as np
from praatio import textgrid
import pdb


class DiarizationScorer:
    def __init__(
        self,
        ref_rttm_file: str,
        hyp_output_file: str,
        hyp_format: str,
        scoring_dir: str,
        speaker_order_method: str,
        spk_format: str = "spk_idx",
        skip_interval: int = 1,
        frame_duration: float = 0.1, 
        kernel_size: int = 11,
        spk_count_dur: float = 0.02,
        spk_count_skip: float = 0.02,
        segment_dur: float = 8,
        segment_skip: float = 0.8,
        test_wav_scp: str = None,
        apply_clustering: bool = False,
        textgrid_dir: str = None
    ):
        self.ref_rttm_file = ref_rttm_file
        self.hyp_output_file = hyp_output_file
        self.hyp_format = hyp_format
        self.output_dir = scoring_dir
        self.textgrid_dir = textgrid_dir
        self.wav_scp_path = test_wav_scp
        self.speaker_order_method = speaker_order_method
        self.spk_format = spk_format
        self.skip_interval = skip_interval
        self.apply_clustering = apply_clustering
        self.frame_duration = frame_duration
        self.kernel_size = kernel_size

        self.spk_count_dur = spk_count_dur
        self.spk_count_skip = spk_count_skip
        self.segment_dur = segment_dur
        self.segment_skip = segment_skip

        self.spk_map = self.get_spk_order()

    @staticmethod
    def read_file(file_path: str) -> List[str]:
        with open(file_path, "r") as f:
            return f.readlines()

    @staticmethod
    def write_file(content: List[str], file_path: str):
        with open(file_path, "w") as f:
            f.writelines(content)
        f.close()

    def get_spk_order(self) -> Dict[str, Dict[int, str]]:
        rows = self.read_file(self.ref_rttm_file)
        arr_order, most_order, count = {}, {}, {}

        for row in rows:
            parts = row.strip().split()
            file_id, spk, dur = parts[1], parts[-3], float(parts[-6])
            arr_order.setdefault(file_id, {})
            most_order.setdefault(file_id, {})
            count.setdefault(file_id, Counter())

            if spk not in arr_order[file_id]:
                arr_order[file_id][spk] = len(arr_order[file_id]) + 1
            count[file_id][spk] += dur

        for file_id in count:
            for spk, _ in count[file_id].most_common():
                most_order[file_id][spk] = len(most_order[file_id]) + 1

        reverse = {}
        for file_id in arr_order:
            reverse[file_id] = {}
            order = arr_order if self.speaker_order_method == "arrive" else most_order
            for spk, idx in order[file_id].items():
                reverse[file_id][idx] = spk
        return reverse

    def make_hyp_rttm(self):
        content = self.read_file(self.hyp_output_file)
        out_dict = {}

        for row in content:
            try:
                key = row.strip().split()[0]
                file_info = key.split("_")[-2]
                file_id, start_offset = file_info.split("-")[0], float(file_info.split("-")[1])
                end_offset = float(file_info.split("-")[2])
                if self.skip_interval is not None and end_offset % self.skip_interval != 0:
                    continue
            except:
                continue

            if self.hyp_format == "event":
                out_dict.setdefault(file_id, {})
                all_events = re.findall((r'(<spk\d>.\(\d+\.\d+, \d+\.\d+\))'), row.strip())
                for curr_event in all_events:
                    try:
                        parts = curr_event.split()
                        spk_idx = int(parts[0][4])
                        if spk_idx in self.spk_map[file_id]:
                            curr_start = float(parts[1].strip("(").strip(","))
                            curr_end = float(parts[2].strip(")"))
                            start = round(curr_start+start_offset, 2)
                            end = round(curr_end+start_offset, 2)
                            if end-start<0.1: continue
                            out_dict[file_id].setdefault(spk_idx, []).append([start, end])
                    except:
                        continue
            elif self.hyp_format == "frame": # convert to event based model
                out_dict.setdefault(file_id, {})
                max_frame = int(np.ceil((end_offset-start_offset) / self.frame_duration))
                out = row.strip().split()[1:]
                for i,s in enumerate(out):
                    out[i]=re.findall((r'(\d+)'), s)[0] # extract numerical numbers only
                if len(out) < max_frame:
                    pad_zeros = [0] * (max_frame - len(out))
                    out.extend(pad_zeros)
                out = out[:max_frame]
                out_dict[file_id][(start_offset,end_offset)]= np.asarray(out, dtype=int)      

        # apply median filter on frame-based output and generated event-based output
        merged = self.merge_out_dict(out_dict)
        if self.hyp_format == "event":
            frame_output = self.event2frame(merged)
            merged = self.apply_median_filter(frame_output)
        else: # hyp_format == frame
            merged = self.apply_median_filter(merged)

        if self.apply_clustering:
            seg = self.get_segmentations(merged)
            rttm = self.apply_pyannote_clustering(seg)
            self.write_file(rttm, os.path.join(self.output_dir, "hyp.rttm"))
        else:
            out_lines = []
            for file_id in merged:
                for spk in merged[file_id]:
                    for start, end in merged[file_id][spk]:
                        dur = round(end - start, 2)
                        out_lines.append(f"SPEAKER {file_id} 1 {start} {dur} <NA> <NA> {spk} <NA> <NA>\n")
            self.write_file(out_lines, os.path.join(self.output_dir, "hyp.rttm"))

    def merge_out_dict(self, out_dict):
        merged = {}
        if self.hyp_format=="event":
            for file_id, spks in out_dict.items():
                merged[file_id] = {}
                for spk, intervals in spks.items():
                    intervals = sorted(intervals, key=lambda x: x[0])
                    merged[file_id][spk] = [intervals[0]]
                    for start, end in intervals[1:]:
                        if start <= merged[file_id][spk][-1][1]:
                            merged[file_id][spk][-1][1] = max(merged[file_id][spk][-1][1], end)
                        else:
                            merged[file_id][spk].append([start, end])
        else:
            for file_id in out_dict:
                merged.setdefault(file_id, [])
                sorted_timestamps=sorted(out_dict[file_id].keys())
                print(sorted_timestamps)
                prev_timestamp=None
                for timestamp in sorted_timestamps:
                    if prev_timestamp is None: # first timestamp
                        assert timestamp[0]==0, f"file id, {file_id}, timestamp, {timestamp}"
                        merged[file_id].extend(out_dict[file_id][timestamp])
                    else: # prev_timestamp is not None
                        assert timestamp[0]==prev_timestamp[-1], f"file id, {file_id}, timestamp[0], {timestamp[0]}, prev timestamp, {prev_timestamp[-1]}"
                        merged[file_id].extend(out_dict[file_id][timestamp])
                    prev_timestamp = timestamp
                        
        return merged

    def event2frame(self, merged):
        out_frame={}
        for file_id in merged:
            max_time = 0
            for spk_id in merged[file_id]:
                max_time=max(max_time, merged[file_id][spk_id][-1][-1])

            num_frames = int(np.ceil(max_time / self.frame_duration))
            num_spks = len(merged[file_id])
            labels = np.zeros((num_spks, num_frames), dtype=int)

            for spk_id in merged[file_id]:
                for start, end in merged[file_id][spk_id]:
                    start_idx, end_idx = int(start / self.frame_duration), int(np.ceil(end / self.frame_duration))
                    labels[spk_id - 1][start_idx:end_idx] = 1

            index = labels[0, :].copy()
            for i in range(1, num_spks):
                index += np.power(2, i) * labels[i, :]
            out_frame[file_id] = index # single stream
        return out_frame
            
    def apply_median_filter(self, frame_mat):
        filtered = {}
        for file_id in frame_mat:
            index = frame_mat[file_id]
            index = medfilt(index, kernel_size=self.kernel_size)
            num_spks=len(self.spk_map[file_id])
            num_frames = index.shape[0]

            labels = np.array([(index >> i) & 1 for i in range(num_spks)])
            filtered[file_id] = {spk + 1: [] for spk in range(num_spks)}

            for spk in range(num_spks):
                active, start = labels[spk], None
                for i, val in enumerate(active):
                    if val and start is None:
                        start = i
                    elif not val and start is not None:
                        filtered[file_id][spk + 1].append([round(start * self.frame_duration, 1), round(i * self.frame_duration, 1)])
                        start = None
                if start is not None:
                    filtered[file_id][spk + 1].append([round(start * self.frame_duration, 1), round(num_frames * self.frame_duration, 1)])
        return filtered

    def get_segmentations(self, merged):
        wav_lines = self.read_file(self.wav_scp_path)
        wav_dict = {line.strip().split()[0]: line.strip().split()[1] for line in wav_lines}
        segmentations = {}
        for file_id in merged:
            audio_path = wav_dict[file_id]
            rate, audio = read(audio_path)
            max_dur = len(audio) / rate
            num_spk = len(merged[file_id])
            total_frames = int((max_dur - self.spk_count_dur) / self.spk_count_skip)
            spk_activities = np.zeros((total_frames, num_spk))
            for spk_id in merged[file_id]:
                for start, end in merged[file_id][spk_id]:
                    s_idx, e_idx = int(start / self.spk_count_skip), int(end / self.spk_count_skip)
                    spk_activities[s_idx:e_idx, spk_id - 1] = 1
            total_chunks = int((max_dur - self.segment_dur) / self.segment_skip) + 1
            frame_len = int((self.segment_dur - self.spk_count_dur) / self.spk_count_skip) + 1
            segments = np.zeros((total_chunks, frame_len, num_spk))
            for i in range(total_chunks):
                s_idx = int(i * self.segment_skip / self.spk_count_skip)
                e_idx = min(s_idx + frame_len, len(spk_activities))
                segments[i, :e_idx - s_idx] = spk_activities[s_idx:e_idx]
            segmentations[file_id] = segments
        return segmentations

    def apply_pyannote_clustering(self, segmentations):
        # conditional import
        from diarizen.pipelines.inference import DiariZenPipeline
        from pyannote.core import SlidingWindowFeature, SlidingWindow
        
        wav_lines = self.read_file(self.wav_scp_path)
        wav_dict = {line.strip().split()[0]: line.strip().split()[1] for line in wav_lines}
        receptive_field = SlidingWindow(start=0.0, step=self.spk_count_skip, duration=self.spk_count_dur)
        frame = SlidingWindow(start=0.0, step=self.segment_skip, duration=self.segment_dur)
        diar_pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-meeting-base")
        diar_pipeline.max_n_speakers = 4
        out_rttm = ""
        for file_id, seg in segmentations.items():
            seg_input = SlidingWindowFeature(seg, frame)
            diar_result = diar_pipeline(wav_dict[file_id], seg_input, receptive_field, file_id)
            out_rttm += diar_result.to_rttm()
        return out_rttm.splitlines(True)

    def make_textgrid(self):
        hyp_dict = self.rttm_to_interval(os.path.join(self.output_dir, "hyp.rttm"), "hyp")
        ref_dict = self.rttm_to_interval(os.path.join(self.output_dir, "ref.rttm"), "ref")
        for file_id in hyp_dict:
            tg = textgrid.Textgrid()
            for spk in hyp_dict[file_id]:
                tier = textgrid.IntervalTier(f"{spk} hyp", hyp_dict[file_id][spk], 0, hyp_dict[file_id][spk][-1][1])
                tg.addTier(tier)
            for spk in ref_dict[file_id]:
                tier = textgrid.IntervalTier(f"{spk} ref", ref_dict[file_id][spk], 0, ref_dict[file_id][spk][-1][1])
                tg.addTier(tier)
            tg.save(os.path.join(self.textgrid_dir, f"{file_id}.TextGrid"), format="long_textgrid", includeBlankSpaces=True)

    def rttm_to_interval(self, rttm_file, file_type="hyp"):
        rows = self.read_file(rttm_file)
        out = {}
        for row in rows:
            parts = row.split()
            file_id, start, dur, spk = parts[1], float(parts[3]), float(parts[4]), parts[-3]
            out.setdefault(file_id, {}).setdefault(spk, []).append((round(start, 2), round(start + dur, 2), file_type))
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_rttm_file", type=str)
    parser.add_argument("--hyp_output_file", type=str)
    parser.add_argument("--hyp_format", type=str, choices=["event", "frame"])
    parser.add_argument("--scoring_dir", type=str)
    parser.add_argument("--textgrid_dir", type=str, default=None)
    parser.add_argument("--test_wav_scp", type=str, default=None)
    parser.add_argument("--speaker_order_method", type=str, choices=["arrive", "most"])
    parser.add_argument("--spk_format", type=str, choices=["spk_id", "spk_idx"], default="spk_idx")
    parser.add_argument("--skip_interval", type=int, default=1)
    parser.add_argument("--apply_clustering", action="store_true")
    args = parser.parse_args()

    scorer = DiarizationScorer(
        ref_rttm_file=args.ref_rttm_file,
        hyp_output_file=args.hyp_output_file,
        hyp_format=args.hyp_format,
        scoring_dir=args.scoring_dir,
        speaker_order_method=args.speaker_order_method,
        spk_format=args.spk_format,
        skip_interval=args.skip_interval,
        test_wav_scp=args.test_wav_scp,
        apply_clustering=args.apply_clustering,
        textgrid_dir=args.textgrid_dir
    )

    scorer.make_hyp_rttm()
    if args.textgrid_dir:
        scorer.make_textgrid()


if __name__ == "__main__":
    main()
