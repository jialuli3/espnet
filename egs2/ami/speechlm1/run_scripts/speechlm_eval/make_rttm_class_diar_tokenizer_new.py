import os
import re
import argparse
from collections import Counter
from typing import List, Dict
from scipy.io.wavfile import read
from scipy.signal import medfilt
import numpy as np
from praatio import textgrid
from speaker_matching_class import SpeakerSegmentMatcher
import pdb
import sys


class DiarizationScorer:
    def __init__(
        self,
        ref_rttm_file: str,
        hyp_output_file: str,
        hyp_format: str,
        tokenizer: str,
        scoring_dir: str,
        speaker_order_method: str,
        spk_format: str = "spk_idx",
        skip_interval: int = 1,
        frame_duration: float = 0.1, 
        kernel_size: int = 11,
        spk_count_dur: float = 0.02,
        spk_count_skip: float = 0.02,
        output_threshold: float = 0.5, 
        segment_dur: float = 5,
        segment_skip: float = 0.5,
        test_wav_scp: str = None,
        apply_clustering: bool = False,
        use_multistream: bool = False,
        use_multistream_subtask: bool = False,
        textgrid_dir: str = None,
        dataset_name: str = "ami",
        reco2dur: str = None,
        overlap_type: str = None,
        use_ipu: bool = False,
        use_dur_format: bool = False,
        check_overlap_sad: bool = False,
        check_overlap_od: bool = False,
        apply_local_speaker_matching: bool = False,
    ):
        self.ref_rttm_file = ref_rttm_file
        self.hyp_output_file = hyp_output_file
        self.hyp_format = hyp_format
        self.tokenizer = tokenizer
        self.output_dir = scoring_dir
        self.textgrid_dir = textgrid_dir
        self.wav_scp_path = test_wav_scp
        self.speaker_order_method = speaker_order_method
        self.spk_format = spk_format
        self.skip_interval = skip_interval
        self.apply_clustering = apply_clustering
        self.use_multistream = use_multistream
        self.use_multistream_subtask = use_multistream_subtask
        self.frame_duration = frame_duration
        self.kernel_size = kernel_size
        self.output_threshold = output_threshold

        self.spk_count_dur = spk_count_dur
        self.spk_count_skip = spk_count_skip
        self.segment_dur = segment_dur
        self.segment_skip = segment_skip
        self.dataset_name = dataset_name
        self.overlap_type = overlap_type
        self.use_ipu = use_ipu
        self.use_dur_format = use_dur_format
        self.check_overlap_sad = check_overlap_sad
        self.check_overlap_od = check_overlap_od
        self.apply_local_speaker_matching = apply_local_speaker_matching

        self.reco2dur = {}
        if reco2dur is not None:
            content = self.read_file(reco2dur)
            for row in content:
                self.reco2dur[row.split()[0]]=round(float(row.strip().split()[1]),1)

        self.spk_map = self.get_spk_order()

    def read_file(self, file_path: str) -> List[str]:
        with open(file_path, "r") as f:
            return f.readlines()

    def write_file(self, content: List[str], file_path: str):
        with open(file_path, "w") as f:
            f.writelines(content)
        f.close()

    @staticmethod
    def get_overlap_interval(s1,e1,s2,e2):
        if s1<=e2 and s2<=e1:
            return (max(s1,s2),min(e1,e2))
        return None

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

    def parse_sad_token(self, row, start_offset):
        all_events = re.findall((r'(<spk> <bot> <\d+\.\d+> <\d+\.\d+> <eot>)'), row)
        out_sad = []
        for curr_event in all_events:
            parts = curr_event.split()
            parts = [p.strip("<").strip(">") for p in parts]
            curr_start = float(parts[2])
            curr_end = float(parts[3])
            start = round(curr_start+start_offset, 2)
            end = round(curr_end+start_offset, 2)
            out_sad.append((start,end))
        return out_sad

    def parse_od_token(self, row, start_offset):
        all_events = re.findall((r'(<overlap> <bot> <\d+\.\d+> <\d+\.\d+> <eot>)'), row)
        out_od = []
        for curr_event in all_events:
            parts = curr_event.split()
            parts = [p.strip("<").strip(">") for p in parts]
            curr_start = float(parts[2])
            curr_end = float(parts[3])
            start = round(curr_start+start_offset, 2)
            end = round(curr_end+start_offset, 2)
            out_od.append((start,end))
        return out_od

    def parse_event_token(self, row, rttm_out, start_offset, sad_out=None):
        all_events = re.findall((r'(<spk\d> <bot> <\d+\.\d+> <\d+\.\d+> <eot>)'), row)
        for curr_event in all_events:
            try:
                parts = curr_event.split()
                parts = [p.strip("<").strip(">") for p in parts]
                spk_idx = int(parts[0][3])              
                curr_start = float(parts[2])
                curr_end = float(parts[3])
                start = round(curr_start+start_offset, 2)
                end = round(curr_end+start_offset, 2)
                if self.check_overlap_sad:
                    found_overlap=False
                    for start_sad, end_sad in sad_out:
                        if self.get_overlap_interval(start, end, start_sad, end_sad) is not None:
                            found_overlap=True
                            start = max(start, start_sad)
                            end = min(end, end_sad)
                            break
                    if not found_overlap: continue
                if end-start<self.output_threshold: continue
                rttm_out.setdefault(spk_idx, []).append([start, end])
            except:
                continue  

        return rttm_out

    def parse_sad_od_frame_token(self, row, max_frame, format="sad"):
        bit_coded = []
        for frame_output in row:
            frame_output = frame_output.strip("<").strip(">")
            try:
                if format=="sad" and frame_output=="sil":
                    bit_coded.append(0)
                elif format=="od" and frame_output=="no_overlap":
                    bit_coded.append(0)
                elif format=="sad" and frame_output=="spk":
                    bit_coded.append(1)
                elif format=="od" and frame_output=="overlap":
                    bit_coded.append(1)
            except:
                #print(frame_output)
                continue
        if len(bit_coded) < max_frame:
            pad_zeros = [0] * (max_frame - len(bit_coded))
            bit_coded.extend(pad_zeros)
        bit_coded = bit_coded[:max_frame]
        return bit_coded

    def parse_frame_token(self, row, max_frame):
        bit_coded = []
        for frame_output in row:
            frame_output = frame_output.strip("<").strip(">")
            try:
                if frame_output=="sil":
                    bit_coded.append(0)
                elif frame_output.startswith("spk"):
                    spk_id = int(frame_output[-1])
                    bit_coded.append(1 << (spk_id - 1))
                elif frame_output.startswith("overlap_spk"): # startswith overlap
                    spk_ids = frame_output.split("_")[2:]
                    spk_ids = [int(i) for i in spk_ids]
                    bit_coded.append(sum(1 << (i - 1) for i in spk_ids))
            except:
                continue

        if len(bit_coded) < max_frame:
            pad_zeros = [0] * (max_frame - len(bit_coded))
            bit_coded.extend(pad_zeros)
        bit_coded = bit_coded[:max_frame]
        return bit_coded

    def parse_frame_multistream_token(self, frame_rows, max_frame):
        frame_out_dict = {}
        for spk_idx, curr_str in enumerate(frame_rows):
            frame_out_dict.setdefault(spk_idx, [])
            parts = curr_str.strip().split(" ")
            for i in range(1, len(parts)):
                p = parts[i].strip("<").strip(">")
                if p=="sil":
                    frame_out_dict[spk_idx].append(0)
                elif p==f"spk{spk_idx+1}":
                    frame_out_dict[spk_idx].append(1)
                elif p=="end_sd":
                    break

        sorted_spk = sorted(frame_out_dict.keys())
        for spk_idx in sorted_spk:
            if len(frame_out_dict[spk_idx]) < max_frame:
                pad_zeros = [0] * (max_frame - len(frame_out_dict[spk_idx]))
                frame_out_dict[spk_idx].extend(pad_zeros) 
            else:
                frame_out_dict[spk_idx] = frame_out_dict[spk_idx][:max_frame]                     
        frame_out = np.stack([frame_out_dict[k] for k in sorted_spk])
        bit_coded = frame_out[0,:].copy()
        for i in range(1, frame_out.shape[0]):
            bit_coded += np.power(2, i) * frame_out[i, :]
        return bit_coded

    def parse_frame_timestamp_token(self, row, rttm_out, start_offset): # experimental
        bit_coded = []
        curr_start, curr_end = None, None
        spk_id = None
        for i,frame_output in enumerate(row):
            frame_output = frame_output.strip("<").strip(">")
            curr_event = []
            try:
                if "sc_start" in frame_output:
                    curr_start = float(row[i+1].strip("<").strip(">"))
                    spk_id = row[i+2].strip("<").strip(">")
                elif "sc_end" in frame_output:
                    curr_end = float(row[i+1].strip("<").strip(">"))
                    if spk_id is not None:
                        start = round(curr_start+start_offset, 2)
                        end = round(curr_end+start_offset, 2)
                    if end-start<self.output_threshold: continue
                    if spk_id.startswith("overlap_spk"):
                        curr_spk_ids=spk_id.split("_")[2:]
                        for curr_spk_id in curr_spk_ids:
                            rttm_out.setdefault(int(curr_spk_id), []).append([start, end])
                    else:
                        curr_spk_id = int(spk_id[-1])
                        rttm_out.setdefault(curr_spk_id, []).append([start, end])

            except:
                #print(frame_output)
                continue

        return rttm_out

    def get_curr_rttm_output(self, row, file_id, start_offset, end_offset):
        rttm_out={}
        if self.hyp_format == "event":
            row=row.strip().split(" ")[1:]
            sd_row = row
            try:
                start_idx = row.index("<start_sd>")
                end_idx = row.index("<end_sd>")
                sd_row = row[start_idx:end_idx]
            except:
                pass
            sd_row = " ".join(sd_row)

            if self.check_overlap_sad:
                sad_row = row
                try:
                    start_idx = row.index("<start_sad>")
                    end_idx = row.index("<end_sad>")
                    sad_row = row[start_idx:end_idx]
                except:
                    pass
                sad_row = " ".join(sad_row)
                sad_out=self.parse_sad_token(sad_row, start_offset)
                rttm_out=self.parse_event_token(sd_row, rttm_out, start_offset, sad_out)
            else:
                rttm_out=self.parse_event_token(sd_row, rttm_out, start_offset)
                       
        elif self.hyp_format == "frame":
            max_frame = int(np.ceil((end_offset-start_offset) / self.frame_duration))
            if self.use_multistream:
                out = " ".join(row.strip().split()[1:])
                rows = out.split(" | ")
                frame_out_dict = {}
                
                frame_rows=[]
                sad_row, sad_out = [],[]
                for row in rows:
                    parts = row.strip().split(" ")
                    parts_0 = parts[0].strip("<").strip(">")
                    if parts_0.startswith("start_sd"):
                        frame_rows.append(row)
                    elif parts_0.startswith("start_sad"):
                        sad_row = row.split(" ")
                        sad_coded = self.parse_sad_od_frame_token(sad_row, max_frame, "sad")

                if self.use_multistream_subtask:
                    frame_rows = frame_rows[0].split(" ")
                    bit_coded = self.parse_frame_token(frame_rows, max_frame)
                else:
                    bit_coded = self.parse_frame_multistream_token(frame_rows, max_frame)
                if self.check_overlap_sad:
                    bit_coded, sad_coded = np.asarray(bit_coded), np.asarray(sad_coded)
                    bit_coded = np.asarray(bit_coded * sad_coded, dtype=int)

                rttm_out[(start_offset,end_offset)]= np.asarray(bit_coded, dtype=int)

            else: # single stream
                row = row.strip().split(" ")[1:]
                start_idx, end_idx = 0, len(row)
                if self.check_overlap_sad:
                    sad_coded = None
                    try:
                        start_sad_idx = row.index("<start_sad>")
                        end_sad_idx = row.index("<end_sad>")
                        sad_row = row[start_sad_idx:end_sad_idx]
                        sad_coded = self.parse_sad_od_frame_token(sad_row, max_frame, "sad")
                    except:
                        pass 

                if self.check_overlap_od:
                    start_od_idx = row.index("<start_od>")
                    end_od_idx = row.index("<end_od>")
                    od_row = row[start_od_idx:end_od_idx]
                    od_coded = self.parse_sad_od_frame_token(od_row, max_frame, "od")

                try:
                    start_idx = row.index("<start_sd>")
                    end_idx = row.index("<end_sd>")
                    row = row[start_idx+1:end_idx]
                except:
                    pass

                bit_coded = self.parse_frame_token(row, max_frame)

                if self.check_overlap_sad and sad_coded is not None:
                    bit_coded, sad_coded = np.asarray(bit_coded), np.asarray(sad_coded)
                    bit_coded = np.asarray(bit_coded * sad_coded, dtype=int)
                
                if self.check_overlap_od:
                    bit_coded, od_coded = np.asarray(bit_coded), np.asarray(od_coded)
                    for j, od_code in enumerate(od_coded):
                        curr_spks=[i+1 for i in range(5) if (bit_coded[j] >> i) & 1]
                        if od_code==0 and len(curr_spks)>1: #overlap conflict
                            if j>=1:
                                prev_spks = [i+1 for i in range(5) if (bit_coded[j-1] >> i) & 1]
                                if len(prev_spks) == 0: # choose previous spk
                                    bit_coded[j]=bit_coded[j-1]
                            else: # choose the first spk
                                bit_coded[j]=curr_spks[0] # may resolve the spk conflict later 

                rttm_out[(start_offset,end_offset)]= np.asarray(bit_coded, dtype=int)  
        
        elif self.hyp_format == "frame_timestamp":
            row=row.strip().split(" ")[1:]
            sd_row = row
            try:
                start_idx = row.index("<start_sd>")
                end_idx = row.index("<end_sd>")
                sd_row = row[start_idx:end_idx]
            except:
                pass
            sd_row = " ".join(sd_row)
            if self.check_overlap_sad:
                sad_row = row
                try:
                    start_idx = row.index("<start_sad>")
                    end_idx = row.index("<end_sad>")
                    sad_row = row[start_idx:end_idx]
                except:
                    pass
                sad_row = " ".join(sad_row)
                sad_out=self.parse_sad_token(sad_row, start_offset)
                rttm_out=self.parse_event_token(sd_row, rttm_out, start_offset, sad_out)

            else:
                sd_row = row
                try:
                    end_idx = row.index("<end_count>")+1
                    sd_row = row[end_idx:]
                except:
                    pass
                rttm_out=self.parse_frame_timestamp_token(sd_row, rttm_out, start_offset)


        return rttm_out

    def write_out_rttm(self, merged, out_file_name):
        out_lines = []
        out_file_id_set = set()
        for file_id in merged:
            for spk in merged[file_id]:
                for start, end in merged[file_id][spk]:
                    dur = round(end - start, 2)
                    if self.dataset_name in ["ami","aishell4","alimeeting","synthetic"]:
                        out_file_id = file_id.split("-")[0]
                        out_file_id_set.add(out_file_id)
                        if self.hyp_format == "frame" and not self.apply_local_speaker_matching:
                            start += float(file_id.split("-")[1])

                        out_lines.append(f"SPEAKER {out_file_id} 1 {start} {dur} <NA> <NA> {spk} <NA> <NA>\n")
                    else:
                        out_lines.append(f"SPEAKER {file_id} 1 {start} {dur} <NA> <NA> {spk} <NA> <NA>\n")
        self.write_file(out_lines, os.path.join(self.output_dir, out_file_name))
    
    def merge_two_rttm(self, rttm1, rttm2, rttm_out):
        rttm1_file = os.path.join(self.output_dir, rttm1)
        rttm2_file = os.path.join(self.output_dir, rttm2)
        rttm_out_file = os.path.join(self.output_dir, rttm_out)

        with open(rttm1_file, 'r') as f1, open(rttm2_file, 'r') as f2, open(rttm_out_file, 'w') as fout:
            fout.write(f1.read())
            fout.write(f2.read())

    def merge_ami(self, input_dict):
        out_dict={}
        for file_info in input_dict:
            file_id = file_info.split("-")[0]
            offset = float(file_info.split("-")[1])
            out_dict.setdefault(file_id, {})
            for spk in input_dict[file_info]:
                if self.hyp_format == "frame":
                    for start,end in input_dict[file_info][spk]:
                        out_dict[file_id].setdefault(spk,[]).append([start+offset, end+offset])                
                else:
                    out_dict[file_id].setdefault(spk,[]).extend(input_dict[file_info][spk])
        return out_dict

    def make_hyp_rttm(self):
        def extract_key(s):
            file_id = s.split("_")[-2]
            info = file_id.split("-")
            corpus, start, end = info[0], info[1], info[2]
            return (corpus, int(start), int(end))

        content = self.read_file(self.hyp_output_file)
        out_dict = {}
        
        if self.dataset_name in ["ami", "aishell4", "alimeeting", "synthetic"]:
            all_ids = []
            content_dict={}
            for line in content:
                all_ids.append(line.split(" ")[0])
                content_dict[line.split(" ")[0]]=line
            sorted_lines = sorted(all_ids, key=extract_key)

            all_rows = []
            for i,line in enumerate(sorted_lines):
                corpus, start, end = extract_key(line)
                if start % self.skip_interval != 0:
                    if i<len(sorted_lines)-1 and extract_key(sorted_lines[i+1])[0]!=corpus:
                        all_rows.append(content_dict[line])
                    if i==len(sorted_lines)-1:
                        all_rows.append(content_dict[line])
                else:
                        all_rows.append(content_dict[line])
        else:
            all_rows = content

        for row in all_rows:
            try:
                key = row.strip().split()[0]
                if self.dataset_name in ["ami", "aishell4", "alimeeting", "synthetic"]:
                    if self.dataset_name == "ami":
                        file_id = key.split("_")[-2]
                    elif self.dataset_name == "aishell4":
                        file_id = key.split("_")[-3]+"_"+key.split("_")[-2]
                    elif self.dataset_name == "alimeeting":
                        file_id = key.split("_")[-4]+"_"+key.split("_")[-3]+"_"+key.split("_")[-2]
                    elif self.dataset_name == "synthetic":
                        file_id = "_".join(key.split("_")[8:20])
                    start_offset, end_offset = float(file_id.split("-")[-2]), float(file_id.split("-")[-1])
                if self.dataset_name == "librimix": # librimix
                    file_id = re.findall((r'(\d+-\d+-\d+(?:_\d+-\d+-\d+)+)'), key)[0]
                    start_offset, end_offset = 0, self.reco2dur[file_id]
                
            except:
                continue
            curr_rttm = self.get_curr_rttm_output(row, file_id, start_offset, end_offset)
            out_dict[file_id] = curr_rttm

        # apply median filter on frame-based output and generated event-based output
        if self.hyp_format == "event" or self.hyp_format == "frame_timestamp":
            if self.apply_local_speaker_matching:
                matcher = SpeakerSegmentMatcher(out_dict, skip = self.skip_interval)
                merged = matcher.run()
            else:
                merged = self.merge_out_dict(out_dict, "event")
            frame_output = self.event2frame(merged)
            merged = self.apply_median_filter(frame_output)
        elif self.hyp_format == "frame": 
            if self.apply_local_speaker_matching:
                merged = self.frame2event(out_dict)
                matcher = SpeakerSegmentMatcher(merged, skip = self.skip_interval)
                merged = matcher.run()
                merged = self.event2frame(merged)
            else:
                merged = self.merge_out_dict(out_dict,"frame")

            merged = self.apply_median_filter(merged)
        elif self.hyp_format == "event_frame" or self.hyp_format == "event_frame_2tasks":
            merged={}
            for curr_format in ["event", "frame"]:
                curr_out_dict = {}
                for file_id in out_dict:
                    curr_out_dict[file_id]=out_dict[file_id][curr_format]
                merged[curr_format] = self.merge_out_dict(curr_out_dict, curr_format) 
                if curr_format == "event":
                    merged[curr_format] = self.event2frame(merged[curr_format])
                merged[curr_format] = self.apply_median_filter(merged[curr_format])       

        if self.apply_clustering:
            if not self.apply_local_speaker_matching:
                merged = self.merge_ami(merged)
            seg = self.get_segmentations(merged)
            rttm = self.apply_pyannote_clustering(seg)
            self.write_file(rttm, os.path.join(self.output_dir, "hyp.rttm"))
        else:
            if self.hyp_format=="event_frame" or self.hyp_format == "event_frame_2tasks":
                self.write_out_rttm(merged["event"], "hyp_event.rttm")
                self.write_out_rttm(merged["frame"], "hyp_frame.rttm")
                # merge two
                self.merge_two_rttm("hyp_event.rttm", "hyp_frame.rttm", "hyp_event_frame.rttm")
            else:
                self.write_out_rttm(merged, "hyp.rttm")


    def merge_out_dict(self, out_dict, curr_hyp_format):
        merged = {}
        if curr_hyp_format == "event":
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
        else: # frame
            for file_id in out_dict:
                for key in out_dict[file_id]:
                    merged[file_id] = out_dict[file_id][key]

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
                    labels[min(spk_id - 1, num_spks-1)][start_idx:end_idx] = 1

            try:
                index = labels[0, :].copy()
                for i in range(1, num_spks):
                    index += np.power(2, i) * labels[i, :]
                out_frame[file_id] = index # single stream
            except:
                print(file_id, "no conversation found!")
        return out_frame
            
    def frame2event(self, merged):
        out_event={}
        for file_id in merged:
            max_time = int(file_id.split("-")[2])-int(file_id.split("-")[1])
            start_offset = int(file_id.split("-")[1])
            num_frames = int(np.ceil(max_time / self.frame_duration))
            for key in merged[file_id]:
                bitcoded_stream = merged[file_id][key]
                break
            max_val = np.max(bitcoded_stream)
            num_spks = int(np.floor(np.log2(max_val))) + 1 if max_val > 0 else 1

            # Decode speaker activity
            for spk in range(num_spks):
                # Extract activity for this speaker
                activity = ((bitcoded_stream >> spk) & 1).astype(int)

                # Find segments (start, end)
                segments = []
                in_segment = False
                for i, val in enumerate(activity):
                    if val == 1 and not in_segment:
                        in_segment = True
                        start_frame = i
                    elif val == 0 and in_segment:
                        end_frame = i
                        segments.append((
                            start_frame * self.frame_duration + start_offset,
                            end_frame * self.frame_duration + start_offset
                        ))
                        in_segment = False
                if in_segment:
                    # Close any open segment at the end
                    segments.append([
                        start_frame * self.frame_duration + start_offset,
                        num_frames * self.frame_duration + start_offset
                    ])

                if segments:
                    out_event.setdefault(file_id, {})[spk+1] = segments  # match speaker indexing in event2frame

        return out_event

    def apply_median_filter(self, frame_mat):
        # It converts frame-level multi-speaker binary activity indicators into speaker segment timestamps, with median filtering applied to smooth the frame-wise predictions.
        filtered = {}
        for file_id in frame_mat:
            index = frame_mat[file_id]
            index = medfilt(index, kernel_size=self.kernel_size)
            spk_map_id = file_id
            if self.dataset_name in ["ami", "aishell4", "alimeeting","synthetic"]:
                spk_map_id = file_id.split("-")[0]
            num_spks=len(self.spk_map[spk_map_id])
            num_frames = index.shape[0]
            labels = np.array([(index.astype(int) >> i) & 1 for i in range(num_spks)])
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
        self.wav_dict = {line.strip().split()[0]: line.strip().split()[1] for line in wav_lines}

        segmentations = {}
        for file_id in merged:
            audio_path = self.wav_dict[file_id]
            rate, audio = read(audio_path)
            max_dur = len(audio) / rate
            num_spk = len(merged[file_id])
            total_frames = int((max_dur - self.spk_count_dur) / self.spk_count_skip)
            spk_activities = np.zeros((total_frames, num_spk))
            #for spk_id in merged[file_id]:
            spks = list(merged[file_id].keys())
            for i in range(len(merged[file_id].keys())):
                spk_id = spks[i]
                for start, end in merged[file_id][spk_id]:
                    s_idx, e_idx = int(start / self.spk_count_skip), int(end / self.spk_count_skip)
                    #spk_activities[s_idx:e_idx, spk_id - 1] = 1
                    spk_activities[s_idx:e_idx, i] = 1

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
        
        receptive_field = SlidingWindow(start=0.0, step=self.spk_count_skip, duration=self.spk_count_dur)
        frame = SlidingWindow(start=0.0, step=self.segment_skip, duration=self.segment_dur)
        diar_pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")
        #, cache_dir="/work/nvme/bbjs/jialuli3/cache")
        diar_pipeline.max_speakers = 4
        out_rttm = ""
        for file_id, seg in segmentations.items():
            seg_input = SlidingWindowFeature(seg, frame)
            diar_result = diar_pipeline(self.wav_dict[file_id], seg_input, receptive_field, file_id)
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
    print("Raw sys.argv:", sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_rttm_file", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--hyp_output_file", type=str)
    parser.add_argument("--hyp_format", type=str, choices=["event", "frame", "event_frame", "event_frame_2tasks", "frame_timestamp"])
    parser.add_argument("--tokenizer", type=str, default="diar_tokenizer", choices=["text_bpe", "diar_tokenizer"])
    parser.add_argument("--scoring_dir", type=str)
    parser.add_argument("--textgrid_dir", type=str, default=None)
    parser.add_argument("--test_wav_scp", type=str, default=None)
    parser.add_argument("--overlap_type", type=str, default=None)
    parser.add_argument("--reco2dur", type=str, default=None)
    parser.add_argument("--speaker_order_method", type=str, choices=["arrive", "most"])
    parser.add_argument("--spk_format", type=str, choices=["spk_id", "spk_idx"], default="spk_idx")
    parser.add_argument("--skip_interval", type=int, default=1)
    parser.add_argument("--apply_clustering", action="store_true")
    parser.add_argument("--use_multistream", action="store_true")
    parser.add_argument("--use_multistream_subtask", action="store_true")
    parser.add_argument("--use_ipu", action="store_true")
    parser.add_argument("--use_dur_format", action="store_true")
    parser.add_argument("--check_overlap_sad", action="store_true")
    parser.add_argument("--check_overlap_od", action="store_true")
    parser.add_argument("--apply_local_speaker_matching", action="store_true")

    args = parser.parse_args()

    scorer = DiarizationScorer(
        ref_rttm_file=args.ref_rttm_file,
        dataset_name=args.dataset_name,
        hyp_output_file=args.hyp_output_file,
        hyp_format=args.hyp_format,
        tokenizer=args.tokenizer,
        scoring_dir=args.scoring_dir,
        speaker_order_method=args.speaker_order_method,
        spk_format=args.spk_format,
        skip_interval=args.skip_interval,
        test_wav_scp=args.test_wav_scp,
        apply_clustering=args.apply_clustering,
        textgrid_dir=args.textgrid_dir,
        reco2dur=args.reco2dur,
        overlap_type=args.overlap_type,
        use_multistream=args.use_multistream,
        use_multistream_subtask=args.use_multistream_subtask,
        use_ipu=args.use_ipu,
        use_dur_format=args.use_dur_format,
        check_overlap_sad=args.check_overlap_sad,
        check_overlap_od=args.check_overlap_od,
        apply_local_speaker_matching=args.apply_local_speaker_matching
    )

    scorer.make_hyp_rttm()
    # if args.textgrid_dir:
    #     scorer.make_textgrid()


if __name__ == "__main__":
    main()
