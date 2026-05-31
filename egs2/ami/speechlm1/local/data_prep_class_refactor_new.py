import os
import re
import json
import argparse
import numpy as np
from scipy.io.wavfile import read, write
from collections import Counter
import logging
import pdb

class DiarizationPreprocessor:
    def __init__(self, 
        dataset_dir,
        output_dir, 
        wav_out_dir,
        output_format,
        dur=3, 
        skip=1, 
        mic="sdm",
        pit_method="arrive",
        spk_format="spk_idx",
        overlap_type="",
        curr_sets=["train","dev","test"],
        use_spk_count=False,
        use_spk_count_after=False,
        use_spk_count_after1=False,
        use_random_durs=False,
        use_multistream=False,
        use_multistream_subtask=False,
        use_multistream_2tasks=False,
        use_dur_format=False,
        use_sad=False,
        use_od=False,
        use_sad_after=False,
        use_od_after=False,
        use_od_before=False,
        use_ipu=False,
        use_local_speaker=False,
        use_sc_token=False, # use speaker change token in frame-based model
        use_sc_token_2types=False, # use speaker change token in frame-based model
        use_sc_token_1type=False, # use speaker change token in frame-based model
        use_timestamp=False, 
        use_padding=False,
        frame_res=0.1,
        context_length=0,
        random_durs=[30, 20, 10],
        dataset_name="ami",
        ):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.wav_out_dir = wav_out_dir
        self.output_format = output_format
        self.dur = dur
        self.skip = skip
        self.mic = mic
        self.pit_method=pit_method
        self.use_spk_count=use_spk_count
        self.use_spk_count_after=use_spk_count_after
        self.use_spk_count_after1=use_spk_count_after1
        self.spk_format=spk_format
        self.frame_res=frame_res
        self.text_out_event = None # used to convert to frame based
        self.use_random_durs = use_random_durs
        self.use_multistream = use_multistream
        self.use_multistream_subtask = use_multistream_subtask
        self.use_multistream_2tasks = use_multistream_2tasks
        if self.use_multistream_subtask or self.use_multistream_2tasks:
            self.use_multistream=True
        self.use_dur_format=use_dur_format
        self.use_sc_token = use_sc_token
        self.use_sc_token_2types = use_sc_token_2types
        self.use_sc_token_1type = use_sc_token_1type
        self.use_sad = use_sad
        self.use_od = use_od
        self.use_sad_after = use_sad_after
        self.use_od_after = use_od_after
        self.use_od_before = use_od_before
        self.use_ipu=use_ipu
        self.use_local_speaker=use_local_speaker
        self.use_timestamp = use_timestamp
        self.use_padding=use_padding
        self.random_durs = random_durs
        self.dataset_name=dataset_name
        self.overlap_type=overlap_type
        self.context_length=context_length
        self.target_length = self.dur - 2*self.context_length

        # get speaker orders
        self.speak_order_maps={}
        self.rttm_files={}
        self.file_ids={}
        self.segment_files={}
        self.reco2dur={}
        
        self.sc_token_dict = None
        if self.use_sc_token:
            self.sc_token_dict={"1": ["<sc_start>","<sc_end>"], \
                        "2": ["<overlap_sc_start_2>","<overlap_sc_end_2>"],\
                        "3": ["<overlap_sc_start_3>","<overlap_sc_end_3>"],\
                        "others": ["<overlap_sc_start>","<overlap_sc_end>"]                            
                        }
        elif self.use_sc_token_2types:
            self.sc_token_dict={"1": ["<sc_start>","<sc_end>"], \
                            "2": ["<overlap_sc_start>","<overlap_sc_end>"],\
                            "3": ["<overlap_sc_start>","<overlap_sc_end>"],\
                            "others": ["<overlap_sc_start>","<overlap_sc_end>"]                            
                            }
        elif self.use_sc_token_1type:
            self.sc_token_dict={"1": ["<sc_start>","<sc_end>"], \
                            "2": ["<sc_start>","<sc_end>"],\
                            "3": ["<sc_start>","<sc_end>"],\
                            "others": ["<sc_start>","<sc_end>"]                                
                            }
        if self.dataset_name in ["ami","aishell4","alimeeting"]:
            for curr_set in curr_sets:
                if self.use_ipu:
                    self.rttm_files[curr_set] = os.path.join(self.output_dir, curr_set, "sorted.ipu_rttm")
                else:
                    self.rttm_files[curr_set] = os.path.join(self.output_dir, curr_set, "sorted.rttm")
                segment_file_name = f"segments_dur{self.dur}_skip{self.skip}"
                if self.context_length>0:
                    segment_file_name += f"_context{self.context_length}"
                self.segment_files[curr_set]=os.path.join(self.output_dir, curr_set, segment_file_name)
                if self.use_random_durs:
                    self.segment_files[curr_set]=os.path.join(self.output_dir, curr_set, \
                        "segments_random_dur_"+"_".join(str(x) for x in self.random_durs))
                    print(self.segment_files[curr_set])
                if self.dataset_name == "ami":
                    self.file_ids[curr_set]=f"local/split_{curr_set}.orig"
                else:
                    self.file_ids[curr_set]=os.path.join(self.output_dir, curr_set, "file_id")

                self.speak_order_maps[curr_set]=self.get_spk_order(self.rttm_files[curr_set], self.pit_method)
        else: # librimix
            for curr_set in curr_sets:
                if self.use_ipu:
                    self.rttm_files[curr_set] = os.path.join(self.output_dir, curr_set, "sorted.ipu_rttm")
                else:
                    self.rttm_files[curr_set] = os.path.join(self.output_dir, curr_set, "sorted.rttm")
                self.speak_order_maps[curr_set]=self.get_spk_order(self.rttm_files[curr_set], self.pit_method)
                self.reco2dur[curr_set] = os.path.join(self.output_dir, curr_set, "reco2dur")

    @staticmethod
    def read_file(file_name):
        with open(file_name, "r") as f:
            return f.readlines()

    @staticmethod
    def write_file(content, file_name):
        with open(file_name, "w") as f:
            f.writelines(content)

    @staticmethod
    def get_overlap_interval(s1,e1,s2,e2):
        if s1<=e2 and s2<=e1:
            return (max(s1,s2),min(e1,e2))
        return None

    def get_spk_order(self, rttm_file, method="arrive"):
        '''
        get speaker order based on rttm file
        '''
        #method can be arrive_time_order or most_time_order
        rttm_rows = self.read_file(rttm_file)
        arr_time_spk_order={}
        most_time_spk_order={}
        count_time={}

        for curr_rttm in rttm_rows:
            curr_rttm = curr_rttm.strip()
            curr_id = curr_rttm.split()[1]
            curr_spk = curr_rttm.split()[-3]
            dur = float(curr_rttm.split()[-6])
            if curr_id not in arr_time_spk_order:
                arr_time_spk_order[curr_id]={}
                most_time_spk_order[curr_id]={}
                count_time[curr_id]=Counter()

            if curr_spk not in arr_time_spk_order[curr_id]:
                arr_time_spk_order[curr_id][curr_spk]=len(arr_time_spk_order[curr_id])+1

            count_time[curr_id][curr_spk]+=dur
        
        for curr_id in count_time:
            sorted_counter=count_time[curr_id].most_common()
            for curr_spk,_ in sorted_counter:
                most_time_spk_order[curr_id][curr_spk]=len(most_time_spk_order[curr_id])+1

        if method=="arrive":
            return arr_time_spk_order

        return most_time_spk_order

    def prep_wav_scp_aishell_alimeeting(self, curr_set):
        os.makedirs(os.path.join(self.output_dir, curr_set), exist_ok=True)
        os.makedirs(self.wav_out_dir, exist_ok=True)

        output_path = os.path.join(self.output_dir, curr_set, "wav.scp")
        segment_file_name = f"segments_dur{self.dur}_skip{self.skip}"
        if self.context_length>0:
            segment_file_name += f"_context{self.context_length}"

        output_segment_path = os.path.join(self.output_dir, curr_set, segment_file_name)

        file_id = self.file_ids[curr_set]
        all_ids = self.read_file(file_id)
        wav_files = self.read_file(output_path)
        wav_files_dict={}
        for row in wav_files:
            row = row.strip("\n")
            wav_files_dict[row.split()[0]]=row.split()[1]
        out, out_segments = [], []

        for curr_id, wav_filename in wav_files_dict.items():
            rate, data = read(wav_filename)
            print(wav_filename, data.shape)
            if data.ndim > 1: # handle two channel wav files
                data = data[:, 0]
                new_wav_path = os.path.join(self.wav_out_dir, os.path.basename(curr_wav_file))
                write(new_wav_path, rate, data)
                out[-1] = f"{curr_id} {new_wav_path}\n"

            total_dur = int(len(data) / rate) + 1
            for time_idx in range(0-self.context_length, total_dur+self.context_length, self.skip):
                start, end = int(float(time_idx)), int(float(min(time_idx + self.dur, len(data) / rate)))
                start = max(0, start)
                end = min(total_dur, end)
                if end - start > self.target_length//2:
                    if self.context_length>0:
                        ref_start = start + self.context_length
                        if end-start<self.dur and start==0:
                            ref_start = start                        
                        ref_end = min(ref_start + self.target_length, end)
                        out_segments.append(f"{curr_id}-{ref_start}-{ref_end}-{start}-{end} {curr_id} {start} {end}\n")        
                    else:
                        out_segments.append(f"{curr_id}-{start}-{end} {curr_id} {start} {end}\n")
            
            if end < len(data) / rate:
                start = end
                end = int(len(data) / rate)
                if end - start > self.target_length//2:
                    if self.context_length>0:
                        ref_start = start + self.context_length
                        ref_end = min(ref_start + self.target_length, end)
                        out_segments.append(f"{curr_id}-{start}-{end}-{ref_start}-{ref_end} {curr_id} {start} {end}\n")   
                    else:
                        out_segments.append(f"{curr_id}-{start}-{end} {curr_id} {start} {end}\n")

        if not os.path.exists(output_path):
            self.write_file(out, output_path)
        self.write_file(out_segments, output_segment_path)

    def prep_wav_scp(self, curr_set):
        os.makedirs(os.path.join(self.output_dir, curr_set), exist_ok=True)
        os.makedirs(self.wav_out_dir, exist_ok=True)

        pdb.set_trace()
        output_path = os.path.join(self.output_dir, curr_set, "wav.scp")
        segment_file_name = f"segments_dur{self.dur}_skip{self.skip}"
        if self.context_length>0:
            segment_file_name += f"_context{self.context_length}"

        output_segment_path = os.path.join(self.output_dir, curr_set, segment_file_name)

        file_id = self.file_ids[curr_set]
        all_ids = self.read_file(file_id)
        out, out_segments = [], []

        for curr_id in map(str.strip, all_ids):
            if curr_id in ["IS1003b", "IS1007d", "IB4005"]:
                continue

            wav_filename = f"{curr_id}.Array1-01.wav" if self.mic == "sdm" else f"{curr_id}.Mix-Headset.wav"
            curr_wav_file = os.path.join(self.dataset_dir, curr_id, "audio", wav_filename)

            out.append(f"{curr_id} {curr_wav_file}\n")
            rate, data = read(curr_wav_file)
            print(curr_wav_file, data.shape)

            if data.ndim > 1: # handle two channel wav files
                data = data[:, 0]
                new_wav_path = os.path.join(self.wav_out_dir, os.path.basename(curr_wav_file))
                write(new_wav_path, rate, data)
                out[-1] = f"{curr_id} {new_wav_path}\n"

            total_dur = int(len(data) / rate) + 1
            for time_idx in range(0-self.context_length, total_dur+self.context_length, self.skip):
                start, end = int(float(time_idx)), int(float(min(time_idx + self.dur, len(data) / rate)))
                start = max(0, start)
                end = min(total_dur, end)
                if end - start > self.target_length//2:
                    if self.context_length>0:
                        ref_start = start + self.context_length
                        if end-start<self.dur and start==0:
                            ref_start = start                        
                        ref_end = min(ref_start + self.target_length, end)
                        out_segments.append(f"{curr_id}-{ref_start}-{ref_end}-{start}-{end} {curr_id} {start} {end}\n")        
                    else:
                        out_segments.append(f"{curr_id}-{start}-{end} {curr_id} {start} {end}\n")
            
            if end < len(data) / rate:
                start = end
                end = int(len(data) / rate)
                if end - start > self.target_length//2:
                    if self.context_length>0:
                        ref_start = start + self.context_length
                        ref_end = min(ref_start + self.target_length, end)
                        out_segments.append(f"{curr_id}-{start}-{end}-{ref_start}-{ref_end} {curr_id} {start} {end}\n")   
                    else:
                        out_segments.append(f"{curr_id}-{start}-{end} {curr_id} {start} {end}\n")

        if not os.path.exists(output_path):
            self.write_file(out, output_path)
        self.write_file(out_segments, output_segment_path)

    def prep_random_wav_scp(self, curr_set):
        os.makedirs(os.path.join(self.output_dir, curr_set), exist_ok=True)
        os.makedirs(self.wav_out_dir, exist_ok=True)

        output_path = os.path.join(self.output_dir, curr_set, "wav.scp")
        output_segment_path = os.path.join(self.output_dir, curr_set, "segments_random_dur_"+"_".join(str(x) for x in self.random_durs))

        file_id = self.file_ids[curr_set]
        all_ids = self.read_file(file_id)
        out, out_segments = [], {}

        for curr_id in map(str.strip, all_ids):
            if curr_id in ["IS1003b", "IS1007d", "IB4005"]:
                continue

            wav_filename = f"{curr_id}.Array1-01.wav" if self.mic == "sdm" else f"{curr_id}.Mix-Headset.wav"
            curr_wav_file = os.path.join(self.dataset_dir, curr_id, "audio", wav_filename)

            out.append(f"{curr_id} {curr_wav_file}\n")
            rate, data = read(curr_wav_file)
            print(curr_wav_file, data.shape)

            if data.ndim > 1: # handle two channel wav files
                data = data[:, 0]
                new_wav_path = os.path.join(self.wav_out_dir, os.path.basename(curr_wav_file))
                write(new_wav_path, rate, data)
                out[-1] = f"{curr_id} {new_wav_path}\n"

            total_dur = int(len(data) / rate) + 1

            out_segments.setdefault(curr_id, {})
            # generate 30s duration first 
            num_utt = np.inf
            for random_dur in self.random_durs:
                start, end = 0, random_dur
                out_segments[curr_id].setdefault(random_dur, [])
                out_segments[curr_id][random_dur].append(f"{curr_id}-{start}-{end} {curr_id} {start} {end}\n")

                #while end < total_dur and len(out_segments[curr_id][random_dur])<num_utt:
                while end < total_dur:
                    offset = np.random.randint(5, 15) # increment 5-15 seconds every time
                    start = start + offset
                    end = min(start+random_dur, total_dur)
                    if end-start>5:
                        out_segments[curr_id][random_dur].append(f"{curr_id}-{start}-{end} {curr_id} {start} {end}\n")
                
                if np.isinf(num_utt): # set num_utt at the largest duration
                    num_utt = len(out_segments[curr_id][random_dur])
                print(curr_id, random_dur, len(out_segments[curr_id][random_dur]))
            
        final_out_segments=[]
        for curr_id in out_segments:
            for curr_dur in out_segments[curr_id]:
                final_out_segments.extend(out_segments[curr_id][curr_dur])
        if not os.path.exists(output_path):
                self.write_file(out, output_path)
        self.write_file(final_out_segments, output_segment_path)

    def get_multistream_event_string(self, spk_out, curr_id, curr_string):
        all_events = re.findall((r'(<spk\d> <bot> <\d+\.\d+> <\d+\.\d+> <eot>)'), curr_string) + \
                        re.findall((r'(<overlap_spk[_\d]+> <bot> <\d+\.\d+> <\d+\.\d+> <eot>)'), curr_string) # assume as diar tokens
        for curr_event in all_events:
            parts = curr_event.split()
            spk_idx = parts[0].strip("<").strip(">")
            if spk_idx not in spk_out:
                spk_out[spk_idx] = f"<{spk_idx}> "
            spk_out[spk_idx] += " ".join(parts[1:])+" "
        out_string = f"{curr_id} "
        for spk_idx in spk_out:
            out_string+=f"{spk_out[spk_idx]}| "
        out_string=out_string[:-2]+"\n"
        return out_string

    def remap_local_speaker(self, out_event):
        out_event_local = {"sd":{}, "count":{}, "sad":{}, "od":{}}
        for curr_key in out_event["sd"]:
            unique_spks = []
            for global_spk_id, offset_start, offset_end in out_event["sd"][curr_key]:
                if global_spk_id not in unique_spks:
                    unique_spks.append(global_spk_id)
            for global_spk_id, offset_start, offset_end in out_event["sd"][curr_key]:
                out_event_local["sd"].setdefault(curr_key, []).append((f"<spk{unique_spks.index(global_spk_id)+1}>", offset_start, offset_end))
            for global_spk_id in out_event["count"][curr_key]:
                out_event_local["count"].setdefault(curr_key, {})
                out_event_local["count"][curr_key][f"<spk{unique_spks.index(global_spk_id)+1}>"]=out_event["count"][curr_key][global_spk_id]
            # add sad and od keys back
            out_event_local["sad"].setdefault(curr_key, [])
            out_event_local["od"].setdefault(curr_key, [])
        return out_event_local

    def get_ami_info(self, curr_set):
        # initialization of dictionary
        out_event = {"count":{}, "sad":{}, "od":{}, "sd":{}}
        out_frame = {"count":{}, "sad":{}, "od":{}, "sd":{}, "mat":{}}

        # read necessary files
        segment_file=self.segment_files[curr_set]
        rttm_rows = self.read_file(self.rttm_files[curr_set])
        segment_rows = self.read_file(segment_file)

        speak_order_map = self.speak_order_maps[curr_set]

        for row in segment_rows:
            parts = row.strip().split()
            segment_id = parts[0]
            curr_id = parts[1]
            start = int(float(parts[2]))
            end = int(float(parts[3]))
            curr_key = segment_id
            out_event["sd"].setdefault(curr_key, [])
            out_event["count"].setdefault(curr_key, Counter())
            out_event["sad"].setdefault(curr_key, [])
            out_event["od"].setdefault(curr_key, [])

        prev_id = None
        for row in rttm_rows:
            parts = row.strip().split()
            curr_id = parts[1]
            rttm_start = float(parts[-7])
            if self.use_ipu:
                rttm_end = float(parts[-6])
            else:
                dur = float(parts[-6])
                rttm_end = rttm_start + dur
            curr_spk = parts[-3]
            curr_type = parts[-2] # used for ipu
            spk_id = speak_order_map[curr_id][curr_spk]
            # fill out the text transcript
            if curr_id!=prev_id:
                curr_keys = [key for key in out_event["sd"] if key.startswith(curr_id)]
                prev_id = curr_id
            for curr_key in curr_keys:
                curr_id_ = curr_key.split("-")[0]
                start,end=float(curr_key.split("-")[1]), float(curr_key.split("-")[2])
                overlap_interval = self.get_overlap_interval(start, end, rttm_start, rttm_end)

                if overlap_interval is not None:
                    offset_start, offset_end = round(overlap_interval[0]-start,1), round(overlap_interval[1]-start,1)
                    if offset_end-offset_start<0.1: continue # at least 20ms

                    out_event["sd"][curr_key].append((f"<spk{spk_id}>", offset_start, offset_end)) # spk turn including full range
                    out_event["count"][curr_key][f"<spk{spk_id}>"]+=1

        # convert event2frame
        out_event, out_frame = self.event2frame(out_event, out_frame, speak_order_map)
        return out_event, out_frame

    def event2frame(self, out_event, out_frame, speak_order_map):
        for curr_id in out_event["sd"]:
            if self.dataset_name in ["ami", "aishell4", "alimeeting"]:
                dur = float(curr_id.split("-")[2])-float(curr_id.split("-")[1])
                curr_id_ = curr_id.split("-")[0]
                num_spks = len(speak_order_map[curr_id_])
                frame_mat = np.zeros((num_spks, int(dur/self.frame_res)))
            else:
                dur = self.dur_dict[curr_id]
                num_spks = len(speak_order_map[curr_id])
                frame_mat = np.zeros((num_spks, int(dur/self.frame_res)))
            frame_single_stream = [] # frame list for appending current tokens
            sad_single_stream = [] # frame list for appending sad token
            od_single_stream = [] # frame list for appending od token
            
            for spk_id, curr_start, curr_end in out_event["sd"][curr_id]:
                curr_start_idx = round(curr_start/self.frame_res)
                curr_end_idx = round(curr_end/self.frame_res)
                spk_id = int(spk_id.strip("<").strip(">")[-1])
                frame_mat[spk_id-1][curr_start_idx:curr_end_idx]=1

            prev_token, curr_token = None, None
            prev_sad_token, sad_token, prev_od_token, od_token = None, None, None, None
            prev_start, prev_sad_start, prev_od_start = 0.0, 0.0, 0.0 

            for i in range(frame_mat.shape[1]):
                curr_active_spks=[]
                for j in range(num_spks):
                    if frame_mat[j][i]==1:
                        curr_active_spks.append(j+1)
                if len(curr_active_spks)==0: # silence
                    curr_token = "<sil>"
                    sad_token = "<sil>"
                    od_token = "<no_overlap>"
                elif len(curr_active_spks)==1:
                    curr_token = f"<spk{curr_active_spks[0]}>"
                    sad_token = "<spk>"
                    od_token = "<no_overlap>"
                elif len(curr_active_spks)<=3:
                    spk_list = "_".join(str(s) for s in curr_active_spks)
                    curr_token = f"<overlap_spk_{spk_list}>"
                    sad_token = "<spk>"
                    od_token = f"<overlap>"                   
                else:
                    curr_token = "<overlap_4_more>" # 4 or more
                    sad_token = "<spk>"
                    od_token = "<overlap>"    

                if prev_token != curr_token and prev_token is not None: # add overlapping information on SC
                    if prev_token!="<sil>":
                        if prev_token.startswith("<overlap") and self.overlap_type == "ovl":
                            out_event["sd"][curr_id].append((prev_token, prev_start, round(i*self.frame_res,1)))
                            out_event["count"][curr_id].setdefault(prev_token, 0)
                            out_event["count"][curr_id][prev_token]+=1
                               
                    prev_start = round(i*self.frame_res,1)

                # add sad information for event-based model
                if prev_sad_token != sad_token and prev_sad_token is not None: # add overlapping information
                    if prev_sad_token!="<sil>":
                        out_event["sad"][curr_id].append((prev_sad_token, prev_sad_start, round(i*self.frame_res,1)))
                    prev_sad_start = round(i*self.frame_res,1)

                # add overlap detection information for event-based model
                if prev_od_token != od_token and prev_od_token is not None: # add overlapping information
                    if prev_od_token!="<no_overlap>":
                        out_event["od"][curr_id].append((prev_od_token, prev_od_start, round(i*self.frame_res,1)))
                    prev_od_start = round(i*self.frame_res,1)                

                prev_token = curr_token
                prev_sad_token = sad_token
                prev_od_token = od_token
                frame_single_stream.append(curr_token)
                sad_single_stream.append(sad_token)
                od_single_stream.append(od_token)
            
            # handle last frame
            if prev_sad_token =="<spk>": # add sad information
                out_event["sad"][curr_id].append((prev_sad_token, prev_sad_start, round((i+1)*self.frame_res,1)))
            if prev_od_token =="<overlap>": # add overlapping information
                out_event["od"][curr_id].append((prev_od_token, prev_od_start, round((i+1)*self.frame_res,1)))
           
            out_frame["sd"][curr_id]=frame_single_stream
            out_frame["mat"][curr_id]=frame_mat
            out_frame["sad"][curr_id]=sad_single_stream
            out_frame["od"][curr_id]=od_single_stream
            out_frame["count"]=out_event["count"]
        return out_event, out_frame

    def get_librimix_info(self, curr_set):
        # initialization of dictionary
        out_event = {"count":{}, "sad":{}, "od":{}, "sd":{}}
        out_frame = {"count":{}, "sad":{}, "od":{}, "sd":{}, "mat":{}}

        # read necessary files
        rttm_rows = self.read_file(self.rttm_files[curr_set])
        speak_order_map = self.speak_order_maps[curr_set]
        dur_file = self.read_file(self.reco2dur[curr_set])
        self.dur_dict = {}
        for row in dur_file:
            row=row.strip()
            self.dur_dict[row.split(" ")[0]]=float(row.split(" ")[1])

        # get rttm-based event segments
        for row in rttm_rows:
            parts = row.strip().split()
            curr_id = parts[1]
            curr_spk = parts[-3]
            spk_id = speak_order_map[curr_id][curr_spk]

            out_event["sd"].setdefault(curr_id, [])
            out_event["count"].setdefault(curr_id, Counter())
            out_event["sad"].setdefault(curr_id, [])
            out_event["od"].setdefault(curr_id, [])
            
            rttm_start = round(float(parts[-7]),1)
            dur = float(parts[-6])
            rttm_end = round(rttm_start + dur, 1)
            out_event["sd"][curr_id].append((f"<spk{spk_id}>", rttm_start, rttm_end)) # spk turn including full range
            out_event["count"][curr_id][f"<spk{spk_id}>"]+=1                
        
        # convert event2frame
        out_event, out_frame = self.event2frame(out_event, out_frame, speak_order_map)

        return out_event, out_frame

    def convert_frame_output(self, curr_frame_list):
        curr_frame_stream = []
        prev_token = None
        for i, curr_token in enumerate(curr_frame_list):
            if i==0:
                # compute the first frame status
                prev_token = curr_token
                if prev_token.startswith("<sil"):
                    prev_num_spks = 0
                elif prev_token.startswith("<spk"):
                    prev_num_spks = 1
                elif prev_token.startswith("<overlap_spk"):
                    prev_num_spks = len(prev_token.strip("<").strip(">").split("_")[2:])
                else:
                    prev_num_spks = 4

                sc_token = None
                if self.sc_token_dict is not None:
                    if prev_num_spks <=3 and prev_num_spks >= 1:
                        sc_token=self.sc_token_dict[str(prev_num_spks)][0]
                    elif prev_num_spks >3:
                        sc_token=self.sc_token_dict["others"][0]

                if prev_num_spks>0 and self.use_timestamp:
                    curr_frame_stream.append(f"<{round(i*self.frame_res, 1)}>")
                if sc_token is not None:
                    curr_frame_stream.append(sc_token)

            else:
                # compute current frame status
                if curr_token.startswith("<sil"):
                    curr_num_spks = 0
                elif curr_token.startswith("<spk"):
                    curr_num_spks = 1
                elif curr_token.startswith("<overlap_spk"):
                    curr_num_spks = len(curr_token.strip("<").strip(">").split("_")[2:])
                else:
                    curr_num_spks = 4

                if (self.sc_token_dict is not None) and prev_token != curr_token: # speaker change
                    if prev_num_spks <=3 and prev_num_spks >= 1:
                        curr_frame_stream.append(self.sc_token_dict[str(prev_num_spks)][1])
                    elif prev_num_spks >3:
                        curr_frame_stream.append(self.sc_token_dict["others"][1])

                    if prev_num_spks>0 and self.use_timestamp:
                        curr_frame_stream.append(f"<{round(i*self.frame_res, 1)}>")

                    if curr_num_spks <=3 and curr_num_spks >= 1:
                        curr_frame_stream.append(self.sc_token_dict[str(curr_num_spks)][0])
                    elif curr_num_spks >3:
                        curr_frame_stream.append(self.sc_token_dict["others"][0])

                    if curr_num_spks>0 and self.use_timestamp:
                        curr_frame_stream.append(f"<{round(i*self.frame_res, 1)}>")

                prev_token = curr_token
                prev_num_spks=curr_num_spks

            curr_frame_stream.append(curr_token)
        
        if self.sc_token_dict is not None:
            if prev_num_spks <=3 and prev_num_spks >= 1:
                curr_frame_stream.append(self.sc_token_dict[str(prev_num_spks)][1])
            elif prev_num_spks >3:
                curr_frame_stream.append(self.sc_token_dict["others"][1])

        if prev_num_spks>0 and self.use_timestamp:
            curr_frame_stream.append(f"<{round((i+1)*self.frame_res, 1)}>")

        if self.use_padding and len(curr_frame_list)<300: # 30s
            curr_frame_stream.extend(["<sil>"]*(300-len(curr_frame_list)))

        return curr_frame_stream
    
    def convert_sad_output(self, curr_sad_list):
        sad_out =""
        if self.sc_token_dict is not None:
            prev_token = None
            if curr_sad_list[0]=="<spk>": # handle first frame
                sad_out += "<spk_start> "
                if self.use_timestamp:
                    sad_out += "<0.0> "
            for i,curr_token in enumerate(curr_sad_list):
                if curr_token!=prev_token and prev_token is not None:
                    if curr_token == "<spk>":
                        sad_out += "<spk_start> "
                    else: # sil
                        sad_out += "<spk_end> "
                    if self.use_timestamp:
                        sad_out += f"<{round(i*self.frame_res, 1)}> "

                sad_out += curr_token+" "
                prev_token = curr_token
            if curr_token == "<spk>":
                sad_out += "<spk_end> "
                if self.use_timestamp:
                    sad_out += f"<{round((i+1)*self.frame_res, 1)}> "
        else:
            sad_out += " ".join(curr_sad_list)
        if self.use_padding and len(curr_sad_list)<300: # 30s
            padding_list = ["<sil>"]*(300-len(curr_sad_list))
            sad_out += " "+" ".join(padding_list)
        return sad_out

    def convert_od_output(self, curr_od_list):
        od_out =""
        if self.sc_token_dict is not None:
            prev_token = None
            if curr_od_list[0]=="<overlap>": # handle first frame
                od_out += "<overlap_start> "
                if self.use_timestamp:
                    od_out += "<0.0> "
            for i,curr_token in enumerate(curr_od_list):
                if curr_token!=prev_token and prev_token is not None:
                    if curr_token == "<overlap>":
                        od_out += "<overlap_start> "
                    else: # sil
                        od_out += "<overlap_end> "
                    if self.use_timestamp:
                        od_out += f"<{round(i*self.frame_res, 1)}> "

                od_out += curr_token+" "
                prev_token = curr_token
            if curr_token == "<overlap>":
                od_out += "<overlap_end> "
                if self.use_timestamp:
                    od_out += f"<{round((i+1)*self.frame_res, 1)}> "
        else:
            od_out += " ".join(curr_od_list)

        if self.use_padding and len(curr_od_list)<300: # 30s
            padding_list = ["<sil>"]*(300-len(curr_od_list))
            od_out += " "+" ".join(padding_list)
        return od_out

    def get_event_out(self, token, start, end):
        if self.use_dur_format:
            dur = round(float(end)-float(start),1)
            return f"{token} <s_{start}> <d_{dur}> "
        return f"{token} <bot> <{start}> <{end}> <eot> "

    def convert_output(self, out_event, out_frame):
        out={"event":[],"frame":[],"event_frame":[],"event_frame_2tasks":[]}

        if self.use_multistream:
            # compute event-based model
            for j, curr_id in enumerate(out_event["sd"]):
                curr_event_out = {"count":"", "sad":"", "od":"", "sd":""}
                curr_frame_out = {"count":"", "sad":"", "od":"", "sd":""}

                # add sad info
                if self.use_sad:
                    curr_frame_out["sad"] += "<start_sad> "
                    curr_frame_out["sad"] += self.convert_sad_output(out_frame["sad"][curr_id])
                    curr_frame_out["sad"] += " <end_sad> "

                # add sad info
                if self.use_od:
                    curr_frame_out["od"] += "<start_od> "
                    curr_frame_out["od"] += self.convert_od_output(out_frame["od"][curr_id])
                    curr_frame_out["od"] += "<end_od> "

                # add counter info
                if self.use_spk_count or self.use_spk_count_after:
                    curr_event_out["count"] += "<start_count> " # single stream case
                    for spk_idx, count in out_event["count"][curr_id].items():
                        spk_idx = spk_idx.strip("<").strip(">")
                        if count>10 and self.target_length<=30:
                            print("extra count", curr_id, spk_idx, count)
                            count = min(10, count)
                        curr_event_out["count"] += f"<{spk_idx}_count> <{count}_count> "

                    curr_event_out["count"] += "<end_count> "
                    curr_frame_out["count"] = curr_event_out["count"]

                # put each task as one stream
                # compute event-based model
                if self.use_multistream_subtask: # put each task as one stream
                    curr_frame_stream = ""

                    if self.use_sad:
                        curr_frame_stream+=curr_frame_out["sad"] + "| "

                    if self.use_od:
                        curr_frame_stream+=curr_frame_out["od"] + "| "

                    if self.use_spk_count:
                        curr_frame_stream+=curr_frame_out["count"] + "| "

                    # compute frame-based model
                    # put each task as one stream
                    curr_frame_sd_stream = ["<start_sd>"]
                    curr_frame_sd_stream.extend(self.convert_frame_output(out_frame["sd"][curr_id]))
                    curr_frame_sd_stream.append("<end_sd>")
                    curr_frame_stream += " ".join(curr_frame_sd_stream)

                    if self.use_spk_count_after:
                        curr_frame_stream+=" | "+ curr_frame_out["count"] 

                    out["frame"].append(f"{curr_id} {curr_frame_stream}\n")


                else: # put each speaker as one stream, work less better than single stream
                    # compute frame-based model
                    curr_frame_stream = ""
                    if self.use_sad:
                        curr_frame_stream+=curr_frame_out["sad"] + "| "

                    if self.use_od:
                        curr_frame_stream+=curr_frame_out["od"] + "| "

                    if self.use_spk_count:
                        curr_frame_stream+=curr_frame_out["count"] + "| "
                        
                    frame_mat = out_frame["mat"][curr_id]
                    curr_frame_dict = {}
                    for spk_id in range(frame_mat.shape[0]):
                        curr_frame_dict[f"spk{spk_id+1}"]=[]
                        prev_token = None
                        for frame_idx in range(frame_mat.shape[1]):
                            if (prev_token != frame_mat[spk_id][frame_idx]) and (prev_token is not None) and (self.use_sc_token):
                                if prev_token == 0: curr_frame_dict[f"spk{spk_id+1}"].append(f"<sc_start>")
                                if prev_token == 1: curr_frame_dict[f"spk{spk_id+1}"].append(f"<sc_end>")

                            if frame_mat[spk_id][frame_idx] == 1:
                                curr_frame_dict[f"spk{spk_id+1}"].append(f"<spk{spk_id+1}>")
                            else:
                                curr_frame_dict[f"spk{spk_id+1}"].append(f"<sil>")

                            prev_token = frame_mat[spk_id][frame_idx]

                    sorted_spk = sorted(curr_frame_dict.keys())
                    for spk in sorted_spk:
                        curr_frame_stream += "<start_sd> "+ " ".join(curr_frame_dict[spk])+" <end_sd>"
                        curr_frame_stream+=" | "
                    curr_frame_stream = curr_frame_stream[:-3]

                    out["frame"].append(f"{curr_id} {curr_frame_stream}\n")
        else: # single stream case
            for j, curr_id in enumerate(out_event["sd"]):
                curr_event_out = ""
                curr_frame_out = ""

                # compute subtask output
                if self.use_od_before:
                    curr_event_out += "<start_od> "
                    for od_token, start, end in out_event["od"][curr_id]:
                        curr_event_out += self.get_event_out(od_token, start, end)

                    curr_event_out += "<end_od> "
                    curr_frame_out += "<start_od> "
                    curr_od_out = self.convert_od_output(out_frame["od"][curr_id])

                    curr_frame_out += curr_od_out + " <end_od> "

                if self.use_sad:
                    curr_event_out += "<start_sad> "
                    for sad_token, start, end in out_event["sad"][curr_id]:
                        curr_event_out += self.get_event_out(sad_token, start, end)

                    curr_event_out += "<end_sad> "
                    curr_frame_out += "<start_sad> "
                    sad_out = self.convert_sad_output(out_frame["sad"][curr_id])
                    curr_frame_out += sad_out + " <end_sad> "

                if self.use_od:
                    curr_event_out += "<start_od> "
                    for od_token, start, end in out_event["od"][curr_id]:
                        curr_event_out += self.get_event_out(od_token, start, end)

                    curr_event_out += "<end_od> "
                    curr_frame_out += "<start_od> "
                    curr_od_out = self.convert_od_output(out_frame["od"][curr_id])

                    curr_frame_out += curr_od_out + " <end_od> "
                
                if self.use_spk_count:
                    curr_event_out += "<start_count> " # single stream case
                    curr_frame_out += "<start_count> "
                    for spk_idx, count in out_event["count"][curr_id].items():
                        spk_idx = spk_idx.strip("<").strip(">")
                        if count>10 and self.target_length<=30:
                            print("extra count", curr_id, spk_idx, count)
                            count = min(10, count)
                        curr_event_out += f"<{spk_idx}_count> <{count}_count> "
                        curr_frame_out += f"<{spk_idx}_count> <{count}_count> "
                    
                    curr_event_out += "<end_count> "
                    curr_frame_out += "<end_count> "

                # compute event-based sd output
                curr_event_out+="<start_sd> "
                sorted_sd_data = sorted(out_event["sd"][curr_id], key=lambda x: x[1])

                for spk_id, start, end in sorted_sd_data:
                    if self.overlap_type=="" and spk_id.startswith("<overlap"):
                        continue

                    curr_event_out += self.get_event_out(spk_id, start, end)

                if len(sorted_sd_data) == 0:
                    if self.dataset_name in ["ami", "aishell4", "alimeeting"]:
                        dur = float(curr_id.split("-")[2])-float(curr_id.split("-")[1])
                    else:
                        dur = self.dur_dict[curr_id]
                    curr_event_out += self.get_event_out("<sil>", 0.0, round(float(dur),1))

                curr_event_out+="<end_sd> "

                # compute frame-based output
                if self.sc_token_dict is not None:
                    curr_frame_stream = ["<start_sd>"]
                    curr_frame_stream.extend(self.convert_frame_output(out_frame["sd"][curr_id]))
                    
                    curr_frame_stream.append("<end_sd>")
                    curr_frame_out += " ".join(curr_frame_stream)
                else:
                    if self.use_padding and len(out_frame["sd"][curr_id])<300: # 30s
                        padding_list=["<sil>"]*(300-len(out_frame["sd"][curr_id]))
                        curr_frame_out += "<start_sd> "+" ".join(out_frame["sd"][curr_id]) +" "+" ".join(padding_list)+" <end_sd> "
                    else:
                        curr_frame_out += "<start_sd> "+" ".join(out_frame["sd"][curr_id]) +" <end_sd> "

                if self.use_spk_count_after1:
                    curr_event_out += "<start_count> " # single stream case
                    curr_frame_out += " <start_count> "
                    for spk_idx, count in out_event["count"][curr_id].items():
                        spk_idx = spk_idx.strip("<").strip(">")
                        if count>10 and self.target_length<=30:
                            print("extra count", curr_id, spk_idx, count)
                            count = min(10, count)
                        curr_event_out += f"<{spk_idx}_count> <{count}_count> "
                        curr_frame_out += f"<{spk_idx}_count> <{count}_count> "
                    
                    curr_event_out += "<end_count> "
                    curr_frame_out += "<end_count> "

                if self.use_od_after:
                    curr_event_out += "<start_od> "
                    for od_token, start, end in out_event["od"][curr_id]:
                        curr_event_out += self.get_event_out(od_token, start, end)

                    curr_event_out += "<end_od> "
                    curr_frame_out += "<start_od> "
                    curr_od_out = self.convert_od_output(out_frame["od"][curr_id])

                    curr_frame_out += curr_od_out + " <end_od> "

                if self.use_spk_count_after:
                    curr_event_out += "<start_count> " # single stream case
                    curr_frame_out += " <start_count> "
                    for spk_idx, count in out_event["count"][curr_id].items():
                        spk_idx = spk_idx.strip("<").strip(">")
                        if count>10 and self.target_length<=30:
                            print("extra count", curr_id, spk_idx, count)
                            count = min(10, count)
                        curr_event_out += f"<{spk_idx}_count> <{count}_count> "
                        curr_frame_out += f"<{spk_idx}_count> <{count}_count> "
                    
                    curr_event_out += "<end_count> "
                    curr_frame_out += "<end_count> "

                if self.use_sad_after:
                    curr_event_out += "<start_sad> "
                    for sad_token, start, end in out_event["sad"][curr_id]:
                        curr_event_out += self.get_event_out(sad_token, start, end)

                    curr_event_out += "<end_sad> "
                    curr_frame_out += "<start_sad> "
                    sad_out = self.convert_sad_output(out_frame["sad"][curr_id])
                    curr_frame_out += sad_out + " <end_sad> "

                # write final output
                out["event"].append(f"{curr_id} {curr_event_out}\n")
                out["frame"].append(f"{curr_id} {curr_frame_out}\n")                
        return out

    def write_out_file(self, out):
        print(f"writing file...")
        out_file_name = f"diar_tokens_{self.output_format}"

        if self.output_format in ["frame","event_frame","event_frame_2tasks"]:
            if self.use_sc_token:
                out_file_name+=f"_sc"
            elif self.use_sc_token_2types:
                out_file_name+=f"_sc2"
            elif self.use_sc_token_1type:
                out_file_name+=f"_sc1"

        if self.use_od_before:
            out_file_name+=f"_od"   

        if self.use_sad:
            out_file_name+=f"_sad"

        if self.use_spk_count_after1:
            out_file_name+=f"_spk_count_after"

        if self.use_od_after:
            out_file_name+=f"_od_after"   

        if self.use_sad_after:
            out_file_name+=f"_sad_after"   

        if self.use_od:
            out_file_name+=f"_od"                

        if self.overlap_type!="":
            out_file_name+=f"_{self.overlap_type}"

        if self.use_spk_count:
            out_file_name+=f"_spk_count"

        if self.use_spk_count_after:
            out_file_name+=f"_spk_count_after"

        if self.use_padding:
            out_file_name+=f"_pad"

        if self.use_multistream:
            out_file_name+=f"_multi"

        if self.use_multistream_subtask:
            out_file_name+=f"_subtask"

        if self.use_multistream_2tasks:
            out_file_name+=f"_2tasks"

        if self.use_ipu:
            out_file_name+=f"_ipu"

        if self.use_timestamp:
            out_file_name+=f"_time"

        if self.use_dur_format:
            out_file_name+=f"_durf"

        if self.use_local_speaker:
            out_file_name+=f"_local"
        
        # if self.dataset_name=="ami":
        if self.dataset_name in ["ami", "aishell4", "alimeeting"]:
            if self.use_random_durs:
                out_file_name+="_random_"+"_".join(str(x) for x in self.random_durs)
            else:
                out_file_name+=f"_dur{self.dur}_skip{self.skip}"
            
            if self.context_length>0:
                out_file_name+=f"_context{self.context_length}"

        out_path=os.path.join(self.output_dir, curr_set, out_file_name)
        print(f"writting to file {out_path}")
        self.write_file(out[self.output_format], out_path)

    def prep_librimix(self, curr_set):
        out_event, out_frame = self.get_librimix_info(curr_set)
        out = self.convert_output(out_event, out_frame)
        self.write_out_file(out)

    def prep_ami(self, curr_set):
        out_event, out_frame = self.get_ami_info(curr_set)
        out = self.convert_output(out_event, out_frame)
        self.write_out_file(out)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare diarization wav.scp and segments files")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for Kaldi files")
    parser.add_argument("--output_format", type=str, required=True, choices=["event", "frame", "event_frame", "event_frame_2tasks"])
    parser.add_argument("--wav_out_dir", type=str, required=True, help="Directory to save single-channel wav files")
    parser.add_argument("--mic", type=str, default="sdm", choices=["sdm", "ihm"], help="Microphone type")
    parser.add_argument("--dur", type=int, default=3, help="Duration of each segment in seconds")
    parser.add_argument("--skip", type=int, default=1, help="Skip interval in seconds")
    parser.add_argument("--context_length", type=int, default=0, help="Duration of context length")
    parser.add_argument("--pit_method", type=str, default="arrive", choices=["arrive", "most"], help="speaker arrival order or most frequently speaker")
    parser.add_argument("--spk_format", type=str, default="spk_idx", choices=["spk_idx", "spk_id"])
    parser.add_argument("--overlap_type", type=str, default="", choices=["","od", "ovl"])
    parser.add_argument("--curr_sets", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--use_random_durs", action="store_true", help="whether to use random durations")
    parser.add_argument("--use_spk_count", action="store_true")
    parser.add_argument("--use_spk_count_after", action="store_true")
    parser.add_argument("--use_spk_count_after1", action="store_true")
    parser.add_argument("--use_sad", action="store_true")
    parser.add_argument("--use_sad_after", action="store_true")
    parser.add_argument("--use_od", action="store_true")
    parser.add_argument("--use_od_before", action="store_true")
    parser.add_argument("--use_od_after", action="store_true")
    parser.add_argument("--use_multistream", action="store_true")
    parser.add_argument("--use_multistream_subtask", action="store_true")
    parser.add_argument("--use_multistream_2tasks", action="store_true")
    parser.add_argument("--use_sc_token", action="store_true")
    parser.add_argument("--use_sc_token_2types", action="store_true")
    parser.add_argument("--use_sc_token_1type", action="store_true")
    parser.add_argument("--use_ipu", action="store_true")
    parser.add_argument("--use_timestamp", action="store_true")
    parser.add_argument("--use_local_speaker", action="store_true")
    parser.add_argument("--use_dur_format", action="store_true")
    parser.add_argument("--use_padding", action="store_true")
    parser.add_argument("--frame_res", type=float, default=0.1, help="frame resolution")
    parser.add_argument("--data_outputs", nargs="+", default=["wav", "text_event", "text_frame"])

    args = parser.parse_args()

    processor = DiarizationPreprocessor(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        output_format=args.output_format,
        wav_out_dir=args.wav_out_dir,
        mic=args.mic,
        dur=args.dur,
        skip=args.skip,
        context_length=args.context_length,
        pit_method=args.pit_method,
        spk_format=args.spk_format,
        overlap_type=args.overlap_type,
        curr_sets=args.curr_sets,
        use_spk_count=args.use_spk_count,
        use_spk_count_after=args.use_spk_count_after,
        use_spk_count_after1=args.use_spk_count_after1,
        use_sad=args.use_sad,
        use_od=args.use_od,
        use_sad_after=args.use_sad_after,
        use_od_after=args.use_od_after,
        use_od_before=args.use_od_before,
        use_random_durs=args.use_random_durs,
        use_multistream=args.use_multistream,
        use_multistream_subtask=args.use_multistream_subtask,
        use_multistream_2tasks=args.use_multistream_2tasks,
        use_sc_token=args.use_sc_token,
        use_sc_token_2types=args.use_sc_token_2types,
        use_sc_token_1type=args.use_sc_token_1type,
        use_dur_format=args.use_dur_format,
        use_ipu=args.use_ipu,
        use_local_speaker=args.use_local_speaker,
        use_timestamp=args.use_timestamp,
        use_padding=args.use_padding,
        frame_res=args.frame_res,
    )

    # prepare wav file
    for item in args.data_outputs:
        for curr_set in args.curr_sets:
            print(f"current set {curr_set}")
            if item == "wav":
                if args.dataset_name == "ami":
                    if args.use_random_durs:
                        print("preparing wav file")
                        processor.prep_random_wav_scp(curr_set)                
                    else:
                        processor.prep_wav_scp(curr_set)
                else:
                    processor.prep_wav_scp_aishell_alimeeting(curr_set)


            #if args.dataset_name == "ami":
            if args.dataset_name in ["ami", "aishell4", "alimeeting"]:
                processor.prep_ami(curr_set)
            else: # librimix
                processor.prep_librimix(curr_set)
