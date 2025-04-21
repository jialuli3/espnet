import os
import re
import json
import argparse
import numpy as np
from scipy.io.wavfile import read, write
from collections import Counter
import pdb

class DiarizationPreprocessor:
    def __init__(self, 
        dataset_dir,
        output_dir, 
        wav_out_dir,
        dur=3, 
        skip=1, 
        mic="sdm",
        pit_method="arrive",
        spk_format="spk_idx",
        curr_sets=["train","dev","test"],
        use_extra_info=False,
        use_random_durs=False,
        frame_res=0.1,
        random_durs=[30, 20, 15, 8],
        ):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.wav_out_dir = wav_out_dir
        self.dur = dur
        self.skip = skip
        self.mic = mic
        self.pit_method=pit_method
        self.use_extra_info=use_extra_info
        self.spk_format=spk_format
        self.frame_res=frame_res
        self.text_out_event = None # used to convert to frame based
        self.use_random_durs = use_random_durs
        self.random_durs = random_durs

        # get speaker orders
        self.speak_order_maps={}
        self.rttm_files={}
        self.file_ids={}
        self.segment_files={}

        for curr_set in curr_sets:
            self.rttm_files[curr_set] = os.path.join(self.output_dir, curr_set, "rttm")
            self.segment_files[curr_set]=os.path.join(self.output_dir, curr_set, f"segments_dur{dur}_skip{skip}")
            if self.use_random_durs:
                self.segment_files[curr_set]=os.path.join(self.output_dir, curr_set, \
                    f"segments_random_dur_{self.random_durs[0]}_{self.random_durs[1]}_{self.random_durs[2]}_{self.random_durs[3]}")
                print(self.segment_files[curr_set])
            self.file_ids[curr_set]=f"local/split_{curr_set}.orig"
            self.speak_order_maps[curr_set]=self.get_spk_order(self.rttm_files[curr_set], self.pit_method)

        # np.random.seed(2025)

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

    def prep_wav_scp(self, curr_set):
        os.makedirs(os.path.join(self.output_dir, curr_set), exist_ok=True)
        os.makedirs(self.wav_out_dir, exist_ok=True)

        output_path = os.path.join(self.output_dir, curr_set, "wav.scp")
        output_segment_path = os.path.join(self.output_dir, curr_set, f"segments_dur{self.dur}_skip{self.skip}")

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
            for time_idx in range(0, total_dur, self.skip):
                start, end = round(time_idx, 2), round(min(time_idx + self.dur, len(data) / rate), 2)
                if end - start >= 1:
                    out_segments.append(f"{curr_id}-{start}-{end} {curr_id} {start} {end}\n")

            if end < len(data) / rate:
                start = round(end, 2)
                end = round(len(data) / rate, 2)
                out_segments.append(f"{curr_id}-{start}-{end} {curr_id} {start} {end}\n")

        if not os.path.exists(output_path):
            self.write_file(out, output_path)
        self.write_file(out_segments, output_segment_path)


    def prep_random_wav_scp(self, curr_set):
        os.makedirs(os.path.join(self.output_dir, curr_set), exist_ok=True)
        os.makedirs(self.wav_out_dir, exist_ok=True)

        output_path = os.path.join(self.output_dir, curr_set, "wav.scp")
        output_segment_path = os.path.join(self.output_dir, curr_set, f"segments_random_dur_{self.random_durs[0]}_{self.random_durs[1]}_{self.random_durs[2]}_{self.random_durs[3]}")

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
                    if end-start>1:
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

    def get_text_out(self, text_out_dict, text_count_dict=None, format_type="event", output_format="text"):
        out=[]
        for curr_id in text_out_dict:
            for start,end in text_out_dict[curr_id]: 
                if text_out_dict[curr_id][(start,end)]=="" and format_type=="event":
                    if output_format=="text":
                        text_out_dict[curr_id][(start,end)]="<space>"
                    else: # diarization tokens
                        text_out_dict[curr_id][(start,end)]="<sil>"
                if text_count_dict is not None:
                    sorted_c=dict(sorted(text_count_dict[curr_id][(start,end)].items()))
                    counter_out = str(dict(sorted_c)).replace("'","")
                    out.append(f"{curr_id}-{int(start)}-{int(end)} {counter_out} {text_out_dict[curr_id][(start,end)]}\n")
                else:           
                    out.append(f"{curr_id}-{int(start)}-{int(end)} {text_out_dict[curr_id][(start,end)]}\n")
        return out

    def prep_event(self, curr_set, output_format):
        if not os.path.exists(self.segment_files[curr_set]):
            print("Prepare wav and segment files")
            self.prep_wav_scp(curr_set)

        segment_file=self.segment_files[curr_set]
        rttm_rows = self.read_file(self.rttm_files[curr_set])
        segment_rows = self.read_file(segment_file)

        speak_order_map = self.speak_order_maps[curr_set]

        # initialization of dictionary
        text_out_event = {}
        text_out_event_count = {}
        text_out_frame = {}

        for row in segment_rows:
            parts = row.strip().split()
            curr_id = parts[1]
            start = float(parts[2])
            end = float(parts[3])
            if curr_id not in text_out_event:
                text_out_event[curr_id]={} # initialize the output dictionary with start,end stamps
                text_out_event_count[curr_id]={}
            text_out_event[curr_id][(start,end)]=""
            text_out_event_count[curr_id][(start,end)]=Counter({f"<spk{key}>": 0 for key in range(1, len(speak_order_map[curr_id])+1)})

        for row in rttm_rows:
            parts = row.strip().split()
            curr_id = parts[1]
            rttm_start = float(parts[-7])
            dur = float(parts[-6])
            rttm_end = rttm_start + dur
            curr_spk = parts[-3]
            spk_id = speak_order_map[curr_id][curr_spk]

            # fill out the text transcript
            for start,end in text_out_event[curr_id]:
                overlap_interval = self.get_overlap_interval(start, end, rttm_start, rttm_end)
                if overlap_interval is not None:
                    offset_start, offset_end = round(overlap_interval[0]-start,1), round(overlap_interval[1]-start,1)
                    if offset_end-offset_start<0.1: continue # at least 20ms
                    text_out_event_count[curr_id][(start,end)][f"<spk{spk_id}>"]+=1
                    
                    if output_format=="text":
                        if self.spk_format=="spk_idx":
                            text_out_event[curr_id][(start,end)]+="{"+f"<spk{spk_id}> ({offset_start}, {offset_end})"+"} "
                        else:
                            text_out_event[curr_id][(start,end)]+="{"+f"<{curr_spk}> ({offset_start}, {offset_end})"+"} "
                    else: # output_format diar_tokens
                        assert self.spk_format=="spk_idx"
                        text_out_event[curr_id][(start,end)]+=f"<spk{spk_id}> <bot> <{offset_start}> <{offset_end}> <eot> "


        time_info = "_".join(segment_file.split("_")[-2:])
        if self.use_random_durs:
            time_info = "_".join(segment_file.split("_")[-5:])
        out_file_event_name=f"{output_format}_{self.pit_method}_event_{time_info}"

        if self.spk_format!="spk_idx":
            out_file_event_name+=f"_{self.spk_format}"
        if self.use_extra_info:
            out_file_event_name+=f"_extra_info"

        out_text_event_path=os.path.join(self.output_dir, curr_set, out_file_event_name)

        if self.use_extra_info:
            text_event_out = self.get_text_out(text_out_event, text_count_dict=text_out_event_count, format_type="event", output_format=output_format)
        else:
            text_event_out = self.get_text_out(text_out_event, format_type="event", output_format=output_format)

        self.write_file(text_event_out, out_text_event_path)
        print(f"write event file to {out_text_event_path}")
        self.text_out_event = text_out_event

    def prep_frame(self, curr_set, output_format):
        # convert event-based to frame-based model
        speak_order_map = self.speak_order_maps[curr_set]
        segment_file=self.segment_files[curr_set]
        text_out_frame={}

        for curr_id in self.text_out_event:
            text_out_frame[curr_id]={}
            for start,end in self.text_out_event[curr_id]:
                dur = end-start
                if dur<0.1:
                    continue
                frame_mat = np.zeros((len(speak_order_map[curr_id]), round(dur/self.frame_res)))
                
                if output_format == "text":
                    all_events = re.findall((r'(<spk\d>.\(\d+\.\d+, \d+\.\d+\))'), self.text_out_event[curr_id][(start,end)])
                    for curr_event in all_events:
                        parts=curr_event.split()
                        spk_id = int(parts[0][4])
                        curr_start = float(parts[1].strip("(").strip(","))
                        curr_end = float(parts[2].strip(")"))

                        curr_start_idx = round(curr_start/self.frame_res)
                        curr_end_idx = round(curr_end/self.frame_res)
                        frame_mat[spk_id-1][curr_start_idx:curr_end_idx]=1
                    
                    frame_idx=frame_mat[0,:]
                    for i in range(1,len(speak_order_map[curr_id])):
                        frame_idx +=(2**i)*(frame_mat[i,:])
                    frame_idx = np.array(frame_idx, dtype=int)
                    out_string=" ".join(map(str, frame_idx.ravel()))+" "

                    if self.use_extra_info: # add special tokens for transitions
                        out_string=""
                        prev_spk = frame_idx[0]
                        if prev_spk>0:
                            out_string+=f"<speech> {frame_idx[0]} "
                        for i in range(1,len(frame_idx)):
                            if prev_spk!=frame_idx[i]: # when speaker changes                            
                                if prev_spk==0 or frame_idx[i]==0: # change back to silence
                                    out_string+="<speech> "
                                if prev_spk>0 and frame_idx[i]>0:
                                    out_string+="<sc> "
                                if frame_idx[i]>len(frame_mat)+1: # frame_idx[i]>len(frame_mat)+1:
                                    out_string+="<overlap> "

                            out_string+=str(frame_idx[i])+" "
                            prev_spk=frame_idx[i]
                else: # diar tokens
                    all_events = re.findall((r'(<spk\d> <bot> <\d+\.\d+> <\d+\.\d+> <eot>)'), self.text_out_event[curr_id][(start,end)])
                    for curr_event in all_events:
                        parts = curr_event.split()
                        spk_id = int(parts[0][4])
                        curr_start = float(parts[2].strip("<").strip(">"))
                        curr_end = float(parts[3].strip("<").strip(">"))

                        curr_start_idx = round(curr_start/self.frame_res)
                        curr_end_idx = round(curr_end/self.frame_res)
                        frame_mat[spk_id-1][curr_start_idx:curr_end_idx]=1
                    
                    frame_idx=frame_mat[0,:]
                    out_string=""
                    for i in range(frame_mat.shape[1]):
                        curr_active_spks=[]
                        for j in range(len(speak_order_map[curr_id])):
                            if frame_mat[j][i]==1:
                                curr_active_spks.append(j+1)
                        if len(curr_active_spks)==0: # silence
                            out_string+="<sil> "
                        elif len(curr_active_spks)==1:
                            out_string+=f"<spk{curr_active_spks[0]}> "
                        elif len(curr_active_spks)<=3:
                            spk_list = "_".join(str(s) for s in curr_active_spks)
                            out_string += f"<overlap_spk_{spk_list}> "
                        else:
                            out_string+="<overlap> "

                text_out_frame[curr_id][(start,end)]=out_string
    
        time_info = "_".join(segment_file.split("_")[-2:])
        if self.use_random_durs:
            time_info = "_".join(segment_file.split("_")[-5:])
        out_file_frame_name=f"{output_format}_{self.pit_method}_frame_{time_info}"


        if self.spk_format!="spk_idx":
            out_file_frame_name+=f"_{self.spk_format}"
        if self.use_extra_info:
            out_file_frame_name+=f"_extra_info"

        out_text_frame_path=os.path.join(self.output_dir, curr_set, out_file_frame_name)
        text_frame_out = self.get_text_out(text_out_frame, format_type="frame", output_format=output_format)
        self.write_file(text_frame_out, out_text_frame_path)
        print(f"write frame file to {out_text_frame_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare diarization wav.scp and segments files")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to AMI dataset root")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for Kaldi files")
    parser.add_argument("--wav_out_dir", type=str, required=True, help="Directory to save single-channel wav files")
    parser.add_argument("--mic", type=str, default="sdm", choices=["sdm", "ihm"], help="Microphone type")
    parser.add_argument("--dur", type=int, default=3, help="Duration of each segment in seconds")
    parser.add_argument("--skip", type=int, default=1, help="Skip interval in seconds")
    parser.add_argument("--pit_method", type=str, default="arrive", choices=["arrive", "most"], help="speaker arrival order or most frequently speaker")
    parser.add_argument("--spk_format", type=str, default="spk_idx", choices=["spk_idx", "spk_id"])
    parser.add_argument("--curr_sets", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--use_random_durs", action="store_true", help="whether to use random durations")
    parser.add_argument("--use_extra_info", action="store_true")
    parser.add_argument("--frame_res", type=float, default=0.1, help="frame resolution")
    parser.add_argument("--data_outputs", nargs="+", default=["wav", "text_event", "text_frame"])

    args = parser.parse_args()

    processor = DiarizationPreprocessor(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        wav_out_dir=args.wav_out_dir,
        mic=args.mic,
        dur=args.dur,
        skip=args.skip,
        pit_method=args.pit_method,
        spk_format=args.spk_format,
        curr_sets=args.curr_sets,
        use_extra_info=args.use_extra_info,
        use_random_durs=args.use_random_durs,
        frame_res=args.frame_res,
    )

    # prepare wav file
    for item in args.data_outputs:
        for curr_set in args.curr_sets:
            print(f"current set {curr_set}")
            if item == "wav":
                if args.use_random_durs:
                    print("preparing wav file")
                    processor.prep_random_wav_scp(curr_set)                
                else:
                    processor.prep_wav_scp(curr_set)

            if item == "text":
                print("preparing event file")
                processor.prep_event(curr_set, "text")
                print("preparing frame file")
                processor.prep_frame(curr_set, "text")
            if item == "diar_tokens":
                print("preparing event file")
                processor.prep_event(curr_set, "diar_tokens")
                print("preparing frame file")
                processor.prep_frame(curr_set, "diar_tokens")



