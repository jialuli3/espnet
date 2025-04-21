import os
from scipy.io.wavfile import read,write
from collections import Counter
import numpy as np
import re
import pdb

def read_file(file_name):
    f=open(file_name,"r")
    content=f.readlines()
    f.close()
    return content

def write_file(content, file_name):
    f=open(file_name,"w")
    f.writelines(content)
    f.close()

def prep_wav_scp(file_id, output_dir, curr_set, dataset_dir, wav_out_dir, mic="sdm", dur=3, skip=1):
    if not os.path.exists(os.path.join(output_dir,curr_set)):
        os.makedirs(os.path.join(output_dir,curr_set), exist_ok=True)
    if not os.path.exists(wav_out_dir): # used to store one channel audio
        os.mkdir(wav_out_dir)

    output_path=os.path.join(output_dir,curr_set,"wav.scp")
    output_segment_path=os.path.join(output_dir,curr_set,f"segments_dur{dur}_skip{skip}")

    all_ids = read_file(file_id)
    out = []
    out_segments=[]

    for curr_id in all_ids:
        curr_id = curr_id.strip()
        if curr_id in ["IS1003b", "IS1007d", "IB4005"]: # follow Brno setup
            continue
        if mic=="sdm":
            curr_wav_file=os.path.join(dataset_dir, curr_id, "audio", f"{curr_id}.Array1-01.wav")
        else: # ihm
            curr_wav_file=os.path.join(dataset_dir, curr_id, "audio", f"{curr_id}.Mix-Headset.wav")

        out.append(f"{curr_id} {curr_wav_file}\n")
        rate,data = read(curr_wav_file)
        print(curr_wav_file, data.shape)
        if len(data.shape)>1:
            data=data[:,0]
            write(os.path.join(wav_out_dir,os.path.basename(curr_wav_file)), rate, data)
            out[-1]=f"{curr_id} {os.path.join(wav_out_dir,os.path.basename(curr_wav_file))}\n"
        total_dur = int(len(data)/rate)+1

        for time_idx in range(0,total_dur,skip):
            start = round(time_idx,2)
            end = round(min(time_idx + dur, len(data)/rate),2)
            if end-start<1: continue
            out_segments.append(f"{curr_id}-{start}-{end} {curr_id} {start} {end}\n")

        if end<len(data)/rate:
            start = round(end,2)
            end = round(len(data)/rate,2)
            out_segments.append(f"{curr_id}-{start}-{end} {curr_id} {start} {end}\n")

    if not os.path.exists(output_path):
       write_file(out, output_path)
    write_file(out_segments, output_segment_path)

def get_spk_order(rttm_file, method="arrive"):
    '''
    get speaker order based on rttm file
    '''
    #method can be arrive_time_order or most_time_order
    rttm_rows = read_file(rttm_file)
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

def get_overlap_interval(s1,e1,s2,e2):
    if s1<=e2 and s2<=e1:
        return (max(s1,s2),min(e1,e2))
    return None

def get_text_out(text_out_dict, text_count_dict=None, format_type="event"):
    out=[]
    for curr_id in text_out_dict:
        for start,end in text_out_dict[curr_id]: 
            if text_out_dict[curr_id][(start,end)]=="":
                if format_type=="event":
                    text_out_dict[curr_id][(start,end)]="<space>"
                else:
                    continue
            if text_count_dict is not None:
                sorted_c=dict(sorted(text_count_dict[curr_id][(start,end)].items()))
                counter_out = str(dict(sorted_c)).replace("'","")
                out.append(f"{curr_id}-{int(start)}-{int(end)} {counter_out} {text_out_dict[curr_id][(start,end)]}\n")
            else:           
                out.append(f"{curr_id}-{int(start)}-{int(end)} {text_out_dict[curr_id][(start,end)]}\n")
    return out

def get_text_out_entire_record(text_out_dict):
    out=[]
    for curr_id in text_out_dict:
        out.append(f"{curr_id} {text_out_dict[curr_id]}\n")
    return out

def prep_text(rttm_file, segment_file, output_dir, curr_set, dataset_dir, \
                pit_method="arrive", spk_format="spk_idx", frame_res=0.1, use_special_tokens=False):
    '''
    pit_method: arrive_time_order, most_time_order
    output_format: event, frame
    spk_format: spk_idx, spk_id, for event-based model only
    '''

    rttm_rows = read_file(rttm_file)
    segment_rows = read_file(segment_file)
    speak_order_map=get_spk_order(rttm_file, pit_method)
    text_out_event = {}
    text_out_event_count = {}

    text_out_frame = {}

    for segment_row in segment_rows:
        segment_row = segment_row.strip()
        curr_id = segment_row.split()[1]
        start,end = float(segment_row.split()[2]), float(segment_row.split()[3])
        if curr_id not in text_out_event:
            text_out_event[curr_id]={} # initialize the output dictionary with start,end stamps
            text_out_event_count[curr_id]={}
            text_out_frame[curr_id]={}
        text_out_event[curr_id][(start,end)]=""
        text_out_event_count[curr_id][(start,end)]=Counter({f"<spk{key}>": 0 for key in range(1, len(speak_order_map[curr_id])+1)})
        text_out_frame[curr_id][(start,end)]=""
    
    for rttm_row in rttm_rows:
        curr_rttm = rttm_row.strip()
        curr_id = curr_rttm.split()[1]
        rttm_start = float(curr_rttm.split()[-7])
        dur = float(curr_rttm.split()[-6])
        rttm_end = rttm_start+dur
        curr_spk = curr_rttm.split()[-3]
        spk_id = speak_order_map[curr_id][curr_spk]
        
        # fill out the text transcript
        for start,end in text_out_event[curr_id]:
            overlap_interval = get_overlap_interval(start, end, rttm_start, rttm_end)
            if overlap_interval is not None:
                offset_start, offset_end = round(overlap_interval[0]-start,1), round(overlap_interval[1]-start,1)
                if offset_end-offset_start<0.1: continue # at least 20ms
                text_out_event_count[curr_id][(start,end)][f"<spk{spk_id}>"]+=1
                if spk_format=="spk_idx":
                    text_out_event[curr_id][(start,end)]+="{"+f"<spk{spk_id}> ({offset_start}, {offset_end})"+"} "
                else:
                    text_out_event[curr_id][(start,end)]+="{"+f"<{curr_spk}> ({offset_start}, {offset_end})"+"} "

    # compute frame-based text output based on event-based text output
    for curr_id in text_out_event:
        for start,end in text_out_event[curr_id]:
            dur = end-start
            if dur<0.1:
                continue
            frame_mat = np.zeros((len(speak_order_map[curr_id]), round(dur/frame_res)))
            
            all_events = re.findall((r'(<spk\d>.\(\d+\.\d+, \d+\.\d+\))'), text_out_event[curr_id][(start,end)])
            for curr_event in all_events:
                spk_id = int(curr_event.split()[0][4])
                curr_start = float(curr_event.split()[1].strip("(").strip(","))
                curr_end = float(curr_event.split()[2].strip(")"))

                curr_start_idx = round(curr_start/frame_res)
                curr_end_idx = round(curr_end/frame_res)
                # print(curr_start, curr_end, curr_start_idx, curr_end_idx)
                frame_mat[spk_id-1][curr_start_idx:curr_end_idx]=1
            
            frame_idx=frame_mat[0,:]
            for i in range(1,len(speak_order_map[curr_id])):
                frame_idx +=(2**i)*(frame_mat[i,:])
            frame_idx = np.array(frame_idx, dtype=int)
            out_string=" ".join(map(str, frame_idx.ravel()))+" "
            if use_special_tokens:
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
            text_out_frame[curr_id][(start,end)]=out_string
    
    time_info = "_".join(segment_file.split("_")[-2:])
    out_file_event_name=f"text_{pit_method}_event_{time_info}"
    out_file_frame_name=f"text_{pit_method}_frame_{time_info}"

    if spk_format!="spk_idx":
        out_file_event_name+=f"_{spk_format}"
        out_file_frame_name+=f"_{spk_format}"
    if use_special_tokens:
        out_file_event_name+=f"_special_tokens"
        out_file_frame_name+=f"_special_tokens"

    out_text_event_path=os.path.join(output_dir, curr_set, out_file_event_name)
    out_text_frame_path=os.path.join(output_dir, curr_set, out_file_frame_name)

    if use_special_tokens:
        text_event_out = get_text_out(text_out_event, text_count_dict=text_out_event_count, format_type="event")
    else:
        text_event_out = get_text_out(text_out_event, format_type="event")
    text_frame_out = get_text_out(text_out_frame, format_type="frame")

    write_file(text_event_out, out_text_event_path)
    write_file(text_frame_out, out_text_frame_path)


def prep_text_entire_record(rttm_file, output_dir, curr_set, dataset_dir, \
                pit_method="arrive"):
    '''
    pit_method: arrive_time_order, most_time_order
    output_format: event, frame
    '''

    rttm_rows = read_file(rttm_file)
    speak_order_map=get_spk_order(rttm_file, pit_method)
    text_out_event = {}
    text_out_frame = {}
    
    for rttm_row in rttm_rows:
        curr_rttm = rttm_row.strip()
        curr_id = curr_rttm.split()[1]
        rttm_start = round(float(curr_rttm.split()[-7]), 1)
        dur = round(float(curr_rttm.split()[-6]), 1)
        rttm_end = round(rttm_start+dur, 1)
        curr_spk = curr_rttm.split()[-3]
        spk_id = speak_order_map[curr_id][curr_spk]
        
        if curr_id not in text_out_event:
            text_out_event[curr_id]=""
        text_out_event[curr_id]+="{"+f"<spk{spk_id}> ({rttm_start}, {rttm_end})"+"} "

    # compute frame-based text output based on event-based text output
    #out_text_event_path=os.path.join(output_dir, curr_set, f"text_{pit_method}_event")
    out_text_frame_path=os.path.join(output_dir, curr_set, f"text_{pit_method}_frame")

    #text_event_out = get_text_out_entire_record(text_out_event)
    text_frame_out = get_text_out_entire_record(text_out_frame)

    #write_file(text_event_out, out_text_event_path)
    write_file(text_frame_out, out_text_frame_path)

np.set_printoptions(linewidth=np.inf)
# prepare wav files    
dataset_dir="/ocean/projects/cis210027p/shared/corpora/amicorpus"
pit_method="arrive"
spk_format="spk_idx"
use_special_tokens=False
mic="ihm"
dur=8
skip=8
#for curr_set in ["dev", "test", "train"]:
# for curr_set in ["test"]:
#     # prep_wav_scp(f"split_{curr_set}.orig", "../data", curr_set, dataset_dir, "../data/wav", dur=dur, skip=skip)
#     prep_text(f"../data/{curr_set}/rttm", f"../data/{curr_set}/segments_dur{dur}_skip{skip}", "../data", curr_set,\
#                 dataset_dir, pit_method=pit_method, spk_format=spk_format, use_special_tokens=use_special_tokens)
#     #prep_text_entire_record(f"../data/{curr_set}/rttm", "../data", curr_set, dataset_dir)

#for curr_set in ["dev_ihm", "train_ihm"]:
for curr_set in ["test_ihm"]:
    prep_wav_scp(f"split_{curr_set}.orig", "../data", curr_set, dataset_dir, "../data/wav", mic, dur=dur, skip=skip)
    prep_text(f"../data/{curr_set}/rttm", f"../data/{curr_set}/segments_dur{dur}_skip{skip}", "../data", curr_set,\
               dataset_dir, pit_method=pit_method, spk_format=spk_format, use_special_tokens=use_special_tokens)
    #prep_text_entire_record(f"../data/{curr_set}/rttm", "../data", curr_set, dataset_dir)

