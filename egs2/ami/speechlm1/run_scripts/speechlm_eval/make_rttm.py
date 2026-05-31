import os
from scipy.io.wavfile import read,write
from scipy.signal import medfilt
from collections import Counter
import numpy as np
import re
import os
from praatio import textgrid
import argparse

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

    arr_time_spk_order_reverse={}
    most_time_spk_order_reverse={}
    
    for curr_id in arr_time_spk_order:
        arr_time_spk_order_reverse[curr_id]={}
        most_time_spk_order_reverse[curr_id]={}
        for curr_spk in arr_time_spk_order[curr_id]:
            spk_idx = arr_time_spk_order[curr_id][curr_spk]
            arr_time_spk_order_reverse[curr_id][spk_idx]=curr_spk
            spk_idx = most_time_spk_order[curr_id][curr_spk]
            most_time_spk_order_reverse[curr_id][spk_idx]=curr_spk

    if method=="arrive":
        return arr_time_spk_order_reverse
    return most_time_spk_order_reverse

def merge_out_dict(out_dict):
    merge_dict={}
    for file_id in out_dict:
        merge_dict[file_id]={}
        for spk_id in out_dict[file_id]:
            curr_interval = sorted(out_dict[file_id][spk_id], key=lambda x: x[0])
            #if file_id == "EN2002a":
            #    print("before merged", file_id, spk_id, curr_interval)
            merge_dict[file_id][spk_id]=[curr_interval[0]]
            for start,end in curr_interval[1:]:
                if start<=merge_dict[file_id][spk_id][-1][1]:
                    merge_dict[file_id][spk_id][-1][1]=max(merge_dict[file_id][spk_id][-1][1], end)
                else:
                    merge_dict[file_id][spk_id].append([start,end])
            #if file_id == "EN2002a":
            #    print("merged", file_id, spk_id, merge_dict[file_id][spk_id])
    return merge_dict

def apply_median_filter_func(merge_dict, frame_duration=0.1, kernel_size = 11):
    """
    Convert segments into 0.1s framewise labels.

    """

    merge_dict_median={}
    # generate frame-based level features from event-based format
    for file_id in merge_dict:
        max_time = 0
        for spk_id in merge_dict[file_id]:
            max_time=max(max_time, merge_dict[file_id][spk_id][-1][-1])
        
        num_frames = int(np.ceil(max_time / frame_duration))
        num_spks = len(merge_dict[file_id])
        all_speakers_labels = np.zeros((num_spks,num_frames), dtype=int)

        for spk_id in merge_dict[file_id]:
            for start, end in merge_dict[file_id][spk_id]:
                start_idx = int(np.floor(start / frame_duration))
                end_idx = int(np.ceil(end / frame_duration))
                all_speakers_labels[spk_id-1][start_idx:end_idx]=1

        frame_idx=np.array(all_speakers_labels[0, :], dtype=int)
        for i in range(1,num_spks):
            frame_idx +=np.power(2, i)*(all_speakers_labels[i,:])
        frame_idx = np.array(frame_idx, dtype=int)
        
        # apply median filter
        frame_idx = medfilt(frame_idx, kernel_size=kernel_size)

        # convert back to event-based format
        all_speakers_labels = np.zeros((num_spks,num_frames))

        for i in range(num_spks):
            all_speakers_labels[i] = (frame_idx >> i) & 1

        merge_dict_median[file_id]={}

        for spk in range(num_spks):
            active = all_speakers_labels[spk]
            merge_dict_median[file_id][spk+1]=[]
            start = None

            for i in range(num_frames):
                if active[i] == 1 and start is None:
                    start = i
                elif active[i] == 0 and start is not None:
                    end = i
                    merge_dict_median[file_id][spk+1].append([round(start * frame_duration, 1), round(end * frame_duration, 1)])
                    start = None

            # Handle case where last frame is active
            if start is not None:
                end = num_frames
                merge_dict_median[file_id][spk+1].append([round(start * frame_duration, 1), round(end * frame_duration, 1)])

    return merge_dict_median

def get_segmentations(merge_dict, wav_folder_path, spk_count_dur=0.02, spk_count_skip=0.02,\
    segment_dur=8, segment_skip=0.8): # follow Brno people's choice
    wav_files = read_file(wav_folder_path)
    wav_dict={}
    for row in wav_files:
        row=row.strip().split()
        wav_dict[row[0]]=row[1]

    out_segmentations={}
    #generate pyannote segments, default 8s with skip 0.8s
    #receptive field duration 0.025s with 0.02s skip
    for file_id in merge_dict:
        num_spk = len(merge_dict[file_id])
        print("file_id", file_id)
        rate, audio_data = read(wav_dict[file_id])
        max_dur = len(audio_data)/rate
        print("max dur", max_dur)

        total_frames=int((max_dur-spk_count_dur)/spk_count_skip)
        spk_activities = np.zeros((total_frames, num_spk))
        print("total_frames", total_frames)

        all_spks = sorted(list(merge_dict[file_id].keys()))
        for spk_idx in all_spks:
            for start,end in merge_dict[file_id][spk_idx]:
                start_idx = int(start/spk_count_skip)
                end_idx = int(end/spk_count_skip)
                spk_activities[start_idx:min(end_idx, total_frames), spk_idx-1]=1
        
        total_segments_chunk = int((max_dur-segment_dur)/segment_skip)+1
        print("total_segments_chunk", total_segments_chunk)
        segment_frame=int((segment_dur-spk_count_dur)/spk_count_skip)+1 # 400
        print("segment_frame", segment_frame)
        curr_segment_spk_activities = np.zeros((total_segments_chunk, segment_frame, num_spk))

        for i in range(total_segments_chunk):
            start_idx = int(i*segment_skip/spk_count_skip)
            end_idx = min(start_idx+segment_frame, len(spk_activities))
            if start_idx>=end_idx: 
                print(i, start_idx, end_idx)
                continue
            curr_segment_spk_activities[i, :end_idx-start_idx, :]=spk_activities[start_idx:end_idx, :]
        out_segmentations[file_id]=curr_segment_spk_activities
        
    return out_segmentations

def apply_pyannote_clustering(segmentations, wav_folder_path, spk_count_dur=0.02, spk_count_skip=0.02, segment_dur=8, segment_skip=0.8):
    from diarizen.pipelines.inference import DiariZenPipeline
    from pyannote.core import SlidingWindowFeature, SlidingWindow
    
    wav_files = read_file(wav_folder_path)
    diar_pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-meeting-base")
    diar_pipeline.max_n_speakers = 4
    receptive_field=SlidingWindow(start=0.0, step=spk_count_skip, duration=spk_count_dur)
    frame = SlidingWindow(start=0.0, step=segment_skip, duration=segment_dur)

    out_rttm=""
    wav_dict={}
    for row in wav_files:
        row=row.strip().split()
        wav_dict[row[0]]=row[1]

    for file_id in segmentations:
        wav_file = wav_dict[file_id]
        curr_spk_activities = segmentations[file_id]
        segmentation_input = SlidingWindowFeature(curr_spk_activities, frame)
        diar_results = diar_pipeline(wav_file, segmentation_input, receptive_field, file_id)
        out_rttm+=diar_results.to_rttm()
    return out_rttm

def make_hyp_rttm(hyp_output_file, hyp_format, hyp_output_dir, spk_map, test_wav_scp, spk_format="spk_id", apply_clustering=False, skip_interval=None, apply_median_filter=True):
    content=read_file(hyp_output_file)
    out_dict={}
    print("start making hyp rttm")
    for row in content:
        try:    
            key=row.strip().split()[0]
            file_info = key.split("_")[-2]
            file_id, start_offset = file_info.split("-")[0], float(file_info.split("-")[1])
            end_offset = float(file_info.split("-")[2])
            if skip_interval is not None:
                if end_offset % skip_interval !=0: continue
        except:
            continue
        if hyp_format=="event":
            if spk_format=="spk_idx":
                all_events = re.findall((r'(<spk\d>.\(\d+\.\d+, \d+\.\d+\))'), row.strip())
                for curr_event in all_events:
                    try:
                        spk_idx = int(curr_event.split()[0][4])
                        if spk_idx in spk_map[file_id]:
                            #spk_id = spk_map[file_id][spk_idx]
                            spk_id = spk_idx
                            curr_start = float(curr_event.split()[1].strip("(").strip(","))
                            curr_end = float(curr_event.split()[2].strip(")"))
                            start = round(curr_start+start_offset, 2)
                            end = round(curr_end+start_offset, 2)
                            if end-start<0.1: continue
                            if file_id not in out_dict:
                                out_dict[file_id]={}
                            if spk_id not in out_dict[file_id]:
                                out_dict[file_id][spk_id]=[]
                            out_dict[file_id][spk_id].append([start, end])
                    except:
                        continue
            else: # spk_format == "spk_id"
                all_events = re.findall((r'(<\w+\d+\w+>.\(\d+\.\d+, \d+\.\d+\))'), row.strip())
                for curr_event in all_events:
                    try:
                        spk_id = curr_event.split()[0].strip("<").strip(">")
                        curr_start = float(curr_event.split()[1].strip("(").strip(","))
                        curr_end = float(curr_event.split()[2].strip(")"))
                        start = round(curr_start+start_offset, 2)
                        end = round(curr_end+start_offset, 2)
                        if end-start<0.1: continue
                        if file_id not in out_dict:
                            out_dict[file_id]={}
                        if spk_id not in out_dict[file_id]:
                            out_dict[file_id][spk_id]=[]
                        out_dict[file_id][spk_id].append([start, end])
                    except:
                        continue            

    ### merge dictionary
    print("merge dict")
    merge_dict=merge_out_dict(out_dict)
    print("apply median filter")
    if apply_median_filter:
        merge_dict=apply_median_filter_func(merge_dict)

    # get segmentations for pyannote clustering
    if apply_clustering:
        segmentations = get_segmentations(merge_dict, test_wav_scp)
        out_rttm = apply_pyannote_clustering(segmentations, test_wav_scp)
        write_file(out_rttm, os.path.join(hyp_output_dir, "hyp.rttm"))   
        return 
    
    content=[]
    for file_id in merge_dict:
        for spk_id in merge_dict[file_id]:
            for start,end in merge_dict[file_id][spk_id]:
                dur = round(end-start,2)
                content.append(f"SPEAKER {file_id} 1 {start} {dur} <NA> <NA> {spk_id} <NA> <NA>\n")
    write_file(content, os.path.join(hyp_output_dir, "hyp.rttm"))    

def rttm_to_interval(rttm_file, file_type="hyp"):
    content=read_file(rttm_file)
    out_dict={}
    for row in content:
        file_id = row.split(" ")[1]
        spk_id = row.split(" ")[-3]
        start,dur = round(float(row.split(" ")[3]),2), round(float(row.split(" ")[4]),2)
        if file_id not in out_dict:
            out_dict[file_id]={}
        if spk_id not in out_dict[file_id]:
            out_dict[file_id][spk_id]=[]
        out_dict[file_id][spk_id].append((start, round(start+dur,2), file_type))
    return out_dict

def make_textgrid(hyp_rttm_file, ref_rttm_file, textgrid_dir):
    hyp_dict = rttm_to_interval(hyp_rttm_file, "hyp")
    ref_dict = rttm_to_interval(ref_rttm_file, "ref")

    for file_id in hyp_dict:
        tg = textgrid.Textgrid()
        for spk_id in hyp_dict[file_id]:
            curr_spk_tier = textgrid.IntervalTier(f'{spk_id} hyp', hyp_dict[file_id][spk_id], 0, hyp_dict[file_id][spk_id][-1][1])
            tg.addTier(curr_spk_tier)
        for spk_id in ref_dict[file_id]:
            curr_spk_tier = textgrid.IntervalTier(f'{spk_id} ref', ref_dict[file_id][spk_id], 0, ref_dict[file_id][spk_id][-1][1])
            tg.addTier(curr_spk_tier)
        out_tg_path = os.path.join(textgrid_dir, f"{file_id}.TextGrid")
        tg.save(out_tg_path, format="long_textgrid", includeBlankSpaces=True)

def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(description="diarization scoring pipeline.")
    
    parser.add_argument(
        "--ref_rttm_file", 
        type=str, 
        help="reference rttm file"
    )

    parser.add_argument(
        "--uem_file", 
        type=str, 
        help="reference uem file"
    )

    parser.add_argument(
        "--hyp_output_file", 
        type=str, 
        help="hypothesis output file"
    )

    parser.add_argument(
        "--hyp_format", 
        type=str,
        choices=["event","frame"], 
        help="hyp format, can be either event for frame"
    )

    parser.add_argument(
        "--scoring_dir", 
        type=str,
        help="output file path"
    )

    parser.add_argument(
        "--textgrid_dir", 
        type=str,
        default=None,
        help="output textgrid path for visualization"
    )

    parser.add_argument(
        "--test_wav_scp", 
        type=str,
        default=None,
        help="path to entire wavfile scp "
    )

    parser.add_argument(
        "--speaker_order_method", 
        type=str,
        choices=["arrive","most"], 
        help="can be either arrive time order or most time order"
    )

    parser.add_argument(
        "--spk_format", 
        type=str,
        choices=["spk_id","spk_idx"], 
        default="spk_idx",
        help="can be speaker index or actual speaker ID"
    )

    parser.add_argument(
        "--skip_interval", 
        type=int,
        default=1,
        help="whether to use overlap intervals for inference"
    )

    parser.add_argument(
        "--apply_clustering", 
        action="store_true",  # This will set it to True if the flag is provided
        help="whether to apply speaker clustering from pyannote framework"
    )

    args = parser.parse_args()
    print(args)

    #get speaker order first
    spk_map=get_spk_order(args.ref_rttm_file,args.speaker_order_method)
    make_hyp_rttm(args.hyp_output_file, args.hyp_format, args.scoring_dir, spk_map, args.test_wav_scp, args.spk_format, args.apply_clustering, args.skip_interval)
    if args.textgrid_dir is not None:
        make_textgrid(os.path.join(args.scoring_dir, "hyp.rttm"), os.path.join(args.scoring_dir, "ref.rttm"), args.textgrid_dir)

if __name__ == "__main__":
    main()
