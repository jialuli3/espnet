from pathlib import Path
import json

# 1) Find the json this script is splitting (adjust path)
js = json.load(open("../dump_synthetic/raw_codec_ssl_sd_event_sad_od_dur30_skip10_alimeeting/dev/data.json"))

print(len(js))                         # how many utts
print(list(js["examples"])[:5])             # sample keys

# 2) Create a fake utt that looks like the broken one
utt = "data_simu_wav_swb_sr" + "\x00"*10
print(utt in js["examples"])                       # this should be False
print("data_simu_wav_swb_sr" in js)    # maybe True
