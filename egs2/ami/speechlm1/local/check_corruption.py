import kaldiio

#spec = "../dump_synthetic/raw_codec_ssl_sd_event_sad_od_dur30_skip10_alimeeting/swb_sre_tr_ns5_beta25_5000/data/wav_codec_ssl_ESPnet.4.ark:311526243"

spec = "../5_spks/AMI/raw_codec_ssl_sd_event_sad_od_dur30_skip10_ami/train/data/wav_codec_ESPnet.1.ark:420161"

x = kaldiio.load_mat(spec)  # or load_ark, depending on type

print(type(x), getattr(x, "shape", None))
