import kaldiio

# scp_path = 'dump/raw_codec_ssl_sd_event_dur3_skip1_ami/dev/wav.scp'
# for key, array in kaldiio.load_scp_sequential(scp_path):
#     print(f"{key}: shape = {array.shape}")
#     break

features = kaldiio.load_mat('dump/raw_codec_ssl_sd_event_dur30_skip10_ami/dev/data/wav_codec_ssl_ESPnet.1.ark:1699542')
print(features, features.shape)