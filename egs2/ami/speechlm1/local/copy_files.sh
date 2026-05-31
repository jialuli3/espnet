for curr_set in dev test train;
do
  orig_file="../data/${curr_set}/diar_tokens_arrive_frame_dur8_skip6"
  target_file="../dump/audio_raw_codec_ssl_sd_frame_dur8_skip6_diar_model_ami/${curr_set}/"
  echo $orig_file
  echo $target_file
  cp $orig_file $target_file
  target_file="../dump/raw_codec_ssl_sd_frame_dur8_skip6_diar_model_ami/${curr_set}/"
  echo $target_file
  cp $orig_file $target_file
done

orig_file="../data/test_skip8/diar_tokens_arrive_frame_dur8_skip8"
target_file="../dump/raw_codec_ssl_sd_frame_dur8_skip6_diar_model_ami/test_skip8/diar_tokens_arrive_frame_dur8_skip6"
cp $orig_file $target_file

target_file="../dump/audio_raw_codec_ssl_sd_frame_dur8_skip6_diar_model_ami/test_skip8/diar_tokens_arrive_frame_dur8_skip6"
cp $orig_file $target_file

