input_dir=/work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/dump_synthetic/raw_codec_ssl_sd_event_sad_od_dur30_skip10_alimeeting/test
curr_part=swb_sre_cv_ns5_beta25_500
output_dir=/work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/dump_synthetic/raw_codec_ssl_sd_event_sad_od_dur30_skip10_alimeeting

awk "\$1 ~ /^data_simu_wav_${curr_part}/" \
    $input_dir/wav.scp > $output_dir/$curr_part/wav.scp
awk 'NR==FNR {keep[$1]=1; next} ($1 in keep)' $output_dir/$curr_part/wav.scp  $input_dir/utt2num_samples > $output_dir/$curr_part/utt2num_samples
awk 'NR==FNR {keep[$1]=1; next} ($1 in keep)' $output_dir/$curr_part/wav.scp  $input_dir/diar_tokens_event_sad_od_dur30_skip10 > $output_dir/$curr_part/diar_tokens_event_sad_od_dur30_skip10

python make_json.py --input $output_dir/$curr_part/wav.scp --output $output_dir/$curr_part/data.json