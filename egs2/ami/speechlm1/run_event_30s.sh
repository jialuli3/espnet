# run.sh
train_set="train"
valid_set="dev"
test_set="test"
train_config="conf/train_delay.yaml"
inference_config="conf/decode_sd.yaml"
task="codec_ssl_sd_event_dur30_skip10"

# need to load cuda
./speechlm.sh \
  --task ${task} \
  --data_name "ami" \
  --train_set "${train_set}" \
  --valid_set "${valid_set}" \
  --test_sets "${test_set}"  \
  --train_config "${train_config}" \
  --inference_config "${inference_config}" \
  --audio_format wav \
  --dur 30 --skip 10 \
  --codec_choice ESPnet --codec_hf_model_tag ftshijt/espnet_codec_dac_large_v1.4_360epoch \
  --ssl_choice espnet_hubert --ssl_nlayer 18 --ssl_checkpoint_path exp/kmeans/38epoch.pth --ssl_kmeans_path exp/kmeans/xeus_18_5000clusters/km_5000.mdl --ssl_batch_bins 1000000 \
  --subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B \
  --ngpu 1 \
  --gpu_inference true \
  --inference_model valid.ce_loss.ave_3best.pth \
  --nj 1 --inference_nj 1 \
  --skip_interval 30 \
  --apply_clustering false \
  --stage 10 --stop_stage 10
