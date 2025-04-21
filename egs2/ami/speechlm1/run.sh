# run.sh
train_set="train"
valid_set="dev"
test_set="test"

./speechlm.sh \
  --task codec_ssl_sd_event \
  --data_name "ami" \
  --train_set "${train_set}" \
  --valid_set "${valid_set}" \
  --test_sets "${test_set}"  \
  --audio_format wav \
  --codec_choice ESPnet --codec_hf_model_tag ftshijt/espnet_codec_dac_large_v1.4_360epoch \
  --ssl_choice espnet_hubert --ssl_nlayer 18 --ssl_checkpoint_path exp/kmeans/38epoch.pth --ssl_kmeans_path exp/kmeans/xeus_18_5000clusters/km_5000.mdl --ssl_batch_bins 5000000 \
  --subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B \
  --ngpu 1 \
  --nj 32 --inference_nj 32 \
  --stage 8 --stop_stage 8
