# run.sh
train_set="train"
valid_set="dev"
test_set="test_skip8"
train_config="conf/train_delay_epoch50_lr_set2.yaml"
inference_config="conf/decode_sd.yaml"
task="codec_ssl_sd_event_dur8_skip6"

stage=2              # Processes starts from the specified stage.
stop_stage=2     # Processes is stopped at the specified stage.

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ./speechlm.sh \
    --task ${task} \
    --data_name "ami" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}"  \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --audio_format wav \
    --dur 8 --skip 8 \
    --codec_stage 1 \
    --codec_choice ESPnet --codec_hf_model_tag ftshijt/espnet_codec_dac_large_v1.4_360epoch \
    --ssl_choice espnet_hubert --ssl_nlayer 18 --ssl_checkpoint_path exp/kmeans/38epoch.pth --ssl_kmeans_path exp/kmeans/xeus_18_5000clusters/km_5000.mdl --ssl_batch_bins 2000000 \
    --subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B \
    --ngpu 2 \
    --cmd_backend "slurm_train" \
    --nj 1 --inference_nj 1 \
    --stage 8 --stop_stage 8
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ./speechlm.sh \
    --task ${task} \
    --data_name "ami" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}"  \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --audio_format wav \
    --dur 8 --skip 8 \
    --ngpu 1 \
    --gpu_inference true \
    --inference_model valid.ce_loss.ave_3best.pth \
    --nj 1 --inference_nj 1 \
    --apply_clustering true \
    --stage 9 --stop_stage 10
fi