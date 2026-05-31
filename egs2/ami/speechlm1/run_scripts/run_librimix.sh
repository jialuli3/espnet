#!/bin/bash

train_set="train"
valid_set="dev"
test_set="test"
train_config="conf/train_delay_delta_warmup4000_lre-5_target_only.yaml"
inference_config="conf/decode_sd.yaml"
task="codec_ssl_sd_event_librimix_diar_model"

stage=1              # Processes starts from the specified stage.
stop_stage=1     # Processes is stopped at the specified stage.

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ./speechlm.sh \
    --task ${task} \
    --data_name "librimix" \
    --data_dir "data_librimix" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}"  \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --audio_format wav \
    --codec_choice ESPnet --codec_hf_model_tag ftshijt/espnet_codec_dac_large_v1.4_360epoch \
    --ssl_choice espnet_hubert --ssl_nlayer 18 --ssl_checkpoint_path exp/kmeans/38epoch.pth --ssl_kmeans_path exp/kmeans/xeus_18_5000clusters/km_5000.mdl --ssl_batch_bins 2000000 \
    --subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B \
    --nj 32 --inference_nj 1 \
    --ngpu 1 \
    --data_outputs "diar_tokens" \
    --dumpdir "dump_librimix" \
    --cmd_backend "slurm_short_hours" \
    --codec_stage 1 \
    --stage 7 --stop_stage 7
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ./speechlm.sh \
    --task ${task} \
    --data_name "librimix" \
    --data_dir "data_librimix" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}"  \
    --train_config "${train_config}" \
    --audio_format wav \
    --nj 1 --inference_nj 1 \
    --ngpu 1 \
    --data_outputs "diar_tokens" \
    --dumpdir "dump_librimix" \
    --cmd_backend "slurm_short_hours" \
    --stage 8 --stop_stage 8
fi