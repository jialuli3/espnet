#!/bin/bash

train_set="train"
valid_set="dev"
test_set="test"
train_config="conf/train_target_pretrain_librimix_lr2e-4.yaml"
inference_config="conf/decode_diar_token.yaml"
task="codec_ssl_sd_event_sad_od"
inference_model="valid.acc_all.ave_3best.pth"
data_dir="data_librimix_new"
dump_dir="dump_librimix_new"
exp_dir="exp_librimix_new"
dataset_name="librimix"
#inference_model="latest.pth"


stage=3             # Processes starts from the specified stage.
stop_stage=3     # Processes is stopped at the specified stage.

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  ./speechlm.sh \
    --task ${task} \
    --data_name ${dataset_name} \
    --data_dir "${data_dir}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}"  \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --audio_format wav \
    --codec_choice ESPnet --codec_hf_model_tag ftshijt/espnet_codec_dac_large_v1.4_360epoch \
    --ssl_choice espnet_hubert --ssl_nlayer 18 --ssl_checkpoint_path exp/kmeans/38epoch.pth --ssl_kmeans_path exp/kmeans/xeus_18_5000clusters/km_5000.mdl --ssl_batch_bins 1600000 \
    --subword_choice huggingface --subword_model HuggingFaceTB/SmolLM-1.7B \
    --nj 4 --inference_nj 1 \
    --ngpu 1 \
    --data_outputs "diar_tokens" \
    --dumpdir "${dump_dir}" \
    --cmd_backend "slurm" \
    --codec_stage 100 \
    --use_sad true \
    --use_od true \
    --diar_token_list "${data_dir}/diar_corpus/diar_corpus_event_sad_od_spk_count_spk5_dur30" \
    --expdir "${exp_dir}" \
    --output_format "event" \
    --dur 30 --skip 10 \
    --stage 5 --stop_stage 7
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  ./speechlm.sh \
    --task ${task} \
    --data_name ${dataset_name} \
    --data_dir "${data_dir}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}"  \
    --train_config "${train_config}" \
    --audio_format wav \
    --nj 1 --inference_nj 1 \
    --ngpu 1 \
    --data_outputs "diar_tokens" \
    --dumpdir "${dump_dir}" \
    --cmd_backend "slurm_1gpu" \
    --expdir "${exp_dir}" \
    --stage 8 --stop_stage 8
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  ./speechlm.sh \
    --task ${task} \
    --data_name ${dataset_name} \
    --data_dir "${data_dir}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}"  \
    --train_config "${train_config}" \
    --inference_model "${inference_model}" \
    --inference_config "${inference_config}" \
    --nj 1 --inference_nj 1 \
    --ngpu 1 \
    --data_outputs "diar_tokens" \
    --dumpdir "${dump_dir}" \
    --cmd_backend "slurm" \
    --expdir "${exp_dir}" \
    --tokenizer "diar_tokenizer" \
    --output_format "event" \
    --check_overlap_sad true \
    --apply_clustering false \
    --apply_local_speaker_matching false \
    --gpu_inference true \
    --collar 0.0 \
    --skip_interval 30 \
    --decoding true \
    --stage 9 --stop_stage 10
fi