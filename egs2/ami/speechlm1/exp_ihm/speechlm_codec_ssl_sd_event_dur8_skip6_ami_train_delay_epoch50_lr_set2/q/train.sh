#!/bin/bash
cd /ocean/projects/cis210027p/jlix/espnet/egs2/ami/speechlm1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python3 -m espnet2.bin.speechlm_train --use_preprocessor true --token_list data/token_list/codec_ssl_sd_event_dur8_skip6_ami/token_list.json --token_bias data/token_list/codec_ssl_sd_event_dur8_skip6_ami/token_bias.json --non_linguistic_symbols none --cleaner none --g2p g2p_en --subword_choice sentencepiece --subword_model dump/raw_codec_ssl_sd_event_dur8_skip6_ami/train_ihm/token_lists/text_bpe --multi_task_dataset true --sharded_dataset true --resume true --output_dir exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2 --config conf/train_delay_epoch50_lr_set2.yaml --sampler_allow_duplication true --train_data_path_and_name_and_type dump/raw_codec_ssl_sd_event_dur8_skip6_ami/train_ihm/stats/split2/JOB/data.json,_,dataset_json --valid_data_path_and_name_and_type dump/raw_codec_ssl_sd_event_dur8_skip6_ami/dev_ihm/stats/split2/JOB/data.json,_,dataset_json --train_shape_file exp_ihm/speechlm_stats_codec_ssl_sd_event_dur8_skip6_ami/train/split2/dec_seq_shape.JOB --valid_shape_file exp_ihm/speechlm_stats_codec_ssl_sd_event_dur8_skip6_ami/valid/split2/dec_seq_shape.JOB --ngpu 2 --multiprocessing_distributed True 
EOF
) >exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/train.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/train.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( python3 -m espnet2.bin.speechlm_train --use_preprocessor true --token_list data/token_list/codec_ssl_sd_event_dur8_skip6_ami/token_list.json --token_bias data/token_list/codec_ssl_sd_event_dur8_skip6_ami/token_bias.json --non_linguistic_symbols none --cleaner none --g2p g2p_en --subword_choice sentencepiece --subword_model dump/raw_codec_ssl_sd_event_dur8_skip6_ami/train_ihm/token_lists/text_bpe --multi_task_dataset true --sharded_dataset true --resume true --output_dir exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2 --config conf/train_delay_epoch50_lr_set2.yaml --sampler_allow_duplication true --train_data_path_and_name_and_type dump/raw_codec_ssl_sd_event_dur8_skip6_ami/train_ihm/stats/split2/JOB/data.json,_,dataset_json --valid_data_path_and_name_and_type dump/raw_codec_ssl_sd_event_dur8_skip6_ami/dev_ihm/stats/split2/JOB/data.json,_,dataset_json --train_shape_file exp_ihm/speechlm_stats_codec_ssl_sd_event_dur8_skip6_ami/train/split2/dec_seq_shape.JOB --valid_shape_file exp_ihm/speechlm_stats_codec_ssl_sd_event_dur8_skip6_ami/valid/split2/dec_seq_shape.JOB --ngpu 2 --multiprocessing_distributed True  ) &>>exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/train.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/train.log
echo '#' Accounting: end_time=$time2 >>exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/train.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/train.log
echo '#' Finished at `date` with status $ret >>exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/train.log
[ $ret -eq 137 ] && exit 100;
touch exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/q/done.1112422
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  --time 48:00:00 -p GPU-shared --gres=gpu:h100-80:2 -c 2 --job-name exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/train.log  --open-mode=append -e exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/q/train.log -o exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/q/train.log  /ocean/projects/cis210027p/jlix/espnet/egs2/ami/speechlm1/exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/q/train.sh >>exp_ihm/speechlm_codec_ssl_sd_event_dur8_skip6_ami_train_delay_epoch50_lr_set2/q/train.log 2>&1
