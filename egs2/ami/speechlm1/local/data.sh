#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

dataset_dir="/ocean/projects/cis210027p/shared/corpora/amicorpus"
dataset_name="ami"
output_dir="data"
wav_out_dir="${output_dir}/wav"
pit_method="arrive" # can be arrive or most time ordered
output_format="event" # can be event or frame-based
dur=3 # duration of each audio file
skip=3 # skip duration 
spk_format="spk_idx"
overlap_type=""
mic="sdm"
use_spk_count=false
use_spk_count_after=false
use_spk_count_after1=false
use_random_durs=false
use_multistream=false
use_multistream_subtask=false
use_multistream_2tasks=false
use_sc_token=false
use_sc_token_2types=false
use_sc_token_1type=false
use_sad=false
use_od=false
use_sad_after=false
use_od_after=false
use_od_before=false
use_ipu=false
use_timestamp=false
use_dur_format=false
use_local_speaker=false
use_padding=false
check_overlap_sad=false
rttm_file=
frame_res=0.1
context_length=0
curr_sets=
data_outputs=
_opts=

log "$0 $*"
. utils/parse_options.sh

log "Data preparation"
log "Prepare dataset ${curr_sets}"
log "Parsing the ${data_outputs}"

if ${use_spk_count}; then
    _opts+="--use_spk_count "
fi

if ${use_spk_count_after}; then
    _opts+="--use_spk_count_after "
fi

if ${use_spk_count_after1}; then
    _opts+="--use_spk_count_after1 "
fi

if ${use_random_durs}; then
    _opts+="--use_random_durs "
fi

if ${use_multistream}; then
    _opts+="--use_multistream "
fi

if ${use_multistream_subtask}; then
    _opts+="--use_multistream_subtask "
fi

if ${use_multistream_2tasks}; then
    _opts+="--use_multistream_2tasks "
fi

if ${use_sc_token}; then
    _opts+="--use_sc_token "
fi

if ${use_sc_token_2types}; then
    _opts+="--use_sc_token_2types "
fi

if ${use_sc_token_1type}; then
    _opts+="--use_sc_token_1type "
fi

if ${use_sad}; then
    _opts+="--use_sad "
fi

if ${use_od}; then
    _opts+="--use_od "
fi

if ${use_sad_after}; then
    _opts+="--use_sad_after "
fi

if ${use_od_after}; then
    _opts+="--use_od_after "
fi

if ${use_od_before}; then
    _opts+="--use_od_before "
fi

if ${use_timestamp}; then
    _opts+="--use_timestamp "
fi

if ${use_local_speaker}; then
    _opts+="--use_local_speaker "
fi

if ${use_dur_format}; then
    _opts+="--use_dur_format "
fi

if ${use_padding}; then
    _opts+="--use_padding "
fi

if ${use_ipu}; then
    log "use ipu script, ${rttm_file}, ${output_dir}/${curr_sets}/ipu_rttm"
    python local/generate_turn_taking_label.py \
    --rttm "${rttm_file}" \
    --output_path "${output_dir}/${curr_sets}/ipu_rttm"

    _opts+="--use_ipu "
    sort -k2,2 -k4,4n ${output_dir}/${curr_sets}/ipu_rttm > ${output_dir}/${curr_sets}/sorted.ipu_rttm

fi

python local/data_prep_class_refactor_new.py \
  --dataset_dir "${dataset_dir}" \
  --dataset_name "${dataset_name}" \
  --output_dir "${output_dir}" \
  --output_format "${output_format}" \
  --wav_out_dir "${wav_out_dir}" \
  --mic "${mic}" \
  --dur "${dur}" \
  --skip "${skip}" \
  --pit_method "${pit_method}" \
  --spk_format "${spk_format}" \
  --overlap_type "${overlap_type}" \
  --curr_sets "${curr_sets}" \
  --frame_res "${frame_res}" \
  --data_outputs "${data_outputs}" \
  --context_length "${context_length}" \
  ${_opts} 


log "Successfully finished. [elapsed=${SECONDS}s]"
