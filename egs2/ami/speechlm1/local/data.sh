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
output_dir="data"
wav_out_dir="${output_dir}/wav"
pit_method="arrive" # can be arrive or most time ordered
output_format="event" # can be event or frame-based
dur=3 # duration of each audio file
skip=3 # skip duration 
spk_format="spk_idx"
mic="sdm"
use_extra_info=false
use_random_durs=false
frame_res=0.1
curr_sets=
data_outputs=
_opts=

log "$0 $*"
. utils/parse_options.sh

log "Data preparation"
log "Prepare dataset ${curr_sets}"
log "Parsing the ${data_outputs}"

if ${use_extra_info}; then
    _opts+="--use_extra_info "
fi

if ${use_random_durs}; then
    _opts+="--use_random_durs "
fi

python local/data_prep_class.py \
  --dataset_dir "${dataset_dir}" \
  --output_dir "${output_dir}" \
  --wav_out_dir "${wav_out_dir}" \
  --mic "${mic}" \
  --dur "${dur}" \
  --skip "${skip}" \
  --pit_method "${pit_method}" \
  --spk_format "${spk_format}" \
  --curr_sets "${curr_sets}" \
  --frame_res "${frame_res}" \
  --data_outputs "${data_outputs}" \
  ${_opts} 


log "Successfully finished. [elapsed=${SECONDS}s]"
