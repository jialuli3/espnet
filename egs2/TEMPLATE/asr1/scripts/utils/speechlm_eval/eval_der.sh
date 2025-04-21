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

# . ./path.sh
# . ./cmd.sh

stage=1
stop_stage=100
nj=1
inference_nj=1

gen_dir=
output_dir=
# rttm options
ref_rttm_file=
uem_file=
hyp_format=
speaker_order_method=
collar=0.25
spk_format=spk_idx
test_wav_scp=
apply_clustering=false

python=python3
skip_interval=1
_opts=

log "$0 $*"
. utils/parse_options.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Make rttm file"
    
    _scoredir="${output_dir}/scoring"
    _textgrid_dir="${output_dir}/textgrids"
    _opts+="--skip_interval ${skip_interval} "

    if [ "${skip_interval}" -eq 1 ]; then
        _scoredir+="_overlap"
        _textgrid_dir+="_overlap"
    fi


    if ${apply_clustering}; then
        _scoredir+="_clustered"
        _textgrid_dir+="_clustered"
        _opts+="--apply_clustering"
    fi

    mkdir -p "${_scoredir}"
    mkdir -p "${_textgrid_dir}"

    cp $ref_rttm_file ${_scoredir}/ref.rttm
    gen_text=${gen_dir}/gen_list

    log "apply clustering ${apply_clustering}"
    log "opt ${_opts}"
    python scripts/utils/speechlm_eval/make_rttm_class.py \
        --ref_rttm_file ${ref_rttm_file} \
        --hyp_output_file ${gen_text} \
        --hyp_format ${hyp_format} \
        --scoring_dir ${_scoredir} \
        --speaker_order_method ${speaker_order_method} \
        --textgrid_dir ${_textgrid_dir} \
        --spk_format ${spk_format} \
        --test_wav_scp ${test_wav_scp} \
        ${_opts}

        # Scoring
    log "compute DER"
    spyder ${_scoredir}/ref.rttm ${_scoredir}/hyp.rttm -u ${uem_file} -p -c ${collar} > ${_scoredir}/results
    log "write to file ${_scoredir}/results"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
