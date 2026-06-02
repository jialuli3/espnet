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
stop_stage=1
nj=1
inference_nj=1

gen_dir=
output_dir=
# rttm options
ref_rttm_file=
uem_file=
hyp_format=
tokenizer=
speaker_order_method=
collar=0.0
spk_format=spk_idx
test_wav_scp=
apply_clustering=false
use_multistream=false
use_multistream_subtask=false
use_ipu=false
use_dur_format=false
check_overlap_sad=false
check_overlap_od=false
apply_local_speaker_matching=false
reco2dur=
overlap_type=

data_name="ami"
python=python3
skip_interval=1
_opts=

log "$0 $*"
. utils/parse_options.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Make rttm file"
    
    _scoredir="${output_dir}/scoring"
    _textgrid_dir="${output_dir}/textgrids"

    if [[ "${skip_interval}" -eq 1 && ${data_name} == "ami" ]]; then
        _scoredir+="_overlap"
        _textgrid_dir+="_overlap"
    fi

    if [[ -n "${reco2dur}" ]]; then
        _opts+="--reco2dur ${reco2dur} "
    fi

    if [[ -n "${overlap_type}" ]]; then
        _opts+="--overlap_type ${overlap_type} "
    fi

    if ${apply_clustering}; then
        _scoredir+="_clustered"
        _textgrid_dir+="_clustered"
        _opts+="--apply_clustering "
    fi

    if ${use_multistream}; then
        _opts+="--use_multistream "
    fi

    if ${use_multistream_subtask}; then
        _opts+="--use_multistream_subtask "
    fi

    if ${use_ipu}; then
        _opts+="--use_ipu "
    fi

    if ${use_dur_format}; then
        _opts+="--use_dur_format "
    fi

    if ${check_overlap_sad}; then
        _opts+="--check_overlap_sad "
    fi

    if ${check_overlap_od}; then
        _opts+="--check_overlap_od "
    fi

    if ${apply_local_speaker_matching}; then
        _opts+="--apply_local_speaker_matching "
    fi

    mkdir -p "${_scoredir}"
    mkdir -p "${_textgrid_dir}"

    cp $ref_rttm_file ${_scoredir}/ref.rttm
    gen_text=${gen_dir}/gen_list

    log "opts ${_opts}"
    python scripts/utils/speechlm_eval/make_rttm_class_diar_tokenizer_new.py \
        --ref_rttm_file ${ref_rttm_file} \
        --dataset_name ${data_name} \
        --hyp_output_file ${gen_text} \
        --hyp_format ${hyp_format} \
        --tokenizer ${tokenizer} \
        --scoring_dir ${_scoredir} \
        --speaker_order_method ${speaker_order_method} \
        --textgrid_dir ${_textgrid_dir} \
        --spk_format ${spk_format} \
        --test_wav_scp ${test_wav_scp} \
        --skip_interval ${skip_interval} \
        ${_opts}

        # Scoring
    log "compute DER"
    if [ ${data_name} == "librimix" ]; then
        if [ ${hyp_format} == "event_frame" ] || [ ${hyp_format} == "event_frame_2tasks" ]; then
            spyder ${_scoredir}/ref.rttm ${_scoredir}/hyp_event.rttm -p -c ${collar} > ${_scoredir}/results_event_c${collar}
            spyder ${_scoredir}/ref.rttm ${_scoredir}/hyp_frame.rttm -p -c ${collar} > ${_scoredir}/results_frame_c${collar}
            spyder ${_scoredir}/ref.rttm ${_scoredir}/hyp_event_frame.rttm -p -c ${collar} > ${_scoredir}/results_event_frame_c${collar}
            log "write to file ${_scoredir}/results_event_frame_c${collar}"
        else
            out_filename=${_scoredir}/results_c${collar}
            if ${check_overlap_sad}; then
                out_filename+="_sad"
            fi
            if ${check_overlap_od}; then
                out_filename+="_od"
            fi

            spyder ${_scoredir}/ref.rttm ${_scoredir}/hyp.rttm -p -c ${collar} > ${out_filename}
            log "write to file ${out_filename}"
        fi
    else
        out_filename=${_scoredir}/results_c${collar}_int${skip_interval}
        if ${check_overlap_sad}; then
            out_filename+="_sad"
        fi
        if ${check_overlap_od}; then
            out_filename+="_od"
        fi
        if ${apply_local_speaker_matching}; then
            out_filename+="_sm"
        fi
        spyder ${_scoredir}/ref.rttm ${_scoredir}/hyp.rttm -u ${uem_file} -p -c ${collar} > ${out_filename}
        log "write to file ${out_filename}"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "compute count stats"
    _opts=""

    if ${use_multistream}; then
        _opts+="--use_multistream "
    fi

    if ${use_multistream_subtask}; then
        _opts+="--use_multistream_subtask "
    fi
    
    if ${use_ipu}; then
        _opts+="--use_ipu "
    fi

    if [ ${data_name} == "ami" ]; then
        _opts+="--skip_interval ${skip_interval} "
    fi

    ref_text=$(find ${gen_dir} -type f -name 'diar_tokens*' ! -name '*.tmp')
    gen_text=${gen_dir}/gen_list
    log "gen_text ${gen_text}"

    python scripts/utils/speechlm_eval/make_counter_comparison.py \
        --ref_out_file ${ref_text} \
        --hyp_out_file ${gen_text} \
        --hyp_format ${hyp_format} \
        --scoring_dir ${_scoredir} \
        --dataset_name ${data_name} \
        ${_opts}    
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
