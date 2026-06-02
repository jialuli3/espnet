#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from dataclasses import dataclass
from typing import List, Tuple


# (1) Modality Definitions
@dataclass
class Modality:
    discrete: bool = True


MODALITIES = {}
# Discrete
MODALITIES["codec"] = Modality()
MODALITIES["ssl"] = Modality()
MODALITIES["codec_ssl"] = Modality()
MODALITIES["text_bpe"] = Modality()
MODALITIES["g2p"] = Modality()
MODALITIES["spk"] = Modality()
MODALITIES["diar_tokenizer"] = Modality()
MODALITIES["diar_tokenizer_multistream"] = Modality()
MODALITIES["class"] = Modality()
MODALITIES["bool"] = Modality()
MODALITIES["video_ssl"] = Modality()
MODALITIES["svs_lb"] = Modality()

# continuous
MODALITIES["wav"] = Modality(discrete=False)
MODALITIES["text_emb"] = Modality(discrete=False)
MODALITIES["ssl_feat"] = Modality(discrete=False)

# dialogue
MODALITIES["dialogue"] = Modality()

# END OF MODALITY DEFINITION #


# (2) Task Definition
@dataclass
class SpeechLMTaskTemplate:
    conditions: List[Tuple[str, str, str]]
    targets: List[Tuple[str, str, str]]
    use_task_identifier: bool = True
    fixed_length_key: str = ""

    @property
    def data_triplets(self):
        all_entries = self.conditions + self.targets
        return all_entries
    
    @property
    def n_conditions(self):
        return len(self.conditions)
    
    @property
    def n_targets(self):
        return len(self.targets)

    @property
    def data_triplets_string(self):
        ans = ""
        for entry in self.data_triplets:
            ans = ans + ",".join(entry) + " "
        return ans

    @property
    def condition_string(self):
        ans = ""
        for entry in self.conditions:
            ans = ans + ",".join(entry) + " "
        return ans

    @property
    def target_string(self):
        ans = ""
        for entry in self.targets:
            ans = ans + ",".join(entry) + " "
        return ans


SPEECHLM_TASKS = dict()

SPEECHLM_TASKS["textlm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["audiolm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["ssl_audiolm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("wav.scp", "ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "g2p", "text"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["ssl_tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text")],
    targets=[("wav.scp", "ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["bpe_tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["asr"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["ssl_asr"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "ssl", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["mt"] = SpeechLMTaskTemplate(
    conditions=[("src_text", "text_bpe", "text")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["text2audio"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_emb", "kaldi_ark")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["visual_tts"] = SpeechLMTaskTemplate(
    conditions=[
        ("text", "g2p", "text"),
        ("utt2spk", "spk", "text"),
        ("video.scp", "video_ssl", "kaldi_ark"),
    ],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

# SPEECHLM_TASKS["vc"] = SpeechLMTaskTemplate(
#     conditions=[("src_wav.scp", "codec", "kaldi_ark"), ("utt2spk", "spk", "text")],
#     targets=[("wav.scp", "codec", "kaldi_ark")],
# )

# SPEECHLM_TASKS["ssl2codec"] = SpeechLMTaskTemplate(
#     conditions=[("ssl_wav.scp", "ssl", "kaldi_ark"), ("utt2spk", "spk", "text")],
#     targets=[("wav.scp", "codec", "kaldi_ark")],
# )

# SPEECHLM_TASKS["svs"] = SpeechLMTaskTemplate(
#     conditions=[("label", "svs_lb", "text")],
#     targets=[("wav.scp", "codec", "kaldi_ark")],
# )

# SPEECHLM_TASKS["mt"] = SpeechLMTaskTemplate(
#     conditions=[("src_text", "text_bpe", "text")],
#     targets=[("text", "text_bpe", "text")],
# )

# SPEECHLM_TASKS["st"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "ssl", "kaldi_ark")],
#     targets=[("src_text", "text_bpe", "text"), ("text", "text_bpe", "text")],
# )

# SPEECHLM_TASKS["se"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec", "kaldi_ark")],
#     targets=[("spk1.scp", "codec", "kaldi_ark")],
# )

# codec_ssl tasks:
# SPEECHLM_TASKS["codec_ssl_sd_event_ovl_aux_spk_count_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_ovl_aux_spk_count_dur30_skip10", "diar_tokenizer", "diar_tokens_event_ovl_aux_spk_count_dur30_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_ovl_aux_spk_count_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_od_ovl_aux_spk_count_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_od_ovl_aux_spk_count_dur30_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_ovl_aux_spk_count_dur30_skip5"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_od_ovl_aux_spk_count_dur30_skip5", "diar_tokenizer", "diar_tokens_event_sad_od_ovl_aux_spk_count_dur30_skip5")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_ovl_aux_spk_count_dur20_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_od_ovl_aux_spk_count_dur20_skip10", "diar_tokenizer", "diar_tokens_event_sad_od_ovl_aux_spk_count_dur20_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_ovl_aux_spk_count_dur20_skip5"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_od_ovl_aux_spk_count_dur20_skip5", "diar_tokenizer", "diar_tokens_event_sad_od_ovl_aux_spk_count_dur20_skip5")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_ovl_aux_spk_count_dur10_skip3"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_od_ovl_aux_spk_count_dur10_skip3", "diar_tokenizer", "diar_tokens_event_sad_od_ovl_aux_spk_count_dur10_skip3")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_ovl_aux_spk_count_durf_dur20_skip5"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_od_ovl_aux_spk_count_durf_dur20_skip5", "diar_tokenizer", "diar_tokens_event_sad_od_ovl_aux_spk_count_durf_dur20_skip5")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_ovl_aux_spk_count_local_dur20_skip5"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_od_ovl_aux_spk_count_local_dur20_skip5", "diar_tokenizer", "diar_tokens_event_sad_od_ovl_aux_spk_count_local_dur20_skip5")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_ovl_aux_spk_count_ipu_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_od_ovl_aux_spk_count_ipu_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_od_ovl_aux_spk_count_ipu_dur30_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_ovl_aux_spk_count_random"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_ovl_aux_spk_count_random_30_20_10", "diar_tokenizer", "diar_tokens_event_ovl_aux_spk_count_random_30_20_10")], 
# )

SPEECHLM_TASKS["codec_ssl_sd_frame_sad_multi_subtask_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sad_multi_subtask_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sad_multi_subtask_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_multi_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_multi_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_multi_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_sad_multi_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sad_multi_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sad_multi_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_multi_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sc_sad_multi_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sc_sad_multi_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_spk_count_multi_subtask_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sc_sad_spk_count_multi_subtask_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sc_sad_spk_count_multi_subtask_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_sc_spk_count_after_multi_subtask_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sc_spk_count_after_multi_subtask_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sc_spk_count_after_multi_subtask_dur30_skip10")], 
)

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_ovl_aux_spk_count_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_dur30_skip10", "diar_tokenizer", "diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_dur30_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_ovl_aux_spk_count_dur20_skip5"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_dur20_skip5", "diar_tokenizer", "diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_dur20_skip5")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_ovl_aux_spk_count_dur10_skip5"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_dur10_skip5", "diar_tokenizer", "diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_dur10_skip5")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_ovl_aux_spk_count_dur10_skip3"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_dur10_skip3", "diar_tokenizer", "diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_dur10_skip3")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_ovl_aux_spk_count_time_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_time_dur30_skip10", "diar_tokenizer", "diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_time_dur30_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_ovl_aux_spk_count_time_dur20_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_time_dur20_skip10", "diar_tokenizer", "diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_time_dur20_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_ovl_aux_spk_count_time_dur20_skip5"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_time_dur20_skip5", "diar_tokenizer", "diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_time_dur20_skip5")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_ovl_aux_spk_count_multi_subtask_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_multi_subtask_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_multi_subtask_dur30_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_ovl_aux_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_ovl_aux_dur30_skip10", "diar_tokenizer", "diar_tokens_frame_sc_ovl_aux_dur30_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_ovl_aux_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_ovl_aux_dur30_skip10", "diar_tokenizer", "diar_tokens_frame_sc_sad_ovl_aux_dur30_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_od_ovl_aux_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_od_ovl_aux_dur30_skip10", "diar_tokenizer", "diar_tokens_frame_sc_od_ovl_aux_dur30_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_ovl_aux_spk_count_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_ovl_aux_spk_count_dur30_skip10", "diar_tokenizer", "diar_tokens_frame_sc_ovl_aux_spk_count_dur30_skip10")], 
# )

SPEECHLM_TASKS["codec_ssl_sd_event_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_dur30_skip10", "diar_tokenizer", "diar_tokens_event_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_dur10_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_od_dur10_skip10", "diar_tokenizer", "diar_tokens_event_sad_od_dur10_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_od_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_od_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_cache_event_sad_od_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[
        ("cache_wav.scp", "codec_ssl", "multicol_kaldi_ark"),
        ("wav.scp", "codec_ssl", "kaldi_ark"),
    ],
    targets=[("diar_tokens_event_sad_od_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_od_dur30_skip10")],
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_dur60_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_od_dur60_skip10", "diar_tokenizer", "diar_tokens_event_sad_od_dur60_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_spk_count_after_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_od_spk_count_after_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_od_spk_count_after_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_od_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_od_dur30_skip10", "diar_tokenizer", "diar_tokens_event_od_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_od_after_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_od_after_dur30_skip10", "diar_tokenizer", "diar_tokens_event_od_after_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_after_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_od_after_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_od_after_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_after_spk_count_after_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_od_after_spk_count_after_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_od_after_spk_count_after_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_spk_count_after_od_after_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_spk_count_after_od_after_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_spk_count_after_od_after_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_after_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_after_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_after_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_od_sad_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_od_sad_dur30_skip10", "diar_tokenizer", "diar_tokens_event_od_sad_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_od_sad_spk_count_after_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_od_sad_spk_count_after_dur30_skip10", "diar_tokenizer", "diar_tokens_event_od_sad_spk_count_after_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_ovl_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_ovl_dur30_skip10", "diar_tokenizer", "diar_tokens_event_ovl_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_spk_count_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_spk_count_dur30_skip10", "diar_tokenizer", "diar_tokens_event_spk_count_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_spk_count_after_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_spk_count_after_dur30_skip10", "diar_tokenizer", "diar_tokens_event_spk_count_after_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_spk_count_after_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_spk_count_after_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_spk_count_after_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_ovl_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_ovl_dur30_skip10", "diar_tokenizer", "diar_tokens_event_sad_ovl_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_dur30_skip10", "diar_tokenizer", "diar_tokens_frame_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_sc_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sc_dur30_skip10", "diar_tokenizer", "diar_tokens_frame_sc_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sc_sad_dur30_skip10", "diar_tokenizer", "diar_tokens_frame_sc_sad_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_multi_subtask_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sc_sad_od_multi_subtask_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sc_sad_od_multi_subtask_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_multi_subtask_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sc_sad_multi_subtask_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sc_sad_multi_subtask_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_spk_count_multi_subtask_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sc_sad_od_spk_count_multi_subtask_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sc_sad_od_spk_count_multi_subtask_dur30_skip10")], 
)

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc1_sad_multi_subtask_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc1_sad_multi_subtask_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sc1_sad_multi_subtask_dur30_skip10")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc2_sad_multi_subtask_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc2_sad_multi_subtask_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sc2_sad_multi_subtask_dur30_skip10")], 
# )

SPEECHLM_TASKS["codec_ssl_sd_frame_sc_od_multi_subtask_dur30_skip10"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sc_od_multi_subtask_dur30_skip10", "diar_tokenizer_multistream", "diar_tokens_frame_sc_od_multi_subtask_dur30_skip10")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event", "diar_tokenizer", "diar_tokens_event")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_od"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_od", "diar_tokenizer", "diar_tokens_event_od")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_od_after"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_od_after", "diar_tokenizer", "diar_tokens_event_od_after")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad", "diar_tokenizer", "diar_tokens_event_sad")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_ovl"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_ovl", "diar_tokenizer", "diar_tokens_event_ovl")], 
)


# SPEECHLM_TASKS["codec_ssl_sd_event_od_ovl_aux"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_od_ovl_aux", "diar_tokenizer", "diar_tokens_event_od_ovl_aux")], 
# )

SPEECHLM_TASKS["codec_ssl_sd_event_spk_count"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_spk_count", "diar_tokenizer", "diar_tokens_event_spk_count")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_spk_count_after"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_spk_count_after", "diar_tokenizer", "diar_tokens_event_sad_spk_count_after")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_spk_count_after"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_od_spk_count_after", "diar_tokenizer", "diar_tokens_event_sad_od_spk_count_after")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_sad_od"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_sad_od", "diar_tokenizer", "diar_tokens_event_sad_od")], 
)

SPEECHLM_TASKS["codec_ssl_sd_event_spk_count_after"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_event_spk_count_after", "diar_tokenizer", "diar_tokens_event_spk_count_after")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame", "diar_tokenizer", "diar_tokens_frame")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_pad"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_pad", "diar_tokenizer", "diar_tokens_frame_pad")], 
)


SPEECHLM_TASKS["codec_ssl_sd_frame_sad_pad_multi_subtask"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sad_pad_multi_subtask", "diar_tokenizer_multistream", "diar_tokens_frame_sad_pad_multi_subtask")], 
)

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc", "diar_tokenizer", "diar_tokens_frame_sc")], 
# )

SPEECHLM_TASKS["codec_ssl_sd_frame_sad"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sad", "diar_tokenizer", "diar_tokens_frame_sad")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_od"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_od", "diar_tokenizer", "diar_tokens_frame_od")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_od_after"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_od_after", "diar_tokenizer", "diar_tokens_frame_od_after")], 
)


SPEECHLM_TASKS["codec_ssl_sd_frame_sad_od"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sad_od", "diar_tokenizer", "diar_tokens_frame_sad_od")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_spk_count_after"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_spk_count_after", "diar_tokenizer", "diar_tokens_frame_spk_count_after")], 
)

SPEECHLM_TASKS["codec_ssl_sd_frame_sad_spk_count_after"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sad_spk_count_after", "diar_tokenizer", "diar_tokens_frame_sad_spk_count_after")], 
)


SPEECHLM_TASKS["codec_ssl_sd_frame_sad_od_spk_count_after"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("diar_tokens_frame_sad_od_spk_count_after", "diar_tokenizer", "diar_tokens_frame_sad_od_spk_count_after")], 
)

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_multi_subtask"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_multi_subtask", "diar_tokenizer_multistream", "diar_tokens_frame_sc_sad_multi_subtask")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_ovl_aux_spk_count"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_ovl_aux_spk_count", "diar_tokenizer", "diar_tokens_frame_sc_ovl_aux_spk_count")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_od_ovl_aux"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_od_ovl_aux", "diar_tokenizer", "diar_tokens_frame_sc_od_ovl_aux")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_ovl_aux"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_ovl_aux", "diar_tokenizer", "diar_tokens_frame_sc_sad_ovl_aux")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_ovl_aux_spk_count"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_od_ovl_aux_spk_count", "diar_tokenizer", "diar_tokens_frame_sc_sad_od_ovl_aux_spk_count")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_ovl_aux_spk_count_time"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_time", "diar_tokenizer", "diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_time")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_ovl_aux_spk_count"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_ovl_aux_spk_count", "diar_tokenizer", "diar_tokens_event_ovl_aux_spk_count")], 
# )


# SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_ovl_aux_spk_count"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_od_ovl_aux_spk_count", "diar_tokenizer", "diar_tokens_event_sad_od_ovl_aux_spk_count")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_sad_ovl_aux"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_ovl_aux", "diar_tokenizer", "diar_tokens_event_sad_ovl_aux")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_frame_sc_sad_od_ovl_aux_spk_count"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_frame_sc_sad_od_ovl_aux_spk_count", "diar_tokenizer", "diar_tokens_event_frame_sc_sad_od_ovl_aux_spk_count")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_frame_sc_sad_od_ovl_aux_spk_count_multi_subtask"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_frame_sc_sad_od_ovl_aux_spk_count_multi_subtask", "diar_tokenizer_multistream", "diar_tokens_event_frame_sc_sad_od_ovl_aux_spk_count_multi_subtask")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_frame_2tasks_sc_sad_od_ovl_aux_spk_count_multi_2tasks"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_frame_2tasks_sc_sad_od_ovl_aux_spk_count_multi_2tasks", "diar_tokenizer_multistream", "diar_tokens_event_frame_2tasks_sc_sad_od_ovl_aux_spk_count_multi_2tasks")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_ovl_aux_spk_count_multi_subtask"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_od_ovl_aux_spk_count_multi_subtask", "diar_tokenizer_multistream", "diar_tokens_event_sad_od_ovl_aux_spk_count_multi_subtask")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_ovl_aux_spk_count_ipu"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_ovl_aux_spk_count_ipu", "diar_tokenizer", "diar_tokens_event_ovl_aux_spk_count_ipu")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_sad_od_ovl_aux_spk_count_ipu"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_sad_od_ovl_aux_spk_count_ipu", "diar_tokenizer", "diar_tokens_event_sad_od_ovl_aux_spk_count_ipu")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_ovl_aux_spk_count_multi"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_ovl_aux_spk_count_multi", "diar_tokenizer_multistream", "diar_tokens_event_ovl_aux_spk_count_multi")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_spk_count_multi"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_spk_count_multi", "diar_tokenizer_multistream", "diar_tokens_event_spk_count_multi")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_overlap_exp_spk_count"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_event_overlap_exp_spk_count", "diar_tokenizer", "diar_tokens_event_overlap_exp_spk_count")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame", "diar_tokenizer", "diar_tokens_frame")], # frame based model 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_ovl_aux_spk_count_multi"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_ovl_aux_spk_count_multi", "diar_tokenizer_multistream", "diar_tokens_frame_sc_ovl_aux_spk_count_multi")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_sc_sad_od_ovl_aux_spk_count_multi_subtask"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_multi_subtask", "diar_tokenizer_multistream", "diar_tokens_frame_sc_sad_od_ovl_aux_spk_count_multi_subtask")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_dur8_skip6_diar_model"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("diar_tokens_frame_dur8_skip6", "diar_tokenizer", "diar_tokens_frame_dur8_skip6")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_dur8_skip6"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("text_frame_dur8_skip6", "text_bpe", "text_frame_dur8_skip6")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_dur3_skip1"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("text_event_dur3_skip1", "text_bpe", "text_event_dur3_skip1")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_event_dur30_skip10_spk_id"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("text_event_dur30_skip10_spk_id", "text_bpe", "text_event_dur30_skip10_spk_id")], 
# )

# SPEECHLM_TASKS["codec_ssl_sd_frame_dur30_skip10"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("text_frame_dur30_skip10_0.1", "text_bpe", "text_frame_dur30_skip10_0.1")], # frame based model 
# )

# SPEECHLM_TASKS["codec_ssl_asr"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("text", "text_bpe", "text")],
# )

# SPEECHLM_TASKS["codec_ssl_tts"] = SpeechLMTaskTemplate(
#     conditions=[("text", "text_bpe", "text"), ("utt2spk", "spk", "text")],
#     targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
# )

# SPEECHLM_TASKS["codec_ssl_plain_tts"] = SpeechLMTaskTemplate(
#     conditions=[("text", "text_bpe", "text")],
#     targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
# )

# SPEECHLM_TASKS["codec_ssl_audiolm"] = SpeechLMTaskTemplate(
#     conditions=[],
#     targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
# )

# SPEECHLM_TASKS["codec_ssl_se"] = SpeechLMTaskTemplate(
#     conditions=[("mix.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     fixed_length_key="mix.scp",
# )

# SPEECHLM_TASKS["codec_ssl_tse"] = SpeechLMTaskTemplate(
#     conditions=[("mix.scp", "codec_ssl", "kaldi_ark"), ("utt2spk", "spk", "text")],
#     targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     fixed_length_key="mix.scp",
# )

# SPEECHLM_TASKS["aac_codecssl"] = SpeechLMTaskTemplate(
#     conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
#     targets=[("text", "text_bpe", "text")],
# )

# SPEECHLM_TASKS["ag_codecssl"] = SpeechLMTaskTemplate(
#     conditions=[("text", "text_bpe", "text")],
#     targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
# )

# SPEECHLM_TASKS["text_dialogue"] = SpeechLMTaskTemplate(
#     conditions=[],
#     targets=[("dialogue", "dialogue", "dialogue_json")],
# )

# SPEECHLM_TASKS["audio_dialogue"] = SpeechLMTaskTemplate(
#     conditions=[],
#     targets=[("dialogue", "dialogue", "dialogue_json")],
# )

# END OF TASK DEFINITION #

# (3) Special token definition
# a. always reserve 256 slots for special tokens
#    0-31:    general special tokens
#    32-63:   modality identifier
#    64-127:  task identifier
#    128-255: reserved for future
# b. don't delete / modify it, otherwise the model trained
#    previously can become incompatible. New tokens can be
#    added - there are enough slots
special_tokens = [
    "<pad>",
    "<unk>",
    "<blank>",
    "<space>",
    "<continuous_placeholder>",
    "<sos/eos>",
    "<local_sos/eos>",
    "<unkown_task_identifer>",
    "<system_prompt>",
    "<user_input>",
    "<assistant_output>",
    "<eou>",
    "<enroll_spk1>",
    "<enroll_spk2>",
    "<enroll_spk3>",
    "<enroll_spk4>",
    "<enroll_spk5>",
]


def pad_until(token_list, until):
    if len(token_list) >= until:
        return token_list
    for idx in range(len(token_list), until):
        token_list.append(f"<unused_token_{idx}>")
    return token_list


special_tokens = pad_until(special_tokens, 32)

for m in MODALITIES.keys():
    special_tokens.append(f"<{m}_start/end>")
special_tokens = pad_until(special_tokens, 64)

for m in SPEECHLM_TASKS.keys():
    special_tokens.append(f"<{m}_task>")
special_tokens = pad_until(special_tokens, 128)

special_tokens = pad_until(special_tokens, 256)

# END OF SPECIAL TOKEN DEFINITION #
