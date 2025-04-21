import numpy as np
from espnet2.bin.spk_inference import Speech2Embedding

# from checkpoints trained by oneself
# model_path="voxcelebs12_rawnet3_checkpoint/40epoch.pth"
# speech2spk_embed = Speech2Embedding(model_file=model_path, train_config="config.yaml")
# embedding = speech2spk_embed(np.zeros(32000))

speech2spk_embed = Speech2Embedding.from_pretrained(model_tag="espnet/voxcelebs12_rawnet3")
embedding = speech2spk_embed(np.zeros(16500))

