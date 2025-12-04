import torch
import pathlib
import json


# model/model_util.py
from pathlib import Path

class ModelDumper:
    def __init__(self, path, seed, cpt_name, modalities, args):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        modalities_name = "_".join(sorted(modalities.keys())) if isinstance(modalities, dict) else str(modalities)

        # ---- 容错：命名相关参数给默认值 ----
        fusion = getattr(args, "fusion_type", "na")
        latent = getattr(args, "modality_latent_len", "na")

        root_seed = root / f"seed_{seed}"
        root_seed.mkdir(parents=True, exist_ok=True)

        self.task_path_str = (root_seed / f"{cpt_name}_{modalities_name}_{args.lr}_{args.batch_size}_{fusion}").__str__()
        self.cross_seeds   = (root       / f"{cpt_name}_{modalities_name}_{args.lr}_{args.batch_size}_{latent}_{fusion}").__str__()


    

class Modality:
    def __init__(self, max_freq, freq_bands, freq_base, input_dim, modality_latent_len,
                 modelity_weight = 1., token_fusion_topk_ratio = 0.02) -> None:
        self.max_freq = max_freq
        self.freq_bands = freq_bands
        self.freq_base = freq_base
        self.input_dim = input_dim
        self.modality_latent_len = modality_latent_len
        self.modelity_weight = modelity_weight
        self.token_fusion_topk_ratio = token_fusion_topk_ratio