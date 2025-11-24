import torch
import pathlib
import json

class ModelDumper(object):
    def __init__(self, root, seed, cpt_name, modality, args) -> None:
        root = pathlib.Path(root)
        root.mkdir(exist_ok=True)
        root_seed = root.joinpath(f"{seed}")
        root_seed.mkdir(exist_ok=True)
        modalities_name = ""
        # all_modalities = 
        for kk in modality:
            modalities_name += f"{kk}_{modality[kk].modality_latent_len}_{modality[kk].modelity_weight:.2f}"
        self.model_path = root_seed.joinpath(
            f"{cpt_name}_{modalities_name}.pth"
        )
        self.task_path_str = root_seed.joinpath(f"{cpt_name}_{modalities_name}_{args.lr}_{args.batch_size}_{args.fusion_type}").__str__()
        
        self.cross_seeds = root.joinpath(f"{cpt_name}_{modalities_name}_{args.lr}_{args.batch_size}_{args.modality_latent_len}_{args.fusion_type}").__str__()

        
    def dump(self, model: torch.nn.Module):
        print("Save checkpoint to {}".format(self.model_path))
        torch.save(model.state_dict(), self.model_path)
        
    def dump_json(self, dict_data):
        with open(f"{self.task_path_str}.json", 'w') as fin:
            json.dump(dict_data, fin, indent=2)
            
    def dump_results(self, dict_data):
        with open(f"{self.task_path_str}_results.json", 'w') as fin:
            json.dump(dict_data, fin, indent=2)
    
    def load_results(self):
        with open(f"{self.task_path_str}_results.json", 'r') as fin:
            return json.load(fin)
        
    
    def dump_json_cross_seeds(self, dict_data):
        with open(f"{self.cross_seeds}_results.json", 'w') as fin:
            json.dump(dict_data, fin, indent=2)
        
    

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