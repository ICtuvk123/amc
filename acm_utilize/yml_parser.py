import yaml
import easydict
import os
import sys
sys.path.append(os.environ['ACTIVE_ROOT'])
from model.model_util import Modality

def input_dim(input_axis, freq_bands, channels):
    return input_axis * ((freq_bands * 2)+1) + channels


class YmlConfig(object):
    def __init__(self, yml_path) -> None:
        self.yml_path = yml_path
        with open(self.yml_path, 'r') as fin:
            self.obj = easydict.EasyDict(yaml.load(fin, yaml.Loader))
        
        
    def flush(self):
        with open(self.yml_path, 'r') as fin:
            self.obj = easydict.EasyDict(yaml.load(fin, yaml.Loader))
    
    def parse_to_modality(self, modality_dict: easydict.EasyDict):

        max_freq = modality_dict.get('max_freq', 1)
        freq_bands = modality_dict.get('freq_bands', 6)
        freq_base = modality_dict.get('freq_base', 2)
        additional_dim = modality_dict.get('additional_dim', 384)
        modality_latent_len = modality_dict.get('modality_latent_len', 20)
        modelity_weight = modality_dict.get('modelity_weight', 0.5)
        # modality_latent_len = modality_dict.get('modality_latent_len', 20)

        # For 'cut_into' mode (patch-based processing), use additional_dim directly
        # For other modes, use frequency encoding calculation
        input_dim_value = additional_dim

        return Modality(
            max_freq=max_freq,
            freq_bands=freq_bands,
            freq_base=freq_base,
            input_dim=input_dim_value,
            modality_latent_len=modality_latent_len,
            modelity_weight = modelity_weight
        )
        
            
if __name__ == '__main__':
    data = YmlConfig("config/multi_modal_config.yml")
    print(data.obj.modality['screenImg'])