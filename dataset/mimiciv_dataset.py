import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
import random
import csv
import os
from collections import Counter
import torch
from torchvision import transforms
import pathlib

class MIMICIVDataset(Dataset):
    def __init__(self, data_root, modalities, split) -> None:
        super().__init__()
        self.data_root = pathlib.Path(data_root)
        self.modalities = modalities
        self.dict_data = {}
        for mm in self.modalities:
            self.dict_data[mm] = torch.load(self.data_root.joinpath(f"{mm}.pt"), weights_only=False)
        label_df = pd.read_csv(self.data_root.joinpath('labels.csv'), index_col='subject_id')
        self.labels = label_df['one_year_mortality'].values.astype(np.int64)
        self.n_labels = len(set(self.labels))
        with open(self.data_root.joinpath('PTID_splits_mimic.json')) as json_file:
            data_split = json.load(json_file)

        if split =='training':
            self.noise_std = 0.0
        else:
            self.noise_std = 0

        data_ids = list(set(data_split[split]))
        id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
        
        self.data_idx = [id_to_idx[id] for id in data_ids if id in id_to_idx]
        if 'lab_x' in self.dict_data:
            train_ids = list(set(data_split['training']))
            train_idx = [id_to_idx[id] for id in train_ids if id in id_to_idx]
            lab_feature_idx = self.dict_data['lab_x'][train_idx].sum(0) != 0
            self.dict_data['lab_x'] = self.dict_data['lab_x'][:, lab_feature_idx]

        train_data_ids = list(set(data_split['training']))
        train_id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
        train_data_idx = [train_id_to_idx[id] for id in train_data_ids if id in train_id_to_idx]
        self.data_min = {modality: data[train_data_idx].min(0) for modality, data in self.dict_data.items()}
        self.data_max = {modality: data[train_data_idx].max(0) for modality, data in self.dict_data.items()}
        self.norm_index = {modality: self.data_min[modality] !=  self.data_max[modality] for modality, data in self.dict_data.items()}
        
        # self.data_dict = {modality: data[indices] for modality, data in data_dict.items()}
        for mm in self.norm_index:
            self.dict_data[mm][:,self.norm_index[mm]] = self.dict_data[mm][:,self.norm_index[mm]] - self.data_min[mm][self.norm_index[mm]] / (self.data_max[mm][self.norm_index[mm]] - self.data_min[mm][self.norm_index[mm]])
        
    
    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        didx = self.data_idx[idx]
        # print(didx)
        ret_item = {}
        ret_item['target'] = self.labels[didx]
        ret_item['idx'] = didx
        for mm in self.dict_data:
            # ret_item[mm] = self.dict_data[mm][didx]
            ret_item[mm] = np.expand_dims(self.dict_data[mm][didx], axis=0)
            ret_item[mm] += np.random.randn(*ret_item[mm].shape) * self.noise_std
            # ret_item[mm] = np.expand_dims(self.dict_data[mm][didx], axis=-1)
            
        return ret_item
    
def get_dataset(data_dir, modalities):
    ds_train = MIMICIVDataset(data_dir, modalities=modalities, split='training')
    ds_val = MIMICIVDataset(data_dir, modalities=modalities, split='validation')
    ds_test = MIMICIVDataset(data_dir, modalities=modalities, split='testing')
    
    return ds_train, ds_val, ds_test