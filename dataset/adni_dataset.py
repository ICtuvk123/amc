import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
import scanpy as sc
import pickle
from torchvision.transforms import Compose, ToTensor, Normalize
import nibabel as nib
import random
import csv
import os
from collections import Counter
import torch
from torchvision import transforms
import pathlib
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler

def process_labels(adni_subj, diagnosis_path):
    diagnosis_df = pd.read_csv(diagnosis_path)
    subject_ids = []
    dates = []
    with open(adni_subj, "r") as fin:
        for line in fin:
            line = line.strip()
            parts = line.split("_")
            subject_id = "_".join(parts[:3])
            date = parts[-1]
            subject_ids.append(subject_id)
            dates.append(date)
    label_df = pd.DataFrame({"PTID": subject_ids, "EXAMDATE": dates})
    label_df["EXAMDATE"] = pd.to_datetime(label_df["EXAMDATE"])
    diagnosis_df["EXAMDATE"] = pd.to_datetime(diagnosis_df["EXAMDATE"])

    diagnosis_df = diagnosis_df.sort_values(by="EXAMDATE")

    def get_closest_date(row, df):
        # Filter diagnosis_df for the same PTID
        same_id = df[df["PTID"] == row["PTID"]]
        if not same_id.empty:
            # Find the closest date
            closest_date_idx = (same_id["EXAMDATE"] - row["EXAMDATE"]).abs().idxmin()
            return df.loc[closest_date_idx]

    closest_dates = label_df.apply(get_closest_date, df=diagnosis_df, axis=1)

    label_df["DIAGNOSIS"] = closest_dates["DIAGNOSIS"]
    label_df["DIAGNOSIS"].fillna(1.0, inplace=True)
    return label_df

def load_image(image_np, label_df, template_path):
    labels = [label_df.loc[i, "DIAGNOSIS"] for i in range(len(image_np))]
    img_template = nib.load(template_path)
    img_fdata = img_template.get_fdata(dtype=np.float32)
    mask_gm = (img_fdata == 150).ravel()

    transform = Compose(
        [
            ToTensor(),
            Normalize(mean=[img_fdata.mean()], std=[img_fdata.std()]),
        ]
    )

    image_data = {}
    for i in range(len(image_np)):
        subj1 = image_np[i]
        i_label = label_df.loc[i]

        subj_gm_3d = np.zeros(mask_gm.shape, dtype=np.float32)
        subj_gm_3d.ravel()[mask_gm] = subj1
        subj_gm_3d = subj_gm_3d.reshape(img_fdata.shape)

        subj_gm_3d = transform(subj_gm_3d)
        image_data[i_label["PTID"]] = subj_gm_3d[None, :, :, :]

    return image_data

class ADNIDataset(Dataset):
    def __init__(
        self,
        data_root,
        modalities,
        split,
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.data_root = pathlib.Path(data_root)
        with open(self.data_root.joinpath("PTID_splits.json"), "r") as fin:
            self.data_ptid = json.load(fin)[split]
            
        self.genomic_data = None
        all_modality = {}
        if "genomic" in self.modalities:
            genomic_path = self.data_root.joinpath("genomics/genomic_merged.h5ad")
            if os.path.exists(f"{str(genomic_path)}.cache"):
                with open(f"{str(genomic_path)}.cache", "rb") as fin:
                    self.genomic_data = pickle.load(fin)
            else:
                df = sc.read_h5ad(genomic_path).to_df()
                df = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=0)
                arr = df.values
                scaler = MinMaxScaler(feature_range=(-1, 1))
                arr = scaler.fit_transform(arr)
                genomic_dict = {}
                for i, pid in enumerate(df.index):
                    genomic_dict[pid] = arr[i]
                self.genomic_data = genomic_dict
                with open(f"{str(genomic_path)}.cache", "wb") as fout:
                    pickle.dump(self.genomic_data, fout)
        
        diagnosis_path = self.data_root.joinpath("diagnosis/DXSUM_PDXCONV_22Apr2024.csv")
        if os.path.exists(f"{str(diagnosis_path)}.cache"):
            self.label = pd.read_csv(f"{str(diagnosis_path)}.cache")
        else:
            adni_subj = self.data_root.joinpath("images/240412-Voxel-Level-Imaging/ADNI_subj.txt")
            self.label = process_labels(adni_subj, diagnosis_path)
            self.label.to_csv(f"{diagnosis_path}.cache")

        self.image_data = None
        if "image" in self.modalities:
            image_path = self.data_root.joinpath("images/240412-Voxel-Level-Imaging/ADNI_G.npy")
            if os.path.exists(f"{str(image_path)}.cache"):
                with open(f"{str(image_path)}.cache", "rb") as fin:
                    self.image_data = pickle.load(fin)
            else:
                image_template_path = self.data_root.joinpath("images/240412-Voxel-Level-Imaging/BLSA_SPGR+MPRAGE_averagetemplate_muse_seg_DS222.nii.gz")
                image_np = np.load(image_path, mmap_mode="r")
                self.image_data = load_image(image_np, self.label, image_template_path)
                with open(f"{str(image_path)}.cache", "wb") as fout:
                    pickle.dump(self.image_data, fout)
            # all_modality['images'] = self.image_data

        self.biospecimen = None
        biospecimen_path = self.data_root.joinpath("biospecimen/biospecimen_merged.csv")
        if "biospecimen" in self.modalities:
            self.biospecimen = pd.read_csv(biospecimen_path)
            self.biospecimen.set_index("PTID", inplace=True)
            self.biospecimen.fillna(0, inplace=True)
            # all_modality['biospecimen'] = self.biospecimen

        self.clinical = None
        if "clinical" in self.modalities:
            clinical_path = self.data_root.joinpath("clinical/clinical_merged.csv")
            self.clinical = pd.read_csv(clinical_path, index_col=0)
            columns_to_exclude = [
                col
                for col in self.clinical.columns
                if col.startswith("PTCOGBEG")
                or col.startswith("PTADDX")
                or col.startswith("PTADBEG")
            ]
            if len(columns_to_exclude) > 0:
                self.clinical = self.clinical.drop(columns_to_exclude, axis=1)
                print("Drop Columns:", columns_to_exclude)
            self.clinical.fillna(0, inplace=True)
            # all_modality['clinical'] = self.clinical

        self.label = self.label.set_index("PTID")

        tmp_data_ptid = []
        # print(self.data_ptid)
        for ptid in self.data_ptid:
            appd = True
            # print(ptid)
            data_frame = {}
            if self.genomic_data is not None:
                if ptid not in self.genomic_data:
                    appd = False
            if self.image_data is not None:
                if ptid not in self.image_data:
                    appd = False
            if self.biospecimen is not None:
                if ptid not in self.biospecimen.index:
                    appd = False
            if self.clinical is not None:
                if ptid not in self.clinical.index:
                    appd = False
            if ptid not in self.label.index:
                appd = False
            if appd:
                tmp_data_ptid.append(ptid)
            
        self.data_ptid = tmp_data_ptid
        if self.image_data is not None:
            self.image_template_key = list(self.image_data.keys())[0]
        # print(self.data_ptid)
        self.all_labels = self.label#.loc[self.data_ptid]["DIAGNOSIS"]
        # print(set(self.all_labels.to_numpy().tolist()))
        # self.all_labels = (self.all_labels.to_numpy().astype(np.int32) - 1).tolist()
        # print(self.data_ptid)
        count_dict = {}
        for pid in self.data_ptid:
            tmp_label = self.all_labels.loc[pid]["DIAGNOSIS"]
            tmp_label = tmp_label.values[0] if hasattr(tmp_label, "values") else tmp_label
            tmp_label = int(tmp_label - 1)
            if tmp_label not in count_dict:
                count_dict[tmp_label] = 0
            count_dict[tmp_label] += 1
        print(count_dict)
        
    def __len__(self):
        return len(self.data_ptid)

    def __getitem__(self, idx):
        frame_data = {}

        frame_data["idx"] = idx
        ptid = self.data_ptid[idx]
        # print(ptid)
        # print(self.genomic_data.obs[ptid])
        if self.genomic_data is not None:
            # print(ptid in self.genomic_data.var.index)
            # if ptid in self.genomic_data.var.index:
            frame_data["genomic"] = self.genomic_data[ptid]
            # print(frame_data["genomic"])
            frame_data["genomic"] = np.nan_to_num(frame_data["genomic"], nan=0.0).astype(np.float32)
            # print(frame_data["genomic"])
        tmp_label = self.all_labels.loc[ptid]["DIAGNOSIS"]
        tmp_label = tmp_label.values[0] if hasattr(tmp_label, "values") else tmp_label
        frame_data["target"] = int(tmp_label - 1)
        if self.image_data is not None:
            # if ptid in self.image_data:
            frame_data["image"] = self.image_data[ptid]

        if self.biospecimen is not None:
            # if ptid in self.biospecimen.index:
            frame_data["biospecimen"] = self.biospecimen.loc[ptid].values.astype(
                np.float32
            )

        if self.clinical is not None:
            # if ptid in self.clinical.index:
            frame_data["clinical"] = self.clinical.loc[ptid].values.astype(
                np.float32
            )
        return frame_data
    
def get_dataset(data_dir, modalities):
    ds_train = ADNIDataset(data_dir, modalities=modalities, split='training')
    ds_val = ADNIDataset(data_dir, modalities=modalities, split='validation')
    ds_test = ADNIDataset(data_dir, modalities=modalities, split='testing')
    
    return ds_train, ds_val, ds_test