"""Implements dataloaders for ENRICO dataset."""


from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
import random
import csv
import os
import json
from collections import Counter

import torch
from torchvision import transforms

import einops
from PIL import Image
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from tqdm import trange

def _add_screen_elements(tree, element_list):
    """Helper function for extracting UI elements from hierarchy. (unused?)"""
    if 'children' in tree and len(tree['children']) > 0:
        # we are at an intermediate node
        for child in tree['children']:
            _add_screen_elements(child, element_list)
    else:
        # we are at a leaf node
        if 'bounds' in tree and 'componentLabel' in tree:
            # valid leaf node
            nodeBounds = tree['bounds']
            nodeLabel = tree['componentLabel']
            node = (nodeBounds, nodeLabel)
            element_list.append(node)


class EnricoDataset(Dataset):
    """Implements torch dataset class for ENRICO dataset."""
    
    def __init__(self, data_dir, mode="train", 
                 noise_level=0, img_dim_x=128, 
                 img_dim_y=256, 
                 train_split=0.65, val_split=0.15, normalize_image=False, 
                 seq_len=64, hmmt = False, cut_into = False, 
                 flip_label = False, max_step = None, modalities = 'all'):
        """Instantiate ENRICO dataset.

        Args:
            data_dir (str): Data directory.
            mode (str, optional): What data to extract. Defaults to "train".
            noise_level (int, optional): Noise level, as defined in robustness. Defaults to 0.
            img_noise (bool, optional): Whether to apply noise to images or not. Defaults to False.
            wireframe_noise (bool, optional): Whether to apply noise to wireframes or not. Defaults to False.
            img_dim_x (int, optional): Image width. Defaults to 128.
            img_dim_y (int, optional): Image height. Defaults to 256.
            random_seed (int, optional): Seed to split dataset on and shuffle data on. Defaults to 42.
            train_split (float, optional): Percentage of training data split. Defaults to 0.65.
            val_split (float, optional): Percentage of validation data split. Defaults to 0.15.
            test_split (float, optional): Percentage of test data split. Defaults to 0.2.
            normalize_image (bool, optional): Whether to normalize image or not Defaults to False.
            seq_len (int, optional): Length of sequence. Defaults to 64.
        """
        super(EnricoDataset, self).__init__()
        self.noise_level = noise_level
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.seq_len = seq_len
        csv_file = os.path.join(data_dir, "design_topics.csv")
        self.img_dir = os.path.join(data_dir, "screenshots")
        self.wireframe_dir = os.path.join(data_dir, "wireframes")
        self.hierarchy_dir = os.path.join(data_dir, "hierarchies")
        self.hmmt = hmmt
        self.cut_into = cut_into
        self.flip_label = flip_label
        self.max_step = max_step
        self.modalities = modalities

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            example_list = list(reader)

        # the wireframe files are corrupted for these files
        IGNORES = set(["50105", "50109"])
        example_list = [
            e for e in example_list if e['screen_id'] not in IGNORES]

        self.example_list = example_list

        keys = list(range(len(example_list)))
        # shuffle and create splits
        # random.Random(random_seed).shuffle(keys)
        random.shuffle(keys)
        
        img_transforms = [
            transforms.ToTensor(),
            transforms.Resize((img_dim_y, img_dim_x))
        ]

        if mode == "train":
            # train split is at the front
            start_index = 0
            stop_index = int(len(example_list) * train_split)
            img_transforms.append(transforms.RandomCrop((img_dim_y, img_dim_x)))
            img_transforms.append(transforms.RandomHorizontalFlip())
            img_transforms.append(transforms.GaussianBlur(3))
            img_transforms.append(transforms.RandomVerticalFlip())
            # img_transforms.append(transforms.GaussianBlur())
        elif mode == "val":
            # val split is in the middle
            start_index = int(len(example_list) * train_split)
            stop_index = int(len(example_list) * (train_split + val_split))
        elif mode == "test":
            # test split is at the end
            start_index = int(len(example_list) * (train_split + val_split))
            stop_index = len(example_list)

        # only keep examples in the current split
        keys = keys[start_index:stop_index]
        self.keys = keys

        
        
        if normalize_image:
            img_transforms.append(transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        # pytorch image transforms
        self.img_transforms = transforms.Compose(img_transforms)

        # make maps
        topics = set()
        for e in example_list:
            topics.add(e['topic'])
        topics = sorted(list(topics))

        idx2Topic = {}
        topic2Idx = {}

        for i in range(len(topics)):
            idx2Topic[i] = topics[i]
            topic2Idx[topics[i]] = i
        self.max_label = len(topics) - 1
        self.idx2Topic = idx2Topic
        self.topic2Idx = topic2Idx

        UI_TYPES = ["Text", "Text Button", "Icon", "Card", "Drawer", "Web View", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab",
                    "Background Image", "Image", "Video", "Input", "Number Stepper", "Checkbox", "Radio Button", "Pager Indicator", "On/Off Switch", "Modal", "Slider", "Advertisement", "Date Picker", "Map View"]

        idx2Label = {}
        label2Idx = {}

        for i in range(len(UI_TYPES)):
            idx2Label[i] = UI_TYPES[i]
            label2Idx[UI_TYPES[i]] = i

        self.idx2Label = idx2Label
        self.label2Idx = label2Idx
        self.ui_types = UI_TYPES
        
        self.img_cache = {}
        self.frame_cache = {}
        self.n_labels = len(topics)
        
        for i in trange(len(self.keys)):
            example = self.example_list[self.keys[i]]
            screenId = example['screen_id']
            # image modality
            img_path = os.path.join(self.img_dir, screenId + ".jpg")
            if img_path not in self.img_cache:
                screenImg = Image.open(img_path).convert("RGBA")
                screenImg = screenImg.convert("RGB")
                self.img_cache[img_path] = screenImg
            
            frame_path = os.path.join(
                self.wireframe_dir, screenId + ".png")
            if frame_path not in self.frame_cache:
                screenWireframeImg = Image.open(frame_path).convert("RGBA")
                screenWireframeImg = screenWireframeImg.convert("RGB")
                self.frame_cache[frame_path] = screenWireframeImg


    def __len__(self):
        """Get number of samples in dataset."""
        if self.max_step is None:
            return len(self.keys)
        else:
            return self.max_step

    def featurizeElement(self, element):
        """Convert element into tuple of (bounds, one-hot-label)."""
        bounds, label = element
        labelOneHot = [0 for _ in range(len(self.ui_types))]
        labelOneHot[self.label2Idx[label]] = 1
        return bounds, labelOneHot

    def __getitem__(self, idx):
        """Get item in dataset.

        Args:
            idx (int): Index of data to get.

        Returns:
            list: List of (screen image, screen wireframe image, screen label)
        """
        # print(self.max_step)
        if self.max_step is not None:
            idx = np.random.randint(len(self.keys))

        example = self.example_list[self.keys[idx]]
        screenId = example['screen_id']
        # image modality
        if self.modalities in {'all', 'screenImg'}:
            img_path = os.path.join(self.img_dir, screenId + ".jpg")
            if img_path not in self.img_cache:
                screenImg = Image.open(img_path).convert("RGBA")
                screenImg = screenImg.convert("RGB")
                self.img_cache[img_path] = screenImg
            else:
                screenImg = self.img_cache[img_path]
            # screenImg = self.screenImgList[idx]

            screenImg = self.img_transforms(screenImg)
        # wireframe image modality
        if self.modalities in {'all', 'screenWireframeImg'}:
            frame_path = os.path.join(
                self.wireframe_dir, screenId + ".png")
            if frame_path not in self.frame_cache:
                screenWireframeImg = Image.open(frame_path).convert("RGBA")
                screenWireframeImg = screenWireframeImg.convert("RGB")
                self.frame_cache[frame_path] = screenWireframeImg
            else:
                screenWireframeImg = self.frame_cache[frame_path]
            # screenWireframeImg = self.screenWireframeImgList[idx]
            screenWireframeImg = self.img_transforms(screenWireframeImg)
        # label
        if self.flip_label:
            screenLabel = self.max_label - self.topic2Idx[example['topic']]
        else:
            screenLabel = self.topic2Idx[example['topic']]
        # print(type(screenImg))
        # ret_data = [idx]
        ret_data = {'idx': idx}
        if self.modalities in {'all', 'screenImg'}:
            if not self.hmmt:
                screenImg = screenImg.permute(1, 2, 0)
            if self.cut_into:
                screenImg = einops.rearrange(screenImg, '(h p1) (w p2) c -> p1 p2 (h w c)', p1=16, p2=16)
            # ret_data.append(screenImg)
            ret_data['screenImg'] = screenImg
        if self.modalities in {'all', 'screenWireframeImg'}:
            if not self.hmmt:
                screenWireframeImg = screenWireframeImg.permute(1, 2, 0)
            if self.cut_into:
                screenWireframeImg = einops.rearrange(screenWireframeImg, '(h p1) (w p2) c -> p1 p2 (h w c)', p1=16, p2=16)
            # ret_data.append(screenWireframeImg)
            ret_data['screenWireframeImg'] = screenWireframeImg
        # ret_data.append(screenLabel)
        ret_data['target'] = screenLabel
            
        return ret_data

@dataclass
class EnricoCollect(object):
    def __call__(self, instances: Sequence[Dict]):
        ret_dict = {}
        for item in instances:
            for kk in item:
                if kk not in ret_dict:
                    ret_dict[kk] = []
                ret_dict[kk].append(item[kk] if type(item[kk]) is torch.Tensor else torch.tensor(item[kk]))
        
        for kk in ret_dict:
            ret_dict[kk] = torch.stack(ret_dict[kk])
        return ret_dict

def get_dataset(data_dir, hmmt = False, 
                cut_into = False, x_dim = 128, 
                y_dim=256, flip_label = False,
                max_step = None,
                modalities = "all"):
    ds_train = EnricoDataset(data_dir, mode="train", hmmt = hmmt, 
                             cut_into=cut_into, noise_level=0, 
                             img_dim_x=x_dim, img_dim_y=y_dim, 
                             flip_label=flip_label, max_step=max_step, modalities=modalities)
    ds_val = EnricoDataset(data_dir, mode="val", hmmt = hmmt, cut_into=cut_into, img_dim_x=x_dim, img_dim_y=y_dim, flip_label=flip_label, modalities=modalities)
    ds_test = EnricoDataset(data_dir, mode='test', hmmt = hmmt, cut_into=cut_into, img_dim_x=x_dim, img_dim_y=y_dim, modalities=modalities)
    
    return ds_train, ds_val, ds_test

def get_dataloader(data_dir, batch_size=32, 
                   num_workers=0, hmmt = False, 
                   train_shuffle=True, return_class_weights=True, 
                   cut_into = False, img_noise = False, 
                   wireframe_noise = False, x_dim = 128, 
                   y_dim=256, flip_label = False,
                   max_step = None):
    """Get dataloaders for this dataset.

    Args:
        data_dir (str): Data directory.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 0.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        return_class_weights (bool, optional): Whether to return class weights or not. Defaults to True.

    Returns:
        tuple: Tuple of ((train dataloader, validation dataloader, test dataloader), class_weights) if return_class_weights, otherwise just the dataloaders
    """
    ds_train = EnricoDataset(data_dir, mode="train", hmmt = hmmt, 
                             cut_into=cut_into, img_noise=img_noise, 
                             wireframe_noise=wireframe_noise, noise_level=0.3, 
                             img_dim_x=x_dim, img_dim_y=y_dim, 
                             flip_label=flip_label, max_step=max_step)
    ds_val = EnricoDataset(data_dir, mode="val", hmmt = hmmt, cut_into=cut_into, img_dim_x=x_dim, img_dim_y=y_dim, flip_label=flip_label)

    targets = []
    class_counter = Counter()
    for i in range(len(ds_train.keys)):
        example_topic = ds_train.example_list[ds_train.keys[i]]['topic']
        targets.append(example_topic)
        class_counter[example_topic] += 1

    weights = []
    for t in targets:
        weights.append(1 / class_counter[t])

    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, len(ds_train))

    if return_class_weights:
        class_counter = Counter()
        for i in range(len(ds_train.keys)):
            example_topic = ds_train.example_list[ds_train.keys[i]]['topic']
            class_counter[ds_train.topic2Idx[example_topic]] += 1
        weights = []
        for i in range(len(ds_train.topic2Idx)):
            weights.append(class_counter[i])

        weights2 = []
        for w in weights:
            weights2.append(1 / (w / sum(weights)))
        weights3 = []
        for w in weights2:
            weights3.append(w / sum(weights2))
        weights = weights3

    dl_train = DataLoader(ds_train, num_workers=num_workers,
                          sampler=sampler, batch_size=batch_size, pin_memory=True)
    # dl_train = DataLoader(ds_train, shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    dl_val = DataLoader(ds_val, shuffle=False,
                        num_workers=num_workers, batch_size=batch_size)
    # dl_val = DataLoader(ds_val, num_workers=num_workers, sampler=sampler, batch_size=batch_size)

    ds_test_img = []
    ds_test_wireframe = []
    dl_test = dict()
    # Add image noise
    for i in range(1):
        ds_test_img.append(EnricoDataset(
            data_dir, mode='test', img_noise=False, noise_level=i/10, hmmt = hmmt, cut_into=cut_into, img_dim_x=x_dim, img_dim_y=y_dim))
    dl_test['image'] = [DataLoader(test, shuffle=False, num_workers=num_workers,
                                   batch_size=batch_size) for test in ds_test_img]
    # Add wireframe image noise
    for i in range(1):
        ds_test_wireframe.append(EnricoDataset(
            data_dir, mode='test', wireframe_noise=False, noise_level=i/10, hmmt = hmmt, cut_into=cut_into, img_dim_x=x_dim, img_dim_y=y_dim))
    dl_test['wireframe image'] = [DataLoader(test, shuffle=False, num_workers=num_workers,
                                             batch_size=batch_size) for test in ds_test_wireframe]
    # print(dl_test)
    dls = tuple([dl_train, dl_val, dl_test])
    if return_class_weights:
        return dls, weights
    else:
        return dls
