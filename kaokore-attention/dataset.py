#!/usr/bin/env python3

import csv
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
import numpy as np
# import lycon
import pandas as pd
# import keras as k
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
import wandb


def verify_str_arg(value, valid_values):
    assert value in valid_values
    return value


def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def load_labels(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        return [{
            headers[column_index]: row[column_index]
            for column_index in range(len(row))
        }
            for row in reader]


class Kaokore(Dataset):

    def __init__(self, root, split='train', category='gender', transform=None):
        label = category
        self.root = os.path.expanduser(root)

        self.split = verify_str_arg(split, ['train', 'dev', 'test'])

        self.category = verify_str_arg(category, ['gender', 'status'])

        self.gen_to_cls = {'male': 0, 'female': 1} if label == 'gender' else {'noble': 0, 'warrior': 1,
                                                                                   'incarnation': 2, 'commoner': 3}
        self.cls_to_gen = {v: k for k, v in self.gen_to_cls.items()}

        labels = load_labels(os.path.join(root, 'labels.csv'))
        self.entries = [
            (label_entry['image'], int(label_entry[category]))
            for label_entry in labels
            if label_entry['set'] == split and os.path.exists(
                os.path.join(self.root, 'images_256', label_entry['image']))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        image_filename, label = self.entries[index]

        image_filepath = os.path.join(self.root, 'images_256', image_filename)
        image = image_loader(image_filepath)
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def gen_val_transforms(args):
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((args.image_size, args.image_size)),
                               transforms.Normalize([1., 1., 1.], [0.5, 0.5, 0.5])])


def gen_train_transforms(args):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
        transforms.RandomApply(transforms=[
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        ], p=0.5),
        transforms.Normalize([1., 1., 1.], [0.5, 0.5, 0.5])

    ]

    )


