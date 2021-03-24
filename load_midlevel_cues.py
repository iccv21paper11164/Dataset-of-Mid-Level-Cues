import os
import argparse
import numpy as np
import random
import pickle
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler

from einops import rearrange
import einops as ein
import torch 
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data.dataset_of_midlevel_cues import MidLevelCuesDataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_size', type=int, default=512,
        help='Input image size. (default: 512)')
    parser.add_argument(
        '--normalize_rgb', type=bool, default=False,
        help='Normalize the rgb images. (default: False)')    
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size for data loader (default: 4)')
    parser.add_argument(
        '--num_workers', type=int, default=16,
        help='Number of workers for DataLoader. (default: 16)')
    parser.add_argument(
        '--data_path', type=str, default='dataset/',
        help='Root directory of the dataset (default: dataset/)')
    parser.add_argument(
        "--midlevel_cues", nargs="+", default=["rgb", "normal"],
        help='The mid-level cues loaded in each batch of data (default: [rgb, normal])')

    args = parser.parse_args()

    tasks = args.midlevel_cues

    opt_dataset = MidLevelCuesDataset.Options(
        tasks=tasks,
        buildings=['apartment_2'],
        data_path=args.data_path,
        transform='DEFAULT',
        image_size=args.image_size,
        normalize_rgb=args.normalize_rgb,
        randomize_views=True
    )
        
    dataset = MidLevelCuesDataset(options=opt_dataset)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
    )

    print("Loading one batch from the dataset...")
    batch = next(iter(dataloader))
    print("Loaded points : ", batch['point'])
    print("MidLevel Cues : ")
    for task in tasks:
        midlevel_cue = batch[task]
        print(f'{task} : ', midlevel_cue.shape)
