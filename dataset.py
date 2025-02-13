import os
from concurrent.futures import ThreadPoolExecutor
import requests
from io import BytesIO
import numpy as np
import cv2
from collections import Counter
from ast import literal_eval
import itertools
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==============


# ==== Some useless utils ==========

def get_filepaths_recursive(dir):
    import itertools
    return list(itertools.chain.from_iterable(
        [os.path.join(root, file) for file in files]
        for root, _, files in os.walk(dir)
    ))


def get_filenames_recursive(dir):
    import itertools
    return list(itertools.chain.from_iterable(
        files
        for root, _, files in os.walk(dir)
    ))


def filter_by_extension(filenames, ext):
    return [f for f in filenames if f.endswith(ext)]


def parallelize(func, inputs, n_workers=4):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(func, inputs))
    
    return results


def copy_images_to(paths, out_dir, n_workers=4):
    from functools import partial
    os.makedirs(out_dir, exist_ok=True)
    
    def copy_img(args):
        try:
            index, image_path = args
            ext = image_path.split('.')[-1]
            dst = os.path.join(out_dir, f'image_{index}.{ext}')
            shutil.copy(image_path, dst)
        except FileNotFoundError:
            print(f"ERROR: image_path: {image_path}, extension: {ext}. Skipping this image...")

    index_image_pairs = zip(itertools.count(start=0, step=1), paths)
    parallelize(copy_img, index_image_pairs, n_workers=n_workers)



# =====================================
class WatermarkDataset(torch.utils.data.Dataset):
    """Some Information about WatermarkDataset"""
    def __init__(self):
        super(WatermarkDataset, self).__init__()

    def __getitem__(self, index):
        return 

    def __len__(self):
        return


image_dir = 'data/images'
# print(Counter([filename.split('.')[1] for filename in get_filenames_recursive(image_dir)]))
# print(get_filenames_recursive(image_dir)[:10])
image_paths = get_filepaths_recursive(image_dir)
# print(*image_paths[:10], sep='\n')

copy_images_to(image_paths, 'data/images_', n_workers=4)



