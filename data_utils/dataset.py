import os
from concurrent.futures import ThreadPoolExecutor
import requests
from io import BytesIO
import numpy as np
import cv2
import itertools
import shutil
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==============


# ==== Some useless utils ==========
def read_file(path):
    with open(path, 'r') as f:
        return f.readlines()


def save_iterable(path, _iter):
    with open(path, 'w') as f:
        for line in _iter:
            f.write(f'{line}\n')
    print(f'Saved successfully to {path}')


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


def resize_and_save_images(paths, out_dir, n_workers=4):
    from functools import partial
    os.makedirs(out_dir, exist_ok=True)
    
    def copy_img(args):
        try:
            index, image_path = args
            ext = image_path.split('.')[-1]
            dst = os.path.join(out_dir, f'image_{index}.{ext}')
            # shutil.copy(image_path, dst)
            img = cv2.imread(image_path)
            resized = cv2.resize(img, dsize=(600, 600), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(dst, resized)        
        except Exception as e:
            # print(f"ERROR: image_path: {image_path}, extension: {ext}. Skipping this image...")
            print(e)
    
    index_image_pairs = zip(itertools.count(start=0, step=1), paths)
    parallelize(copy_img, index_image_pairs, n_workers=n_workers)


def sort_by_index(_strs):
    import re
    def str_index(_str):
        match = re.search(r'_(\d+)', _str)
        return int(match.group(1)) if match else float('inf')
    
    return sorted(_strs, key=str_index)


def get_image_paths(dir, n=None):
    n = len(os.listdir(dir)) if n is None else n
    valid_ext = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    paths = [os.path.join(dir, filename) for filename in sort_by_index(os.listdir(dir))[:n]
             if any(filename.lower().endswith(ext) for ext in valid_ext)]
    return paths

# =====================================

class WatermarkDataset(Dataset):
    
    def __init__(self, clean_dir, wm_dir, segment_dir, n=None, transform=None):
        super(WatermarkDataset, self).__init__()
        self.clean_dir = clean_dir
        self.wm_dir = wm_dir
        self.segment_dir = segment_dir
        self.clean_images_path = get_image_paths(self.clean_dir, n=n)
        self.wm_images_path = get_image_paths(self.wm_dir, n=n)
        self.segment_images_path = get_image_paths(self.segment_dir, n=n)
        self.transform = transform

    def __getitem__(self, index):
        clean_img = cv2.imread(self.clean_images_path[index], cv2.IMREAD_COLOR)
        wm_img = cv2.imread(self.wm_images_path[index], cv2.IMREAD_COLOR)
        segment_img = cv2.imread(self.segment_images_path[index], cv2.IMREAD_GRAYSCALE)

        if clean_img is None or wm_img is None or segment_img is None:
            raise RuntimeError(f"Error loading image at index {index}")

        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        wm_img = cv2.cvtColor(wm_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            clean_img = self.transform(clean_img)
            wm_img = self.transform(wm_img)
            segment_img = transforms.ToTensor()(segment_img)

        return clean_img, wm_img, segment_img

    def __len__(self):
        return min(len(self.clean_images_path), len(self.segment_images_path), len(self.wm_images_path))


class WatermarkSimpleDataset(Dataset):
    """dataset for primitive pix2pix model"""
    def __init__(self, transform=None):
        super(WatermarkSimpleDataset, self).__init__()
        self.clean_dir = 'data/upscaled'
        self.wm_dir = 'data/watermark_upscaled'
        self.clean_images_path = get_image_paths(self.clean_dir)
        self.wm_images_path = get_image_paths(self.wm_dir)
        self.transform = transform

    def __getitem__(self, index):
        clean_img = cv2.imread(self.clean_images_path[index], cv2.IMREAD_COLOR)
        wm_img = cv2.imread(self.wm_images_path[index], cv2.IMREAD_COLOR)

        if clean_img is None or wm_img is None:
            raise RuntimeError(f"Error loading image at index {index}")

        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        wm_img = cv2.cvtColor(wm_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            clean_img = self.transform(clean_img)
            wm_img = self.transform(wm_img)

        return clean_img, wm_img

    def __len__(self):
        return min(len(self.clean_images_path), len(self.wm_images_path))



# ===================================
watermarks = [
    "CONFIDENTIAL", "SAMPLE", "DRAFT", "PRIVATE", "RESTRICTED", 
    "DO NOT COPY", "FOR INTERNAL USE ONLY", "CLASSIFIED", "COPYRIGHT", 
    "UNAUTHORIZED REPRODUCTION PROHIBITED", "COMPANY NAME CONFIDENTIAL", 
    "INTERNAL DOCUMENT", "COMPANY SECRETS", "PROPERTY OF [YOUR COMPANY]", 
    "BUSINESS CONFIDENTIAL", "PROPRIETARY", "TOP SECRET", "SENSITIVE INFORMATION", 
    "OFFICIAL DOCUMENT", "NOT FOR DISTRIBUTION", "READ-ONLY", "LIMITED ACCESS", 
    "BETA VERSION", "FINAL VERSION", "PROTECTED DOCUMENT", "VERIFIED COPY", 
    "SCANNED COPY", "DO NOT PRINT", "PREVIEW ONLY", "WATERMARKED DOCUMENT", 
    "LEGAL DOCUMENT", "ATTORNEY-CLIENT PRIVILEGED", "NOT FOR RESALE", 
    "CONTROLLED COPY", "CLIENT CONFIDENTIAL", "FINANCIAL REPORT", 
    "BOARD OF DIRECTORS ONLY", "INVESTOR CONFIDENTIAL", "UNDER NDA", 
    "PERSONAL & CONFIDENTIAL", "EMPLOYEE CONFIDENTIAL", "GOVERNMENT DOCUMENT", 
    "FOR RESEARCH PURPOSES", "TRIAL VERSION", "EVALUATION COPY", "SECURITY PROTECTED", 
    "LICENSED COPY", "UNAUTHORIZED USE PROHIBITED", "PERSONAL COPY", 
    "CUSTOMER CONFIDENTIAL", "UNAPPROVED DRAFT", "PATENT PENDING", 
    "RESTRICTED DISTRIBUTION", "FINAL DRAFT", "FOR REVIEW ONLY", 
    "FOR DEMONSTRATION PURPOSES", "INTERNAL MEMO", "FOR TRAINING USE ONLY", 
    "VERIFIED AUTHENTIC", "COMPANY POLICY DOCUMENT", "EVIDENCE COPY", 
    "OFFICIAL USE ONLY", "AUDIT COPY", "MEDICAL RECORD", "SECURE FILE", 
    "STRICTLY PRIVATE", "NO REPRODUCTION ALLOWED", "FOR APPROVAL ONLY", 
    "UNDER REVISION", "PENDING REVIEW", "INTERNAL USE", "SENSITIVE DATA", 
    "TRAINING MATERIAL", "FOR CEO REVIEW", "CONTRACT COPY", "MASTER COPY", 
    "NO PUBLIC SHARING", "WATERMARK TEST", "NON-DISCLOSURE AGREEMENT", 
    "LEGAL COMPLIANCE", "BOARD MEETING NOTES", "PRIVATE REPORT", "FOR EDITING ONLY", 
    "CONFIDENTIAL WORK-IN-PROGRESS", "DRAFT PROPOSAL", "STRATEGIC PLAN", 
    "INTERNAL ANALYSIS", "FINANCIAL STATEMENT", "FOR AUDITOR REVIEW", 
    "NON-EDITABLE COPY", "INTELLECTUAL PROPERTY", "DO NOT FORWARD", 
    "PROTECTED CONTENT", "LEGAL COMPLIANCE ONLY", "COMPANY CONFIDENTIAL", 
    "SECURITY CLASSIFIED", "PRIVILEGED & CONFIDENTIAL", "UNDER INVESTIGATION", 
    "SUBJECT TO CHANGE", "EXCLUSIVE CONTENT", "PATENT CONFIDENTIAL", 
    "DO NOT CIRCULATE", "MANAGEMENT EYES ONLY"
]


image_dir = 'data/upscaled'
watermark_dir = 'data/watermark_upscaled'




