import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data_utils.dataloader import wm_dataloader


class DecompNet(nn.Module):
    """Some Information about DecompNet"""
    def __init__(self):
        super(DecompNet, self).__init__()

    def forward(self, x):

        return x


class Discriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self, x):

        return x


class Generator(nn.Module):
    """Some Information about Generator"""
    def __init__(self):
        super(Generator, self).__init__()

    def forward(self, x):

        return x



