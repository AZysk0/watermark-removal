import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch, time, os, pickle
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data_utils.dataloader import wm_dataloader
from data_utils.dataset import WatermarkSimpleDataset


# ========= Helper modules ==========

class BasicResBlock(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, inplanes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.match_identity = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.match_identity(identity)
        out += identity
        out = self.relu(out)
        return out


# ======= ResNet-based GAN ==========

class ResNetRemoval(nn.Module):
    '''watermark removal task, primitive pix2pix model'''
    def __init__(self, img_channels):
        super(ResNetRemoval, self).__init__()
        self.dataset = None
        self.resblock_1 = BasicResBlock(inplanes=img_channels, planes=16)
        self.resblock_2 = BasicResBlock(inplanes=16, planes=3)
        
        self._init_dataset()
        self._init_train_params()
    
    def train(self, epochs=3):
        trainloader = DataLoader(self.dataset, batch_size=4)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        for epoch in range(epochs):
            running_loss = 0.0
            with tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                for i, data in enumerate(pbar):
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    
                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs = self.forward(x)
                    loss = self.criterion(outputs, y)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    pbar.set_postfix(loss=running_loss / (i + 1)) 
                    # pbar.update()
        
            print('Loss: {}'.format(running_loss))
        
        print('Finished Training')
        
        return

    def forward(self, x):
        out = self.resblock_1(x)
        out = self.resblock_2(out)
        return out
        
    def _load_checkpoint(self, path):
        checkpoint = torch.load(path)
    
        # Check if the checkpoint contains a model_state_dict and optimizer_state_dict
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
        
        # Match the keys in the checkpoint to the model's state_dict
        state_dict = self.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if k in state_dict}

        # Load the filtered state_dict into the model
        self.load_state_dict(model_state_dict, strict=False)
    
    def _save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def _init_dataset(self):
        transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.dataset = WatermarkSimpleDataset(transform=transform)
        print('Dataset has been initialized successfully')

    def _init_train_params(self):
        self.criterion = nn.MSELoss()  # primitive loss func
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        print('Loss function and optimizer have been initialized successfully')


class ResNetGenerator(nn.Module):
    """Some Information about ResNetGenerator"""
    def __init__(self):
        super(ResNetGenerator, self).__init__()

    def forward(self, x):

        return x


class ResNetDiscriminator(nn.Module):
    """
    
    """
    def __init__(self):
        super(ResNetDiscriminator, self).__init__()

    def forward(self, x):

        return x


class ResNetGAN():
    """Some Information about ResNetGAN"""
    def __init__(self):
        self.generator = ResNetGenerator()
        self.discriminator = ResNetDiscriminator()

    def train(self):
        raise NotImplementedError


# ========== Attention GAN ================

class AttentionGenerator(nn.Module):
    """Some Information about Generator"""
    def __init__(self):
        super(AttentionGenerator, self).__init__()
        raise NotImplementedError

    def forward(self, x):

        return x


class AttentionDiscriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self):
        super(AttentionDiscriminator, self).__init__()
        raise NotImplementedError

    def forward(self, x):

        return x


class AttentiveGAN(nn.Module):
    """Some Information about AttentiveGAN"""
    def __init__(self):
        super(AttentiveGAN, self).__init__()
        raise NotImplementedError

    def forward(self, x):

        return x


