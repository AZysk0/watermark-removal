from data_utils.dataset import WatermarkDataset
import torch
from torchvision import transforms

def wm_dataloader(batch_size=4, shuffle=False):
    clean_images_dir = 'data/upscaled'
    watermark_images_dir = 'data/watermark_upscaled'
    segmented_watermarks_dir = 'data/segmented'
    
    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = WatermarkDataset(
        clean_images_dir, 
        watermark_images_dir, 
        segmented_watermarks_dir, 
        n=1000, 
        transform=transforms_
    )
    dataloader = torch.utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # dataloader getitem => clean, watermark, wm-mask
    return dataloader





