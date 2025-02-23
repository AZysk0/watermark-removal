import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from models import ResNetRemoval


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

simple_resnet_wm_removal = ResNetRemoval(img_channels=3).to(device=device)
# simple_resnet_wm_removal.train(epochs=3)

checkpoint_save_path = 'checkpoints/simple_resnet.pth'
# simple_resnet_wm_removal._save_checkpoint(checkpoint_save_path)
simple_resnet_wm_removal._load_checkpoint(checkpoint_save_path)

wm_dir = 'data/watermark_upscaled'
clean_dir = 'data/upscaled'

wm_paths = [os.path.join(wm_dir, f) for f in os.listdir(wm_dir) if f.endswith('.png') or f.endswith('.jpg')]
clean_paths = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.png') or f.endswith('.jpg')]

n = 111
test_wm_image_path = wm_paths[n]
test_clean_image_path = clean_paths[n]

test_wm_image = cv2.imread(test_wm_image_path)
test_clean_image = cv2.imread(test_clean_image_path)

test_wm_image = cv2.cvtColor(test_wm_image, cv2.COLOR_BGR2RGB)
test_clean_image = cv2.cvtColor(test_clean_image, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_wm_tensor = transform(test_wm_image).unsqueeze(0).to(device)
test_clean_tensor = transform(test_clean_image).unsqueeze(0).to(device)

simple_resnet_wm_removal.eval()
with torch.no_grad():
    out = simple_resnet_wm_removal(test_wm_tensor)

out_image = out.squeeze(0).cpu().clamp(0, 1).numpy()
out_image = np.transpose(out_image, (1, 2, 0))
out_image = np.clip((out_image * 255).astype(np.uint8), 0, 255)
out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
out_image = cv2.resize(out_image, dsize=(125, 125), interpolation=cv2.INTER_AREA)

cv2.imshow('Restored Image', out_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

