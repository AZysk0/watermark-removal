import os
from models import ResNetRemoval
import cv2
import numpy as np
from torchvision import transforms
import torch


def simple_inference(image):
    weights_path = 'checkpoints/simple_resnet.pth'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    resnet_model = ResNetRemoval(img_channels=3).to(device=device)
    resnet_model._load_checkpoint(weights_path)

    out_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((600, 600)),   # model trained on images of this size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    out_image = transform(out_image).unsqueeze(0).to(device)
    
    resnet_model.eval()
    with torch.no_grad():
        out_image = resnet_model(out_image)

    out_image = out_image.squeeze(0).cpu().clamp(0, 1).numpy()
    out_image = np.transpose(out_image, (1, 2, 0))
    out_image = np.clip((out_image * 255).astype(np.uint8), 0, 255)
    out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
    out_image = cv2.resize(out_image, dsize=(300, 300), interpolation=cv2.INTER_AREA)
    # out_image = cv2.resize(out_image, dsize=image.shape[:2], interpolation=cv2.INTER_LANCZOS4)

    return out_image


wm_dir = 'data/watermark_upscaled'
wm_paths = [os.path.join(wm_dir, f) for f in os.listdir(wm_dir) if f.endswith('.png') or f.endswith('.jpg')]

n = 111
test_wm_image_path = wm_paths[n]

test_image = cv2.imread(test_wm_image_path)
out_image = simple_inference(test_image)

cv2.imshow('Restored Image', out_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




