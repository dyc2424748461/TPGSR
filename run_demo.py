import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import argparse
from easydict import EasyDict
import yaml

# Load configuration
config_path = os.path.join('config', 'super_resolution.yaml')
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)

# Set device to CPU
device = torch.device('cpu')

# Create a simple transform
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
])

# Load a test image
img_path = 'demo/test.png'
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0)

print(f"Loaded image from {img_path}")
print(f"Image tensor shape: {img_tensor.shape}")

# Save the transformed image
transformed_img = transforms.ToPILImage()(img_tensor.squeeze(0))
transformed_img.save('demo_results/transformed.png')

print("Saved transformed image to demo_results/transformed.png")
print("Demo completed successfully!")