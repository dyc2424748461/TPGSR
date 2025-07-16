#!/usr/bin/env python3
"""
TPGSR Google Colab Setup Script
This script sets up the TPGSR repository in Google Colab environment
"""

import os
import sys
import subprocess
import yaml
import gdown
import torch
from PIL import Image, ImageDraw, ImageFont
import re

def run_command(cmd, description=""):
    """Run a shell command and handle errors"""
    print(f"Running: {description}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(f"Success: {description}")
    return True

def check_environment():
    """Check if running in Colab and GPU availability"""
    print("=== Environment Check ===")
    
    # Check if in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
        in_colab = True
    except ImportError:
        print("✗ Not running in Google Colab")
        in_colab = False
    
    # Check GPU
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    return in_colab, cuda_available

def install_dependencies():
    """Install required dependencies"""
    print("\n=== Installing Dependencies ===")
    
    dependencies = [
        "torch==1.2.0 torchvision==0.4.0",
        "numpy==1.18.0",
        "Pillow==6.2.2",
        "lmdb easydict pyfasttext editdistance tensorboardX",
        "pyyaml scipy matplotlib tqdm opencv-python",
        "IPython gdown"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    return True

def download_dataset():
    """Download TextZoom dataset"""
    print("\n=== Downloading TextZoom Dataset ===")
    
    os.makedirs('data', exist_ok=True)
    
    # TextZoom dataset Google Drive ID (you may need to update this)
    dataset_id = "1WKVhB2qFjqQUqy8KVqtgEQZCsn2hZ8kV"
    
    try:
        print("Downloading TextZoom dataset...")
        gdown.download(f'https://drive.google.com/uc?id={dataset_id}', 
                      'data/TextZoom.zip', quiet=False)
        
        print("Extracting dataset...")
        run_command("cd data && unzip -q TextZoom.zip", "Extracting TextZoom")
        
        print("✓ TextZoom dataset downloaded and extracted")
        return True
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        return False

def download_pretrained_models():
    """Download pretrained recognizer models"""
    print("\n=== Downloading Pretrained Models ===")
    
    os.makedirs('pretrained', exist_ok=True)
    
    models = {
        'aster.pth.tar': '1sOqiX9cqOgXV0qbMHTwl5eSV_5_d1gwc',
        'moran.pth': '1YLDHhtc5EyRNyhvNQS6ywC9htkdT4c7q',
        'crnn.pth': '1ooaHefQp0wDATLvOZlsXyLCjWiHSHKX'
    }
    
    for model_name, file_id in models.items():
        try:
            print(f"Downloading {model_name}...")
            gdown.download(f'https://drive.google.com/uc?id={file_id}', 
                          f'pretrained/{model_name}', quiet=False)
            print(f"✓ {model_name} downloaded")
        except Exception as e:
            print(f"✗ Error downloading {model_name}: {e}")
            return False
    
    return True

def update_configuration(cuda_available):
    """Update configuration for Colab environment"""
    print("\n=== Updating Configuration ===")
    
    try:
        # Load configuration
        with open('config/super_resolution.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
        
        # Update paths for Colab
        base_path = '/content/TPGSR' if 'google.colab' in sys.modules else os.getcwd()
        
        config['TRAIN']['train_data_dir'] = [
            f'{base_path}/data/TextZoom/train1',
            f'{base_path}/data/TextZoom/train2'
        ]
        
        config['TRAIN']['VAL']['val_data_dir'] = [
            f'{base_path}/data/TextZoom/test/easy',
            f'{base_path}/data/TextZoom/test/medium',
            f'{base_path}/data/TextZoom/test/hard'
        ]
        
        config['TRAIN']['VAL']['rec_pretrained'] = f'{base_path}/pretrained/aster.pth.tar'
        config['TRAIN']['VAL']['moran_pretrained'] = f'{base_path}/pretrained/moran.pth'
        config['TRAIN']['VAL']['crnn_pretrained'] = f'{base_path}/pretrained/crnn.pth'
        
        # Adjust for GPU/CPU
        if cuda_available:
            config['TRAIN']['cuda'] = True
            config['TRAIN']['batch_size'] = 16
        else:
            config['TRAIN']['cuda'] = False
            config['TRAIN']['batch_size'] = 4
        
        config['TRAIN']['workers'] = 2
        config['TRAIN']['epochs'] = 100
        
        # Save updated configuration
        with open('config/super_resolution.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Configuration updated (CUDA: {cuda_available}, batch_size: {config['TRAIN']['batch_size']})")
        return True
        
    except Exception as e:
        print(f"✗ Error updating configuration: {e}")
        return False

def fix_code_compatibility(cuda_available):
    """Fix code for compatibility issues"""
    print("\n=== Fixing Code Compatibility ===")
    
    try:
        # Fix ptflops imports
        for file_path in ['interfaces/base.py', 'interfaces/super_resolution.py']:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Comment out ptflops related lines
            content = re.sub(r'^(.*ptflops.*)$', r'# \1', content, flags=re.MULTILINE)
            content = re.sub(r'^(.*get_model_complexity_info.*)$', r'# \1', content, flags=re.MULTILINE)
            
            with open(file_path, 'w') as f:
                f.write(content)
        
        # Fix model loading for CPU/GPU compatibility
        with open('interfaces/base.py', 'r') as f:
            content = f.read()
        
        # Add map_location for torch.load calls
        device_str = "torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
        content = re.sub(
            r'torch\.load\(([^)]+)\)',
            f'torch.load(\\1, map_location={device_str})',
            content
        )
        
        # Fix MORAN initialization
        if cuda_available:
            content = re.sub(
                r"inputDataType='torch\.FloatTensor', CUDA=False",
                "inputDataType='torch.cuda.FloatTensor', CUDA=True",
                content
            )
        else:
            content = re.sub(
                r"inputDataType='torch\.cuda\.FloatTensor', CUDA=True",
                "inputDataType='torch.FloatTensor', CUDA=False",
                content
            )
        
        with open('interfaces/base.py', 'w') as f:
            f.write(content)
        
        print("✓ Code compatibility fixes applied")
        return True
        
    except Exception as e:
        print(f"✗ Error fixing code compatibility: {e}")
        return False

def create_demo():
    """Create demo files and test image"""
    print("\n=== Creating Demo ===")
    
    try:
        os.makedirs('demo', exist_ok=True)
        os.makedirs('demo_results', exist_ok=True)
        
        # Create a simple test image with text
        img = Image.new('RGB', (128, 32), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 5), "HELLO", fill='black', font=font)
        img.save('demo/test.png')
        
        # Create demo script
        demo_script = '''
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import yaml
from easydict import EasyDict

# Load configuration
config_path = os.path.join('config', 'super_resolution.yaml')
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
transformed_img.save('demo_results/input.png')

print("Demo preprocessing completed successfully!")
print("Input image saved to demo_results/input.png")
'''
        
        with open('run_demo.py', 'w') as f:
            f.write(demo_script)
        
        print("✓ Demo files created")
        return True
        
    except Exception as e:
        print(f"✗ Error creating demo: {e}")
        return False

def main():
    """Main setup function"""
    print("TPGSR Google Colab Setup")
    print("=" * 50)
    
    # Check environment
    in_colab, cuda_available = check_environment()
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies")
        return False
    
    # Download dataset
    if not download_dataset():
        print("Failed to download dataset")
        return False
    
    # Download pretrained models
    if not download_pretrained_models():
        print("Failed to download pretrained models")
        return False
    
    # Update configuration
    if not update_configuration(cuda_available):
        print("Failed to update configuration")
        return False
    
    # Fix code compatibility
    if not fix_code_compatibility(cuda_available):
        print("Failed to fix code compatibility")
        return False
    
    # Create demo
    if not create_demo():
        print("Failed to create demo")
        return False
    
    print("\n" + "=" * 50)
    print("✓ TPGSR setup completed successfully!")
    print("\nNext steps:")
    print("1. Run demo: python run_demo.py")
    print("2. Start training: python main.py --batch_size=16 --STN --mask --gradient --sr_share --use_distill --without_colorjitter --test_model=TSRN")
    print("3. Check demo_results/ folder for outputs")
    
    return True

if __name__ == "__main__":
    main()