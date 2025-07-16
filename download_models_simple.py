#!/usr/bin/env python3
"""
Simple model download script with direct links
"""

import os
import requests
import subprocess
from tqdm import tqdm

def download_file(url, filename, output_dir):
    """Download file with progress bar"""
    output_path = os.path.join(output_dir, filename)
    
    # Skip if file already exists
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10 * 1024 * 1024:  # > 10MB
        print(f"✓ {filename} already exists")
        return True
    
    print(f"Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ {filename} downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def download_with_command(cmd, filename):
    """Download using command line tools"""
    try:
        print(f"Downloading {filename} using command line...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {filename} downloaded successfully")
            return True
        else:
            print(f"✗ Command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Command failed: {e}")
        return False

def main():
    """Main download function"""
    print("TPGSR Model Downloader (Simple Version)")
    print("=" * 50)
    
    # Create pretrained directory
    pretrained_dir = 'pretrained'
    os.makedirs(pretrained_dir, exist_ok=True)
    
    # Try multiple download methods for each model
    models = [
        {
            'filename': 'aster.pth.tar',
            'methods': [
                # Method 1: Direct download links (if available)
                lambda: download_file('https://github.com/ayumiymk/aster.pytorch/releases/download/v1.0/aster.pth.tar', 'aster.pth.tar', pretrained_dir),
                # Method 2: wget with Google Drive
                lambda: download_with_command('wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1sOqiX9cqOgXV0qbMHTwl5eSV_5_d1gwc\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id=1sOqiX9cqOgXV0qbMHTwl5eSV_5_d1gwc" -O pretrained/aster.pth.tar && rm -rf /tmp/cookies.txt', 'aster.pth.tar'),
                # Method 3: curl
                lambda: download_with_command('curl -L "https://drive.google.com/uc?export=download&id=1sOqiX9cqOgXV0qbMHTwl5eSV_5_d1gwc" -o pretrained/aster.pth.tar', 'aster.pth.tar'),
            ]
        },
        {
            'filename': 'moran.pth',
            'methods': [
                lambda: download_file('https://github.com/Canjie-Luo/MORAN_v2/releases/download/v1.0/moran.pth', 'moran.pth', pretrained_dir),
                lambda: download_with_command('wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1YLDHhtc5EyRNyhvNQS6ywC9htkdT4c7q\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id=1YLDHhtc5EyRNyhvNQS6ywC9htkdT4c7q" -O pretrained/moran.pth && rm -rf /tmp/cookies.txt', 'moran.pth'),
                lambda: download_with_command('curl -L "https://drive.google.com/uc?export=download&id=1YLDHhtc5EyRNyhvNQS6ywC9htkdT4c7q" -o pretrained/moran.pth', 'moran.pth'),
            ]
        },
        {
            'filename': 'crnn.pth',
            'methods': [
                lambda: download_file('https://github.com/meijieru/crnn.pytorch/releases/download/v1.0/crnn.pth', 'crnn.pth', pretrained_dir),
                lambda: download_with_command('wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1ooaHefQp0wDATLvOZlsXyLCjWiHSHKX\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id=1ooaHefQp0wDATLvOZlsXyLCjWiHSHKX" -O pretrained/crnn.pth && rm -rf /tmp/cookies.txt', 'crnn.pth'),
                lambda: download_with_command('curl -L "https://drive.google.com/uc?export=download&id=1ooaHefQp0wDATLvOZlsXyLCjWiHSHKX" -o pretrained/crnn.pth', 'crnn.pth'),
            ]
        }
    ]
    
    success_count = 0
    
    for model in models:
        filename = model['filename']
        methods = model['methods']
        
        print(f"\n=== Downloading {filename} ===")
        
        success = False
        for i, method in enumerate(methods, 1):
            print(f"Attempt {i}/{len(methods)}...")
            try:
                if method():
                    success = True
                    break
            except Exception as e:
                print(f"Method {i} failed: {e}")
        
        if success:
            success_count += 1
        else:
            print(f"✗ All methods failed for {filename}")
    
    print("\n" + "=" * 50)
    print(f"Download Summary: {success_count}/{len(models)} models downloaded")
    
    if success_count < len(models):
        print("\nManual Download Instructions:")
        print("If automatic download fails, please manually download from:")
        print("1. ASTER: https://github.com/ayumiymk/aster.pytorch")
        print("2. MORAN: https://github.com/Canjie-Luo/MORAN_v2") 
        print("3. CRNN: https://github.com/meijieru/crnn.pytorch")
        print("\nPlace the downloaded files in the 'pretrained/' directory")

if __name__ == "__main__":
    main()