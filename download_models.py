#!/usr/bin/env python3
"""
Enhanced model download script with multiple methods
"""

import os
import sys
import requests
import subprocess
from pathlib import Path
import time

def download_with_gdown(file_id, output_path, filename):
    """Download using gdown library"""
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        print(f"Downloading {filename} using gdown...")
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"gdown failed: {e}")
        return False

def download_with_wget(file_id, output_path, filename):
    """Download using wget command"""
    try:
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        cmd = f'wget --no-check-certificate "{url}" -O "{output_path}"'
        print(f"Downloading {filename} using wget...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"wget failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"wget failed: {e}")
        return False

def download_with_curl(file_id, output_path, filename):
    """Download using curl command"""
    try:
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        cmd = f'curl -L "{url}" -o "{output_path}"'
        print(f"Downloading {filename} using curl...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"curl failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"curl failed: {e}")
        return False

def download_with_requests(file_id, output_path, filename):
    """Download using requests library with session"""
    try:
        print(f"Downloading {filename} using requests...")
        
        # First request to get the download warning page
        session = requests.Session()
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        response = session.get(url, stream=True)
        
        # Look for download warning and get the confirm token
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                confirm_url = f'https://drive.google.com/uc?export=download&confirm={value}&id={file_id}'
                response = session.get(confirm_url, stream=True)
                break
        
        # Download the file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return True
    except Exception as e:
        print(f"requests failed: {e}")
        return False

def download_from_alternative_sources(filename, output_path):
    """Try alternative download sources"""
    alternative_urls = {
        'aster.pth.tar': [
            'https://github.com/ayumiymk/aster.pytorch/releases/download/v1.0/aster.pth.tar',
            'https://huggingface.co/spaces/akhaliq/ASTER/resolve/main/aster.pth.tar'
        ],
        'moran.pth': [
            'https://github.com/Canjie-Luo/MORAN_v2/releases/download/v1.0/moran.pth'
        ],
        'crnn.pth': [
            'https://github.com/meijieru/crnn.pytorch/releases/download/v1.0/crnn.pth'
        ]
    }
    
    if filename in alternative_urls:
        for url in alternative_urls[filename]:
            try:
                print(f"Trying alternative source: {url}")
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    return True
            except Exception as e:
                print(f"Alternative source failed: {e}")
                continue
    
    return False

def verify_file(filepath, min_size_mb=10):
    """Verify downloaded file"""
    if not os.path.exists(filepath):
        return False
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb < min_size_mb:
        print(f"File {filepath} is too small ({size_mb:.1f}MB), probably corrupted")
        return False
    
    print(f"File {filepath} verified ({size_mb:.1f}MB)")
    return True

def download_model(file_id, filename, output_dir):
    """Download a model using multiple methods"""
    output_path = os.path.join(output_dir, filename)
    
    # Skip if file already exists and is valid
    if verify_file(output_path):
        print(f"✓ {filename} already exists and is valid")
        return True
    
    print(f"\n=== Downloading {filename} ===")
    
    # Try multiple download methods
    methods = [
        lambda: download_with_gdown(file_id, output_path, filename),
        lambda: download_with_wget(file_id, output_path, filename),
        lambda: download_with_curl(file_id, output_path, filename),
        lambda: download_with_requests(file_id, output_path, filename),
        lambda: download_from_alternative_sources(filename, output_path)
    ]
    
    for i, method in enumerate(methods, 1):
        try:
            print(f"Attempt {i}/5...")
            if method():
                if verify_file(output_path):
                    print(f"✓ {filename} downloaded successfully")
                    return True
                else:
                    print(f"Downloaded file is invalid, trying next method...")
                    if os.path.exists(output_path):
                        os.remove(output_path)
        except Exception as e:
            print(f"Method {i} failed: {e}")
        
        time.sleep(2)  # Wait between attempts
    
    print(f"✗ Failed to download {filename}")
    return False

def main():
    """Main download function"""
    print("TPGSR Model Downloader")
    print("=" * 50)
    
    # Create pretrained directory
    pretrained_dir = 'pretrained'
    os.makedirs(pretrained_dir, exist_ok=True)
    
    # Model information with multiple possible file IDs
    models = {
        'aster.pth.tar': [
            '1sOqiX9cqOgXV0qbMHTwl5eSV_5_d1gwc',
            '1KKahTJDFVbJhTbMBbTFDPNzKCnsjvjpM',
            '1-5JW3wTRkOw7h4_qVqKJqRdNgqYTSCvt'
        ],
        'moran.pth': [
            '1YLDHhtc5EyRNyhvNQS6ywC9htkdT4c7q',
            '1KKahTJDFVbJhTbMBbTFDPNzKCnsjvjpM',
            '1-5JW3wTRkOw7h4_qVqKJqRdNgqYTSCvt'
        ],
        'crnn.pth': [
            '1ooaHefQp0wDATLvOZlsXyLCjWiHSHKX',
            '1KKahTJDFVbJhTbMBbTFDPNzKCnsjvjpM',
            '1-5JW3wTRkOw7h4_qVqKJqRdNgqYTSCvt'
        ]
    }
    
    success_count = 0
    total_count = len(models)
    
    for filename, file_ids in models.items():
        success = False
        for file_id in file_ids:
            if download_model(file_id, filename, pretrained_dir):
                success = True
                break
        
        if success:
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Download Summary: {success_count}/{total_count} models downloaded successfully")
    
    if success_count == total_count:
        print("✓ All models downloaded successfully!")
        return True
    else:
        print("✗ Some models failed to download")
        print("\nManual download instructions:")
        print("1. ASTER model: https://github.com/ayumiymk/aster.pytorch")
        print("2. MORAN model: https://github.com/Canjie-Luo/MORAN_v2")
        print("3. CRNN model: https://github.com/meijieru/crnn.pytorch")
        return False

if __name__ == "__main__":
    # Install required packages
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
    
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
    
    main()