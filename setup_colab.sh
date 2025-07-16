#!/bin/bash

# TPGSR Google Colab Setup Script
echo "Setting up TPGSR in Google Colab..."

# Clone repository (if not already cloned)
if [ ! -d "TPGSR" ]; then
    git clone https://github.com/dyc2424748461/TPGSR.git
    cd TPGSR
fi

# Install dependencies
echo "Installing dependencies..."
pip install torch==1.2.0 torchvision==0.4.0
pip install numpy==1.18.0
pip install Pillow==6.2.2
pip install lmdb easydict pyfasttext editdistance tensorboardX
pip install pyyaml scipy matplotlib tqdm opencv-python
pip install IPython gdown

# Run Python setup script
echo "Running Python setup script..."
python setup_colab.py

echo "Setup completed! You can now run:"
echo "  python run_demo.py"
echo "  python main.py --batch_size=16 --STN --mask --gradient --sr_share --use_distill --without_colorjitter --test_model=TSRN"