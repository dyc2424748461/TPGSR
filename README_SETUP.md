# TPGSR Setup Guide

This guide provides instructions for setting up and running the TPGSR (Text Prior Guided Scene Text Image Super-Resolution) repository.

## Environment Setup

### 1. Create a Python 3.7 environment
```bash
conda create -n tpgsr python=3.7
conda activate tpgsr
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download required datasets and models

#### TextZoom Dataset
Download the TextZoom dataset and place it in the `data/TextZoom` directory.

#### Pretrained Models
Download the following pretrained recognizer models and place them in the `pretrained` directory:
- ASTER: aster.pth.tar
- MORAN: moran.pth
- CRNN: crnn.pth

## Configuration

Update the paths in `config/super_resolution.yaml` to point to your dataset and pretrained models:

```yaml
TRAIN:
  train_data_dir: [
           '/path/to/TextZoom/train1',
           '/path/to/TextZoom/train2',
  ]
  ...
  VAL:
    val_data_dir: [
            '/path/to/TextZoom/test/easy',
            '/path/to/TextZoom/test/medium',
            '/path/to/TextZoom/test/hard',
    ]
    ...
    rec_pretrained: '/path/to/pretrained/aster.pth.tar'
    moran_pretrained: '/path/to/pretrained/moran.pth'
    crnn_pretrained: '/path/to/pretrained/crnn.pth'
```

## Running the Model

### Training
To train the TPGSR-TSRN model:
```bash
./train_TPGSR-TSRN.sh
```

### Demo
To run a demo with a pretrained model:
```bash
./demo.sh
```

## CPU-Only Mode

If you don't have a GPU, modify the following in `config/super_resolution.yaml`:
```yaml
cuda: False
workers: 0
batch_size: 4  # Reduce batch size for CPU
```

## Troubleshooting

### Memory Issues
If you encounter memory issues, try:
1. Reducing batch size in `config/super_resolution.yaml`
2. Setting `workers: 0` to avoid multiprocessing issues
3. Using CPU mode by setting `cuda: False`

### CUDA Errors
If you get CUDA errors but have a GPU, make sure:
1. PyTorch is installed with CUDA support
2. Your CUDA version is compatible with PyTorch 1.2.0 (CUDA 9.2 or 10.0)
3. Your GPU drivers are up to date

## Citation

If you use this code, please cite the original TPGSR paper.