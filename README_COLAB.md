# TPGSR Google Colab Setup

This guide provides instructions for running TPGSR (Text Prior Guided Scene Text Image Super-Resolution) in Google Colab.

## Quick Start

### Method 1: Using Jupyter Notebook (Recommended)

1. Open the `TPGSR_Colab.ipynb` notebook in Google Colab
2. Run all cells sequentially
3. The notebook will automatically:
   - Install dependencies
   - Download TextZoom dataset
   - Download pretrained models (ASTER, MORAN, CRNN)
   - Configure the environment
   - Fix compatibility issues
   - Run a demo

### Method 2: Using Python Script

```bash
# In Google Colab cell:
!git clone https://github.com/dyc2424748461/TPGSR.git
%cd TPGSR
!python setup_colab.py
```

### Method 3: Using Shell Script

```bash
# In Google Colab cell:
!git clone https://github.com/dyc2424748461/TPGSR.git
%cd TPGSR
!chmod +x setup_colab.sh
!./setup_colab.sh
```

## What the Setup Does

1. **Environment Check**: Verifies GPU availability and Colab environment
2. **Dependencies**: Installs PyTorch 1.2.0, torchvision 0.4.0, and other required packages
3. **Dataset Download**: Downloads TextZoom dataset from Google Drive
4. **Model Download**: Downloads pretrained recognizer models:
   - ASTER: `aster.pth.tar`
   - MORAN: `moran.pth`
   - CRNN: `crnn.pth`
5. **Configuration**: Updates paths and settings for Colab environment
6. **Code Fixes**: Applies compatibility fixes for CPU/GPU usage
7. **Demo Setup**: Creates demo files and test images

## Running the Model

### Demo
```python
# Run basic demo
!python run_demo.py
```

### Training
```python
# Start training (will take several hours)
!python main.py --batch_size=16 --STN --mask --gradient --sr_share --use_distill --without_colorjitter --test_model=TSRN
```

### Inference
```python
# Run inference on test data
!python main.py --test --resume=path/to/trained/model.pth
```

## File Structure After Setup

```
TPGSR/
├── config/
│   └── super_resolution.yaml    # Updated configuration
├── data/
│   └── TextZoom/               # Downloaded dataset
│       ├── train1/
│       ├── train2/
│       └── test/
├── pretrained/                 # Downloaded models
│   ├── aster.pth.tar
│   ├── moran.pth
│   └── crnn.pth
├── demo/
│   └── test.png               # Demo test image
├── demo_results/              # Demo outputs
├── TPGSR_Colab.ipynb         # Colab notebook
├── setup_colab.py            # Python setup script
├── setup_colab.sh            # Shell setup script
└── run_demo.py               # Demo script
```

## GPU vs CPU Mode

The setup automatically detects GPU availability:

- **GPU Mode**: Uses CUDA, batch_size=16, faster training
- **CPU Mode**: Uses CPU only, batch_size=4, slower but works without GPU

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch_size in `config/super_resolution.yaml`
2. **Download Failures**: Check Google Drive links and try again
3. **CUDA Errors**: The setup handles CPU/GPU compatibility automatically
4. **Import Errors**: Make sure all dependencies are installed

### Manual Fixes

If automatic setup fails, you can manually:

1. **Fix ptflops issues**:
   ```python
   # Comment out ptflops imports in interfaces/base.py and interfaces/super_resolution.py
   ```

2. **Fix model loading**:
   ```python
   # Add map_location to torch.load calls
   torch.load(model_path, map_location=torch.device('cpu'))
   ```

3. **Update configuration**:
   ```yaml
   # In config/super_resolution.yaml
   cuda: False  # Set to True if GPU available
   batch_size: 4  # Reduce for CPU or limited GPU memory
   ```

## Expected Runtime

- **Setup**: 5-10 minutes
- **Demo**: 1-2 minutes
- **Training**: 2-8 hours (depending on GPU)
- **Inference**: 1-5 minutes

## Memory Requirements

- **Minimum**: 4GB RAM (CPU mode)
- **Recommended**: 12GB GPU memory (GPU mode)
- **Storage**: ~2GB for dataset and models

## Citation

If you use this code, please cite the original TPGSR paper:

```bibtex
@article{tpgsr2021,
  title={Text Prior Guided Scene Text Image Super-resolution},
  author={...},
  journal={...},
  year={2021}
}
```

## Support

For issues specific to this Colab setup, please check:
1. GPU/CPU compatibility settings
2. File paths in configuration
3. Model download completion
4. Dependency versions

For general TPGSR issues, refer to the original repository.