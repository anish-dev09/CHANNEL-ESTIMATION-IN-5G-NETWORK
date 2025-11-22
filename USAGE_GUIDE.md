# AI-Assisted Channel Estimation in 5G — Quick Start Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Demo](#quick-demo)
3. [Dataset Generation](#dataset-generation)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Clone or Set Up Project

```powershell
cd "d:\MY PROJECTS\Channel Estimation In 5g"
```

### Step 2: Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```powershell
python -c "import torch; import numpy; print('✓ Installation successful!')"
```

---

## Quick Demo

Run the quick demonstration to test the setup:

```powershell
python quick_start.py
```

**This will:**
- Simulate a MIMO-OFDM transmission
- Apply LS and MMSE channel estimation
- Generate comparison plots in `results/` folder
- Print performance metrics

**Expected output:**
```
AI-ASSISTED CHANNEL ESTIMATION IN 5G - QUICK DEMO
======================================================================

[1/5] Loading configuration...
      ✓ Configuration loaded

[2/5] Simulating MIMO-OFDM transmission...
      ✓ Transmission simulated

[3/5] Running LS channel estimation...
      ✓ LS NMSE: -18.45 dB

[4/5] Running MMSE channel estimation...
      ✓ MMSE NMSE: -21.32 dB

[5/5] Creating visualizations...
      ✓ Plots saved to 'results/' directory

✓ Demo completed successfully!
```

---

## Dataset Generation

### Basic Dataset Generation

Generate training, validation, and test datasets:

```powershell
python src/dataset_generator.py `
    --train-samples 5000 `
    --val-samples 500 `
    --test-samples 1000 `
    --output-dir data `
    --format npz
```

**Parameters:**
- `--train-samples`: Number of training samples (default: 50000)
- `--val-samples`: Number of validation samples (default: 5000)
- `--test-samples`: Number of test samples (default: 10000)
- `--output-dir`: Output directory (default: `data`)
- `--format`: File format `npz` or `h5` (default: `npz`)
- `--seed`: Random seed for reproducibility (default: 42)

### Small Dataset for Testing (Faster)

```powershell
python src/dataset_generator.py --train-samples 1000 --val-samples 100 --test-samples 200
```

**Expected output:**
```
Channel Estimation Dataset Generation
======================================================================
Generating train dataset with 1000 samples...
100%|████████████████████████| 1000/1000 [02:15<00:00,  7.37it/s]
Dataset saved to data\train.npz

Generating val dataset with 100 samples...
100%|██████████████████████████| 100/100 [00:14<00:00,  7.08it/s]
Dataset saved to data\val.npz

Generating test dataset with 200 samples...
100%|██████████████████████████| 200/200 [00:28<00:00,  7.14it/s]
Dataset saved to data\test.npz

Dataset generation completed!
Training samples: 1000
Validation samples: 100
Test samples: 200
```

---

## Model Training

### Train CNN Model

```powershell
python src/train.py --model cnn --epochs 50 --batch-size 32
```

**Parameters:**
- `--model`: Model type (`cnn`, `lstm`, `hybrid`, `resnet`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 64)
- `--device`: Device (`auto`, `cpu`, `cuda`)
- `--data-dir`: Data directory (default: `data`)
- `--config`: Configuration file (default: `configs/experiment_config.yaml`)

### Train with Custom Configuration

```powershell
python src/train.py `
    --model resnet `
    --epochs 30 `
    --batch-size 16 `
    --device cuda
```

### Monitor Training with TensorBoard

Open a new terminal:

```powershell
tensorboard --logdir logs/tensorboard
```

Then open browser to `http://localhost:6006`

**Expected training output:**
```
Loading datasets...
Train samples: 1000
Val samples: 100

Creating CNN model...

======================================================================
Starting Training
======================================================================
Model parameters: 1,234,567
Device: cpu
Epochs: 50
======================================================================

Epoch 1/50: 100%|█████████| 32/32 [00:45<00:00,  1.42s/it, loss=0.0234]

Epoch 1/50
  Train Loss: 0.023456
  Val Loss: 0.018765
  LR: 0.001000
  ✓ Saved best model (val_loss: 0.018765)

[... training continues ...]

Training Completed!
Best validation loss: 0.012345
```

---

## Evaluation

### Evaluate All Models

Compare LS, MMSE, and trained AI model:

```powershell
python src/evaluate.py `
    --model-types cnn `
    --model-dir models `
    --data-dir data `
    --output-dir results
```

**Parameters:**
- `--model-types`: Models to evaluate (space-separated, e.g., `cnn lstm`)
- `--model-dir`: Directory with trained models (default: `models`)
- `--data-dir`: Data directory (default: `data`)
- `--output-dir`: Results output directory (default: `results`)
- `--batch-size`: Batch size for evaluation (default: 16)

### Evaluate Multiple Models

```powershell
python src/evaluate.py --model-types cnn resnet hybrid
```

**Expected output:**
```
Loading test dataset...
Test samples: 200

Evaluating LS Estimator...
100%|████████████████████████████| 200/200 [00:32<00:00,  6.25it/s]

Evaluating MMSE Estimator...
100%|████████████████████████████| 200/200 [00:45<00:00,  4.44it/s]

Loading CNN model...
Loaded checkpoint from epoch 50 (loss: 0.012345)

Evaluating AI Model...
100%|████████████████████████████| 13/13 [00:08<00:00,  1.56it/s]

======================================================================
CHANNEL ESTIMATION EVALUATION REPORT
======================================================================

Configuration:
  FFT Size: 1024
  CP Length: 72
  MIMO: 2x2

----------------------------------------------------------------------
Results Summary:
----------------------------------------------------------------------

LS:
  NMSE: -18.45 ± 2.34 dB
  MSE:  0.014567 ± 0.002345
  Latency: 12.34 ± 1.23 ms

MMSE:
  NMSE: -21.32 ± 2.10 dB
  MSE:  0.007345 ± 0.001234
  Latency: 25.67 ± 2.45 ms

CNN:
  NMSE: -24.78 ± 1.87 dB
  MSE:  0.003321 ± 0.000876
  Latency: 15.43 ± 1.67 ms

======================================================================

Comparison plot saved to results\comparison.png
Evaluation complete! Results saved to results
```

### View Results

Check the `results/` folder for:
- `comparison.png` - Bar charts comparing methods
- `evaluation_results.json` - Detailed metrics in JSON format
- `evaluation_report.txt` - Text report

---

## Advanced Usage

### Custom Configuration

Edit `configs/experiment_config.yaml` to customize:

```yaml
# Example: Change OFDM parameters
ofdm:
  fft_size: 512  # Changed from 1024
  cp_length: 36
  num_symbols: 14

# Example: Change MIMO configuration
mimo:
  num_tx_antennas: 4  # Changed from 2
  num_rx_antennas: 4
  
# Example: Add more SNR values
simulation:
  snr_range: [-5, 0, 5, 10, 15, 20, 25, 30, 35]  # Added 35 dB
```

### Test Baseline Estimators Standalone

```powershell
python src/baseline_estimators.py
```

### Test Channel Simulator

```powershell
python -c "from src.channel_simulator import *; print('Simulator OK')"
```

### Generate Specific Channel Model Dataset

Edit `dataset_generator.py` to use specific channel:

```python
# In generate_dataset method, change:
channel_type = 'ETU'  # Fixed channel type
doppler = 100  # Fixed Doppler
```

---

## Project Workflow

### Complete Workflow from Scratch

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quick demo
python quick_start.py

# 3. Generate dataset (small for testing)
python src/dataset_generator.py --train-samples 2000 --val-samples 200 --test-samples 400

# 4. Train CNN model
python src/train.py --model cnn --epochs 30 --batch-size 32

# 5. Evaluate
python src/evaluate.py --model-types cnn

# 6. View results
start results\comparison.png
```

### For Production-Scale Experiments

```powershell
# Generate large dataset
python src/dataset_generator.py --train-samples 50000 --val-samples 5000 --test-samples 10000

# Train with more epochs
python src/train.py --model resnet --epochs 100 --batch-size 64 --device cuda

# Comprehensive evaluation
python src/evaluate.py --model-types cnn lstm hybrid resnet
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```powershell
pip install torch torchvision
```

### Issue: "FileNotFoundError: configs/experiment_config.yaml"

**Solution:** Run from project root directory:
```powershell
cd "d:\MY PROJECTS\Channel Estimation In 5g"
```

### Issue: CUDA out of memory

**Solution:** Reduce batch size:
```powershell
python src/train.py --model cnn --batch-size 16
```

Or use CPU:
```powershell
python src/train.py --model cnn --device cpu
```

### Issue: Training too slow

**Solutions:**
1. Use smaller dataset for testing:
   ```powershell
   python src/dataset_generator.py --train-samples 1000
   ```

2. Use fewer epochs:
   ```powershell
   python src/train.py --epochs 20
   ```

3. Enable GPU if available:
   ```powershell
   python src/train.py --device cuda
   ```

### Issue: "No checkpoint found"

**Solution:** Train a model first:
```powershell
python src/train.py --model cnn --epochs 10
```

---

## File Structure Reference

```
Channel Estimation In 5g/
│
├── src/
│   ├── channel_simulator.py     # OFDM + MIMO simulator
│   ├── baseline_estimators.py   # LS and MMSE estimators
│   ├── ai_models.py              # Neural network models
│   ├── dataset_generator.py     # Dataset generation
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── utils.py                  # Utility functions
│
├── configs/
│   └── experiment_config.yaml   # System configuration
│
├── notebooks/
│   └── 01_complete_demonstration.ipynb  # Interactive notebook
│
├── data/                         # Generated datasets (created by scripts)
├── models/                       # Saved model checkpoints
├── results/                      # Evaluation results and plots
├── logs/                         # Training logs
│
├── requirements.txt              # Python dependencies
├── quick_start.py                # Quick demonstration script
└── README.md                     # Project documentation
```

---

## Performance Benchmarks

**Expected results on test dataset:**

| Method | NMSE (dB) | Latency (ms) | Model Size |
|--------|-----------|--------------|------------|
| LS     | -18 to -20 | ~10-15      | N/A        |
| MMSE   | -21 to -23 | ~20-30      | N/A        |
| CNN    | -24 to -26 | ~15-20      | ~5 MB      |
| ResNet | -25 to -27 | ~18-25      | ~8 MB      |
| Hybrid | -26 to -28 | ~25-35      | ~12 MB     |

*Results vary based on SNR, channel model, and pilot density*

---

## Next Steps

1. **Experiment with different architectures:** Try `lstm`, `hybrid`, or `resnet` models
2. **Tune hyperparameters:** Modify `configs/experiment_config.yaml`
3. **Analyze pilot overhead:** Vary pilot density in configuration
4. **Test robustness:** Train on one channel model, test on another
5. **Extend to more antennas:** Change MIMO configuration to 4x4 or 8x8

---

## Support

For issues or questions:
1. Check this guide first
2. Review error messages carefully
3. Ensure you're in the correct directory
4. Verify all dependencies are installed

---

**Last Updated:** November 2025  
**Version:** 1.0
