# Quick Reference: Phase Execution Scripts

## Overview

This document provides quick reference commands for each phase of the AI-assisted channel estimation project.

---

## Phase 3: Dataset Generation

```powershell
# Quick test (100 samples)
python run_phase3_robust.py --samples 100

# Full dataset with checkpointing (RECOMMENDED)
python run_phase3_robust.py --train 10000 --val 1000 --test 2000 --checkpoint 500

# Resume from checkpoint if interrupted
python run_phase3_robust.py --train 10000 --val 1000 --test 2000 --resume

# Verify datasets
python verify_phase3_datasets.py --detailed
```

**Output**: `data/train.npz`, `data/val.npz`, `data/test.npz`

---

## Phase 4: CNN Model Training

```powershell
# Quick test (5 epochs)
python run_phase4_training.py --model cnn --quick

# Full training
python run_phase4_training.py --model cnn --epochs 50

# With specific batch size
python run_phase4_training.py --model cnn --epochs 50 --batch-size 64
```

**Output**: `models/cnn_best.pth`, `models/cnn_history.json`

---

## Phase 5: Evaluation

```powershell
# Evaluate CNN model
python run_phase5_evaluation.py --model cnn

# Evaluate all models with SNR sweep
python run_phase5_evaluation.py --all --snr-sweep

# Baselines only (LS, MMSE)
python run_phase5_evaluation.py --baselines-only
```

**Output**: `results/evaluation_results.json`, `results/nmse_vs_snr.png`

---

## Phase 6-7: Advanced Models

```powershell
# Train LSTM
python run_phase6_advanced_training.py --model lstm --epochs 50

# Train Hybrid CNN-LSTM
python run_phase6_advanced_training.py --model hybrid --epochs 50

# Train ResNet
python run_phase6_advanced_training.py --model resnet --epochs 50

# Train all at once
python run_phase6_advanced_training.py --all --epochs 50
```

**Output**: `models/{model}_best.pth` for each model

---

## Phase 8: Pilot Optimization

```powershell
# Standard analysis
python run_phase8_pilot_optimization.py

# Custom densities
python run_phase8_pilot_optimization.py --densities 0.03 0.05 0.08 0.10 0.15

# More samples for accuracy
python run_phase8_pilot_optimization.py --samples 100
```

**Output**: `results/pilot_optimization.png`, `results/pilot_optimization_report.md`

---

## Phase 9: Hyperparameter Tuning

```powershell
# Quick test
python run_phase9_hyperparameter_tuning.py --model cnn --quick

# Full random search
python run_phase9_hyperparameter_tuning.py --model cnn --trials 20

# Grid search
python run_phase9_hyperparameter_tuning.py --model cnn --method grid
```

**Output**: `results/hyperparameter_tuning/{model}_tuning_results.json`

---

## Phase 10: Final Report

```powershell
# Generate comprehensive report
python run_phase10_final_report.py
```

**Output**: `results/final_report/FINAL_REPORT.md`, `results/final_report/figures/`

---

## Complete Pipeline (Sequential Execution)

```powershell
# Step 1: Generate datasets
python run_phase3_robust.py --train 10000 --val 1000 --test 2000 --checkpoint 500

# Step 2: Train CNN
python run_phase4_training.py --model cnn --epochs 50

# Step 3: Initial evaluation
python run_phase5_evaluation.py --model cnn --snr-sweep

# Step 4: Train advanced models
python run_phase6_advanced_training.py --all --epochs 50

# Step 5: Full comparison
python run_phase5_evaluation.py --all --snr-sweep

# Step 6: Pilot analysis
python run_phase8_pilot_optimization.py --samples 50

# Step 7: Tune best model
python run_phase9_hyperparameter_tuning.py --model cnn --trials 20

# Step 8: Generate final report
python run_phase10_final_report.py
```

---

## Useful Tips

1. **GPU Memory Issues**: Reduce batch size with `--batch-size 16`
2. **Slow Generation**: Use checkpointing with `--checkpoint 500` to save progress
3. **Quick Testing**: All scripts support `--quick` or small sample sizes
4. **Resume Training**: Model checkpoints are saved automatically as `*_best.pth`

---

## File Structure After Completion

```
data/
  ├── train.npz       # Training dataset
  ├── val.npz         # Validation dataset
  └── test.npz        # Test dataset

models/
  ├── cnn_best.pth    # Best CNN model
  ├── lstm_best.pth   # Best LSTM model
  ├── hybrid_best.pth # Best Hybrid model
  └── *_history.json  # Training histories

results/
  ├── evaluation_results.json
  ├── nmse_vs_snr.png
  ├── pilot_optimization.png
  └── final_report/
      ├── FINAL_REPORT.md
      └── figures/
```
