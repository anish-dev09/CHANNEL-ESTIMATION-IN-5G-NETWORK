# AI-Assisted Channel Estimation in 5G — Execution Plan (Phases 3-10)

**Created**: December 21, 2025  
**Updated**: December 22, 2025  
**Status**: Phase 3 In Progress (Dataset Generation)  
**Prerequisite**: Phases 0-2 Completed ✅

---

## 📋 Phase Status Tracker

| Phase | Name | Status | Duration | Dependencies |
|-------|------|--------|----------|--------------|
| Phase 0 | Environment Setup | ✅ Complete | - | None |
| Phase 1 | OFDM/MIMO Simulator | ✅ Complete | - | Phase 0 |
| Phase 2 | Baseline Estimators | ✅ Complete | - | Phase 1 |
| **Phase 3** | Dataset Generation | 🔄 **IN PROGRESS** | ~2 hrs | Phase 1, 2 |
| Phase 4 | CNN Model Training | ⏳ Pending | 2-3 hrs | Phase 3 |
| Phase 5 | Evaluation & Comparison | ⏳ Pending | 1-2 hrs | Phase 4 |
| Phase 6-7 | Advanced AI Models | ⏳ Pending | 3-4 hrs | Phase 5 |
| Phase 8 | Pilot Optimization | ⏳ Pending | 1-2 hrs | Phase 5 |
| Phase 9 | Hyperparameter Tuning | ⏳ Pending | 2-3 hrs | Phase 5 |
| Phase 10 | Final Report | ⏳ Pending | 1-2 hrs | Phase 8, 9 |

**Total Remaining Time**: ~16-22 hours

---

## 🎯 Project Goals Recap

1. **Primary Goal**: Develop AI models that outperform classical LS/MMSE estimators
2. **Secondary Goal**: Reduce pilot overhead from 10% to 5% or less
3. **Metrics**: NMSE (dB), BER, Inference Latency (ms)
4. **Target Performance**:
   - AI NMSE: 3-5 dB better than MMSE
   - Pilot reduction: 50% less pilots with same performance
   - Latency: < 10ms for real-time application

---

# Phase 3: Dataset Generation

**Objective**: Generate comprehensive datasets for training, validation, and testing AI models.

## 3.1 Tasks Overview

| Task | Description | Output |
|------|-------------|--------|
| 3.1 | Fix dataset generator imports | Working script |
| 3.2 | Generate training dataset | `data/train.npz` (10,000 samples) |
| 3.3 | Generate validation dataset | `data/val.npz` (1,000 samples) |
| 3.4 | Generate test dataset | `data/test.npz` (2,000 samples) |
| 3.5 | Verify dataset integrity | Validation report |
| 3.6 | Create PyTorch DataLoader test | Working data pipeline |

## 3.2 Dataset Specifications

```yaml
Training Dataset:
  samples: 10000
  channel_types: [EPA, EVA, ETU]  # Uniform random
  snr_range: [-5, 0, 5, 10, 15, 20, 25, 30]  # dB
  doppler_hz: [10, 50, 100, 200]
  pilot_density: [0.05, 0.10]  # 5% and 10%
  
Validation Dataset:
  samples: 1000
  same distribution as training
  
Test Dataset:
  samples: 2000
  same distribution as training
```

## 3.3 Data Format

```python
# Each sample contains:
{
    'rx_symbols': (14, 2, 599),      # Received OFDM symbols [symbols, rx_ant, subcarriers]
    'tx_symbols': (14, 2, 599),      # Transmitted symbols
    'H_ls': (14, 2, 2, 599),         # LS estimate (input feature)
    'H_true': (14, 2, 2, 599),       # True channel (target)
    'pilot_mask': (14, 599),         # Pilot positions
    'snr_db': float,                 # SNR value
    'channel_type': str,             # EPA/EVA/ETU
    'doppler_hz': float,             # Doppler frequency
    'pilot_density': float           # Pilot density
}
```

## 3.4 Commands

```powershell
# Step 1: Create data directory
mkdir -Force data

# Step 2: Generate training data
python run_dataset_generation.py --split train --samples 10000

# Step 3: Generate validation data
python run_dataset_generation.py --split val --samples 1000

# Step 4: Generate test data
python run_dataset_generation.py --split test --samples 2000

# Step 5: Verify datasets
python verify_datasets.py
```

## 3.5 Success Criteria

- [ ] Training dataset: 10,000 samples saved to `data/train.npz`
- [ ] Validation dataset: 1,000 samples saved to `data/val.npz`
- [ ] Test dataset: 2,000 samples saved to `data/test.npz`
- [ ] All array shapes verified
- [ ] Channel types distributed uniformly
- [ ] SNR range covered uniformly
- [ ] DataLoader test passes

---

# Phase 4: CNN Model Training

**Objective**: Train the CNN-based channel estimator and validate it outperforms baselines.

## 4.1 Tasks Overview

| Task | Description | Output |
|------|-------------|--------|
| 4.1 | Verify model architecture | Model summary |
| 4.2 | Create training script runner | `run_training.py` |
| 4.3 | Train CNN model (50 epochs) | `models/cnn_best.pth` |
| 4.4 | Monitor training with TensorBoard | Loss curves |
| 4.5 | Evaluate on test set | NMSE comparison |
| 4.6 | Compare CNN vs LS vs MMSE | Results table |

## 4.2 Training Configuration

```yaml
Model: CNN
  input_channels: 5  # [rx_real, rx_imag, H_ls_real, H_ls_imag, pilot_mask]
  hidden_channels: [64, 128, 256, 128, 64]
  kernel_size: 3
  dropout: 0.1

Training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: Adam
  scheduler: CosineAnnealing
  early_stopping_patience: 10
```

## 4.3 Commands

```powershell
# Train CNN model
python run_training.py --model cnn --epochs 50 --batch-size 32

# Monitor training (separate terminal)
tensorboard --logdir logs

# Evaluate trained model
python run_evaluation.py --model cnn --checkpoint models/cnn_best.pth
```

## 4.4 Expected Results

| Method | NMSE @ 10dB | NMSE @ 20dB |
|--------|-------------|-------------|
| LS | ~-15 dB | ~-20 dB |
| MMSE | ~-18 dB | ~-25 dB |
| CNN | ~-21 dB | ~-28 dB |

## 4.5 Success Criteria

- [ ] CNN training converges (loss decreasing)
- [ ] Validation loss < Training loss (no overfitting)
- [ ] CNN NMSE > MMSE NMSE by at least 2 dB
- [ ] Model checkpoint saved
- [ ] TensorBoard logs generated

---

# Phase 5: Advanced AI Models

**Objective**: Train and compare multiple neural network architectures.

## 5.1 Models to Train

| Model | Architecture | Expected Advantage |
|-------|--------------|-------------------|
| LSTM | 3-layer BiLSTM | Temporal dynamics |
| Hybrid | CNN + LSTM | Spatial + Temporal |
| ResNet | 4 residual blocks | Deeper network |

## 5.2 Commands

```powershell
# Train LSTM model
python run_training.py --model lstm --epochs 50

# Train Hybrid model
python run_training.py --model hybrid --epochs 50

# Train ResNet model
python run_training.py --model resnet --epochs 50

# Compare all models
python compare_models.py --models cnn,lstm,hybrid,resnet
```

## 5.3 Success Criteria

- [ ] All 4 models trained successfully
- [ ] Comparison table generated
- [ ] Best model identified
- [ ] Training curves for all models

---

# Phase 6: Training Pipeline Optimization

**Objective**: Improve training efficiency and model performance.

## 6.1 Optimizations

| Optimization | Purpose |
|--------------|---------|
| Learning rate warmup | Stable early training |
| Gradient clipping | Prevent exploding gradients |
| Data augmentation | Improve generalization |
| Mixed precision (FP16) | Faster training on GPU |

## 6.2 Commands

```powershell
# Retrain best model with optimizations
python run_training.py --model best --epochs 100 --use-amp --warmup-epochs 5
```

## 6.3 Success Criteria

- [ ] Training time reduced by 20%+
- [ ] Final NMSE improved by 0.5+ dB
- [ ] No training instabilities

---

# Phase 7: Evaluation & Comparison

**Objective**: Comprehensive evaluation across all conditions.

## 7.1 Evaluation Scenarios

| Scenario | Parameters |
|----------|------------|
| SNR Sweep | -5 to 30 dB (step 5) |
| Channel Models | EPA, EVA, ETU separately |
| Doppler Effects | 10, 50, 100, 200 Hz |
| Pilot Density | 1%, 2%, 5%, 10% |

## 7.2 Metrics

- **NMSE** (dB): Primary accuracy metric
- **MSE**: Absolute error metric
- **BER**: Bit error rate after equalization
- **Latency**: Inference time per sample

## 7.3 Commands

```powershell
# Full evaluation
python run_full_evaluation.py --models ls,mmse,cnn,lstm,hybrid,resnet

# Generate plots
python generate_evaluation_plots.py
```

## 7.4 Expected Outputs

```
results/
├── snr_sweep_comparison.png
├── channel_model_comparison.png
├── doppler_analysis.png
├── latency_comparison.png
├── evaluation_results.json
└── evaluation_report.txt
```

## 7.5 Success Criteria

- [ ] All methods evaluated at all SNR points
- [ ] BER computed for all methods
- [ ] Latency benchmarked
- [ ] Publication-quality plots generated

---

# Phase 8: Pilot Overhead Optimization

**Objective**: Demonstrate AI advantage with reduced pilot density.

## 8.1 Experiment Design

Test all methods at pilot densities: 10%, 5%, 2%, 1%

**Key Question**: At what pilot density does AI match baseline performance at 10%?

## 8.2 Expected Results

| Pilot Density | LS NMSE | MMSE NMSE | CNN NMSE |
|---------------|---------|-----------|----------|
| 10% | -15 dB | -18 dB | -22 dB |
| 5% | -10 dB | -14 dB | -20 dB |
| 2% | -5 dB | -8 dB | -16 dB |
| 1% | 0 dB | -2 dB | -12 dB |

**Key Finding**: CNN at 5% achieves same performance as MMSE at 10% → 50% pilot reduction!

## 8.3 Commands

```powershell
# Pilot density sweep
python pilot_overhead_analysis.py --densities 0.01,0.02,0.05,0.10
```

## 8.4 Success Criteria

- [ ] AI outperforms baseline at same pilot density
- [ ] Identified minimum pilot density for target NMSE
- [ ] Pilot reduction benefit quantified

---

# Phase 9: Hyperparameter Tuning

**Objective**: Optimize model configuration for best performance.

## 9.1 Search Space

```python
search_space = {
    'hidden_channels': [[32,64,128], [64,128,256], [64,128,256,128,64]],
    'kernel_size': [3, 5, 7],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
    'batch_size': [16, 32, 64, 128]
}
```

## 9.2 Commands

```powershell
# Run hyperparameter optimization (50 trials)
python hyperparameter_optimization.py --trials 50 --model cnn

# Train with optimal parameters
python run_training.py --config configs/optimal_config.yaml
```

## 9.3 Success Criteria

- [ ] Optimal hyperparameters identified
- [ ] Performance improved by 0.5+ dB over default
- [ ] Configuration saved for reproducibility

---

# Phase 10: Final Report & Visualization

**Objective**: Create publication-ready results and documentation.

## 10.1 Deliverables

| Deliverable | Description |
|-------------|-------------|
| `results/final_comparison.png` | Main NMSE vs SNR plot |
| `results/ber_comparison.png` | BER vs SNR plot |
| `results/pilot_overhead.png` | Pilot density analysis |
| `results/latency_accuracy.png` | Trade-off analysis |
| `FINAL_RESULTS.md` | Complete results summary |
| `notebooks/final_demo.ipynb` | Interactive demonstration |

## 10.2 Publication-Ready Figures

1. **Figure 1**: NMSE vs SNR for all methods
2. **Figure 2**: BER comparison after equalization
3. **Figure 3**: Pilot overhead reduction capability
4. **Figure 4**: Inference latency comparison
5. **Figure 5**: Channel estimation visualization (true vs estimated)

## 10.3 Commands

```powershell
# Generate all final plots
python generate_final_figures.py

# Create results summary
python generate_results_summary.py

# Update README with results
python update_readme_results.py
```

## 10.4 Success Criteria

- [ ] All figures generated at 300 DPI
- [ ] Results summary complete
- [ ] README updated with final results
- [ ] Jupyter demo notebook working
- [ ] All code documented

---

# 📊 Final Expected Results Summary

## Performance Comparison (@ 15 dB SNR, 10% pilots)

| Method | NMSE (dB) | BER | Latency (ms) | Parameters |
|--------|-----------|-----|--------------|------------|
| LS | -16.5 | 8.2e-3 | 0.3 | 0 |
| MMSE | -19.8 | 4.1e-3 | 1.8 | 0 |
| CNN | -23.4 | 1.9e-3 | 2.5 | ~500K |
| LSTM | -22.8 | 2.1e-3 | 4.2 | ~800K |
| Hybrid | -24.1 | 1.6e-3 | 5.8 | ~1.2M |
| ResNet | -23.9 | 1.7e-3 | 3.1 | ~600K |

## Key Findings

1. **Best AI Model**: Hybrid CNN-LSTM (highest accuracy)
2. **Best Trade-off**: CNN (good accuracy, fast inference)
3. **Pilot Reduction**: AI at 5% ≈ MMSE at 10% (50% saving)
4. **Latency**: All AI models < 10ms (real-time capable)

---

# 🚀 Quick Start Commands

```powershell
# Phase 3: Generate Data
python run_dataset_generation.py --split train --samples 10000
python run_dataset_generation.py --split val --samples 1000
python run_dataset_generation.py --split test --samples 2000

# Phase 4: Train CNN
python run_training.py --model cnn --epochs 50

# Phase 7: Evaluate
python run_full_evaluation.py

# Phase 10: Final Report
python generate_final_figures.py
```

---

**Document Version**: 1.0  
**Last Updated**: December 21, 2025  
**Next Action**: Execute Phase 3 - Dataset Generation
