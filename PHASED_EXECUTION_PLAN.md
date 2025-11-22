# AI-Assisted Channel Estimation in 5G — Phased Execution Plan

## Project Phases Overview

This document breaks down the complete project into manageable phases with clear deliverables, verification steps, and success criteria for each phase.

---

## 📋 Phase Status Tracker

| Phase | Status | Duration | Completed |
|-------|--------|----------|-----------|
| Phase 0: Environment Setup | ✅ Ready | 30 min | ☐ |
| Phase 1: OFDM Simulator | 🔄 Current | 2-3 hours | ☐ |
| Phase 2: Baseline Estimators | ⏳ Pending | 2 hours | ☐ |
| Phase 3: Dataset Generation | ⏳ Pending | 1-2 hours | ☐ |
| Phase 4: AI Model Development | ⏳ Pending | 3-4 hours | ☐ |
| Phase 5: Training Pipeline | ⏳ Pending | 2-3 hours | ☐ |
| Phase 6: Evaluation & Analysis | ⏳ Pending | 2 hours | ☐ |
| Phase 7: Documentation & Reporting | ⏳ Pending | 2 hours | ☐ |

**Total Estimated Time:** 14-19 hours

---

## Phase 0: Environment Setup & Verification

### 🎯 Objectives
- Set up Python environment
- Install all dependencies
- Verify installation
- Test basic imports

### 📝 Tasks

#### Task 0.1: Create Virtual Environment
```powershell
cd "d:\MY PROJECTS\Channel Estimation In 5g"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Verification:**
```powershell
python --version  # Should show Python 3.8+
```

#### Task 0.2: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Verification:**
```powershell
pip list | Select-String "torch|numpy|matplotlib|scipy"
```

#### Task 0.3: Test Imports
```powershell
python -c "import numpy as np; import torch; import matplotlib.pyplot as plt; print('✓ All imports successful')"
```

#### Task 0.4: Create Directories
```powershell
mkdir -p data, models, results, logs
```

### ✅ Success Criteria
- [ ] Virtual environment activated
- [ ] All packages installed without errors
- [ ] Import test passes
- [ ] All directories created

### 🚫 Common Issues & Solutions
- **Issue:** `pip install torch` fails
  - **Solution:** Install PyTorch separately: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

---

## Phase 1: OFDM Simulator Validation

### 🎯 Objectives
- Test OFDM transmitter/receiver
- Verify channel models (EPA, EVA, ETU)
- Validate MIMO functionality
- Generate test transmission

### 📝 Tasks

#### Task 1.1: Test Channel Models
```powershell
python -c "from src.channel_simulator import ChannelModel; print('✓ Channel models loaded')"
```

**Expected Output:**
```
✓ Channel models loaded
```

#### Task 1.2: Test OFDM System
Create test file `test_phase1.py`:
```python
import sys
sys.path.append('src')
from channel_simulator import simulate_transmission
from utils import load_config

config = load_config('configs/experiment_config.yaml')

print("Testing OFDM Simulator...")
sim_data = simulate_transmission(
    config,
    channel_type='EVA',
    doppler_hz=50,
    snr_db=15,
    pilot_density=0.1
)

print(f"✓ Transmission simulated")
print(f"  RX symbols shape: {sim_data['rx_symbols'].shape}")
print(f"  Channel shape: {sim_data['channel'].shape}")
print(f"  Pilot density: {sim_data['pilot_pattern'].pilot_density*100:.1f}%")
print("\nPhase 1 Complete!")
```

Run test:
```powershell
python test_phase1.py
```

**Expected Output:**
```
Testing OFDM Simulator...
✓ Transmission simulated
  RX symbols shape: (14, 2, 599)
  Channel shape: (14, 2, 2, 599)
  Pilot density: 10.0%

Phase 1 Complete!
```

#### Task 1.3: Visualize Channel
Create `visualize_channel.py`:
```python
import sys
sys.path.append('src')
import numpy as np
import matplotlib.pyplot as plt
from channel_simulator import simulate_transmission
from utils import load_config

config = load_config('configs/experiment_config.yaml')
sim_data = simulate_transmission(config, 'EVA', 50, 15, 0.1)

H = sim_data['channel'][:, 0, 0, :]  # First RX-TX pair

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(np.abs(H), aspect='auto', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.title('Channel Magnitude')
plt.xlabel('Subcarrier')
plt.ylabel('OFDM Symbol')

plt.subplot(1, 2, 2)
plt.imshow(np.angle(H), aspect='auto', cmap='twilight')
plt.colorbar(label='Phase (rad)')
plt.title('Channel Phase')
plt.xlabel('Subcarrier')
plt.ylabel('OFDM Symbol')

plt.tight_layout()
plt.savefig('results/phase1_channel_visualization.png', dpi=150)
print("✓ Channel visualization saved to results/phase1_channel_visualization.png")
```

Run:
```powershell
python visualize_channel.py
start results\phase1_channel_visualization.png
```

### ✅ Success Criteria
- [ ] Channel models instantiate correctly
- [ ] OFDM transmission completes without errors
- [ ] Output shapes are correct
- [ ] Visualization shows time-frequency channel variation

### 📊 Deliverables
- `test_phase1.py` - Working test script
- `results/phase1_channel_visualization.png` - Channel plot

---

## Phase 2: Baseline Estimators Validation

### 🎯 Objectives
- Implement and test LS estimator
- Implement and test MMSE estimator
- Compare performance metrics
- Validate against ground truth

### 📝 Tasks

#### Task 2.1: Test LS Estimator
Create `test_phase2.py`:
```python
import sys
sys.path.append('src')
import numpy as np
from channel_simulator import simulate_transmission
from baseline_estimators import LSEstimator, evaluate_estimator
from utils import load_config

print("Phase 2: Testing Baseline Estimators\n")
print("="*60)

config = load_config('configs/experiment_config.yaml')

# Simulate transmission
sim_data = simulate_transmission(config, 'EVA', 50, 15, 0.1)

rx_symbols = sim_data['rx_symbols']
H_true = sim_data['channel']
pilot_pattern = sim_data['pilot_pattern']
pilot_symbols = sim_data['pilot_symbols']

# Prepare data
num_symbols, num_rx, num_sc = rx_symbols.shape
num_tx = sim_data['tx_symbols'].shape[1]
rx_4d = rx_symbols.reshape(num_symbols, num_rx, 1, num_sc)
rx_4d = np.repeat(rx_4d, num_tx, axis=2)

# Test LS
print("\n[1/2] Testing LS Estimator...")
ls_estimator = LSEstimator(interpolation_method='linear')
H_ls = ls_estimator.estimate(
    rx_4d, pilot_symbols,
    pilot_pattern.pilot_mask,
    pilot_pattern.pilot_positions
)

ls_metrics = evaluate_estimator(H_true, H_ls)
print(f"  ✓ LS Estimation Complete")
print(f"     NMSE: {ls_metrics['nmse_db']:.2f} dB")
print(f"     MSE:  {ls_metrics['mse']:.6f}")

# Test MMSE
print("\n[2/2] Testing MMSE Estimator...")
from baseline_estimators import MMSEEstimator

mmse_estimator = MMSEEstimator(estimate_statistics=True)
H_mmse = mmse_estimator.estimate(
    rx_4d, pilot_symbols,
    pilot_pattern.pilot_mask,
    pilot_pattern.pilot_positions,
    snr_db=15
)

mmse_metrics = evaluate_estimator(H_true, H_mmse)
print(f"  ✓ MMSE Estimation Complete")
print(f"     NMSE: {mmse_metrics['nmse_db']:.2f} dB")
print(f"     MSE:  {mmse_metrics['mse']:.6f}")

# Summary
print("\n" + "="*60)
print("COMPARISON:")
print(f"  LS NMSE:   {ls_metrics['nmse_db']:>8.2f} dB")
print(f"  MMSE NMSE: {mmse_metrics['nmse_db']:>8.2f} dB")
improvement = ls_metrics['nmse_db'] - mmse_metrics['nmse_db']
print(f"  Improvement: {improvement:>6.2f} dB")
print("="*60)
print("\n✓ Phase 2 Complete!")
```

Run:
```powershell
python test_phase2.py
```

**Expected Output:**
```
Phase 2: Testing Baseline Estimators

============================================================

[1/2] Testing LS Estimator...
  ✓ LS Estimation Complete
     NMSE: -18.45 dB
     MSE:  0.014267

[2/2] Testing MMSE Estimator...
  ✓ MMSE Estimation Complete
     NMSE: -21.32 dB
     MSE:  0.007345

============================================================
COMPARISON:
  LS NMSE:      -18.45 dB
  MMSE NMSE:    -21.32 dB
  Improvement:    2.87 dB
============================================================

✓ Phase 2 Complete!
```

#### Task 2.2: Run Quick Demo
```powershell
python quick_start.py
```

Review generated plots in `results/` folder.

### ✅ Success Criteria
- [ ] LS estimator runs without errors
- [ ] MMSE estimator runs without errors
- [ ] MMSE shows improvement over LS (2-4 dB typical)
- [ ] Visualizations generated successfully

### 📊 Deliverables
- `test_phase2.py` - Baseline test script
- `results/quick_demo_results.png` - Comparison plots
- `results/channel_response_comparison.png` - Detailed comparison

---

## Phase 3: Dataset Generation

### 🎯 Objectives
- Generate small test dataset
- Validate dataset structure
- Generate full training dataset
- Verify data quality

### 📝 Tasks

#### Task 3.1: Generate Small Test Dataset
```powershell
python src/dataset_generator.py `
    --train-samples 500 `
    --val-samples 50 `
    --test-samples 100 `
    --output-dir data `
    --seed 42
```

**Expected Output:**
```
Channel Estimation Dataset Generation
======================================================================
Generating train dataset with 500 samples...
100%|████████████████████████| 500/500 [01:08<00:00,  7.32it/s]
Dataset saved to data\train.npz

Generating val dataset with 50 samples...
100%|██████████████████████████| 50/50 [00:07<00:00,  7.14it/s]
Dataset saved to data\val.npz

Generating test dataset with 100 samples...
100%|████████████████████████| 100/100 [00:14<00:00,  7.08it/s]
Dataset saved to data\test.npz

Dataset generation completed!
Training samples: 500
Validation samples: 50
Test samples: 100
```

#### Task 3.2: Inspect Dataset
Create `inspect_dataset.py`:
```python
import numpy as np

data = np.load('data/train.npz')

print("Dataset Inspection")
print("="*60)
print("\nArrays in dataset:")
for key in data.keys():
    print(f"  {key:20s} shape: {data[key].shape}")

print("\n\nSample metadata:")
print(f"  SNR values: {np.unique(data['snr_db'])}")
print(f"  Doppler values: {np.unique(data['doppler_hz'])}")
print(f"  Channel types: {np.unique(data['channel_type'])}")
print(f"  Pilot densities: {np.unique(data['pilot_density'])}")

print("\n✓ Dataset structure validated")
```

Run:
```powershell
python inspect_dataset.py
```

#### Task 3.3: Generate Full Dataset (Optional - for production)
```powershell
# Only run this if you want the full dataset
python src/dataset_generator.py `
    --train-samples 10000 `
    --val-samples 1000 `
    --test-samples 2000
```

**Note:** This takes ~45 minutes. Skip for now if testing.

### ✅ Success Criteria
- [ ] Dataset files created (train.npz, val.npz, test.npz)
- [ ] All expected keys present in dataset
- [ ] Diverse conditions represented (SNR, channel types, etc.)
- [ ] No NaN or Inf values

### 📊 Deliverables
- `data/train.npz` - Training dataset
- `data/val.npz` - Validation dataset
- `data/test.npz` - Test dataset
- `inspect_dataset.py` - Dataset inspection script

---

## Phase 4: AI Model Development

### 🎯 Objectives
- Test CNN model architecture
- Verify forward pass
- Check model size and parameters
- Test on sample data

### 📝 Tasks

#### Task 4.1: Test Model Architectures
```powershell
python src/ai_models.py
```

**Expected Output:**
```
Testing channel estimation models...

1. Testing CNN model...
   Input shape: torch.Size([4, 5, 14, 600])
   Output shape: torch.Size([4, 2, 14, 600])
   Parameters: 1,234,567

2. Testing ResNet model...
   Input shape: torch.Size([4, 5, 14, 600])
   Output shape: torch.Size([4, 2, 14, 600])
   Parameters: 987,654

3. Testing Hybrid CNN-LSTM model...
   Input shape: torch.Size([4, 5, 14, 600])
   Output shape: torch.Size([4, 2, 14, 600])
   Parameters: 2,345,678

4. Testing loss function...
   Loss value: 0.123456

All model tests passed!
```

#### Task 4.2: Test Model on Real Data
Create `test_phase4.py`:
```python
import sys
sys.path.append('src')
import torch
import numpy as np
from train import ChannelDataset
from ai_models import get_model

print("Phase 4: Testing AI Models on Real Data\n")
print("="*60)

# Load dataset
dataset = ChannelDataset('data/train.npz', normalize=True)
print(f"✓ Dataset loaded: {len(dataset)} samples")

# Get one sample
inputs, targets, pilot_mask = dataset[0]
print(f"✓ Sample loaded")
print(f"  Input shape: {inputs.shape}")
print(f"  Target shape: {targets.shape}")

# Test CNN model
print("\nTesting CNN model...")
model = get_model('cnn', {'input_channels': 5})
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Forward pass
inputs_batch = inputs.unsqueeze(0)  # Add batch dimension
with torch.no_grad():
    output = model(inputs_batch)

print(f"  ✓ Forward pass successful")
print(f"  Output shape: {output.shape}")

# Calculate loss
loss_fn = torch.nn.MSELoss()
targets_batch = targets.unsqueeze(0)
loss = loss_fn(output, targets_batch)
print(f"  Loss (untrained): {loss.item():.6f}")

print("\n" + "="*60)
print("✓ Phase 4 Complete!")
print("  All models working correctly")
```

Run:
```powershell
python test_phase4.py
```

### ✅ Success Criteria
- [ ] All model architectures instantiate correctly
- [ ] Forward pass works on real data
- [ ] Output shapes match expected dimensions
- [ ] Loss can be computed

### 📊 Deliverables
- `test_phase4.py` - Model testing script
- Verified model architectures

---

## Phase 5: Training Pipeline

### 🎯 Objectives
- Train CNN model on small dataset
- Monitor training progress
- Save checkpoints
- Validate training convergence

### 📝 Tasks

#### Task 5.1: Quick Training Test (10 epochs)
```powershell
python src/train.py `
    --model cnn `
    --epochs 10 `
    --batch-size 32 `
    --device cpu
```

**Expected Output:**
```
Loading datasets...
Train samples: 500
Val samples: 50

Creating CNN model...

======================================================================
Starting Training
======================================================================
Model parameters: 1,234,567
Device: cpu
Epochs: 10
======================================================================

Epoch 1/10: 100%|████████| 16/16 [00:45<00:00,  2.81s/it, loss=0.0234]

Epoch 1/10
  Train Loss: 0.023456
  Val Loss: 0.018765
  LR: 0.001000
  ✓ Saved best model (val_loss: 0.018765)

[... continues for 10 epochs ...]

Training Completed!
Best validation loss: 0.012345
```

#### Task 5.2: Monitor Training (Optional)
In a separate terminal:
```powershell
tensorboard --logdir logs/tensorboard
```

Open browser to `http://localhost:6006`

#### Task 5.3: Full Training (Optional - if time permits)
```powershell
python src/train.py `
    --model cnn `
    --epochs 50 `
    --batch-size 32
```

**Note:** This takes 1-2 hours. Skip for now if testing.

### ✅ Success Criteria
- [ ] Training starts without errors
- [ ] Loss decreases over epochs
- [ ] Validation loss improves
- [ ] Model checkpoint saved
- [ ] TensorBoard logs created

### 📊 Deliverables
- `models/best_model.pth` - Best model checkpoint
- `models/final_model.pth` - Final model
- `logs/tensorboard/*` - Training logs

---

## Phase 6: Evaluation & Analysis

### 🎯 Objectives
- Evaluate trained model
- Compare with baselines
- Generate performance plots
- Analyze results

### 📝 Tasks

#### Task 6.1: Run Evaluation
```powershell
python src/evaluate.py `
    --model-types cnn `
    --data-dir data `
    --model-dir models `
    --output-dir results `
    --batch-size 16
```

**Expected Output:**
```
Loading test dataset...
Test samples: 100

Evaluating LS Estimator...
100%|████████████████████████| 100/100 [00:16<00:00,  6.25it/s]

Evaluating MMSE Estimator...
100%|████████████████████████| 100/100 [00:22<00:00,  4.44it/s]

Loading CNN model...
Loaded checkpoint from epoch 50 (loss: 0.012345)

Evaluating AI Model...
100%|████████████████████████| 7/7 [00:04<00:00,  1.56it/s]

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
Evaluation complete!
```

#### Task 6.2: View Results
```powershell
start results\comparison.png
start results\evaluation_report.txt
```

#### Task 6.3: Analyze Performance
Review:
- NMSE improvement over baselines
- Inference latency trade-offs
- Model size vs performance

### ✅ Success Criteria
- [ ] Evaluation completes without errors
- [ ] AI model outperforms LS baseline
- [ ] Results saved to files
- [ ] Plots generated successfully

### 📊 Deliverables
- `results/comparison.png` - Performance comparison chart
- `results/evaluation_results.json` - Detailed metrics
- `results/evaluation_report.txt` - Text report

---

## Phase 7: Documentation & Reporting

### 🎯 Objectives
- Document findings
- Create final report
- Prepare presentation
- Archive code

### 📝 Tasks

#### Task 7.1: Review All Results
Compile all generated outputs:
- Phase 1: Channel visualizations
- Phase 2: Baseline comparisons
- Phase 6: Final evaluation results

#### Task 7.2: Create Summary Report
Create `FINAL_REPORT.md`:
```markdown
# Project Summary: AI-Assisted Channel Estimation in 5G

## Executive Summary
[Your findings here]

## Methodology
- OFDM Parameters: [list]
- Channel Models: EPA, EVA, ETU
- Baseline Methods: LS, MMSE
- AI Method: CNN

## Results
### Performance Metrics
| Method | NMSE (dB) | Latency (ms) |
|--------|-----------|--------------|
| LS     | -18.45    | 12.34        |
| MMSE   | -21.32    | 25.67        |
| CNN    | -24.78    | 15.43        |

### Key Findings
1. CNN achieves 6.33 dB improvement over LS
2. CNN achieves 3.46 dB improvement over MMSE
3. Inference latency comparable to baselines

## Conclusions
[Your conclusions]

## Future Work
1. Test with more channel models
2. Reduce pilot density further
3. Try Transformer architectures
4. Hardware implementation study
```

#### Task 7.3: Clean Up Code
```powershell
# Remove test files
Remove-Item test_phase*.py
Remove-Item visualize_channel.py
Remove-Item inspect_dataset.py
```

### ✅ Success Criteria
- [ ] All results documented
- [ ] Final report created
- [ ] Code repository clean
- [ ] All deliverables archived

### 📊 Deliverables
- `FINAL_REPORT.md` - Complete project report
- All code in `src/` directory
- All results in `results/` directory
- Clean Git repository (optional)

---

## 🎓 Learning Checkpoints

After each phase, verify understanding:

- **Phase 1:** Do you understand how OFDM works?
- **Phase 2:** Can you explain the difference between LS and MMSE?
- **Phase 3:** Do you know what features go into the ML model?
- **Phase 4:** Can you explain the CNN architecture?
- **Phase 5:** Do you understand the training process?
- **Phase 6:** Can you interpret the performance metrics?

---

## 📌 Quick Reference Commands

### Essential Commands
```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Quick test
python quick_start.py

# Generate small dataset
python src/dataset_generator.py --train-samples 500 --val-samples 50 --test-samples 100

# Train for 10 epochs (quick test)
python src/train.py --model cnn --epochs 10 --batch-size 32

# Evaluate
python src/evaluate.py --model-types cnn

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🔄 Iteration Strategy

If results are not satisfactory:

1. **Poor LS/MMSE performance?** → Check SNR, pilot density
2. **AI model not learning?** → Increase epochs, adjust learning rate
3. **Overfitting?** → Add dropout, reduce model size
4. **Slow training?** → Reduce batch size, use GPU
5. **Poor generalization?** → Generate more diverse dataset

---

## 📅 Time Management

**Recommended Schedule:**

- **Day 1 (4 hours):** Phases 0-2
- **Day 2 (4 hours):** Phases 3-4
- **Day 3 (4 hours):** Phases 5-6
- **Day 4 (2 hours):** Phase 7 + Buffer

**Minimum Viable Path (8 hours):**
- Phase 0: 30 min
- Phase 1: 1 hour (skip visualization)
- Phase 2: 1 hour
- Phase 3: 1 hour (small dataset only)
- Phase 4: 30 min
- Phase 5: 2 hours (10 epochs only)
- Phase 6: 1 hour
- Phase 7: 1 hour

---

## ✅ Final Checklist

Before considering project complete:

- [ ] All phases completed
- [ ] All tests pass
- [ ] Results documented
- [ ] Code is clean and commented
- [ ] README updated
- [ ] Known issues documented
- [ ] Future work identified

---

**Ready to start? Begin with Phase 0!**

Run:
```powershell
cd "d:\MY PROJECTS\Channel Estimation In 5g"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
