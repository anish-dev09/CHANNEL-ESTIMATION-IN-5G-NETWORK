# Phase 2: Baseline Estimators Validation

**Status:** 🔄 Ready to Start  
**Duration:** 1-2 hours  
**Prerequisites:** Phase 1 completed ✅

---

## 🎯 Phase Objectives

- ✅ Implement and test LS (Least Squares) estimator
- ✅ Implement and test MMSE (Minimum Mean Square Error) estimator
- ✅ Compare performance across different SNR levels
- ✅ Test different interpolation methods
- ✅ Benchmark execution time

---

## 📋 Task Checklist

- [ ] Task 2.1: Test LS estimator
- [ ] Task 2.2: Test MMSE estimator  
- [ ] Task 2.3: Compare LS vs MMSE performance
- [ ] Task 2.4: Test interpolation methods
- [ ] Task 2.5: SNR sweep analysis
- [ ] Task 2.6: Generate baseline performance report

---

## Task 2.1: Test LS Estimator

Test the Least Squares channel estimator.

**Command:**
```powershell
python test_phase2_ls.py
```

**Expected Output:**
```
======================================================================
PHASE 2 - LS ESTIMATOR VALIDATION
======================================================================

Testing LS estimator with linear interpolation...
  SNR: 20 dB
  NMSE: ~0 to -5 dB
  ✓ LS estimator working correctly
```

---

## Task 2.2: Test MMSE Estimator

Test the MMSE channel estimator.

**Command:**
```powershell
python test_phase2_mmse.py
```

**Expected Output:**
```
======================================================================
PHASE 2 - MMSE ESTIMATOR VALIDATION
======================================================================

Testing MMSE estimator...
  SNR: 20 dB
  NMSE: ~-3 to -8 dB (better than LS)
  ✓ MMSE estimator working correctly
```

---

## Task 2.3: Compare LS vs MMSE

Run comprehensive comparison.

**Command:**
```powershell
python test_phase2_comparison.py
```

**Expected Output:**
```
Performance Comparison:
  LS   NMSE: 0.06 dB
  MMSE NMSE: -0.26 dB
  
  MMSE improvement: ~0.3 dB
  ✓ MMSE outperforms LS as expected
```

---

## Task 2.4: Test Interpolation Methods

Test different interpolation schemes (linear, cubic, nearest).

**Command:**
```powershell
python test_phase2_interpolation.py
```

**Expected Results:**
- Linear: Good balance
- Cubic: Best performance (smooth)
- Nearest: Fastest but least accurate

---

## Task 2.5: SNR Sweep Analysis

Test estimators across SNR range (-5 to 30 dB).

**Command:**
```powershell
python test_phase2_snr_sweep.py
```

**Expected Pattern:**
- Low SNR (<5 dB): High NMSE, both methods struggle
- Medium SNR (10-20 dB): MMSE shows advantage
- High SNR (>25 dB): Both converge, LS sufficient

---

## Task 2.6: Generate Performance Report

Create comprehensive baseline report.

**Command:**
```powershell
python generate_baseline_report.py
```

**Output Files:**
- `results/phase2_ls_performance.png`
- `results/phase2_mmse_performance.png`
- `results/phase2_comparison_chart.png`
- `results/phase2_baseline_report.txt`

---

## 📊 Expected Results

After completing Phase 2, you should have:

### Performance Metrics:

| Estimator | NMSE (15 dB) | NMSE (20 dB) | Complexity |
|-----------|--------------|--------------|------------|
| LS        | 0-2 dB       | -2 to 0 dB   | Low        |
| MMSE      | -2 to 0 dB   | -5 to -2 dB  | Medium     |

### Key Findings:
- MMSE consistently outperforms LS by 2-4 dB
- LS is faster (no covariance estimation)
- MMSE requires accurate noise variance estimation
- Cubic interpolation best for both methods

---

## ✅ Phase 2 Success Criteria

Check all boxes before proceeding to Phase 3:

- [ ] LS estimator produces reasonable channel estimates
- [ ] MMSE estimator outperforms LS
- [ ] Both estimators tested at multiple SNR levels
- [ ] Interpolation methods compared
- [ ] Execution time benchmarked
- [ ] Performance plots generated
- [ ] Baseline report created

---

## 🐛 Troubleshooting

### Issue: High NMSE Values
**Problem:** NMSE > 5 dB even at high SNR

**Solution:** Check:
- Pilot density (should be ≥ 5%)
- Interpolation method
- Channel coherence time

### Issue: MMSE No Better Than LS
**Problem:** MMSE NMSE ≈ LS NMSE

**Solution:**
- Verify noise variance estimation
- Check channel statistics estimation
- Ensure sufficient pilot density

---

## 📈 Performance Benchmarks

Document these values:

| Metric | LS Value | MMSE Value |
|--------|----------|------------|
| NMSE @ 10 dB | ___ dB | ___ dB |
| NMSE @ 15 dB | ___ dB | ___ dB |
| NMSE @ 20 dB | ___ dB | ___ dB |
| Execution Time | ___ ms | ___ ms |
| Memory Usage | ___ MB | ___ MB |

---

## 🔄 Next Steps

Once all success criteria are met:

1. ✅ Mark all checkboxes above
2. 📊 Review performance plots
3. 📝 Note baseline performance for AI comparison
4. ➡️ **Proceed to Phase 3: Dataset Generation**

---

## 💡 Key Learnings from Phase 2

After completing this phase, you should understand:

1. **LS Estimation:**
   - Simple matrix inversion at pilot locations
   - Fast but noise-sensitive
   - Good for high SNR scenarios

2. **MMSE Estimation:**
   - Leverages channel statistics
   - Requires noise variance knowledge
   - Better noise suppression

3. **Interpolation:**
   - Critical for filling non-pilot locations
   - Trade-off between accuracy and smoothness
   - Cubic works best for slowly varying channels

4. **Performance Trade-offs:**
   - Accuracy vs. complexity
   - Pilot overhead vs. estimation quality
   - Real-time feasibility considerations

---

**Ready to start Phase 2? Let's create the test scripts!**
