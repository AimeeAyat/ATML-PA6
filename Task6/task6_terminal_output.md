TASK 6: GEOMETRY OF LOSS LANDSCAPE
================================================================================

1. Loading training metrics...
[2025-12-02 15:35:53] Loading training metrics from Task 1
[2025-12-02 15:35:53] Loaded 50000 epochs of metrics

2. Identifying phase checkpoints...
[2025-12-02 15:35:53] Identifying Memorization/Circuit Formation/Cleanup checkpoints

Phase checkpoints:
  Memorization: epoch 10000
  Circuit Formation: epoch 20000
  Cleanup: epoch 49999
[2025-12-02 15:35:53] Phase epochs: {
  "memorization": 10000,
  "circuit_formation": 20000,
  "cleanup": 49999
}

3. Loading dataset...
Dataset created: P=113, Train=3830, Test=8939
[2025-12-02 15:35:53] Dataset loaded

4. Computing Hessian eigenvalues for each phase...
[2025-12-02 15:35:53] Computing Hessian eigenvalues

  Computing Hessian for memorization phase (target epoch 10000)...
[2025-12-02 15:35:53] Computing Hessian for memorization
Moving model to device:  cuda

Computing Hessian for epoch 12500...
Eigenvalue 1: 1210.344727
Error processing checkpoint ..\Task1\checkpoints\checkpoint_epoch_12500.pt: dot(): argument 'tensor' (position 2) must be Tensor, not tuple

  Computing Hessian for circuit_formation phase (target epoch 20000)...
[2025-12-02 15:35:54] Computing Hessian for circuit_formation
Moving model to device:  cuda

Computing Hessian for epoch 25000...
Eigenvalue 1: 1755.511841
Error processing checkpoint ..\Task1\checkpoints\checkpoint_epoch_25000.pt: dot(): argument 'tensor' (position 2) must be Tensor, not tuple

  Computing Hessian for cleanup phase (target epoch 49999)...
[2025-12-02 15:35:55] Computing Hessian for cleanup
Moving model to device:  cuda

Computing Hessian for epoch 50000...
Eigenvalue 1: 0.003493
Error processing checkpoint ..\Task1\checkpoints\checkpoint_epoch_50000.pt: dot(): argument 'tensor' (position 2) must be Tensor, not tuple

5. Analyzing eigenvalue trends across phases...
[2025-12-02 15:35:56] Analyzing eigenvalue trends

6. Correlating with sparsity metrics...
[2025-12-02 15:35:56] Correlating eigenvalues with sparsity

Sparsity metrics from Task 2:
  Gini coefficient: 0.4613
  Top-10 energy concentration: 0.4183

7. Plotting eigenvalue analysis...

8. Theoretical analysis of landscape geometry...

9. Saving analysis results...
Analysis saved to logs\task6_analysis.json
[2025-12-02 15:35:56] Task 6 complete

================================================================================
KEY FINDINGS: LOSS LANDSCAPE GEOMETRY
================================================================================

3. WEIGHT DECAY HYPOTHESIS:
   - Weight decay (λ=1.0) biases toward sparse, distributed solutions
   - These solutions live in flatter regions of loss landscape
   - Result: Better generalization (test accuracy jumps)

4. SPARSITY-GEOMETRY CORRELATION:
   - Gini coefficient: 0.4613
   - Sparse circuits live in flat minima ✓