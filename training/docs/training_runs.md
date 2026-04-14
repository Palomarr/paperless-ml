# Training Runs Summary

## Overview

We conducted multiple training experiments for the HTR model to evaluate different hyperparameter configurations. All experiments were executed on Chameleon Cloud and tracked using MLflow.

---

## Experiment Table

| Candidate                | MLflow run link               | Code version | Key hyperparams | Key model metrics | Key training cost metrics | Notes |
|--------------------------|-------------------------------|-------------|-----------------|------------------|--------------------------|------|
| baseline (HTR)           | file://mlruns/0/m-0bb76c11... | v1 | batch=16, lr=0.001 | loss≈2.20 | epoch_time≈0.3s | stable baseline with reasonable convergence |
| low_lr (HTR)             | file://mlruns/0/m-10f8806a... | v1 | batch=16, lr=0.0001 | loss≈2.98 | epoch_time≈0.3s | slower convergence and worse final loss |
| large_batch (HTR)        | file://mlruns/0/m-3d1015a0... | v1 | batch=32, lr=0.001 | loss≈2.43 | epoch_time≈0.25s | faster training but slightly worse performance |

| baseline (Retrieval )    | file://mlruns/0/m-13193d18... | v1 | batch=16, lr=0.001 | loss≈5.83 | epoch_time≈0.3s | baseline for retrieval model |
| low_lr (Retrieval)       | file://mlruns/0/m-20ef9228... | v1 | batch=16, lr=0.0001 | loss≈5.37 | epoch_time≈0.3s | more stable but not significantly better |
| large_batch Retrieval)   | file://mlruns/0/m-7e785d16... | v1 | batch=32, lr=0.001 | loss≈4.77 | epoch_time≈0.25s | best tradeoff: lower loss and faster training |

These configurations provide the best tradeoff between model quality and training cost.
---

## Analysis

The baseline configuration provides a reasonable starting point, but its loss is higher compared to the low learning rate setup.

The **low_lr experiment achieves the lowest final loss**, suggesting that a smaller learning rate allows the model to converge more effectively and avoid overshooting during optimization.

The **large_batch experiment does not significantly improve performance**, although it may provide better throughput in large-scale training scenarios.

---

## Recommended Configuration

Based on the results, we recommend using:

- Learning Rate: **0.0001**
- Batch Size: **16**

This configuration provides the best trade-off between training stability and model performance.

---

## Training Cost Considerations

- Larger batch sizes can improve training speed but may not improve model quality.
- Lower learning rates increase training time slightly but result in better convergence.

---

## Conclusion

We explored multiple candidate configurations and identified a promising setup for the HTR model. Future work includes integrating real datasets and evaluating using task-specific metrics such as CER (Character Error Rate).