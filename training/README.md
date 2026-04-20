# Paperless-ngx Training 

This repository contains the **training subsystem** for our Paperless-ngx ML system project.  
It supports two complementary ML models:

1. **HTR model** for handwriting text recognition  
2. **Retrieval model** for semantic search

This subsystem is designed to support the full training lifecycle required by the course project:

- training
- evaluation
- experiment tracking with MLflow
- quality gate checking
- ONNX export
- Triton packaging
- serving handoff

---

## Team

- Dongting Gao — Training
- Yikai Sun — Serving
- Elnath Zhao — Data

---

## Project Context

Our team extends **Paperless-ngx** with two ML features:

- **Handwriting Text Recognition (HTR)** at upload time
- **Semantic Search Retrieval** at query time

The training subsystem is responsible for:

- training candidate models
- comparing multiple runs
- logging all runs in MLflow
- evaluating model quality
- applying quality gates before promotion
- exporting approved models to ONNX
- packaging models into Triton-compatible format for serving

---

## Repository Structure

```text
paperless_training/
│
├── configs/
│   ├── train_config.yaml
│   ├── eval_config.yaml
│   ├── export_config.yaml
│   ├── quality_gate.yaml
│   └── handoff_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── htr_input/
│   │   ├── retrieval_input/
│   │   └── feedback_events/
│   │
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   ├── htr_input_sample.json
│   ├── htr_output_sample.json
│   ├── retrieval_input_sample.json
│   └── retrieval_output_sample.json
│
├── docs/
│   ├── training_design.md
│   ├── training_runs.md
│   ├── data_contract.md
│   ├── model_contract.md
│   ├── serving_handoff.md
│   ├── quality_gate_spec.md
│   └── deployment_note.md
│
├── mlruns/
│
├── models/
│   ├── model.py
│   ├── trocr_model.py
│   └── retrieval_model.py
│
├── outputs/
│   ├── checkpoints/
│   ├── exported/
│   ├── triton_repo/
│   └── reports/
│
├── trainer/
│   ├── trainer.py
│   ├── loader.py
│   ├── metrics.py
│   ├── utils.py
│   ├── quality_gate.py
│   └── mlflow_helper.py
│
├── Dockerfile
├── requirements.txt
├── README.md
├── train.py
├── eval.py
├── export_model.py
├── package_for_triton.py
└── provision_chameleon.ipynb