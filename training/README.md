# HTR fine-tune — TrOCR-small on IAM

Produces a domain-adapted TrOCR checkpoint for our ml-gateway, plus an
ONNX export compatible with `Dockerfile.fastapi-ort`. Complements (does
not replace) Dongting's `paperless_training_integration` retraining-
pipeline mechanism: her pipeline is the scheduler/gate/promote scaffold;
this script is the training body that produces the weights.

## What it does

1. Reads IAM parquet shards from MinIO (`warehouse/iam_dataset/{train,validation,test}/`)
2. Fine-tunes `microsoft/trocr-small-handwritten` with HuggingFace Trainer
3. Evaluates CER on IAM test split
4. Exports to ONNX via `optimum.onnxruntime.ORTModelForVision2Seq`
5. Uploads fine-tuned safetensors + ONNX to
   `warehouse/models/trocr-ft-v<VERSION>/` on MinIO
6. Logs run to MLflow (params, metrics, tags pointing at MinIO path)
7. Optionally registers as `htr` in MLflow's model registry
   (`--register` flag) so downstream `@production` alias promotion works

## Prerequisites

- IAM already ingested (`warehouse/iam_dataset/train/*.parquet` present on MinIO)
- MLflow reachable at `http://mlflow:5000` on `paperless_ml_net`
- MinIO reachable at `http://minio:9000` with `minioadmin/minioadmin`
- GPU available (P100 sufficient; fp16 auto-enabled on CUDA)

## Run

### One-off on the Chameleon node (via Docker run)

```bash
cd ~/paperless-ml/training
sg docker -c 'docker build -t htr-finetune .'

# ~30-60 min on P100 for 2 epochs over IAM (~6500 rows)
sg docker -c 'docker run --rm --network paperless_ml_net --gpus all \
  -e MINIO_ENDPOINT=minio:9000 \
  -e MINIO_ACCESS_KEY=minioadmin \
  -e MINIO_SECRET_KEY=minioadmin \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  htr-finetune \
    --epochs 2 --batch-size 8 --register'
```

### Via compose (once wired — see docker-compose.yml next iteration)

```bash
docker compose --profile training run --rm htr-finetune --epochs 2 --register
```

## Arguments

| Flag | Default | What |
|---|---|---|
| `--model` | `microsoft/trocr-small-handwritten` | Base checkpoint |
| `--epochs` | 2 | Training epochs (IAM pretrained TrOCR is strong — 1-3 is usual) |
| `--batch-size` | 8 | Per-device batch size (P100 fits 8 at fp16) |
| `--lr` | 5e-5 | Learning rate |
| `--mlflow-uri` | `$MLFLOW_TRACKING_URI` or `http://mlflow:5000` | Tracking server |
| `--experiment` | `htr-finetune` | MLflow experiment name |
| `--register` | off | Register as `htr` in MLflow registry |
| `--version-tag` | timestamp | Tag for MinIO artifact path |
| `--skip-onnx-export` | off | Smoke-test train loop without ONNX step |

## Output layout

After a successful run with version tag `20260424_143012`:

```
s3://paperless-datalake/warehouse/models/trocr-ft-v20260424_143012/
├── model/                      ← HuggingFace format
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
└── onnx/                       ← ONNX format (optimum export)
    ├── config.json
    ├── encoder_model.onnx
    ├── decoder_model.onnx
    ├── decoder_with_past_model.onnx
    └── ... (preprocessor files)
```

ml-gateway's `Dockerfile.fastapi-ort` variant reads from `/models`
(set via `ONNX_DIR=/models`) so the `onnx/` subtree is what gets mounted
or pulled at serving-container boot.

## How it integrates with the full system

```
         ┌─────────────────────────────────────────────┐
         │  Dongting's paperless_training_integration  │
         │  (retraining pipeline scaffold)             │
         │                                             │
         │  scheduler → train.py → eval → gate → alias │
         └───────────────────┬─────────────────────────┘
                             │
                  "train.py" can be any training body, including this one:
                             ▼
         ┌─────────────────────────────────────────────┐
         │  paperless-ml/training/                     │
         │  finetune_trocr.py                          │
         │                                             │
         │  IAM parquet → TrOCR fine-tune → CER → ONNX │
         │  → upload to MinIO + register in MLflow     │
         └───────────────────┬─────────────────────────┘
                             │
                             ▼
         ┌─────────────────────────────────────────────┐
         │  ml-gateway (Dockerfile.fastapi-ort, P100)  │
         │                                             │
         │  Reads ONNX checkpoint from MinIO at boot   │
         │  (HTR_MODEL_URI env → optimum ORTModel)     │
         │  Serves /htr requests with fine-tuned model │
         └─────────────────────────────────────────────┘
```

## Notes on serving contract

The ONNX export via `optimum.onnxruntime.ORTModelForVision2Seq` produces
three ONNX files (encoder, decoder, decoder-with-past) that match the
loading pattern in `serving/src/fastapi_app/app_ort.py`. The serving
container's `ONNX_DIR=/models` env var points at whichever directory
contains these — either a baked-in default (HF stock) or a MinIO-
downloaded fine-tune (via `HTR_MODEL_URI`).

## Known risks

- **TrOCR encoder-decoder ONNX export can fail** on certain transformers
  versions. If this script's ONNX step errors, re-run with
  `--skip-onnx-export`, debug ONNX separately, and fall back to serving
  the safetensors checkpoint via `Dockerfile.fastapi-cpu` (PyTorch path).
- **fp16 training precision** — safe on P100 for TrOCR-small. If you see
  NaN losses, drop fp16 via editing `Seq2SeqTrainingArguments` in the
  script.
- **Memory** — IAM train split (~6500 rows × 384² × 3 channels) is ~2.8 GB
  at fp16 + gradient memory. P100's 16 GB VRAM fits this with batch
  size 8; drop to 4 if OOM.
