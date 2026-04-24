"""Fine-tune microsoft/trocr-small-handwritten on IAM parquet shards.

Architectural role: this is the *training body* for our serving path.
Dongting's paperless_training_integration provides the retraining-pipeline
*mechanism* (scheduler → gate → register → alias). This script is what
produces the specific model weights the mechanism promotes.

Read: IAM parquet shards from MinIO at
    s3://paperless-datalake/warehouse/iam_dataset/train/*.parquet
    s3://paperless-datalake/warehouse/iam_dataset/validation/*.parquet
    s3://paperless-datalake/warehouse/iam_dataset/test/*.parquet
Parquet schema (per Elnath's ingester commit 4922cd2):
    image_id       str    — unique identifier
    image_png      bytes  — PNG encoded
    transcription  str    — ground-truth text
    split          str    — train/test/validation marker

Writes:
    - Fine-tuned safetensors checkpoint at
      s3://paperless-datalake/warehouse/models/trocr-ft-v<N>/model/
    - Exported ONNX (via optimum) at
      s3://paperless-datalake/warehouse/models/trocr-ft-v<N>/onnx/
    - MLflow run with CER on test split, hyperparameters, tags
    - (Optional, flag-gated) registered model version on MLflow's
      htr registry so downstream alias promotion works

Run:
    # Inside a GPU-enabled container with MLflow + MinIO reachable
    python finetune_trocr.py \
        --epochs 3 \
        --batch-size 8 \
        --mlflow-uri http://mlflow:5000 \
        --register

    # Or via compose (once docker-compose adds a service entry)
    docker compose run --rm htr-finetune

Design notes:
    - Uses HuggingFace Trainer for training loop (well-tested, handles
      mixed precision, logging, eval, checkpointing automatically).
    - CER metric via `jiwer.cer` — standard HTR metric, easy to interpret.
    - ONNX export via `optimum.onnxruntime.ORTModelForVision2Seq` — the
      canonical HF export path for TrOCR encoder-decoder models.
    - Parquet reading via minio + pyarrow streaming (no local-disk copy);
      HF Dataset.from_generator feeds Trainer without pre-materializing.
    - Small epoch counts (1-3) are expected — TrOCR pre-training is strong,
      fine-tune is for domain adaptation, not from-scratch learning.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from jiwer import cer
from minio import Minio
from PIL import Image
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

import mlflow
from datasets import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s finetune_trocr: %(message)s",
)
log = logging.getLogger("finetune_trocr")


# ─── Config ───────────────────────────────────────────────────────────

DEFAULT_MODEL = "microsoft/trocr-small-handwritten"
IAM_BUCKET = "paperless-datalake"
IAM_PREFIX = "warehouse/iam_dataset"
MODELS_PREFIX = "warehouse/models"
REGISTERED_MODEL_NAME = "htr"


def minio_client() -> Minio:
    return Minio(
        os.environ.get("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )


# ─── Data loading ─────────────────────────────────────────────────────


def iter_parquet_rows(mc: Minio, bucket: str, prefix: str, split: str) -> Iterator[dict]:
    """Yield rows from all parquet shards under `prefix/split/`.

    Streams each shard's bytes into memory one at a time rather than
    downloading to disk. IAM splits are ~100-500 MB each — fits.
    """
    shards = sorted(
        o.object_name
        for o in mc.list_objects(bucket, prefix=f"{prefix}/{split}/", recursive=True)
        if o.object_name.endswith(".parquet")
    )
    if not shards:
        raise RuntimeError(f"no parquet shards under s3://{bucket}/{prefix}/{split}/")
    log.info("found %d shards for split=%s", len(shards), split)

    for shard in shards:
        resp = mc.get_object(bucket, shard)
        try:
            data = resp.read()
        finally:
            resp.close()
            resp.release_conn()
        table = pq.read_table(io.BytesIO(data))
        for row in table.to_pylist():
            yield row


def build_hf_dataset(mc: Minio, processor: TrOCRProcessor, split: str) -> Dataset:
    """Build a HuggingFace Dataset for the given IAM split.

    Processes images + transcriptions eagerly into pixel_values + labels.
    For IAM's ~6500 train rows × (3 × 384 × 384 pixels) this is ~2.8 GB
    in memory — fine for the P100 node.
    """
    pixel_values = []
    label_ids = []
    max_text_len = 128  # IAM lines are typically <80 chars; 128 is safe ceiling

    for row in iter_parquet_rows(mc, IAM_BUCKET, IAM_PREFIX, split):
        try:
            img = Image.open(io.BytesIO(row["image_png"])).convert("RGB")
            encoded = processor(images=img, return_tensors="pt")
            pixel_values.append(encoded.pixel_values[0])

            text = (row.get("transcription") or "").strip()
            if not text:
                continue
            ids = processor.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_text_len,
            ).input_ids
            # Replace pad token with -100 so it's ignored in the loss
            ids = [(i if i != processor.tokenizer.pad_token_id else -100) for i in ids]
            label_ids.append(torch.tensor(ids))
        except Exception as e:
            log.debug("skipping bad row: %s", e)

    if len(pixel_values) != len(label_ids):
        # Align (shouldn't normally mismatch but defensive)
        n = min(len(pixel_values), len(label_ids))
        pixel_values = pixel_values[:n]
        label_ids = label_ids[:n]

    log.info("built dataset for split=%s: %d rows", split, len(pixel_values))

    return Dataset.from_dict(
        {
            "pixel_values": [pv.numpy() for pv in pixel_values],
            "labels": [l.numpy() for l in label_ids],
        }
    )


# ─── Metric ────────────────────────────────────────────────────────────


def make_compute_metrics(processor: TrOCRProcessor):
    """Return a compute_metrics(eval_pred) that computes CER."""

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred
        # Replace -100 in labels with pad_token_id so decoder can read them
        label_ids = np.where(
            label_ids == -100, processor.tokenizer.pad_token_id, label_ids
        )

        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_text = processor.batch_decode(label_ids, skip_special_tokens=True)

        # jiwer.cer accepts list-of-strings reference + hypothesis
        pairs = [(r, h) for r, h in zip(label_text, pred_text) if r.strip()]
        if not pairs:
            return {"cer": 1.0}
        refs, hyps = zip(*pairs)
        return {"cer": cer(list(refs), list(hyps))}

    return compute_metrics


# ─── ONNX export ───────────────────────────────────────────────────────


def export_to_onnx(checkpoint_dir: Path, onnx_dir: Path) -> None:
    """Export fine-tuned checkpoint to ONNX via optimum.

    optimum's ORTModelForVision2Seq handles the encoder-decoder export
    dance — splits encoder + decoder + decoder-with-past into separate
    ONNX files, which is what app_ort.py expects under /models.
    """
    # Delay-import so the module loads even if optimum isn't installed
    from optimum.onnxruntime import ORTModelForVision2Seq

    log.info("exporting %s → ONNX at %s", checkpoint_dir, onnx_dir)
    ort_model = ORTModelForVision2Seq.from_pretrained(
        checkpoint_dir, export=True
    )
    ort_model.save_pretrained(onnx_dir)
    log.info("ONNX export complete: %s", list(onnx_dir.glob("*.onnx")))


# ─── MinIO upload ──────────────────────────────────────────────────────


def upload_dir_to_minio(mc: Minio, local_dir: Path, bucket: str, prefix: str) -> None:
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir)
        key = f"{prefix}/{rel.as_posix()}"
        log.info("  uploading %s → s3://%s/%s (%s bytes)",
                 rel, bucket, key, path.stat().st_size)
        mc.fput_object(bucket, key, str(path))


# ─── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    )
    parser.add_argument("--experiment", default="htr-finetune")
    parser.add_argument(
        "--register",
        action="store_true",
        help=f"Register fine-tuned model as '{REGISTERED_MODEL_NAME}' in MLflow",
    )
    parser.add_argument(
        "--version-tag",
        default=None,
        help="Version tag for MinIO upload path (default: YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--skip-onnx-export",
        action="store_true",
        help="Skip the ONNX export step (useful for smoke-testing the train loop)",
    )
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Version tag for artifact paths
    if args.version_tag:
        version = args.version_tag
    else:
        from datetime import datetime, timezone
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    log.info("=" * 60)
    log.info("TrOCR fine-tune on IAM")
    log.info("  base model:   %s", args.model)
    log.info("  epochs:       %d", args.epochs)
    log.info("  batch size:   %d", args.batch_size)
    log.info("  lr:           %g", args.lr)
    log.info("  version tag:  %s", version)
    log.info("  MLflow URI:   %s", args.mlflow_uri)
    log.info("=" * 60)

    # Set up MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    mc = minio_client()
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)

    # Configure decoder start + pad tokens for Seq2Seq training
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    log.info("building HF datasets from MinIO parquet")
    train_ds = build_hf_dataset(mc, processor, "train")
    val_ds = build_hf_dataset(mc, processor, "validation")
    test_ds = build_hf_dataset(mc, processor, "test")

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "checkpoint"
        onnx_dir = Path(tmp) / "onnx"

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            predict_with_generate=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            fp16=torch.cuda.is_available(),
            logging_steps=20,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            report_to="none",  # We handle MLflow logging manually
            seed=args.seed,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=default_data_collator,
            compute_metrics=make_compute_metrics(processor),
            tokenizer=processor.feature_extractor,
        )

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            log.info("MLflow run: %s", run_id)
            mlflow.log_params(
                {
                    "base_model": args.model,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "seed": args.seed,
                    "train_rows": len(train_ds),
                    "val_rows": len(val_ds),
                    "test_rows": len(test_ds),
                }
            )

            log.info("starting training")
            trainer.train()

            log.info("evaluating on test split")
            test_metrics = trainer.evaluate(test_ds)
            log.info("test metrics: %s", test_metrics)
            mlflow.log_metrics(
                {f"test_{k.replace('eval_', '')}": v for k, v in test_metrics.items()
                 if isinstance(v, (int, float))}
            )

            # Save processor + model
            log.info("saving fine-tuned checkpoint")
            trainer.save_model(str(output_dir))
            processor.save_pretrained(output_dir)

            # Export to ONNX
            if not args.skip_onnx_export:
                export_to_onnx(output_dir, onnx_dir)

            # Upload to MinIO
            minio_prefix = f"{MODELS_PREFIX}/trocr-ft-v{version}"
            log.info("uploading artifacts to s3://%s/%s", IAM_BUCKET, minio_prefix)
            upload_dir_to_minio(mc, output_dir, IAM_BUCKET, f"{minio_prefix}/model")
            if not args.skip_onnx_export:
                upload_dir_to_minio(mc, onnx_dir, IAM_BUCKET, f"{minio_prefix}/onnx")

            mlflow.set_tag("minio_path", f"s3://{IAM_BUCKET}/{minio_prefix}")
            mlflow.set_tag("version", version)
            mlflow.set_tag("base_model", args.model)

            if args.register:
                log.info("registering MLflow model as '%s'", REGISTERED_MODEL_NAME)
                mlflow.register_model(
                    f"runs:/{run_id}/",
                    REGISTERED_MODEL_NAME,
                )

            log.info("=" * 60)
            log.info("DONE. MinIO artifacts: s3://%s/%s", IAM_BUCKET, minio_prefix)
            log.info("  CER on test: %.4f", test_metrics.get("eval_cer", -1.0))
            log.info("=" * 60)


if __name__ == "__main__":
    main()
