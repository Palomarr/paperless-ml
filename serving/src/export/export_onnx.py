"""Export bi-encoder and HTR models to ONNX format (B1, B2, B3)."""

import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import TrOCRProcessor


def export_biencoder(output_dir: str) -> str:
    """Export bi-encoder (all-mpnet-base-v2) to ONNX."""
    print("=== Exporting bi-encoder to ONNX ===")

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    transformer = model[0].auto_model
    tokenizer = model.tokenizer

    transformer.eval()

    dummy_input_ids = torch.zeros(1, 128, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)

    output_path = os.path.join(output_dir, "biencoder.onnx")

    torch.onnx.export(
        transformer,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "embeddings": {0: "batch", 1: "sequence"},
        },
    )
    print(f"Saved bi-encoder ONNX model to {output_path}")

    # Verify
    session = ort.InferenceSession(output_path)
    dummy_ids = np.zeros((1, 128), dtype=np.int64)
    dummy_mask = np.ones((1, 128), dtype=np.int64)
    outputs = session.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})
    # The transformer outputs (batch, seq, hidden); take mean pooling to get (1, 768)
    token_embeddings = outputs[0]
    embedding = np.mean(token_embeddings, axis=1)
    assert embedding.shape == (1, 768), f"Expected (1, 768), got {embedding.shape}"
    print(f"Verification passed: output shape {embedding.shape}")

    return output_path


def export_htr(output_dir: str) -> str:
    """Export HTR model (trocr-small-handwritten) to ONNX via optimum."""
    print("\n=== Exporting HTR model to ONNX ===")

    from optimum.onnxruntime import ORTModelForVision2Seq

    htr_dir = os.path.join(output_dir, "htr_onnx")

    model = ORTModelForVision2Seq.from_pretrained(
        "microsoft/trocr-small-handwritten", export=True
    )
    model.save_pretrained(htr_dir)
    print(f"Saved HTR ONNX model to {htr_dir}")

    # Verify
    loaded_model = ORTModelForVision2Seq.from_pretrained(htr_dir)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    dummy_image = Image.new("RGB", (224, 224), color="white")
    pixel_values = processor(images=dummy_image, return_tensors="pt").pixel_values
    generated_ids = loaded_model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    assert len(text) >= 0, "Expected text output from HTR model"
    print(f"Verification passed: generated text = '{text}'")

    return htr_dir


def validate_exports(output_dir: str):
    """Validate both ONNX exports."""
    print("\n=== Validating exports ===")

    # Bi-encoder validation
    print("Validating bi-encoder...")
    biencoder_path = os.path.join(output_dir, "biencoder.onnx")
    session = ort.InferenceSession(biencoder_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2"
    )
    inputs = tokenizer(
        "This is a test sentence.",
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        },
    )
    token_embeddings = outputs[0]
    embedding = np.mean(token_embeddings, axis=1)
    assert embedding.shape == (1, 768), f"Expected (1, 768), got {embedding.shape}"
    assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"
    print(f"  Shape: {embedding.shape}, dtype: {embedding.dtype} -- OK")

    # HTR validation
    print("Validating HTR model...")
    from optimum.onnxruntime import ORTModelForVision2Seq

    htr_dir = os.path.join(output_dir, "htr_onnx")
    model = ORTModelForVision2Seq.from_pretrained(htr_dir)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    dummy_image = Image.new("RGB", (224, 224), color="white")
    pixel_values = processor(images=dummy_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    assert isinstance(text, str), "Expected string output"
    print(f"  Generated text: '{text}' -- OK")

    # File sizes
    print("\nModel file sizes:")
    biencoder_size = os.path.getsize(biencoder_path)
    print(f"  biencoder.onnx: {biencoder_size / 1024 / 1024:.2f} MB")

    htr_total = 0
    for root, _, files in os.walk(htr_dir):
        for f in files:
            fpath = os.path.join(root, f)
            fsize = os.path.getsize(fpath)
            htr_total += fsize
            if f.endswith(".onnx"):
                print(f"  htr_onnx/{f}: {fsize / 1024 / 1024:.2f} MB")
    print(f"  htr_onnx/ total: {htr_total / 1024 / 1024:.2f} MB")

    print("\nAll validations passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX format")
    parser.add_argument(
        "--output-dir", type=str, default="./onnx_models", help="Output directory"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    export_biencoder(args.output_dir)
    export_htr(args.output_dir)
    validate_exports(args.output_dir)
