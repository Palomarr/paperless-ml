"""Apply dynamic quantization to ONNX models."""

import argparse
import os

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_biencoder(input_path: str, output_path: str):
    """Dynamic quantization for MiniLM bi-encoder."""
    print("=== Quantizing bi-encoder ===")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
    )
    orig_size = os.path.getsize(input_path) / 1e6
    quant_size = os.path.getsize(output_path) / 1e6
    print(f"  Original: {orig_size:.2f} MB -> Quantized: {quant_size:.2f} MB "
          f"({(1 - quant_size / orig_size) * 100:.1f}% reduction)")

    # Verify output matches FP32
    sess_fp32 = ort.InferenceSession(input_path, providers=["CPUExecutionProvider"])
    sess_quant = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    dummy_ids = np.ones((1, 128), dtype=np.int64)
    dummy_mask = np.ones((1, 128), dtype=np.int64)
    feeds = {"input_ids": dummy_ids, "attention_mask": dummy_mask}
    out_fp32 = sess_fp32.run(None, feeds)[0]
    out_quant = sess_quant.run(None, feeds)[0]
    diff = np.abs(out_fp32 - out_quant).mean()
    print(f"  Mean absolute diff vs FP32: {diff:.6f}")
    return output_path


def quantize_htr_encoder(input_dir: str, output_dir: str):
    """Dynamic quantization for TrOCR encoder ONNX only."""
    print("\n=== Quantizing TrOCR encoder ===")
    os.makedirs(output_dir, exist_ok=True)

    encoder_path = os.path.join(input_dir, "encoder_model.onnx")
    encoder_out = os.path.join(output_dir, "encoder_model.onnx")

    quantize_dynamic(
        model_input=encoder_path,
        model_output=encoder_out,
        weight_type=QuantType.QUInt8,
    )
    orig_size = os.path.getsize(encoder_path) / 1e6
    quant_size = os.path.getsize(encoder_out) / 1e6
    print(f"  Encoder: {orig_size:.2f} MB -> {quant_size:.2f} MB "
          f"({(1 - quant_size / orig_size) * 100:.1f}% reduction)")

    # Copy decoder files unchanged (autoregressive decoding is harder to quantize)
    import shutil
    for fname in os.listdir(input_dir):
        if fname != "encoder_model.onnx":
            src = os.path.join(input_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  Copied {fname} (unchanged)")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX models")
    parser.add_argument(
        "--onnx-dir", type=str, default="./onnx_models",
        help="Directory containing FP32 ONNX models",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./onnx_models_quantized",
        help="Output directory for quantized models",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    quantize_biencoder(
        input_path=os.path.join(args.onnx_dir, "biencoder.onnx"),
        output_path=os.path.join(args.output_dir, "biencoder.onnx"),
    )

    quantize_htr_encoder(
        input_dir=os.path.join(args.onnx_dir, "htr_onnx"),
        output_dir=os.path.join(args.output_dir, "htr_onnx"),
    )

    print("\nDone. Quantized models saved to:", args.output_dir)
