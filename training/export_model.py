import os
import json
import yaml
import torch


# =========================
# 可选：根据你项目实际改 import
# =========================
from models.trocr_model import TrOCRModel
from models.retrieval_model import RetrievalModel


def build_model(model_key):
    if model_key == "htr":
        return TrOCRModel()
    elif model_key == "retrieval":
        return RetrievalModel()
    else:
        raise ValueError(f"Unknown model: {model_key}")


def make_dummy_input(cfg):
    shape = cfg["input"]["shape"]
    dtype = cfg["input"]["dtype"]

    if dtype == "float32":
        return torch.randn(*shape)
    elif dtype == "int64":
        # 常见 token 输入
        return torch.randint(low=0, high=1000, size=shape, dtype=torch.long)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def export_one(model_key, cfg, device):
    src_ckpt = cfg["source_checkpoint"]
    export_path = cfg["export_path"]

    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    model = build_model(model_key)
    model.to(device)

    if not os.path.exists(src_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {src_ckpt}")

    state = torch.load(src_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dummy_input = make_dummy_input(cfg).to(device)

    input_name = cfg["input"]["name"]
    output_name = cfg["output"]["name"]

    # 支持动态 batch
    dynamic_axes = {
        input_name: {0: "batch_size"},
        output_name: {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        opset_version=13,
    )

    if not os.path.exists(export_path):
        raise RuntimeError(f"Export failed for {model_key}")

    return {
        "model": model_key,
        "checkpoint": src_ckpt,
        "export_path": export_path,
        "input_shape": cfg["input"]["shape"],
        "output_shape": cfg["output"]["shape"],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("configs/export_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    results = []

    for model_key, model_cfg in config["models"].items():
        print(f"\n===== Exporting {model_key} =====")
        info = export_one(model_key, model_cfg, device)
        print(f"Exported to: {info['export_path']}")
        results.append(info)

    # 写报告
    report_path = "outputs/exported/export_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== EXPORT DONE =====")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()