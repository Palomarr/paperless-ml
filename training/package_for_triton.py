import os
import yaml
import shutil


# =========================
# 生成 Triton config.pbtxt
# =========================
def generate_config_pbtxt(model_name, cfg):
    input_cfg = cfg["input"]
    output_cfg = cfg["output"]
    serving_cfg = cfg["serving"]

    input_name = input_cfg["name"]
    input_dtype = input_cfg["dtype"].upper()
    input_dims = input_cfg["shape"][1:]  # Triton不包含batch维

    output_name = output_cfg["name"]
    output_dtype = output_cfg["dtype"].upper()
    output_dims = output_cfg["shape"][1:]

    max_batch_size = serving_cfg.get("max_batch_size", 32)

    config = f"""
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}

input [
  {{
    name: "{input_name}"
    data_type: TYPE_{input_dtype}
    dims: {input_dims}
  }}
]

output [
  {{
    name: "{output_name}"
    data_type: TYPE_{output_dtype}
    dims: {output_dims}
  }}
]

dynamic_batching {{}}
"""

    return config


# =========================
# 打包单个模型
# =========================
def package_model(model_name, cfg):
    source_onnx = cfg["exported_model_path"]
    repo_root = "outputs/triton_repo"
    model_dir = os.path.join(repo_root, model_name)
    version_dir = os.path.join(model_dir, "1")

    os.makedirs(version_dir, exist_ok=True)

    # 拷贝 ONNX 模型
    target_model_path = os.path.join(version_dir, "model.onnx")
    shutil.copyfile(source_onnx, target_model_path)

    # 生成 config.pbtxt
    config_str = generate_config_pbtxt(model_name, cfg)
    config_path = os.path.join(model_dir, "config.pbtxt")

    with open(config_path, "w") as f:
        f.write(config_str)

    print(f"Packaged model: {model_name}")
    print(f"  ONNX -> {target_model_path}")
    print(f"  Config -> {config_path}")


# =========================
# 主函数
# =========================
def main():
    with open("configs/handoff_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for model_name, model_cfg in config["models"].items():
        print(f"\n===== Packaging {model_name} =====")

        if not os.path.exists(model_cfg["exported_model_path"]):
            raise FileNotFoundError(
                f"ONNX not found: {model_cfg['exported_model_path']}"
            )

        package_model(model_name, model_cfg)

    print("\n===== ALL MODELS PACKAGED FOR TRITON =====")


if __name__ == "__main__":
    main()