import os
import json
import yaml
import torch
import mlflow

from trainer.loader import build_dataloader
from trainer.metrics import compute_accuracy, compute_recall_at_k
from models.model import get_model


def embedding_dim_check(embeddings, expected_dim):
    return embeddings.shape[1] == expected_dim


def evaluate_model(model_type, model_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = model_config["data"]["path"]
    batch_size = model_config["evaluation"]["batch_size"]
    metrics = model_config["evaluation"]["metrics"]

    loader = build_dataloader(data_path, batch_size)

    # 这里先默认评估 baseline
    checkpoint_path = f"outputs/checkpoints/{model_type}_baseline.pt"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(model_type)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_outputs = []
    all_targets = []

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            all_outputs.append(outputs.cpu())
            all_targets.append(y.cpu())

    avg_loss = total_loss / total_samples
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    results = {"loss": avg_loss}

    if model_type == "htr":
        if "accuracy" in metrics:
            acc = compute_accuracy(all_outputs, all_targets)
            results["accuracy"] = acc

    elif model_type == "retrieval":
        embeddings = all_outputs

        if "recall_at_k" in metrics:
            recall = compute_recall_at_k(embeddings, all_targets, k=5)
            results["recall_at_k"] = recall

        if "embedding_dim_check" in metrics:
            ok = embedding_dim_check(embeddings, expected_dim=128)
            results["embedding_dim_check"] = int(ok)

    return results


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("evaluation")

    with open("configs/eval_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.makedirs("outputs/reports", exist_ok=True)
    final_report = {}

    for model_type, model_cfg in config["models"].items():
        print(f"\n===== Evaluating {model_type} =====")

        with mlflow.start_run(run_name=f"eval_{model_type}"):
            results = evaluate_model(model_type, model_cfg)

            for k, v in results.items():
                print(f"{k}: {v}")
                mlflow.log_metric(k, v)

            mlflow.log_param("model_type", model_type)
            final_report[model_type] = results

    with open("outputs/reports/eval_report.json", "w") as f:
        json.dump(final_report, f, indent=4)

    print("\nEvaluation done. Report saved to outputs/reports/eval_report.json")


if __name__ == "__main__":
    main()