import os
import time
import torch
import mlflow

from trainer.loader import build_dataloader
from trainer.metrics import compute_accuracy, compute_recall_at_k
from trainer.utils import get_device, save_model
from trainer.mlflow_helper import log_basic_params, log_environment
from models.model import get_model


def train(model_type, exp_config, model_config):
    device = get_device()

    batch_size = exp_config["batch_size"]
    epochs = exp_config["epochs"]
    lr = exp_config["learning_rate"]
    data_path = model_config["data"]["path"]

    run_name = exp_config["name"]

    with mlflow.start_run(run_name=run_name):
        log_basic_params(exp_config, model_type)
        log_environment()

        loader = build_dataloader(data_path, batch_size)

        model = get_model(model_type)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            total_loss = 0.0
            all_outputs = []
            all_targets = []

            for x, y in loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = model(x)

                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                all_outputs.append(outputs.detach().cpu())
                all_targets.append(y.detach().cpu())

            avg_loss = total_loss / len(loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)

            outputs = torch.cat(all_outputs)
            targets = torch.cat(all_targets)

            if model_type == "htr":
                acc = compute_accuracy(outputs, targets)
                mlflow.log_metric("accuracy", acc, step=epoch)

            elif model_type == "retrieval":
                recall = compute_recall_at_k(outputs, targets, k=5)
                mlflow.log_metric("recall_at_k", recall, step=epoch)

            epoch_time = time.time() - epoch_start
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)

            print(f"[{model_type}] Epoch {epoch}, Loss: {avg_loss:.4f}")

        total_time = time.time() - start_time
        mlflow.log_metric("total_training_time", total_time)

        os.makedirs("outputs/checkpoints", exist_ok=True)
        save_path = f"outputs/checkpoints/{run_name}.pt"
        save_model(model, save_path)

        print(f"Saved model to: {save_path}")