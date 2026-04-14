import os
import time
import torch
import mlflow
from torch.utils.data import DataLoader
from models.model import HTRModel, RetrievalModel
from data.loader import JSONDataset

os.makedirs("saved_models", exist_ok=True)

try:
    mlflow.set_tracking_uri("file:./mlruns")
except Exception:
    pass


def train(exp_config, global_config):
    batch_size = exp_config["batch_size"]
    epochs = exp_config["epochs"]
    lr = exp_config["learning_rate"]
    model_type = global_config["model"]["name"]
    data_path = global_config["data"]["path"]
    run_name = f"{exp_config['name']}_{model_type}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "model_type": model_type
        })

        dataset = JSONDataset(data_path)
        loader = DataLoader(dataset, batch_size=batch_size)

        if model_type == "htr":
            model = HTRModel()
        elif model_type == "retrieval":
            model = RetrievalModel()
        else:
            raise ValueError("Unknown model type")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0.0

            for x, y in loader:
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"[{model_type}] Epoch {epoch}, Loss: {avg_loss:.4f}")

            mlflow.log_metric("loss", avg_loss, step=epoch)

            epoch_time = time.time() - epoch_start
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)

        total_time = time.time() - start_time
        mlflow.log_metric("total_training_time", total_time)

        model_path = f"saved_models/{run_name}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved locally to: {model_path}")