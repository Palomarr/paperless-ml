import mlflow
import subprocess


def setup_mlflow():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("training")


def log_basic_params(exp_config, model_type):
    mlflow.log_params({
        "experiment_name": exp_config["name"],
        "batch_size": exp_config["batch_size"],
        "epochs": exp_config["epochs"],
        "learning_rate": exp_config["learning_rate"],
        "model_type": model_type
    })


def log_environment():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlflow.log_param("device", device)

    if device == "cuda":
        mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        mlflow.log_param("git_sha", git_sha)
    except Exception:
        pass