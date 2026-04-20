import yaml
import mlflow
from mlflow.tracking import MlflowClient


# ========================
# 从 MLflow 读取最新 run 的 metrics
# ========================
def get_latest_metrics(client, experiment_name, run_name_prefix):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    for run in runs:
        if run.data.tags.get("mlflow.runName", "").startswith(run_name_prefix):
            return run.data.metrics

    raise ValueError(f"No run found for {run_name_prefix}")


# ========================
# 核心：执行 gate 判断
# ========================
def check_gate(model_type, metrics, gate_config):
    required = gate_config.get("required_metrics", [])
    thresholds = gate_config.get("thresholds", {})
    fail_on_missing = gate_config.get("fail_on_missing_metrics", True)

    # 1. 检查 required metrics
    for m in required:
        if m not in metrics:
            if fail_on_missing:
                return False, f"Missing metric: {m}"
            else:
                continue

    # 2. 检查 thresholds
    for key, value in thresholds.items():

        # max_loss
        if key == "max_loss":
            if metrics.get("loss", float("inf")) > value:
                return False, f"Loss too high: {metrics.get('loss')}"

        # min_accuracy
        elif key == "min_accuracy":
            if metrics.get("accuracy", 0) < value:
                return False, f"Accuracy too low: {metrics.get('accuracy')}"

        # min_recall_at_k
        elif key == "min_recall_at_k":
            if metrics.get("recall_at_k", 0) < value:
                return False, f"Recall too low: {metrics.get('recall_at_k')}"

        # embedding_dim_check
        elif key == "expected_embedding_dim":
            if metrics.get("embedding_dim_check", 0) != 1:
                return False, "Embedding dim check failed"

    return True, "PASS"


# ========================
# 主函数
# ========================
def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("quality_gate")

    client = MlflowClient()

    with open("configs/quality_gate.yaml", "r") as f:
        config = yaml.safe_load(f)

    results = {}

    for model_type, model_cfg in config["models"].items():
        print(f"\n===== Checking {model_type} =====")

        gate_cfg = model_cfg["gate"]

        if not gate_cfg.get("enabled", True):
            print("Gate disabled")
            continue

        # 从 evaluation experiment 取结果
        metrics = get_latest_metrics(
            client,
            experiment_name="evaluation",
            run_name_prefix=f"eval_{model_type}"
        )

        passed, reason = check_gate(model_type, metrics, gate_cfg)

        results[model_type] = {
            "status": "PASS" if passed else "FAIL",
            "reason": reason
        }

        print(f"{model_type}: {results[model_type]}")

        # log 到 MLflow
        with mlflow.start_run(run_name=f"gate_{model_type}"):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("status", results[model_type]["status"])
            mlflow.log_param("reason", results[model_type]["reason"])

    print("\n===== FINAL RESULT =====")
    for k, v in results.items():
        print(k, ":", v)


if __name__ == "__main__":
    main()